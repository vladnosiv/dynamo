// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trace-only SGLang latency reference with unbounded historical L1 reuse.
//!
//! This is intentionally independent of routing and cache capacity. Requests
//! at one source timestamp observe the same immutable pre-bucket history; only
//! complete pages from strictly older timestamp buckets are reusable.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, ensure};
use dynamo_tokens::SequenceHash;

use crate::common::perf_model::PerfModel;
use crate::common::protocols::MockEngineArgs;
use crate::loadgen::{ReplayRequestHashes, ReplayRequestPayload};
use crate::replay::collector::IdealL1RequestBound;

const DEFAULT_CHUNKED_PREFILL_SIZE: usize = 8192;

pub(super) struct IdealL1ZeroQueueTracker {
    block_size: usize,
    chunked_prefill_size: usize,
    total_kv_tokens: usize,
    speedup_ratio: f64,
    decode_speedup_ratio: f64,
    perf_model: Arc<PerfModel>,
    history: HashSet<SequenceHash>,
    pending_bucket: Vec<SequenceHash>,
    current_timestamp_ms: Option<f64>,
}

impl IdealL1ZeroQueueTracker {
    pub(super) fn from_args(args: &MockEngineArgs) -> Self {
        Self {
            block_size: args.block_size,
            chunked_prefill_size: args
                .sglang
                .as_ref()
                .and_then(|sglang| sglang.chunked_prefill_size)
                .unwrap_or(DEFAULT_CHUNKED_PREFILL_SIZE),
            total_kv_tokens: args.num_gpu_blocks.saturating_mul(args.block_size),
            speedup_ratio: args.speedup_ratio,
            decode_speedup_ratio: args.decode_speedup_ratio,
            perf_model: Arc::clone(&args.perf_model),
            history: HashSet::new(),
            pending_bucket: Vec::new(),
            current_timestamp_ms: None,
        }
    }

    pub(super) fn observe(
        &mut self,
        request: &ReplayRequestPayload,
        replay_hashes: Option<&ReplayRequestHashes>,
        arrival_time_ms: f64,
        output_length: usize,
    ) -> Result<IdealL1RequestBound> {
        ensure!(
            arrival_time_ms.is_finite(),
            "ideal L1 bound requires a finite arrival timestamp"
        );
        self.advance_timestamp_bucket(arrival_time_ms)?;

        let computed_hashes;
        let hashes = if let Some(hashes) = replay_hashes {
            hashes
        } else {
            computed_hashes = ReplayRequestHashes::from_tokens(
                request.materialized_tokens().unwrap_or_default(),
                u32::try_from(self.block_size)?,
            );
            &computed_hashes
        };
        let input_length = request.input_length();
        let max_reusable_pages = input_length.saturating_sub(1) / self.block_size;
        let reusable_pages = hashes
            .sequence_hashes
            .iter()
            .take(max_reusable_pages)
            .take_while(|hash| self.history.contains(*hash))
            .count();
        let reusable_tokens = reusable_pages.saturating_mul(self.block_size);
        let recomputed_tokens = input_length.saturating_sub(reusable_tokens);
        let ttft_ms = self.isolated_ttft_ms(
            input_length,
            output_length,
            reusable_tokens,
            recomputed_tokens,
        )?;

        self.pending_bucket
            .extend(hashes.sequence_hashes.iter().copied());
        Ok(IdealL1RequestBound {
            reusable_tokens,
            recomputed_tokens,
            ttft_ms,
        })
    }

    fn advance_timestamp_bucket(&mut self, arrival_time_ms: f64) -> Result<()> {
        if let Some(current) = self.current_timestamp_ms {
            ensure!(
                arrival_time_ms >= current,
                "ideal L1 bound requires non-decreasing arrivals, got {arrival_time_ms} after {current}"
            );
            if arrival_time_ms > current {
                self.history.extend(self.pending_bucket.drain(..));
                self.current_timestamp_ms = Some(arrival_time_ms);
            }
        } else {
            self.current_timestamp_ms = Some(arrival_time_ms);
        }
        Ok(())
    }

    fn isolated_ttft_ms(
        &self,
        input_length: usize,
        output_length: usize,
        reusable_tokens: usize,
        recomputed_tokens: usize,
    ) -> Result<f64> {
        let mut duration_ms = 0.0;
        let mut computed_tokens = 0usize;
        let mut remaining_tokens = recomputed_tokens;
        while remaining_tokens > 0 {
            let chunk_tokens = remaining_tokens.min(self.chunked_prefill_size);
            let prefix_tokens = reusable_tokens.saturating_add(computed_tokens);
            computed_tokens = computed_tokens.saturating_add(chunk_tokens);
            let context_tokens = reusable_tokens.saturating_add(computed_tokens);
            let prefill_ms =
                self.perf_model
                    .predict_prefill_time(1, context_tokens, prefix_tokens)?;
            duration_ms += scale_duration(prefill_ms, self.speedup_ratio);
            remaining_tokens -= chunk_tokens;
        }

        if output_length > 0 {
            let decode_ms = self.perf_model.predict_decode_time(
                1,
                input_length,
                input_length,
                self.total_kv_tokens,
            )?;
            duration_ms +=
                scale_duration(decode_ms, self.speedup_ratio * self.decode_speedup_ratio);
        }
        Ok(duration_ms)
    }
}

fn scale_duration(duration_ms: f64, ratio: f64) -> f64 {
    if ratio > 0.0 && duration_ms > 0.0 {
        duration_ms / ratio
    } else {
        duration_ms
    }
}

#[cfg(test)]
mod tests {
    use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs, SglangArgs};
    use crate::loadgen::ReplayRequestPayload;

    use super::*;

    fn args() -> MockEngineArgs {
        MockEngineArgs {
            engine_type: EngineType::Sglang,
            block_size: 256,
            num_gpu_blocks: 1024,
            sglang: Some(SglangArgs {
                chunked_prefill_size: Some(512),
                ..Default::default()
            }),
            ..MockEngineArgs::default()
        }
    }

    fn request(tokens: Vec<u32>) -> ReplayRequestPayload {
        ReplayRequestPayload::materialized(DirectRequest {
            tokens,
            max_output_tokens: 1,
            ..Default::default()
        })
    }

    #[test]
    fn same_timestamp_does_not_observe_peer_history() {
        let mut tracker = IdealL1ZeroQueueTracker::from_args(&args());
        let tokens = (0..1024).collect::<Vec<u32>>();
        let first = request(tokens);
        let hashes = ReplayRequestHashes::from_tokens(first.materialized_tokens().unwrap(), 256);
        let first_bound = tracker.observe(&first, Some(&hashes), 0.0, 1).unwrap();
        let second_bound = tracker.observe(&first, Some(&hashes), 0.0, 1).unwrap();
        let later_bound = tracker.observe(&first, Some(&hashes), 1.0, 1).unwrap();

        assert_eq!(first_bound.reusable_tokens, 0);
        assert_eq!(second_bound.reusable_tokens, 0);
        assert_eq!(later_bound.reusable_tokens, 768);
        assert_eq!(later_bound.recomputed_tokens, 256);
    }
}
