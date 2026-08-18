// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use crate::common::protocols::MockEngineArgs;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{
    ActiveLoad, ActiveSequenceEvent, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::queue::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, DefaultWorkerSelector, LocalScheduler, SequencePublisher,
};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ReplayNoopPublisher;

impl SequencePublisher for ReplayNoopPublisher {
    fn enqueue_event(&self, _event: ActiveSequenceEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn publish_load(&self, _load: ActiveLoad) {}

    fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: usize, _: usize) {}
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ReplayWorkerConfig {
    pub(super) max_num_batched_tokens: u64,
    pub(super) total_kv_blocks: u64,
    pub(super) data_parallel_start_rank: u32,
    pub(super) data_parallel_size: u32,
}

impl WorkerConfigLike for ReplayWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        self.data_parallel_start_rank
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        Some(self.max_num_batched_tokens)
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        Some(self.total_kv_blocks)
    }
}

pub(super) type ReplayScheduler =
    LocalScheduler<ReplayNoopPublisher, ReplayWorkerConfig, DefaultWorkerSelector>;

pub(in crate::replay) fn replay_worker_config(args: &MockEngineArgs) -> ReplayWorkerConfig {
    ReplayWorkerConfig {
        max_num_batched_tokens: args
            .max_num_batched_tokens
            .map(|tokens| tokens as u64)
            .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS),
        total_kv_blocks: args.num_gpu_blocks as u64,
        data_parallel_start_rank: 0,
        data_parallel_size: args.dp_size.max(1),
    }
}

pub(super) fn replay_workers_with_configs(
    args: &MockEngineArgs,
    num_workers: usize,
) -> HashMap<WorkerId, ReplayWorkerConfig> {
    let worker_config = replay_worker_config(args);
    (0..num_workers)
        .map(|worker_idx| (worker_idx as WorkerId, worker_config.clone()))
        .collect()
}

pub(super) fn replay_slots(
    args: &MockEngineArgs,
    workers_with_configs: &HashMap<WorkerId, ReplayWorkerConfig>,
) -> Arc<ActiveSequencesMultiWorker<ReplayNoopPublisher>> {
    let dp_range = workers_with_configs
        .iter()
        .map(|(&worker_id, config)| {
            (
                worker_id,
                (config.data_parallel_start_rank, config.data_parallel_size),
            )
        })
        .collect();
    // NOTE: Offline replay must retire requests through explicit lifecycle events. Wall-clock
    // expiry is a live-router cleanup heuristic and must not observe simulator CPU time: a
    // healthy replay may spend minutes of wall time advancing seconds of virtual time. Keep
    // expiry disabled here until replay has a liveness-aware definition of a stale request; do
    // not mask replay dead ends by expiring requests that are still live in virtual time.
    Arc::new(ActiveSequencesMultiWorker::new_without_expiry(
        ReplayNoopPublisher,
        args.block_size,
        dp_range,
        false,
        0,
        "replay",
    ))
}

pub(super) fn replay_selector(config: &KvRouterConfig) -> DefaultWorkerSelector {
    #[cfg(feature = "replay-bench")]
    if super::canonical_replay_active() {
        return DefaultWorkerSelector::new_seeded(Some(config.clone()), "replay", 0xD1A0_5EED);
    }

    DefaultWorkerSelector::new(Some(config.clone()), "replay")
}

pub(crate) fn replay_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let explicit_router_config = router_config.is_some();
    let mut config = router_config.unwrap_or_default();
    if !explicit_router_config
        && args
            .sglang
            .as_ref()
            .and_then(|sglang| sglang.hicache.as_ref())
            .is_some()
    {
        // Shared blocks beyond the local device prefix receive half credit by
        // default. Callers can still override this through KvRouterConfig.
        config.shared_cache_multiplier = 0.5;
    }
    if let Some(policy) = args.router_queue_policy {
        config.router_queue_policy = policy;
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::protocols::{
        EngineType, SglangArgs, SglangHiCacheArgs, SglangHiCacheSwaCheckpoint,
        SglangHiCacheWritePolicy,
    };

    fn hicache_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(256)
            .kv_bytes_per_token(Some(1))
            .sglang(Some(SglangArgs {
                page_size: Some(256),
                hicache: Some(SglangHiCacheArgs {
                    write_policy: SglangHiCacheWritePolicy::WriteBack,
                    l2_capacity_blocks: 1,
                    l1_checkpoint_capacity: 1,
                    l2_checkpoint_capacity: 1,
                    swa_checkpoint: SglangHiCacheSwaCheckpoint {
                        interval_tokens: 256,
                        bytes: 1,
                    },
                    l3_capacity_gib: 1,
                    io_tokens_per_second: 160_000,
                }),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    #[test]
    fn hicache_default_router_credits_shared_prefix() {
        let args = hicache_args();
        assert_eq!(
            replay_router_config(&args, None).shared_cache_multiplier,
            0.5
        );
        let metadata = crate::replay::canonical_router_metadata_for_args(
            crate::replay::ReplayRouterMode::KvRouter,
            None,
            &args,
        )
        .unwrap();
        assert_eq!(
            metadata.config.unwrap().shared_cache_multiplier,
            0.5,
            "canonical provenance must record the effective runtime default"
        );
    }

    #[test]
    fn explicit_router_config_controls_shared_prefix_weight() {
        let explicit = KvRouterConfig {
            shared_cache_multiplier: 0.25,
            ..Default::default()
        };
        assert_eq!(
            replay_router_config(&hicache_args(), Some(explicit)).shared_cache_multiplier,
            0.25
        );
    }
}
