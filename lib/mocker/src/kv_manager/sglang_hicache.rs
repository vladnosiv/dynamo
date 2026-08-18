// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang HiCache model used by replay.
//!
//! The model deliberately owns no radix tree. Device residency remains
//! authoritative in [`SglangKvManager`](super::SglangKvManager); this module
//! consumes the same router events as production and models worker-local L2,
//! the separate device SWA-checkpoint budget, and one byte-bounded shared L3.
//! L1<->L2 traffic is returned as logical-token IO debt to the scheduler.
//! L3 replacement is group-coherent: a FULL block and its optional checkpoint
//! sidecar are evicted together.

use std::collections::{BTreeSet, HashMap};

use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, LocalBlockHash, RouterEvent, SharedCacheHits,
    StorageTier, WorkerWithDpRank, compute_next_seq_hash,
};
use serde::Serialize;

use crate::common::protocols::{SglangHiCacheArgs, SglangHiCacheWritePolicy};
use crate::kv_manager::sglang_backend::is_sglang_hicache_checkpoint_event;

pub(crate) const BYTES_PER_GIB: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StorageGeometry {
    full_bytes_per_block: u64,
    checkpoint_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ComponentMask {
    full: bool,
    checkpoint: bool,
}

impl ComponentMask {
    fn bytes(self, geometry: StorageGeometry) -> u64 {
        u64::from(self.full) * geometry.full_bytes_per_block
            + u64::from(self.checkpoint) * geometry.checkpoint_bytes
    }

    fn union(self, other: Self) -> Self {
        Self {
            full: self.full || other.full,
            checkpoint: self.checkpoint || other.checkpoint,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ResidentPage {
    host: ComponentMask,
    device: ComponentMask,
    host_full_access: u64,
    host_checkpoint_access: u64,
    device_checkpoint_access: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BackupOutcome {
    stored: ComponentMask,
    inserted_full: bool,
}

#[derive(Debug)]
struct WorkerL2 {
    full_capacity_blocks: usize,
    checkpoint_capacity: usize,
    clock: u64,
    pages: HashMap<ExternalSequenceBlockHash, ResidentPage>,
    full_lru: BTreeSet<(u64, ExternalSequenceBlockHash)>,
    checkpoint_lru: BTreeSet<(u64, ExternalSequenceBlockHash)>,
    device_checkpoint_lru: BTreeSet<(u64, ExternalSequenceBlockHash)>,
}

impl WorkerL2 {
    fn new(config: &SglangHiCacheArgs) -> Self {
        Self {
            full_capacity_blocks: config.l2_capacity_blocks,
            checkpoint_capacity: config.l2_checkpoint_capacity,
            clock: 0,
            pages: HashMap::new(),
            full_lru: BTreeSet::new(),
            checkpoint_lru: BTreeSet::new(),
            device_checkpoint_lru: BTreeSet::new(),
        }
    }

    fn page_mut(&mut self, hash: ExternalSequenceBlockHash) -> &mut ResidentPage {
        self.pages.entry(hash).or_insert(ResidentPage {
            host: ComponentMask::default(),
            device: ComponentMask::default(),
            host_full_access: 0,
            host_checkpoint_access: 0,
            device_checkpoint_access: 0,
        })
    }

    fn mark_device_stored(&mut self, hash: ExternalSequenceBlockHash, components: ComponentMask) {
        self.page_mut(hash).device.full |= components.full;
        if components.checkpoint && !self.has_device_checkpoint(hash) {
            self.page_mut(hash).device.checkpoint = true;
        }
        self.touch_device_boundary(hash);
    }

    fn mark_device_checkpoint_removed(&mut self, hash: ExternalSequenceBlockHash) {
        let Some(page) = self.pages.get_mut(&hash) else {
            return;
        };
        if page.device.checkpoint {
            self.device_checkpoint_lru
                .remove(&(page.device_checkpoint_access, hash));
            page.device.checkpoint = false;
        }
        self.prune(hash);
    }

    fn mark_device_removed(&mut self, hash: ExternalSequenceBlockHash) -> ComponentMask {
        let Some(page) = self.pages.get_mut(&hash) else {
            return ComponentMask::default();
        };
        let removed = page.device;
        if page.device.checkpoint {
            self.device_checkpoint_lru
                .remove(&(page.device_checkpoint_access, hash));
        }
        page.device = ComponentMask::default();
        self.prune(hash);
        removed
    }

    fn clear_device(&mut self) {
        self.device_checkpoint_lru.clear();
        for page in self.pages.values_mut() {
            page.device = ComponentMask::default();
        }
        self.pages
            .retain(|_, page| page.host.full || page.host.checkpoint);
    }

    fn insert(
        &mut self,
        hash: ExternalSequenceBlockHash,
        components: ComponentMask,
    ) -> BackupOutcome {
        let mut outcome = BackupOutcome::default();
        if components.full && !self.has_full(hash) {
            while self.full_resident_blocks() >= self.full_capacity_blocks {
                let Some((_, victim)) = self
                    .full_lru
                    .iter()
                    .find(|(_, candidate)| {
                        self.pages
                            .get(candidate)
                            .is_some_and(|page| !page.device.full)
                    })
                    .copied()
                else {
                    break;
                };
                self.remove_full(victim);
            }
            if self.full_resident_blocks() < self.full_capacity_blocks {
                self.page_mut(hash).host.full = true;
                outcome.inserted_full = true;
            }
        }

        if self.has_full(hash) && components.checkpoint && !self.has_checkpoint(hash) {
            while self.checkpoint_resident_blocks() >= self.checkpoint_capacity {
                let Some((_, victim)) = self.checkpoint_lru.pop_first() else {
                    break;
                };
                self.remove_checkpoint(victim);
            }
            if self.checkpoint_resident_blocks() < self.checkpoint_capacity {
                self.page_mut(hash).host.checkpoint = true;
            }
        }
        self.touch(hash);
        outcome.stored = self
            .pages
            .get(&hash)
            .map(|page| page.host)
            .unwrap_or_default();
        outcome
    }

    fn touch(&mut self, hash: ExternalSequenceBlockHash) {
        let Some(page) = self.pages.get_mut(&hash) else {
            return;
        };
        self.clock = self.clock.saturating_add(1);
        if page.host.full {
            self.full_lru.remove(&(page.host_full_access, hash));
            page.host_full_access = self.clock;
            self.full_lru.insert((page.host_full_access, hash));
        }
        if page.host.checkpoint {
            self.checkpoint_lru
                .remove(&(page.host_checkpoint_access, hash));
            page.host_checkpoint_access = self.clock;
            self.checkpoint_lru
                .insert((page.host_checkpoint_access, hash));
        }
    }

    fn touch_device_boundary(&mut self, hash: ExternalSequenceBlockHash) {
        let Some(page) = self.pages.get_mut(&hash) else {
            return;
        };
        if !page.device.checkpoint {
            return;
        }
        self.clock = self.clock.saturating_add(1);
        self.device_checkpoint_lru
            .remove(&(page.device_checkpoint_access, hash));
        page.device_checkpoint_access = self.clock;
        self.device_checkpoint_lru
            .insert((page.device_checkpoint_access, hash));
    }

    fn remove_full(&mut self, hash: ExternalSequenceBlockHash) {
        let Some(page) = self.pages.get_mut(&hash) else {
            return;
        };
        self.full_lru.remove(&(page.host_full_access, hash));
        self.checkpoint_lru
            .remove(&(page.host_checkpoint_access, hash));
        page.host = ComponentMask::default();
        self.prune(hash);
    }

    fn remove_checkpoint(&mut self, hash: ExternalSequenceBlockHash) {
        let Some(page) = self.pages.get_mut(&hash) else {
            return;
        };
        self.checkpoint_lru
            .remove(&(page.host_checkpoint_access, hash));
        page.host.checkpoint = false;
        self.prune(hash);
    }

    fn full_resident_blocks(&self) -> usize {
        self.full_lru.len()
    }

    fn checkpoint_resident_blocks(&self) -> usize {
        self.checkpoint_lru.len()
    }

    fn has_full(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.pages.get(&hash).is_some_and(|page| page.host.full)
    }

    fn has_checkpoint(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.pages
            .get(&hash)
            .is_some_and(|page| page.host.checkpoint)
    }

    fn has_device_checkpoint(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.pages
            .get(&hash)
            .is_some_and(|page| page.device.checkpoint)
    }

    fn device_components(&self, hash: ExternalSequenceBlockHash) -> ComponentMask {
        self.pages
            .get(&hash)
            .map(|page| page.device)
            .unwrap_or_default()
    }

    fn prune(&mut self, hash: ExternalSequenceBlockHash) {
        if self.pages.get(&hash).is_some_and(|page| {
            !page.host.full && !page.host.checkpoint && !page.device.full && !page.device.checkpoint
        }) {
            self.pages.remove(&hash);
        }
    }

    fn extension_blocks(
        &self,
        sequence: &[ExternalSequenceBlockHash],
        device_blocks: usize,
    ) -> usize {
        let mut best = device_blocks;
        for (index, &hash) in sequence.iter().enumerate().skip(device_blocks) {
            if !self.has_full(hash) {
                break;
            }
            if self.has_checkpoint(hash) {
                best = index + 1;
            }
        }
        best.saturating_sub(device_blocks)
    }

    fn device_prefix_blocks(&self, sequence: &[ExternalSequenceBlockHash]) -> usize {
        let mut best = 0;
        for (index, &hash) in sequence.iter().enumerate() {
            let Some(page) = self.pages.get(&hash) else {
                break;
            };
            if !page.device.full {
                break;
            }
            if page.device.checkpoint {
                best = index + 1;
            }
        }
        best
    }
}

#[derive(Clone, Copy, Debug)]
struct L3Page {
    components: ComponentMask,
    last_access: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SglangL3Report {
    pub capacity_bytes: u64,
    pub used_bytes: u64,
    pub peak_bytes: u64,
    pub inserted_bytes: u64,
    pub evicted_bytes: u64,
    pub rejected_bytes: u64,
    pub page_evictions: u64,
    pub final_pages: u64,
    pub final_full_blocks: u64,
    pub final_checkpoint_blocks: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SglangHiCacheReport {
    pub write_policy: SglangHiCacheWritePolicy,
    pub block_size_tokens: usize,
    pub full_bytes_per_block: u64,
    pub l2_capacity_blocks_per_worker: usize,
    pub swa_checkpoint_interval_tokens: usize,
    pub swa_checkpoint_bytes: u64,
    pub l1_checkpoint_capacity_per_worker: usize,
    pub l2_checkpoint_capacity_per_worker: usize,
    pub io_tokens_per_second: u64,
    pub workers_with_l2_state: usize,
    pub final_l2_full_blocks: usize,
    pub final_l2_checkpoint_blocks: usize,
    pub d2h_tokens: u64,
    pub h2d_tokens: u64,
    pub total_host_io_tokens: u64,
    /// Complete prompt tokens reusable on the selected worker immediately
    /// after lower-tier hydration, before chunked-prefill admissions can count
    /// chunks computed by the request itself as cache reuse.
    pub route_input_tokens: u64,
    pub route_reused_tokens: u64,
    pub route_recomputed_tokens: u64,
    pub route_token_hit_rate: f64,
    /// Cache reuse that survived until the request's first scheduler
    /// admission, after queue-time eviction and capacity pressure.
    pub admission_input_tokens: u64,
    pub admission_reused_tokens: u64,
    pub admission_recomputed_tokens: u64,
    pub admission_token_hit_rate: f64,
    pub l3_prefetched_pages: u64,
    pub l3: SglangL3Report,
}

#[derive(Debug)]
struct SharedL3 {
    geometry: StorageGeometry,
    capacity_bytes: u64,
    used_bytes: u64,
    peak_bytes: u64,
    inserted_bytes: u64,
    evicted_bytes: u64,
    rejected_bytes: u64,
    page_evictions: u64,
    clock: u64,
    pages: HashMap<ExternalSequenceBlockHash, L3Page>,
    lru: BTreeSet<(u64, ExternalSequenceBlockHash)>,
}

impl SharedL3 {
    fn new(capacity_bytes: u64, geometry: StorageGeometry) -> Self {
        Self {
            geometry,
            capacity_bytes,
            used_bytes: 0,
            peak_bytes: 0,
            inserted_bytes: 0,
            evicted_bytes: 0,
            rejected_bytes: 0,
            page_evictions: 0,
            clock: 0,
            pages: HashMap::new(),
            lru: BTreeSet::new(),
        }
    }

    fn put(&mut self, hash: ExternalSequenceBlockHash, offered: ComponentMask) -> bool {
        let resident = self
            .pages
            .get(&hash)
            .map(|page| page.components)
            .unwrap_or_default();
        let desired = resident.union(offered);
        let missing_bytes = desired
            .bytes(self.geometry)
            .saturating_sub(resident.bytes(self.geometry));
        // Mooncake missing-key puts do not refresh ordinary LRU recency when
        // every requested object is already resident.
        if missing_bytes == 0 {
            return true;
        }
        if desired.bytes(self.geometry) > self.capacity_bytes {
            self.rejected_bytes = self.rejected_bytes.saturating_add(missing_bytes);
            return false;
        }

        while self.used_bytes.saturating_add(missing_bytes) > self.capacity_bytes {
            let Some((_, victim)) = self
                .lru
                .iter()
                .find(|(_, candidate)| *candidate != hash)
                .copied()
            else {
                self.rejected_bytes = self.rejected_bytes.saturating_add(missing_bytes);
                return false;
            };
            self.evict(victim);
        }

        self.clock = self.clock.saturating_add(1);
        if let Some(page) = self.pages.get_mut(&hash) {
            self.lru.remove(&(page.last_access, hash));
            page.components = desired;
            page.last_access = self.clock;
        } else {
            self.pages.insert(
                hash,
                L3Page {
                    components: desired,
                    last_access: self.clock,
                },
            );
        }
        self.lru.insert((self.clock, hash));
        self.used_bytes = self.used_bytes.saturating_add(missing_bytes);
        self.inserted_bytes = self.inserted_bytes.saturating_add(missing_bytes);
        self.peak_bytes = self.peak_bytes.max(self.used_bytes);
        true
    }

    fn evict(&mut self, hash: ExternalSequenceBlockHash) {
        let Some(page) = self.pages.remove(&hash) else {
            return;
        };
        self.lru.remove(&(page.last_access, hash));
        let bytes = page.components.bytes(self.geometry);
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
        self.evicted_bytes = self.evicted_bytes.saturating_add(bytes);
        self.page_evictions = self.page_evictions.saturating_add(1);
    }

    fn has_full(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.pages
            .get(&hash)
            .is_some_and(|page| page.components.full)
    }

    fn has_checkpoint(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.pages
            .get(&hash)
            .is_some_and(|page| page.components.checkpoint)
    }

    fn prefix_blocks(&self, sequence: &[ExternalSequenceBlockHash]) -> usize {
        let mut best = 0;
        for (index, &hash) in sequence.iter().enumerate() {
            if !self.has_full(hash) {
                break;
            }
            if self.has_checkpoint(hash) {
                best = index + 1;
            }
        }
        best
    }

    fn touch_prefix(&mut self, sequence: &[ExternalSequenceBlockHash], blocks: usize) {
        for &hash in sequence.iter().take(blocks) {
            let Some(page) = self.pages.get_mut(&hash) else {
                continue;
            };
            self.lru.remove(&(page.last_access, hash));
            self.clock = self.clock.saturating_add(1);
            page.last_access = self.clock;
            self.lru.insert((page.last_access, hash));
        }
    }

    fn components(&self, hash: ExternalSequenceBlockHash) -> ComponentMask {
        self.pages
            .get(&hash)
            .map(|page| page.components)
            .unwrap_or_default()
    }

    fn report(&self) -> SglangL3Report {
        SglangL3Report {
            capacity_bytes: self.capacity_bytes,
            used_bytes: self.used_bytes,
            peak_bytes: self.peak_bytes,
            inserted_bytes: self.inserted_bytes,
            evicted_bytes: self.evicted_bytes,
            rejected_bytes: self.rejected_bytes,
            page_evictions: self.page_evictions,
            final_pages: self.pages.len() as u64,
            final_full_blocks: self
                .pages
                .values()
                .filter(|page| page.components.full)
                .count() as u64,
            final_checkpoint_blocks: self
                .pages
                .values()
                .filter(|page| page.components.checkpoint)
                .count() as u64,
        }
    }
}

/// Replay-owned hierarchical state. Device events update this state at their
/// normal visibility boundary; routing probes never mutate LRU state.
#[derive(Debug)]
pub(crate) struct SglangHiCacheState {
    config: SglangHiCacheArgs,
    block_size: usize,
    geometry: StorageGeometry,
    block_positions: HashMap<ExternalSequenceBlockHash, usize>,
    workers: HashMap<WorkerWithDpRank, WorkerL2>,
    l3: SharedL3,
    d2h_tokens: u64,
    l3_prefetched_pages: u64,
}

impl SglangHiCacheState {
    pub(crate) fn new(
        config: SglangHiCacheArgs,
        block_size: usize,
        full_bytes_per_block: u64,
    ) -> Self {
        let capacity_bytes = config
            .l3_capacity_gib
            .checked_mul(BYTES_PER_GIB)
            .expect("validated SGLang L3 capacity must fit in u64");
        let geometry = StorageGeometry {
            full_bytes_per_block,
            checkpoint_bytes: config.swa_checkpoint.bytes,
        };
        Self {
            config,
            block_size,
            geometry,
            block_positions: HashMap::new(),
            workers: HashMap::new(),
            l3: SharedL3::new(capacity_bytes, geometry),
            d2h_tokens: 0,
            l3_prefetched_pages: 0,
        }
    }

    pub(crate) fn sequence_hashes(
        local_hashes: &[LocalBlockHash],
    ) -> Vec<ExternalSequenceBlockHash> {
        let mut parent = None;
        local_hashes
            .iter()
            .copied()
            .map(|local| {
                let hash = match parent {
                    Some(parent_hash) => {
                        ExternalSequenceBlockHash(compute_next_seq_hash(parent_hash, local))
                    }
                    None => ExternalSequenceBlockHash(local.0),
                };
                parent = Some(hash.0);
                hash
            })
            .collect()
    }

    /// Apply one device event and return newly serialized D2H traffic in
    /// logical tokens. A backup already resident in L2 creates no new debt.
    pub(crate) fn apply_router_event(&mut self, event: &RouterEvent) -> usize {
        let checkpoint_event = is_sglang_hicache_checkpoint_event(&event.event);
        if !checkpoint_event && event.storage_tier != StorageTier::Device {
            return 0;
        }
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let mut d2h_tokens = 0_usize;
        if checkpoint_event {
            match &event.event.data {
                KvCacheEventData::Stored(store) => {
                    for block in &store.blocks {
                        let offered = ComponentMask {
                            full: false,
                            checkpoint: true,
                        };
                        self.worker_mut(worker)
                            .mark_device_stored(block.block_hash, offered);
                        if self.config.write_policy == SglangHiCacheWritePolicy::WriteThrough {
                            d2h_tokens = d2h_tokens.saturating_add(self.backup(
                                worker,
                                block.block_hash,
                                offered,
                            ));
                        }
                    }
                }
                KvCacheEventData::Removed(remove) => {
                    for &hash in &remove.block_hashes {
                        self.worker_mut(worker).mark_device_checkpoint_removed(hash);
                    }
                }
                KvCacheEventData::Cleared => {}
            }
            self.d2h_tokens = self
                .d2h_tokens
                .saturating_add(u64::try_from(d2h_tokens).unwrap_or(u64::MAX));
            return d2h_tokens;
        }
        match &event.event.data {
            KvCacheEventData::Stored(store) => {
                let mut position = store
                    .start_position
                    .map(|position| position as usize)
                    .or_else(|| {
                        store
                            .parent_hash
                            .and_then(|hash| self.block_positions.get(&hash).copied())
                            .map(|position| position + 1)
                    })
                    .unwrap_or(0);
                let checkpoint_interval_blocks =
                    self.config.swa_checkpoint.interval_tokens / self.block_size;
                for block in &store.blocks {
                    self.block_positions.insert(block.block_hash, position);
                    let has_checkpoint = (position + 1) % checkpoint_interval_blocks == 0;
                    let device_components = ComponentMask {
                        full: true,
                        checkpoint: false,
                    };
                    self.worker_mut(worker)
                        .mark_device_stored(block.block_hash, device_components);
                    if self.config.write_policy == SglangHiCacheWritePolicy::WriteThrough {
                        // The ordinary KV event describes the FULL block. The
                        // simulator-private checkpoint event tracks independent
                        // sidecar eviction, while the absolute block position tells
                        // WT whether this store crosses a checkpoint boundary.
                        let offered = ComponentMask {
                            full: true,
                            checkpoint: has_checkpoint,
                        };
                        d2h_tokens = d2h_tokens.saturating_add(self.backup(
                            worker,
                            block.block_hash,
                            offered,
                        ));
                    }
                    position += 1;
                }
            }
            KvCacheEventData::Removed(remove) => {
                for &hash in &remove.block_hashes {
                    let components = self
                        .workers
                        .get(&worker)
                        .map(|state| state.device_components(hash))
                        .unwrap_or_default();
                    if self.config.write_policy == SglangHiCacheWritePolicy::WriteBack {
                        d2h_tokens =
                            d2h_tokens.saturating_add(self.backup(worker, hash, components));
                    }
                    self.worker_mut(worker).mark_device_removed(hash);
                }
            }
            KvCacheEventData::Cleared => self.worker_mut(worker).clear_device(),
        }
        self.d2h_tokens = self
            .d2h_tokens
            .saturating_add(u64::try_from(d2h_tokens).unwrap_or(u64::MAX));
        d2h_tokens
    }

    fn worker_mut(&mut self, worker: WorkerWithDpRank) -> &mut WorkerL2 {
        if !self.workers.contains_key(&worker) {
            let state = WorkerL2::new(&self.config);
            self.workers.insert(worker, state);
        }
        self.workers
            .get_mut(&worker)
            .expect("worker L2 state was just inserted")
    }

    fn backup(
        &mut self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
        components: ComponentMask,
    ) -> usize {
        let outcome = self.worker_mut(worker).insert(hash, components);
        if outcome.stored.full {
            self.l3.put(hash, outcome.stored);
        }
        usize::from(outcome.inserted_full).saturating_mul(self.block_size)
    }

    pub(crate) fn l2_extension_blocks(
        &self,
        worker: WorkerWithDpRank,
        sequence: &[ExternalSequenceBlockHash],
        device_blocks: usize,
    ) -> usize {
        self.workers
            .get(&worker)
            .map(|l2| l2.extension_blocks(sequence, device_blocks))
            .unwrap_or(0)
    }

    pub(crate) fn device_prefix_blocks(
        &self,
        worker: WorkerWithDpRank,
        sequence: &[ExternalSequenceBlockHash],
    ) -> usize {
        self.workers
            .get(&worker)
            .map(|state| state.device_prefix_blocks(sequence))
            .unwrap_or(0)
    }

    pub(crate) fn shared_hits(&self, sequence: &[ExternalSequenceBlockHash]) -> SharedCacheHits {
        let prefix = self.l3.prefix_blocks(sequence);
        if prefix == 0 {
            SharedCacheHits::default()
        } else {
            SharedCacheHits::from_ranges(vec![0..u32::try_from(prefix).unwrap_or(u32::MAX)])
        }
    }

    /// Materialize the selected lower-tier prefix into the target worker's L2
    /// and return the number of logical pages available for L1 load-back.
    pub(crate) fn prefetch_selected_prefix(
        &mut self,
        worker: WorkerWithDpRank,
        sequence: &[ExternalSequenceBlockHash],
        device_blocks: usize,
    ) -> usize {
        let local =
            device_blocks.saturating_add(self.l2_extension_blocks(worker, sequence, device_blocks));
        let shared = self.l3.prefix_blocks(sequence);
        let target = local.max(shared);
        if device_blocks > 0 {
            self.worker_mut(worker)
                .touch_device_boundary(sequence[device_blocks - 1]);
        }
        if target <= device_blocks {
            return device_blocks;
        }

        for &hash in sequence.iter().take(target) {
            let components = self.l3.components(hash);
            if components.full || components.checkpoint {
                self.worker_mut(worker).insert(hash, components);
            } else {
                self.worker_mut(worker).touch(hash);
            }
        }
        self.l3.touch_prefix(sequence, shared.min(target));
        let admitted = device_blocks
            .saturating_add(self.l2_extension_blocks(worker, sequence, device_blocks))
            .min(target);
        self.l3_prefetched_pages = self
            .l3_prefetched_pages
            .saturating_add(u64::try_from(admitted.saturating_sub(local)).unwrap_or(u64::MAX));
        admitted
    }

    pub(crate) fn report(&self) -> SglangHiCacheReport {
        SglangHiCacheReport {
            write_policy: self.config.write_policy,
            block_size_tokens: self.block_size,
            full_bytes_per_block: self.geometry.full_bytes_per_block,
            l2_capacity_blocks_per_worker: self.config.l2_capacity_blocks,
            swa_checkpoint_interval_tokens: self.config.swa_checkpoint.interval_tokens,
            swa_checkpoint_bytes: self.config.swa_checkpoint.bytes,
            l1_checkpoint_capacity_per_worker: self.config.l1_checkpoint_capacity,
            l2_checkpoint_capacity_per_worker: self.config.l2_checkpoint_capacity,
            io_tokens_per_second: self.config.io_tokens_per_second,
            workers_with_l2_state: self.workers.len(),
            final_l2_full_blocks: self
                .workers
                .values()
                .map(WorkerL2::full_resident_blocks)
                .sum(),
            final_l2_checkpoint_blocks: self
                .workers
                .values()
                .map(WorkerL2::checkpoint_resident_blocks)
                .sum(),
            d2h_tokens: self.d2h_tokens,
            h2d_tokens: 0,
            total_host_io_tokens: self.d2h_tokens,
            route_input_tokens: 0,
            route_reused_tokens: 0,
            route_recomputed_tokens: 0,
            route_token_hit_rate: 0.0,
            admission_input_tokens: 0,
            admission_reused_tokens: 0,
            admission_recomputed_tokens: 0,
            admission_token_hit_rate: 0.0,
            l3_prefetched_pages: self.l3_prefetched_pages,
            l3: self.l3.report(),
        }
    }

    pub(crate) fn remove_worker(&mut self, worker_id: u64) {
        self.workers
            .retain(|worker, _| worker.worker_id != worker_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::protocols::SglangHiCacheSwaCheckpoint;
    use dynamo_kv_router::protocols::{
        KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData, RouterEvent, StorageTier,
    };

    const FULL_BYTES: u64 = 1_000;
    const CHECKPOINT_BYTES: u64 = 8_000;
    const BUNDLE_BYTES: u64 = FULL_BYTES + CHECKPOINT_BYTES;

    fn geometry() -> StorageGeometry {
        StorageGeometry {
            full_bytes_per_block: FULL_BYTES,
            checkpoint_bytes: CHECKPOINT_BYTES,
        }
    }

    fn config(policy: SglangHiCacheWritePolicy, l3_capacity_gib: u64) -> SglangHiCacheArgs {
        SglangHiCacheArgs {
            write_policy: policy,
            l2_capacity_blocks: 2,
            l1_checkpoint_capacity: 1,
            l2_checkpoint_capacity: 1,
            swa_checkpoint: SglangHiCacheSwaCheckpoint {
                interval_tokens: 256,
                bytes: CHECKPOINT_BYTES,
            },
            l3_capacity_gib,
            io_tokens_per_second: 160_000,
        }
    }

    #[test]
    fn full_block_and_checkpoint_are_the_only_byte_components() {
        let components = ComponentMask {
            full: true,
            checkpoint: true,
        };
        assert_eq!(components.bytes(geometry()), BUNDLE_BYTES);
    }

    #[test]
    fn l3_evicts_all_components_of_one_logical_page() {
        let mut l3 = SharedL3::new(BUNDLE_BYTES, geometry());
        let first = ExternalSequenceBlockHash(1);
        let second = ExternalSequenceBlockHash(2);
        assert!(l3.put(
            first,
            ComponentMask {
                full: true,
                checkpoint: true,
            },
        ));
        assert!(l3.put(
            second,
            ComponentMask {
                full: true,
                checkpoint: true,
            },
        ));
        assert!(!l3.pages.contains_key(&first));
        assert!(l3.pages.contains_key(&second));
        assert_eq!(l3.report().page_evictions, 1);
        assert_eq!(l3.report().used_bytes, BUNDLE_BYTES);
    }

    #[test]
    fn prefix_requires_full_chain_and_checkpoint_boundary() {
        let mut l3 = SharedL3::new(BUNDLE_BYTES * 3, geometry());
        let sequence = [
            ExternalSequenceBlockHash(1),
            ExternalSequenceBlockHash(2),
            ExternalSequenceBlockHash(3),
        ];
        for &hash in &sequence {
            assert!(l3.put(
                hash,
                ComponentMask {
                    full: true,
                    checkpoint: false,
                },
            ));
        }
        assert_eq!(l3.prefix_blocks(&sequence), 0);
        assert!(l3.put(
            sequence[1],
            ComponentMask {
                full: true,
                checkpoint: true,
            },
        ));
        assert_eq!(l3.prefix_blocks(&sequence), 2);
    }

    #[test]
    fn write_back_waits_for_device_removal() {
        let mut state = SglangHiCacheState::new(
            config(SglangHiCacheWritePolicy::WriteBack, 1),
            256,
            FULL_BYTES,
        );
        let worker = WorkerWithDpRank::new(7, 0);
        let hash = ExternalSequenceBlockHash(11);
        let components = ComponentMask {
            full: true,
            checkpoint: true,
        };
        state
            .worker_mut(worker)
            .mark_device_stored(hash, components);
        assert_eq!(state.shared_hits(&[hash]).total_hits, 0);
        assert_eq!(state.backup(worker, hash, components), 256);
        assert_eq!(state.shared_hits(&[hash]).total_hits, 1);
    }

    #[test]
    fn write_through_stores_checkpoint_only_at_configured_boundaries() {
        let mut config = config(SglangHiCacheWritePolicy::WriteThrough, 1);
        config.swa_checkpoint.interval_tokens = 512;
        let mut state = SglangHiCacheState::new(config, 256, FULL_BYTES);
        let first_hash = ExternalSequenceBlockHash(11);
        let second_hash = ExternalSequenceBlockHash(12);
        let first_event = RouterEvent::with_storage_tier(
            7,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: first_hash,
                        tokens_hash: LocalBlockHash(11),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
            StorageTier::Device,
        );
        let second_event = RouterEvent::with_storage_tier(
            7,
            KvCacheEvent {
                event_id: 2,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: Some(first_hash),
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: second_hash,
                        tokens_hash: LocalBlockHash(12),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
            StorageTier::Device,
        );

        assert_eq!(state.apply_router_event(&first_event), 256);
        assert_eq!(state.apply_router_event(&second_event), 256);
        let report = state.report();
        assert_eq!(report.l3.used_bytes, FULL_BYTES * 2 + CHECKPOINT_BYTES);
        assert_eq!(report.l3.final_full_blocks, 2);
        assert_eq!(report.l3.final_checkpoint_blocks, 1);
        assert_eq!(state.shared_hits(&[first_hash, second_hash]).total_hits, 2);
    }

    #[test]
    fn zero_l3_capacity_keeps_local_write_through_io() {
        let mut state = SglangHiCacheState::new(
            config(SglangHiCacheWritePolicy::WriteThrough, 0),
            256,
            FULL_BYTES,
        );
        let event = RouterEvent::with_storage_tier(
            7,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(11),
                        tokens_hash: LocalBlockHash(11),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
            StorageTier::Device,
        );

        assert_eq!(state.apply_router_event(&event), 256);
        let report = state.report();
        assert_eq!(report.l3.used_bytes, 0);
        assert_eq!(report.l3.rejected_bytes, BUNDLE_BYTES);
        assert_eq!(report.d2h_tokens, 256);
    }

    #[test]
    fn l3_prefetch_returns_only_the_prefix_admitted_to_l2() {
        let mut state = SglangHiCacheState::new(
            config(SglangHiCacheWritePolicy::WriteBack, 1),
            256,
            FULL_BYTES,
        );
        let worker = WorkerWithDpRank::new(7, 0);
        let sequence = [
            ExternalSequenceBlockHash(1),
            ExternalSequenceBlockHash(2),
            ExternalSequenceBlockHash(3),
        ];
        let full = ComponentMask {
            full: true,
            checkpoint: true,
        };
        for hash in sequence {
            assert!(state.l3.put(hash, full));
        }

        // The test L2 has room for only two FULL blocks, so sequentially
        // staging a three-page prefix retains a tail but no usable prefix.
        assert_eq!(state.prefetch_selected_prefix(worker, &sequence, 0), 0);
        assert_eq!(state.report().l3_prefetched_pages, 0);
    }

    #[test]
    fn deduplicated_put_does_not_refresh_lru() {
        let mut l3 = SharedL3::new(BUNDLE_BYTES * 2, geometry());
        let first = ExternalSequenceBlockHash(1);
        let second = ExternalSequenceBlockHash(2);
        let third = ExternalSequenceBlockHash(3);
        let full = ComponentMask {
            full: true,
            checkpoint: true,
        };
        assert!(l3.put(first, full));
        assert!(l3.put(second, full));
        assert!(l3.put(first, full));
        assert!(l3.put(third, full));
        assert!(!l3.pages.contains_key(&first));
        assert!(l3.pages.contains_key(&second));
        assert!(l3.pages.contains_key(&third));
    }

    #[test]
    fn sequence_hashes_match_sglang_event_chain() {
        let local = [LocalBlockHash(10), LocalBlockHash(20), LocalBlockHash(30)];
        let sequence = SglangHiCacheState::sequence_hashes(&local);
        assert_eq!(sequence[0], ExternalSequenceBlockHash(10));
        assert_eq!(
            sequence[1],
            ExternalSequenceBlockHash(compute_next_seq_hash(10, local[1]))
        );
        assert_eq!(
            sequence[2],
            ExternalSequenceBlockHash(compute_next_seq_hash(sequence[1].0, local[2]))
        );
    }
}
