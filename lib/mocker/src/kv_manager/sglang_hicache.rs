// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang HiCache model used by replay.
//!
//! The model deliberately owns no radix tree. Device residency remains
//! authoritative in [`SglangKvManager`](super::SglangKvManager); this module
//! consumes the same router events as production and models worker-local L2,
//! the separate device SWA/state budget, and one byte-bounded shared L3.
//! L1<->L2 traffic is returned as logical-token IO debt to the scheduler.
//! L3 replacement is group-coherent: all
//! physical objects belonging to one logical 256-token page are evicted
//! together.

use std::collections::{BTreeSet, HashMap};

use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, LocalBlockHash, RouterEvent, SharedCacheHits,
    StorageTier, WorkerWithDpRank, compute_next_seq_hash,
};
use serde::Serialize;

use crate::common::protocols::{SglangHiCacheArgs, SglangHiCacheWritePolicy};
use crate::kv_manager::sglang_backend::is_sglang_hicache_metadata_event;

pub(crate) const BYTES_PER_GIB: u64 = 1 << 30;
pub(crate) const DSV4_GEOMETRY_ID: &str = "dsv4-flash-tp4-attn-cp4-fp32-v1";

pub(crate) const DSV4_C4_BYTES_PER_PAGE: u64 = 786_240;
pub(crate) const DSV4_C4_INDEXER_BYTES_PER_PAGE: u64 = 177_408;
pub(crate) const DSV4_C128_BYTES_PER_PAGE: u64 = 34_560;
pub(crate) const DSV4_SWA_BYTES_PER_PAGE: u64 = 6_439_680;
pub(crate) const DSV4_C4_STATE_BYTES_PER_PAGE: u64 = 1_376_256;
pub(crate) const DSV4_C4_INDEXER_STATE_BYTES_PER_PAGE: u64 = 344_064;

pub(crate) const DSV4_REGULAR_BYTES_PER_PAGE: u64 =
    DSV4_C4_BYTES_PER_PAGE + DSV4_C4_INDEXER_BYTES_PER_PAGE + DSV4_C128_BYTES_PER_PAGE;
pub(crate) const DSV4_TRAILING_BYTES_PER_PAGE: u64 =
    DSV4_SWA_BYTES_PER_PAGE + DSV4_C4_STATE_BYTES_PER_PAGE + DSV4_C4_INDEXER_STATE_BYTES_PER_PAGE;
pub(crate) const DSV4_FULL_BUNDLE_BYTES_PER_PAGE: u64 =
    DSV4_REGULAR_BYTES_PER_PAGE + DSV4_TRAILING_BYTES_PER_PAGE;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ComponentMask {
    regular: bool,
    trailing: bool,
}

impl ComponentMask {
    fn bytes(self) -> u64 {
        u64::from(self.regular) * DSV4_REGULAR_BYTES_PER_PAGE
            + u64::from(self.trailing) * DSV4_TRAILING_BYTES_PER_PAGE
    }

    fn union(self, other: Self) -> Self {
        Self {
            regular: self.regular || other.regular,
            trailing: self.trailing || other.trailing,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ResidentPage {
    host: ComponentMask,
    device: ComponentMask,
    host_full_access: u64,
    host_swa_access: u64,
    device_swa_access: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BackupOutcome {
    stored: ComponentMask,
    inserted_full: bool,
}

#[derive(Debug)]
struct WorkerL2 {
    full_capacity_pages: usize,
    swa_capacity_pages: usize,
    clock: u64,
    pages: HashMap<ExternalSequenceBlockHash, ResidentPage>,
    full_lru: BTreeSet<(u64, ExternalSequenceBlockHash)>,
    swa_lru: BTreeSet<(u64, ExternalSequenceBlockHash)>,
    device_swa_lru: BTreeSet<(u64, ExternalSequenceBlockHash)>,
}

impl WorkerL2 {
    fn new(config: &SglangHiCacheArgs, page_size: usize) -> Self {
        Self {
            full_capacity_pages: config.l2_full_capacity_tokens / page_size,
            swa_capacity_pages: config.l2_swa_capacity_tokens / page_size,
            clock: 0,
            pages: HashMap::new(),
            full_lru: BTreeSet::new(),
            swa_lru: BTreeSet::new(),
            device_swa_lru: BTreeSet::new(),
        }
    }

    fn page_mut(&mut self, hash: ExternalSequenceBlockHash) -> &mut ResidentPage {
        self.pages.entry(hash).or_insert(ResidentPage {
            host: ComponentMask::default(),
            device: ComponentMask::default(),
            host_full_access: 0,
            host_swa_access: 0,
            device_swa_access: 0,
        })
    }

    fn mark_device_stored(&mut self, hash: ExternalSequenceBlockHash, components: ComponentMask) {
        self.page_mut(hash).device.regular |= components.regular;
        if components.trailing && !self.has_device_trailing(hash) {
            self.page_mut(hash).device.trailing = true;
        }
        self.touch_device_boundary(hash);
    }

    fn mark_device_trailing_removed(&mut self, hash: ExternalSequenceBlockHash) {
        let Some(page) = self.pages.get_mut(&hash) else {
            return;
        };
        if page.device.trailing {
            self.device_swa_lru.remove(&(page.device_swa_access, hash));
            page.device.trailing = false;
        }
        self.prune(hash);
    }

    fn mark_device_removed(&mut self, hash: ExternalSequenceBlockHash) -> ComponentMask {
        let Some(page) = self.pages.get_mut(&hash) else {
            return ComponentMask::default();
        };
        let removed = page.device;
        if page.device.trailing {
            self.device_swa_lru.remove(&(page.device_swa_access, hash));
        }
        page.device = ComponentMask::default();
        self.prune(hash);
        removed
    }

    fn clear_device(&mut self) {
        self.device_swa_lru.clear();
        for page in self.pages.values_mut() {
            page.device = ComponentMask::default();
        }
        self.pages
            .retain(|_, page| page.host.regular || page.host.trailing);
    }

    fn insert(
        &mut self,
        hash: ExternalSequenceBlockHash,
        components: ComponentMask,
    ) -> BackupOutcome {
        let mut outcome = BackupOutcome::default();
        if components.regular && !self.has_regular(hash) {
            while self.full_resident_pages() >= self.full_capacity_pages {
                let Some((_, victim)) = self
                    .full_lru
                    .iter()
                    .find(|(_, candidate)| {
                        self.pages
                            .get(candidate)
                            .is_some_and(|page| !page.device.regular)
                    })
                    .copied()
                else {
                    break;
                };
                self.remove_regular(victim);
            }
            if self.full_resident_pages() < self.full_capacity_pages {
                self.page_mut(hash).host.regular = true;
                outcome.inserted_full = true;
            }
        }

        if self.has_regular(hash) && components.trailing && !self.has_trailing(hash) {
            while self.swa_resident_pages() >= self.swa_capacity_pages {
                let Some((_, victim)) = self.swa_lru.pop_first() else {
                    break;
                };
                self.remove_trailing(victim);
            }
            if self.swa_resident_pages() < self.swa_capacity_pages {
                self.page_mut(hash).host.trailing = true;
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
        if page.host.regular {
            self.full_lru.remove(&(page.host_full_access, hash));
            page.host_full_access = self.clock;
            self.full_lru.insert((page.host_full_access, hash));
        }
        if page.host.trailing {
            self.swa_lru.remove(&(page.host_swa_access, hash));
            page.host_swa_access = self.clock;
            self.swa_lru.insert((page.host_swa_access, hash));
        }
    }

    fn touch_device_boundary(&mut self, hash: ExternalSequenceBlockHash) {
        let Some(page) = self.pages.get_mut(&hash) else {
            return;
        };
        if !page.device.trailing {
            return;
        }
        self.clock = self.clock.saturating_add(1);
        self.device_swa_lru.remove(&(page.device_swa_access, hash));
        page.device_swa_access = self.clock;
        self.device_swa_lru.insert((page.device_swa_access, hash));
    }

    fn remove_regular(&mut self, hash: ExternalSequenceBlockHash) {
        let Some(page) = self.pages.get_mut(&hash) else {
            return;
        };
        self.full_lru.remove(&(page.host_full_access, hash));
        self.swa_lru.remove(&(page.host_swa_access, hash));
        page.host = ComponentMask::default();
        self.prune(hash);
    }

    fn remove_trailing(&mut self, hash: ExternalSequenceBlockHash) {
        let Some(page) = self.pages.get_mut(&hash) else {
            return;
        };
        self.swa_lru.remove(&(page.host_swa_access, hash));
        page.host.trailing = false;
        self.prune(hash);
    }

    fn full_resident_pages(&self) -> usize {
        self.full_lru.len()
    }

    fn swa_resident_pages(&self) -> usize {
        self.swa_lru.len()
    }

    fn has_regular(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.pages.get(&hash).is_some_and(|page| page.host.regular)
    }

    fn has_trailing(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.pages.get(&hash).is_some_and(|page| page.host.trailing)
    }

    fn has_device_trailing(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.pages
            .get(&hash)
            .is_some_and(|page| page.device.trailing)
    }

    fn device_components(&self, hash: ExternalSequenceBlockHash) -> ComponentMask {
        self.pages
            .get(&hash)
            .map(|page| page.device)
            .unwrap_or_default()
    }

    fn prune(&mut self, hash: ExternalSequenceBlockHash) {
        if self.pages.get(&hash).is_some_and(|page| {
            !page.host.regular
                && !page.host.trailing
                && !page.device.regular
                && !page.device.trailing
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
            if !self.has_regular(hash) {
                break;
            }
            if self.has_trailing(hash) {
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
            if !page.device.regular {
                break;
            }
            if page.device.trailing {
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
    pub final_regular_pages: u64,
    pub final_trailing_pages: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SglangHiCacheReport {
    pub geometry_id: String,
    pub write_policy: SglangHiCacheWritePolicy,
    pub page_size_tokens: usize,
    pub l1_swa_capacity_tokens_per_worker: usize,
    pub l2_full_capacity_tokens_per_worker: usize,
    pub l2_swa_capacity_tokens_per_worker: usize,
    pub io_tokens_per_second: u64,
    pub c4_bytes_per_page: u64,
    pub c4_indexer_bytes_per_page: u64,
    pub c128_bytes_per_page: u64,
    pub swa_bytes_per_page: u64,
    pub c4_state_bytes_per_page: u64,
    pub c4_indexer_state_bytes_per_page: u64,
    pub regular_bytes_per_page: u64,
    pub trailing_bytes_per_page: u64,
    pub full_bundle_bytes_per_page: u64,
    pub workers_with_l2_state: usize,
    pub final_l2_full_pages: usize,
    pub final_l2_swa_pages: usize,
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
    fn new(capacity_bytes: u64) -> Self {
        Self {
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
        let missing_bytes = desired.bytes().saturating_sub(resident.bytes());
        // Mooncake missing-key puts do not refresh ordinary LRU recency when
        // every requested object is already resident.
        if missing_bytes == 0 {
            return true;
        }
        if desired.bytes() > self.capacity_bytes {
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
        let bytes = page.components.bytes();
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
        self.evicted_bytes = self.evicted_bytes.saturating_add(bytes);
        self.page_evictions = self.page_evictions.saturating_add(1);
    }

    fn has_regular(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.pages
            .get(&hash)
            .is_some_and(|page| page.components.regular)
    }

    fn has_trailing(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.pages
            .get(&hash)
            .is_some_and(|page| page.components.trailing)
    }

    fn prefix_blocks(&self, sequence: &[ExternalSequenceBlockHash]) -> usize {
        let mut best = 0;
        for (index, &hash) in sequence.iter().enumerate() {
            if !self.has_regular(hash) {
                break;
            }
            if self.has_trailing(hash) {
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
            final_regular_pages: self
                .pages
                .values()
                .filter(|page| page.components.regular)
                .count() as u64,
            final_trailing_pages: self
                .pages
                .values()
                .filter(|page| page.components.trailing)
                .count() as u64,
        }
    }
}

/// Replay-owned hierarchical state. Device events update this state at their
/// normal visibility boundary; routing probes never mutate LRU state.
#[derive(Debug)]
pub(crate) struct SglangHiCacheState {
    config: SglangHiCacheArgs,
    page_size: usize,
    workers: HashMap<WorkerWithDpRank, WorkerL2>,
    l3: SharedL3,
    d2h_tokens: u64,
    l3_prefetched_pages: u64,
}

impl SglangHiCacheState {
    pub(crate) fn new(config: SglangHiCacheArgs, page_size: usize) -> Self {
        let capacity_bytes = config
            .l3_capacity_gib
            .checked_mul(BYTES_PER_GIB)
            .expect("validated SGLang L3 capacity must fit in u64");
        Self {
            config,
            page_size,
            workers: HashMap::new(),
            l3: SharedL3::new(capacity_bytes),
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
        let metadata_event = is_sglang_hicache_metadata_event(&event.event);
        if !metadata_event && event.storage_tier != StorageTier::Device {
            return 0;
        }
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let mut d2h_tokens = 0_usize;
        if metadata_event {
            match &event.event.data {
                KvCacheEventData::Stored(store) => {
                    for block in &store.blocks {
                        let offered = ComponentMask {
                            regular: false,
                            trailing: true,
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
                        self.worker_mut(worker).mark_device_trailing_removed(hash);
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
                for block in &store.blocks {
                    let device_components = ComponentMask {
                        regular: true,
                        trailing: false,
                    };
                    self.worker_mut(worker)
                        .mark_device_stored(block.block_hash, device_components);
                    if self.config.write_policy == SglangHiCacheWritePolicy::WriteThrough {
                        // SGLang WT serializes the complete physical representation for
                        // every logical page. Device trailing-state residency is still
                        // tracked separately through the metadata sideband because the
                        // ordinary KV event describes only the radix-cache page.
                        let offered = ComponentMask {
                            regular: true,
                            trailing: true,
                        };
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
            let state = WorkerL2::new(&self.config, self.page_size);
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
        if outcome.stored.regular {
            self.l3.put(hash, outcome.stored);
        }
        usize::from(outcome.inserted_full).saturating_mul(self.page_size)
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
            if components.regular || components.trailing {
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
            geometry_id: DSV4_GEOMETRY_ID.to_string(),
            write_policy: self.config.write_policy,
            page_size_tokens: self.page_size,
            l1_swa_capacity_tokens_per_worker: self.config.l1_swa_capacity_tokens,
            l2_full_capacity_tokens_per_worker: self.config.l2_full_capacity_tokens,
            l2_swa_capacity_tokens_per_worker: self.config.l2_swa_capacity_tokens,
            io_tokens_per_second: self.config.io_tokens_per_second,
            c4_bytes_per_page: DSV4_C4_BYTES_PER_PAGE,
            c4_indexer_bytes_per_page: DSV4_C4_INDEXER_BYTES_PER_PAGE,
            c128_bytes_per_page: DSV4_C128_BYTES_PER_PAGE,
            swa_bytes_per_page: DSV4_SWA_BYTES_PER_PAGE,
            c4_state_bytes_per_page: DSV4_C4_STATE_BYTES_PER_PAGE,
            c4_indexer_state_bytes_per_page: DSV4_C4_INDEXER_STATE_BYTES_PER_PAGE,
            regular_bytes_per_page: DSV4_REGULAR_BYTES_PER_PAGE,
            trailing_bytes_per_page: DSV4_TRAILING_BYTES_PER_PAGE,
            full_bundle_bytes_per_page: DSV4_FULL_BUNDLE_BYTES_PER_PAGE,
            workers_with_l2_state: self.workers.len(),
            final_l2_full_pages: self
                .workers
                .values()
                .map(WorkerL2::full_resident_pages)
                .sum(),
            final_l2_swa_pages: self
                .workers
                .values()
                .map(WorkerL2::swa_resident_pages)
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
    use crate::common::protocols::SglangHiCacheStorageLayout;
    use dynamo_kv_router::protocols::{
        KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData, RouterEvent, StorageTier,
    };

    fn config(policy: SglangHiCacheWritePolicy, l3_capacity_gib: u64) -> SglangHiCacheArgs {
        SglangHiCacheArgs {
            write_policy: policy,
            l1_swa_capacity_tokens: 256,
            l2_full_capacity_tokens: 512,
            l2_swa_capacity_tokens: 256,
            l3_capacity_gib,
            io_tokens_per_second: 160_000,
            storage_layout: SglangHiCacheStorageLayout::Dsv4FlashTp4AttnCp4Fp32,
        }
    }

    #[test]
    fn dsv4_bundle_uses_six_source_qualified_objects() {
        assert_eq!(DSV4_REGULAR_BYTES_PER_PAGE, 998_208);
        assert_eq!(DSV4_TRAILING_BYTES_PER_PAGE, 8_160_000);
        assert_eq!(DSV4_FULL_BUNDLE_BYTES_PER_PAGE, 9_158_208);
    }

    #[test]
    fn l3_evicts_all_components_of_one_logical_page() {
        let mut l3 = SharedL3::new(DSV4_FULL_BUNDLE_BYTES_PER_PAGE);
        let first = ExternalSequenceBlockHash(1);
        let second = ExternalSequenceBlockHash(2);
        assert!(l3.put(
            first,
            ComponentMask {
                regular: true,
                trailing: true,
            },
        ));
        assert!(l3.put(
            second,
            ComponentMask {
                regular: true,
                trailing: true,
            },
        ));
        assert!(!l3.pages.contains_key(&first));
        assert!(l3.pages.contains_key(&second));
        assert_eq!(l3.report().page_evictions, 1);
        assert_eq!(l3.report().used_bytes, DSV4_FULL_BUNDLE_BYTES_PER_PAGE);
    }

    #[test]
    fn prefix_requires_regular_chain_and_trailing_boundary() {
        let mut l3 = SharedL3::new(DSV4_FULL_BUNDLE_BYTES_PER_PAGE * 3);
        let sequence = [
            ExternalSequenceBlockHash(1),
            ExternalSequenceBlockHash(2),
            ExternalSequenceBlockHash(3),
        ];
        for &hash in &sequence {
            assert!(l3.put(
                hash,
                ComponentMask {
                    regular: true,
                    trailing: false,
                },
            ));
        }
        assert_eq!(l3.prefix_blocks(&sequence), 0);
        assert!(l3.put(
            sequence[1],
            ComponentMask {
                regular: true,
                trailing: true,
            },
        ));
        assert_eq!(l3.prefix_blocks(&sequence), 2);
    }

    #[test]
    fn write_back_waits_for_device_removal() {
        let mut state =
            SglangHiCacheState::new(config(SglangHiCacheWritePolicy::WriteBack, 1), 256);
        let worker = WorkerWithDpRank::new(7, 0);
        let hash = ExternalSequenceBlockHash(11);
        let components = ComponentMask {
            regular: true,
            trailing: true,
        };
        state
            .worker_mut(worker)
            .mark_device_stored(hash, components);
        assert_eq!(state.shared_hits(&[hash]).total_hits, 0);
        assert_eq!(state.backup(worker, hash, components), 256);
        assert_eq!(state.shared_hits(&[hash]).total_hits, 1);
    }

    #[test]
    fn write_through_stores_complete_physical_bundle_per_page() {
        let mut state =
            SglangHiCacheState::new(config(SglangHiCacheWritePolicy::WriteThrough, 1), 256);
        let hash = ExternalSequenceBlockHash(11);
        let event = RouterEvent::with_storage_tier(
            7,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: hash,
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
        assert_eq!(report.l3.used_bytes, DSV4_FULL_BUNDLE_BYTES_PER_PAGE);
        assert_eq!(report.l3.inserted_bytes, DSV4_FULL_BUNDLE_BYTES_PER_PAGE);
        assert_eq!(report.l3.final_regular_pages, 1);
        assert_eq!(report.l3.final_trailing_pages, 1);
    }

    #[test]
    fn zero_l3_capacity_keeps_local_write_through_io() {
        let mut state =
            SglangHiCacheState::new(config(SglangHiCacheWritePolicy::WriteThrough, 0), 256);
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
        assert_eq!(report.l3.rejected_bytes, DSV4_FULL_BUNDLE_BYTES_PER_PAGE);
        assert_eq!(report.d2h_tokens, 256);
    }

    #[test]
    fn l3_prefetch_returns_only_the_prefix_admitted_to_l2() {
        let mut state =
            SglangHiCacheState::new(config(SglangHiCacheWritePolicy::WriteBack, 1), 256);
        let worker = WorkerWithDpRank::new(7, 0);
        let sequence = [
            ExternalSequenceBlockHash(1),
            ExternalSequenceBlockHash(2),
            ExternalSequenceBlockHash(3),
        ];
        let full = ComponentMask {
            regular: true,
            trailing: true,
        };
        for hash in sequence {
            assert!(state.l3.put(hash, full));
        }

        // The test L2 has room for only two regular pages, so sequentially
        // staging a three-page prefix retains a tail but no usable prefix.
        assert_eq!(state.prefetch_selected_prefix(worker, &sequence, 0), 0);
        assert_eq!(state.report().l3_prefetched_pages, 0);
    }

    #[test]
    fn deduplicated_put_does_not_refresh_lru() {
        let mut l3 = SharedL3::new(DSV4_FULL_BUNDLE_BYTES_PER_PAGE * 2);
        let first = ExternalSequenceBlockHash(1);
        let second = ExternalSequenceBlockHash(2);
        let third = ExternalSequenceBlockHash(3);
        let full = ComponentMask {
            regular: true,
            trailing: true,
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
