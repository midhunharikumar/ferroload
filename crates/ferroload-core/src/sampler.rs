//! Deterministic distributed sampler over the `world_size x num_workers` grid.
//!
//! Produces a disjoint, reproducible slice of sample indices per (rank, worker)
//! from `(seed, epoch)`, using a seeded block-shuffle. No external RNG crate —
//! a small splitmix64 keeps it dependency-free and bit-stable across platforms.

/// Launch topology (see DESIGN.md §14.4). `num_nodes == 1` => single-node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    pub num_nodes: u32,
    pub node_rank: u32,
    pub local_size: u32,
    pub world_size: u32,
}

impl Topology {
    pub fn single(local_size: u32) -> Self {
        Topology {
            num_nodes: 1,
            node_rank: 0,
            local_size,
            world_size: local_size.max(1),
        }
    }
}

/// splitmix64 — tiny deterministic PRNG.
#[inline]
fn mix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn seed_for(seed: u64, epoch: u64) -> u64 {
    let mut s = seed ^ (epoch.wrapping_mul(0xD1B54A32D192ED03));
    mix(&mut s);
    s
}

/// Fisher–Yates shuffle of `v` with a seeded splitmix64.
fn shuffle<T>(v: &mut [T], state: &mut u64) {
    let n = v.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = (mix(state) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    pub total: usize,
    pub world_size: u32,
    pub rank: u32,
    pub num_workers: u32,
    pub worker_id: u32,
    pub seed: u64,
    pub shuffle_block: usize,
    pub shuffle: bool,
}

impl Sampler {
    pub fn new(total: usize, world_size: u32, rank: u32, num_workers: u32, worker_id: u32) -> Self {
        assert!(rank < world_size.max(1), "rank out of range");
        assert!(worker_id < num_workers.max(1), "worker_id out of range");
        Sampler {
            total,
            world_size: world_size.max(1),
            rank,
            num_workers: num_workers.max(1),
            worker_id,
            seed: 0,
            shuffle_block: 1024,
            shuffle: true,
        }
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn shuffle_block(mut self, b: usize) -> Self {
        self.shuffle_block = b.max(1);
        self
    }
    pub fn with_shuffle(mut self, on: bool) -> Self {
        self.shuffle = on;
        self
    }

    fn global_worker(&self) -> u32 {
        self.rank * self.num_workers + self.worker_id
    }
    fn num_buckets(&self) -> u32 {
        self.world_size * self.num_workers
    }

    /// Build the full permuted order for an epoch (block-shuffle of 0..total).
    fn permuted(&self, epoch: u64) -> Vec<u32> {
        let mut order: Vec<u32> = (0..self.total as u32).collect();
        if !self.shuffle || self.total < 2 {
            return order;
        }
        let mut state = seed_for(self.seed, epoch);

        // shuffle block order, then within each block — preserves shard locality.
        let bs = self.shuffle_block;
        let nblocks = self.total.div_ceil(bs);
        let mut block_ids: Vec<usize> = (0..nblocks).collect();
        shuffle(&mut block_ids, &mut state);

        let mut out = Vec::with_capacity(self.total);
        for &blk in &block_ids {
            let start = blk * bs;
            let end = (start + bs).min(self.total);
            let mut chunk: Vec<u32> = order[start..end].to_vec();
            shuffle(&mut chunk, &mut state);
            out.extend(chunk);
        }
        order = out;
        order
    }

    /// Indices assigned to this (rank, worker) for `epoch`, skipping `resume_from`
    /// already-consumed items of this worker's slice.
    pub fn indices(&self, epoch: u64, resume_from: usize) -> Vec<u32> {
        let perm = self.permuted(epoch);
        let g = self.global_worker() as usize;
        let step = self.num_buckets() as usize;
        perm.into_iter()
            .skip(g)
            .step_by(step)
            .skip(resume_from)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn partition_is_disjoint_and_complete() {
        let (total, ws, nw) = (1000usize, 4u32, 3u32);
        let mut all = Vec::new();
        for rank in 0..ws {
            for wid in 0..nw {
                let s = Sampler::new(total, ws, rank, nw, wid).seed(42);
                all.extend(s.indices(0, 0));
            }
        }
        let set: BTreeSet<u32> = all.iter().copied().collect();
        assert_eq!(all.len(), total, "no duplicates across grid");
        assert_eq!(set.len(), total, "complete cover 0..total");
        assert_eq!(*set.iter().next().unwrap(), 0);
        assert_eq!(*set.iter().next_back().unwrap(), (total - 1) as u32);
    }

    #[test]
    fn deterministic_same_seed_epoch() {
        let a = Sampler::new(200, 2, 1, 2, 0).seed(7).indices(3, 0);
        let b = Sampler::new(200, 2, 1, 2, 0).seed(7).indices(3, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn epoch_changes_order() {
        let s = Sampler::new(500, 1, 0, 1, 0).seed(7);
        assert_ne!(s.indices(0, 0), s.indices(1, 0));
    }

    #[test]
    fn resume_skips_consumed() {
        let s = Sampler::new(100, 2, 0, 1, 0).seed(1);
        let full = s.indices(0, 0);
        let resumed = s.indices(0, 10);
        assert_eq!(&full[10..], &resumed[..]);
    }

    #[test]
    fn no_shuffle_is_identity_strided() {
        let s = Sampler::new(10, 1, 0, 2, 1).with_shuffle(false);
        // buckets=2, global worker=1 -> indices 1,3,5,7,9
        assert_eq!(s.indices(0, 0), vec![1, 3, 5, 7, 9]);
    }
}
