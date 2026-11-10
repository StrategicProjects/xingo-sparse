//! # Chapter 5.8.5 — Block partitions
//!
//! *New with task L1.7a (PLAN.md trilha L): the contract's
//! `BlockPartition`.*
//!
//! ## Purpose
//! The matrices of optimization are assembled from blocks, and the code
//! around them keeps asking where a block begins: the simplex method prices
//! the structural columns of `[A | ±I]` and treats the logical ones apart
//! (Chapter 5.1); the interior-point method stacks `[−D  Aᵀ; A  δI]` and
//! must know that row `n + i` is the dual of constraint `i`
//! (Chapter 5.14, [`KktForm::Augmented`](crate::KktForm::Augmented));
//! a decomposition (Dantzig–Wolfe, Benders) names coupling rows and
//! subproblem blocks. The matrix stays one [`CscMatrix`](crate::CscMatrix); a
//! [`BlockPartition`] is the small companion object that records the
//! *boundaries* of its row and column blocks, so that assembly with
//! [`Triplets`](crate::Triplets), extraction and bookkeeping agree on them
//! instead of each recomputing `n + i`.
//!
//! ## Notation
//! An `m × n` matrix is partitioned into `p ≥ 0` row blocks and `q ≥ 0`
//! column blocks by the *boundary lists*
//!
//! ```text
//! row_blocks = [r_0, r_1, …, r_p],  0 = r_0 < r_1 < … < r_p = m,
//! col_blocks = [c_0, c_1, …, c_q],  0 = c_0 < c_1 < … < c_q = n.
//! ```
//!
//! Row block `k` is the half-open range `r_k..r_{k+1}`; column block `l` is
//! `c_l..c_{l+1}`; block `(k, l)` is their product. Strict increase means a
//! block is never empty; a dimension of size `0` has the single boundary
//! list `[0]` and no block. Two partitions of the same matrix are equal iff
//! their lists are.
//!
//! The two examples of the contract:
//!
//! * the computational form `[A | ±I]` of the simplex, `A ∈ R^{m×n}`: one
//!   row block, `row_blocks = [0, m]`, and two column blocks,
//!   `col_blocks = [0, n, n + m]` — column `j` is structural iff
//!   `col_block_of(j) = 0`;
//! * the augmented system `K = [−D  Aᵀ; A  δI]` of order `n + m`:
//!   `row_blocks = col_blocks = [0, n, n + m]` — the primal block first, as
//!   [`KktForm::Augmented`](crate::KktForm::Augmented) requires.
//!
//! ## Algorithm B (Block of an index)
//! Given a boundary list `b = [b_0, …, b_p]` and an index `0 ≤ i < b_p`.
//!
//! B1. Find, by binary search, the number `k` of boundaries `≤ i`
//!     (`partition_point`): `k ≥ 1` since `b_0 = 0 ≤ i`, and `k ≤ p` since
//!     `i < b_p`.
//! B2. Return `k − 1`: the block `b_{k−1}..b_k` contains `i`.
//!
//! Cost `O(log p)`.
//!
//! ## Invariants
//! (B1) `row_blocks` is non-empty, starts at `0` and is strictly
//!      increasing; its last element is `m`.
//! (B2) The same for `col_blocks` and `n`.
//! (B3) `m ≤ u32::MAX` and `n ≤ u32::MAX`: the dimensions fit the index
//!      space of the containers ((C4) of Chapter 5.8).
//!
//! Checked by [`BlockPartition::check_invariants`]; both constructors
//! return the first violation as a [`SparseError`]. The fields are public,
//! as in the contract; a value edited by hand should be checked again.
//!
//! ## References
//! Maros, *Computational Techniques of the Simplex Method* (2003), §5.1
//! (the computational form); Wright, *Primal-Dual Interior-Point Methods*
//! (1997), ch. 11 (the augmented system); the contract of the crate
//! (`BlockPartition`).

use core::ops::Range;

use crate::SparseError;
use crate::index::dim_to_u32;

/// Block boundaries of a partitioned matrix: `row_blocks` and `col_blocks`
/// are boundary lists starting at `0` and ending at `m`, `n` (invariants
/// (B1)–(B3)).
///
/// ```
/// use xingo_sparse::BlockPartition;
/// // [A | I] with A 3 × 5: one row block, structural then logical columns.
/// let p = BlockPartition::from_sizes(&[3], &[5, 3]).unwrap();
/// assert_eq!((p.n_rows(), p.n_cols()), (3, 8));
/// assert_eq!(p.col_block(1), 5..8);
/// assert_eq!(p.col_block_of(6), 1); // a logical column
/// // The augmented system [−D Aᵀ; A δI] of the same A: order 5 + 3.
/// let k = BlockPartition::new(vec![0, 5, 8], vec![0, 5, 8]).unwrap();
/// assert_eq!(k.row_block(1).start, 5); // the dual of constraint i is row 5 + i
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockPartition {
    /// Row boundaries `[0 = r_0 < r_1 < … < r_p = m]`.
    pub row_blocks: Vec<usize>,
    /// Column boundaries `[0 = c_0 < c_1 < … < c_q = n]`.
    pub col_blocks: Vec<usize>,
}

impl BlockPartition {
    /// A partition from its boundary lists, validated ((B1)–(B3)).
    pub fn new(row_blocks: Vec<usize>, col_blocks: Vec<usize>) -> Result<Self, SparseError> {
        let p = Self {
            row_blocks,
            col_blocks,
        };
        p.check_invariants()?;
        Ok(p)
    }

    /// A partition from the sizes of its blocks (prefix sums): `[A | I]`
    /// with `A` of `m × n` is `from_sizes(&[m], &[n, m])`. An error if a
    /// size is `0` (blocks are never empty) or a dimension exceeds
    /// `u32::MAX`.
    pub fn from_sizes(row_sizes: &[usize], col_sizes: &[usize]) -> Result<Self, SparseError> {
        Self::new(boundaries(row_sizes), boundaries(col_sizes))
    }

    /// Checks (B1)–(B3).
    pub fn check_invariants(&self) -> Result<(), SparseError> {
        check_boundaries("row", &self.row_blocks)?;
        check_boundaries("column", &self.col_blocks)
    }

    /// Number of rows `m` (the last row boundary).
    #[inline]
    pub fn n_rows(&self) -> usize {
        *self
            .row_blocks
            .last()
            .expect("(B1): non-empty boundary list")
    }

    /// Number of columns `n` (the last column boundary).
    #[inline]
    pub fn n_cols(&self) -> usize {
        *self
            .col_blocks
            .last()
            .expect("(B2): non-empty boundary list")
    }

    /// Number of row blocks `p`.
    #[inline]
    pub fn n_row_blocks(&self) -> usize {
        self.row_blocks.len() - 1
    }

    /// Number of column blocks `q`.
    #[inline]
    pub fn n_col_blocks(&self) -> usize {
        self.col_blocks.len() - 1
    }

    /// The rows `r_k..r_{k+1}` of row block `k`.
    ///
    /// # Panics
    /// If `k ≥ p`.
    #[inline]
    pub fn row_block(&self, k: usize) -> Range<usize> {
        assert!(k < self.n_row_blocks(), "row block {k} out of range");
        self.row_blocks[k]..self.row_blocks[k + 1]
    }

    /// The columns `c_l..c_{l+1}` of column block `l`.
    ///
    /// # Panics
    /// If `l ≥ q`.
    #[inline]
    pub fn col_block(&self, l: usize) -> Range<usize> {
        assert!(l < self.n_col_blocks(), "column block {l} out of range");
        self.col_blocks[l]..self.col_blocks[l + 1]
    }

    /// The row block containing row `i` (Algorithm B).
    ///
    /// # Panics
    /// If `i ≥ m`.
    pub fn row_block_of(&self, i: usize) -> usize {
        assert!(i < self.n_rows(), "row {i} out of range");
        block_of(&self.row_blocks, i)
    }

    /// The column block containing column `j` (Algorithm B).
    ///
    /// # Panics
    /// If `j ≥ n`.
    pub fn col_block_of(&self, j: usize) -> usize {
        assert!(j < self.n_cols(), "column {j} out of range");
        block_of(&self.col_blocks, j)
    }
}

/// Prefix sums of `sizes`, starting at `0`.
fn boundaries(sizes: &[usize]) -> Vec<usize> {
    let mut b = Vec::with_capacity(sizes.len() + 1);
    let mut acc = 0;
    b.push(acc);
    for &s in sizes {
        acc += s;
        b.push(acc);
    }
    b
}

/// (B1)/(B2) and (B3) for one boundary list; `what` is `"row"` or
/// `"column"`.
fn check_boundaries(what: &'static str, b: &[usize]) -> Result<(), SparseError> {
    let Some((&first, _)) = b.split_first() else {
        return Err(SparseError::Malformed(format!(
            "{what} block boundaries are empty (expected at least [0])"
        )));
    };
    if first != 0 {
        return Err(SparseError::Malformed(format!(
            "{what} block boundaries start at {first}, not 0"
        )));
    }
    if let Some(k) = b.windows(2).position(|w| w[0] >= w[1]) {
        return Err(SparseError::Malformed(format!(
            "{what} block boundaries not strictly increasing at block {k}: {} ≥ {}",
            b[k],
            b[k + 1]
        )));
    }
    // (B3): the last element is the dimension.
    dim_to_u32(what, b[b.len() - 1]).map(|_| ())
}

/// Algorithm B on a valid boundary list, `i` below its last element.
#[inline]
fn block_of(b: &[usize], i: usize) -> usize {
    let k = b.partition_point(|&x| x <= i);
    debug_assert!(
        k >= 1 && k < b.len(),
        "Algorithm B: index {i} outside {b:?}"
    );
    k - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn ex_5_8_5_simplex_and_kkt_partitions() {
        // [A | ±I], A 4 × 6.
        let p = BlockPartition::from_sizes(&[4], &[6, 4]).unwrap();
        assert_eq!(p.row_blocks, vec![0, 4]);
        assert_eq!(p.col_blocks, vec![0, 6, 10]);
        assert_eq!((p.n_rows(), p.n_cols()), (4, 10));
        assert_eq!((p.n_row_blocks(), p.n_col_blocks()), (1, 2));
        assert_eq!(p.row_block(0), 0..4);
        assert_eq!(p.col_block(0), 0..6);
        assert_eq!(p.col_block(1), 6..10);
        for j in 0..10 {
            assert_eq!(p.col_block_of(j), usize::from(j >= 6));
            assert_eq!(p.row_block_of(j % 4), 0);
        }
        // [−D Aᵀ; A δI], n = 6, m = 4.
        let k = BlockPartition::new(vec![0, 6, 10], vec![0, 6, 10]).unwrap();
        assert_eq!(k, BlockPartition::from_sizes(&[6, 4], &[6, 4]).unwrap());
        assert_eq!(k.row_block(1), 6..10);
        assert_eq!(k.row_block_of(5), 0);
        assert_eq!(k.row_block_of(6), 1);
        assert_eq!(k.col_block_of(9), 1);
        // An empty dimension: the single boundary [0] and no block.
        let e = BlockPartition::new(vec![0], vec![0, 2]).unwrap();
        assert_eq!((e.n_rows(), e.n_row_blocks()), (0, 0));
        assert_eq!(BlockPartition::from_sizes(&[], &[2]).unwrap(), e);
    }

    #[test]
    fn rejects_bad_boundaries() {
        let bad = |r: Vec<usize>, c: Vec<usize>| BlockPartition::new(r, c).unwrap_err();
        assert!(matches!(bad(vec![], vec![0]), SparseError::Malformed(_)));
        assert!(matches!(
            bad(vec![1, 2], vec![0]),
            SparseError::Malformed(_)
        ));
        assert!(matches!(
            bad(vec![0, 2, 2], vec![0]),
            SparseError::Malformed(_)
        ));
        assert!(matches!(
            bad(vec![0], vec![0, 3, 1]),
            SparseError::Malformed(_)
        ));
        // A zero block size.
        assert!(matches!(
            BlockPartition::from_sizes(&[3, 0], &[1]),
            Err(SparseError::Malformed(_))
        ));
        #[cfg(target_pointer_width = "64")]
        assert!(matches!(
            bad(vec![0, 1 << 33], vec![0]),
            SparseError::TooLarge { what: "row", .. }
        ));
        // A value edited by hand fails the check.
        let mut p = BlockPartition::from_sizes(&[2], &[2]).unwrap();
        p.col_blocks[0] = 1;
        assert!(p.check_invariants().is_err());
    }

    #[test]
    #[should_panic(expected = "row 4 out of range")]
    fn block_of_rejects_an_index_beyond_the_dimension() {
        let p = BlockPartition::from_sizes(&[4], &[1]).unwrap();
        let _ = p.row_block_of(4);
    }

    #[test]
    #[should_panic(expected = "column block 1 out of range")]
    fn block_range_rejects_a_block_beyond_the_count() {
        let p = BlockPartition::from_sizes(&[4], &[1]).unwrap();
        let _ = p.col_block(1);
    }

    proptest! {
        /// `from_sizes` tiles `0..m` with the blocks in order; Algorithm B
        /// agrees with a linear scan on every index.
        #[test]
        fn blocks_tile_the_dimension(
            rows in proptest::collection::vec(1_usize..6, 0..6),
            cols in proptest::collection::vec(1_usize..6, 0..6),
        ) {
            let p = BlockPartition::from_sizes(&rows, &cols).unwrap();
            prop_assert!(p.check_invariants().is_ok());
            prop_assert_eq!(p.n_row_blocks(), rows.len());
            prop_assert_eq!(p.n_col_blocks(), cols.len());
            prop_assert_eq!(p.n_rows(), rows.iter().sum::<usize>());
            let mut next = 0;
            for (k, &s) in rows.iter().enumerate() {
                let r = p.row_block(k);
                prop_assert_eq!(r.clone(), next..next + s);
                for i in r {
                    prop_assert_eq!(p.row_block_of(i), k);
                }
                next += s;
            }
            prop_assert_eq!(next, p.n_rows());
            for j in 0..p.n_cols() {
                let scan = (0..cols.len()).find(|&l| p.col_block(l).contains(&j)).unwrap();
                prop_assert_eq!(p.col_block_of(j), scan);
            }
        }
    }
}
