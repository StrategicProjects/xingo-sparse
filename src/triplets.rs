//! # Chapter 5.8.4 — The coordinate builder
//!
//! *New with task L1.7a (PLAN.md trilha L): the contract's `Triplets<T>`,
//! the last container of v0.1 after [`Pattern`](crate::Pattern) (L1.3).*
//!
//! ## Purpose
//! A compressed matrix (Chapter 5.8.1) is awkward to *build*: its entries
//! must arrive column by column, rows sorted, duplicates merged. Code that
//! assembles a matrix — a model reader, the interior-point method stacking
//! `[−D  Aᵀ; A  δI]` block by block, a test writing a small example — knows
//! its entries in no particular order and sometimes twice. [`Triplets`] is
//! that coordinate (COO) form: three parallel arrays `(row, col, val)` that
//! grow by [`Triplets::push`] in any order, and one conversion,
//! [`Triplets::to_csc`], that sorts, sums the duplicates and validates.
//!
//! [`CscMatrix::from_triplets`] (Chapter 5.8.1) does the same for a slice
//! of `(Row, Col, T)` tuples and is what the solver uses; it is unchanged.
//! The builder is the *incremental* form for code outside the solver —
//! plain `u32` indices, the arrays public as in the contract, buffers that
//! can be cleared and refilled at every iteration of a method whose
//! pattern is fixed while its values change.
//!
//! ## Notation
//! A builder of an `m × n` matrix holds `K` triplets `(i_k, j_k, v_k)`,
//! `k = 0, …, K−1`, in *insertion order* (the order of the `push` calls).
//! The matrix it denotes has entry `a_ij = Σ_{k : i_k = i, j_k = j} v_k`,
//! the sum taken in insertion order.
//!
//! ## Algorithm C′ (Compression of a coordinate list)
//! Algorithm C of Chapter 5.8.1 on the three parallel arrays of the
//! builder (the same four steps, so that the two constructors agree bit
//! for bit on the same list).
//!
//! C′1. [Validate.] Reject arrays of unequal length ((D1) below), a
//!      dimension above `u32::MAX`, an index `i_k ≥ m` or `j_k ≥ n`, and a
//!      non-finite `v_k`. Nothing is validated earlier: `push` only
//!      appends, so an index out of range is reported here, by
//!      [`SparseError::IndexOutOfRange`], never panicked on.
//! C′2. [Bucket by row.] Counting-sort the triplets by row, stably: row
//!      `i` receives the pairs `(j_k, v_k)` in insertion order.
//! C′3. [Transpose.] Algorithm T of Chapter 5.8 on the arrays of C′2. Each
//!      column now lists its rows in non-decreasing order, and the
//!      duplicates of `(i, j)` are adjacent and still in insertion order.
//! C′4. [Sum and compact.] Sweep each column once, replacing every run of
//!      equal row indices by the sum of its values, accumulated left to
//!      right — that is, *in insertion order, without reassociation*: the
//!      result of `push(0, 0, 1e16); push(0, 0, 1.0); push(0, 0, −1e16)`
//!      is `(1e16 + 1) − 1e16 = 0` in `f64`, while pushing `−1e16` before
//!      `1.0` gives `1`. Reject a sum that overflows to `±∞`. A sum that is
//!      exactly zero is **not stored**: the output must satisfy (C3) of
//!      Chapter 5.8, which forbids explicit zeros in a [`CscMatrix`]. Zeros
//!      are therefore *accepted as input* — `push(i, j, 0.0)` is legal and
//!      takes part in the sum — and absent from the output exactly when the
//!      sum cancels. Rebuild `col_ptr`.
//!
//! Every step is a stable counting sort or a left-to-right sweep, so the
//! output, rounding of duplicate sums included, depends only on the
//! sequence of pushes — never on hashing, allocation or thread. Cost
//! `O(K + m + n)` time and space.
//!
//! *Divergence from the contract.* The contract's `to_csc` "drops nothing
//! (zeros are kept)", under its invariant (S1), which admits stored zeros.
//! The implementation of L1.2 kept the first edition's stricter (C3) —
//! no explicit zeros, because the solver's counts rely on it (D-0088, the
//! containers' invariants) — so a `CscMatrix` with a stored zero would
//! fail its own [`CscMatrix::check_invariants`]. `to_csc` keeps (C3);
//! what survives of the contract's sentence is that a zero *input* is not
//! an error. Listed for ratification (task L1.7a).
//!
//! ## Invariants
//! (D1) `row.len() = col.len() = val.len()`: the number of triplets,
//!      [`Triplets::len`]. Maintained by every method; a value built from
//!      the public fields is checked by C′1.
//! (D2) `n_rows ≤ u32::MAX` and `n_cols ≤ u32::MAX` ((C4) of Chapter 5.8);
//!      [`Triplets::new`] asserts it, as [`CscMatrix::new`] does.
//! (D3) After a successful `to_csc`: every `row[k] < n_rows`, every
//!      `col[k] < n_cols`, every `val[k]` finite. Not maintained by `push`
//!      (checked in C′1); [`Triplets::check_invariants`] checks (D1)–(D3)
//!      on demand.
//! (D4) `to_csc` returns a matrix satisfying (C1)–(C4) of Chapter 5.8
//!      whose entry `(i, j)` is the insertion-order sum of the pushes at
//!      `(i, j)`, or an error; asserted in debug builds.
//!
//! ## References
//! Davis, *Direct Methods for Sparse Linear Systems* (2006), §2.1
//! (the triplet form) and §2.5 (`cs_compress`, `cs_dupl`); Gustavson
//! (1978); Chapter 5.8.1 (Algorithm C); the contract of the crate
//! (`Triplets<T>`).

use crate::compressed::transpose;
use crate::index::dim_to_u32;
use crate::real::{Real, is_exact_zero};
use crate::{CscMatrix, SparseError};

/// Coordinate builder: push `(i, j, v)` in any order, duplicates summed in
/// insertion order, then [`Triplets::to_csc`] (Algorithm C′).
///
/// The fields are public, as in the contract, so a list assembled by
/// other means can be converted; `to_csc` validates them ((D1)–(D3)).
#[derive(Clone, Debug, Default)]
pub struct Triplets<T: Real = f64> {
    /// Number of rows `m`.
    pub n_rows: usize,
    /// Number of columns `n`.
    pub n_cols: usize,
    /// Row index `i_k` of each triplet.
    pub row: Vec<u32>,
    /// Column index `j_k` of each triplet.
    pub col: Vec<u32>,
    /// Value `v_k` of each triplet.
    pub val: Vec<T>,
}

impl<T: Real> Triplets<T> {
    /// An empty builder of an `m × n` matrix.
    ///
    /// # Panics
    /// If `m` or `n` exceeds `u32::MAX` ((D2); as [`CscMatrix::new`]).
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        assert!(
            dim_to_u32("rows", n_rows).is_ok(),
            "row count exceeds u32::MAX"
        );
        assert!(
            dim_to_u32("columns", n_cols).is_ok(),
            "column count exceeds u32::MAX"
        );
        Self {
            n_rows,
            n_cols,
            row: Vec::new(),
            col: Vec::new(),
            val: Vec::new(),
        }
    }

    /// The stored entries of `a`, one triplet each, column by column: a
    /// builder that [`Triplets::to_csc`] maps back to `a` exactly, and to
    /// which further entries can be pushed.
    pub fn from_csc(a: &CscMatrix<T>) -> Self {
        let mut t = Self::new(a.n_rows(), a.n_cols());
        t.reserve(a.nnz());
        for (j, w) in a.col_ptr.windows(2).enumerate() {
            let jj = u32::try_from(j).expect("column count exceeds u32::MAX");
            for p in w[0]..w[1] {
                t.row.push(a.row_idx[p]);
                t.col.push(jj);
                t.val.push(a.val[p]);
            }
        }
        t
    }

    /// Appends the triplet `(i, j, v)`. Indices are not checked here (they
    /// are in `to_csc`, step C′1); a zero `v` is legal (step C′4).
    #[inline]
    pub fn push(&mut self, i: u32, j: u32, v: T) {
        self.row.push(i);
        self.col.push(j);
        self.val.push(v);
    }

    /// Number of triplets pushed so far (`K`; duplicates counted).
    #[inline]
    pub fn len(&self) -> usize {
        self.val.len()
    }

    /// `true` iff no triplet has been pushed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.val.is_empty()
    }

    /// Removes every triplet, keeping the dimensions and the allocated
    /// buffers (for a matrix rebuilt with new values at every iteration).
    pub fn clear(&mut self) {
        self.row.clear();
        self.col.clear();
        self.val.clear();
    }

    /// Reserves room for `additional` more triplets.
    pub fn reserve(&mut self, additional: usize) {
        self.row.reserve(additional);
        self.col.reserve(additional);
        self.val.reserve(additional);
    }

    /// Checks (D1)–(D3): equal lengths, dimensions in `u32`, indices in
    /// range, values finite. The validation step C′1 of [`Triplets::to_csc`].
    pub fn check_invariants(&self) -> Result<(), SparseError> {
        // (D1)
        let k = self.row.len();
        if self.col.len() != k {
            return Err(SparseError::Dimension {
                what: "triplet column array",
                expected: k,
                found: self.col.len(),
            });
        }
        if self.val.len() != k {
            return Err(SparseError::Dimension {
                what: "triplet value array",
                expected: k,
                found: self.val.len(),
            });
        }
        // (D2)
        dim_to_u32("rows", self.n_rows)?;
        dim_to_u32("columns", self.n_cols)?;
        // (D3)
        for (k, (&i, &j)) in self.row.iter().zip(&self.col).enumerate() {
            if i as usize >= self.n_rows {
                return Err(SparseError::IndexOutOfRange {
                    what: "row",
                    index: i as usize,
                    bound: self.n_rows,
                });
            }
            if j as usize >= self.n_cols {
                return Err(SparseError::IndexOutOfRange {
                    what: "column",
                    index: j as usize,
                    bound: self.n_cols,
                });
            }
            let v = self.val[k];
            if !v.is_finite() {
                return Err(SparseError::InvalidValue {
                    what: "triplet value",
                    index: k,
                    value: v.to_f64(),
                });
            }
        }
        Ok(())
    }

    /// Algorithm C′: the `m × n` matrix the triplets denote — sorted,
    /// duplicates summed in insertion order, zero sums not stored (C3) —
    /// or the first violation of (D1)–(D3) found.
    pub fn to_csc(&self) -> Result<CscMatrix<T>, SparseError> {
        // C′1. Validate.
        self.check_invariants()?;
        let (m, n, k) = (self.n_rows, self.n_cols, self.len());

        // C′2. Stable counting sort by row: CSR with unsorted columns.
        let mut rptr = vec![0_usize; m + 1];
        for &i in &self.row {
            rptr[i as usize + 1] += 1;
        }
        for i in 0..m {
            rptr[i + 1] += rptr[i];
        }
        let mut next = rptr[..m].to_vec();
        let mut cidx = vec![0_u32; k];
        let mut cval = vec![T::ZERO; k];
        for ((&i, &j), &v) in self.row.iter().zip(&self.col).zip(&self.val) {
            let q = next[i as usize];
            next[i as usize] += 1;
            cidx[q] = j;
            cval[q] = v;
        }

        // C′3. Transpose: rows non-decreasing within each column.
        let (col_ptr, mut row_idx, mut val) = transpose(n, &rptr, &cidx, &cval);

        // C′4. Sum runs of equal rows left to right, skip exact zeros,
        // compact in place.
        let mut out_ptr = Vec::with_capacity(n + 1);
        out_ptr.push(0);
        let mut w = 0;
        for (j, win) in col_ptr.windows(2).enumerate() {
            let (mut p, end) = (win[0], win[1]);
            while p < end {
                let i = row_idx[p];
                let mut s = val[p];
                p += 1;
                while p < end && row_idx[p] == i {
                    s += val[p];
                    p += 1;
                }
                if !s.is_finite() {
                    return Err(SparseError::InvalidValue {
                        what: "sum of duplicate entries in column",
                        index: j,
                        value: s.to_f64(),
                    });
                }
                if !is_exact_zero(s) {
                    row_idx[w] = i;
                    val[w] = s;
                    w += 1;
                }
            }
            out_ptr.push(w);
        }
        row_idx.truncate(w);
        val.truncate(w);

        let a = CscMatrix {
            n_rows: m,
            col_ptr: out_ptr,
            row_idx,
            val,
        };
        // (D4)
        debug_assert!(a.check_invariants().is_ok(), "{:?}", a.check_invariants());
        Ok(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::approx_eq;
    use crate::{Col, Row};
    use proptest::prelude::*;

    #[test]
    fn ex_5_8_4_small_example() {
        // A = [ 1 0 2 ]
        //     [ 0 0 3 ]   with (0,2) split as 0.5 + 1.5 and a zero at (1,0).
        let mut t = Triplets::new(2, 3);
        t.push(1, 2, 3.0);
        t.push(0, 2, 0.5);
        t.push(0, 0, 1.0);
        t.push(1, 0, 0.0);
        t.push(0, 2, 1.5);
        assert_eq!((t.len(), t.is_empty()), (5, false));
        assert!(t.check_invariants().is_ok());
        let a = t.to_csc().unwrap();
        assert_eq!(a.col_ptr, vec![0, 1, 1, 3]);
        assert_eq!(a.row_idx, vec![0, 0, 1]);
        assert_eq!(a.val, vec![1.0, 2.0, 3.0]);
        // The same list through Algorithm C gives the same matrix.
        let c = CscMatrix::from_triplets(
            2,
            3,
            &[
                (Row(1), Col(2), 3.0),
                (Row(0), Col(2), 0.5),
                (Row(0), Col(0), 1.0),
                (Row(1), Col(0), 0.0),
                (Row(0), Col(2), 1.5),
            ],
        )
        .unwrap();
        assert_eq!(a, c);
        // Clear keeps the dimensions; the empty builder is the zero matrix.
        t.clear();
        assert!(t.is_empty());
        assert_eq!(t.to_csc().unwrap(), CscMatrix::zeros(2, 3));
        t.reserve(10);
        assert!(t.row.capacity() >= 10 && t.col.capacity() >= 10 && t.val.capacity() >= 10);
    }

    #[test]
    fn duplicates_are_summed_in_insertion_order_without_reassociation() {
        // (1e16 + 1) − 1e16 = 0 in f64: the entry cancels and is not stored.
        let mut t = Triplets::new(1, 1);
        t.push(0, 0, 1e16);
        t.push(0, 0, 1.0);
        t.push(0, 0, -1e16);
        let a = t.to_csc().unwrap();
        assert_eq!(a.nnz(), 0);
        assert!(a.check_invariants().is_ok());
        // (1e16 − 1e16) + 1 = 1: another order, another sum.
        let mut t = Triplets::new(1, 1);
        t.push(0, 0, 1e16);
        t.push(0, 0, -1e16);
        t.push(0, 0, 1.0);
        let a = t.to_csc().unwrap();
        assert_eq!(a.val, vec![1.0]);
    }

    #[test]
    fn zeros_are_accepted_and_not_stored() {
        let mut t = Triplets::new(2, 2);
        t.push(0, 0, 0.0);
        t.push(1, 1, 2.0);
        t.push(1, 1, 0.0);
        t.push(0, 1, 1.0);
        t.push(0, 1, -1.0);
        let a = t.to_csc().unwrap();
        assert_eq!(a.col_ptr, vec![0, 0, 1]);
        assert_eq!(a.row_idx, vec![1]);
        assert_eq!(a.val, vec![2.0]);
    }

    #[test]
    fn rejects_bad_input() {
        let mut t = Triplets::new(1, 1);
        t.push(1, 0, 1.0);
        assert!(matches!(
            t.to_csc(),
            Err(SparseError::IndexOutOfRange {
                what: "row",
                index: 1,
                bound: 1
            })
        ));
        let mut t = Triplets::new(1, 1);
        t.push(0, 3, 1.0);
        assert!(matches!(
            t.to_csc(),
            Err(SparseError::IndexOutOfRange {
                what: "column",
                index: 3,
                bound: 1
            })
        ));
        let mut t = Triplets::new(1, 1);
        t.push(0, 0, f64::NAN);
        assert!(matches!(
            t.to_csc(),
            Err(SparseError::InvalidValue {
                what: "triplet value",
                index: 0,
                ..
            })
        ));
        let mut t = Triplets::new(1, 1);
        t.push(0, 0, f64::MAX);
        t.push(0, 0, f64::MAX);
        assert!(matches!(
            t.to_csc(),
            Err(SparseError::InvalidValue {
                what: "sum of duplicate entries in column",
                index: 0,
                ..
            })
        ));
        // (D1) through the public fields.
        let t = Triplets {
            n_rows: 1,
            n_cols: 1,
            row: vec![0],
            col: vec![],
            val: vec![1.0],
        };
        assert!(matches!(
            t.check_invariants(),
            Err(SparseError::Dimension {
                what: "triplet column array",
                expected: 1,
                found: 0
            })
        ));
        let t = Triplets {
            n_rows: 1,
            n_cols: 1,
            row: vec![0],
            col: vec![0],
            val: Vec::<f64>::new(),
        };
        assert!(matches!(
            t.to_csc(),
            Err(SparseError::Dimension {
                what: "triplet value array",
                ..
            })
        ));
        // (D2) through the public fields.
        #[cfg(target_pointer_width = "64")]
        {
            let t = Triplets::<f64> {
                n_rows: 1 << 33,
                ..Triplets::new(0, 0)
            };
            assert!(matches!(t.to_csc(), Err(SparseError::TooLarge { .. })));
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "exceeds u32::MAX")]
    fn new_rejects_a_dimension_beyond_u32() {
        let _ = Triplets::<f64>::new(1 << 33, 1);
    }

    #[test]
    fn from_csc_lists_the_entries_column_by_column() {
        let a = CscMatrix::from_raw_parts(2, vec![0, 1, 1, 3], vec![0, 0, 1], vec![1.0, 2.0, 3.0])
            .unwrap();
        let t = Triplets::from_csc(&a);
        assert_eq!((t.n_rows, t.n_cols, t.len()), (2, 3, 3));
        assert_eq!(t.row, vec![0, 0, 1]);
        assert_eq!(t.col, vec![0, 2, 2]);
        assert_eq!(t.val, vec![1.0, 2.0, 3.0]);
        assert_eq!(t.to_csc().unwrap(), a);
    }

    /// Strategy for `(m, n, triplets)` with small-integer values (duplicate
    /// sums exact in any order, frequent exact cancellations).
    fn triplets() -> impl Strategy<Value = (usize, usize, Vec<(u32, u32, f64)>)> {
        (0_usize..8, 0_usize..8).prop_flat_map(|(m, n)| {
            let entries = if m == 0 || n == 0 {
                Just(Vec::new()).boxed()
            } else {
                proptest::collection::vec(
                    (0..m, 0..n, -3_i8..=3).prop_map(|(i, j, v)| {
                        (
                            u32::try_from(i).unwrap(),
                            u32::try_from(j).unwrap(),
                            f64::from(v),
                        )
                    }),
                    0..40,
                )
                .boxed()
            };
            (Just(m), Just(n), entries)
        })
    }

    /// A CSC matrix from the strategy, plus a shuffle of `0..nnz + extra`.
    fn csc_and_shuffle() -> impl Strategy<Value = (CscMatrix, Vec<usize>, Vec<(u32, u32)>)> {
        triplets().prop_flat_map(|(m, n, t)| {
            let mut b = Triplets::new(m, n);
            for &(i, j, v) in &t {
                b.push(i, j, v);
            }
            let a = b.to_csc().unwrap();
            // Extra zero pushes at arbitrary positions (in range), which
            // must leave the matrix unchanged.
            let extra = if m == 0 || n == 0 {
                Just(Vec::new()).boxed()
            } else {
                proptest::collection::vec(
                    (0..m, 0..n)
                        .prop_map(|(i, j)| (u32::try_from(i).unwrap(), u32::try_from(j).unwrap())),
                    0..6,
                )
                .boxed()
            };
            let nnz = a.nnz();
            (Just(a), extra).prop_flat_map(move |(a, extra)| {
                let order: Vec<usize> = (0..nnz + extra.len()).collect();
                (Just(a), Just(order).prop_shuffle(), Just(extra))
            })
        })
    }

    proptest! {
        /// Algorithm C′ agrees with Algorithm C on the same list (same
        /// sums, same order of summation): the builder is a second entry to
        /// the same construction.
        #[test]
        fn c_prime_agrees_with_c((m, n, t) in triplets()) {
            let mut b = Triplets::new(m, n);
            for &(i, j, v) in &t {
                b.push(i, j, v);
            }
            prop_assert_eq!(b.len(), t.len());
            prop_assert!(b.check_invariants().is_ok());
            let a = b.to_csc().unwrap();
            prop_assert!(a.check_invariants().is_ok());
            let tuples: Vec<(Row, Col, f64)> =
                t.iter().map(|&(i, j, v)| (Row(i), Col(j), v)).collect();
            let c = CscMatrix::from_triplets(m, n, &tuples).unwrap();
            prop_assert_eq!(a, c);
        }

        /// Round trip: CSC → triplets in a random order, with extra zero
        /// pushes anywhere → the identical CSC (structure and values).
        #[test]
        fn round_trip_in_any_order_with_zeros((a, order, extra) in csc_and_shuffle()) {
            let t = Triplets::from_csc(&a);
            prop_assert_eq!(t.to_csc().unwrap(), a.clone());
            let nnz = a.nnz();
            let mut shuffled = Triplets::new(a.n_rows(), a.n_cols());
            for &k in &order {
                if k < nnz {
                    shuffled.push(t.row[k], t.col[k], t.val[k]);
                } else {
                    let (i, j) = extra[k - nnz];
                    shuffled.push(i, j, 0.0);
                }
            }
            prop_assert_eq!(shuffled.len(), nnz + extra.len());
            let back = shuffled.to_csc().unwrap();
            prop_assert!(back.check_invariants().is_ok());
            prop_assert_eq!(back, a);
        }

        /// The matrix denoted equals the dense accumulation entry by entry
        /// (values are small integers, so the order of summation is moot).
        #[test]
        fn matches_the_dense_reference((m, n, t) in triplets()) {
            let mut d = vec![vec![0.0; n]; m];
            let mut b = Triplets::new(m, n);
            for &(i, j, v) in &t {
                d[i as usize][j as usize] += v;
                b.push(i, j, v);
            }
            let a = b.to_csc().unwrap();
            let mut nnz = 0;
            for (i, row) in d.iter().enumerate() {
                for (j, &dij) in row.iter().enumerate() {
                    let aij = a.get(Row::from_usize(i), Col::from_usize(j));
                    prop_assert!(approx_eq(aij, dij, 0.0));
                    nnz += usize::from(!is_exact_zero(dij));
                }
            }
            prop_assert_eq!(a.nnz(), nnz);
        }
    }
}
