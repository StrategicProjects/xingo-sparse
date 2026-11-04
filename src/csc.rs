//! # Chapter 5.8.1 — Compressed sparse column storage
//!
//! ## Purpose
//! [`CscMatrix`] is the canonical storage of the constraint matrix `A`. It is
//! physically identical to an Arrow `ListArray` of `(row, val)` and to R's
//! `Matrix::dgCMatrix` (`PLAN.md` §4.2), so the fields are public for
//! zero-copy interoperability; [`CscMatrix::check_invariants`] guards them.
//!
//! ## Notation
//! `A ∈ R^{m×n}`; column `j` holds the pairs `(row_idx[p], val[p])` for
//! `p ∈ [col_ptr[j], col_ptr[j+1])`. A *triplet* `(i, j, v)` is one entry of
//! the coordinate (COO) format; a triplet list may contain duplicates, which
//! denote the sum of their values, and zeros.
//!
//! ## Algorithm C (Compression of a triplet list)
//! Given `m`, `n` and triplets `(i_k, j_k, v_k)`, `k = 0, …, K−1`.
//!
//! C1. [Validate.] Reject an index `i_k ≥ m` or `j_k ≥ n`, and a non-finite
//!     `v_k`.
//! C2. [Bucket by row.] Counting-sort the triplets by row, stably: row `i`
//!     receives the pairs `(j_k, v_k)` in input order. This is a CSR array
//!     with unsorted column indices and possible duplicates.
//! C3. [Transpose.] Apply Algorithm T (Chapter 5.8) to the arrays of C2. In
//!     the result every column lists its row indices in non-decreasing order,
//!     and duplicates `(i, j)` are adjacent and still in input order.
//! C4. [Sum and drop.] Sweep each column once, replacing each run of equal
//!     row indices by the sum of its values (accumulated in input order) and
//!     dropping sums that are exactly zero. Reject a sum that overflows to
//!     `±∞`. Compact the arrays in place and rebuild `col_ptr`.
//!
//! The output satisfies (C1)–(C4) of Chapter 5.8. Because every step is a
//! stable counting sort, the result — including the rounding of duplicate
//! sums — depends only on the input, never on hashing or allocation order.
//! Cost: `O(K + m + n)` time and space.
//!
//! Only *exact* zeros are dropped: deciding that `1e-17` is noise is a
//! modelling decision taken by the solver's presolve with its tolerances
//! (`xingo_core::Tolerances::zero`), not by storage.
//!
//! ## Invariants
//! (C1)–(C4) of Chapter 5.8 with `ptr = col_ptr`, `idx = row_idx`,
//! `n_major = n`, `n_minor = m`; checked by [`CscMatrix::check_invariants`]
//! and asserted in debug builds after every construction.
//!
//! ## References
//! Davis, *Direct Methods for Sparse Linear Systems* (2006), §2.2–2.5
//! (`cs_compress`, `cs_dupl`); Gustavson (1978).

use crate::compressed::{check_compressed, transpose};
use crate::index::{Col, Row, dim_to_u32};
use crate::real::{Real, is_exact_zero};
use crate::{CsrMatrix, SparseError};

/// Sparse matrix in compressed sparse column (CSC) format.
///
/// See the module documentation for the invariants. Equality is structural
/// (same dimensions, same stored entries in the same order), which for
/// matrices satisfying the invariants is equality of the matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct CscMatrix<T: Real = f64> {
    /// Number of rows `m`.
    pub n_rows: usize,
    /// Column pointers, length `n + 1`; column `j` occupies
    /// `col_ptr[j]..col_ptr[j + 1]`.
    pub col_ptr: Vec<usize>,
    /// Row index of each stored entry, strictly increasing within a column.
    pub row_idx: Vec<u32>,
    /// Value of each stored entry; finite and nonzero.
    pub val: Vec<T>,
}

impl<T: Real> CscMatrix<T> {
    /// The `m × n` zero matrix (the contract's name; the same as
    /// [`CscMatrix::zeros`]).
    ///
    /// # Panics
    /// If `m` or `n` exceeds `u32::MAX`.
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self::zeros(n_rows, n_cols)
    }

    /// The `m × n` zero matrix.
    ///
    /// # Panics
    /// If `m` or `n` exceeds `u32::MAX`.
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
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
            col_ptr: vec![0; n_cols + 1],
            row_idx: Vec::new(),
            val: Vec::new(),
        }
    }

    /// Builds a matrix from raw compressed arrays, validating every invariant.
    pub fn from_raw_parts(
        n_rows: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<u32>,
        val: Vec<T>,
    ) -> Result<Self, SparseError> {
        let a = Self {
            n_rows,
            col_ptr,
            row_idx,
            val,
        };
        a.check_invariants()?;
        Ok(a)
    }

    /// Builds an `m × n` matrix from triplets `(i, j, v)` by Algorithm C:
    /// duplicates are summed in input order and exact zeros are dropped.
    pub fn from_triplets(
        n_rows: usize,
        n_cols: usize,
        triplets: &[(Row, Col, T)],
    ) -> Result<Self, SparseError> {
        dim_to_u32("rows", n_rows)?;
        dim_to_u32("columns", n_cols)?;

        // C1. Validate.
        for (k, &(i, j, v)) in triplets.iter().enumerate() {
            if i.index() >= n_rows {
                return Err(SparseError::IndexOutOfRange {
                    what: "row",
                    index: i.index(),
                    bound: n_rows,
                });
            }
            if j.index() >= n_cols {
                return Err(SparseError::IndexOutOfRange {
                    what: "column",
                    index: j.index(),
                    bound: n_cols,
                });
            }
            if !v.is_finite() {
                return Err(SparseError::InvalidValue {
                    what: "triplet value",
                    index: k,
                    value: v.to_f64(),
                });
            }
        }

        // C2. Stable counting sort by row: CSR with unsorted columns.
        let mut rptr = vec![0_usize; n_rows + 1];
        for &(i, _, _) in triplets {
            rptr[i.index() + 1] += 1;
        }
        for i in 0..n_rows {
            rptr[i + 1] += rptr[i];
        }
        let mut next = rptr[..n_rows].to_vec();
        let mut cidx = vec![0_u32; triplets.len()];
        let mut cval = vec![T::ZERO; triplets.len()];
        for &(i, j, v) in triplets {
            let q = next[i.index()];
            next[i.index()] += 1;
            cidx[q] = j.0;
            cval[q] = v;
        }

        // C3. Transpose: rows non-decreasing within each column.
        let (col_ptr, mut row_idx, mut val) = transpose(n_cols, &rptr, &cidx, &cval);

        // C4. Sum runs of equal rows, drop exact zeros, compact in place.
        let mut out_ptr = Vec::with_capacity(n_cols + 1);
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

        let a = Self {
            n_rows,
            col_ptr: out_ptr,
            row_idx,
            val,
        };
        debug_assert!(a.check_invariants().is_ok(), "{:?}", a.check_invariants());
        Ok(a)
    }

    /// Number of rows `m`.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns `n`.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.col_ptr.len() - 1
    }

    /// Number of stored entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }

    /// Row indices and values of column `j`.
    #[inline]
    pub fn col(&self, j: Col) -> (&[u32], &[T]) {
        let r = self.col_ptr[j.index()]..self.col_ptr[j.index() + 1];
        (&self.row_idx[r.clone()], &self.val[r])
    }

    /// Entry `a_ij`, or `0` if not stored; `O(log nnz(column j))`.
    pub fn get(&self, i: Row, j: Col) -> T {
        let (rows, vals) = self.col(j);
        rows.binary_search(&i.0).map_or(T::ZERO, |p| vals[p])
    }

    /// `y ← y + α A x` with dense `x` (length `n`) and `y` (length `m`),
    /// column by column: `y_i ← y_i + a_ij (α x_j)` for the stored entries,
    /// skipping the columns with `x_j = 0`.
    ///
    /// # Panics
    /// If `x.len() != n` or `y.len() != m`.
    pub fn gaxpy(&self, alpha: T, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.n_cols(), "gaxpy: x has the wrong length");
        assert_eq!(y.len(), self.n_rows, "gaxpy: y has the wrong length");
        for (j, &xj) in x.iter().enumerate() {
            if is_exact_zero(xj) {
                continue;
            }
            let t = alpha * xj;
            let (rows, vals) = self.col(Col::from_usize(j));
            for (&i, &v) in rows.iter().zip(vals) {
                y[i as usize] += v * t;
            }
        }
    }

    /// The same matrix in row-wise (CSR) storage, by Algorithm T.
    pub fn to_csr(&self) -> CsrMatrix<T> {
        let (row_ptr, col_idx, val) =
            transpose(self.n_rows, &self.col_ptr, &self.row_idx, &self.val);
        let r = CsrMatrix {
            n_cols: self.n_cols(),
            row_ptr,
            col_idx,
            val,
        };
        debug_assert!(r.check_invariants().is_ok(), "{:?}", r.check_invariants());
        r
    }

    /// The transpose `Aᵀ` in CSC storage, by Algorithm T.
    #[must_use]
    pub fn transpose(&self) -> CscMatrix<T> {
        let (col_ptr, row_idx, val) =
            transpose(self.n_rows, &self.col_ptr, &self.row_idx, &self.val);
        let t = CscMatrix {
            n_rows: self.n_cols(),
            col_ptr,
            row_idx,
            val,
        };
        debug_assert!(t.check_invariants().is_ok(), "{:?}", t.check_invariants());
        t
    }

    /// Checks invariants (C1)–(C4).
    pub fn check_invariants(&self) -> Result<(), SparseError> {
        check_compressed(
            "column",
            "row",
            self.n_rows,
            &self.col_ptr,
            &self.row_idx,
            &self.val,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::approx_eq;
    use proptest::prelude::*;

    /// Dense reference: accumulate the triplets into an `m × n` array.
    fn dense_reference(m: usize, n: usize, t: &[(Row, Col, f64)]) -> Vec<Vec<f64>> {
        let mut d = vec![vec![0.0; n]; m];
        for &(i, j, v) in t {
            d[i.index()][j.index()] += v;
        }
        d
    }

    /// Strategy for `(m, n, triplets)`. Values are small integers so that
    /// duplicate sums are exact regardless of summation order, and exact
    /// cancellations (explicit zeros after summing) are frequent.
    fn triplets() -> impl Strategy<Value = (usize, usize, Vec<(Row, Col, f64)>)> {
        (0_usize..8, 0_usize..8).prop_flat_map(|(m, n)| {
            let entries = if m == 0 || n == 0 {
                Just(Vec::new()).boxed()
            } else {
                proptest::collection::vec(
                    (0..m, 0..n, -3_i8..=3).prop_map(|(i, j, v)| {
                        (Row::from_usize(i), Col::from_usize(j), f64::from(v))
                    }),
                    0..40,
                )
                .boxed()
            };
            (Just(m), Just(n), entries)
        })
    }

    #[test]
    fn ex_1_1_small_example() {
        // A = [ 1 0 2 ]
        //     [ 0 0 3 ]   with a duplicate (0,2) split as 0.5 + 1.5 and an
        //                 explicit zero at (1,0).
        let t = [
            (Row(1), Col(2), 3.0),
            (Row(0), Col(2), 0.5),
            (Row(0), Col(0), 1.0),
            (Row(1), Col(0), 0.0),
            (Row(0), Col(2), 1.5),
        ];
        let a = CscMatrix::from_triplets(2, 3, &t).unwrap();
        assert_eq!(a.col_ptr, vec![0, 1, 1, 3]);
        assert_eq!(a.row_idx, vec![0, 0, 1]);
        assert_eq!(a.val, vec![1.0, 2.0, 3.0]);
        assert!(approx_eq(a.get(Row(0), Col(2)), 2.0, 0.0));
        assert!(is_exact_zero(a.get(Row(1), Col(1))));
    }

    #[test]
    fn rejects_bad_input() {
        let oob = CscMatrix::from_triplets(1, 1, &[(Row(1), Col(0), 1.0)]);
        assert!(matches!(
            oob,
            Err(SparseError::IndexOutOfRange { what: "row", .. })
        ));
        let nan = CscMatrix::from_triplets(1, 1, &[(Row(0), Col(0), f64::NAN)]);
        assert!(matches!(nan, Err(SparseError::InvalidValue { .. })));
        let ovf = CscMatrix::from_triplets(
            1,
            1,
            &[(Row(0), Col(0), f64::MAX), (Row(0), Col(0), f64::MAX)],
        );
        assert!(matches!(ovf, Err(SparseError::InvalidValue { .. })));
    }

    #[test]
    fn raw_parts_are_validated() {
        assert!(CscMatrix::from_raw_parts(2, vec![0, 2], vec![0, 1], vec![1.0, 2.0]).is_ok());
        // Unsorted rows.
        assert!(CscMatrix::from_raw_parts(2, vec![0, 2], vec![1, 0], vec![1.0, 2.0]).is_err());
        // Duplicate rows.
        assert!(CscMatrix::from_raw_parts(2, vec![0, 2], vec![0, 0], vec![1.0, 2.0]).is_err());
        // Explicit zero.
        assert!(CscMatrix::from_raw_parts(2, vec![0, 1], vec![0], vec![0.0]).is_err());
        // Row out of range.
        assert!(CscMatrix::from_raw_parts(1, vec![0, 1], vec![1], vec![1.0]).is_err());
        // Pointer not starting at zero / decreasing / wrong length.
        assert!(CscMatrix::<f64>::from_raw_parts(2, vec![1, 1], vec![], vec![]).is_err());
        assert!(CscMatrix::from_raw_parts(2, vec![0, 1, 0], vec![0], vec![1.0]).is_err());
        assert!(CscMatrix::from_raw_parts(2, vec![0, 2], vec![0], vec![1.0]).is_err());
        assert!(CscMatrix::<f64>::from_raw_parts(2, vec![], vec![], vec![]).is_err());
    }

    #[test]
    fn gaxpy_matches_the_dense_product() {
        // A = [[1, 0, 2], [0, 0, 3]]
        let a = CscMatrix::from_raw_parts(2, vec![0, 1, 1, 3], vec![0, 0, 1], vec![1.0, 2.0, 3.0])
            .unwrap();
        let mut y = vec![1.0, -1.0];
        a.gaxpy(2.0, &[1.0, 5.0, -1.0], &mut y);
        assert!(approx_eq(y[0], 1.0 + 2.0 * (1.0 - 2.0), 0.0));
        assert!(approx_eq(y[1], -1.0 + 2.0 * (-3.0), 0.0));
        assert_eq!(CscMatrix::<f64>::new(2, 3), CscMatrix::zeros(2, 3));
    }

    #[test]
    fn zeros_is_valid() {
        let z = CscMatrix::<f64>::zeros(3, 4);
        assert!(z.check_invariants().is_ok());
        assert_eq!((z.n_rows(), z.n_cols(), z.nnz()), (3, 4, 0));
        assert_eq!(z.to_csr().to_csc(), z);
    }

    proptest! {
        /// CSC built from triplets equals the dense reference entry by entry,
        /// stores no zeros and satisfies (C1)–(C4).
        #[test]
        fn triplets_match_dense_reference((m, n, t) in triplets()) {
            let a = CscMatrix::from_triplets(m, n, &t).unwrap();
            prop_assert!(a.check_invariants().is_ok());
            let d = dense_reference(m, n, &t);
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

        /// CSC → CSR → CSC is the identity, and the CSR holds the same entries.
        #[test]
        fn csc_csr_round_trip((m, n, t) in triplets()) {
            let a = CscMatrix::from_triplets(m, n, &t).unwrap();
            let r = a.to_csr();
            prop_assert!(r.check_invariants().is_ok());
            prop_assert_eq!((r.n_rows(), r.n_cols(), r.nnz()), (m, n, a.nnz()));
            for i in 0..m {
                for j in 0..n {
                    let (i, j) = (Row::from_usize(i), Col::from_usize(j));
                    prop_assert!(approx_eq(r.get(i, j), a.get(i, j), 0.0));
                }
            }
            prop_assert_eq!(r.to_csc(), a.clone());
            prop_assert_eq!(a.transpose().transpose(), a);
        }
    }
}
