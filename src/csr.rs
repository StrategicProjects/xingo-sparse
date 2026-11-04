//! # Chapter 5.8.2 — Compressed sparse row storage
//!
//! ## Purpose
//! [`CsrMatrix`] stores `A` by rows. The simplex method computes the pivot
//! row `α_r = ρ_r A_N` row-wise when `ρ_r` is sparse (Hall & McKinnon 2005),
//! and presolve and domain propagation walk constraints row by row. The CSR
//! copy is derived from the canonical [`CscMatrix`] on demand by
//! [`CscMatrix::to_csr`]; whoever needs it repeatedly (the solver) owns and
//! caches it, so the solver's `Model` stays a plain value.
//!
//! ## Notation
//! Row `i` holds the pairs `(col_idx[p], val[p])` for
//! `p ∈ [row_ptr[i], row_ptr[i+1])`. The arrays `(row_ptr, col_idx, val)` of
//! `A` are exactly the CSC arrays of `Aᵀ`.
//!
//! ## Invariants
//! (C1)–(C4) of Chapter 5.8 with `ptr = row_ptr`, `idx = col_idx`,
//! `n_major = m`, `n_minor = n`; checked by [`CsrMatrix::check_invariants`].
//!
//! ## References
//! Hall & McKinnon, "Hyper-sparsity in the revised simplex method and how to
//! exploit it", Comput. Optim. Appl. 32, 2005; Davis (2006), §2.5.

use crate::compressed::{check_compressed, transpose};
use crate::index::{Col, Row};
use crate::real::Real;
use crate::{CscMatrix, SparseError};

/// Sparse matrix in compressed sparse row (CSR) format.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix<T: Real = f64> {
    /// Number of columns `n`.
    pub n_cols: usize,
    /// Row pointers, length `m + 1`; row `i` occupies
    /// `row_ptr[i]..row_ptr[i + 1]`.
    pub row_ptr: Vec<usize>,
    /// Column index of each stored entry, strictly increasing within a row.
    pub col_idx: Vec<u32>,
    /// Value of each stored entry; finite and nonzero.
    pub val: Vec<T>,
}

impl<T: Real> CsrMatrix<T> {
    /// Builds a matrix from raw compressed arrays, validating every invariant.
    pub fn from_raw_parts(
        n_cols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<u32>,
        val: Vec<T>,
    ) -> Result<Self, SparseError> {
        let r = Self {
            n_cols,
            row_ptr,
            col_idx,
            val,
        };
        r.check_invariants()?;
        Ok(r)
    }

    /// Number of rows `m`.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.row_ptr.len() - 1
    }

    /// Number of columns `n`.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Number of stored entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }

    /// Column indices and values of row `i`.
    #[inline]
    pub fn row(&self, i: Row) -> (&[u32], &[T]) {
        let r = self.row_ptr[i.index()]..self.row_ptr[i.index() + 1];
        (&self.col_idx[r.clone()], &self.val[r])
    }

    /// Entry `a_ij`, or `0` if not stored; `O(log nnz(row i))`.
    pub fn get(&self, i: Row, j: Col) -> T {
        let (cols, vals) = self.row(i);
        cols.binary_search(&j.0).map_or(T::ZERO, |p| vals[p])
    }

    /// The same matrix in column-wise (CSC) storage, by Algorithm T.
    pub fn to_csc(&self) -> CscMatrix<T> {
        let (col_ptr, row_idx, val) =
            transpose(self.n_cols, &self.row_ptr, &self.col_idx, &self.val);
        let a = CscMatrix {
            n_rows: self.n_rows(),
            col_ptr,
            row_idx,
            val,
        };
        debug_assert!(a.check_invariants().is_ok(), "{:?}", a.check_invariants());
        a
    }

    /// Checks invariants (C1)–(C4).
    pub fn check_invariants(&self) -> Result<(), SparseError> {
        check_compressed(
            "row",
            "column",
            self.n_cols,
            &self.row_ptr,
            &self.col_idx,
            &self.val,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::{approx_eq, is_exact_zero};

    #[test]
    fn row_access() {
        let r = CsrMatrix::from_raw_parts(3, vec![0, 2, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0])
            .unwrap();
        assert_eq!(r.n_rows(), 3);
        assert_eq!(r.row(Row(0)), (&[0_u32, 2][..], &[1.0, 2.0][..]));
        assert!(approx_eq(r.get(Row(2), Col(1)), 3.0, 0.0));
        assert!(is_exact_zero(r.get(Row(1), Col(1))));
        assert!(CsrMatrix::from_raw_parts(3, vec![0, 1], vec![3], vec![1.0_f64]).is_err());
    }
}
