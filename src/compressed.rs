//! # Chapter 5.8 — Sparse matrices and vectors
//!
//! *Chapter 1.2 of the first edition, moved here from `xingo-core` with
//! task L1.2 and made generic in the scalar `T: `[`Real`]
//! (Chapter 5.6);
//! `xingo-core` re-exports the `f64` instances at their old paths.*
//!
//! ## Purpose
//! Constraint matrices of real LPs have a handful of nonzeros per column, so
//! the solver stores only the nonzeros. This chapter defines the two
//! compressed matrix formats and the work vector used by the simplex kernels:
//!
//! * [`CscMatrix`](crate::CscMatrix) — compressed sparse column, the canonical storage of `A`
//!   (columns are what the simplex method prices and enters; Chapter
//!   5.8.1);
//! * [`CsrMatrix`](crate::CsrMatrix) — compressed sparse row, the same matrix stored by rows,
//!   built on demand for row-wise operations (pivot row `α_r = ρ_r A_N`,
//!   presolve, propagation; Chapter 5.8.2);
//! * [`SparseVec`](crate::SparseVec) — a dense work array with an explicit nonzero pattern
//!   (Chapter 5.8.3);
//! * [`Triplets`](crate::Triplets) — the coordinate builder: entries pushed in any order,
//!   duplicates summed, compressed by Algorithm C′ (Chapter 5.8.4; task
//!   L1.7a);
//! * [`BlockPartition`](crate::BlockPartition) — the row/column block boundaries of a
//!   partitioned matrix such as `[A | ±I]` or `[−D Aᵀ; A δI]` (Chapter
//!   5.8.5; task L1.7a).
//!
//! ## Notation
//! `A ∈ R^{m×n}` with `nnz` stored entries. In CSC, the entries of column `j`
//! are `row_idx[p], val[p]` for `p ∈ [col_ptr[j], col_ptr[j+1])`. CSR is the
//! same with the roles of rows and columns exchanged; physically, the CSR
//! arrays of `A` are exactly the CSC arrays of `Aᵀ`. We call the indexed
//! dimension (columns for CSC) the *major* one and the other the *minor* one.
//!
//! ## Algorithm T (Transposition of a compressed matrix)
//! Given major pointers `ptr` (length `n_major + 1`), minor indices `idx` and
//! values `val`, produce the compressed arrays of the transpose.
//!
//! T1. [Count.] For each stored entry with minor index `i`, increment
//!     `cnt[i]`.
//! T2. [Prefix sums.] Set `tptr[i] = Σ_{k<i} cnt[k]`; then `tptr[n_minor] =
//!     nnz`. Copy `next ← tptr[0..n_minor]`.
//! T3. [Scatter.] For each major index `j` in increasing order and each
//!     entry `(i, v)` of `j` in storage order, write `(j, v)` at position
//!     `next[i]` and increment `next[i]`.
//!
//! Because T3 visits `j` in increasing order, the major indices inside every
//! output list come out sorted, and entries that tie keep their input order
//! (the sort is stable). Cost: `O(nnz + n_major + n_minor)` time, no
//! comparisons.
//!
//! ## Invariants of a compressed matrix
//! (C1) `ptr.len() = n_major + 1`, `ptr[0] = 0`, `ptr` non-decreasing and
//!      `ptr[n_major] = idx.len() = val.len() = nnz`.
//! (C2) Inside each major list, minor indices are strictly increasing (sorted,
//!      no duplicates) and smaller than `n_minor`.
//! (C3) No explicit zeros: every stored value is finite and nonzero. (The
//!      contract's (S1) admits zeros; the first edition's stricter rule is
//!      kept, since the solver relies on it — e.g. the factorization's
//!      `nnz(B)` statistics and the presolve's counts.)
//! (C4) `n_major` and `n_minor` fit in `u32`.
//!
//! Algorithm T maps a structure satisfying (C1)–(C4) to one satisfying them.
//!
//! ## References
//! Gustavson, "Two fast algorithms for sparse matrices: multiplication and
//! permuted transposition", ACM TOMS 4(3), 1978; Duff, Erisman & Reid,
//! *Direct Methods for Sparse Matrices* (1986), §2.13; Davis, *Direct Methods
//! for Sparse Linear Systems* (SIAM, 2006), §2.4–2.5 (`cs_transpose`,
//! `cs_compress`, `cs_dupl`).

use crate::SparseError;
use crate::index::dim_to_u32;
use crate::real::{Real, is_exact_zero};

/// Compressed arrays `(ptr, idx, val)` of a matrix.
pub(crate) type Compressed<T> = (Vec<usize>, Vec<u32>, Vec<T>);

/// Algorithm T: transposes compressed arrays with `n_major = ptr.len() − 1`
/// major lists and `n_minor` minor indices.
///
/// Preconditions: (C1), indices `< n_minor`, `n_major ≤ u32::MAX`.
pub(crate) fn transpose<T: Real>(
    n_minor: usize,
    ptr: &[usize],
    idx: &[u32],
    val: &[T],
) -> Compressed<T> {
    let n_major = ptr.len() - 1;
    let nnz = ptr[n_major];

    // T1. Count entries per minor index.
    let mut tptr = vec![0_usize; n_minor + 1];
    for &i in &idx[..nnz] {
        tptr[i as usize] += 1;
    }

    // T2. Exclusive prefix sums; tptr[n_minor] ends at nnz.
    let mut acc = 0;
    for c in &mut tptr {
        let cnt = *c;
        *c = acc;
        acc += cnt;
    }
    debug_assert_eq!(tptr[n_minor], nnz);
    let mut next = tptr[..n_minor].to_vec();

    // T3. Scatter by increasing major index (stable).
    let mut tidx = vec![0_u32; nnz];
    let mut tval = vec![T::ZERO; nnz];
    for (j, w) in ptr.windows(2).enumerate() {
        let jj = u32::try_from(j).expect("major dimension exceeds u32::MAX");
        for p in w[0]..w[1] {
            let i = idx[p] as usize;
            let q = next[i];
            next[i] += 1;
            tidx[q] = jj;
            tval[q] = val[p];
        }
    }
    (tptr, tidx, tval)
}

/// Checks invariants (C1)–(C4) of compressed arrays. `major`/`minor` name the
/// dimensions in error messages (e.g. `"column"`, `"row"`).
pub(crate) fn check_compressed<T: Real>(
    major: &'static str,
    minor: &'static str,
    n_minor: usize,
    ptr: &[usize],
    idx: &[u32],
    val: &[T],
) -> Result<(), SparseError> {
    // (C4)
    dim_to_u32(minor, n_minor)?;
    let Some((&first, _)) = ptr.split_first() else {
        return Err(SparseError::Malformed(format!(
            "{major} pointer array is empty"
        )));
    };
    dim_to_u32(major, ptr.len() - 1)?;
    // (C1)
    if first != 0 {
        return Err(SparseError::Malformed(format!(
            "{major} pointer starts at {first}, not 0"
        )));
    }
    let nnz = ptr[ptr.len() - 1];
    if idx.len() != nnz {
        return Err(SparseError::Dimension {
            what: "index array",
            expected: nnz,
            found: idx.len(),
        });
    }
    if val.len() != nnz {
        return Err(SparseError::Dimension {
            what: "value array",
            expected: nnz,
            found: val.len(),
        });
    }
    for (j, w) in ptr.windows(2).enumerate() {
        if w[0] > w[1] {
            return Err(SparseError::Malformed(format!(
                "{major} pointer decreases at {major} {j}: {} > {}",
                w[0], w[1]
            )));
        }
        // (C2)
        let list = &idx[w[0]..w[1]];
        if let Some(&last) = list.last()
            && last as usize >= n_minor
        {
            return Err(SparseError::IndexOutOfRange {
                what: minor,
                index: last as usize,
                bound: n_minor,
            });
        }
        if let Some(k) = list.windows(2).position(|p| p[0] >= p[1]) {
            return Err(SparseError::Malformed(format!(
                "{minor} indices of {major} {j} not strictly increasing at offset {k}"
            )));
        }
    }
    // (C3)
    for (p, &v) in val.iter().enumerate() {
        if !v.is_finite() || is_exact_zero(v) {
            return Err(SparseError::InvalidValue {
                what: "stored value",
                index: p,
                value: v.to_f64(),
            });
        }
    }
    Ok(())
}
