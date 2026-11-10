//! # Chapter 5.16 — Dense kernels for supernodal blocks
//!
//! *New with task L1.5 (PLAN.md trilha L; the contract's §5.8, decision
//! Q4).*
//!
//! ## Purpose
//! A supernodal factorization (L1.6, `SupernodalLdl`) spends its time in a
//! handful of dense operations on the blocks of `L`: factorize a diagonal
//! block, solve the block below it against that factor, and subtract the
//! products into the blocks of the ancestors. [`DenseKernel`] names them
//! with their BLAS/LAPACK names (`potrf`, `sytrf`, `trsm`, `gemm`, `syrk`)
//! so a backend can be swapped without touching the sparse code:
//! [`NaiveDense`] — straightforward loops, generic in `T`, always available,
//! the oracle of the others — and `FaerDense` (feature `faer`), faer's
//! optimized `f64` kernels. A scalar type other than `f64` (the
//! double-double of `xingo-extprec`) uses [`NaiveDense`].
//!
//! ## Notation
//! A [`DenseBlock`] `A` is `rows × cols`, column-major: `a_ij =
//! data[i + j·rows]`. Square blocks are read and written **in their lower
//! triangle only** (the strict upper triangle is left untouched), like
//! LAPACK with `uplo = 'L'`. For [`DenseKernel::sytrf`], `P A Pᵀ = L B Lᵀ`
//! with `P` a permutation ([`Perm`], `fwd[k]` = the original index placed
//! `k`-th), `L` unit lower triangular and `B` block diagonal with `1×1`
//! and `2×2` blocks; `α = (1 + √17)/8 ≈ 0.6404` is Bunch & Kaufman's
//! constant, which minimizes the bound on element growth.
//!
//! ## Algorithm P (`potrf`, Cholesky by columns)
//! For `j = 0, …, n − 1`: `d ← a_jj − Σ_{k<j} l_jk²`; if `d ≤ 0` (or not
//! finite) fail with [`DenseError::NotPositiveDefinite`] at `j`;
//! `l_jj ← √d`; for `i > j`, `l_ij ← (a_ij − Σ_{k<j} l_ik l_jk) / l_jj`.
//!
//! ## Algorithm B (`sytrf`, Bunch–Kaufman partial pivoting)
//! On a working copy `W` of the symmetric matrix, `k ← 0`; while `k < n`:
//! B1. `λ ← max_{i>k} |w_ik|`, `r` its first row index. If `λ = 0`, the
//!     column is already eliminated: a `1×1` pivot `w_kk` (possibly zero).
//! B2. Else if `|w_kk| ≥ α λ`: `1×1` pivot at `k`.
//! B3. Else `σ ← max_{i≥k, i≠r} |w_ir|`: if `|w_kk| σ ≥ α λ²`, `1×1` at
//!     `k`; else if `|w_rr| ≥ α σ`, swap `r` into position `k` and take a
//!     `1×1` pivot; else swap `r` into position `k + 1` and take the `2×2`
//!     pivot `E = W[k..k+2, k..k+2]`.
//! B4. A swap exchanges rows and columns of `W`, the rows of the computed
//!     columns of `L`, and the entries of `P`.
//! B5. Eliminate: `1×1` — `l_ik ← w_ik / w_kk` and `W ← W − l d lᵀ` on the
//!     trailing block; `2×2` — `[l_ik, l_i,k+1] ← [w_ik, w_i,k+1] E⁻¹` and
//!     `W ← W − [w_k w_k+1] E⁻¹ [w_k w_k+1]ᵀ`; advance `k` by 1 or 2.
//! Output in `A`: `B`'s diagonal on the diagonal, the off-diagonal of each
//! `2×2` block at `(k + 1, k)` (where `L` has an implicit zero), `L` below;
//! `pivot_2x2[k] = pivot_2x2[k + 1] = true` for a `2×2` block. The inertia
//! is that of `B` (Sylvester; Chapter 5.14, step K4).
//!
//! ## Algorithms T, G, S (`trsm_lower`, `gemm_sub`, `syrk_sub`)
//! `trsm_lower(L, B, false)`: `B ← L⁻¹ B` by forward substitution, column
//! by column of `B`; `trsm_lower(L, B, true)`: `B ← B L⁻ᵀ`, i.e. the rows
//! of `B` solved against `L` (the "solve the block below the diagonal"
//! step of a left-looking Cholesky). `L` is non-unit lower triangular (the
//! output of `potrf`). `gemm_sub(A, B, C)`: `C ← C − A Bᵀ`. `syrk_sub(A,
//! C)`: `C ← C − A Aᵀ` on the lower triangle of `C`. Sums run over `k` in
//! increasing order.
//!
//! *Loop order of [`NaiveDense`] (task L1.6b).* The first edition summed
//! each entry of the result in place (`c_ij ← c_ij − Σ_k a_ik b_jk`), which
//! reads `A` across its rows — a stride of `rows` between consecutive `k`
//! — and made `SupernodalLdl<f64, NaiveDense>` 14× slower than faer's
//! kernels on dfl001 (D-0099). The kernels now run their innermost loop
//! down one column (contiguous: an `axpy` on the column of `C` with the
//! column `k` of `A` scaled by `b_jk`), in blocks of [`GEMM_BLOCK_ROWS`]
//! rows and [`GEMM_BLOCK_K`] columns of `A` so that the block of `A` in
//! use stays in cache while every column of `C` passes over it. No SIMD
//! and no `unsafe`: plain slices, which the compiler vectorizes. The
//! additions still run over `k` in increasing order, so the result is
//! deterministic; its rounding differs from the first edition's (the sum
//! is accumulated into `c_ij` instead of formed apart and subtracted once).
//!
//! ## Invariants
//! (D1) After `potrf`, `L Lᵀ = A` (lower triangle) to working precision.
//! (D2) After `sytrf`, `P A Pᵀ = L B Lᵀ` to working precision, `B` has
//!      only `1×1` and `2×2` blocks, every `2×2` block has a negative
//!      determinant or is indefinite-safe (`|e_21| ≥ λ > 0`), and the
//!      returned inertia sums to `n`. Checked by the tests against the
//!      reconstruction and against `NaiveDense` vs `FaerDense`.
//! (D3) Every kernel leaves the strict upper triangle of a square block
//!      untouched.
//!
//! ## References
//! Bunch & Kaufman, "Some stable methods for calculating inertia and
//! solving symmetric linear systems", Math. Comp. 31, 1977; Golub & Van
//! Loan, *Matrix Computations*, 4th ed. 2013, §4.2 and §4.4; Higham,
//! *Accuracy and Stability of Numerical Algorithms*, 2nd ed. 2002, ch.
//! 10–11; LAPACK `dpotrf`/`dsytrf`/`dtrsm`/`dgemm`/`dsyrk` (interfaces,
//! read); faer 0.24.4 `linalg::{cholesky, triangular_solve, matmul}` (the
//! API, read, not copied).

use crate::kkt::count_inertia;
use crate::real::{Real, is_exact_zero, le, lt};
use crate::{DenseError, Inertia, Perm};

/// A column-major dense block: `a_ij = data[i + j·rows]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseBlock<T: Real = f64> {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// The entries, column by column (`rows · cols` of them).
    pub data: Vec<T>,
}

impl<T: Real> DenseBlock<T> {
    /// The `rows × cols` zero block.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::ZERO; rows * cols],
        }
    }

    /// Entry `(i, j)`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[i + j * self.rows]
    }

    /// Sets entry `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: T) {
        self.data[i + j * self.rows] = v;
    }

    fn is_consistent(&self) -> bool {
        self.data.len() == self.rows * self.cols
    }

    fn square(&self) -> Result<usize, DenseError> {
        if self.is_consistent() && self.rows == self.cols {
            Ok(self.rows)
        } else {
            Err(DenseError::Dimension)
        }
    }
}

/// The kernels a supernodal factorization needs (BLAS-3 names kept for
/// recognizability). Square blocks are read and written in their lower
/// triangle (invariant (D3)).
pub trait DenseKernel<T: Real> {
    /// In-place Cholesky `A = L Lᵀ` of a square block (Algorithm P).
    fn potrf(&self, a: &mut DenseBlock<T>) -> Result<(), DenseError>;

    /// In-place `P A Pᵀ = L B Lᵀ` with `1×1`/`2×2` pivots (Bunch–Kaufman,
    /// Algorithm B); returns the inertia and writes the permutation into
    /// `perm` and the `2×2` marks into `pivot_2x2` (both resized to `n`).
    fn sytrf(
        &self,
        a: &mut DenseBlock<T>,
        perm: &mut Perm,
        pivot_2x2: &mut Vec<bool>,
    ) -> Result<Inertia, DenseError>;

    /// `B ← B L⁻ᵀ` (`transpose = true`) or `B ← L⁻¹ B` (`false`), `L`
    /// square, non-unit lower triangular.
    ///
    /// # Panics
    /// If the dimensions do not agree.
    fn trsm_lower(&self, l: &DenseBlock<T>, b: &mut DenseBlock<T>, transpose: bool);

    /// `C ← C − A Bᵀ`.
    ///
    /// # Panics
    /// If the dimensions do not agree.
    fn gemm_sub(&self, a: &DenseBlock<T>, b: &DenseBlock<T>, c: &mut DenseBlock<T>);

    /// `C ← C − A Aᵀ`, lower triangle of `C`.
    ///
    /// # Panics
    /// If the dimensions do not agree.
    fn syrk_sub(&self, a: &DenseBlock<T>, c: &mut DenseBlock<T>);

    /// A short name for logs and tables.
    fn name(&self) -> &'static str;
}

/// Reference kernels: straightforward loops, always available, the oracle
/// of the others in tests.
///
/// ```
/// use xingo_sparse::{DenseBlock, DenseKernel, NaiveDense};
/// let mut a = DenseBlock { rows: 2, cols: 2, data: vec![4.0, 2.0, 0.0, 5.0] };
/// NaiveDense.potrf(&mut a).unwrap();
/// assert_eq!((a.get(0, 0), a.get(1, 0), a.get(1, 1)), (2.0, 1.0, 2.0));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NaiveDense;

/// Rows of `C` and `A` handled per block in [`NaiveDense::gemm_sub`] and
/// [`NaiveDense::syrk_sub`] (a cache-blocking size, not a numerical
/// threshold): `GEMM_BLOCK_ROWS × GEMM_BLOCK_K` entries of `A` — 128 KiB
/// in `f64` — stay in cache while the columns of `C` pass over them.
pub const GEMM_BLOCK_ROWS: usize = 256;
/// Columns of `A` (the summation index `k`) per block; see
/// [`GEMM_BLOCK_ROWS`].
pub const GEMM_BLOCK_K: usize = 64;

/// Bunch & Kaufman's `α = (1 + √17)/8` (a constant of Algorithm B, not a
/// tuning knob: it minimizes the growth bound).
fn bk_alpha<T: Real>() -> T {
    (T::ONE + T::from_f64(17.0).sqrt()) / T::from_f64(8.0)
}

impl<T: Real> DenseKernel<T> for NaiveDense {
    fn potrf(&self, a: &mut DenseBlock<T>) -> Result<(), DenseError> {
        let n = a.square()?;
        for j in 0..n {
            let d = a.get(j, j) - (0..j).map(|k| a.get(j, k) * a.get(j, k)).sum::<T>();
            if !d.is_finite() || le(d, T::ZERO, T::ZERO) {
                return Err(DenseError::NotPositiveDefinite { position: j });
            }
            let ljj = d.sqrt();
            a.set(j, j, ljj);
            for i in j + 1..n {
                let s = a.get(i, j) - (0..j).map(|k| a.get(i, k) * a.get(j, k)).sum::<T>();
                a.set(i, j, s / ljj);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)] // Algorithm B, steps B1–B5 in one place
    fn sytrf(
        &self,
        a: &mut DenseBlock<T>,
        perm: &mut Perm,
        pivot_2x2: &mut Vec<bool>,
    ) -> Result<Inertia, DenseError> {
        let n = a.square()?;
        let alpha = bk_alpha::<T>();
        // The working copy W (full, symmetric) and the factor L.
        let mut w = vec![T::ZERO; n * n];
        for j in 0..n {
            for i in j..n {
                w[i + j * n] = a.get(i, j);
                w[j + i * n] = a.get(i, j);
            }
        }
        let mut l = vec![T::ZERO; n * n];
        let mut fwd: Vec<usize> = (0..n).collect();
        let mut two = vec![false; n];
        // `first[k]`: a 2×2 block starts at k.
        let mut first = vec![false; n];
        let mut diag = vec![T::ZERO; n];
        let mut off = vec![T::ZERO; n];
        let swap = |w: &mut [T], l: &mut [T], fwd: &mut [usize], k: usize, r: usize| {
            if k == r {
                return;
            }
            for c in 0..n {
                w.swap(k + c * n, r + c * n);
            }
            for c in 0..n {
                w.swap(c + k * n, c + r * n);
            }
            for c in 0..k {
                l.swap(k + c * n, r + c * n);
            }
            fwd.swap(k, r);
        };
        let mut k = 0;
        while k < n {
            // B1.
            let mut lambda = T::ZERO;
            let mut r = k;
            for i in k + 1..n {
                if lt(lambda, w[i + k * n].abs()) {
                    lambda = w[i + k * n].abs();
                    r = i;
                }
            }
            let akk = w[k + k * n].abs();
            let size = if is_exact_zero(lambda) || !lt(akk, alpha * lambda) {
                1 // B1, B2
            } else {
                // B3.
                let mut sigma = T::ZERO;
                for i in k..n {
                    if i != r {
                        sigma = sigma.max(w[i + r * n].abs());
                    }
                }
                if !lt(akk * sigma, alpha * lambda * lambda) {
                    1
                } else if !lt(w[r + r * n].abs(), alpha * sigma) {
                    swap(&mut w, &mut l, &mut fwd, k, r);
                    1
                } else {
                    swap(&mut w, &mut l, &mut fwd, k + 1, r);
                    2
                }
            };
            // B5.
            if size == 1 {
                let d = w[k + k * n];
                diag[k] = d;
                l[k + k * n] = T::ONE;
                if !is_exact_zero(d) {
                    for i in k + 1..n {
                        l[i + k * n] = w[i + k * n] / d;
                    }
                    for j in k + 1..n {
                        let ljd = l[j + k * n] * d;
                        for i in j..n {
                            let v = w[i + j * n] - l[i + k * n] * ljd;
                            w[i + j * n] = v;
                            w[j + i * n] = v;
                        }
                    }
                }
                k += 1;
            } else {
                let (e11, e21, e22) = (w[k + k * n], w[k + 1 + k * n], w[k + 1 + (k + 1) * n]);
                let det = e11 * e22 - e21 * e21;
                diag[k] = e11;
                diag[k + 1] = e22;
                off[k] = e21;
                two[k] = true;
                two[k + 1] = true;
                first[k] = true;
                l[k + k * n] = T::ONE;
                l[k + 1 + (k + 1) * n] = T::ONE;
                for i in k + 2..n {
                    let (x, y) = (w[i + k * n], w[i + (k + 1) * n]);
                    // [x y] E⁻¹, E⁻¹ = [e22 −e21; −e21 e11] / det.
                    l[i + k * n] = (x * e22 - y * e21) / det;
                    l[i + (k + 1) * n] = (y * e11 - x * e21) / det;
                }
                for j in k + 2..n {
                    let (xj, yj) = (w[j + k * n], w[j + (k + 1) * n]);
                    for i in j..n {
                        let v = w[i + j * n] - (l[i + k * n] * xj + l[i + (k + 1) * n] * yj);
                        w[i + j * n] = v;
                        w[j + i * n] = v;
                    }
                }
                k += 2;
            }
        }
        // Output.
        for j in 0..n {
            a.set(j, j, diag[j]);
            for i in j + 1..n {
                let v = if first[j] && i == j + 1 {
                    off[j]
                } else {
                    l[i + j * n]
                };
                a.set(i, j, v);
            }
        }
        *perm = Perm::from_fwd(fwd.into_iter().map(crate::perm::idx32).collect())
            .map_err(|_| DenseError::Dimension)?;
        pivot_2x2.clear();
        pivot_2x2.extend_from_slice(&two);
        Ok(block_inertia(a, pivot_2x2))
    }

    fn trsm_lower(&self, l: &DenseBlock<T>, b: &mut DenseBlock<T>, transpose: bool) {
        let n = l.square().expect("trsm_lower: L must be square");
        assert!(b.is_consistent(), "trsm_lower: B is inconsistent");
        let rows = b.rows;
        if transpose {
            // X Lᵀ = B: row i of X solves L xᵢ = bᵢ (forward substitution),
            // all rows at once, column by column of B: column j of X is
            // (b_j − Σ_{k<j} l_jk x_k) / l_jj — contiguous axpys.
            assert_eq!(b.cols, n, "trsm_lower: B must have n columns");
            for j in 0..n {
                let (done, rest) = b.data.split_at_mut(j * rows);
                let xj = &mut rest[..rows];
                for k in 0..j {
                    let t = l.get(j, k);
                    for (x, &y) in xj.iter_mut().zip(&done[k * rows..(k + 1) * rows]) {
                        *x -= y * t;
                    }
                }
                let d = l.get(j, j);
                for x in xj {
                    *x = *x / d;
                }
            }
        } else {
            // L X = B, column c of B: right-looking forward substitution,
            // the update of the rows below k contiguous.
            assert_eq!(rows, n, "trsm_lower: B must have n rows");
            for c in 0..b.cols {
                let col = &mut b.data[c * n..(c + 1) * n];
                for k in 0..n {
                    let x = col[k] / l.get(k, k);
                    col[k] = x;
                    let (_, below) = col.split_at_mut(k + 1);
                    for (y, &lik) in below.iter_mut().zip(&l.data[k + 1 + k * n..(k + 1) * n]) {
                        *y -= lik * x;
                    }
                }
            }
        }
    }

    fn gemm_sub(&self, a: &DenseBlock<T>, b: &DenseBlock<T>, c: &mut DenseBlock<T>) {
        assert!(
            a.is_consistent()
                && b.is_consistent()
                && c.is_consistent()
                && a.cols == b.cols
                && c.rows == a.rows
                && c.cols == b.rows,
            "gemm_sub: dimensions do not agree"
        );
        let (rows, kk) = (a.rows, a.cols);
        let mut i0 = 0;
        while i0 < rows {
            let i1 = (i0 + GEMM_BLOCK_ROWS).min(rows);
            let mut k0 = 0;
            while k0 < kk {
                let k1 = (k0 + GEMM_BLOCK_K).min(kk);
                for j in 0..c.cols {
                    let cj = &mut c.data[j * rows + i0..j * rows + i1];
                    for k in k0..k1 {
                        let t = b.data[j + k * b.rows];
                        let ak = &a.data[k * rows + i0..k * rows + i1];
                        for (x, &av) in cj.iter_mut().zip(ak) {
                            *x -= av * t;
                        }
                    }
                }
                k0 = k1;
            }
            i0 = i1;
        }
    }

    fn syrk_sub(&self, a: &DenseBlock<T>, c: &mut DenseBlock<T>) {
        let n = c.square().expect("syrk_sub: C must be square");
        assert!(
            a.is_consistent() && a.rows == n,
            "syrk_sub: dimensions do not agree"
        );
        let kk = a.cols;
        let mut k0 = 0;
        while k0 < kk {
            let k1 = (k0 + GEMM_BLOCK_K).min(kk);
            for j in 0..n {
                let cj = &mut c.data[j * n + j..(j + 1) * n];
                for k in k0..k1 {
                    let t = a.data[j + k * n];
                    let ak = &a.data[k * n + j..(k + 1) * n];
                    for (x, &av) in cj.iter_mut().zip(ak) {
                        *x -= av * t;
                    }
                }
            }
            k0 = k1;
        }
    }

    fn name(&self) -> &'static str {
        "naive"
    }
}

/// The inertia of the block diagonal `B` stored by `sytrf`.
fn block_inertia<T: Real>(a: &DenseBlock<T>, pivot_2x2: &[bool]) -> Inertia {
    let n = a.rows;
    let mut ones = Vec::new();
    let mut twos = Vec::new();
    let mut k = 0;
    while k < n {
        if pivot_2x2[k] && k + 1 < n {
            twos.push((a.get(k, k), a.get(k + 1, k), a.get(k + 1, k + 1)));
            k += 2;
        } else {
            ones.push(a.get(k, k));
            k += 1;
        }
    }
    count_inertia(ones.into_iter(), twos.into_iter())
}

/// faer's dense kernels (`f64` only in v0.1; extended types use
/// [`NaiveDense`]). `sytrf` uses faer's Bunch–Kaufman with partial
/// pivoting, whose pivot choices may differ from [`NaiveDense`]'s in ties:
/// both satisfy (D2).
#[cfg(feature = "faer")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaerDense;

#[cfg(feature = "faer")]
mod faer_impl {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::cholesky::lblt::factor::{LbltParams, PivotingStrategy};
    use faer::linalg::cholesky::llt::factor::LltRegularization;
    use faer::linalg::matmul::triangular::BlockStructure;
    use faer::{Accum, ColMut, MatMut, MatRef, Par, Spec};

    use super::{DenseBlock, DenseError, DenseKernel, FaerDense, Inertia, Perm, block_inertia};
    use crate::real::is_exact_zero;

    fn view(a: &DenseBlock<f64>) -> MatRef<'_, f64> {
        MatRef::from_column_major_slice(&a.data, a.rows, a.cols)
    }

    fn view_mut(a: &mut DenseBlock<f64>) -> MatMut<'_, f64> {
        MatMut::from_column_major_slice_mut(&mut a.data, a.rows, a.cols)
    }

    impl DenseKernel<f64> for FaerDense {
        fn potrf(&self, a: &mut DenseBlock<f64>) -> Result<(), DenseError> {
            let n = a.square()?;
            let params = Spec::default();
            let mut buf = MemBuffer::new(
                faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
                    n,
                    Par::Seq,
                    params,
                ),
            );
            faer::linalg::cholesky::llt::factor::cholesky_in_place(
                view_mut(a),
                LltRegularization::default(),
                Par::Seq,
                MemStack::new(&mut buf),
                params,
            )
            .map(|_| ())
            .map_err(|e| match e {
                faer::linalg::cholesky::llt::factor::LltError::NonPositivePivot { index } => {
                    DenseError::NotPositiveDefinite { position: index }
                }
            })
        }

        fn sytrf(
            &self,
            a: &mut DenseBlock<f64>,
            perm: &mut Perm,
            pivot_2x2: &mut Vec<bool>,
        ) -> Result<Inertia, DenseError> {
            let n = a.square()?;
            let mut params: Spec<LbltParams, f64> = Spec::default();
            params.config.pivoting = PivotingStrategy::Partial;
            let mut subdiag = vec![0.0_f64; n];
            let mut fwd = vec![0_usize; n];
            let mut inv = vec![0_usize; n];
            let mut buf = MemBuffer::new(
                faer::linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, f64>(
                    n,
                    Par::Seq,
                    params,
                ),
            );
            faer::linalg::cholesky::lblt::factor::cholesky_in_place(
                view_mut(a),
                ColMut::from_slice_mut(&mut subdiag).as_diagonal_mut(),
                &mut fwd,
                &mut inv,
                Par::Seq,
                MemStack::new(&mut buf),
                params,
            );
            pivot_2x2.clear();
            pivot_2x2.resize(n, false);
            let mut k = 0;
            while k < n {
                if k + 1 < n && !is_exact_zero(subdiag[k]) {
                    a.set(k + 1, k, subdiag[k]);
                    pivot_2x2[k] = true;
                    pivot_2x2[k + 1] = true;
                    k += 2;
                } else {
                    k += 1;
                }
            }
            *perm = Perm::from_fwd(fwd.into_iter().map(crate::perm::idx32).collect())
                .map_err(|_| DenseError::Dimension)?;
            Ok(block_inertia(a, pivot_2x2))
        }

        fn trsm_lower(&self, l: &DenseBlock<f64>, b: &mut DenseBlock<f64>, transpose: bool) {
            let n = l.square().expect("trsm_lower: L must be square");
            if transpose {
                assert_eq!(b.cols, n, "trsm_lower: B must have n columns");
                // X Lᵀ = B  ⇔  L Xᵀ = Bᵀ.
                faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                    view(l),
                    view_mut(b).transpose_mut(),
                    Par::Seq,
                );
            } else {
                assert_eq!(b.rows, n, "trsm_lower: B must have n rows");
                faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                    view(l),
                    view_mut(b),
                    Par::Seq,
                );
            }
        }

        fn gemm_sub(&self, a: &DenseBlock<f64>, b: &DenseBlock<f64>, c: &mut DenseBlock<f64>) {
            assert!(
                a.cols == b.cols && c.rows == a.rows && c.cols == b.rows,
                "gemm_sub: dimensions do not agree"
            );
            faer::linalg::matmul::matmul(
                view_mut(c),
                Accum::Add,
                view(a),
                view(b).transpose(),
                -1.0,
                Par::Seq,
            );
        }

        fn syrk_sub(&self, a: &DenseBlock<f64>, c: &mut DenseBlock<f64>) {
            let n = c.square().expect("syrk_sub: C must be square");
            assert_eq!(a.rows, n, "syrk_sub: dimensions do not agree");
            faer::linalg::matmul::triangular::matmul(
                view_mut(c),
                BlockStructure::TriangularLower,
                Accum::Add,
                view(a),
                BlockStructure::Rectangular,
                view(a).transpose(),
                BlockStructure::Rectangular,
                -1.0,
                Par::Seq,
            );
        }

        fn name(&self) -> &'static str {
            "faer"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::approx_eq;
    use proptest::prelude::*;

    /// A deterministic pseudo-random block (xorshift).
    fn block(rows: usize, cols: usize, seed: u64) -> DenseBlock {
        let mut s = seed | 1;
        let data = (0..rows * cols)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                #[allow(clippy::cast_precision_loss)]
                let u = (s >> 11) as f64 / (1_u64 << 53) as f64;
                2.0 * u - 1.0
            })
            .collect();
        DenseBlock { rows, cols, data }
    }

    /// A symmetric block: `M + Mᵀ` (indefinite in general), or
    /// `M Mᵀ + n I` (positive definite).
    fn symmetric(n: usize, seed: u64, spd: bool) -> DenseBlock {
        let m = block(n, n, seed);
        let mut a = DenseBlock::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                #[allow(clippy::cast_precision_loss)]
                let v = if spd {
                    (0..n).map(|k| m.get(i, k) * m.get(j, k)).sum::<f64>()
                        + if i == j { n as f64 } else { 0.0 }
                } else {
                    m.get(i, j) + m.get(j, i)
                };
                a.set(i, j, v);
            }
        }
        a
    }

    fn close(x: &DenseBlock, y: &DenseBlock, lower_only: bool) -> bool {
        (0..x.cols).all(|j| {
            (if lower_only { j } else { 0 }..x.rows)
                .all(|i| approx_eq(x.get(i, j), y.get(i, j), 1e-10))
        })
    }

    /// `‖P A Pᵀ − L B Lᵀ‖_max` from the output of `sytrf`.
    fn sytrf_error(a0: &DenseBlock, f: &DenseBlock, perm: &Perm, two: &[bool]) -> f64 {
        let n = a0.rows;
        let mut l = DenseBlock::zeros(n, n);
        let mut b = DenseBlock::zeros(n, n);
        let mut k = 0;
        while k < n {
            if two[k] {
                b.set(k, k, f.get(k, k));
                b.set(k + 1, k + 1, f.get(k + 1, k + 1));
                b.set(k + 1, k, f.get(k + 1, k));
                b.set(k, k + 1, f.get(k + 1, k));
                l.set(k, k, 1.0);
                l.set(k + 1, k + 1, 1.0);
                for i in k + 2..n {
                    l.set(i, k, f.get(i, k));
                    l.set(i, k + 1, f.get(i, k + 1));
                }
                k += 2;
            } else {
                b.set(k, k, f.get(k, k));
                l.set(k, k, 1.0);
                for i in k + 1..n {
                    l.set(i, k, f.get(i, k));
                }
                k += 1;
            }
        }
        let mut err = 0.0_f64;
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0;
                for p in 0..n {
                    for q in 0..n {
                        s += l.get(i, p) * b.get(p, q) * l.get(j, q);
                    }
                }
                let (pi, pj) = (perm.fwd[i] as usize, perm.fwd[j] as usize);
                let aij = if pi >= pj {
                    a0.get(pi, pj)
                } else {
                    a0.get(pj, pi)
                };
                err = err.max((s - aij).abs());
            }
        }
        err
    }

    #[test]
    fn potrf_rejects_indefinite_blocks() {
        let mut a = DenseBlock {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 0.0, 1.0],
        };
        assert_eq!(
            NaiveDense.potrf(&mut a.clone()),
            Err(DenseError::NotPositiveDefinite { position: 1 })
        );
        #[cfg(feature = "faer")]
        assert_eq!(
            FaerDense.potrf(&mut a),
            Err(DenseError::NotPositiveDefinite { position: 1 })
        );
        let mut r = DenseBlock::<f64>::zeros(2, 3);
        assert_eq!(NaiveDense.potrf(&mut r), Err(DenseError::Dimension));
        let mut p = Perm::identity(0);
        let mut t = Vec::new();
        assert_eq!(
            NaiveDense.sytrf(&mut r, &mut p, &mut t),
            Err(DenseError::Dimension)
        );
        a.data.pop();
        assert_eq!(NaiveDense.potrf(&mut a), Err(DenseError::Dimension));
    }

    #[test]
    fn sytrf_finds_2x2_pivots_and_inertia() {
        // [0 1; 1 0]: no 1×1 pivot is acceptable.
        let mut a = DenseBlock {
            rows: 2,
            cols: 2,
            data: vec![0.0, 1.0, 0.0, 0.0],
        };
        let mut p = Perm::identity(0);
        let mut two = Vec::new();
        let s = NaiveDense.sytrf(&mut a, &mut p, &mut two).unwrap();
        assert_eq!(
            s,
            Inertia {
                positive: 1,
                negative: 1,
                zero: 0
            }
        );
        assert_eq!(two, vec![true, true]);
        // A singular block has a zero pivot.
        let mut z = DenseBlock {
            rows: 2,
            cols: 2,
            data: vec![1.0, 0.0, 0.0, 0.0],
        };
        let s = NaiveDense.sytrf(&mut z, &mut p, &mut two).unwrap();
        assert_eq!(s.zero, 1);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]
        /// (D1), (D2) and NaiveDense vs FaerDense: equal to working
        /// precision (factors of potrf, trsm, gemm, syrk entry by entry;
        /// sytrf by reconstruction and identical inertia).
        #[test]
        fn kernels_agree(n in 1_usize..24, k in 1_usize..9, seed in any::<u64>()) {
            // potrf.
            let spd = symmetric(n, seed, true);
            let mut x = spd.clone();
            NaiveDense.potrf(&mut x).unwrap();
            let mut llt = DenseBlock::zeros(n, n);
            for j in 0..n {
                for i in j..n {
                    llt.set(i, j, (0..=j).map(|p| x.get(i, p) * x.get(j, p)).sum());
                }
            }
            prop_assert!(close(&llt, &spd, true));
            // sytrf on an indefinite block.
            let ind = symmetric(n, seed ^ 0x9e37, false);
            let mut f = ind.clone();
            let mut p = Perm::identity(0);
            let mut two = Vec::new();
            let s = NaiveDense.sytrf(&mut f, &mut p, &mut two).unwrap();
            prop_assert_eq!(s.positive + s.negative + s.zero, n);
            prop_assert!(sytrf_error(&ind, &f, &p, &two) <= 1e-9);
            // trsm, gemm, syrk.
            let b = block(k, n, seed ^ 0x7);
            let mut t1 = b.clone();
            NaiveDense.trsm_lower(&x, &mut t1, true);
            let b2 = block(n, k, seed ^ 0xb);
            let mut t2 = b2.clone();
            NaiveDense.trsm_lower(&x, &mut t2, false);
            let a3 = block(k, n, seed ^ 0xd);
            let mut c3 = block(k, k, seed ^ 0x11);
            let c3_0 = c3.clone();
            NaiveDense.gemm_sub(&a3, &b, &mut c3);
            for i in 0..k {
                for j in 0..k {
                    let v: f64 = c3_0.get(i, j) - (0..n).map(|p| a3.get(i, p) * b.get(j, p)).sum::<f64>();
                    prop_assert!(approx_eq(v, c3.get(i, j), 1e-12));
                }
            }
            let mut c4 = symmetric(k, seed ^ 0x13, false);
            let c4_0 = c4.clone();
            NaiveDense.syrk_sub(&a3, &mut c4);
            #[cfg(feature = "faer")]
            {
                let mut y = spd.clone();
                FaerDense.potrf(&mut y).unwrap();
                prop_assert!(close(&x, &y, true));
                let mut g = ind.clone();
                let mut q = Perm::identity(0);
                let mut two_f = Vec::new();
                let sf = FaerDense.sytrf(&mut g, &mut q, &mut two_f).unwrap();
                prop_assert_eq!(s, sf);
                prop_assert!(sytrf_error(&ind, &g, &q, &two_f) <= 1e-9);
                let mut u1 = b.clone();
                FaerDense.trsm_lower(&x, &mut u1, true);
                prop_assert!(close(&t1, &u1, false));
                let mut u2 = b2.clone();
                FaerDense.trsm_lower(&x, &mut u2, false);
                prop_assert!(close(&t2, &u2, false));
                let mut d3 = c3_0.clone();
                FaerDense.gemm_sub(&a3, &b, &mut d3);
                prop_assert!(close(&c3, &d3, false));
                let mut d4 = c4_0.clone();
                FaerDense.syrk_sub(&a3, &mut d4);
                prop_assert!(close(&c4, &d4, true));
                // (D3): the strict upper triangle is untouched.
                prop_assert!((0..k).all(|j| (0..j).all(|i| approx_eq(d4.get(i, j), c4_0.get(i, j), 0.0))));
            }
            prop_assert!((0..k).all(|j| (0..j).all(|i| approx_eq(c4.get(i, j), c4_0.get(i, j), 0.0))));
            // trsm really solved: t1 Lᵀ = b.
            for i in 0..k {
                for j in 0..n {
                    let v: f64 = (0..=j).map(|p| t1.get(i, p) * x.get(j, p)).sum();
                    prop_assert!(approx_eq(v, b.get(i, j), 1e-9));
                }
            }
        }
    }
}
