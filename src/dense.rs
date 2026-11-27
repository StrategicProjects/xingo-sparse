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
//! ## Algorithm P (`potrf`, Cholesky by columns, left-looking)
//! For `j = 0, …, n − 1`:
//! P1. \[Update.\] For `k = 0, …, j − 1`, in this order: `a_ij ← a_ij −
//!     l_ik · l_jk` for `i = j, …, n − 1` (a contiguous column update
//!     with the column `k` of `L` from row `j` down, scaled by `l_jk`).
//! P2. \[Pivot.\] `d ← a_jj`; if `d ≤ 0` or `d` is not finite, fail with
//!     [`DenseError::NotPositiveDefinite`] at `j`; `l_jj ← √d`.
//! P3. \[Scale.\] `l_ij ← a_ij / l_jj` for `i > j`.
//!
//! *Loop order (task L1.8).* Until 0.3.x each entry was formed as `a_ij −
//! Σ_{k<j} l_ik l_jk` with the sum accumulated apart, along a row of `L`
//! (a stride of `n` between terms), which no kernel can run on vector
//! lanes with the same bits. P1 subtracts term by term, the statement of
//! `gemm_sub` and `trsm_lower` (`x ← x − a · t`, the entry of the column
//! first), so a kernel that vectorizes those vectorizes `potrf` bit for
//! bit. The operands, the order of `k` and the order of the factors are
//! those of 0.3.x; the factor differs from 0.3.x's in rounding from the
//! third column on (and possibly in the sign of a zero in the first), no
//! bound changes (Higham 2002, Theorem 10.3, holds for either order), and
//! a block on the edge of definiteness may fail in one order and pass in
//! the other. On failure the lower triangle is left partially factorized,
//! as before.
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
//! ## Algorithm B′ (`FaerDense::sytrf`; task I08)
//! *faer's partial pivoting.* faer 0.24.4's Bunch–Kaufman with partial
//! pivoting (`linalg::cholesky::lblt::factor`, `lblt_unblocked` and
//! `lblt_blocked_step`, strategy `PivotingStrategy::Partial`; reported as
//! sarah-quinones/faer-rs#288) states the second test of B3 as `|w_kk| ≥ α σ
//! (σ/λ)` — `(alpha * gamma_r) * (gamma_r / gamma_i)` — where Bunch &
//! Kaufman (and LAPACK `dsytf2`, `ABSAKK ≥ ALPHA·COLMAX·(COLMAX/ROWMAX)`)
//! have `|w_kk| ≥ α λ (λ/σ)`, i.e. `|w_kk| σ ≥ α λ²`. Since `σ ≥ λ`, faer's
//! threshold is the larger: when `α λ²/σ ≤ |w_kk| < α σ²/λ`, Bunch–Kaufman
//! takes the `1×1` pivot `w_kk` and faer goes on — to the `1×1` pivot `w_rr`
//! (`|w_rr| ≥ α σ`: stable) or to the `2×2` pivot `E`, which then lacks the
//! bound `|e₁₁ e₂₂| < α² e₂₁²` the test exists to give: `det E` may be zero
//! (`[1 2 0; 2 4 8; 0 8 0]`, nonsingular: `E = [1 2; 2 4]`, and faer's
//! `1/(d₀₀ d₁₁ − 1)` is `1/0`, a NaN in `L`), or tiny, and the growth is
//! unbounded (Algorithm M then sees a zero pivot or a huge `L₂₁`). Until
//! 0.5.x this kernel asked faer for `Partial`, with no recorded reason
//! (Algorithm M does not depend on it: it needs only that a column with no
//! off-diagonal entry be a `1×1` pivot without elimination, which every
//! strategy of faer does).
//!
//! *Rook pivoting.* This kernel asks faer for `PivotingStrategy::Rook`,
//! whose tests are right: a `2×2` pivot is taken only on two columns `i₀,
//! i₁` each of which holds the largest off-diagonal entry of the other
//! (`γ = λ = σ = |e₂₁|`) with `|e₁₁|, |e₂₂| < α γ`, so `|e₁₁ e₂₂| < α² λ²`
//! and `|det E| > (1 − α²) λ²`; a `1×1` pivot `d` has `|d| ≥ α γ` (Ashcraft,
//! Grimes & Lewis 1998). Its pivots are not those of [`NaiveDense`]'s
//! partial pivoting (Algorithm B); both satisfy (D2).
//!
//! The check of Algorithm B′ stays, as a net:
//! B′1. Copy the lower triangle of `A`; run faer's rook-pivoting `LBLᵀ`.
//! B′2. For each `2×2` block `E` at `k` faer returned, with `λ = |e₂₁|` and
//!      `σ = max(λ, max_{i≥k+2} |w_{i,k+1}|)`, where `w_{i,k+1} = l_ik e₂₁ +
//!      l_i,k+1 e₂₂` recovers column `k + 1` of the working matrix at the
//!      step (`[l_ik l_i,k+1] = [w_ik w_i,k+1] E⁻¹`, step B5): accept iff
//!      `E`, the `l`s and `σ` are finite and `|e₁₁| σ < α λ²` — B3's
//!      condition for a `2×2` pivot, which a rook `2×2` pivot meets (`σ = λ`,
//!      `|e₁₁| < α λ`) up to the rounding of the recovered `σ`.
//! B′3. If every block is accepted, return faer's factor; else restore the
//!      copy and run Algorithm B ([`NaiveDense`]'s `sytrf`) on it.
//! A block that passes B′2 has `|e₁₁ e₂₂| < α λ²/σ · α σ = α² λ²`, so
//! `|det E| > (1 − α²) λ²`, as in Bunch–Kaufman. `σ` in B′2 is recomputed
//! and carries rounding: a block on the edge of the test may be refused
//! (then Algorithm B runs: a different but equally valid factor), never a
//! block far from it. Algorithm B has no blocking (`O(n³)` in plain loops):
//! when B′3 runs on a large front it costs far more than faer's kernel —
//! which is why the strategy, not the net, is the fix.
//!
//! *Measured (task I08).* Algorithm M (`Indefinite`, `Regularization::none()`)
//! on 3 005 random `L₀ Δ₀ L₀ᵀ` (exact integers, order ≤ 27) × the three
//! option sets of the tests: 9 015 factorizations — refused, factors with
//! another inertia, and factors that differ in any bit from 0.5.x's among
//! its 8 960 successes; and the median of three numeric factorizations of
//! random augmented `[−D Aᵀ; A 0]` with large fronts (order `10⁴`, front
//! 1 898; `6·10⁴`, front 3 587; `1.5·10⁵`, front 1 439, one run), release
//! build, one session:
//!
//! | `sytrf` | refused | wrong inertia | bits changed | `10⁴` | `6·10⁴` | `1.5·10⁵` |
//! |---|---|---|---|---|---|---|
//! | faer `Partial` (0.5.x) | 55 | 0 | — | 0.12 s | refused | 22.7 s |
//! | faer `Partial` + B′ | 0 | 0 | 1 131 | 1.05 s | 13.4 s | 22.2 s |
//! | faer `Rook` + B′ (this kernel) | 0 | 0 | 2 203 | 0.12 s | 4.2 s | 22.7 s |
//!
//! With `Partial`, B′3 ran on 302 blocks of the 9 303 of the small
//! matrices (default options) and on 31 blocks of more than 64 columns of
//! the large ones, up to 1 898 columns — the cost in the second row. With
//! `Rook` it ran on none.
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
//! ## Algorithms V, W (`solve_unit_lower`, `solve_unit_lower_t`; task L1.8)
//! The two halves of a supernode's part in Algorithm S of Chapter 5.17,
//! on the stored block `L` (`n_r × n_c`, `n_r ≥ n_c`, unit lower
//! trapezoidal: the diagonal slots hold `Δ̂` and are never read) and a
//! contiguous vector `y` of `n_r` entries — the `n_c` unknowns of the
//! supernode followed by the gathered entries of its pattern rows.
//! `skip[j]` (empty: all false) marks the first column of a `2×2` pivot,
//! whose slot `(j + 1, j)` holds `e₂₁` and is not an entry of `L`.
//!
//! V1. For `j = 0, …, n_c − 1`, with `o = j + 1 + [skip_j]`:
//!     `y_i ← y_i − l_ij · y_j` for `i = o, …, n_r − 1`, in this order.
//!
//! W1. For `j = n_c − 1, …, 0`, with `o` as in V1: `a ← D(l_{o..n_c, j},
//!     y_{o..n_c})`, `b ← D(l_{n_c.., j}, y_{n_c..})`, `y_j ← y_j − (a + b)`.
//! D.  `D(u, v)` over `n` entries: four accumulators `a_k`, `k = i mod 4`
//!     over the first `4⌊n/4⌋` entries, `a_k ← a_k + u_i · v_i`; a fifth,
//!     `t`, over the rest; result `((a₀ + a₁) + (a₂ + a₃)) + t`.
//!
//! Every product has the entry of `L` first. These are the operations of
//! the second edition's solve entry by entry (each `y_i` of V1 receives
//! its updates in increasing `j`, whether the rows are those of the
//! diagonal block or the gathered ones), so:
//!
//! (D4) The provided methods return the bits of the solve of 0.3.x. An
//!      implementation *should* return the bits of the provided methods;
//!      one that does not must say so in its documentation (as `FaerDense`
//!      does for the factorization kernels). `NaiveDense` and `FaerDense`
//!      do not override them in 0.4.0. Tested with a kernel that sends
//!      every supernode through the provided methods (Chapter 5.17).
//! (D5) [`DenseKernel::solve_min_rows`] is a cost switch, never a
//!      numerical one: by (D4) the solution does not depend on it for a
//!      conforming kernel.
//!
//! ## Invariants
//! (D1) After `potrf`, `L Lᵀ = A` (lower triangle) to working precision.
//! (D2) After `sytrf`, `P A Pᵀ = L B Lᵀ` to working precision, `B` has
//!      only `1×1` and `2×2` blocks, every `2×2` block has a negative
//!      determinant or is indefinite-safe (`|e_21| ≥ λ > 0`), and the
//!      returned inertia sums to `n`. Checked by the tests against the
//!      reconstruction and against `NaiveDense` vs `FaerDense`. Every `2×2`
//!      block satisfies Bunch–Kaufman's test `|e₁₁| σ < α λ²` (for
//!      `FaerDense` by step B′2, with the recovered `σ`).
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
//! API, read, not copied) and `linalg::cholesky::bunch_kaufman::factor`
//! (the pivot tests of `lblt_unblocked` and `lblt_blocked_step`, partial
//! and rook, read for Algorithm B′); Ashcraft, Grimes & Lewis, "Accurate
//! symmetric indefinite linear equation solvers", SIAM J. Matrix Anal.
//! Appl. 20(2), 1998 (rook pivoting).

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

    /// The smallest row count `n_r` of a supernode block for which the
    /// solves of `SupernodalLdl` go through this kernel (gather, hook,
    /// scatter; step S0′ of Chapter 5.17). `usize::MAX` — the default —
    /// keeps the loops of the second edition everywhere. A cost switch
    /// only (invariant (D5)).
    fn solve_min_rows(&self) -> usize {
        usize::MAX
    }

    /// Forward half of a supernode's solve, `y ← L⁻¹ y` (Algorithm V).
    /// `l` is the stored block (`rows ≥ cols`, unit lower trapezoidal,
    /// diagonal slots ignored); `skip` is empty or has `l.cols` marks (the
    /// first column of each `2×2` pivot).
    ///
    /// # Panics
    /// If `y.len() != l.rows`, `l.rows < l.cols`, `l` is inconsistent, or
    /// `skip` has another length.
    fn solve_unit_lower(&self, l: &DenseBlock<T>, skip: &[bool], y: &mut [T]) {
        check_solve(l, skip, y);
        let (nr, nc) = (l.rows, l.cols);
        for j in 0..nc {
            let o = solve_offset(skip, j);
            let xj = y[j];
            let col = &l.data[j * nr..(j + 1) * nr];
            for (yi, &lij) in y[o..].iter_mut().zip(&col[o..]) {
                *yi -= lij * xj;
            }
        }
    }

    /// Backward half, `y_{..cols} ← L₁₁⁻ᵀ (y_{..cols} − L₂₁ᵀ y_{cols..})`
    /// (Algorithm W); `y[cols..]` is read only.
    ///
    /// # Panics
    /// As [`DenseKernel::solve_unit_lower`].
    fn solve_unit_lower_t(&self, l: &DenseBlock<T>, skip: &[bool], y: &mut [T]) {
        check_solve(l, skip, y);
        let (nr, nc) = (l.rows, l.cols);
        for j in (0..nc).rev() {
            let o = solve_offset(skip, j);
            let col = &l.data[j * nr..(j + 1) * nr];
            let a = dot4(&col[o..nc], &y[o..nc]);
            let b = dot4(&col[nc..], &y[nc..]);
            y[j] -= a + b;
        }
    }
}

/// The argument checks of Algorithms V and W.
fn check_solve<T: Real>(l: &DenseBlock<T>, skip: &[bool], y: &[T]) {
    assert!(
        l.is_consistent()
            && l.rows >= l.cols
            && y.len() == l.rows
            && (skip.is_empty() || skip.len() == l.cols),
        "solve_unit_lower: dimensions do not agree"
    );
}

/// `o = j + 1 + [skip_j]` of Algorithms V and W.
#[inline]
fn solve_offset(skip: &[bool], j: usize) -> usize {
    j + 1 + usize::from(skip.get(j).copied().unwrap_or(false))
}

/// Step D of Algorithm W: `Σ u_i v_i` with four independent partial sums
/// over the chunks of four and a fifth over the tail, combined as
/// `((a₀ + a₁) + (a₂ + a₃)) + t` (the dependent chain of a single
/// accumulator bounds the backward sweep by the latency of the
/// multiply-add, not by memory: 2× on maros-r7's normal equations, L1.6a).
/// The order of the additions is fixed, so the result is deterministic;
/// the backward sweep of Algorithm S of Chapter 5.17 uses it too.
#[inline]
pub(crate) fn dot4<T: Real>(u: &[T], v: &[T]) -> T {
    let mut acc = [T::ZERO; 4];
    let (cu, ru) = u.as_chunks::<4>();
    let (cv, rv) = v.as_chunks::<4>();
    for (pu, pv) in cu.iter().zip(cv) {
        for k in 0..4 {
            acc[k] += pu[k] * pv[k];
        }
    }
    let mut tail = T::ZERO;
    for (&p, &q) in ru.iter().zip(rv) {
        tail += p * q;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
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
            let (done, rest) = a.data.split_at_mut(j * n);
            let cj = &mut rest[j..n];
            // P1.
            for k in 0..j {
                let t = done[j + k * n];
                for (x, &l) in cj.iter_mut().zip(&done[j + k * n..(k + 1) * n]) {
                    *x -= l * t;
                }
            }
            // P2.
            let d = cj[0];
            if !d.is_finite() || le(d, T::ZERO, T::ZERO) {
                return Err(DenseError::NotPositiveDefinite { position: j });
            }
            let ljj = d.sqrt();
            cj[0] = ljj;
            // P3.
            for x in &mut cj[1..] {
                *x = *x / ljj;
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
/// [`NaiveDense`]). `sytrf` uses faer's Bunch–Kaufman with rook pivoting,
/// whose pivots differ from [`NaiveDense`]'s partial pivoting: both satisfy
/// (D2). A factor whose `2×2` pivots fail Bunch–Kaufman's test is refused
/// and the block refactorized by [`NaiveDense`] (Algorithm B′ of Chapter
/// 5.16; faer's *partial* pivoting, which this kernel used until 0.5.x,
/// can return a singular `2×2` pivot).
#[cfg(feature = "faer")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaerDense;

#[cfg(feature = "faer")]
pub(crate) mod faer_impl {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::cholesky::lblt::factor::{LbltParams, PivotingStrategy};
    use faer::linalg::cholesky::llt::factor::LltRegularization;
    use faer::linalg::matmul::triangular::BlockStructure;
    use faer::{Accum, ColMut, MatMut, MatRef, Par, Spec};

    use super::{
        DenseBlock, DenseError, DenseKernel, FaerDense, Inertia, NaiveDense, Perm, bk_alpha,
        block_inertia,
    };
    use crate::real::{is_exact_zero, lt};

    /// Step B′2 of Algorithm B′: every `2×2` block of the factor `f` (in
    /// the layout of [`DenseKernel::sytrf`]) passes Bunch–Kaufman's test
    /// `|e₁₁| σ < α λ²`, with `λ = |e₂₁|` and `σ` the largest off-diagonal
    /// entry of the block's second column at its step, recovered from `L`
    /// and `E`; a non-finite value fails.
    pub(super) fn bunch_kaufman_2x2_holds(f: &DenseBlock<f64>, two: &[bool]) -> bool {
        let n = f.rows;
        let alpha = bk_alpha::<f64>();
        let mut k = 0;
        while k < n {
            if !two[k] {
                k += 1;
                continue;
            }
            let (e11, e21, e22) = (f.get(k, k), f.get(k + 1, k), f.get(k + 1, k + 1));
            let lambda = e21.abs();
            let mut sigma = lambda;
            for i in k + 2..n {
                let w = (f.get(i, k) * e21 + f.get(i, k + 1) * e22).abs();
                if !w.is_finite() {
                    return false;
                }
                sigma = sigma.max(w);
            }
            let lhs = e11.abs() * sigma;
            let rhs = alpha * lambda * lambda;
            if !(lhs.is_finite() && rhs.is_finite() && e22.is_finite() && lt(lhs, rhs)) {
                return false;
            }
            k += 2;
        }
        true
    }

    fn view(a: &DenseBlock<f64>) -> MatRef<'_, f64> {
        MatRef::from_column_major_slice(&a.data, a.rows, a.cols)
    }

    fn view_mut(a: &mut DenseBlock<f64>) -> MatMut<'_, f64> {
        MatMut::from_column_major_slice_mut(&mut a.data, a.rows, a.cols)
    }

    /// faer's `LBLᵀ` with the given pivoting `strategy`, in place, in the
    /// layout of [`DenseKernel::sytrf`] (`e₂₁` written at `(k + 1, k)`, the
    /// marks in `pivot_2x2`); returns the permutation's `fwd`. Step B′1 of
    /// Algorithm B′ (with [`PivotingStrategy::Rook`]) without the check of
    /// B′2; with [`PivotingStrategy::Partial`], the kernel of 0.5.x.
    pub(crate) fn faer_lblt(
        a: &mut DenseBlock<f64>,
        pivot_2x2: &mut Vec<bool>,
        strategy: PivotingStrategy,
    ) -> Result<Vec<usize>, DenseError> {
        let n = a.square()?;
        let mut params: Spec<LbltParams, f64> = Spec::default();
        params.config.pivoting = strategy;
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
        Ok(fwd)
    }

    /// Algorithm B′ with faer's pivoting `strategy`: [`FaerDense::sytrf`]
    /// with [`PivotingStrategy::Rook`].
    pub(crate) fn faer_sytrf_checked(
        a: &mut DenseBlock<f64>,
        perm: &mut Perm,
        pivot_2x2: &mut Vec<bool>,
        strategy: PivotingStrategy,
    ) -> Result<Inertia, DenseError> {
        a.square()?;
        // B′1.
        let copy = a.data.clone();
        let fwd = faer_lblt(a, pivot_2x2, strategy)?;
        // B′2, B′3.
        if !bunch_kaufman_2x2_holds(a, pivot_2x2) {
            a.data.copy_from_slice(&copy);
            return NaiveDense.sytrf(a, perm, pivot_2x2);
        }
        *perm = Perm::from_fwd(fwd.into_iter().map(crate::perm::idx32).collect())
            .map_err(|_| DenseError::Dimension)?;
        Ok(block_inertia(a, pivot_2x2))
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
            faer_sytrf_checked(a, perm, pivot_2x2, PivotingStrategy::Rook)
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

    /// Algorithm P entry by entry (task L1.8): the factors of 32 random
    /// `5×5` blocks are, bit for bit, those of a reference written with the
    /// statement `x ← x − a · t` spelled out (entry of the column first, `k`
    /// increasing), and some differ from the 0.3.x order (a sum formed
    /// apart).
    #[test]
    #[allow(clippy::needless_range_loop)] // entries (i, j) of a matrix
    fn potrf_follows_its_statement_entry_by_entry() {
        let n = 5;
        let bits = |x: f64| x.to_bits();
        let mut differs = 0;
        for seed in 1..=32_u64 {
            let a0 = symmetric(n, seed, true);
            let mut got = a0.clone();
            NaiveDense.potrf(&mut got).unwrap();
            let mut l = vec![vec![0.0_f64; n]; n];
            for (i, row) in l.iter_mut().enumerate() {
                for (j, x) in row.iter_mut().enumerate().take(i + 1) {
                    *x = a0.get(i, j);
                }
            }
            for j in 0..n {
                for k in 0..j {
                    let t = l[j][k];
                    for row in l.iter_mut().skip(j) {
                        let a = row[k];
                        row[j] -= a * t;
                    }
                }
                let ljj = l[j][j].sqrt();
                l[j][j] = ljj;
                for row in l.iter_mut().skip(j + 1) {
                    row[j] /= ljj;
                }
            }
            for j in 0..n {
                for i in j..n {
                    assert_eq!(
                        bits(got.get(i, j)),
                        bits(l[i][j]),
                        "seed {seed}: ({i}, {j})"
                    );
                }
                for i in 0..j {
                    assert_eq!(
                        bits(got.get(i, j)),
                        bits(a0.get(i, j)),
                        "(D3) at ({i}, {j})"
                    );
                }
            }
            // The 0.3.x order: each entry's sum formed apart.
            let mut old = vec![vec![0.0_f64; n]; n];
            for j in 0..n {
                let d = a0.get(j, j) - (0..j).map(|k| old[j][k] * old[j][k]).sum::<f64>();
                old[j][j] = d.sqrt();
                for i in j + 1..n {
                    let s = a0.get(i, j) - (0..j).map(|k| old[i][k] * old[j][k]).sum::<f64>();
                    old[i][j] = s / old[j][j];
                }
            }
            if (0..n).any(|j| (j..n).any(|i| bits(old[i][j]) != bits(l[i][j]))) {
                differs += 1;
            }
        }
        // The statement is not the 0.3.x one: some blocks round differently.
        assert!(differs > 0, "no block tells the two orders apart");
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

    /// Algorithm B′ on the smallest block faer 0.24.4's partial pivoting
    /// takes a singular pivot on (task I08; faer-rs#288): `[1 2 0; 2 4 8; 0
    /// 8 0]` (det `−64`). Bunch–Kaufman takes the `1×1` pivot `1` (`|w₀₀| σ
    /// = 8 ≥ α λ² = 2.56`); faer's `Partial`, with `λ` and `σ` exchanged (`1
    /// < α σ²/λ = 20.5`), takes `E = [1 2; 2 4]`, which is singular, and its
    /// factor holds non-finite values — the sentinel of the faer defect:
    /// when faer is fixed, the first assertions fail. Algorithm B′ with
    /// `Partial` refuses it (B′2) and returns Algorithm B's factor bit for
    /// bit (B′3). `FaerDense` (rook) takes the `2×2` pivot `[4 8; 8 0]` on
    /// rows 1 and 2, which B′2 accepts.
    #[cfg(feature = "faer")]
    #[test]
    fn faer_dense_refuses_faers_singular_2x2_pivot() {
        use faer::linalg::cholesky::lblt::factor::PivotingStrategy;
        let a0 = DenseBlock {
            rows: 3,
            cols: 3,
            data: vec![1.0, 2.0, 0.0, 9.0, 4.0, 8.0, 9.0, 9.0, 0.0],
        };
        // faer alone (the kernel of 0.5.x): the singular 2×2 pivot.
        let mut raw = a0.clone();
        let mut two = Vec::new();
        let fwd = faer_impl::faer_lblt(&mut raw, &mut two, PivotingStrategy::Partial).unwrap();
        assert_eq!(fwd, vec![0, 1, 2]);
        assert_eq!(two, vec![true, true, false]);
        assert_eq!(
            (raw.get(0, 0), raw.get(1, 0), raw.get(1, 1)),
            (1.0, 2.0, 4.0)
        );
        assert!(raw.data.iter().any(|x| !x.is_finite()));
        assert!(!faer_impl::bunch_kaufman_2x2_holds(&raw, &two));
        // Algorithm B′ with Partial: Algorithm B's factor, bit for bit.
        let (mut f, mut g) = (a0.clone(), a0.clone());
        let (mut p, mut q) = (Perm::identity(0), Perm::identity(0));
        let (mut tf, mut tg) = (Vec::new(), Vec::new());
        let sf = faer_impl::faer_sytrf_checked(&mut f, &mut p, &mut tf, PivotingStrategy::Partial)
            .unwrap();
        let sg = NaiveDense.sytrf(&mut g, &mut q, &mut tg).unwrap();
        assert_eq!(
            sf,
            Inertia {
                positive: 2,
                negative: 1,
                zero: 0
            }
        );
        assert_eq!((sf, &p, &tf), (sg, &q, &tg));
        let bits = |b: &DenseBlock| b.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&f), bits(&g));
        // (D3): the strict upper triangle (the 9s) is untouched.
        assert_eq!((f.get(0, 1), f.get(0, 2), f.get(1, 2)), (9.0, 9.0, 9.0));
        assert!(sytrf_error(&a0, &f, &p, &tf) <= 1e-12);
        // FaerDense (rook): faer's factor, accepted by B′2.
        let mut h = a0.clone();
        let mut th = Vec::new();
        let sh = FaerDense.sytrf(&mut h, &mut p, &mut th).unwrap();
        assert_eq!(sh, sf);
        assert_eq!(p.fwd, vec![1, 2, 0]);
        assert_eq!(th, vec![true, true, false]);
        assert_eq!((h.get(0, 0), h.get(1, 0), h.get(1, 1)), (4.0, 8.0, 0.0));
        assert!(faer_impl::bunch_kaufman_2x2_holds(&h, &th));
        assert!(h.data.iter().all(|x| x.is_finite()));
        assert_eq!((h.get(0, 1), h.get(0, 2), h.get(1, 2)), (9.0, 9.0, 9.0));
        assert!(sytrf_error(&a0, &h, &p, &th) <= 1e-12);
        // A 2×2 pivot that Bunch–Kaufman takes is kept: [0 1; 1 0].
        let mut b = DenseBlock {
            rows: 2,
            cols: 2,
            data: vec![0.0, 1.0, 0.0, 0.0],
        };
        let mut t = Vec::new();
        faer_impl::faer_lblt(&mut b.clone(), &mut t, PivotingStrategy::Partial).unwrap();
        FaerDense.sytrf(&mut b, &mut p, &mut t).unwrap();
        assert_eq!(t, vec![true, true]);
        assert!(faer_impl::bunch_kaufman_2x2_holds(&b, &t));
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
                // (D2): every 2×2 block passes Bunch–Kaufman's test (B′2).
                prop_assert!(faer_impl::bunch_kaufman_2x2_holds(&g, &two_f));
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
