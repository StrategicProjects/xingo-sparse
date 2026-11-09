//! # Chapter 5.14 — KKT systems: analyze once, factorize many
//!
//! *New with task L1.5 (PLAN.md trilha L; the contract's §5.7).*
//!
//! ## Purpose
//! At every iteration a primal-dual interior-point method solves a linear
//! system whose matrix has a *fixed pattern* and *changing values*: the
//! normal equations `A D Aᵀ` or the augmented (KKT) system
//! `[−D, Aᵀ; A, 0]`, where the diagonal `D > 0` spreads over many orders
//! of magnitude as the iterates approach the boundary (Wright 1997, ch.
//! 11). The interface below is the shape of CHOLMOD (Chen, Davis, Hager &
//! Rajamanickam 2008): a *symbolic* phase that depends on the pattern only
//! (ordering, elimination tree, structure of the factor) is done once, a
//! *numeric* phase is repeated for every new set of values, and a *solve*
//! phase uses the factor. On top of it sit two things that every backend
//! shares and that therefore live here, not in a backend:
//!
//! * the **regularization policy** (Altman & Gondzio 1999): which static
//!   shifts `ρ`, `δ` enter which diagonal block, which sign each pivot must
//!   have, which pivots a backend may bump, and the inertia check (S6);
//! * **iterative refinement** against the matrix the caller really wants
//!   solved ([`KktSolver::solve_refined`], a provided method), which keeps
//!   its best iterate (S7).
//!
//! ## Notation
//! `K ∈ R^{N×N}` symmetric, given by its **lower triangle** in a
//! [`CscMatrix`] (entries `i ≥ j`; CHOLMOD's `stype < 0`). A matrix that
//! stores both triangles is accepted when its pattern is symmetric: the
//! entries above the diagonal are then ignored. Two forms ([`KktForm`]):
//!
//! * `NormalEquations`: `K = A D Aᵀ` (`N = m`), positive (semi)definite;
//! * `Augmented { n_primal: n, n_dual: m }`: `K = [−D, Aᵀ; A, 0]`
//!   (`N = n + m`), the `n` primal unknowns first.
//!
//! A [`Regularization`] `(ρ, δ, τ, β)` = (`primal`, `dual`, `pivot_tol`,
//! `bump`) defines the matrix the backend factorizes,
//! ```text
//!   K_reg = K + diag(−ρ I_n, δ I_m)   (augmented),   K_reg = K + δ I_m   (normal),
//! ```
//! and the *pivot signs* `σ_k = −1` for the primal rows, `+1` for the dual
//! rows (all `+1` for the normal equations). The *inertia* of a symmetric
//! matrix is the triple ([`Inertia`]) of the numbers of positive, negative
//! and zero eigenvalues; by Sylvester's law it equals the count of
//! positive, negative and zero pivots of any `P K Pᵀ = L Δ Lᵀ` with `Δ`
//! diagonal, and, for `Δ` block diagonal with `1×1` and `2×2` blocks,
//! the sum of the inertias of the blocks. The *relative residual* of `x`
//! is `res(x) = ‖K x − b‖_∞ / (1 + ‖b‖_∞)`.
//!
//! ## Why the regularized augmented system needs no pivoting
//! With `D > 0`, `ρ > 0` and `δ > 0`, `K_reg` is *quasi-definite*: its
//! `(1,1)` block is negative definite and its `(2,2)` block positive
//! definite. Vanderbei (1995) proved that every symmetric permutation of a
//! quasi-definite matrix has an `LDLᵀ` factorization with diagonal `Δ`,
//! the first block's pivots negative and the second's positive; Gill,
//! Saunders & Shinnerl (1996) bound its stability by the size of the
//! regularization. So the ordering can be chosen for sparsity alone, at
//! analysis time — the property that makes "analyze once" possible for the
//! augmented system — and the inertia is known in advance: `(m, n, 0)`.
//! Without regularization (`ρ = 0` or `δ = 0`) this guarantee is lost, and
//! a backend needs `2×2` pivots (Bunch & Kaufman 1977) or dynamic
//! regularization to factorize.
//!
//! *On the sign convention.* The contract's (S6) writes the expected
//! inertia of the augmented system as `(n, m, 0)` for the matrix
//! `[−D − ρI, Aᵀ; A, δI]`; for that matrix the `n` primal pivots are
//! *negative* and the `m` dual pivots *positive*, so with the field order
//! `(positive, negative, zero)` of [`Inertia`] the triple is
//! `{ positive: m, negative: n, zero: 0 }` — which is what this chapter
//! checks (`n` primal and `m` dual eigenvalues, each block with its own
//! sign; listed for ratification in L1.5).
//!
//! ## Algorithm K (the regularization policy shared by every backend)
//! K1. [Validate.] `K` satisfies (S1); it is square, of order `N` equal to
//!     the form's (`n + m` for the augmented system); its lower triangle
//!     lies inside the pattern given to the analysis (missing entries are
//!     zeros: an IPM value that became exactly zero leaves the CSC
//!     pattern, (C3) of Chapter 5.8).
//! K2. [Static regularization.] Assemble `K_reg`: the lower triangle of `K`
//!     plus a full diagonal, with `−ρ` added to the primal diagonal and
//!     `+δ` to the dual one (normal equations: `+δ` everywhere; `ρ` does
//!     not apply, the caller folds it into `D`).
//! K2′. [Symmetric equilibration (Ruiz 2001).] Scale `K_reg` to
//!     `K̃ = S K_reg S`, `S = diag(s)` with powers of two: start from
//!     `s = 1`; in each pass compute, for every row `i` of the *scaled*
//!     matrix (both triangles), `r_i = max_j |k̃_ij|` and multiply `s_i` by
//!     `⌊1/√r_i⌉₂` (rows without entries keep `s_i`); stop after a pass
//!     that changes no `s_i` — exact, the factors being powers of two — or
//!     after [`EQUILIBRATION_PASSES`] passes. Every row of `K̃` then has
//!     its largest entry in `[2^{-1}, 2^{1}]` roughly, whatever the spread
//!     of `D`. The scaling is a congruence, so the inertia of `K̃` is that
//!     of `K_reg` (Sylvester). Being by powers of two it is exact, and an
//!     `LDLᵀ` without pivoting *commutes* with it: the factor of `K̃` is
//!     `(S L S⁻¹)(S Δ S)(S L S⁻¹)ᵀ` bit for bit, barring overflow and
//!     underflow. So K2′ changes exactly two things: the thresholds `τ`, `β`
//!     of K3 become *relative* to the magnitude of each row, and the
//!     intermediate values stay in range. The backend factorizes `K̃` and
//!     solves with `S F̃⁻¹ S`.
//! K3. [Dynamic regularization.] If `τ = pivot_tol` is `Some`, the backend
//!     replaces every pivot `d_k` with `σ_k d_k ≤ τ` (too small, or of the
//!     wrong sign) by `σ_k β` while it eliminates — the dynamic part of
//!     Altman & Gondzio's (1999) primal-dual regularization, in the form
//!     faer's `LdltRegularization` applies — and counts the bumped pivots. Bumping
//!     needs the sign `σ_k`, which only the form knows — the policy is
//!     here; the elimination loop that applies it is the backend's.
//! K4. [Inertia (S6).] From the pivots of the factor, count the inertia.
//!     If it is not `(N, 0, 0)` (normal) or `(m, n, 0)` (augmented),
//!     return [`KktError::WrongInertia`]. With `τ = Some`, every pivot has
//!     the sign `σ_k` by K3, so the check cannot fail: the factor is that
//!     of a perturbation of `K_reg` with the right inertia, and refinement
//!     (Algorithm F) measures how far it is from `K`.
//!
//! ## Choosing the regularization
//! Static shifts work only above the rounding level of the matrix: a
//! shift `δ < ε_T · max|k_ij|` changes nothing in floating point. At the
//! end of an interior-point method `D` spreads to `10^{±10}` and
//! `max|K| ≈ 10^{10}`, so the absolute `ρ = δ = 10⁻⁸` of
//! [`Regularization::default`] no longer makes `K_reg` numerically
//! quasi-definite: on the Netlib LPs of L1.5 it produced wrong-signed
//! pivots in 139 of 558 systems, and 32 of them failed even with bumping
//! (overflow, or a residual above `10⁻¹⁰`). Shifts relative to the matrix,
//! [`Regularization::relative_to`]`(k, 10⁻⁸)`, gave the right inertia
//! without a single bump and a refined residual below `10⁻¹²` in all 558
//! (L1.5, D3). The default keeps the absolute values of the contract;
//! an interior-point method should pass relative ones.
//!
//! ## Algorithm F (Iterative refinement with the best iterate; (S7))
//! Given a factor `F ≈ K_reg`, the matrix `K` to be solved (the
//! *unregularized* one, or `K_reg` itself — whichever the caller passes),
//! `b` and a [`Refinement`] `(max_iters, target)`:
//! F1. `x ← F⁻¹ b`; `r ← b − K x`; `ρ ← res(x)`; `x* ← x`, `ρ* ← ρ`;
//!     `t ← 0`.
//! F2. If `ρ* ≤ target`, `t = max_iters`, or `ρ` is not finite, go to F4.
//! F3. `t ← t + 1`; `x ← x + F⁻¹ r`; `r ← b − K x`; `ρ ← res(x)`. If
//!     `ρ < ρ*`, set `x* ← x`, `ρ* ← ρ`. Go to F2.
//! F4. Return `x*` with `iters = t`, `residual = ρ*` and
//!     `converged = (ρ* ≤ target)`.
//!
//! The iteration continues from its current iterate even when a step does
//! not improve on the best (the residual of fixed-precision refinement
//! need not decrease monotonically near its floor, and stopping at the
//! first non-improving step left `wood1p` at `4·10⁻⁹`, L1.5, D5); (S7)
//! holds because only `x*` is returned.
//!
//! The residual is computed with the stored lower triangle of `K`
//! (`y_i += k_ij x_j` and, for `i ≠ j`, `y_j += k_ij x_i`), in the working
//! precision of `T` — fixed-precision refinement, which brings the
//! componentwise backward error of a backward-stable factorization down to
//! the order of the unit roundoff (Skeel 1980; Higham 2002, ch. 12); with
//! a regularized factor it also removes the bias of the shifts (Altman &
//! Gondzio 1999, §6). When `K` is singular but `b` is consistent the
//! iteration still converges in the residual.
//!
//! ## Invariants
//! (K1) The pattern given to [`KktSolver::analyze`] is square and either
//!      lower triangular or structurally symmetric
//!      ([`KktError::NotSymmetric`] otherwise).
//! (S6) (the contract's) [`KktSolver::factorize`] returns the inertia of
//!      the factor it computed; it is `(N, 0, 0)` for the normal equations
//!      and `(m, n, 0)` = `{positive: m, negative: n, zero: 0}` for the
//!      augmented system, or the call fails with
//!      [`KktError::WrongInertia`]. Asserted on every successful
//!      factorization in debug builds.
//! (S7) (the contract's) [`KktSolver::solve_refined`] never returns a
//!      residual larger than that of its first solve: it keeps the best
//!      iterate (step F3). Asserted in debug builds.
//!
//! ## References
//! Chen, Davis, Hager & Rajamanickam, "Algorithm 887: CHOLMOD, supernodal
//! sparse Cholesky factorization and update/downdate", ACM TOMS 35(3),
//! 2008 (the analyze/factorize/solve shape, the lower-triangle input, the
//! supernodal/simplicial selection by flops per `nnz(L)` — read, not
//! copied); Altman & Gondzio, "Regularized symmetric indefinite systems in
//! interior point methods for linear and quadratic optimization", Optim.
//! Methods Softw. 11–12, 1999; Vanderbei, "Symmetric quasidefinite
//! matrices", SIAM J. Optim. 5(1), 1995; Gill, Saunders & Shinnerl, "On
//! the stability of Cholesky factorization for symmetric quasidefinite
//! systems", SIAM J. Matrix Anal. Appl. 17(1), 1996; Wright,
//! *Primal-Dual Interior-Point Methods*, SIAM 1997, ch. 11; Bunch &
//! Kaufman, "Some stable methods for calculating inertia and solving
//! symmetric linear systems", Math. Comp. 31, 1977; Skeel, "Iterative
//! refinement implies numerical stability for Gaussian elimination", Math.
//! Comp. 35, 1980; Higham, *Accuracy and Stability of Numerical
//! Algorithms*, 2nd ed. 2002, ch. 11–12; Ruiz, "A scaling algorithm to
//! equilibrate both rows and columns norms in matrices", RAL-TR-2001-034,
//! 2001 (symmetric variant, rounded to powers of two here); Stellato et al., "OSQP: an
//! operator splitting solver for quadratic programs", Math. Prog. Comp.
//! 12, 2020 (quasi-definite `LDLᵀ` without pivoting in an optimization
//! solver).

use crate::real::{Real, is_exact_zero, is_nan, le, lt};
use crate::{CscMatrix, KktError, Ordering, Pattern, SparseError};

/// Inertia `(positive, negative, zero)` of a symmetric factorization: the
/// numbers of positive, negative and zero pivots (eigenvalues, by
/// Sylvester's law).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Inertia {
    /// Positive pivots.
    pub positive: usize,
    /// Negative pivots.
    pub negative: usize,
    /// Zero pivots.
    pub zero: usize,
}

/// Regularization of a quasi-definite KKT matrix (Altman & Gondzio 1999;
/// Algorithm K).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Regularization<T: Real = f64> {
    /// `ρ ≥ 0`, added to the primal block (which becomes `−D − ρI`);
    /// ignored for the normal equations.
    pub primal: T,
    /// `δ ≥ 0`, added to the dual block (`δI`), or to the whole diagonal
    /// of the normal equations (`A D Aᵀ + δI`).
    pub dual: T,
    /// Below this magnitude (or with the wrong sign) a pivot is bumped to
    /// `±bump` with the sign the inertia expects (dynamic regularization,
    /// step K3); `None` disables it.
    pub pivot_tol: Option<T>,
    /// The magnitude `β > 0` of a bumped pivot.
    pub bump: T,
}

impl<T: Real> Default for Regularization<T> {
    /// `ρ = δ = 10⁻⁸`, and pivots of the wrong sign or of magnitude at
    /// most `τ = 10⁻¹³` bumped to `±β = ±2·10⁻⁷` — the order of the
    /// defaults of current interior-point codes (Clarabel's documented
    /// static and dynamic regularization; not tuned here). `β ≠ δ` on
    /// purpose: a dual pivot equal to `δ` is not mistaken for a bump by
    /// the backends that count bumps by value (Chapter 5.15).
    fn default() -> Self {
        Self {
            primal: T::from_f64(1e-8),
            dual: T::from_f64(1e-8),
            pivot_tol: Some(T::from_f64(1e-13)),
            bump: T::from_f64(2e-7),
        }
    }
}

impl<T: Real> Regularization<T> {
    /// No regularization at all: `ρ = δ = 0`, no bumping (the factorization
    /// then fails on a zero pivot, or with the wrong inertia).
    #[must_use]
    pub fn none() -> Self {
        Self {
            primal: T::ZERO,
            dual: T::ZERO,
            pivot_tol: None,
            bump: T::ZERO,
        }
    }

    /// Static shifts relative to the magnitude of `k`: `ρ = δ = factor ·
    /// max|k_ij|`, with the dynamic part of [`Regularization::default`].
    ///
    /// An absolute `δ` stops regularizing once it falls below the rounding
    /// level `ε_T · max|k_ij|` — at the end of an interior-point method,
    /// where `D` spreads to `10^{±10}`, `10⁻⁸` does (Chapter 5.14, "Choosing
    /// the regularization"). `factor = 10⁻⁸` passed every Netlib case of
    /// L1.5 in the three regimes.
    #[must_use]
    pub fn relative_to(k: &CscMatrix<T>, factor: T) -> Self {
        let big = inf_norm(&k.val);
        Self {
            primal: factor * big,
            dual: factor * big,
            ..Self::default()
        }
    }

    /// `ρ > 0` and `δ > 0`: the regularized augmented matrix is
    /// quasi-definite (it factorizes without pivoting, in any order).
    pub fn is_quasi_definite(&self) -> bool {
        lt(T::ZERO, self.primal) && lt(T::ZERO, self.dual)
    }
}

/// Iterative refinement parameters (Algorithm F).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Refinement<T: Real = f64> {
    /// Refinement steps after the first solve.
    pub max_iters: usize,
    /// Stop when `‖K x − b‖_∞ / (1 + ‖b‖_∞)` is at or below this.
    pub target_residual: T,
}

impl<T: Real> Default for Refinement<T> {
    /// Ten steps, target `10⁻¹²`.
    fn default() -> Self {
        Self {
            max_iters: 10,
            target_residual: T::from_f64(1e-12),
        }
    }
}

/// What refinement achieved (S7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefineReport<T: Real = f64> {
    /// Refinement steps taken (solves after the first).
    pub iters: usize,
    /// `‖K x − b‖_∞ / (1 + ‖b‖_∞)` of the returned `x`.
    pub residual: T,
    /// `residual ≤ target_residual`.
    pub converged: bool,
}

/// Which system the interior-point method hands over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KktForm {
    /// `A D Aᵀ + δI`, positive definite.
    NormalEquations,
    /// `[−D − ρI, Aᵀ; A, δI]`, quasi-definite; the `n_primal` primal rows
    /// come first.
    Augmented {
        /// `n`, the order of the `(1,1)` block.
        n_primal: usize,
        /// `m`, the order of the `(2,2)` block.
        n_dual: usize,
    },
}

impl KktForm {
    /// The inertia (S6) requires of a factor of order `n`.
    pub fn expected_inertia(self, n: usize) -> Inertia {
        match self {
            Self::NormalEquations => Inertia {
                positive: n,
                negative: 0,
                zero: 0,
            },
            Self::Augmented { n_primal, n_dual } => Inertia {
                positive: n_dual,
                negative: n_primal,
                zero: 0,
            },
        }
    }
}

/// Analyze/factorize/solve of symmetric systems with a fixed pattern (the
/// CHOLMOD shape, read, not copied). Regularization (its policy,
/// Algorithm K) and refinement (Algorithm F) are shared by every backend.
///
/// ```
/// # #[cfg(feature = "faer")] {
/// use xingo_sparse::{
///     Amd, Col, CscMatrix, FaerLdl, KktForm, KktSolver, Refinement, Regularization, Row,
/// };
/// // K = [−D, Aᵀ; A, 0] with D = diag(1, 2), A = [1 1]: lower triangle.
/// let k = CscMatrix::from_triplets(3, 3, &[
///     (Row(0), Col(0), -1.0), (Row(2), Col(0), 1.0),
///     (Row(1), Col(1), -2.0), (Row(2), Col(1), 1.0),
/// ])
/// .unwrap();
/// let form = KktForm::Augmented { n_primal: 2, n_dual: 1 };
/// let solver = FaerLdl::default();
/// let sym = solver.analyze(&k.pattern(), form, &Amd::default()).unwrap();
/// let (num, inertia) = solver.factorize(&sym, &k, Regularization::default()).unwrap();
/// assert_eq!(inertia, form.expected_inertia(3)); // (S6): 1 positive, 2 negative
/// let b = [1.0, 0.0, 3.0];
/// let mut x = [0.0; 3];
/// let report = solver.solve_refined(&num, &k, &b, &mut x, Refinement::default());
/// assert!(report.converged && report.residual <= 1e-12);
/// # }
/// ```
pub trait KktSolver<T: Real> {
    /// The result of the symbolic phase.
    type Symbolic;
    /// The result of the numeric phase.
    type Numeric;

    /// Symbolic phase on the lower triangle of `pattern` (plus a full
    /// diagonal) with the given ordering; done once per pattern.
    fn analyze(
        &self,
        pattern: &Pattern,
        form: KktForm,
        ordering: &dyn Ordering,
    ) -> Result<Self::Symbolic, KktError>;

    /// Numeric phase on the lower triangle of `k` (inside the analyzed
    /// pattern) with regularization (Algorithm K); returns the factor and
    /// its inertia (S6).
    fn factorize(
        &self,
        symbolic: &Self::Symbolic,
        k: &CscMatrix<T>,
        reg: Regularization<T>,
    ) -> Result<(Self::Numeric, Inertia), KktError>;

    /// `x ← F⁻¹ x` in place, dense, with `F` the regularized factor.
    ///
    /// # Panics
    /// If `x.len()` is not the order of the factor.
    fn solve(&self, numeric: &Self::Numeric, x: &mut [T]);

    /// Solve followed by iterative refinement against `k` (Algorithm F);
    /// keeps the best iterate (S7). `k` is the matrix to be solved — the
    /// unregularized one the factor approximates, or any other with the
    /// same order; only its lower triangle is read. `x` is overwritten.
    ///
    /// # Panics
    /// If `b`, `x` and `k` do not have the order of the factor.
    fn solve_refined(
        &self,
        numeric: &Self::Numeric,
        k: &CscMatrix<T>,
        b: &[T],
        x: &mut [T],
        refinement: Refinement<T>,
    ) -> RefineReport<T> {
        let n = b.len();
        assert_eq!(x.len(), n, "solve_refined: x has the wrong length");
        assert!(
            k.n_rows == n && k.n_cols() == n,
            "solve_refined: k has the wrong order"
        );
        let scale = T::ONE + inf_norm(b);
        // F1.
        x.copy_from_slice(b);
        self.solve(numeric, x);
        let mut r = vec![T::ZERO; n];
        residual(k, x, b, &mut r);
        let first = inf_norm(&r) / scale;
        let mut best = first;
        let mut iters = 0;
        let mut d = vec![T::ZERO; n];
        let mut cur = x.to_vec();
        let mut res = first;
        // F2–F3.
        while res.is_finite()
            && !le(best, refinement.target_residual, T::ZERO)
            && iters < refinement.max_iters
        {
            iters += 1;
            d.copy_from_slice(&r);
            self.solve(numeric, &mut d);
            for (c, &di) in cur.iter_mut().zip(&d) {
                *c += di;
            }
            residual(k, &cur, b, &mut r);
            res = inf_norm(&r) / scale;
            if res.is_finite() && lt(res, best) {
                x.copy_from_slice(&cur);
                best = res;
            }
        }
        // F4.
        debug_assert!(
            !best.is_finite() || !first.is_finite() || le(best, first, T::ZERO),
            "(S7): the returned residual exceeds the first one"
        );
        RefineReport {
            iters,
            residual: best,
            converged: best.is_finite() && le(best, refinement.target_residual, T::ZERO),
        }
    }

    /// Nonzeros of the factor (for the IPM log).
    fn nnz_factor(&self, numeric: &Self::Numeric) -> usize;

    /// A short name for logs and tables.
    fn name(&self) -> &'static str;
}

/// `‖v‖_∞` (`0` for an empty vector; NaN if an entry is NaN — `max`
/// alone would hide it).
pub(crate) fn inf_norm<T: Real>(v: &[T]) -> T {
    let mut m = T::ZERO;
    for &x in v {
        if is_nan(x) {
            return x;
        }
        m = m.max(x.abs());
    }
    m
}

/// `y ← K x` with `K` given by the lower triangle of `k` (entries above
/// the diagonal ignored).
pub(crate) fn sym_lower_matvec<T: Real>(k: &CscMatrix<T>, x: &[T], y: &mut [T]) {
    y.fill(T::ZERO);
    for (j, &xj) in x.iter().enumerate() {
        for p in k.col_ptr[j]..k.col_ptr[j + 1] {
            let i = k.row_idx[p] as usize;
            if i < j {
                continue;
            }
            let v = k.val[p];
            y[i] += v * xj;
            if i != j {
                y[j] += v * x[i];
            }
        }
    }
}

/// `r ← b − K x` (lower triangle of `k`).
fn residual<T: Real>(k: &CscMatrix<T>, x: &[T], b: &[T], r: &mut [T]) {
    sym_lower_matvec(k, x, r);
    for (ri, &bi) in r.iter_mut().zip(b) {
        *ri = bi - *ri;
    }
}

/// The lower triangle of a symmetric pattern plus a full diagonal, as the
/// symbolic phase of every backend consumes it (step K1 for the pattern,
/// invariant (K1)). The order `N` is checked against the form.
#[cfg_attr(not(feature = "faer"), allow(dead_code))] // used by every backend
pub(crate) fn lower_with_diagonal(pattern: &Pattern, form: KktForm) -> Result<Pattern, KktError> {
    pattern.check_invariants()?;
    let n = pattern.n_cols();
    if pattern.n_rows != n {
        return Err(KktError::NotSymmetric);
    }
    if let KktForm::Augmented { n_primal, n_dual } = form
        && n_primal + n_dual != n
    {
        return Err(KktError::Structure(SparseError::Dimension {
            what: "KKT order (n_primal + n_dual)",
            expected: n_primal + n_dual,
            found: n,
        }));
    }
    let has_upper = (0..n).any(|j| pattern.col(j).first().is_some_and(|&i| (i as usize) < j));
    if has_upper && pattern.transpose() != *pattern {
        return Err(KktError::NotSymmetric);
    }
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::with_capacity(pattern.nnz() / 2 + n);
    col_ptr.push(0);
    for j in 0..n {
        let rows = pattern.col(j);
        let start = rows.partition_point(|&i| (i as usize) < j);
        let below = &rows[start..];
        if below.first() != Some(&crate::perm::idx32(j)) {
            row_idx.push(crate::perm::idx32(j));
        }
        row_idx.extend_from_slice(below);
        col_ptr.push(row_idx.len());
    }
    let l = Pattern {
        n_rows: n,
        col_ptr,
        row_idx,
    };
    debug_assert_eq!(l.check_invariants(), Ok(()));
    Ok(l)
}

/// The regularization is usable: `ρ, δ, β ≥ 0` finite, and with dynamic
/// regularization `τ > 0` and `β > 0`.
#[cfg_attr(not(feature = "faer"), allow(dead_code))] // used by every backend
pub(crate) fn check_regularization<T: Real>(reg: &Regularization<T>) -> Result<(), KktError> {
    for (what, x) in [
        ("primal regularization", reg.primal),
        ("dual regularization", reg.dual),
        ("bump", reg.bump),
    ] {
        if !x.is_finite() || lt(x, T::ZERO) {
            return Err(KktError::Structure(SparseError::InvalidValue {
                what,
                index: 0,
                value: x.to_f64(),
            }));
        }
    }
    if let Some(tau) = reg.pivot_tol
        && (!tau.is_finite() || !lt(T::ZERO, tau) || !lt(T::ZERO, reg.bump))
    {
        return Err(KktError::Structure(SparseError::InvalidValue {
            what: "pivot_tol and bump (both must be positive)",
            index: 0,
            value: tau.to_f64(),
        }));
    }
    Ok(())
}

/// The pivot signs `σ_k` of step K3: `−1` on the primal rows, `+1` on the
/// dual rows (all `+1` for the normal equations).
#[cfg_attr(not(feature = "faer"), allow(dead_code))] // used by every backend
pub(crate) fn pivot_signs(form: KktForm, n: usize) -> Vec<i8> {
    match form {
        KktForm::NormalEquations => vec![1; n],
        KktForm::Augmented { n_primal, .. } => {
            (0..n).map(|k| if k < n_primal { -1 } else { 1 }).collect()
        }
    }
}

/// Step K2: scatters the lower triangle of `k` into the values of the
/// analyzed lower pattern `l` (which has a full diagonal) and adds the
/// static shifts. Fails (K1) if `k` does not satisfy (S1), has the wrong
/// order, or has a lower entry outside `l`.
#[cfg_attr(not(feature = "faer"), allow(dead_code))] // used by every backend
pub(crate) fn assemble_regularized<T: Real>(
    l: &Pattern,
    form: KktForm,
    k: &CscMatrix<T>,
    reg: &Regularization<T>,
) -> Result<Vec<T>, KktError> {
    k.check_invariants()?;
    let n = l.n_cols();
    if k.n_rows != n || k.n_cols() != n {
        return Err(KktError::Structure(SparseError::Dimension {
            what: "KKT matrix order (as analyzed)",
            expected: n,
            found: k.n_rows.max(k.n_cols()),
        }));
    }
    let mut val = vec![T::ZERO; l.nnz()];
    for j in 0..n {
        let lrows = l.col(j);
        let base = l.col_ptr[j];
        let mut q = 0;
        for p in k.col_ptr[j]..k.col_ptr[j + 1] {
            let i = k.row_idx[p];
            if (i as usize) < j {
                continue;
            }
            // Both lists are increasing: a merge.
            while q < lrows.len() && lrows[q] < i {
                q += 1;
            }
            if q == lrows.len() || lrows[q] != i {
                return Err(KktError::Structure(SparseError::Malformed(format!(
                    "KKT entry ({i}, {j}) is outside the analyzed pattern"
                ))));
            }
            val[base + q] = k.val[p];
        }
        // The diagonal is the first entry of column j of l.
        debug_assert_eq!(lrows[0] as usize, j);
        let shift = match form {
            KktForm::NormalEquations => reg.dual,
            KktForm::Augmented { n_primal, .. } => {
                if j < n_primal {
                    -reg.primal
                } else {
                    reg.dual
                }
            }
        };
        if !is_exact_zero(shift) {
            val[base] += shift;
        }
    }
    Ok(val)
}

/// Step K4 over a list of `1×1` pivots and `2×2` blocks `(a, b, c)` =
/// `[a b; b c]`: the inertia by Sylvester's law.
pub(crate) fn count_inertia<T: Real>(
    ones: impl Iterator<Item = T>,
    twos: impl Iterator<Item = (T, T, T)>,
) -> Inertia {
    let mut s = Inertia::default();
    let mut one = |d: T| {
        if lt(T::ZERO, d) {
            s.positive += 1;
        } else if lt(d, T::ZERO) {
            s.negative += 1;
        } else {
            s.zero += 1;
        }
    };
    for d in ones {
        one(d);
    }
    for (a, b, c) in twos {
        // det < 0: one eigenvalue of each sign; det > 0: both of the sign
        // of the trace; det = 0: one zero, the other of the sign of the
        // trace.
        let det = a * c - b * b;
        if lt(det, T::ZERO) {
            one(T::ONE);
            one(-T::ONE);
        } else if lt(T::ZERO, det) {
            one(a + c);
            one(a + c);
        } else {
            one(T::ZERO);
            one(a + c);
        }
    }
    s
}

/// Most passes of the symmetric equilibration (step K2′). A cap, not a
/// threshold: with power-of-two factors the iteration stops by itself
/// when no factor changes, which happens after a few passes in practice
/// (Ruiz 2001 reports convergence of the infinity-norm iteration to
/// within a factor 2 in a handful of steps).
pub const EQUILIBRATION_PASSES: usize = 20;

/// Step K2′: symmetric equilibration of the lower pattern `l` with values
/// `val` (full diagonal present), in place; returns `s` (powers of two).
#[cfg_attr(not(feature = "faer"), allow(dead_code))] // used by every backend
pub(crate) fn equilibrate_symmetric<T: Real>(l: &Pattern, val: &mut [T]) -> Vec<T> {
    let n = l.n_cols();
    let mut s = vec![T::ONE; n];
    let mut rmax = vec![T::ZERO; n];
    let mut step = vec![T::ONE; n];
    for _ in 0..EQUILIBRATION_PASSES {
        rmax.fill(T::ZERO);
        for j in 0..n {
            let range = l.col_ptr[j]..l.col_ptr[j + 1];
            for (&i, &x) in l.row_idx[range.clone()].iter().zip(&val[range]) {
                let i = i as usize;
                let v = x.abs();
                rmax[i] = rmax[i].max(v);
                rmax[j] = rmax[j].max(v);
            }
        }
        let mut changed = false;
        for i in 0..n {
            step[i] = if rmax[i].is_finite() && lt(T::ZERO, rmax[i]) {
                pow2_round(rmax[i].sqrt().recip())
            } else {
                T::ONE
            };
            changed |= !is_exact_zero(step[i] - T::ONE);
        }
        if !changed {
            break;
        }
        for j in 0..n {
            let range = l.col_ptr[j]..l.col_ptr[j + 1];
            for (&i, x) in l.row_idx[range.clone()].iter().zip(&mut val[range]) {
                *x = *x * step[i as usize] * step[j];
            }
        }
        for (si, &t) in s.iter_mut().zip(&step) {
            *si *= t;
        }
    }
    s
}

/// `⌊x⌉₂ = 2^{round(log₂ x)}` for a finite `x > 0`, computed in `f64`
/// (a power of two is exact in every `Real`).
fn pow2_round<T: Real>(x: T) -> T {
    T::from_f64(x.to_f64().log2().round().exp2())
}

/// Step K4: the (S6) check.
#[cfg_attr(not(feature = "faer"), allow(dead_code))] // used by every backend
pub(crate) fn check_inertia(found: Inertia, expected: Inertia) -> Result<Inertia, KktError> {
    if found == expected {
        Ok(found)
    } else {
        Err(KktError::WrongInertia { found, expected })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Col, Row};

    #[test]
    fn lower_with_diagonal_accepts_lower_and_symmetric_patterns() {
        let lower = CscMatrix::from_triplets(3, 3, &[(Row(2), Col(0), 1.0), (Row(1), Col(1), 2.0)])
            .unwrap();
        let l = lower_with_diagonal(&lower.pattern(), KktForm::NormalEquations).unwrap();
        assert_eq!(l.col_ptr, vec![0, 2, 3, 4]);
        assert_eq!(l.row_idx, vec![0, 2, 1, 2]);
        let full = CscMatrix::from_triplets(3, 3, &[(Row(2), Col(0), 1.0), (Row(0), Col(2), 1.0)])
            .unwrap();
        assert_eq!(
            lower_with_diagonal(&full.pattern(), KktForm::NormalEquations).unwrap(),
            l.clone()
        );
        let skew = CscMatrix::from_triplets(3, 3, &[(Row(2), Col(0), 1.0), (Row(0), Col(1), 1.0)])
            .unwrap();
        assert_eq!(
            lower_with_diagonal(&skew.pattern(), KktForm::NormalEquations),
            Err(KktError::NotSymmetric)
        );
        let rect = CscMatrix::<f64>::zeros(2, 3);
        assert_eq!(
            lower_with_diagonal(&rect.pattern(), KktForm::NormalEquations),
            Err(KktError::NotSymmetric)
        );
        assert!(matches!(
            lower_with_diagonal(
                &lower.pattern(),
                KktForm::Augmented {
                    n_primal: 1,
                    n_dual: 1
                }
            ),
            Err(KktError::Structure(_))
        ));
    }

    #[test]
    fn inertia_counts_follow_sylvester() {
        let s = count_inertia(
            [1.0, -2.0, 0.0].into_iter(),
            [(0.0, 1.0, 0.0), (2.0, 1.0, 3.0), (-1.0, 0.5, -1.0)].into_iter(),
        );
        assert_eq!(
            s,
            Inertia {
                positive: 4,
                negative: 4,
                zero: 1
            }
        );
        let expected = KktForm::Augmented {
            n_primal: 2,
            n_dual: 3,
        }
        .expected_inertia(5);
        assert_eq!(expected.positive, 3);
        assert_eq!(expected.negative, 2);
        assert!(check_inertia(s, expected).is_err());
    }

    #[test]
    fn assembly_adds_the_shifts_and_rejects_foreign_entries() {
        let k = CscMatrix::from_triplets(
            3,
            3,
            &[
                (Row(0), Col(0), -1.0),
                (Row(2), Col(0), 1.0),
                (Row(1), Col(1), -2.0),
            ],
        )
        .unwrap();
        let form = KktForm::Augmented {
            n_primal: 2,
            n_dual: 1,
        };
        let l = lower_with_diagonal(&k.pattern(), form).unwrap();
        let reg = Regularization {
            primal: 0.5,
            dual: 0.25,
            pivot_tol: None,
            bump: 0.0,
        };
        let v = assemble_regularized(&l, form, &k, &reg).unwrap();
        assert_eq!(v, vec![-1.5, 1.0, -2.5, 0.25]);
        let other = CscMatrix::from_triplets(3, 3, &[(Row(2), Col(1), 1.0)]).unwrap();
        assert!(matches!(
            assemble_regularized(&l, form, &other, &reg),
            Err(KktError::Structure(_))
        ));
    }

    #[test]
    fn symmetric_matvec_reads_the_lower_triangle_only() {
        let k = CscMatrix::from_triplets(
            2,
            2,
            &[
                (Row(0), Col(0), 2.0),
                (Row(1), Col(0), 3.0),
                (Row(0), Col(1), 99.0),
                (Row(1), Col(1), 4.0),
            ],
        )
        .unwrap();
        let mut y = [0.0; 2];
        sym_lower_matvec(&k, &[1.0, 1.0], &mut y);
        assert_eq!(y, [5.0, 7.0]);
    }
}
