//! # Chapter 5.15 — `FaerLdl`: the v0.1 KKT backend
//!
//! *New with task L1.5 (PLAN.md trilha L; the contract's §5.7, decision
//! D10 "hybrid"). Compiled with the `faer` feature (on by default).*
//!
//! ## Purpose
//! [`FaerLdl`] implements [`KktSolver`] with the sparse symmetric
//! factorizations of the `faer` crate (pure Rust, MIT): `LDLᵀ`
//! (simplicial or supernodal, faer's choice by flops per `nnz(L)` as in
//! CHOLMOD, or forced by [`FaerLdl::supernodal`]) and the intranodal
//! Bunch–Kaufman `LBLᵀ`. What is ours: the fill-reducing ordering (any
//! [`Ordering`], by default our [`crate::Amd`] of Chapter 5.10 — faer's own AMD
//! is never called), the regularization policy and the inertia check
//! (Algorithm K of Chapter 5.14), and the refinement (Algorithm F). The
//! simplex method never touches this chapter (D10).
//!
//! ## Notation
//! As in Chapter 5.14: `K` of order `N` by its lower triangle, `L₀` the
//! analyzed lower pattern with a full diagonal, `π` the ordering
//! ([`crate::Perm`], `fwd[k]` = the row eliminated `k`-th), `σ_k` the pivot
//! signs, `(ρ, δ, τ, β)` the regularization.
//!
//! ## Algorithm A (Analyze)
//! A1. `L₀ ←` the lower triangle of the pattern plus a full diagonal
//!     (invariant (K1) of Chapter 5.14; [`KktError::NotSymmetric`] or
//!     [`KktError::Structure`] otherwise).
//! A2. `π ←` `ordering.order(L₀ + L₀ᵀ, Symmetric)`.
//! A2′. `π ←` the postorder of the elimination tree of `π`, heaviest
//!     child last (Algorithm P of Chapter 5.9): the same `nnz(L)`, but
//!     subtrees contiguous and the largest front next to its parent —
//!     the one child faer's relaxed supernodal amalgamation can merge.
//!     Whatever the ordering's own output convention (the assembly tree of
//!     our [`crate::Amd`], the pivot order of [`crate::Colamd`], a caller's
//!     permutation), the supernodes faer builds are then as good as the
//!     elimination tree allows (task L1.5b: maros-r7's augmented system
//!     factorized 11× slower without this step, with the same
//!     `nnz(L)`).
//! A3. faer's symbolic Cholesky of `L₀` under the custom permutation `π`:
//!     elimination tree, column counts, and the simplicial or supernodal
//!     structure of `L` (threshold from [`FaerLdl::supernodal`]).
//! A4. The pattern of `P L₀ Pᵀ` in the triangle faer's numeric phase reads
//!     (upper for the simplicial structure, lower for the supernodal one),
//!     with the position of every entry of `L₀` in it: each factorization
//!     then permutes its values by a scatter, and calls faer's
//!     *structure-level* numeric phase, which reports how many pivots it
//!     bumped (the top-level one does not; task L1.5b).
//!
//! ## Algorithm N (Numeric factorization)
//! N1. `v ←` the values of `K` on `L₀` (steps K1′–K2 of Chapter 5.14),
//!     equilibrated to `K̃ = S K S` (step K2′), plus the static shifts
//!     (step K2″): `v` holds `K̃_reg`.
//! N2. Choose the kernel:
//!     * **`LDLᵀ`** (diagonal `Δ`, no pivoting) for the normal equations,
//!       for an augmented system that is quasi-definite (`ρ, δ > 0`), and
//!       whenever dynamic regularization is on (`τ = Some`): faer's
//!       left-looking simplicial or supernodal `LDLᵀ` with the signs `σ_k`
//!       (permuted by `π` inside faer), threshold `τ` and bump `β`
//!       (step K3). A zero or non-finite pivot without bumping is
//!       [`KktError::ZeroPivot`] at the original row `fwd[k]`.
//!     * **`LBLᵀ`** (Bunch & Kaufman 1977, `1×1` and `2×2` pivots) for an
//!       augmented system without regularization (`ρ = 0` or `δ = 0`, and
//!       `τ = None`) when the structure is supernodal: faer's *intranodal*
//!       Bunch–Kaufman, whose interchanges stay inside each supernode (the
//!       analyzed structure does not change). With a simplicial structure
//!       faer has no pivoted kernel, and the `LDLᵀ` path runs instead.
//! N3. Read the pivots — the diagonal of `Δ` (first entry of each column
//!     of the simplicial `L`; diagonal of each supernode's dense block),
//!     plus the subdiagonal of `B` for the `2×2` blocks of `LBLᵀ` — and
//!     count the inertia (step K4). The bumped pivots of step K3: faer's
//!     count, exact for the simplicial structure; its supernodal kernel
//!     counts only the positive-sign pivots it bumps, so a negative-sign
//!     pivot equal to `−β` exactly is added — the one count by value left
//!     (it can only overcount, by a pivot that happens to equal `−β` bit
//!     for bit).
//! N3′. (S10): if a stored value of the factor (or of the subdiagonal of
//!     `B`) is not finite, fail with [`KktError::NotFinite`] at the
//!     original row `fwd[k]` of the first such column `k`.
//! N4. (S6): fail with [`KktError::WrongInertia`] unless the inertia is the
//!     one the form requires.
//!
//! ## Algorithm S (Solve)
//! `x ← S x`, faer's permuted triangular solves `x ← Pᵀ L⁻ᵀ Δ⁻¹ L⁻¹ P x`
//! on `K̃`, then `x ← S x` (the factor is of `K̃ = S K_reg S`; and the
//! intranodal permutation and `2×2` blocks for `LBLᵀ`), sequentially
//! (`Par::Seq`; faer is built without its `rayon` feature, so results do
//! not depend on threads). The solve allocates nothing: faer's stack
//! memory for it is sized and allocated once, by the factorization, and
//! kept in the [`FaerNumeric`] behind a mutex (a second thread solving
//! with the same factor at the same moment gets a temporary buffer
//! instead of waiting) — with [`KktSolver::solve_refined_with`] and a
//! reused [`crate::RefineWorkspace`], a refined solve allocates nothing
//! either. Measured in L1.5b with a counting allocator on the 186 Netlib
//! systems of the middle regime: 0 allocations in a second solve and in
//! a second refined solve with the same workspace (the first solve with a
//! factor may make one, the lazy initialization of the mutex on macOS).
//!
//! ## Invariants
//! (F1) The numeric factor shares the symbolic structure it was computed
//!      with (an `Arc`): a [`FaerNumeric`] can be solved with after its
//!      [`FaerSymbolic`] is dropped, and never with another one.
//! (F2) `nnz_factor` is faer's stored length of `L` (diagonal included;
//!      for the supernodal structure, the dense supernodal blocks with
//!      their explicit zeros).
//! (S6) as in Chapter 5.14, checked in step N4 and asserted in debug
//!      builds.
//!
//! ## What was read
//! faer 0.24.4, `sparse::linalg::cholesky` (the public API and its
//! documentation: `factorize_symbolic_cholesky`, `SymmetricOrdering::Custom`,
//! `factorize_numeric_ldlt` with `LdltRegularization`, the intranodal
//! `LBLᵀ`, the supernodal storage read by the solves) and
//! `linalg::cholesky::ldlt::factor` (the dynamic regularization rule, to
//! match step K3) — read, not copied; CHOLMOD (Chen, Davis, Hager &
//! Rajamanickam 2008) for the shape.
//!
//! ## References
//! See Chapter 5.14; the faer documentation (docs.rs/faer/0.24.4,
//! `sparse::linalg::cholesky`); Bunch & Kaufman (1977); Davis, *Direct
//! Methods for Sparse Linear Systems* (2006), ch. 4 and 9.

use std::sync::{Arc, Mutex};

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::{LdltError, LdltRegularization};
use faer::perm::PermRef;
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, IntranodeLbltRef, LdltRef, SymbolicCholesky, SymbolicCholeskyRaw,
    SymmetricOrdering, factorize_symbolic_cholesky, simplicial, supernodal,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par, Side, Spec};

use crate::kkt::{
    assemble_regularized, check_inertia, check_regularization, count_inertia, lower_with_diagonal,
    pivot_signs,
};
use crate::real::{Real, approx_eq, is_exact_zero};
use crate::{
    CscMatrix, Inertia, KktError, KktForm, KktSolver, Ordering, OrderingTarget, Pattern, Perm,
    Regularization, SparseError,
};

/// The v0.1 backend (D10, hybrid): faer's sparse `LDLᵀ` (normal
/// equations, quasi-definite augmented systems) or intranodal `LBLᵀ` with
/// Bunch–Kaufman pivots (unregularized augmented systems), with faer's
/// simplicial/supernodal selection, ordered by our ordering through
/// [`KktSolver::analyze`]. Regularization policy and refinement are the
/// trait's (Chapter 5.14).
///
/// Normal equations `A D Aᵀ + δI`, analyzed once and factorized for two
/// diagonals `D` (the pattern does not change between interior-point
/// iterations):
/// ```
/// use xingo_sparse::{Amd, Col, CscMatrix, FaerLdl, KktForm, KktSolver, Refinement, Regularization, Row};
/// // A = [1 1 0; 0 1 1]; lower triangle of A D Aᵀ.
/// let normal = |d: [f64; 3]| {
///     CscMatrix::from_triplets(2, 2, &[
///         (Row(0), Col(0), d[0] + d[1]), (Row(1), Col(0), d[1]), (Row(1), Col(1), d[1] + d[2]),
///     ])
///     .unwrap()
/// };
/// let solver = FaerLdl::default();
/// let form = KktForm::NormalEquations;
/// let sym = solver.analyze(&normal([1.0; 3]).pattern(), form, &Amd::default()).unwrap();
/// for d in [[1.0, 2.0, 3.0], [1e-6, 1e4, 1e-3]] {
///     let k = normal(d);
///     // δ is taken in the equilibrated matrix (the default).
///     let (num, inertia) = solver.factorize(&sym, &k, Regularization::default()).unwrap();
///     assert_eq!(inertia.positive, 2);
///     let (b, mut x) = ([1.0, -1.0], [0.0; 2]);
///     // Refining against k itself removes the bias of the shift δ, down to
///     // the rounding floor ε‖K‖‖x‖ of the computed residual (≈ 10⁻⁸ for
///     // the second D, whose K has condition ≈ 10⁷).
///     let report = solver.solve_refined(&num, &k, &b, &mut x, Refinement::default());
///     assert!(report.residual <= 1e-8);
/// }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaerLdl {
    /// Force the supernodal path (`Some(true)`), the simplicial one
    /// (`Some(false)`), or let faer choose by flops per `nnz(L)`
    /// (`None`).
    pub supernodal: Option<bool>,
}

/// The symbolic phase of [`FaerLdl`] (opaque): the analyzed lower
/// pattern, the ordering and faer's symbolic factor.
#[derive(Debug)]
pub struct FaerSymbolic {
    form: KktForm,
    lower: Pattern,
    row_idx: Vec<usize>,
    perm: Perm,
    faer: Arc<SymbolicCholesky<usize>>,
    /// `P L₀ Pᵀ` in the triangle faer's numeric phase reads (upper for the
    /// simplicial structure, lower for the supernodal one), with the
    /// position of every entry of `L₀` in it (step A4).
    permuted: Permuted,
}

/// Step A4: the permuted pattern and the scatter map.
#[derive(Debug)]
struct Permuted {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    /// `map[p]` = the position in `row_idx` of entry `p` of `L₀`.
    map: Vec<usize>,
}

/// Step A4 on `L₀` (lower, full diagonal) under `perm`: the upper
/// (`upper = true`) or lower triangle of `P L₀ Pᵀ`, columns sorted.
fn permute_lower(lower: &Pattern, perm: &Perm, upper: bool) -> Permuted {
    let n = lower.n_cols();
    let nnz = lower.nnz();
    let mut entries: Vec<(usize, usize, usize)> = Vec::with_capacity(nnz);
    for j in 0..n {
        for (q, &i) in lower.col(j).iter().enumerate() {
            let (a, b) = (perm.inv[i as usize] as usize, perm.inv[j] as usize);
            let (hi, lo) = (a.max(b), a.min(b));
            let (col, row) = if upper { (hi, lo) } else { (lo, hi) };
            entries.push((col, row, lower.col_ptr[j] + q));
        }
    }
    entries.sort_unstable();
    let mut col_ptr = vec![0_usize; n + 1];
    let mut row_idx = Vec::with_capacity(nnz);
    let mut map = vec![0_usize; nnz];
    for (pos, &(col, row, p)) in entries.iter().enumerate() {
        col_ptr[col + 1] += 1;
        row_idx.push(row);
        map[p] = pos;
    }
    for c in 0..n {
        col_ptr[c + 1] += col_ptr[c];
    }
    Permuted {
        col_ptr,
        row_idx,
        map,
    }
}

impl FaerSymbolic {
    /// The fill-reducing permutation used (`fwd[k]` = row eliminated
    /// `k`-th): the ordering's, postordered (step A2′).
    pub fn perm(&self) -> &Perm {
        &self.perm
    }

    /// `true` if faer chose (or was told to use) the supernodal structure.
    pub fn is_supernodal(&self) -> bool {
        matches!(self.faer.raw(), SymbolicCholeskyRaw::Supernodal(_))
    }

    /// Stored length of the factor `L` (invariant (F2)).
    pub fn nnz_factor(&self) -> usize {
        self.faer.len_val()
    }

    /// The form the pattern was analyzed for.
    pub fn form(&self) -> KktForm {
        self.form
    }
}

/// The numeric factor of [`FaerLdl`] (opaque; invariant (F1)).
#[derive(Debug)]
pub struct FaerNumeric<T: Real> {
    faer: Arc<SymbolicCholesky<usize>>,
    values: Vec<T>,
    /// `Some` for the Bunch–Kaufman path: the subdiagonal of `B` and the
    /// intranodal permutation.
    lblt: Option<Lblt<T>>,
    inertia: Inertia,
    bumped: usize,
    /// `s` of step K2′ (powers of two).
    scale: Vec<T>,
    /// faer's stack memory for the solves (Algorithm S).
    scratch: SolveScratch,
}

/// faer's stack memory for [`KktSolver::solve`], allocated by the
/// factorization; a mutex so that a shared factor stays `Sync`.
struct SolveScratch(Mutex<MemBuffer>);

impl std::fmt::Debug for SolveScratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SolveScratch")
    }
}

#[derive(Debug)]
struct Lblt<T> {
    subdiag: Vec<T>,
    fwd: Vec<usize>,
    inv: Vec<usize>,
}

impl<T: Real> FaerNumeric<T> {
    /// The inertia of the factor (S6).
    pub fn inertia(&self) -> Inertia {
        self.inertia
    }

    /// Pivots that dynamic regularization bumped (step K3).
    pub fn bumped(&self) -> usize {
        self.bumped
    }

    /// `true` if the factor is a Bunch–Kaufman `LBLᵀ` (`2×2` pivots
    /// allowed), `false` for a diagonal `LDLᵀ`.
    pub fn is_pivoted(&self) -> bool {
        self.lblt.is_some()
    }
}

fn faer_error(e: faer::sparse::FaerError) -> KktError {
    match e {
        faer::sparse::FaerError::OutOfMemory => KktError::Backend("faer: out of memory"),
        faer::sparse::FaerError::IndexOverflow => KktError::Backend("faer: index overflow"),
        _ => KktError::Backend("faer: symbolic factorization failed"),
    }
}

/// The diagonal pivots of a faer `LDLᵀ`/`LBLᵀ` value array, in pivot order.
fn pivots(sym: &SymbolicCholesky<usize>, values: &[f64]) -> Vec<f64> {
    match sym.raw() {
        SymbolicCholeskyRaw::Simplicial(s) => {
            let l = s.factor();
            let col_ptr = l.col_ptr();
            let rows = l.row_idx();
            (0..s.nrows())
                .map(|k| {
                    debug_assert_eq!(rows[col_ptr[k]], k, "the diagonal leads each column");
                    values[col_ptr[k]]
                })
                .collect()
        }
        SymbolicCholeskyRaw::Supernodal(s) => {
            let mut d = Vec::with_capacity(s.nrows());
            for t in 0..s.n_supernodes() {
                let begin = s.supernode_begin()[t];
                let ncols = s.supernode_end()[t] - begin;
                let nrows = s.nnz_per_super()[t] + ncols;
                let block = &values[s.col_ptr_for_val()[t]..s.col_ptr_for_val()[t + 1]];
                d.extend((0..ncols).map(|c| block[c * nrows + c]));
            }
            d
        }
    }
}

/// Step N3′: the first column (pivot order) of the factor holding a NaN
/// or an infinity, if any.
fn first_non_finite(sym: &SymbolicCholesky<usize>, values: &[f64]) -> Option<usize> {
    let p = values.iter().position(|x| !x.is_finite())?;
    Some(match sym.raw() {
        SymbolicCholeskyRaw::Simplicial(s) => {
            // The column whose range holds p.
            s.col_ptr().partition_point(|&c| c <= p) - 1
        }
        SymbolicCholeskyRaw::Supernodal(s) => {
            let ptr = s.col_ptr_for_val();
            let t = ptr.partition_point(|&c| c <= p) - 1;
            let ncols = s.supernode_end()[t] - s.supernode_begin()[t];
            let nrows = s.nnz_per_super()[t] + ncols;
            s.supernode_begin()[t] + (p - ptr[t]) / nrows
        }
    })
}

/// Step N2, Bunch–Kaufman path: faer's intranodal `LBLᵀ`; returns the
/// pivot data, the `1×1` pivots and the `2×2` blocks (step N3).
#[allow(clippy::type_complexity)]
fn factor_lblt(
    sym: &SymbolicCholesky<usize>,
    a: SparseColMatRef<'_, usize, f64>,
    values: &mut [f64],
) -> Result<(Lblt<f64>, Vec<f64>, Vec<(f64, f64, f64)>), KktError> {
    let n = sym.nrows();
    let mut subdiag = vec![0.0_f64; n];
    let mut fwd = vec![0_usize; n];
    let mut inv = vec![0_usize; n];
    let mut buf = MemBuffer::try_new(
        sym.factorize_numeric_intranode_lblt_scratch::<f64>(Par::Seq, Spec::default()),
    )
    .map_err(|_| KktError::Backend("faer: out of memory"))?;
    sym.factorize_numeric_intranode_lblt(
        values,
        &mut subdiag,
        &mut fwd,
        &mut inv,
        a,
        Side::Lower,
        Par::Seq,
        MemStack::new(&mut buf),
        Spec::default(),
    );
    let d = pivots(sym, values);
    // N3: 2×2 blocks where the subdiagonal is nonzero.
    let mut ones = Vec::with_capacity(n);
    let mut twos = Vec::new();
    let mut p = 0;
    while p < n {
        if p + 1 < n && !is_exact_zero(subdiag[p]) {
            twos.push((d[p], subdiag[p], d[p + 1]));
            p += 2;
        } else {
            ones.push(d[p]);
            p += 1;
        }
    }
    Ok((Lblt { subdiag, fwd, inv }, ones, twos))
}

impl KktSolver<f64> for FaerLdl {
    type Symbolic = FaerSymbolic;
    type Numeric = FaerNumeric<f64>;

    fn analyze(
        &self,
        pattern: &Pattern,
        form: KktForm,
        ordering: &dyn Ordering,
    ) -> Result<FaerSymbolic, KktError> {
        // A1.
        let lower = lower_with_diagonal(pattern, form)?;
        let n = lower.n_cols();
        // A2.
        let full = lower.symmetric()?;
        let perm = ordering.order(&full, OrderingTarget::Symmetric)?;
        if perm.len() != n {
            return Err(KktError::Structure(SparseError::Dimension {
                what: "ordering length",
                expected: n,
                found: perm.len(),
            }));
        }
        // A2′.
        let perm = full.etree_postorder(&perm);
        // A3.
        let row_idx: Vec<usize> = lower.row_idx.iter().map(|&i| i as usize).collect();
        let fwd: Vec<usize> = perm.fwd.iter().map(|&i| i as usize).collect();
        let inv: Vec<usize> = perm.inv.iter().map(|&i| i as usize).collect();
        let a = SymbolicSparseColMatRef::new_checked(n, n, &lower.col_ptr, None, &row_idx);
        let threshold = match self.supernodal {
            None => SupernodalThreshold::AUTO,
            Some(true) => SupernodalThreshold::FORCE_SUPERNODAL,
            Some(false) => SupernodalThreshold::FORCE_SIMPLICIAL,
        };
        let params = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: threshold,
            ..Default::default()
        };
        let faer = factorize_symbolic_cholesky(
            a,
            Side::Lower,
            SymmetricOrdering::Custom(PermRef::new_checked(&fwd, &inv, n)),
            params,
        )
        .map_err(faer_error)?;
        // A4.
        let upper = matches!(faer.raw(), SymbolicCholeskyRaw::Simplicial(_));
        let permuted = permute_lower(&lower, &perm, upper);
        Ok(FaerSymbolic {
            form,
            lower,
            row_idx,
            perm,
            faer: Arc::new(faer),
            permuted,
        })
    }

    #[allow(clippy::too_many_lines)] // steps N1–N4 in order, as in the chapter
    fn factorize(
        &self,
        symbolic: &FaerSymbolic,
        k: &CscMatrix<f64>,
        reg: Regularization<f64>,
    ) -> Result<(FaerNumeric<f64>, Inertia), KktError> {
        let form = symbolic.form;
        let sym = &*symbolic.faer;
        let n = symbolic.lower.n_cols();
        check_regularization(&reg)?;
        // N1.
        let (v, scale) = assemble_regularized(&symbolic.lower, form, k, &reg)?;
        let a = SparseColMatRef::new(
            SymbolicSparseColMatRef::new_checked(
                n,
                n,
                &symbolic.lower.col_ptr,
                None,
                &symbolic.row_idx,
            ),
            &v,
        );
        let mut values = vec![0.0_f64; sym.len_val()];
        let supernodal = matches!(sym.raw(), SymbolicCholeskyRaw::Supernodal(_));
        // N2.
        let use_lblt = matches!(form, KktForm::Augmented { .. })
            && !reg.is_quasi_definite()
            && reg.pivot_tol.is_none()
            && supernodal;
        let (lblt, d, twos, faer_bumps) = if use_lblt {
            let (lblt, ones, twos) = factor_lblt(sym, a, &mut values)?;
            (Some(lblt), ones, twos, 0)
        } else {
            // The signs in pivot order, and K̃_reg permuted by the map of
            // step A4: faer's lower-level numeric phases, which report the
            // count of bumped pivots (step N3).
            let signs = pivot_signs(form, n);
            let fwd = &symbolic.perm.fwd;
            let signs: Vec<i8> = (0..n).map(|k| signs[fwd[k] as usize]).collect();
            let pm = &symbolic.permuted;
            let mut pv = vec![0.0_f64; v.len()];
            for (&dst, &x) in pm.map.iter().zip(&v) {
                pv[dst] = x;
            }
            let a_perm = SparseColMatRef::new(
                SymbolicSparseColMatRef::new_checked(n, n, &pm.col_ptr, None, &pm.row_idx),
                &pv,
            );
            let regularization = LdltRegularization {
                dynamic_regularization_signs: reg.pivot_tol.map(|_| signs.as_slice()),
                dynamic_regularization_delta: reg.bump,
                dynamic_regularization_epsilon: reg.pivot_tol.unwrap_or(0.0),
            };
            let req = match sym.raw() {
                SymbolicCholeskyRaw::Simplicial(_) => {
                    simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, f64>(n)
                }
                SymbolicCholeskyRaw::Supernodal(s) => {
                    supernodal::factorize_supernodal_numeric_ldlt_scratch::<usize, f64>(
                        s,
                        Par::Seq,
                        Spec::default(),
                    )
                }
            };
            let mut buf =
                MemBuffer::try_new(req).map_err(|_| KktError::Backend("faer: out of memory"))?;
            let stack = MemStack::new(&mut buf);
            let result = match sym.raw() {
                SymbolicCholeskyRaw::Simplicial(s) => {
                    simplicial::factorize_simplicial_numeric_ldlt(
                        &mut values,
                        a_perm,
                        regularization,
                        s,
                        stack,
                    )
                }
                SymbolicCholeskyRaw::Supernodal(s) => {
                    supernodal::factorize_supernodal_numeric_ldlt(
                        &mut values,
                        a_perm,
                        regularization,
                        s,
                        Par::Seq,
                        stack,
                        Spec::default(),
                    )
                }
            };
            let bumps = match result {
                Ok(info) => info.dynamic_regularization_count,
                Err(LdltError::ZeroPivot { index }) => {
                    // faer 0.24 reports the simplicial pivot one-based and
                    // the supernodal one zero-based.
                    let k = if supernodal { index } else { index - 1 };
                    return Err(KktError::ZeroPivot {
                        position: symbolic.perm.fwd[k] as usize,
                    });
                }
            };
            let d = pivots(sym, &values);
            // faer's supernodal kernel counts only the bumps of the
            // positive-sign pivots; a bumped negative-sign pivot is −β
            // exactly (step N3).
            let extra = if supernodal && reg.pivot_tol.is_some() {
                d.iter()
                    .zip(&signs)
                    .filter(|&(&x, &sg)| sg < 0 && approx_eq(x, -reg.bump, 0.0))
                    .count()
            } else {
                0
            };
            (None, d, Vec::new(), bumps + extra)
        };
        // N3′ (step K3′, (S10)).
        let bad = first_non_finite(sym, &values).or_else(|| {
            lblt.as_ref()
                .and_then(|b| b.subdiag.iter().position(|x| !x.is_finite()))
        });
        if let Some(k) = bad {
            return Err(KktError::NotFinite {
                position: symbolic.perm.fwd[k] as usize,
            });
        }
        // N3.
        let bumped = faer_bumps;
        let found = count_inertia(d.into_iter(), twos.into_iter());
        // N4 (S6).
        let inertia = check_inertia(found, form.expected_inertia(n))?;
        debug_assert_eq!(inertia, form.expected_inertia(n), "(S6)");
        Ok((
            FaerNumeric {
                faer: Arc::clone(&symbolic.faer),
                values,
                lblt,
                inertia,
                bumped,
                scale,
                scratch: SolveScratch(Mutex::new(
                    MemBuffer::try_new(sym.solve_in_place_scratch::<f64>(1, Par::Seq))
                        .map_err(|_| KktError::Backend("faer: out of memory"))?,
                )),
            },
            inertia,
        ))
    }

    fn solve(&self, numeric: &FaerNumeric<f64>, x: &mut [f64]) {
        let sym = &*numeric.faer;
        let n = sym.nrows();
        assert_eq!(x.len(), n, "solve: x has the wrong length");
        for (xi, &si) in x.iter_mut().zip(&numeric.scale) {
            *xi *= si;
        }
        // The factor's own scratch; a temporary one only if another thread
        // holds it (or a panic poisoned it).
        let mut held = numeric.scratch.0.try_lock().ok();
        let mut temp;
        let stack = if let Some(buf) = held.as_deref_mut() {
            MemStack::new(buf)
        } else {
            temp = MemBuffer::new(sym.solve_in_place_scratch::<f64>(1, Par::Seq));
            MemStack::new(&mut temp)
        };
        let rhs = MatMut::from_column_major_slice_mut(x, n, 1);
        match &numeric.lblt {
            None => LdltRef::new(sym, &numeric.values).solve_in_place_with_conj(
                Conj::No,
                rhs,
                Par::Seq,
                stack,
            ),
            Some(b) => IntranodeLbltRef::new(
                sym,
                &numeric.values,
                &b.subdiag,
                PermRef::new_checked(&b.fwd, &b.inv, n),
            )
            .solve_in_place_with_conj(Conj::No, rhs, Par::Seq, stack),
        }
        for (xi, &si) in x.iter_mut().zip(&numeric.scale) {
            *xi *= si;
        }
    }

    fn nnz_factor(&self, numeric: &FaerNumeric<f64>) -> usize {
        numeric.faer.len_val()
    }

    fn name(&self) -> &'static str {
        "faer-ldl"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::le;
    use crate::{Amd, Col, Natural, Refinement, Row, ShiftScale};
    use proptest::prelude::*;

    /// xorshift in `[0, 1)`.
    struct Rng(u64);
    impl Rng {
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            #[allow(clippy::cast_precision_loss)]
            let u = (self.0 >> 11) as f64 / (1_u64 << 53) as f64;
            u
        }
    }

    /// A random sparse `m × n` matrix with a nonzero in every row (full
    /// row rank with probability one) and the KKT systems built from it
    /// with `D = 10^{±spread·u}`.
    fn kkt(m: usize, n: usize, spread: f64, seed: u64) -> (CscMatrix, CscMatrix, KktForm) {
        let mut r = Rng(seed | 1);
        let mut a = Vec::new();
        for i in 0..m {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let j = (r.unit() * n as f64) as usize % n;
            a.push((i, j, 1.0 + r.unit()));
            for jj in 0..n {
                if jj != j && r.unit() < 0.2 {
                    a.push((i, jj, 2.0 * r.unit() - 1.0));
                }
            }
        }
        let d: Vec<f64> = (0..n)
            .map(|_| 10f64.powf(spread * (2.0 * r.unit() - 1.0)))
            .collect();
        // Augmented, lower triangle: −D on the diagonal, A below.
        let mut t = Vec::new();
        for (j, &dj) in d.iter().enumerate() {
            t.push((Row::from_usize(j), Col::from_usize(j), -dj));
        }
        for &(i, j, v) in &a {
            t.push((Row::from_usize(n + i), Col::from_usize(j), v));
        }
        let aug = CscMatrix::from_triplets(n + m, n + m, &t).unwrap();
        // Normal equations A D Aᵀ, lower triangle.
        let mut dense = vec![vec![0.0; m]; m];
        for &(i, j, v) in &a {
            for &(p, q, w) in &a {
                if q == j && p <= i {
                    dense[i][p] += v * d[j] * w;
                }
            }
        }
        let mut tn = Vec::new();
        for (i, row) in dense.iter().enumerate() {
            for (p, &v) in row.iter().enumerate() {
                if !crate::real::is_exact_zero(v) {
                    tn.push((Row::from_usize(i), Col::from_usize(p), v));
                }
            }
        }
        let normal = CscMatrix::from_triplets(m, m, &tn).unwrap();
        (
            aug,
            normal,
            KktForm::Augmented {
                n_primal: n,
                n_dual: m,
            },
        )
    }

    fn rhs(n: usize, seed: u64) -> Vec<f64> {
        let mut r = Rng(seed | 3);
        (0..n).map(|_| 2.0 * r.unit() - 1.0).collect()
    }

    /// Factorizes `k` with `reg` and refines against the unregularized
    /// `K` (the measure of L1.5b), with the right-hand side `K x*` for
    /// `x* = b`.
    fn solve_ok(
        k: &CscMatrix,
        form: KktForm,
        reg: Regularization,
        supernodal: Option<bool>,
        b: &[f64],
    ) -> (Inertia, f64, usize) {
        let s = FaerLdl { supernodal };
        let sym = s.analyze(&k.pattern(), form, &Amd::default()).unwrap();
        let (num, inertia) = s.factorize(&sym, k, reg).unwrap();
        // A manufactured solution: b = K x*, as the right-hand sides of an
        // interior-point method scale with K.
        let mut rhs = vec![0.0; b.len()];
        crate::kkt::sym_lower_matvec(k, b, &mut rhs);
        let mut x = vec![0.0; b.len()];
        let rep = s.solve_refined(&num, k, &rhs, &mut x, Refinement::default());
        (inertia, rep.residual, num.bumped())
    }

    #[test]
    fn augmented_and_normal_systems_solve_with_the_expected_inertia() {
        for (seed, spread) in [(1, 0.0), (2, 4.0), (3, 8.0)] {
            let (aug, normal, form) = kkt(12, 30, spread, seed);
            for sn in [None, Some(true), Some(false)] {
                let (i, res, _) =
                    solve_ok(&aug, form, Regularization::default(), sn, &rhs(42, seed));
                assert_eq!(i, form.expected_inertia(42));
                assert!(le(res, 1e-10, 0.0), "aug seed {seed} sn {sn:?}: {res}");
                let (i, res, _) = solve_ok(
                    &normal,
                    KktForm::NormalEquations,
                    Regularization::default(),
                    sn,
                    &rhs(12, seed),
                );
                assert_eq!(i, KktForm::NormalEquations.expected_inertia(12));
                assert!(le(res, 1e-10, 0.0), "normal seed {seed} sn {sn:?}: {res}");
            }
        }
    }

    #[test]
    fn unregularized_augmented_system_takes_the_bunch_kaufman_path() {
        let (aug, _, form) = kkt(10, 25, 2.0, 9);
        let s = FaerLdl {
            supernodal: Some(true),
        };
        let sym = s.analyze(&aug.pattern(), form, &Natural).unwrap();
        let (num, inertia) = s.factorize(&sym, &aug, Regularization::none()).unwrap();
        assert!(num.is_pivoted());
        assert_eq!(inertia, form.expected_inertia(35));
        let b = rhs(35, 9);
        let mut x = vec![0.0; 35];
        let rep = s.solve_refined(&num, &aug, &b, &mut x, Refinement::default());
        assert!(rep.converged, "{rep:?}");
    }

    #[test]
    fn s6_errors_zero_pivot_and_wrong_inertia() {
        // K = [−1, 0; 0, 0]: the dual pivot is exactly zero.
        let k = CscMatrix::from_triplets(2, 2, &[(Row(0), Col(0), -1.0)]).unwrap();
        let form = KktForm::Augmented {
            n_primal: 1,
            n_dual: 1,
        };
        let s = FaerLdl {
            supernodal: Some(false),
        };
        let sym = s.analyze(&k.pattern(), form, &Natural).unwrap();
        assert_eq!(
            s.factorize(&sym, &k, Regularization::none()).err(),
            Some(KktError::ZeroPivot { position: 1 })
        );
        // With the Bunch–Kaufman path the zero pivot shows in the inertia.
        let sn = FaerLdl {
            supernodal: Some(true),
        };
        let sym = sn.analyze(&k.pattern(), form, &Natural).unwrap();
        assert!(matches!(
            sn.factorize(&sym, &k, Regularization::none()),
            Err(KktError::WrongInertia { .. })
        ));
        // Static regularization makes it quasi-definite.
        let (_, inertia) = s
            .factorize(
                &s.analyze(&k.pattern(), form, &Natural).unwrap(),
                &k,
                Regularization::default(),
            )
            .unwrap();
        assert_eq!(inertia, form.expected_inertia(2));
        // Indefinite "normal equations": wrong inertia without bumping,
        // the expected one (with bumps) with it.
        let ind = CscMatrix::from_triplets(
            2,
            2,
            &[
                (Row(0), Col(0), 1.0),
                (Row(1), Col(0), 2.0),
                (Row(1), Col(1), 1.0),
            ],
        )
        .unwrap();
        let sym = s
            .analyze(&ind.pattern(), KktForm::NormalEquations, &Natural)
            .unwrap();
        assert_eq!(
            s.factorize(&sym, &ind, Regularization::none()).err(),
            Some(KktError::WrongInertia {
                found: Inertia {
                    positive: 1,
                    negative: 1,
                    zero: 0
                },
                expected: Inertia {
                    positive: 2,
                    negative: 0,
                    zero: 0
                }
            })
        );
        let reg = Regularization {
            primal: 0.0,
            dual: 0.0,
            pivot_tol: Some(1e-13),
            bump: 1e-8,
            scale: ShiftScale::Equilibrated,
        };
        let (num, inertia) = s.factorize(&sym, &ind, reg).unwrap();
        assert_eq!(inertia.positive, 2);
        assert_eq!(num.bumped(), 1);
        // A pivot equal to δ is not a bump.
        let (aug, _, form) = kkt(3, 5, 0.0, 4);
        let sym = s.analyze(&aug.pattern(), form, &Natural).unwrap();
        let (num, _) = s.factorize(&sym, &aug, Regularization::default()).unwrap();
        assert_eq!(num.bumped(), 0);
    }

    #[test]
    fn s10_overflow_is_an_error_and_non_finite_input_is_refused() {
        // Pivots bumped to a tiny β make L overflow (step K3′).
        let t = |v: f64| {
            CscMatrix::from_triplets(
                3,
                3,
                &[
                    (Row(0), Col(0), v),
                    (Row(1), Col(0), 1.0),
                    (Row(2), Col(0), 1.0),
                    (Row(1), Col(1), 1.0),
                    (Row(2), Col(1), 1.0),
                    (Row(2), Col(2), 1.0),
                ],
            )
            .unwrap()
        };
        let k = t(1e-310);
        let reg = Regularization {
            primal: 0.0,
            dual: 0.0,
            pivot_tol: Some(1e-300),
            bump: 1e-300,
            scale: ShiftScale::Equilibrated,
        };
        for sn in [Some(false), Some(true)] {
            let s = FaerLdl { supernodal: sn };
            let sym = s
                .analyze(&k.pattern(), KktForm::NormalEquations, &Natural)
                .unwrap();
            assert!(
                matches!(
                    s.factorize(&sym, &k, reg),
                    Err(KktError::NotFinite { .. } | KktError::ZeroPivot { .. })
                ),
                "{sn:?}"
            );
            // A NaN in K never reaches the factorization (step K1′).
            let mut nan = t(1.0);
            nan.val[0] = f64::NAN;
            assert!(matches!(
                s.factorize(&sym, &nan, Regularization::default()),
                Err(KktError::Structure(_))
            ));
        }
        let s = FaerLdl {
            supernodal: Some(false),
        };
        let sym = s
            .analyze(&k.pattern(), KktForm::NormalEquations, &Natural)
            .unwrap();
        assert!(matches!(
            s.factorize(&sym, &k, reg),
            Err(KktError::NotFinite { position: 1 | 2 })
        ));
    }

    #[test]
    fn gmres_refinement_removes_the_shift_on_a_nearly_singular_system() {
        // A D Aᵀ with two nearly parallel rows of A: λ_min ≈ 10⁻¹² ≪ δ, where
        // the classical iteration x ← x + F⁻¹ r contracts by δ/(λ + δ) ≈ 1
        // per step; GMRES (Algorithm F) converges in a few.
        let a = [
            [1.0, 2.0, 0.0, 1.0],
            [1.0, 2.0, 1e-6, 1.0],
            [0.0, 1.0, 1.0, 3.0],
        ];
        let mut t = Vec::new();
        for i in 0..3 {
            for p in 0..=i {
                let v: f64 = (0..4).map(|j| a[i][j] * a[p][j]).sum();
                t.push((Row::from_usize(i), Col::from_usize(p), v));
            }
        }
        let k = CscMatrix::from_triplets(3, 3, &t).unwrap();
        let s = FaerLdl::default();
        let sym = s
            .analyze(&k.pattern(), KktForm::NormalEquations, &Amd::default())
            .unwrap();
        let (num, _) = s.factorize(&sym, &k, Regularization::default()).unwrap();
        let xs = [1.0, -1.0, 0.5];
        let mut b = vec![0.0; 3];
        crate::kkt::sym_lower_matvec(&k, &xs, &mut b);
        let mut x = vec![0.0; 3];
        let first = s.solve_refined(
            &num,
            &k,
            &b,
            &mut x,
            Refinement {
                max_iters: 0,
                target_residual: 0.0,
            },
        );
        let rep = s.solve_refined(&num, &k, &b, &mut x, Refinement::default());
        assert!(rep.converged && rep.iters <= 4, "{first:?} → {rep:?}");
        assert!(le(rep.residual, first.residual, 0.0));
    }

    #[test]
    fn bumps_are_counted_with_both_signs_and_both_structures() {
        // K = [0 1; 1 0], one primal and one dual unknown, no static
        // shift: the primal pivot 0 is bumped to −β (a negative-sign bump),
        // the dual one is then 1/β > 0.
        let k = CscMatrix::from_triplets(2, 2, &[(Row(1), Col(0), 1.0)]).unwrap();
        let form = KktForm::Augmented {
            n_primal: 1,
            n_dual: 1,
        };
        let reg = Regularization {
            primal: 0.0,
            dual: 0.0,
            pivot_tol: Some(1e-13),
            bump: 1e-8,
            scale: ShiftScale::Equilibrated,
        };
        // A normal-equations matrix with a zero pivot: a positive-sign bump.
        let z = CscMatrix::from_triplets(
            3,
            3,
            &[
                (Row(0), Col(0), 1.0),
                (Row(2), Col(0), 1.0),
                (Row(2), Col(2), 1.0),
            ],
        )
        .unwrap();
        for sn in [Some(false), Some(true)] {
            let s = FaerLdl { supernodal: sn };
            let sym = s.analyze(&k.pattern(), form, &Natural).unwrap();
            let (num, inertia) = s.factorize(&sym, &k, reg).unwrap();
            assert_eq!(inertia, form.expected_inertia(2));
            assert_eq!(num.bumped(), 1, "{sn:?}");
            let sym = s
                .analyze(&z.pattern(), KktForm::NormalEquations, &Natural)
                .unwrap();
            let (num, _) = s.factorize(&sym, &z, reg).unwrap();
            assert_eq!(num.bumped(), 2, "{sn:?}"); // pivots 1 (zero row) and 2 (1 − 1 = 0)
        }
    }

    #[test]
    fn structural_errors() {
        let s = FaerLdl::default();
        let skew = CscMatrix::from_triplets(
            2,
            2,
            &[
                (Row(1), Col(0), 1.0),
                (Row(1), Col(1), 1.0),
                (Row(0), Col(0), 1.0),
                (Row(0), Col(1), 3.0),
            ],
        )
        .unwrap();
        // Symmetric pattern with both triangles: accepted.
        assert!(
            s.analyze(&skew.pattern(), KktForm::NormalEquations, &Amd::default())
                .is_ok()
        );
        let up = CscMatrix::from_triplets(2, 2, &[(Row(0), Col(1), 1.0)]).unwrap();
        assert_eq!(
            s.analyze(&up.pattern(), KktForm::NormalEquations, &Amd::default())
                .err(),
            Some(KktError::NotSymmetric)
        );
        let sym = s
            .analyze(
                &CscMatrix::<f64>::zeros(2, 2).pattern(),
                KktForm::NormalEquations,
                &Amd::default(),
            )
            .unwrap();
        let k = CscMatrix::from_triplets(2, 2, &[(Row(1), Col(0), 1.0), (Row(0), Col(0), 1.0)])
            .unwrap();
        assert!(matches!(
            s.factorize(&sym, &k, Regularization::default()),
            Err(KktError::Structure(_))
        ));
        let k3 = CscMatrix::<f64>::zeros(3, 3);
        assert!(matches!(
            s.factorize(&sym, &k3, Regularization::default()),
            Err(KktError::Structure(_))
        ));
        let bad = Regularization {
            pivot_tol: Some(1e-13),
            bump: 0.0,
            ..Regularization::default()
        };
        assert!(matches!(
            s.factorize(&sym, &CscMatrix::zeros(2, 2), bad),
            Err(KktError::Structure(_))
        ));
        // A value that vanished (left the CSC pattern) is a zero inside the
        // analyzed one.
        let (aug, _, form) = kkt(4, 8, 1.0, 5);
        let sym = s.analyze(&aug.pattern(), form, &Amd::default()).unwrap();
        let mut thin = aug.clone();
        thin.row_idx.remove(1);
        thin.val.remove(1);
        for p in &mut thin.col_ptr[1..] {
            *p -= 1;
        }
        assert!(s.factorize(&sym, &thin, Regularization::default()).is_ok());
    }

    #[test]
    fn s7_refinement_keeps_the_best_iterate() {
        let (aug, _, form) = kkt(8, 20, 4.0, 77);
        let s = FaerLdl::default();
        let sym = s.analyze(&aug.pattern(), form, &Amd::default()).unwrap();
        let b = rhs(28, 77);
        // A factor of a far-away matrix (δ = ρ = 10): refinement against
        // the unregularized K improves or stops, never worsens.
        let far = Regularization {
            primal: 10.0,
            dual: 10.0,
            pivot_tol: None,
            bump: 0.0,
            scale: ShiftScale::Absolute,
        };
        let (num, _) = s.factorize(&sym, &aug, far).unwrap();
        let mut x = vec![0.0; 28];
        let none = s.solve_refined(
            &num,
            &aug,
            &b,
            &mut x,
            Refinement {
                max_iters: 0,
                target_residual: 0.0,
            },
        );
        let rep = s.solve_refined(
            &num,
            &aug,
            &b,
            &mut x,
            Refinement {
                max_iters: 50,
                target_residual: 0.0,
            },
        );
        assert!(le(rep.residual, none.residual, 0.0));
        assert_eq!(none.iters, 0);
        assert!(!rep.converged);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        /// (S6) and the refined residual on random KKT systems in the three
        /// IPM regimes, with every structure choice.
        #[test]
        fn random_kkt_systems(m in 1_usize..15, extra in 0_usize..20, regime in 0_usize..3, sn in 0_usize..3, seed in any::<u64>()) {
            let n = m + extra;
            let spread = [0.0, 4.0, 9.0][regime];
            let supernodal = [None, Some(true), Some(false)][sn];
            let (aug, normal, form) = kkt(m, n, spread, seed);
            let (i, res, _) = solve_ok(&aug, form, Regularization::default(), supernodal, &rhs(n + m, seed));
            prop_assert_eq!(i, form.expected_inertia(n + m));
            prop_assert!(le(res, 1e-10, 0.0), "augmented: {}", res);
            let (i, _, _) = solve_ok(&normal, KktForm::NormalEquations, Regularization::default(), supernodal, &rhs(m, seed));
            prop_assert_eq!(i, KktForm::NormalEquations.expected_inertia(m));
        }
    }
}
