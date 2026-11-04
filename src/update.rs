//! # Chapter 5.5 — The Forrest–Tomlin update and the refactorization policy
//!
//! ## Purpose
//! A simplex iteration replaces one basic column: `B' = B + (a_q − B e_r)
//! e_rᵀ`, where `r` is the leaving *position*. Refactorizing at every
//! iteration would cost far more than the iteration itself, so the factors
//! are *updated*. Forrest and Tomlin (1972) keep `L` untouched, replace one
//! column of `U` by the transformed entering column (the *spike*) and
//! restore triangularity by eliminating one row, recorded as a row eta.
//! Their update keeps `U` as sparse as a fresh factorization in practice,
//! which the product form of the inverse does not.
//!
//! ## Notation
//! As in Chapter 5.4, `B = Pᵀ L R₁⁻¹ ⋯ R_t⁻¹ U Qᵀ`. Let `s` be the spike
//! `R_t ⋯ R_1 L⁻¹ a_q` (indexed by rows, recorded by
//! [`crate::LuFactor::ftran_for_update`]), `α_q = B⁻¹ a_q` and `α_r` its entry
//! at position `r`. Let `τ = step(r)`, `p = p_τ` the pivot row of that
//! step and `u_old = u_{p r}` the diagonal entry of `U` there.
//!
//! ## Algorithm U (Forrest–Tomlin update, classical variant)
//! U1. [Locate.] Find `τ = step(r)` and `p = p_τ`. Fail with
//!     [`LuError::NoSpike`] if no spike was recorded since the factor
//!     last changed.
//! U2. [Spike row.] Copy row `p` of `U` (entries `u_pj`, `step(j) > τ`) into
//!     a dense work vector `w` indexed by positions, and let `σ ← s_p`.
//! U3. [Eliminate.] Think of column `r` of `U` as replaced by `s` and of
//!     step `τ` as moved to the end of the pivot sequence (rows and columns
//!     shifted cyclically). Row `p` is then the only row below the diagonal
//!     and must be eliminated with the rows of the later steps, in step
//!     order: for the steps `k > τ` with `j = q_k`, if `w_j` is not
//!     negligible, set `μ_k ← w_j / u_{p_k j}`, `w_j ← 0`,
//!     `w_l ← w_l − μ_k u_{p_k l}` for the entries of row `p_k`, and
//!     `σ ← σ − μ_k s_{p_k}`. Since row `p_k` only has entries in later
//!     steps, the fill lands where it is eliminated later.
//! U4. [Test.] The new diagonal is `d = σ`. Since `det B' = α_r det B` and
//!     the cyclic shifts of rows and columns cancel in sign, exact arithmetic
//!     gives `d = α_r u_old`. Let `ζ = |s_p| + Σ_k |μ_k s_{p_k}|`, the size
//!     of the terms summed into `σ` (its rounding error is `O(ε ζ)`). If
//!     `|d| ≤ τ_abs max(1, ζ)`, the new basis is singular:
//!     return [`LuError::Singular`] with position `r` and row `p`. If
//!     `|d − α_r u_old| > ε_FT max(|d|, |α_r u_old|)`, the update is
//!     numerically unreliable: return
//!     [`UpdateOutcome::RefactorNeeded`]`(`[`RefactorReason::Unstable`]`)`.
//!     In both cases nothing has been modified: the factor still represents
//!     `B`.
//! U5. [Commit.] Delete the old column `r` and the off-diagonal row `p` from
//!     both copies of `U`; insert the spike entries `s_i` (`i ≠ p`,
//!     `|s_i| > ε_drop`) as the new column `r`; set `u_{pr} ← d`; move
//!     step `τ` to the end of the sequence (its slot `τ` becomes *vacant*
//!     and the step is appended, so the step numbers grow beyond `m − 1`
//!     until the next factorization; only their order matters); store the
//!     row eta
//!     `R = I − e_p μᵀ` if `μ ≠ 0`. Now `R L⁻¹ P B' Q' = U'` with `U'` upper
//!     triangular in the new step order.
//! U6. [Policy.] Count the update and decide whether to ask for a
//!     refactorization (below).
//!
//! The variant is the *classical* one: the spike is stored as a column of
//! `U` and the elimination of row `p` uses a dense position-indexed work
//! vector. The steps of U3 with a nonzero `w_j` are visited through a
//! min-priority queue of steps (task P1.4) rather than by scanning
//! `τ+1..m`, U5 inserts the spike from its recorded pattern and moves step
//! `τ` to the end by leaving a vacant slot (no renumbering), so the cost of
//! an update is proportional to the entries involved, as with the linked
//! permutation lists of Suhl's variant.
//!
//! ## Refactorization policy
//! The factor never refactorizes by itself — the columns of `B` belong to
//! the caller. After a successful update, [`crate::LuFactor::update`]
//! returns [`UpdateOutcome::RefactorNeeded`] with the first applicable
//! reason, the update *already applied* and the factor valid:
//!
//! * [`RefactorReason::UpdateLimit`] — `refactor_interval` updates (default
//!   100) since the last factorization;
//! * [`RefactorReason::FillGrowth`] — `nnz(L) + nnz(U) + nnz(etas)` exceeds
//!   `fill_factor` (default 3) times its value right after the
//!   factorization: solves have become more expensive than a refactorization
//!   would make them;
//! * [`RefactorReason::Growth`] — the *updates* have made `U` grow by more
//!   than `growth_limit` (default `10¹⁰`): with `g_0 = max |u_ij|` right
//!   after the factorization and `β = max |b_ij|`,
//!   `max |u_ij| > growth_limit · max(β, g_0)`.
//!
//! The growth trigger is measured *from the fresh factorization*, not from
//! `B`: a refactorization can only undo the growth the updates added, so a
//! basis whose fresh factors already grow past the limit (MAROS-R7 of
//! Netlib with the Markowitz threshold `0.01`: `max |u| / max |b| > 10¹⁰`
//! straight out of the Markowitz phase) must not ask for one at every
//! update — the old test,
//! `max |u_ij| / β > growth_limit`, did exactly that and turned the solve
//! into a loop of refactorizations (278 instead of 854 iterations per
//! second). The absolute growth `max |u_ij| / β` stays available as a
//! diagnostic ([`crate::LuStats::growth_exceeded`]); taking the larger
//! of `β` and `g_0` as the reference makes the trigger identical to the
//! old one when the fresh factor did not grow (`g_0 ≤ β`). The cure of a
//! grown fresh factor is a tighter threshold, a decision of the caller
//! ([`crate::LuFactor::set_options`] with a larger
//! [`crate::LuOptions::rel_pivot_tol`]; the simplex method of Chapter 6
//! does it).
//!
//! Only [`RefactorReason::Unstable`] comes with the update *not* applied.
//! The caller should refactorize before the next solve in every case; with
//! the first three it may postpone (the factor is exact up to rounding).
//!
//! ## The update as a policy (second edition)
//! The contract makes the update algorithm a type parameter of the
//! factorization, `Lu<T, U = ForrestTomlin<T>>`, through the trait
//! [`BasisUpdate`] (D-0080, addendum): an update policy owns its own data
//! — here the row etas `R_e` and the lists of the etas that read or target
//! each row — and reaches the stored factors through `&mut`
//! [`crate::Factors`]. Besides Algorithm U it provides the steps of the
//! solves that apply its data (F2, B2, HF4 and HB4 of Chapter 5.4), called
//! by the solves at the point where the first edition ran them inline.
//! Bartels–Golub (task L1.4) will be the second implementation. The trait
//! is implementable inside the crate only (its arguments are opaque
//! outside), which keeps the factors' invariants in one place.
//!
//! ## Invariants
//! (F1)–(F3) of Chapter 5.3 hold after every successful update (checked by
//! the property tests through [`crate::LuFactor::check_invariants`]); FTRAN and
//! BTRAN after `k` updates agree with a fresh factorization of the updated
//! basis (property test, relative error ≤ 1e-9).
//!
//! ## References
//! Forrest & Tomlin, "Updated triangular factors of the basis to maintain
//! sparsity in the product form simplex method", Math. Programming 2(1),
//! 1972; Suhl & Suhl, "A fast LU update for linear programming", Annals of
//! OR 43, 1993; Maros (2003), §8.3 (Forrest–Tomlin) and §8.5 (stability
//! checks); Koberstein (2005), §5.4; Huangfu & Hall (2018) for the
//! comparison `d` versus `α_r u_old`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt::Debug;

use crate::error::LuError;
use crate::factor::{Factors, LuOptions, VACANT, count_f64, idx32};
use crate::real::{Real, is_exact_zero, is_zero, lt};
use crate::solve::Workspace;
use crate::{Row, SparseError};

/// Why [`crate::LuFactor::update`] asks for a refactorization.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RefactorReason {
    /// `refactor_interval` updates since the last factorization (applied).
    UpdateLimit,
    /// Fill of `L + U + etas` above `fill_factor` times its initial size
    /// (applied).
    FillGrowth,
    /// Growth added by the updates above `growth_limit` (applied; see
    /// [`UpdateStats::update_growth`]).
    Growth,
    /// The new diagonal disagrees with `α_r · u_old`: the update was
    /// **not** applied and the factor still represents the old basis.
    Unstable,
}

/// Result of a successful update.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UpdateOutcome {
    /// The factor now represents the new basis.
    Updated,
    /// The caller should refactorize; see [`RefactorReason`] for whether
    /// the update was applied.
    RefactorNeeded(RefactorReason),
}

/// Counters of the update policy since the last factorization.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UpdateStats {
    /// Updates since the last factorization.
    pub updates: usize,
    /// Multipliers stored in the row etas.
    pub nnz_eta: usize,
    /// Number of row etas (updates whose spike row needed elimination).
    pub n_etas: usize,
    /// Growth added by the updates: `max_u / max(max_basis, max_u_fresh)`
    /// in the terms of [`crate::LuStats`] (at most `1` right after a
    /// factorization). The policy compares it with
    /// [`LuOptions::growth_limit`].
    pub update_growth: f64,
}

/// A basis-update policy over the stored factors (the contract's
/// `BasisUpdate<T>`; see the module documentation). Every implementation
/// is transactional — on `Err` and on [`RefactorReason::Unstable`] the
/// factors are exactly as before — and never modifies `L`.
pub trait BasisUpdate<T: Real>: Default + Clone + Debug {
    /// Replaces the column at position `r` by the entering column whose
    /// spike the last `ftran_for_update` left in `factors`; `alpha_r` is
    /// entry `r` of its result.
    fn update(
        &mut self,
        factors: &mut Factors<T>,
        opts: &LuOptions,
        r: usize,
        alpha_r: T,
    ) -> Result<UpdateOutcome, LuError>;
    /// Step F2 of the dense FTRAN on the row-indexed work array `y`.
    fn apply_ftran(&self, y: &mut [T]);
    /// Step B2 of the dense BTRAN on the row-indexed work array `w`.
    fn apply_btran(&self, w: &mut [T]);
    /// Step HF4 of the hypersparse FTRAN: `list` is the reach `Λ`, marked
    /// in `mark` with `stamp`; rows the etas add are appended and marked.
    /// `queued` is a zero work array with one flag per eta.
    fn apply_ftran_hyper(
        &self,
        y: &mut [T],
        list: &mut Vec<u32>,
        mark: &mut [u32],
        stamp: u32,
        queued: &mut [bool],
    );
    /// Step HB4 of the hypersparse BTRAN, as [`BasisUpdate::apply_ftran_hyper`].
    fn apply_btran_hyper(
        &self,
        w: &mut [T],
        list: &mut Vec<u32>,
        mark: &mut [u32],
        stamp: u32,
        queued: &mut [bool],
    );
    /// Forgets every update: called after each factorization of an
    /// `m × m` basis, with the update growth of the fresh factors (`0`
    /// after a failed factorization).
    fn reset(&mut self, m: usize, update_growth: f64);
    /// The counters since the last factorization.
    fn stats(&self) -> &UpdateStats;
    /// A short name for logs (`"forrest-tomlin"`).
    fn name(&self) -> &'static str;
}

/// Forrest & Tomlin (1972): row etas and a permuted, modified `U`
/// (Algorithm U).
#[derive(Clone, Debug)]
pub struct ForrestTomlin<T: Real> {
    /// Row etas: eta `e` performs `y[eta_target[e]] −= Σ μ_k y[eta_index[k]]`
    /// over its range.
    eta_start: Vec<usize>,
    eta_target: Vec<u32>,
    eta_index: Vec<u32>,
    eta_value: Vec<T>,
    /// For each row `k`, the row etas that read it (`k` among their
    /// indices), in creation order.
    eta_readers: Vec<Vec<u32>>,
    /// For each row `p`, the row etas that target it, in creation order.
    eta_targets: Vec<Vec<u32>>,
    /// Rows with a nonempty `eta_readers` or `eta_targets` list (possibly
    /// repeated), to clear them at the next factorization.
    eta_rows: Vec<u32>,
    stats: UpdateStats,
}

impl<T: Real> Default for ForrestTomlin<T> {
    fn default() -> Self {
        Self {
            eta_start: vec![0],
            eta_target: Vec::new(),
            eta_index: Vec::new(),
            eta_value: Vec::new(),
            eta_readers: Vec::new(),
            eta_targets: Vec::new(),
            eta_rows: Vec::new(),
            stats: UpdateStats::default(),
        }
    }
}

impl<T: Real> BasisUpdate<T> for ForrestTomlin<T> {
    /// Algorithm U. Errors: [`LuError::Singular`] if the new basis is
    /// singular, [`LuError::NoSpike`] without a recorded spike,
    /// [`LuError::NotFactorized`] on an invalid factor,
    /// [`LuError::Structure`] if `r ≥ m`. On every error, and on
    /// [`RefactorReason::Unstable`], the factor is left unchanged.
    fn update(
        &mut self,
        f: &mut Factors<T>,
        opts: &LuOptions,
        r: usize,
        alpha_r: T,
    ) -> Result<UpdateOutcome, LuError> {
        if !f.valid {
            return Err(LuError::NotFactorized);
        }
        if r >= f.m {
            return Err(SparseError::IndexOutOfRange {
                what: "leaving basis position",
                index: r,
                bound: f.m,
            }
            .into());
        }
        if !f.spike_valid {
            return Err(LuError::NoSpike);
        }
        // U1. Locate.
        let tau = f.col_step[r];
        let p = f.step_row[tau] as usize;
        let u_old = f.diag[p];

        // U2–U3. Eliminate the spike row (the factor is not modified).
        let mut ws = std::mem::take(&mut f.ws);
        let (eta, d, zeta) = eliminate_spike_row(f, opts, tau, p, &mut ws);
        f.ws = ws;

        // U4. Singularity and stability tests (nothing modified yet).
        if !lt(T::from_f64(opts.abs_pivot_tol) * zeta.max(T::ONE), d.abs()) {
            return Err(LuError::Singular {
                rank_deficiency: 1,
                cols: vec![r],
                rows: vec![Row::from_usize(p)],
            });
        }
        let expected = alpha_r * u_old;
        let scale = d.abs().max(expected.abs());
        if lt(
            T::from_f64(opts.update_stability_tol) * scale,
            (d - expected).abs(),
        ) {
            return Ok(UpdateOutcome::RefactorNeeded(RefactorReason::Unstable));
        }

        // U5. Commit.
        self.commit(f, opts, r, tau, p, d, eta);
        debug_assert!(f.check_invariants().is_ok(), "{:?}", f.check_invariants());

        // U6. Refactorization policy.
        Ok(self.policy(f, opts))
    }

    /// Step F2.
    #[inline]
    fn apply_ftran(&self, y: &mut [T]) {
        for e in 0..self.eta_target.len() {
            self.apply_eta(e, y);
        }
    }

    /// Step B2.
    #[inline]
    fn apply_btran(&self, w: &mut [T]) {
        for e in (0..self.eta_target.len()).rev() {
            self.apply_eta_t(e, w);
        }
    }

    /// Step HF4: the row etas in creation order through a min-queue.
    fn apply_ftran_hyper(
        &self,
        y: &mut [T],
        list: &mut Vec<u32>,
        mark: &mut [u32],
        stamp: u32,
        queued: &mut [bool],
    ) {
        if self.eta_target.is_empty() {
            return;
        }
        let mut heap: BinaryHeap<Reverse<u32>> = BinaryHeap::new();
        for &i in list.iter() {
            for &e in &self.eta_readers[i as usize] {
                if !queued[e as usize] {
                    queued[e as usize] = true;
                    heap.push(Reverse(e));
                }
            }
        }
        while let Some(Reverse(e)) = heap.pop() {
            queued[e as usize] = false;
            self.apply_eta(e as usize, y);
            let t = self.eta_target[e as usize] as usize;
            if mark[t] != stamp {
                mark[t] = stamp;
                list.push(idx32(t));
                let readers = &self.eta_readers[t];
                let after = readers.partition_point(|&e2| e2 <= e);
                for &e2 in &readers[after..] {
                    if !queued[e2 as usize] {
                        queued[e2 as usize] = true;
                        heap.push(Reverse(e2));
                    }
                }
            }
        }
    }

    /// Step HB4: the row etas in reverse creation order through a
    /// max-queue.
    fn apply_btran_hyper(
        &self,
        w: &mut [T],
        list: &mut Vec<u32>,
        mark: &mut [u32],
        stamp: u32,
        queued: &mut [bool],
    ) {
        if self.eta_target.is_empty() {
            return;
        }
        let mut heap: BinaryHeap<u32> = BinaryHeap::new();
        for &p in list.iter() {
            for &e in &self.eta_targets[p as usize] {
                if !queued[e as usize] {
                    queued[e as usize] = true;
                    heap.push(e);
                }
            }
        }
        while let Some(e) = heap.pop() {
            queued[e as usize] = false;
            self.apply_eta_t(e as usize, w);
            let r = self.eta_start[e as usize]..self.eta_start[e as usize + 1];
            for &k in &self.eta_index[r] {
                let k = k as usize;
                if mark[k] == stamp {
                    continue;
                }
                mark[k] = stamp;
                list.push(idx32(k));
                let targets = &self.eta_targets[k];
                let before = targets.partition_point(|&e2| e2 < e);
                for &e2 in &targets[..before] {
                    if !queued[e2 as usize] {
                        queued[e2 as usize] = true;
                        heap.push(e2);
                    }
                }
            }
        }
    }

    fn reset(&mut self, m: usize, update_growth: f64) {
        self.eta_start = vec![0];
        self.eta_target.clear();
        self.eta_index.clear();
        self.eta_value.clear();
        if self.eta_readers.len() == m {
            for &k in &self.eta_rows {
                self.eta_readers[k as usize].clear();
                self.eta_targets[k as usize].clear();
            }
        } else {
            self.eta_readers = vec![Vec::new(); m];
            self.eta_targets = vec![Vec::new(); m];
        }
        self.eta_rows.clear();
        self.stats = UpdateStats {
            update_growth,
            ..UpdateStats::default()
        };
    }

    #[inline]
    fn stats(&self) -> &UpdateStats {
        &self.stats
    }

    fn name(&self) -> &'static str {
        "forrest-tomlin"
    }
}

impl<T: Real> ForrestTomlin<T> {
    /// Row eta `e` in FTRAN: `y_t ← y_t − Σ_k μ_k y_{p_k}`.
    #[inline]
    fn apply_eta(&self, e: usize, y: &mut [T]) {
        let r = self.eta_start[e]..self.eta_start[e + 1];
        let s: T = self.eta_index[r.clone()]
            .iter()
            .zip(&self.eta_value[r])
            .map(|(&k, &mu)| mu * y[k as usize])
            .sum();
        y[self.eta_target[e] as usize] -= s;
    }

    /// Row eta `e` in BTRAN: if `w_t ≠ 0`, `w_{p_k} ← w_{p_k} − μ_k w_t`.
    #[inline]
    fn apply_eta_t(&self, e: usize, w: &mut [T]) {
        let w_t = w[self.eta_target[e] as usize];
        if is_exact_zero(w_t) {
            return;
        }
        let r = self.eta_start[e]..self.eta_start[e + 1];
        for (&k, &mu) in self.eta_index[r.clone()].iter().zip(&self.eta_value[r]) {
            w[k as usize] -= mu * w_t;
        }
    }

    /// Step U5.
    #[allow(clippy::too_many_arguments)] // the quantities of steps U1–U4
    fn commit(
        &mut self,
        f: &mut Factors<T>,
        opts: &LuOptions,
        r: usize,
        tau: usize,
        p: usize,
        d: T,
        eta: Vec<(u32, T)>,
    ) {
        let drop_tol = T::from_f64(opts.drop_tol);
        let (r32, p32) = (idx32(r), idx32(p));
        // Delete the old column r and the off-diagonal row p of U.
        let old_col: Vec<u32> = f.u_cols.line(r).0.to_vec();
        for i in old_col {
            f.u_rows.remove(i as usize, r32);
        }
        f.u_cols.clear(r);
        let old_row: Vec<u32> = f.u_rows.line(p).0.to_vec();
        for j in old_row {
            f.u_cols.remove(j as usize, p32);
        }
        f.u_rows.clear(p);
        // Insert the spike as the new column r.
        let mut max_u = f.stats.max_u.max(d.abs().to_f64());
        for &i in &f.spike_idx {
            let (iu, s_i) = (i as usize, f.spike[i as usize]);
            if iu == p || is_zero(s_i, drop_tol) {
                continue;
            }
            f.u_cols.push(r, i, s_i);
            f.u_rows.push(iu, r32, s_i);
            max_u = max_u.max(s_i.abs().to_f64());
        }
        f.diag[p] = d;
        // Move step τ to the end: vacate its slot and append it, so that no
        // other step is renumbered (O(1) instead of O(m)).
        f.step_row[tau] = VACANT;
        f.step_col[tau] = VACANT;
        let last = f.step_row.len();
        f.step_row.push(p32);
        f.step_col.push(r32);
        f.row_step[p] = last;
        f.col_step[r] = last;
        // Store the row eta.
        if !eta.is_empty() {
            let e = idx32(self.eta_target.len());
            self.eta_target.push(p32);
            self.eta_targets[p].push(e);
            self.eta_rows.push(p32);
            for (k, mu) in eta {
                self.eta_index.push(k);
                self.eta_value.push(mu);
                self.eta_readers[k as usize].push(e);
                self.eta_rows.push(k);
            }
            self.eta_start.push(self.eta_index.len());
            f.ws.queued.push(false);
        }
        f.spike_valid = false;
        f.stats.max_u = max_u;
        self.stats.updates += 1;
        self.stats.nnz_eta = self.eta_value.len();
        self.stats.n_etas = self.eta_target.len();
        self.stats.update_growth = f.refresh_stats(opts.growth_limit, self.stats.nnz_eta);
    }

    /// Step U6: the refactorization policy.
    fn policy(&self, f: &Factors<T>, opts: &LuOptions) -> UpdateOutcome {
        let s = &f.stats;
        let u = &self.stats;
        let nnz_now = s.nnz_l + s.nnz_u + u.nnz_eta;
        if u.updates >= opts.refactor_interval {
            UpdateOutcome::RefactorNeeded(RefactorReason::UpdateLimit)
        } else if lt(
            opts.fill_factor * count_f64(f.nnz_factor0),
            count_f64(nnz_now),
        ) {
            UpdateOutcome::RefactorNeeded(RefactorReason::FillGrowth)
        } else if lt(opts.growth_limit, u.update_growth) {
            UpdateOutcome::RefactorNeeded(RefactorReason::Growth)
        } else {
            UpdateOutcome::Updated
        }
    }
}

/// Steps U2–U3: returns the row eta `(p_k, μ_k)`, the new diagonal
/// `d = σ` and the scale `ζ = |s_p| + Σ |μ_k s_{p_k}|` of the terms
/// summed into it. The work array `ws.work` (by positions) is zero on
/// entry and on exit.
///
/// The steps `k` with a nonzero `w_{q_k}` are visited in increasing
/// order through a min-priority queue instead of scanning `τ+1..m`:
/// the same operations in the same order as the scan, at a cost
/// proportional to the entries of `U` involved (Chapter 5.4).
fn eliminate_spike_row<T: Real>(
    f: &Factors<T>,
    opts: &LuOptions,
    tau: usize,
    p: usize,
    ws: &mut Workspace<T>,
) -> (Vec<(u32, T)>, T, T) {
    let drop_tol = T::from_f64(opts.drop_tol);
    let w = &mut ws.work;
    let flag = &mut ws.flag;
    let mut heap: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    // U2. Row p of U into the work vector.
    let (idx, val) = f.u_rows.line(p);
    for (&j, &u) in idx.iter().zip(val) {
        w[j as usize] = u;
        flag[j as usize] = true;
        heap.push(Reverse(f.col_step[j as usize]));
    }
    let mut sigma = f.spike[p];
    let mut zeta = sigma.abs();

    // U3. Eliminate row p with the rows of steps τ+1, …, m−1.
    let mut eta: Vec<(u32, T)> = Vec::new();
    while let Some(Reverse(k)) = heap.pop() {
        debug_assert!(k > tau);
        let j = f.step_col[k] as usize;
        let w_j = w[j];
        w[j] = T::ZERO;
        flag[j] = false;
        if is_zero(w_j, drop_tol) {
            continue;
        }
        let p_k = f.step_row[k] as usize;
        let mu = w_j / f.diag[p_k];
        let (idx, val) = f.u_rows.line(p_k);
        for (&l, &u) in idx.iter().zip(val) {
            let l = l as usize;
            if !flag[l] {
                flag[l] = true;
                heap.push(Reverse(f.col_step[l]));
            }
            w[l] -= mu * u;
        }
        sigma -= mu * f.spike[p_k];
        zeta += (mu * f.spike[p_k]).abs();
        eta.push((idx32(p_k), mu));
    }
    (eta, sigma, zeta)
}
