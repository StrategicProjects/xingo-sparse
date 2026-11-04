//! # Chapter 5.4 — FTRAN and BTRAN, dense and hypersparse
//!
//! ## Purpose
//! The simplex method needs `α_q = B⁻¹ a_q` (FTRAN, forward transformation)
//! and `ρ_r = B⁻ᵀ e_r`, `y = B⁻ᵀ c_B` (BTRAN, backward transformation).
//! This chapter applies the factors of Chapter 5.3 and the row etas of
//! Chapter 5.5 to a right-hand side, in one of two ways:
//!
//! * the *dense* Algorithms F and B run over every elimination step,
//!   skipping exact zeros: `O(m + nnz(L) + nnz(U) + nnz(etas))` whatever
//!   the right-hand side;
//! * the *hypersparse* Algorithms HF and HB first compute, by depth-first
//!   search in the graphs of the factors, the set of entries that can
//!   become nonzero (the *reach* of the right-hand side, Gilbert & Peierls
//!   1988), and then perform only the operations of the reach, in a
//!   topological order. Their cost is proportional to the number of
//!   entries and factor nonzeros actually involved — on the large sparse LPs
//!   (Kennington `ken-*`, `pds-*`) a handful out of `m ≈ 10⁵`, which is what
//!   Hall & McKinnon (2005) call *hypersparsity*.
//!
//! The choice between the two is automatic, per call (policy below).
//!
//! ## Notation
//! After `t` updates the factor represents
//! ```text
//!   B = Pᵀ L R₁⁻¹ ⋯ R_t⁻¹ U Qᵀ,
//! ```
//! where `R_e = I − e_{p} μ_eᵀ` is the row eta of update `e` (target row
//! `p`, multipliers `μ_e` on rows pivoted after `p`). FTRAN maps a vector
//! indexed by *rows* to one indexed by *positions*; BTRAN the reverse.
//! For a row `p` let `q(p) = q_{step(p)}` be the position pivoted with it,
//! and for a position `q` let `p(q) = p_{step(q)}`.
//!
//! The four *graphs* of the hypersparse solves have an arc `a → b` when
//! the value at `b` is modified by an operation that reads the value at
//! `a`:
//!
//! * `G_L` on rows: `p → i` for every multiplier `l_ip` of the `L` eta with
//!   pivot row `p` (column-wise `L`, as stored);
//! * `G_U` on rows: `p → i` for every off-diagonal `u_{i q(p)}` of column
//!   `q(p)` of `U` (`U` by columns);
//! * `G_Uᵀ` on positions: `q → j` for every off-diagonal `u_{p(q) j}` of row
//!   `p(q)` of `U` (`U` by rows);
//! * `G_Lᵀ` on rows: `i → p` for every multiplier `l_ip` (`L` by *rows*, a
//!   transposed copy built once per factorization).
//!
//! The row etas are applied in creation order (FTRAN) or in reverse
//! creation order (BTRAN); their dependencies are held by two lists per
//! row: the etas that *read* row `k` (`k` among their `p_k`) and the etas
//! that *target* row `p`.
//!
//! ## Algorithm F (dense FTRAN: solve `B x = b`)
//! F1. [L.] For each `L` eta in pivot order (pivot row `p`, multipliers
//!     `l_ip`): if `y_p ≠ 0`, set `y_i ← y_i − l_ip y_p`.
//! F2. [R.] For each row eta in creation order (target `p`): set
//!     `y_p ← y_p − Σ_k μ_k y_{p_k}`. The vector `y` now holds the *spike*
//!     `R L⁻¹ b` — recorded by [`crate::LuFactor::ftran_for_update`].
//! F3. [U.] For the steps `s` in decreasing order, with pivot `(p_s, q_s)`:
//!     `x_{q_s} ← y_{p_s} / u_{p_s q_s}`; if nonzero, `y_i ← y_i − u_{i q_s}
//!     x_{q_s}` for the entries of column `q_s` of `U` (all in rows of earlier
//!     steps).
//!
//! ## Algorithm B (dense BTRAN: solve `Bᵀ y = c`)
//! B1. [Uᵀ.] For the steps `s` in increasing order:
//!     `w_{p_s} ← c_{q_s} / u_{p_s q_s}`;
//!     if nonzero, `c_j ← c_j − u_{p_s j} w_{p_s}` for the entries of row
//!     `p_s` of `U` (all in positions of later steps).
//! B2. [Rᵀ.] For each row eta in *reverse* creation order (target `p`): if
//!     `w_p ≠ 0`, `w_{p_k} ← w_{p_k} − μ_k w_p`.
//! B3. [Lᵀ.] For each `L` eta in *reverse* pivot order (pivot row `p`):
//!     `w_p ← w_p − Σ_i l_ip w_i`. Then `y = w`.
//!
//! Correctness: F1–F3 apply `L⁻¹`, `R_t ⋯ R_1` and `U⁻¹` (with the
//! permutations implicit in the indexing) to `b`; B1–B3 apply the
//! transposes in the opposite order.
//!
//! ## Algorithm D (reach: depth-first search, Gilbert & Peierls)
//! Given a graph `G` and a set `S` of start nodes, produce the nodes
//! reachable from `S` in a *topological* order (every arc `a → b` between
//! reached nodes has `a` before `b`).
//!
//! D1. [Start.] Take a fresh mark (a stamp: no clearing). Empty the output
//!     list `post`. For each `s ∈ S` not yet marked, mark it and push
//!     `(s, 0)` on the stack; then run D2 until the stack is empty.
//! D2. [Step.] Let `(v, k)` be the top of the stack. If `v` has a `k`-th
//!     successor `w`, advance `k`; if `w` is unmarked, mark it and push
//!     `(w, 0)`. Otherwise pop `v` and append it to `post` (`v` is
//!     *finished*).
//! D3. [Order.] A node finishes after every node reachable from it, so
//!     `post` read backwards is a topological order.
//!
//! The stack is explicit (no recursion: the paths in `G_L` can be `10⁵`
//! long). Cost: `O(|reach| + arcs out of the reach)`.
//!
//! ## Algorithm HF (hypersparse FTRAN)
//! The right-hand side `b` is given with its pattern.
//! HF1. [Load.] Copy the nonzeros of `b` into the row work array `y`;
//!      empty `b` (it receives the result).
//! HF2. [Reach of L.] Run Algorithm D on `G_L` from the rows of HF1.
//! HF3. [Apply L.] For each reached row `p` in topological order: if `p`
//!      is the pivot of an `L` eta and `y_p ≠ 0`, set `y_i ← y_i − l_ip y_p`.
//!      The reached rows form the list `Λ` (marked).
//! HF4. [Row etas.] Put in a min-priority queue every eta that reads a row
//!      of `Λ`. While the queue is not empty, take the smallest eta `e`
//!      (target `t`): `y_t ← y_t − Σ_k μ_k y_{p_k}`; if `t ∉ Λ`, add it to
//!      `Λ` and queue the etas after `e` that read `t`. (An eta before `e`
//!      that reads `t` was applied — or skipped — while `y_t` was still
//!      zero, exactly as in F2.) For [`crate::LuFactor::ftran_for_update`],
//!      record the spike `y` on `Λ`.
//! HF5. [Reach of U.] Run Algorithm D on `G_U` from `Λ`.
//! HF6. [Apply U.] For each reached row `p` in topological order:
//!      `x_{q(p)} ← y_p / u_{p q(p)}`, `y_p ← 0`; if `x_{q(p)} ≠ 0`, set
//!      `y_i ← y_i − u_{i q(p)} x_{q(p)}` along column `q(p)`. The pattern
//!      of the result is the set of positions `q(p)` of the reach.
//!
//! ## Algorithm HB (hypersparse BTRAN)
//! HB1. [Load.] Copy the nonzeros of `c` into the position work array;
//!      empty `c`.
//! HB2. [Reach of Uᵀ.] Run Algorithm D on `G_Uᵀ` from the positions of HB1.
//! HB3. [Apply Uᵀ.] For each reached position `q` in topological order:
//!      `w_{p(q)} ← c_q / u_{p(q) q}`, `c_q ← 0`; if nonzero,
//!      `c_j ← c_j − u_{p(q) j} w_{p(q)}` along row `p(q)`. The rows `p(q)`
//!      form the list `Λ` (marked).
//! HB4. [Row etas, transposed.] Put in a max-priority queue every eta whose
//!      target is in `Λ`. While the queue is not empty, take the largest eta
//!      `e` (target `t`): if `w_t ≠ 0`, `w_{p_k} ← w_{p_k} − μ_k w_t`; for
//!      every `p_k ∉ Λ`, add it to `Λ` and queue the etas *before* `e` that
//!      target `p_k`. (The etas after `e` that target `p_k` already ran,
//!      with `w_{p_k} = 0`: no-ops in B2 too.)
//! HB5. [Reach of Lᵀ.] Run Algorithm D on `G_Lᵀ` from `Λ`.
//! HB6. [Apply Lᵀ.] For each reached row `i` in topological order: if
//!      `w_i ≠ 0`, set `w_p ← w_p − l_ip w_i` for every multiplier of row
//!      `i` of `L`. The result `y = w` has the reach as pattern.
//!
//! The direction of the arcs is the delicate point. In FTRAN `L` and `U`
//! are used by *columns* (scatter: a known value pushes into the rows it
//! modifies), so the graphs are those of the stored columns. In BTRAN the
//! dense algorithm uses `U` by rows (B1 is already a scatter, along row
//! `p(q)`: arcs `q → j`) but `L` by columns as a *gather* (B3 sums over the
//! column of eta `p`): a gather cannot be driven by a reach, so HB6 turns it
//! into a scatter along the rows of `L` (`i → p`), which needs the
//! transposed copy of `L`. The row etas are the opposite: F2 is a gather
//! (driven by the "readers" lists) and B2 a scatter (driven by the
//! "targets" lists).
//!
//! ## Choice of the algorithm (alternation policy)
//! Following Hall & McKinnon (2005, §4) and HiGHS, the hypersparse
//! algorithm is used for a call iff
//! (a) the right-hand side is in sparse mode (it has a pattern);
//! (b) its density `count/m` is at most
//!     [`LuOptions::hyper_rhs_density`](crate::LuOptions) (`0.10`);
//! (c) the *expected* density of the result — an exponential moving
//!     average, with weight [`HYPER_HISTORY_WEIGHT`] for the newest
//!     result, of the result densities of the previous calls of the same
//!     kind (FTRAN, BTRAN, FTRAN for update; kept across factorizations,
//!     starting at 0) — is at most
//!     [`LuOptions::hyper_result_density`](crate::LuOptions)
//!     (`0.10`).
//!
//! When the result fills in, the reach costs more than the dense sweep
//! (the search visits every arc once, then the numerical pass visits them
//! again), so (c) switches a kind of operation to the dense algorithms
//! after about three dense results, and back once results are sparse again
//! (the dense algorithms measure their result density exactly). The
//! per-kind statistics ([`SolveStats`]) are in [`crate::LuStats`].
//!
//! ## Division of labour (second edition)
//! Steps F2, B2, HF4 and HB4 — the row etas — belong to the basis-update
//! policy (Chapter 5.5): they are the methods
//! [`BasisUpdate::apply_ftran`], [`BasisUpdate::apply_btran`],
//! [`BasisUpdate::apply_ftran_hyper`] and [`BasisUpdate::apply_btran_hyper`]
//! of [`crate::ForrestTomlin`], called at exactly the place where the first
//! edition ran them inline, with the same work arrays. Everything else is
//! here, on the stored factors.
//!
//! ## Invariants
//! (V1) On return the [`SparseVec`] is in sparse mode. After a dense solve
//!      its pattern is the exact support of the result (in increasing
//!      order); after a hypersparse solve it is the *symbolic reach*, a
//!      superset of the support that may contain explicit zeros (allowed by
//!      the invariant (V2) of [`SparseVec`]; call [`SparseVec::tidy`] to
//!      drop zeros and tiny entries). Computing the exact support would
//!      cost one more pass for no benefit to the simplex method.
//! (V2) Both work arrays are zero between calls.
//! (V3) The dense and the hypersparse algorithms perform the same
//!      operations except for the order of independent updates (and
//!      HB6, which sums the terms of B3 one at a time): their results agree
//!      to rounding (property test: relative error ≤ 1e-12).
//!
//! ## Complexity
//! Dense: `O(m + nnz(L) + nnz(U) + nnz(etas))`. Hypersparse:
//! `O(r + f + e log e)` with `r` the reach sizes, `f` the factor nonzeros
//! in the columns (or rows) of the reach and `e` the number of row etas
//! involved.
//!
//! ## References
//! J. R. Gilbert and T. Peierls, "Sparse partial pivoting in time
//! proportional to arithmetic operations", SIAM J. Sci. Stat. Comput. 9(5),
//! 1988 (the reach by depth-first search); J. A. J. Hall and K. I. M.
//! McKinnon, "Hyper-sparsity in the revised simplex method and how to
//! exploit it", Comput. Optim. Appl. 32, 2005 (hypersparse FTRAN/BTRAN,
//! the density thresholds); T. A. Davis, *Direct Methods for Sparse Linear
//! Systems* (2006), §3.2 (`cs_reach`, `cs_dfs`); Maros (2003), §5.4–5.5;
//! Forrest & Tomlin (1972); Suhl & Suhl (1990), §4; Koberstein (2005),
//! §5.3; HiGHS `HFactor` (studied, not copied).

use crate::SparseVec;
use crate::factor::{Factors, Lu, VACANT, idx32};
use crate::real::{Real, is_exact_zero, lt};
use crate::update::BasisUpdate;

/// Weight of the newest result in the moving average of the result
/// density of each kind of solve (policy (c) of the module documentation).
pub const HYPER_HISTORY_WEIGHT: f64 = 0.05;

/// "No `L` eta" in the map from rows to `L` etas of [`Factors`].
pub(crate) const NO_ETA: u32 = u32::MAX;

/// Counters of one kind of solve (FTRAN, BTRAN or FTRAN for update),
/// accumulated over the life of the [`Lu`] (not reset by a
/// factorization). The contract's mean densities are the methods
/// [`SolveStats::mean_rhs_density`] and
/// [`SolveStats::mean_result_density`] over exact sums (no division per
/// call).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolveStats {
    /// Number of calls.
    pub calls: usize,
    /// Calls served by the hypersparse algorithm (HF or HB).
    pub hyper_calls: usize,
    /// Sum over the calls of the right-hand side density `count/m`.
    pub rhs_density_sum: f64,
    /// Sum over the calls of the result density `count/m` (pattern size:
    /// the reach for hypersparse calls, the exact support otherwise).
    pub result_density_sum: f64,
}

impl SolveStats {
    /// Mean right-hand side density (0 without calls).
    pub fn mean_rhs_density(&self) -> f64 {
        self.mean(self.rhs_density_sum)
    }

    /// Mean result density (0 without calls).
    pub fn mean_result_density(&self) -> f64 {
        self.mean(self.result_density_sum)
    }

    /// Fraction of the calls served by the hypersparse algorithm.
    pub fn hyper_fraction(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)] // counts far below 2⁵³
        let h = self.hyper_calls as f64;
        self.mean(h)
    }

    fn mean(&self, sum: f64) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let c = self.calls as f64;
            sum / c
        }
    }

    fn record(&mut self, hyper: bool, rhs_density: f64, result_density: f64) {
        self.calls += 1;
        self.hyper_calls += usize::from(hyper);
        self.rhs_density_sum += rhs_density;
        self.result_density_sum += result_density;
    }
}

/// The kinds of solve, for the statistics and the moving averages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Ftran = 0,
    Btran = 1,
    FtranUpdate = 2,
}

/// Work arrays of the solves and of the update (invariant (V2): the value
/// arrays are zero between calls).
#[derive(Clone, Debug)]
pub(crate) struct Workspace<T: Real> {
    /// Value work array (rows in FTRAN, positions in BTRAN and in the
    /// update's spike row).
    pub work: Vec<T>,
    /// Second value work array (rows, BTRAN).
    pub work2: Vec<T>,
    /// Node marks of Algorithm D (and of the lists `Λ`).
    pub mark: Vec<u32>,
    /// Current stamp: `mark[v] == stamp` means marked.
    pub stamp: u32,
    /// The explicit stack of Algorithm D: `(node, next arc)`.
    pub stack: Vec<(u32, u32)>,
    /// The list `Λ`.
    pub list: Vec<u32>,
    /// The postorder of Algorithm D.
    pub post: Vec<u32>,
    /// Row etas currently in the priority queue of HF4/HB4.
    pub queued: Vec<bool>,
    /// Flags by position for the update's elimination (Chapter 5.5).
    pub flag: Vec<bool>,
}

impl<T: Real> Default for Workspace<T> {
    fn default() -> Self {
        Self {
            work: Vec::new(),
            work2: Vec::new(),
            mark: Vec::new(),
            stamp: 0,
            stack: Vec::new(),
            list: Vec::new(),
            post: Vec::new(),
            queued: Vec::new(),
            flag: Vec::new(),
        }
    }
}

impl<T: Real> Workspace<T> {
    /// Sizes the arrays for dimension `m` (all zero).
    pub fn resize(&mut self, m: usize) {
        if self.work.len() != m {
            *self = Self {
                work: vec![T::ZERO; m],
                work2: vec![T::ZERO; m],
                mark: vec![0; m],
                flag: vec![false; m],
                ..Self::default()
            };
        }
    }

    /// A fresh stamp for Algorithm D, step D1.
    pub fn new_stamp(&mut self) -> u32 {
        if self.stamp == u32::MAX {
            self.mark.fill(0);
            self.stamp = 0;
        }
        self.stamp += 1;
        self.stamp
    }
}

/// Algorithm D: the nodes reachable from `seeds` in the graph whose
/// successors of `v` are `adj(v)` are appended to `post` in postorder
/// (read it backwards for a topological order). Uses the current stamp:
/// marked nodes are not entered.
fn reach<'g>(
    mark: &mut [u32],
    stamp: u32,
    stack: &mut Vec<(u32, u32)>,
    seeds: &[u32],
    post: &mut Vec<u32>,
    adj: impl Fn(usize) -> &'g [u32],
) {
    // D1.
    for &s in seeds {
        if mark[s as usize] == stamp {
            continue;
        }
        mark[s as usize] = stamp;
        stack.push((s, 0));
        // D2.
        while let Some(top) = stack.last_mut() {
            let (v, k) = *top;
            let succ = adj(v as usize);
            if let Some(&w) = succ.get(k as usize) {
                top.1 = k + 1;
                if mark[w as usize] != stamp {
                    mark[w as usize] = stamp;
                    stack.push((w, 0));
                }
            } else {
                stack.pop();
                post.push(v);
            }
        }
    }
}

impl<T: Real, U: BasisUpdate<T>> Lu<T, U> {
    /// Overwrites `rhs` (indexed by rows) with `x = B⁻¹ rhs` (indexed by
    /// positions) — Algorithm F or HF (see the alternation policy) — and,
    /// for [`Kind::FtranUpdate`], records the spike `R L⁻¹ a_q` (after step
    /// F2/HF4) for the next update.
    ///
    /// # Panics
    /// If the factor is not valid or `rhs.size() != m` — both are
    /// violations of the calling contract.
    pub(crate) fn ftran_kind(&mut self, rhs: &mut SparseVec<T>, kind: Kind) {
        self.factors.assert_ready(rhs);
        let rhs_density = rhs.density();
        let hyper = self.use_hyper(rhs, kind);
        let mut ws = std::mem::take(&mut self.factors.ws);
        let for_update = kind == Kind::FtranUpdate;
        if hyper {
            self.factors
                .ftran_hyper(&self.update, rhs, &mut ws, for_update);
        } else {
            self.factors
                .ftran_dense(&self.update, rhs, &mut ws, for_update);
        }
        self.factors.ws = ws;
        if for_update {
            self.factors.spike_valid = true;
        }
        self.record(kind, hyper, rhs_density, rhs.density());
    }

    /// Overwrites `rhs` (indexed by positions) with `y = B⁻ᵀ rhs` (indexed
    /// by rows), by Algorithm B or HB (see the alternation policy).
    ///
    /// # Panics
    /// As [`Lu::ftran_kind`].
    pub(crate) fn btran_impl(&mut self, rhs: &mut SparseVec<T>) {
        self.factors.assert_ready(rhs);
        let rhs_density = rhs.density();
        let hyper = self.use_hyper(rhs, Kind::Btran);
        let mut ws = std::mem::take(&mut self.factors.ws);
        if hyper {
            self.factors.btran_hyper(&self.update, rhs, &mut ws);
        } else {
            self.factors.btran_dense(&self.update, rhs, &mut ws);
        }
        self.factors.ws = ws;
        self.record(Kind::Btran, hyper, rhs_density, rhs.density());
    }

    /// The alternation policy (a)–(c).
    fn use_hyper(&self, rhs: &SparseVec<T>, kind: Kind) -> bool {
        rhs.is_sparse()
            && !lt(self.opts.hyper_rhs_density, rhs.density())
            && !lt(
                self.opts.hyper_result_density,
                self.expected_density[kind as usize],
            )
    }

    /// Statistics and moving average of one call.
    fn record(&mut self, kind: Kind, hyper: bool, rhs_density: f64, result_density: f64) {
        let ema = &mut self.expected_density[kind as usize];
        *ema = (1.0 - HYPER_HISTORY_WEIGHT) * *ema + HYPER_HISTORY_WEIGHT * result_density;
        let st = &mut self.factors.stats;
        let s = match kind {
            Kind::Ftran => &mut st.ftran,
            Kind::Btran => &mut st.btran,
            Kind::FtranUpdate => &mut st.ftran_update,
        };
        s.record(hyper, rhs_density, result_density);
    }

    /// Expected result density of each kind of solve (policy (c)), for
    /// diagnostics: `[ftran, btran, ftran_for_update]`.
    pub fn expected_densities(&self) -> [f64; 3] {
        self.expected_density
    }
}

impl<T: Real> Factors<T> {
    fn assert_ready(&self, rhs: &SparseVec<T>) {
        assert!(
            self.valid,
            "solve with a factor that holds no valid factorization"
        );
        assert_eq!(rhs.size(), self.m, "right-hand side has the wrong size");
    }

    /// Records the spike `y` (row-indexed) given on the rows of `rows`,
    /// or on all rows when `rows` is `None`.
    fn record_spike(&mut self, y: &[T], rows: Option<&[u32]>) {
        for &i in &self.spike_idx {
            self.spike[i as usize] = T::ZERO;
        }
        self.spike_idx.clear();
        match rows {
            Some(rows) => {
                for &i in rows {
                    self.spike[i as usize] = y[i as usize];
                }
                self.spike_idx.extend_from_slice(rows);
            }
            None => {
                for (i, (&v, s)) in y.iter().zip(self.spike.iter_mut()).enumerate() {
                    if !is_exact_zero(v) {
                        *s = v;
                        self.spike_idx.push(idx32(i));
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------------
    // Dense algorithms F and B.

    /// Algorithm F.
    fn ftran_dense<U: BasisUpdate<T>>(
        &mut self,
        upd: &U,
        rhs: &mut SparseVec<T>,
        ws: &mut Workspace<T>,
        for_update: bool,
    ) {
        let y = rhs.values_mut();
        self.apply_l(y);
        upd.apply_ftran(y);
        if for_update {
            self.record_spike(y, None);
        }
        self.solve_u(y, &mut ws.work2);
        rhs.rebuild_pattern();
    }

    /// Algorithm B.
    fn btran_dense<U: BasisUpdate<T>>(
        &self,
        upd: &U,
        rhs: &mut SparseVec<T>,
        ws: &mut Workspace<T>,
    ) {
        let c = rhs.values_mut();
        let w = &mut ws.work2;
        self.solve_ut(c, w);
        upd.apply_btran(w);
        self.apply_lt(w);
        // c is all zero after B1; swap the result in and keep (V2).
        c.swap_with_slice(w);
        rhs.rebuild_pattern();
    }

    /// Step F1.
    pub(crate) fn apply_l(&self, y: &mut [T]) {
        for (e, &p) in self.l_pivot.iter().enumerate() {
            let y_p = y[p as usize];
            if is_exact_zero(y_p) {
                continue;
            }
            let r = self.l_start[e]..self.l_start[e + 1];
            for (&i, &l) in self.l_index[r.clone()].iter().zip(&self.l_value[r]) {
                y[i as usize] -= l * y_p;
            }
        }
    }

    /// Step F3: on entry `y` is indexed by rows, on exit by positions; `x`
    /// is a zero scratch array of length `m`, zero again on exit.
    fn solve_u(&self, y: &mut [T], x: &mut [T]) {
        for s in (0..self.step_row.len()).rev() {
            let (p, q) = (self.step_row[s], self.step_col[s]);
            if p == VACANT {
                continue;
            }
            let (p, q) = (p as usize, q as usize);
            let x_q = y[p] / self.diag[p];
            x[q] = x_q;
            if is_exact_zero(x_q) {
                continue;
            }
            let (idx, val) = self.u_cols.line(q);
            for (&i, &u) in idx.iter().zip(val) {
                y[i as usize] -= u * x_q;
            }
        }
        // Every y_p was consumed (y is now zero up to the values x took).
        y.copy_from_slice(x);
        x.fill(T::ZERO);
    }

    /// Step B1: `c` is indexed by positions (and left zero); the result is
    /// written into the zero row-indexed array `w`.
    fn solve_ut(&self, c: &mut [T], w: &mut [T]) {
        for s in 0..self.step_row.len() {
            let (p, q) = (self.step_row[s], self.step_col[s]);
            if p == VACANT {
                continue;
            }
            let (p, q) = (p as usize, q as usize);
            let w_p = c[q] / self.diag[p];
            c[q] = T::ZERO;
            w[p] = w_p;
            if is_exact_zero(w_p) {
                continue;
            }
            let (idx, val) = self.u_rows.line(p);
            for (&j, &u) in idx.iter().zip(val) {
                c[j as usize] -= u * w_p;
            }
        }
    }

    /// Step B3.
    fn apply_lt(&self, w: &mut [T]) {
        for (e, &p) in self.l_pivot.iter().enumerate().rev() {
            let r = self.l_start[e]..self.l_start[e + 1];
            let s: T = self.l_index[r.clone()]
                .iter()
                .zip(&self.l_value[r])
                .map(|(&i, &l)| l * w[i as usize])
                .sum();
            w[p as usize] -= s;
        }
    }

    // ----------------------------------------------------------------------
    // Hypersparse algorithms HF and HB.

    /// The successors of row `p` in `G_L`.
    #[inline]
    fn l_succ(&self, p: usize) -> &[u32] {
        match self.l_eta_of_row[p] {
            NO_ETA => &[],
            e => &self.l_index[self.l_start[e as usize]..self.l_start[e as usize + 1]],
        }
    }

    /// Position pivoted with row `p`.
    #[inline]
    fn q_of(&self, p: usize) -> usize {
        self.step_col[self.row_step[p]] as usize
    }

    /// Row pivoted with position `q`.
    #[inline]
    fn p_of(&self, q: usize) -> usize {
        self.step_row[self.col_step[q]] as usize
    }

    /// Algorithm HF.
    fn ftran_hyper<U: BasisUpdate<T>>(
        &mut self,
        upd: &U,
        rhs: &mut SparseVec<T>,
        ws: &mut Workspace<T>,
        for_update: bool,
    ) {
        // HF1. Load.
        let mut seeds = std::mem::take(&mut ws.list);
        seeds.clear();
        for &i in rhs.pattern().unwrap_or(&[]) {
            let v = rhs.get(i);
            if !is_exact_zero(v) {
                ws.work[i as usize] = v;
                seeds.push(i);
            }
        }
        rhs.clear();
        // HF2. Reach of L.
        let stamp = ws.new_stamp();
        ws.post.clear();
        reach(
            &mut ws.mark,
            stamp,
            &mut ws.stack,
            &seeds,
            &mut ws.post,
            |p| self.l_succ(p),
        );
        // HF3. Apply L in topological order; Λ = the reach.
        let y = &mut ws.work;
        for &p in ws.post.iter().rev() {
            let e = self.l_eta_of_row[p as usize];
            let y_p = y[p as usize];
            if e == NO_ETA || is_exact_zero(y_p) {
                continue;
            }
            let r = self.l_start[e as usize]..self.l_start[e as usize + 1];
            for (&i, &l) in self.l_index[r.clone()].iter().zip(&self.l_value[r]) {
                y[i as usize] -= l * y_p;
            }
        }
        let mut list = seeds;
        list.clear();
        list.extend(ws.post.iter().rev());
        // HF4. Row etas, in creation order through a min-queue
        // (Chapter 5.5: the update policy owns them).
        upd.apply_ftran_hyper(
            &mut ws.work,
            &mut list,
            &mut ws.mark,
            ws.stamp,
            &mut ws.queued,
        );
        if for_update {
            self.record_spike(&ws.work, Some(&list));
        }
        // HF5. Reach of U.
        let stamp = ws.new_stamp();
        ws.post.clear();
        reach(
            &mut ws.mark,
            stamp,
            &mut ws.stack,
            &list,
            &mut ws.post,
            |p| self.u_cols.line(self.q_of(p)).0,
        );
        // HF6. Apply U in topological order.
        let y = &mut ws.work;
        for &p in ws.post.iter().rev() {
            let p = p as usize;
            let q = self.q_of(p);
            let x_q = y[p] / self.diag[p];
            y[p] = T::ZERO;
            rhs.set(idx32(q), x_q);
            if is_exact_zero(x_q) {
                continue;
            }
            let (idx, val) = self.u_cols.line(q);
            for (&i, &u) in idx.iter().zip(val) {
                y[i as usize] -= u * x_q;
            }
        }
        ws.list = list;
    }

    /// Algorithm HB.
    fn btran_hyper<U: BasisUpdate<T>>(
        &self,
        upd: &U,
        rhs: &mut SparseVec<T>,
        ws: &mut Workspace<T>,
    ) {
        // HB1. Load (positions).
        let mut list = std::mem::take(&mut ws.list);
        list.clear();
        for &q in rhs.pattern().unwrap_or(&[]) {
            let v = rhs.get(q);
            if !is_exact_zero(v) {
                ws.work[q as usize] = v;
                list.push(q);
            }
        }
        rhs.clear();
        // HB2. Reach of Uᵀ.
        let stamp = ws.new_stamp();
        ws.post.clear();
        reach(
            &mut ws.mark,
            stamp,
            &mut ws.stack,
            &list,
            &mut ws.post,
            |q| self.u_rows.line(self.p_of(q)).0,
        );
        // HB3. Apply Uᵀ in topological order; Λ = the pivot rows reached.
        list.clear();
        for &q in ws.post.iter().rev() {
            let q = q as usize;
            let p = self.p_of(q);
            let w_p = ws.work[q] / self.diag[p];
            ws.work[q] = T::ZERO;
            ws.work2[p] = w_p;
            list.push(idx32(p));
            if is_exact_zero(w_p) {
                continue;
            }
            let (idx, val) = self.u_rows.line(p);
            for (&j, &u) in idx.iter().zip(val) {
                ws.work[j as usize] -= u * w_p;
            }
        }
        let stamp = ws.new_stamp();
        for &p in &list {
            ws.mark[p as usize] = stamp;
        }
        // HB4. Row etas, in reverse creation order through a max-queue
        // (Chapter 5.5: the update policy owns them).
        upd.apply_btran_hyper(
            &mut ws.work2,
            &mut list,
            &mut ws.mark,
            ws.stamp,
            &mut ws.queued,
        );
        // HB5. Reach of Lᵀ.
        let stamp = ws.new_stamp();
        ws.post.clear();
        reach(
            &mut ws.mark,
            stamp,
            &mut ws.stack,
            &list,
            &mut ws.post,
            |i| &self.lt_pivot[self.lt_start[i]..self.lt_start[i + 1]],
        );
        // HB6. Apply Lᵀ in topological order, then store the result.
        let w = &mut ws.work2;
        for &i in ws.post.iter().rev() {
            let i = i as usize;
            let w_i = w[i];
            if is_exact_zero(w_i) {
                continue;
            }
            let r = self.lt_start[i]..self.lt_start[i + 1];
            for (&p, &l) in self.lt_pivot[r.clone()].iter().zip(&self.lt_value[r]) {
                w[p as usize] -= l * w_i;
            }
        }
        for &i in ws.post.iter().rev() {
            rhs.set(i, w[i as usize]);
            w[i as usize] = T::ZERO;
        }
        ws.list = list;
    }
}
