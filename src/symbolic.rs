//! # Chapter 5.12 — The reusable pivot sequence: [`SymbolicLu`]
//!
//! *New with task L1.3 (PLAN.md trilha L; the contract's `SymbolicLu`,
//! `LuFactor::refactorize_same_pattern` and the pre-order `PreOrder`).*
//!
//! ## Purpose
//! Algorithm L (Chapter 5.3) spends most of its time *deciding*: loading
//! the doubly linked row and column files, keeping the count lists, and
//! searching for Markowitz pivots. When the same pattern comes back with
//! new values — the same basis after a change of scaling or of bounds, a
//! node of branch and bound that restarts from its parent's basis, a
//! sequence of systems with one structure — the decisions of the last
//! factorization are still good ones, and only the arithmetic has to be
//! redone. [`SymbolicLu`] keeps those decisions: the pivot sequence
//! `(p_s, q_s)`, `s = 0, …, m − 1`. [`crate::LuFactor::refactorize_same_pattern`]
//! replays it by a *left-looking* elimination (Algorithm R) that computes
//! one column of `L` and `U` at a time with a sparse triangular solve, the
//! scheme of Gilbert & Peierls (1988) with the pivot rows given instead of
//! searched for.
//!
//! This is weaker than the symbolic object of UMFPACK, which also fixes
//! the patterns of `L` and `U` (a threshold-pivoting Markowitz LU chooses
//! its pivots by value, so its analysis is not separable; the contract,
//! *Divergences*): the patterns are recomputed — cheaply, by the reach of
//! each column — and every pivot is tested again. A pivot refused by the
//! test, or a sequence of another order, sends the call back to Algorithm
//! L, so the result is always a valid factorization (or the diagnosis of a
//! singular basis).
//!
//! The optional COLAMD pre-order of the bump ([`crate::Lu::set_pre_order`],
//! step L5′ of Chapter 5.3) is the other half of the contract's symbolic
//! phase: it fixes the column order of the first factorization from the
//! pattern alone, and the sequence it produces is the one Algorithm R
//! reuses.
//!
//! ## Notation
//! `B` is `m × m` with columns given by a [`ColumnSource`]; the sequence is
//! `p_s` = `pivot_rows[s]`, `q_s` = `pivot_cols[s]`; its first `σ` steps
//! (`singletons`) were singleton pivots of the triangular phase (L3–L4),
//! forced by the structure. At step `s` the rows `p_0, …, p_{s−1}` are
//! *pivoted*, the others *active*. `L_t` is the `L` column of step `t`
//! (multipliers `l_it` for active rows `i` at step `t`), indexed by its
//! pivot row; the graph `G_L` has an edge `p_t → i` for every `l_it`.
//!
//! ## Algorithm R (Refactorization with a given pivot sequence)
//! R1. [Applicable?] If there is no sequence, or its order is not `m`,
//!     fall back to Algorithm L. Validate the source (invariant (B1)).
//! R2. [Scatter.] For `s = 0, 1, …, m − 1`: let `q = q_s`, `p = p_s`;
//!     scatter column `q` of `B` into the dense row-indexed work vector `x`
//!     and its rows into the pattern `X`.
//! R3. [Reach.] Find the set of rows reachable from `X` in `G_L` by depth-
//!     first search; it joins `X`. The pivoted rows with an `L` column, in
//!     reverse postorder of the search, are a topological order of the
//!     reach (Gilbert & Peierls 1988).
//! R4. [Solve.] In that order, for each pivoted row `r` with `x_r ≠ 0`:
//!     `x_i ← x_i − l_ir x_r` for every `l_ir` of `L_r`. Now `x_r` (pivoted
//!     `r`) is the entry `u_{r q}` of `U`, and `x_i` (active `i`) is the
//!     entry `s_iq` of the Schur complement that Algorithm L would have
//!     seen when pivoting column `q`.
//! R5. [Test.] Let `γ = max_{i active} |x_i|`. If `|x_p| ≤ τ_abs`, or `s ≥ σ`
//!     and `|x_p| < u_R γ` (by default `u_R = u`, Markowitz's threshold of
//!     Chapter 5.3; the singleton steps keep the rule of L3–L4: forced
//!     pivots, no relative test), fall back to Algorithm L.
//! R6. [Record.] `u_pq ← x_p`; for each pivoted `r ∈ X` with `|x_r| > ε_drop`
//!     record `u_rq = x_r`; for each active `i ≠ p` with `|x_i| > ε_drop`
//!     record `l_is = x_i / x_p` (the `L` column of step `s`, pivot row
//!     `p`). Clear `x` on `X`; row `p` is pivoted.
//! R7. [Finish.] Step L7 of Chapter 5.3 on the result (the statistics of
//!     the singleton and Markowitz pivots are those of the sequence).
//!
//! ## Invariants
//! (R1) R4 computes the Schur complement column of step `s` exactly as
//!      Algorithm L would (in a different order of the same operations), so
//!      with the same sequence `L U = P B Q` holds to rounding and (F1)–(F3)
//!      of Chapter 5.3 hold (checked by `check_invariants` after step L7 in
//!      debug builds and against the dense product in the tests).
//! (R2) With the default `u_R = u`, the threshold test of R5 is the one
//!      Algorithm L applies to its bump pivots, so a sequence is replayed
//!      only when it would have been acceptable to Algorithm L for these
//!      values. [`crate::Lu::set_refactor_pivot_tol`] may lower `u_R`
//!      (never `u`, which Algorithm L keeps).
//!
//! ## The replay threshold (task L1.5b)
//! A Markowitz search accepts the sparsest pivot that passes `u`, so many
//! recorded pivots sit just above it: in L1.3, 30 of 109 bases (93 Netlib
//! optimal bases and 16 unsymmetric Harwell–Boeing matrices), with every
//! value perturbed by an independent ±1 %, fell back on the first replay.
//! Measured in L1.5b on the same 109, first replay of `B₂` with the
//! sequence of `B₁`:
//!
//! | `u_R` | fallbacks | replays with a residual > 10× the fresh one |
//! |---|---|---|
//! | `u = 10⁻²` (default) | 30 | 0 |
//! | `5·10⁻³` | 19 | 0 |
//! | `10⁻³` | 9 | 0 |
//! | `10⁻⁴` | 2 | 2 (orsirr_1 ×43, degen3 ×290) |
//!
//! (residual `‖B x − b‖_∞ / (‖B‖_∞‖x‖_∞ + ‖b‖_∞)` for `b = B·1`). The
//! default stays `u`, which keeps (R2); `u_R = u/10` is the measured
//! compromise for a caller that replays sequences over drifting values
//! (the simplex method does not replay yet: integration pending, D-0091
//! (a)).
//! (R3) A [`SymbolicLu`] is recorded after every successful factorization
//!      (Algorithm L or R) and dropped after a failed one; its two arrays
//!      are permutations of `0..m` (`debug_assert` in the tests).
//!
//! ## Complexity
//! R2–R6 cost `O(nnz(B) + flops)` — the flops of the elimination itself,
//! with no count lists, no search and no linked files. Step L7 is shared.
//!
//! ## References
//! Gilbert & Peierls, "Sparse partial pivoting in time proportional to
//! arithmetic operations", SIAM J. Sci. Stat. Comput. 9(5), 1988; Davis &
//! Palamadai Natarajan, "Algorithm 907: KLU", ACM TOMS 37(3), 2010
//! (refactorization with the previous pivots); Davis, "Algorithm 832:
//! UMFPACK V4.3", ACM TOMS 30(2), 2004 (symbolic and numeric phases);
//! Suhl & Suhl (1990); the contract of the crate (`SymbolicLu`,
//! *Divergences*).

use crate::basis::ColumnSource;
use crate::error::LuError;
use crate::factor::{Elim, LuOptions, idx32};
use crate::real::{Real, is_exact_zero, is_zero, lt};
use crate::solve::NO_ETA;
use crate::store::NIL;
use crate::update::BasisUpdate;
use crate::{Lu, LuFactor};

/// The reusable part of a factorization: its pivot sequence (Chapter
/// 5.12), consumed by [`crate::LuFactor::refactorize_same_pattern`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolicLu {
    /// `p_s`: the pivot row of step `s`.
    pub pivot_rows: Vec<u32>,
    /// `q_s`: the pivot position (column of `B`) of step `s`.
    pub pivot_cols: Vec<u32>,
    /// The number `σ` of leading steps that were singleton pivots of the
    /// triangular phase (forced; Algorithm R applies no relative test to
    /// them).
    pub singletons: usize,
    /// `nnz(L)` the sequence produced last time.
    pub nnz_l: usize,
    /// `nnz(U)` (diagonal included) the sequence produced last time.
    pub nnz_u: usize,
}

impl SymbolicLu {
    /// The order `m` of the basis the sequence belongs to.
    #[inline]
    pub fn m(&self) -> usize {
        self.pivot_rows.len()
    }
}

impl<T: Real, U: BasisUpdate<T>> Lu<T, U> {
    /// [`LuFactor::refactorize_same_pattern`]: Algorithm R, or Algorithm L
    /// as the fallback.
    pub(crate) fn refactorize_impl<C: ColumnSource<T>>(&mut self, cols: &C) -> Result<(), LuError> {
        // R1.
        let m = cols.m();
        if self.symbolic.as_ref().is_some_and(|s| s.m() == m) {
            self.factors.valid = false;
            self.factors.spike_valid = false;
            cols.validate()?;
            let sym = self.symbolic.take().expect("checked above");
            let prev = self.factors.stats;
            let rel_tol = self.refactor_pivot_tol.unwrap_or(self.opts.rel_pivot_tol);
            let replay = eliminate_fixed(cols, &sym, &self.opts, rel_tol);
            self.symbolic = Some(sym);
            if let Some(mut e) = replay {
                // R7.
                e.col_singletons = prev.col_singletons;
                e.row_singletons = prev.row_singletons;
                e.markowitz_pivots = prev.markowitz_pivots;
                self.factors.m = m;
                let done = self.finish(e);
                debug_assert!(done.is_ok(), "Algorithm R never reports dead columns");
                self.factors.stats.symbolic_reuses += 1;
                return done;
            }
        }
        self.factors.stats.symbolic_fallbacks += 1;
        self.factorize(cols)
    }
}

/// Steps R2–R6; `None` when a pivot is refused (R5) or the sequence is
/// inconsistent.
#[allow(clippy::too_many_lines)] // one step per block, as in the chapter
fn eliminate_fixed<T: Real, C: ColumnSource<T>>(
    cols: &C,
    sym: &SymbolicLu,
    opts: &LuOptions,
    rel_tol: f64,
) -> Option<Elim<T>> {
    let m = cols.m();
    let abs_tol = T::from_f64(opts.abs_pivot_tol);
    let rel_tol = T::from_f64(rel_tol);
    let drop_tol = T::from_f64(opts.drop_tol);
    let mut e = Elim {
        nnz_basis: 0,
        max_basis: T::ZERO,
        col_singletons: 0,
        row_singletons: 0,
        markowitz_pivots: 0,
        step_row: sym.pivot_rows.clone(),
        step_col: sym.pivot_cols.clone(),
        diag: vec![T::ZERO; m],
        l_start: vec![0],
        l_pivot: Vec::new(),
        l_index: Vec::new(),
        l_value: Vec::new(),
        u_trip: Vec::with_capacity(sym.nnz_u.saturating_sub(m)),
        dead: Vec::new(),
    };
    e.l_index.reserve(sym.nnz_l);
    e.l_value.reserve(sym.nnz_l);
    let mut row_step = vec![NIL; m];
    let mut eta_of_row = vec![NO_ETA; m];
    let mut x = vec![T::ZERO; m];
    let mut in_pat = vec![NIL; m];
    let mut done = vec![NIL; m];
    let mut pat: Vec<u32> = Vec::new();
    let mut post: Vec<u32> = Vec::new();
    let mut stack: Vec<(u32, usize)> = Vec::new();
    for s in 0..m {
        let (p, q) = (sym.pivot_rows[s] as usize, sym.pivot_cols[s] as usize);
        if p >= m || q >= m || row_step[p] != NIL {
            return None;
        }
        // R2. Scatter.
        pat.clear();
        cols.for_each_entry(q, |i, v| {
            let iu = i as usize;
            if in_pat[iu] != s {
                in_pat[iu] = s;
                pat.push(i);
            }
            x[iu] += v;
            e.nnz_basis += 1;
            e.max_basis = e.max_basis.max(v.abs());
        });
        // R3. Reach in G_L, by depth-first search from every seed.
        post.clear();
        let eta_start = |r: usize, eta_of_row: &[u32], l_start: &[usize]| match eta_of_row[r] {
            NO_ETA => 0,
            t => l_start[t as usize],
        };
        for t in 0..pat.len() {
            let r0 = pat[t] as usize;
            if done[r0] == s {
                continue;
            }
            done[r0] = s;
            stack.push((idx32(r0), eta_start(r0, &eta_of_row, &e.l_start)));
            while let Some(top) = stack.last_mut() {
                let r = top.0 as usize;
                let eta = eta_of_row[r];
                if eta != NO_ETA && top.1 < e.l_start[eta as usize + 1] {
                    let k = e.l_index[top.1];
                    top.1 += 1;
                    let ku = k as usize;
                    if done[ku] != s {
                        done[ku] = s;
                        if in_pat[ku] != s {
                            in_pat[ku] = s;
                            pat.push(k);
                        }
                        stack.push((k, eta_start(ku, &eta_of_row, &e.l_start)));
                    }
                } else {
                    stack.pop();
                    if eta != NO_ETA {
                        post.push(idx32(r));
                    }
                }
            }
        }
        // R4. Solve with L in topological order.
        for &r in post.iter().rev() {
            let r = r as usize;
            let xr = x[r];
            if is_exact_zero(xr) {
                continue;
            }
            let t = eta_of_row[r] as usize;
            for k in e.l_start[t]..e.l_start[t + 1] {
                let i = e.l_index[k] as usize;
                x[i] -= e.l_value[k] * xr;
            }
        }
        // R5. Threshold test.
        let xp = x[p];
        let gamma = pat
            .iter()
            .filter(|&&i| row_step[i as usize] == NIL)
            .fold(T::ZERO, |g, &i| g.max(x[i as usize].abs()));
        let refused =
            !lt(abs_tol, xp.abs()) || (s >= sym.singletons && lt(xp.abs(), rel_tol * gamma));
        if refused {
            return None;
        }
        // R6. Record U, the diagonal and L; clear x.
        let q32 = idx32(q);
        e.diag[p] = xp;
        let mut any = false;
        for &i in &pat {
            let iu = i as usize;
            let v = x[iu];
            x[iu] = T::ZERO;
            if is_zero(v, drop_tol) || iu == p {
                continue;
            }
            if row_step[iu] == NIL {
                e.l_index.push(i);
                e.l_value.push(v / xp);
                any = true;
            } else {
                e.u_trip.push((i, q32, v));
            }
        }
        if any {
            eta_of_row[p] = idx32(e.l_pivot.len());
            e.l_pivot.push(idx32(p));
            e.l_start.push(e.l_index.len());
        }
        row_step[p] = s;
    }
    Some(e)
}
