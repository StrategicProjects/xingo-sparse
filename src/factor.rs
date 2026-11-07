//! # Chapter 5.3 — Sparse LU factorization of the basis: [`Lu`]
//!
//! ## Purpose
//! Every simplex iteration solves two or three linear systems with the basis
//! matrix `B` (Chapter 5.4). We factorize `B` once into sparse triangular
//! factors and keep them across iterations, modified by the Forrest–Tomlin
//! update (Chapter 5.5) until a fresh factorization is cheaper or safer.
//! Simplex bases are special: most of their columns are logical or nearly
//! triangular, so most pivots can be found *without any arithmetic* by
//! peeling off singletons; only a small *bump* (the kernel) needs genuine
//! Gaussian elimination, done with Markowitz pivoting under a threshold.
//!
//! ## Notation
//! `B ∈ R^{m×m}` has rows `i` and positions `k` (Chapter 5.1). The
//! factorization is a sequence of `m` *steps*; step `s` pivots on the entry
//! `(p_s, q_s)` (row `p_s`, position `q_s`). With the permutation matrices
//! `P` (row `s` of `PB` is row `p_s` of `B`) and `Q` (column `s` of `BQ` is
//! column `q_s` of `B`) we compute
//! ```text
//!   P B Q = L U,
//! ```
//! `L` unit lower triangular and `U` upper triangular. We store both in the
//! *original* indices, never permuting values:
//!
//! * `L` as one column eta per step with a nonzero subdiagonal: the pivot
//!   row `p_s` and the multipliers `l_is` for rows `i` pivoted later;
//! * `U` as its diagonal `u_{p_s q_s}` (indexed by the row `p_s`) and its
//!   off-diagonal entries `u_ij` (row `i`, position `j`, with `step(i) <
//!   step(j)`), held *twice*: by rows and by columns. FTRAN needs `U` by
//!   columns, BTRAN by rows, and the Forrest–Tomlin update must delete a
//!   column *and* a row of `U` — so both copies are dynamic
//!   (Chapter 5.2) and are kept identical.
//!
//! The *active submatrix* `S` during the elimination has row counts `r_i`
//! and column counts `c_j`. A *column singleton* is a column with `c_j = 1`,
//! a *row singleton* a row with `r_i = 1`. The Markowitz merit of a
//! candidate pivot is `(r_i − 1)(c_j − 1)`, an upper bound on the fill it
//! creates. With the threshold `u` ([`LuOptions::rel_pivot_tol`] `∈ (0,
//! 1]`), an entry is *acceptable* iff `|s_ij| ≥ u · max_k |s_kj|` and
//! `|s_ij| > τ_abs` ([`LuOptions::abs_pivot_tol`]). Entries of magnitude
//! at most `ε_drop` ([`LuOptions::drop_tol`]) are negligible.
//!
//! ## Algorithm L (LU factorization of a simplex basis)
//! Given the columns of `B` through a [`crate::ColumnSource`] (in the
//! simplex method: the basis heading and `A`).
//!
//! L1. [Load.] Validate the source (Chapter 5.1, B1). Build `S = B` in
//!     a column file with values and a row file with the pattern only;
//!     record `nnz(B)` and `max |b_ij|`. Bucket columns and rows by count.
//! L2. [Empty columns.] Every active column with `c_j = 0` is *dead*: it
//!     can never be pivoted. Remove it (its position joins the deficiency
//!     list).
//! L3. [Column singleton.] If some column `j` has `c_j = 1`, with entry
//!     `(i, v)`: if `|v| > τ_abs` pivot on `(i, j)` (step L6) — the `L`
//!     column of this step is empty and row `i` is copied to `U` — else
//!     declare `j` dead. Go to L2.
//! L4. [Row singleton.] Otherwise, if some row `i` with `r_i = 1` has its
//!     only entry `(i, j)` above `τ_abs`, pivot on it (step L6) — the `U` row
//!     is empty, the multipliers `s_kj / s_ij` form the `L` column, and no
//!     entry of `S` is updated. Go to L2. Steps L3–L4 constitute the
//!     *triangular phase*: no arithmetic beyond one division per `L` entry.
//!     Singleton pivots are forced (the only entry of their row or column in
//!     any completion of the elimination), so no threshold is applied to
//!     them; they cannot create growth in `U`.
//! L5. [Markowitz.] Otherwise (the *bump*), remove the empty columns as in
//!     L2; if no active column remains, go to L7. Run Algorithm M to obtain
//!     a pivot and a list of dead columns; remove the dead columns, pivot
//!     (step L6) and repeat L5.
//! L5′. [Pre-ordered bump; only with [`Lu::set_pre_order`]`(Some(PreOrder::Colamd))`,
//!     task L1.3.] Instead of L5: compute the
//!     COLAMD order (Chapter 5.10, Algorithm C) of the columns of the bump
//!     on its pattern; for each column `c` in that order that is still
//!     active (after removing the empty columns as in L2): if `γ_c =
//!     max_i |s_ic| ≤ τ_abs` declare `c` dead; otherwise pivot (step L6) on
//!     the acceptable entry `(i, c)` of least row count `r_i` (ties: the
//!     larger `|s_ic|`, then the first in the column) — threshold partial
//!     pivoting inside a fixed column order, as UMFPACK and SuperLU do.
//!     With `pre_order = None` step L5 runs exactly as before, so the
//!     factors are bit for bit those of the second edition.
//! L6. [Pivot on `(p, c)`.] Append the step `(p, c)` and the diagonal
//!     `u_pc = s_pc`. For every other `(i, s_ic)` of column `c` record
//!     `l_ic = s_ic / s_pc`; for every other `(p, s_pj)` of row `p` record
//!     `u_pj = s_pj`. Remove row `p` and column `c` from `S`. Then, for each
//!     `j` of the `U` row, find the positions in column `j` of the rows of
//!     the `L` column (one pass over column `j` against a map of those
//!     rows) and set `s_ij ← s_ij − l_ic u_pj` for every `i` of the `L`
//!     column, in its order, appending fill-in to both files; entries that
//!     cancel to `|s_ij| ≤ ε_drop` are dropped, in the order of column `j`.
//!     Only the entries just updated or filled can cancel — plus, the first
//!     time column `j` is updated, entries of `B` that were negligible from
//!     the start, so such a column is scanned whole once. Update the counts
//!     of the touched rows and columns.
//! L7. [Finish.] If any column is dead, the basis is singular: report its
//!     positions and the rows that never received a pivot
//!     ([`LuError::Singular`]). Otherwise build the row and column
//!     copies of `U`, the inverse step maps, and the statistics, including
//!     the growth factor `max |u_ij| / max |b_ij|` and `max |u_ij|` itself,
//!     the reference of the update policy (Chapter 5.5). Record the pivot
//!     sequence `(p_s, q_s)` as the [`SymbolicLu`] of Chapter 5.12 (into
//!     the arrays of the previous one: `2m` copies, no allocation once
//!     warm). Step L7 reads only the output of the elimination (`Elim`),
//!     so Algorithm R of Chapter 5.12 shares it.
//!
//! ## Algorithm M (Markowitz search with threshold)
//! Let `best` be the acceptable entry of least merit found so far (ties go
//! to the larger ratio `|s_ij| / max_k |s_kj|`), and `e` the number of lines
//! examined that contained an acceptable entry.
//!
//! M1. For `k = 1, 2, …, m`:
//! M2. [Columns of count `k`.] For each active column `j` with `c_j = k`:
//!     compute `γ_j = max_i |s_ij|`; if `γ_j ≤ τ_abs`, mark `j` dead and go
//!     on. Otherwise evaluate every acceptable entry with merit
//!     `(r_i − 1)(k − 1)`. Every unexamined candidate now lies in a row with
//!     `r_i ≥ k` (rows of smaller count were examined whole) and a column
//!     with `c_j ≥ k`, so its merit is at least `(k − 1)²`: stop if `best`
//!     reaches that bound, or if `e ≥ search_limit` and `best` exists.
//! M3. [Rows of count `k`.] The same for each active row `i` with `r_i = k`,
//!     scanning its columns `j` (merit `(k − 1)(c_j − 1)`); the bound for
//!     unexamined candidates is now `k (k − 1)`.
//! M4. Return `best` (possibly none) and the dead columns.
//!
//! The search limit (default 4, [`LuOptions::search_limit`]) is Zlatev's
//! restriction: the merit found after a few lines is almost always the
//! minimum, and the full search would cost `O(nnz(S))` per pivot. If M
//! returns no pivot and no dead column, every active column was examined
//! and found empty, so L5 terminates.
//!
//! ## Invariants
//! Checked by [`LuFactor::check_invariants`] in the tests and after every
//! factorization in debug builds.
//! (F1) The non-vacant slots of `(p_s)` and `(q_s)` are permutations of
//!      `0..m`, and the inverse maps `step(p_s) = step(q_s) = s` agree with
//!      them (right after a factorization there are no vacant slots; each
//!      update adds one, Chapter 5.5).
//! (F2) Every `L` multiplier `l_is` of step `s` has `step(i) > s`, where
//!      `step` is the order of the last factorization (updates reorder `U`
//!      but leave `L` alone).
//! (F3) Every off-diagonal `u_ij` has `step(i) < step(j)`; the row and
//!      column copies of `U` hold exactly the same entries; every diagonal
//!      entry is nonzero.
//! (F4) Right after a factorization, `L U = P B Q` up to rounding; tested
//!      by property tests against the dense product.
//!
//! ### The two files of `S` (task P3.37)
//! The column file and the row file of `S` are *cross-linked* (Chapter 5.2,
//! Algorithm T): every entry holds a hint of the position of its twin in
//! the other file, exact in long lines. Every deletion of L2, L6 and of the
//! cancellations of L6, and every lookup of `s_ij` from the row side (L4,
//! M3), is then `O(1)` in a long line instead of a linear search along it
//! (short lines are still searched, which is cheaper than keeping their
//! hints exact). The deletions keep the lines in the same
//! order as the searches did, so the elimination — pivots, fill, the order
//! of `L` and `U` — is the same, bit for bit. Invariant (S3) of Chapter 5.2
//! is checked after the load and after the triangular phase in debug
//! builds.
//!
//! ## The objects (second edition)
//! [`Lu<T, U>`](Lu) is the factorization object of the contract: the
//! options [`LuOptions`], the stored factors [`Factors<T>`] (the steps,
//! `L`, `U` by rows and by columns, the spike, the work arrays and the
//! statistics [`LuStats`]), the basis-update policy `U` (Chapter 5.5,
//! [`crate::ForrestTomlin`] by default, which owns the row etas) and the
//! moving averages of the density policy (Chapter 5.4). The simplex method
//! programs against the trait [`LuFactor`]. In the first edition all of
//! this was one struct, `Factor`; splitting it moves no arithmetic: the
//! update reaches the factors through `&mut Factors<T>`, and the type
//! parameter `U` is resolved at compile time.
//!
//! `LuOptions` merges the first edition's `FactorOptions` with the two
//! tolerances the factorization borrowed from the solver's `Tolerances`:
//! `markowitz` became [`LuOptions::rel_pivot_tol`] and `zero` became
//! [`LuOptions::drop_tol`] (the solver copies them in when it creates a
//! factor).
//!
//! ## Complexity
//! The triangular phase costs `O(nnz(B))` plus one division per `L` entry.
//! A Markowitz pivot costs `O(Σ_j c_j)` over the columns of its `U` row,
//! plus the bounded search. (Before task P3.37 every deletion also searched
//! its line, `O(Σ_i r_i²)` per factorization: on `hypothyroid-k1` half the
//! time of a factorization.)
//!
//! ## References
//! Markowitz, "The elimination form of the inverse and its application to
//! linear programming", Management Sci. 3, 1957; Suhl & Suhl, "Computing
//! sparse LU factorizations for large-scale linear programming bases", ORSA
//! J. Comput. 2(4), 1990 (triangular phase, count lists, search);
//! Zlatev, "On some pivotal strategies in Gaussian elimination by sparse
//! technique", SIAM J. Numer. Anal. 17, 1980 (restricted search);
//! Maros (2003), ch. 5 (§5.4 LU of the basis) and ch. 8 (§8.2 bump);
//! Davis, *Direct Methods for Sparse Linear Systems* (2006), ch. 6; Duff,
//! Erisman & Reid (1986), ch. 7 and §9.4.

use crate::basis::ColumnSource;
use crate::error::LuError;
use crate::real::{Real, approx_eq, is_exact_zero, is_zero, lt};
use crate::solve::{Kind, NO_ETA, SolveStats, Workspace};
use crate::store::{CountLists, LineStore, NIL, check_links, remove_linked};
use crate::symbolic::SymbolicLu;
use crate::update::{BasisUpdate, ForrestTomlin, UpdateOutcome, UpdateStats};
use crate::{Colamd, Ordering, OrderingTarget, Pattern, Row, SparseError, SparseVec};

/// Options of the factorization, of the solves and of the update policy.
///
/// The names follow the contract (D-0080); the defaults are those of the
/// first edition (`FactorOptions` and the two fields of the solver's
/// `Tolerances` it read), so a default [`Lu`] behaves exactly as the
/// first edition's default `Factor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LuOptions {
    /// Absolute pivot tolerance `τ_abs`: no pivot of magnitude at or below
    /// it is accepted, in the factorization or in an update (`1e-10`).
    pub abs_pivot_tol: f64,
    /// Relative threshold `u ∈ (0, 1]` of Markowitz threshold pivoting
    /// (`1e-2`; the solver's `Tolerances::markowitz`).
    pub rel_pivot_tol: f64,
    /// Number of lines with an acceptable entry after which the Markowitz
    /// search stops (`4`, Zlatev's restriction; at least 1).
    pub search_limit: usize,
    /// Growth limit (`1e10`): the factor is flagged
    /// ([`LuStats::growth_exceeded`]) when `max |u_ij| / max |b_ij|`
    /// exceeds it, and an update asks for a refactorization when the
    /// growth *added by the updates*, [`UpdateStats::update_growth`],
    /// exceeds it (Chapter 5.5, refactorization policy).
    pub growth_limit: f64,
    /// Fill trigger: an update asks for a refactorization when
    /// `nnz(L) + nnz(U) + nnz(etas)` exceeds this multiple of
    /// `nnz(L) + nnz(U)` right after the factorization (`3.0`).
    pub fill_factor: f64,
    /// Number of updates after which a refactorization is requested
    /// (`100`).
    pub refactor_interval: usize,
    /// Relative tolerance of the Forrest–Tomlin stability test: the new
    /// diagonal must agree with `α_r · u_old` within this relative
    /// distance (`1e-8`).
    pub update_stability_tol: f64,
    /// Largest right-hand side density (`count/m`) for which a solve may
    /// use the hypersparse algorithms (Chapter 5.4, policy (b); `0.10`).
    /// A negative value disables them.
    pub hyper_rhs_density: f64,
    /// Largest expected result density for which a solve may use the
    /// hypersparse algorithms (Chapter 5.4, policy (c); `0.10`).
    pub hyper_result_density: f64,
    /// Drop tolerance `ε_drop`: magnitudes at or below it are treated as
    /// zero by the elimination and the update (`1e-12`; the solver's
    /// `Tolerances::zero`).
    pub drop_tol: f64,
}

/// Column pre-orders available to the dynamic LU ([`Lu::set_pre_order`];
/// the contract's `PreOrder`, which it placed in `LuOptions::pre_order`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreOrder {
    /// [`crate::Colamd`] with its defaults on the columns of the bump.
    Colamd,
}

impl Default for LuOptions {
    fn default() -> Self {
        Self {
            abs_pivot_tol: 1e-10,
            rel_pivot_tol: 1e-2,
            search_limit: 4,
            growth_limit: 1e10,
            fill_factor: 3.0,
            refactor_interval: 100,
            update_stability_tol: 1e-8,
            hyper_rhs_density: 0.10,
            hyper_result_density: 0.10,
            drop_tol: 1e-12,
        }
    }
}

impl LuOptions {
    /// `search_limit ≥ 1`, `rel_pivot_tol ∈ (0, 1]`, and the tolerances
    /// finite and non-negative (asserted in debug builds when the options
    /// are installed).
    fn is_valid(&self) -> bool {
        let finite_nonneg = |x: f64| x.is_finite() && !lt(x, 0.0);
        self.search_limit >= 1
            && lt(0.0, self.rel_pivot_tol)
            && !lt(1.0, self.rel_pivot_tol)
            && finite_nonneg(self.abs_pivot_tol)
            && finite_nonneg(self.drop_tol)
            && finite_nonneg(self.update_stability_tol)
    }
}

/// Statistics of the current factor, for logs and the phase-1 benchmark.
/// The update counters are in [`UpdateStats`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LuStats {
    /// Dimension `m` of the basis.
    pub m: usize,
    /// Nonzeros of `B` at the last factorization.
    pub nnz_basis: usize,
    /// Stored multipliers of `L` (its unit diagonal is implicit).
    pub nnz_l: usize,
    /// Entries of `U`, including the `m` diagonal entries.
    pub nnz_u: usize,
    /// Fill-in `nnz_l + nnz_u + nnz_eta − nnz_basis` (saturating at 0;
    /// `nnz_eta` from [`UpdateStats`]).
    pub fill_in: usize,
    /// Pivots taken as column singletons in the triangular phase (L3).
    pub col_singletons: usize,
    /// Pivots taken as row singletons in the triangular phase (L4).
    pub row_singletons: usize,
    /// Pivots chosen by the Markowitz search (L5): the size of the bump.
    pub markowitz_pivots: usize,
    /// Growth factor `max_u / max_basis`.
    pub growth: f64,
    /// Successful factorizations performed by this object.
    pub factorizations: usize,
    /// FTRAN calls (Chapter 5.4), accumulated over the life of the object.
    pub ftran: SolveStats,
    /// BTRAN calls, accumulated over the life of the object.
    pub btran: SolveStats,
    /// FTRAN-for-update calls, accumulated over the life of the object.
    pub ftran_update: SolveStats,
    /// `max |b_ij|` at the last factorization.
    pub max_basis: f64,
    /// Largest `|u_ij|` ever stored since the last factorization.
    pub max_u: f64,
    /// `max |u_ij|` right after the last factorization (before any
    /// update).
    pub max_u_fresh: f64,
    /// `growth > LuOptions::growth_limit`; a diagnostic of the
    /// factorization itself (a fresh factor may already be flagged).
    pub growth_exceeded: bool,
    /// Refactorizations that reused the pivot sequence of the previous
    /// factorization ([`LuFactor::refactorize_same_pattern`], Algorithm R
    /// of Chapter 5.12), accumulated over the life of the object.
    pub symbolic_reuses: usize,
    /// Calls of [`LuFactor::refactorize_same_pattern`] that fell back to a
    /// full factorization (no sequence, another order `m`, or a pivot
    /// refused by the threshold test), accumulated over the life of the
    /// object.
    pub symbolic_fallbacks: usize,
}

/// The stored factors `P, L, U, Q` of a basis (row and column files,
/// MA28/Suhl style), the recorded spike, the work arrays of the solves
/// and the statistics. Opaque: the basis-update policies reach it through
/// [`BasisUpdate`].
#[derive(Clone, Debug)]
pub struct Factors<T: Real> {
    pub(crate) m: usize,
    pub(crate) valid: bool,
    /// `p_s`: pivot row of step `s`.
    pub(crate) step_row: Vec<u32>,
    /// `q_s`: pivot position of step `s`.
    pub(crate) step_col: Vec<u32>,
    /// `step(i)` for rows.
    pub(crate) row_step: Vec<usize>,
    /// `step(j)` for positions.
    pub(crate) col_step: Vec<usize>,
    /// `step(i)` for rows at the last factorization: the order of `L`,
    /// which the updates do not change.
    pub(crate) l_row_step: Vec<usize>,
    /// `L` etas: eta `e` has pivot row `l_pivot[e]` and multipliers
    /// `l_index/l_value[l_start[e]..l_start[e+1]]`.
    pub(crate) l_start: Vec<usize>,
    pub(crate) l_pivot: Vec<u32>,
    pub(crate) l_index: Vec<u32>,
    pub(crate) l_value: Vec<T>,
    /// Index of the `L` eta whose pivot row is `p`, or `NO_ETA` (the
    /// successors of `p` in the graph `G_L`, Chapter 5.4).
    pub(crate) l_eta_of_row: Vec<u32>,
    /// `L` by rows (the graph `G_Lᵀ`): row `i` holds the pivot rows `p`
    /// and multipliers `l_ip` in `lt_pivot/lt_value[lt_start[i]..lt_start[i+1]]`.
    pub(crate) lt_start: Vec<usize>,
    pub(crate) lt_pivot: Vec<u32>,
    pub(crate) lt_value: Vec<T>,
    /// Diagonal of `U`, indexed by the pivot row.
    pub(crate) diag: Vec<T>,
    /// Off-diagonal `U` by rows: line `i` holds `(j, u_ij)`.
    pub(crate) u_rows: LineStore<T>,
    /// Off-diagonal `U` by columns: line `j` holds `(i, u_ij)`.
    pub(crate) u_cols: LineStore<T>,
    /// Spike `R L⁻¹ a_q` recorded by `ftran_for_update` (indexed by rows,
    /// dense, zero outside `spike_idx`).
    pub(crate) spike: Vec<T>,
    /// Rows where `spike` may be nonzero.
    pub(crate) spike_idx: Vec<u32>,
    pub(crate) spike_valid: bool,
    /// Work arrays of the solves and the update (Chapter 5.4).
    pub(crate) ws: Workspace<T>,
    /// `nnz(L) + nnz(U)` right after the factorization (fill trigger).
    pub(crate) nnz_factor0: usize,
    pub(crate) stats: LuStats,
}

impl<T: Real> Default for Factors<T> {
    fn default() -> Self {
        Self {
            m: 0,
            valid: false,
            step_row: Vec::new(),
            step_col: Vec::new(),
            row_step: Vec::new(),
            col_step: Vec::new(),
            l_row_step: Vec::new(),
            l_start: vec![0],
            l_pivot: Vec::new(),
            l_index: Vec::new(),
            l_value: Vec::new(),
            l_eta_of_row: Vec::new(),
            lt_start: vec![0],
            lt_pivot: Vec::new(),
            lt_value: Vec::new(),
            diag: Vec::new(),
            u_rows: LineStore::default(),
            u_cols: LineStore::default(),
            spike: Vec::new(),
            spike_idx: Vec::new(),
            spike_valid: false,
            ws: Workspace::default(),
            nnz_factor0: 0,
            stats: LuStats::default(),
        }
    }
}

/// Sparse LU factors of a basis with a basis-update policy `U`
/// (Forrest–Tomlin by default): the contract's `Lu<T, U>`.
///
/// Usage contract (the simplex method of Chapter 6), through
/// [`LuFactor`]:
/// 1. [`LuFactor::factorize`] with a [`ColumnSource`] (the basis heading
///    and `A`);
/// 2. [`LuFactor::ftran`] / [`LuFactor::btran`] any number of times;
/// 3. to exchange the column at position `r`: [`LuFactor::ftran_for_update`]
///    on the entering column `a_q` (this yields `α_q = B⁻¹ a_q` *and*
///    records the spike), then [`LuFactor::update`]`(r, α_q[r])`;
/// 4. when the update returns [`UpdateOutcome::RefactorNeeded`] (or an
///    error), call [`LuFactor::factorize`] with the new heading — the
///    factor never refactorizes by itself, since only the caller holds the
///    columns (D-0016).
#[derive(Clone, Debug)]
pub struct Lu<T: Real = f64, U: BasisUpdate<T> = ForrestTomlin<T>> {
    pub(crate) opts: LuOptions,
    pub(crate) factors: Factors<T>,
    pub(crate) update: U,
    /// Moving averages of the result density of FTRAN, BTRAN and FTRAN for
    /// update (Chapter 5.4, policy (c)).
    pub(crate) expected_density: [f64; 3],
    /// The pivot sequence of the last successful factorization (Chapter
    /// 5.12); `None` before it and after a failed one.
    pub(crate) symbolic: Option<SymbolicLu>,
    /// Column pre-order of the bump (step L5′); `None`: Algorithm M.
    pub(crate) pre_order: Option<PreOrder>,
}

impl<T: Real, U: BasisUpdate<T>> Default for Lu<T, U> {
    fn default() -> Self {
        Self::new(LuOptions::default())
    }
}

/// What every LU of the crate exposes; the simplex method programs
/// against this trait (the contract's `LuFactor<T>`).
pub trait LuFactor<T: Real> {
    /// Algorithm L: factorizes the `m × m` basis `B` whose columns the
    /// source gives.
    ///
    /// On success the factor represents `B` and all previous updates are
    /// discarded. On [`LuError::Singular`] the factor is invalid until the
    /// next successful call; the error names the dependent positions and
    /// the rows without pivot, so the caller can substitute logicals and
    /// retry. On [`LuError::Structure`] (invariant (B1) violated) nothing
    /// was factorized.
    fn factorize<C: ColumnSource<T>>(&mut self, cols: &C) -> Result<(), LuError>;
    /// Algorithm R (Chapter 5.12): factorizes the `m × m` basis `B` with
    /// the pivot sequence of the last successful factorization — no
    /// search, no count lists — as long as every pivot passes the
    /// threshold test; otherwise (or without a sequence of order `m`) it
    /// falls back to [`LuFactor::factorize`]. Meant for new values on the
    /// same pattern; the outcome and the errors are those of `factorize`,
    /// and [`LuStats::symbolic_reuses`] / [`LuStats::symbolic_fallbacks`]
    /// say which path ran.
    fn refactorize_same_pattern<C: ColumnSource<T>>(&mut self, cols: &C) -> Result<(), LuError>;
    /// The pivot sequence of the last successful factorization, if any.
    fn symbolic(&self) -> Option<&SymbolicLu>;
    /// `v ← B⁻¹ v`: row-indexed in, position-indexed out (D-0016).
    fn ftran(&mut self, v: &mut SparseVec<T>);
    /// `v ← B⁻ᵀ v`: position-indexed in, row-indexed out.
    fn btran(&mut self, v: &mut SparseVec<T>);
    /// [`LuFactor::ftran`] that also records the spike for the next
    /// [`LuFactor::update`].
    fn ftran_for_update(&mut self, v: &mut SparseVec<T>);
    /// Replaces the column at position `r`; `alpha_r` is entry `r` of the
    /// `α_q` returned by the last `ftran_for_update`. The factor never
    /// refactorizes on its own.
    fn update(&mut self, r: usize, alpha_r: T) -> Result<UpdateOutcome, LuError>;
    /// Dimension `m` of the last basis given to `factorize`.
    fn m(&self) -> usize;
    /// `true` iff the factor holds a valid factorization (the last
    /// `factorize` succeeded).
    fn is_valid(&self) -> bool;
    /// Statistics of the current factor.
    fn stats(&self) -> &LuStats;
    /// Statistics of the update policy since the last factorization.
    fn update_stats(&self) -> &UpdateStats;
    /// The options in force.
    fn options(&self) -> &LuOptions;
    /// Replaces the options. The factorization reads them at the next
    /// `factorize`, the solves and the update at their next call (e.g. a
    /// tighter [`LuOptions::rel_pivot_tol`] after numerical trouble,
    /// `PLAN.md` §8).
    fn set_options(&mut self, opts: LuOptions);
    /// Checks invariants (F1)–(F3) of a valid factor (and of its updates).
    fn check_invariants(&self) -> Result<(), SparseError>;
}

/// Converts a count to `f64` for ratios.
#[inline]
#[allow(clippy::cast_precision_loss)] // counts stay far below 2^52
pub(crate) fn count_f64(n: usize) -> f64 {
    n as f64
}

/// A vacant slot of the step sequence: the Forrest–Tomlin update moves a
/// step to the end by appending it and leaving its old slot vacant
/// (Chapter 5.5, step U5), so that no other step is renumbered.
pub(crate) const VACANT: u32 = u32::MAX;

/// Converts an index below `m ≤ u32::MAX` to `u32`.
#[inline]
pub(crate) fn idx32(i: usize) -> u32 {
    u32::try_from(i).expect("index exceeds u32::MAX")
}

impl<T: Real, U: BasisUpdate<T>> Lu<T, U> {
    /// An empty factor (no valid factorization yet).
    pub fn new(opts: LuOptions) -> Self {
        debug_assert!(opts.is_valid(), "{opts:?}");
        Self {
            opts,
            factors: Factors::default(),
            update: U::default(),
            expected_density: [0.0; 3],
            symbolic: None,
            pre_order: None,
        }
    }

    /// The column pre-order of the bump (step L5′; Chapter 5.12): `None`
    /// (the default) runs Algorithm M exactly as before, bit for bit;
    /// `Some(PreOrder::Colamd)` eliminates the bump in the COLAMD order of
    /// its columns, choosing the row by threshold partial pivoting. Read at
    /// the next [`LuFactor::factorize`].
    ///
    /// (The contract has it as a field `LuOptions::pre_order`; a field
    /// would make `LuOptions`, and the simplex's `SimplexOptions` that
    /// holds it, 8 bytes larger, past the 256-byte limit of clippy's
    /// `large_types_passed_by_value` in the simplex and the MIP solver.)
    pub fn set_pre_order(&mut self, pre_order: Option<PreOrder>) {
        self.pre_order = pre_order;
    }

    /// The column pre-order in force ([`Lu::set_pre_order`]).
    #[inline]
    pub fn pre_order(&self) -> Option<PreOrder> {
        self.pre_order
    }

    /// `true` iff a spike is recorded and [`LuFactor::update`] may be
    /// called.
    #[inline]
    pub fn has_spike(&self) -> bool {
        self.factors.valid && self.factors.spike_valid
    }

    /// The basis-update policy (its statistics and name).
    #[inline]
    pub fn update_policy(&self) -> &U {
        &self.update
    }

    /// Step L7: moves the factors out of the elimination (Algorithm L or
    /// Algorithm R of Chapter 5.12), or reports singularity; records the
    /// pivot sequence.
    pub(crate) fn finish(&mut self, k: Elim<T>) -> Result<(), LuError> {
        let f = &mut self.factors;
        let m = f.m;
        let mut stats = LuStats {
            m,
            nnz_basis: k.nnz_basis,
            max_basis: k.max_basis.to_f64(),
            col_singletons: k.col_singletons,
            row_singletons: k.row_singletons,
            markowitz_pivots: k.markowitz_pivots,
            factorizations: f.stats.factorizations,
            ftran: f.stats.ftran,
            btran: f.stats.btran,
            ftran_update: f.stats.ftran_update,
            symbolic_reuses: f.stats.symbolic_reuses,
            symbolic_fallbacks: f.stats.symbolic_fallbacks,
            ..LuStats::default()
        };
        if !k.dead.is_empty() {
            self.symbolic = None;
            let mut cols = k.dead;
            cols.sort_unstable();
            let mut pivoted = vec![false; m];
            for &p in &k.step_row {
                pivoted[p as usize] = true;
            }
            let rows: Vec<Row> = (0..m)
                .filter(|&i| !pivoted[i])
                .map(Row::from_usize)
                .collect();
            debug_assert_eq!(rows.len(), cols.len());
            f.stats = stats;
            self.update.reset(m, 0.0);
            return Err(LuError::Singular {
                rank_deficiency: cols.len(),
                cols,
                rows,
            });
        }

        // Inverse step maps.
        f.row_step = vec![NIL; m];
        f.col_step = vec![NIL; m];
        for (s, (&p, &q)) in k.step_row.iter().zip(&k.step_col).enumerate() {
            f.row_step[p as usize] = s;
            f.col_step[q as usize] = s;
        }
        f.l_row_step.clone_from(&f.row_step);
        f.step_row = k.step_row;
        f.step_col = k.step_col;
        f.l_start = k.l_start;
        f.l_pivot = k.l_pivot;
        f.l_index = k.l_index;
        f.l_value = k.l_value;
        f.diag = k.diag;

        // U by rows and by columns, with a little room for the updates.
        let mut row_len = vec![0_usize; m];
        let mut col_len = vec![0_usize; m];
        for &(i, j, _) in &k.u_trip {
            row_len[i as usize] += 1;
            col_len[j as usize] += 1;
        }
        let slack = |n: &usize| n + 2;
        let row_caps: Vec<usize> = row_len.iter().map(slack).collect();
        let col_caps: Vec<usize> = col_len.iter().map(slack).collect();
        f.u_rows = LineStore::new(&row_caps);
        f.u_cols = LineStore::new(&col_caps);
        let mut max_u = T::ZERO;
        for &(i, j, v) in &k.u_trip {
            f.u_rows.push(i as usize, j, v);
            f.u_cols.push(j as usize, i, v);
            max_u = max_u.max(v.abs());
        }
        for &d in &f.diag {
            max_u = max_u.max(d.abs());
        }

        f.prepare_hypersparse();

        stats.nnz_l = f.l_value.len();
        stats.nnz_u = f.u_rows.live() + m;
        stats.max_u = max_u.to_f64();
        stats.max_u_fresh = stats.max_u;
        stats.factorizations += 1;
        f.stats = stats;
        f.nnz_factor0 = stats.nnz_l + stats.nnz_u;
        let update_growth = f.refresh_stats(self.opts.growth_limit, 0);
        self.update.reset(m, update_growth);
        f.valid = true;
        debug_assert!(f.check_invariants().is_ok(), "{:?}", f.check_invariants());
        // The reusable pivot sequence (Chapter 5.12), into the buffers of
        // the previous one.
        let sym = self.symbolic.get_or_insert_with(SymbolicLu::default);
        sym.pivot_rows.clone_from(&f.step_row);
        sym.pivot_cols.clone_from(&f.step_col);
        sym.singletons = stats.col_singletons + stats.row_singletons;
        sym.nnz_l = stats.nnz_l;
        sym.nnz_u = stats.nnz_u;
        Ok(())
    }
}

impl<T: Real, U: BasisUpdate<T>> LuFactor<T> for Lu<T, U> {
    fn factorize<C: ColumnSource<T>>(&mut self, cols: &C) -> Result<(), LuError> {
        self.factors.valid = false;
        self.factors.spike_valid = false;
        cols.validate()?;
        self.factors.m = cols.m();

        // L1. Load.
        let mut k = Kernel::load(cols, self.opts);
        // L2–L4. Triangular phase.
        k.triangular_phase();
        match self.pre_order {
            // L5. Markowitz phase on the bump.
            None => k.markowitz_phase(),
            // L5′. The bump in its COLAMD column order.
            Some(PreOrder::Colamd) => k.preordered_phase(),
        }
        // L7. Finish.
        self.finish(k.into_elim())
    }

    fn refactorize_same_pattern<C: ColumnSource<T>>(&mut self, cols: &C) -> Result<(), LuError> {
        self.refactorize_impl(cols)
    }

    #[inline]
    fn symbolic(&self) -> Option<&SymbolicLu> {
        self.symbolic.as_ref()
    }

    #[inline]
    fn ftran(&mut self, v: &mut SparseVec<T>) {
        self.ftran_kind(v, Kind::Ftran);
    }

    #[inline]
    fn btran(&mut self, v: &mut SparseVec<T>) {
        self.btran_impl(v);
    }

    #[inline]
    fn ftran_for_update(&mut self, v: &mut SparseVec<T>) {
        self.ftran_kind(v, Kind::FtranUpdate);
    }

    #[inline]
    fn update(&mut self, r: usize, alpha_r: T) -> Result<UpdateOutcome, LuError> {
        self.update
            .update(&mut self.factors, &self.opts, r, alpha_r)
    }

    #[inline]
    fn m(&self) -> usize {
        self.factors.m
    }

    #[inline]
    fn is_valid(&self) -> bool {
        self.factors.valid
    }

    #[inline]
    fn stats(&self) -> &LuStats {
        &self.factors.stats
    }

    #[inline]
    fn update_stats(&self) -> &UpdateStats {
        self.update.stats()
    }

    #[inline]
    fn options(&self) -> &LuOptions {
        &self.opts
    }

    fn set_options(&mut self, opts: LuOptions) {
        debug_assert!(opts.is_valid(), "{opts:?}");
        self.opts = opts;
    }

    fn check_invariants(&self) -> Result<(), SparseError> {
        self.factors.check_invariants()
    }
}

impl<T: Real> Factors<T> {
    /// Step L7, continued: the structures of the hypersparse solves
    /// (Chapter 5.4) — the map from rows to `L` etas, `L` by rows, the work
    /// arrays and an empty spike. (The row etas are emptied by the update
    /// policy, [`BasisUpdate::reset`].)
    fn prepare_hypersparse(&mut self) {
        let m = self.m;
        self.l_eta_of_row.clear();
        self.l_eta_of_row.resize(m, NO_ETA);
        let mut row_len = vec![0_usize; m + 1];
        for (e, &p) in self.l_pivot.iter().enumerate() {
            self.l_eta_of_row[p as usize] = idx32(e);
            for &i in &self.l_index[self.l_start[e]..self.l_start[e + 1]] {
                row_len[i as usize + 1] += 1;
            }
        }
        for i in 0..m {
            row_len[i + 1] += row_len[i];
        }
        let nnz = self.l_index.len();
        self.lt_pivot.clear();
        self.lt_pivot.resize(nnz, 0);
        self.lt_value.clear();
        self.lt_value.resize(nnz, T::ZERO);
        let mut next = row_len.clone();
        for (e, &p) in self.l_pivot.iter().enumerate() {
            let r = self.l_start[e]..self.l_start[e + 1];
            for (&i, &l) in self.l_index[r.clone()].iter().zip(&self.l_value[r]) {
                let at = next[i as usize];
                self.lt_pivot[at] = p;
                self.lt_value[at] = l;
                next[i as usize] += 1;
            }
        }
        self.lt_start = row_len;
        self.ws.resize(m);
        self.ws.queued.clear();
        if self.spike.len() == m {
            for &i in &self.spike_idx {
                self.spike[i as usize] = T::ZERO;
            }
        } else {
            self.spike = vec![T::ZERO; m];
        }
        self.spike_idx.clear();
    }

    /// Recomputes the derived statistics (fill-in, growth) with `nnz_eta`
    /// row-eta multipliers, and returns the growth added by the updates,
    /// `max_u / max(max_basis, max_u_fresh)` (`1` without a reference).
    pub(crate) fn refresh_stats(&mut self, growth_limit: f64, nnz_eta: usize) -> f64 {
        let s = &mut self.stats;
        s.nnz_u = self.u_rows.live() + self.m;
        s.fill_in = (s.nnz_l + s.nnz_u + nnz_eta).saturating_sub(s.nnz_basis);
        s.growth = if lt(0.0, s.max_basis) {
            s.max_u / s.max_basis
        } else {
            1.0
        };
        s.growth_exceeded = lt(growth_limit, s.growth);
        let reference = s.max_basis.max(s.max_u_fresh);
        if lt(0.0, reference) {
            s.max_u / reference
        } else {
            1.0
        }
    }

    /// Checks invariants (F1)–(F3) of a valid factor (and of its updates).
    pub(crate) fn check_invariants(&self) -> Result<(), SparseError> {
        let bad = |msg: String| Err(SparseError::Malformed(msg));
        if !self.valid {
            return bad("factor is not valid".into());
        }
        let m = self.m;
        // (F1)
        if self.step_row.len() != self.step_col.len() {
            return bad("step arrays differ in length".into());
        }
        let mut steps = 0;
        for s in 0..self.step_row.len() {
            let (p, q) = (self.step_row[s], self.step_col[s]);
            if p == VACANT && q == VACANT {
                continue;
            }
            steps += 1;
            let (p, q) = (p as usize, q as usize);
            if p >= m || q >= m || self.row_step[p] != s || self.col_step[q] != s {
                return bad(format!("step {s} inconsistent with the inverse maps"));
            }
        }
        if steps != m {
            return bad(format!("{steps} steps for m = {m}"));
        }
        // (F2), in the step order of the last factorization.
        for e in 0..self.l_pivot.len() {
            let sp = self.l_row_step[self.l_pivot[e] as usize];
            for &i in &self.l_index[self.l_start[e]..self.l_start[e + 1]] {
                if self.l_row_step[i as usize] <= sp {
                    return bad(format!("L eta {e} touches row {i} pivoted earlier"));
                }
            }
        }
        // (F3)
        let mut n_row = 0;
        for i in 0..m {
            if is_exact_zero(self.diag[i]) {
                return bad(format!("zero diagonal at row {i}"));
            }
            let (idx, val) = self.u_rows.line(i);
            n_row += idx.len();
            for (&j, &v) in idx.iter().zip(val) {
                if self.col_step[j as usize] <= self.row_step[i] {
                    return bad(format!("U entry ({i}, {j}) below the diagonal"));
                }
                match self.u_cols.get(j as usize, idx32(i)) {
                    Some(w) if approx_eq(v, w, T::ZERO) => {}
                    _ => return bad(format!("U copies disagree at ({i}, {j})")),
                }
            }
        }
        if n_row != self.u_cols.live() {
            return bad("U row and column copies differ in size".into());
        }
        Ok(())
    }
}

/// The active submatrix and the factors under construction (Algorithm L).
struct Kernel<T: Real> {
    m: usize,
    /// `τ_abs`, `u` and `ε_drop` in the scalar type.
    abs_tol: T,
    rel_tol: T,
    drop_tol: T,
    search_limit: usize,
    /// Column file: line `j` holds `(i, s_ij)`, linked to `rows`
    /// (Chapter 5.2, Algorithm T).
    cols: LineStore<T>,
    /// Row file (pattern only): line `i` holds `(j, ·)`, linked to `cols`.
    rows: LineStore<T>,
    col_lists: CountLists,
    row_lists: CountLists,
    /// Row → relative position in the column being updated, for the
    /// cancelled entries still to remove (`NIL` elsewhere).
    wmap: Vec<usize>,
    /// Row → index in the `L` column of the current pivot (`NIL` elsewhere).
    lmap: Vec<usize>,
    /// Column `j` may still hold a negligible entry of `B` itself
    /// (`|b_ij| ≤ ε_drop`), so its next update scans it whole.
    tiny: Vec<bool>,
    /// Work arrays of step L6 (kept to avoid an allocation per pivot).
    tw: Vec<usize>,
    cancelled: Vec<(usize, u32)>,
    nnz_basis: usize,
    max_basis: T,
    step_row: Vec<u32>,
    step_col: Vec<u32>,
    diag: Vec<T>,
    l_start: Vec<usize>,
    l_pivot: Vec<u32>,
    l_index: Vec<u32>,
    l_value: Vec<T>,
    /// Off-diagonal `U` entries `(i, j, u_ij)` in pivot order.
    u_trip: Vec<(u32, u32, T)>,
    dead: Vec<usize>,
    col_singletons: usize,
    row_singletons: usize,
    markowitz_pivots: usize,
}

/// The output of an elimination — Algorithm L's kernel or Algorithm R of
/// Chapter 5.12 — that step L7 turns into [`Factors`].
pub(crate) struct Elim<T: Real> {
    pub(crate) nnz_basis: usize,
    pub(crate) max_basis: T,
    pub(crate) col_singletons: usize,
    pub(crate) row_singletons: usize,
    pub(crate) markowitz_pivots: usize,
    pub(crate) step_row: Vec<u32>,
    pub(crate) step_col: Vec<u32>,
    pub(crate) diag: Vec<T>,
    pub(crate) l_start: Vec<usize>,
    pub(crate) l_pivot: Vec<u32>,
    pub(crate) l_index: Vec<u32>,
    pub(crate) l_value: Vec<T>,
    /// Off-diagonal `U` entries `(i, j, u_ij)`.
    pub(crate) u_trip: Vec<(u32, u32, T)>,
    pub(crate) dead: Vec<usize>,
}

/// A candidate pivot of Algorithm M.
#[derive(Clone, Copy)]
struct Candidate<T: Real> {
    i: usize,
    j: usize,
    merit: usize,
    ratio: T,
}

impl<T: Real> Candidate<T> {
    /// Smaller merit wins; ties go to the larger ratio `|s_ij| / γ_j`.
    fn beats(&self, other: Option<&Candidate<T>>) -> bool {
        other.is_none_or(|o| {
            self.merit < o.merit || (self.merit == o.merit && lt(o.ratio, self.ratio))
        })
    }
}

impl<T: Real> Kernel<T> {
    /// Step L1.
    fn load<C: ColumnSource<T>>(basis: &C, opts: LuOptions) -> Self {
        let m = basis.m();
        let drop_tol = T::from_f64(opts.drop_tol);
        let mut col_len = vec![0_usize; m];
        let mut row_len = vec![0_usize; m];
        for (k, len) in col_len.iter_mut().enumerate() {
            basis.for_each_entry(k, |i, _| {
                *len += 1;
                row_len[i as usize] += 1;
            });
        }
        let caps = |v: &[usize]| v.iter().map(|&n| n + 4).collect::<Vec<_>>();
        let mut cols = LineStore::new_linked(&caps(&col_len));
        let mut rows = LineStore::new_pattern_linked(&caps(&row_len));
        let mut max_basis = T::ZERO;
        let mut tiny = vec![false; m];
        for (k, tiny_k) in tiny.iter_mut().enumerate() {
            basis.for_each_entry(k, |i, v| {
                *tiny_k |= is_zero(v, drop_tol);
                let (q, r) = (cols.len(k), rows.len(i as usize));
                cols.push_linked(k, i, v, r);
                rows.push_linked(i as usize, idx32(k), T::ZERO, q);
                max_basis = max_basis.max(v.abs());
            });
        }
        debug_assert_eq!(check_links(&cols, &rows, true), Ok(()));
        let mut col_lists = CountLists::new(m, m);
        let mut row_lists = CountLists::new(m, m);
        for j in 0..m {
            col_lists.insert(j, cols.len(j));
        }
        for i in 0..m {
            row_lists.insert(i, rows.len(i));
        }
        Self {
            m,
            abs_tol: T::from_f64(opts.abs_pivot_tol),
            rel_tol: T::from_f64(opts.rel_pivot_tol),
            drop_tol,
            search_limit: opts.search_limit,
            nnz_basis: cols.live(),
            cols,
            rows,
            col_lists,
            row_lists,
            wmap: vec![NIL; m],
            lmap: vec![NIL; m],
            tiny,
            tw: Vec::new(),
            cancelled: Vec::new(),
            max_basis,
            step_row: Vec::with_capacity(m),
            step_col: Vec::with_capacity(m),
            diag: vec![T::ZERO; m],
            l_start: vec![0],
            l_pivot: Vec::new(),
            l_index: Vec::new(),
            l_value: Vec::new(),
            u_trip: Vec::new(),
            dead: Vec::new(),
            col_singletons: 0,
            row_singletons: 0,
            markowitz_pivots: 0,
        }
    }

    /// The output fields, moved (step L7 reads nothing else).
    fn into_elim(self) -> Elim<T> {
        Elim {
            nnz_basis: self.nnz_basis,
            max_basis: self.max_basis,
            col_singletons: self.col_singletons,
            row_singletons: self.row_singletons,
            markowitz_pivots: self.markowitz_pivots,
            step_row: self.step_row,
            step_col: self.step_col,
            diag: self.diag,
            l_start: self.l_start,
            l_pivot: self.l_pivot,
            l_index: self.l_index,
            l_value: self.l_value,
            u_trip: self.u_trip,
            dead: self.dead,
        }
    }

    /// Step L5′: the bump in the COLAMD order of its columns, each column
    /// pivoted on its acceptable entry of least row count (ties: larger
    /// magnitude, then the first in the column).
    fn preordered_phase(&mut self) {
        self.kill_empty_columns();
        for c in self.bump_order() {
            self.kill_empty_columns();
            if !self.col_lists.contains(c) {
                continue;
            }
            let gamma = self
                .cols
                .line(c)
                .1
                .iter()
                .fold(T::ZERO, |g, &v| g.max(v.abs()));
            if !self.above_abs(gamma) {
                self.kill(c);
                continue;
            }
            let u = self.rel_tol;
            let mut best: Option<(usize, usize, T)> = None;
            let (idx, val) = self.cols.line(c);
            for (&i, &v) in idx.iter().zip(val) {
                let (i, a) = (i as usize, v.abs());
                if !self.above_abs(v) || lt(a, u * gamma) {
                    continue;
                }
                let r = self.rows.len(i);
                if best.is_none_or(|(_, rb, ab)| r < rb || (r == rb && lt(ab, a))) {
                    best = Some((i, r, a));
                }
            }
            let (i, _, _) = best.expect("the largest entry is acceptable");
            self.pivot(i, c);
            self.markowitz_pivots += 1;
        }
        self.kill_empty_columns();
        debug_assert_eq!(
            self.col_lists.size(),
            0,
            "every bump column is pivoted or dead"
        );
    }

    /// Step L5′, first part: the COLAMD order (Chapter 5.10, Algorithm C)
    /// of the active columns, on the pattern of the active submatrix.
    fn bump_order(&self) -> Vec<usize> {
        let mut row_of = vec![NIL; self.m];
        let mut n_rows = 0;
        for (i, r) in row_of.iter_mut().enumerate() {
            if self.row_lists.contains(i) {
                *r = n_rows;
                n_rows += 1;
            }
        }
        let active: Vec<usize> = (0..self.m)
            .filter(|&j| self.col_lists.contains(j))
            .collect();
        let mut col_ptr = Vec::with_capacity(active.len() + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
        for &j in &active {
            let start = row_idx.len();
            row_idx.extend(
                self.cols
                    .indices(j)
                    .iter()
                    .map(|&i| idx32(row_of[i as usize])),
            );
            row_idx[start..].sort_unstable();
            col_ptr.push(row_idx.len());
        }
        let pattern = Pattern {
            n_rows,
            col_ptr,
            row_idx,
        };
        debug_assert_eq!(pattern.check_invariants(), Ok(()));
        let perm = Colamd::default()
            .order(&pattern, OrderingTarget::Columns)
            .expect("a valid pattern");
        perm.fwd.iter().map(|&k| active[k as usize]).collect()
    }

    /// `|v| > τ_abs`.
    #[inline]
    fn above_abs(&self, v: T) -> bool {
        lt(self.abs_tol, v.abs())
    }

    /// `s_ij` for the entry at position `r` of row `i` (through its link).
    #[inline]
    fn value_in_row(&self, i: usize, r: usize) -> T {
        let j = self.rows.index_at(i, r) as usize;
        let q = self
            .cols
            .find(j, idx32(i), self.rows.link(i, r))
            .expect("row and column files agree");
        self.cols.value_at(j, q)
    }

    /// The position in the row file of the twin of the entry at position
    /// `q` of column `j` (Algorithm T: its hint, else a search).
    #[inline]
    fn row_twin(&self, j: usize, q: usize) -> (usize, usize) {
        let i = self.cols.index_at(j, q) as usize;
        let r = self
            .rows
            .find(i, idx32(j), self.cols.link(j, q))
            .expect("row and column files agree");
        (i, r)
    }

    /// Removes the dead column `j` from the active submatrix.
    fn kill(&mut self, j: usize) {
        // Its entries leave the row file one by one (Algorithm T; removals
        // from the rows leave column j unchanged), then the column is
        // cleared as a whole.
        for q in 0..self.cols.len(j) {
            let (i, r) = self.row_twin(j, q);
            remove_linked(&mut self.rows, &mut self.cols, i, r);
            self.row_lists.set(i, self.rows.len(i));
        }
        self.cols.clear(j);
        self.col_lists.remove(j);
        self.dead.push(j);
    }

    /// Step L2.
    fn kill_empty_columns(&mut self) {
        while let Some(j) = self.col_lists.first(0) {
            self.kill(j);
        }
    }

    /// Steps L2–L4.
    fn triangular_phase(&mut self) {
        loop {
            self.kill_empty_columns();
            // L3. Column singleton.
            if let Some(j) = self.col_lists.first(1) {
                let (idx, val) = self.cols.line(j);
                let (i, v) = (idx[0] as usize, val[0]);
                if self.above_abs(v) {
                    self.pivot(i, j);
                    self.col_singletons += 1;
                } else {
                    self.kill(j);
                }
                continue;
            }
            // L4. Row singleton with a pivot above τ_abs.
            let mut found = None;
            let mut e = self.row_lists.first(1);
            while let Some(i) = e {
                let j = self.rows.index_at(i, 0) as usize;
                let v = self.value_in_row(i, 0);
                if self.above_abs(v) {
                    found = Some((i, j));
                    break;
                }
                e = self.row_lists.next(i);
            }
            let Some((i, j)) = found else {
                debug_assert_eq!(check_links(&self.cols, &self.rows, false), Ok(()));
                return;
            };
            self.pivot(i, j);
            self.row_singletons += 1;
        }
    }

    /// Step L5.
    fn markowitz_phase(&mut self) {
        loop {
            self.kill_empty_columns();
            if self.col_lists.size() == 0 {
                return;
            }
            let (best, dead) = self.search();
            let killed = !dead.is_empty();
            for j in dead {
                self.kill(j);
            }
            match best {
                Some(c) => {
                    self.pivot(c.i, c.j);
                    self.markowitz_pivots += 1;
                }
                None if killed => {}
                None => {
                    debug_assert!(
                        false,
                        "Markowitz search found neither pivot nor dead column"
                    );
                    return;
                }
            }
        }
    }

    /// Algorithm M.
    fn search(&self) -> (Option<Candidate<T>>, Vec<usize>) {
        let u = self.rel_tol;
        let limit = self.search_limit.max(1);
        let mut best: Option<Candidate<T>> = None;
        let mut dead = Vec::new();
        let mut examined = 0_usize;
        let col_max = |j: usize| {
            self.cols
                .line(j)
                .1
                .iter()
                .fold(T::ZERO, |g, &v| g.max(v.abs()))
        };
        let acceptable = |v: T, gamma: T| self.above_abs(v) && !lt(v.abs(), u * gamma);
        for k in 1..=self.m {
            // M2. Columns of count k.
            let mut e = self.col_lists.first(k);
            while let Some(j) = e {
                e = self.col_lists.next(j);
                let gamma = col_max(j);
                if !self.above_abs(gamma) {
                    dead.push(j);
                    continue;
                }
                let mut any = false;
                let (idx, val) = self.cols.line(j);
                for (&i, &v) in idx.iter().zip(val) {
                    if acceptable(v, gamma) {
                        any = true;
                        let i = i as usize;
                        let c = Candidate {
                            i,
                            j,
                            merit: (self.rows.len(i) - 1) * (k - 1),
                            ratio: v.abs() / gamma,
                        };
                        if c.beats(best.as_ref()) {
                            best = Some(c);
                        }
                    }
                }
                examined += usize::from(any);
                if let Some(b) = &best
                    && (b.merit <= (k - 1) * (k - 1) || examined >= limit)
                {
                    return (best, dead);
                }
            }
            // M3. Rows of count k.
            let mut e = self.row_lists.first(k);
            while let Some(i) = e {
                e = self.row_lists.next(i);
                let mut any = false;
                for (r, &j) in self.rows.indices(i).iter().enumerate() {
                    let j = j as usize;
                    let gamma = col_max(j);
                    let v = self.value_in_row(i, r);
                    if acceptable(v, gamma) {
                        any = true;
                        let c = Candidate {
                            i,
                            j,
                            merit: (k - 1) * (self.cols.len(j) - 1),
                            ratio: v.abs() / gamma,
                        };
                        if c.beats(best.as_ref()) {
                            best = Some(c);
                        }
                    }
                }
                examined += usize::from(any);
                if let Some(b) = &best
                    && (b.merit <= k * (k - 1) || examined >= limit)
                {
                    return (best, dead);
                }
            }
        }
        (best, dead)
    }

    /// Step L6 for column `j` of the `U` row (`u_pj`): `s_ij ← s_ij −
    /// l_ic u_pj` for the `L` column `lcol` (rows marked in `lmap`), fill-in
    /// appended to both files, then the cancellations removed in the order
    /// of the column. `lq` and `cancelled` are work arrays.
    fn update_column(
        &mut self,
        j: usize,
        u_pj: T,
        lcol: &[(u32, T)],
        lq: &mut Vec<usize>,
        cancelled: &mut Vec<(usize, u32)>,
    ) {
        // The positions in column j of the rows of the L column
        // (one pass over the column).
        lq.clear();
        lq.resize(lcol.len(), NIL);
        for (q, &i) in self.cols.line(j).0.iter().enumerate() {
            let k = self.lmap[i as usize];
            if k != NIL {
                lq[k] = q;
            }
        }
        for (k, &(i, l_ic)) in lcol.iter().enumerate() {
            let delta = -l_ic * u_pj;
            let q = lq[k];
            if q == NIL {
                let (q, r) = (self.cols.len(j), self.rows.len(i as usize));
                self.cols.push_linked(j, i, delta, r);
                lq[k] = q;
                self.rows.push_linked(i as usize, idx32(j), T::ZERO, q);
            } else {
                self.cols.values_mut(j)[q] += delta;
            }
        }
        // Cancellations, in the order of the column: only the entries
        // just updated or filled can have become negligible, besides
        // negligible entries of B itself in a column never scanned.
        cancelled.clear();
        if self.tiny[j] {
            self.tiny[j] = false;
            let (idx, val) = self.cols.line(j);
            cancelled.extend(
                idx.iter()
                    .zip(val)
                    .enumerate()
                    .filter(|&(_, (_, &v))| is_zero(v, self.drop_tol))
                    .map(|(q, (&i, _))| (q, i)),
            );
        } else {
            for (k, &(i, _)) in lcol.iter().enumerate() {
                if is_zero(self.cols.value_at(j, lq[k]), self.drop_tol) {
                    cancelled.push((lq[k], i));
                }
            }
            cancelled.sort_unstable();
        }
        // They leave both files in that order; `wmap` holds the
        // current position of each one still to remove and follows
        // the entry that moves into a hole (Algorithm T).
        for &(q, i) in cancelled.iter() {
            self.wmap[i as usize] = q;
        }
        for &(_, i) in cancelled.iter() {
            let q = self.wmap[i as usize];
            let (_, t) = remove_linked(&mut self.cols, &mut self.rows, j, q);
            if q < self.cols.len(j) {
                let moved = self.cols.index_at(j, q) as usize;
                if self.wmap[moved] != NIL {
                    self.wmap[moved] = q;
                }
            }
            let r = self
                .rows
                .find(i as usize, idx32(j), t)
                .expect("row and column files agree");
            remove_linked(&mut self.rows, &mut self.cols, i as usize, r);
            self.wmap[i as usize] = NIL;
        }
    }

    /// Step L6: pivots on `(p, c)` (row `p`, position `c`).
    fn pivot(&mut self, p: usize, c: usize) {
        let (p32, c32) = (idx32(p), idx32(c));
        let piv = self.cols.get(c, p32).expect("pivot entry is stored");

        // L column: multipliers of the other rows of column c, and the
        // hints of their twins in the row file (`tw`, parallel).
        let mut tw = std::mem::take(&mut self.tw);
        tw.clear();
        let mut r_p = NIL;
        let lcol: Vec<(u32, T)> = {
            let (idx, val) = self.cols.line(c);
            let mut lcol = Vec::with_capacity(idx.len().saturating_sub(1));
            for (q, (&i, &v)) in idx.iter().zip(val).enumerate() {
                if i == p32 {
                    r_p = self.cols.link(c, q);
                } else {
                    lcol.push((i, v / piv));
                    tw.push(self.cols.link(c, q));
                }
            }
            lcol
        };
        debug_assert_ne!(r_p, NIL, "the pivot entry is in its column");
        // Column c leaves the row file through the links (Algorithm T), in
        // the order of before: the other rows first, then row p.
        self.cols.clear(c);
        self.col_lists.remove(c);
        for (&(i, _), &t) in lcol.iter().zip(&tw) {
            let i = i as usize;
            let r = self
                .rows
                .find(i, c32, t)
                .expect("row and column files agree");
            remove_linked(&mut self.rows, &mut self.cols, i, r);
        }
        let r_p = self
            .rows
            .find(p, c32, r_p)
            .expect("row and column files agree");
        remove_linked(&mut self.rows, &mut self.cols, p, r_p);

        // U row: the other entries of row p, removed from their columns
        // through the links (which leaves row p itself unchanged); row p is
        // cleared afterwards.
        let mut urow = Vec::with_capacity(self.rows.len(p));
        for r in 0..self.rows.len(p) {
            let j = self.rows.index_at(p, r) as usize;
            let q = self
                .cols
                .find(j, p32, self.rows.link(p, r))
                .expect("row and column files agree");
            let v = self.cols.value_at(j, q);
            remove_linked(&mut self.cols, &mut self.rows, j, q);
            urow.push((idx32(j), v));
        }
        self.rows.clear(p);
        self.row_lists.remove(p);

        // Schur complement update s_ij ← s_ij − l_ic u_pj.
        if !lcol.is_empty() && !urow.is_empty() {
            for (k, &(i, _)) in lcol.iter().enumerate() {
                self.lmap[i as usize] = k;
            }
            // `tw` is free again: it holds the positions in column j.
            let lq = &mut tw;
            let mut cancelled = std::mem::take(&mut self.cancelled);
            for &(j, u_pj) in &urow {
                self.update_column(j as usize, u_pj, &lcol, lq, &mut cancelled);
            }
            for &(i, _) in &lcol {
                self.lmap[i as usize] = NIL;
            }
            self.cancelled = cancelled;
        }
        self.tw = tw;
        for &(i, _) in &lcol {
            let i = i as usize;
            self.row_lists.set(i, self.rows.len(i));
        }
        for &(j, _) in &urow {
            let j = j as usize;
            self.col_lists.set(j, self.cols.len(j));
        }

        // Record the step.
        self.step_row.push(p32);
        self.step_col.push(c32);
        self.diag[p] = piv;
        if !lcol.is_empty() {
            self.l_pivot.push(p32);
            for (i, l) in lcol {
                self.l_index.push(i);
                self.l_value.push(l);
            }
            self.l_start.push(self.l_index.len());
        }
        for (j, u) in urow {
            self.u_trip.push((p32, j, u));
        }
    }
}
