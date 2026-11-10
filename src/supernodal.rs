//! # Chapter 5.17 — `SupernodalLdl`: the own supernodal `LDLᵀ`
//!
//! *New with task L1.6a (PLAN.md trilha L; the contract's §5.7, decision
//! D10 and D-0098). Part 1 of the own factorization: `1×1` pivots for the
//! two regularized systems; the `2×2` pivots of Bunch–Kaufman for the
//! unregularized augmented system are task L1.6b.*
//!
//! ## Purpose
//! [`SupernodalLdl`] implements [`KktSolver`] with no dependency: a
//! supernodal `LDLᵀ` of `P K̃_reg Pᵀ` written here, generic in the scalar
//! `T: Real`, with the dense arithmetic of its blocks delegated to a
//! [`DenseKernel`] (`NaiveDense` always, `FaerDense` with the `faer`
//! feature, Chapter 5.16). What it shares with every backend is the
//! trait's: the ordering (any [`Ordering`], our [`crate::Amd`] by default),
//! the regularization policy and the inertia check (Algorithm K of Chapter
//! 5.14), the refinement (Algorithm F). What is new here is the symbolic
//! phase (elimination tree, relaxed supernodes, the row pattern of every
//! supernode and the scatter map of `K`), the numeric phase (left-looking
//! supernodal elimination with the pivot bumping of step K3 and an exact
//! count of the bumped pivots of both signs) and the solves, which allocate
//! nothing.
//!
//! *Why left-looking.* CHOLMOD (Chen, Davis, Hager & Rajamanickam 2008,
//! §2.3) and faer 0.24 (read, not copied) both factorize supernode by
//! supernode in postorder, each one receiving the updates of its
//! descendants before it is factorized ("left-looking"); the alternative,
//! multifrontal, carries a stack of dense frontal matrices whose peak
//! memory can exceed the factor's by a large factor on wide trees (Liu
//! 1992), while left-looking needs only the factor and three scratch
//! blocks. With the same supernodes the two do the same flops; the choice
//! is memory and simplicity, and the CHOLMOD paper reports no systematic
//! speed advantage of either on its test set.
//!
//! ## Notation
//! As in Chapters 5.14–5.15: `K` of order `N` by its lower triangle, `L₀`
//! the analyzed lower pattern with a full diagonal, `π` the ordering
//! ([`Perm`], `fwd[k]` = the row eliminated `k`-th), `σ_k` the pivot signs,
//! `(ρ, δ, τ, β)` the regularization, `K̃_reg = S K S + Σ` the equilibrated
//! regularized matrix of step K2″ whose factor `P K̃_reg Pᵀ = L Δ Lᵀ` is
//! computed, `L` unit lower triangular, `Δ = diag(d_k)`. Columns are in
//! the *permuted* numbering `0..N` unless said otherwise. The elimination
//! tree has `parent[j]` (Chapter 5.9, Algorithm E) and column counts `c_j`
//! (entries of column `j` of `L`, diagonal included).
//!
//! A *supernode* `s` is a run of consecutive columns `[s₀, s₁)` whose
//! columns of `L` have nested patterns: the pattern of column `j ∈ s`
//! below the diagonal block is the same set `R_s` of `deg(s) = c_{s₁−1} − 1`
//! rows, all `≥ s₁`. Its block of `L` is stored dense, column-major, as a
//! [`DenseBlock`] of `nrows(s) = ncols(s) + deg(s)` rows and
//! `ncols(s) = s₁ − s₀` columns: row `i < ncols(s)` is column `s₀ + i`,
//! row `ncols(s) + q` is the row `R_s[q]`; the diagonal of the block holds
//! `Δ` (the unit diagonal of `L` is implicit) and the strict upper triangle
//! of the diagonal block is never read. A *fundamental* supernode has
//! exactly nested patterns; a *relaxed* one merges a child into its parent
//! at the price of explicit zeros (entries stored where the fundamental
//! structure has none).
//!
//! ## Algorithm A (Analyze)
//! A1. `L₀ ←` the lower triangle of the pattern plus a full diagonal
//!     (invariant (K1) of Chapter 5.14; [`KktError::NotSymmetric`] or
//!     [`KktError::Structure`] otherwise); `G ← L₀ + L₀ᵀ`.
//! A2. `π ← ordering.order(G, Symmetric)` (its length checked).
//! A2′. `π ←` the postorder of the elimination tree of `π` with the
//!     heaviest child last (Algorithm P of Chapter 5.9): every subtree is
//!     a run of consecutive columns, so every chain of only children is a
//!     candidate supernode, and the largest child sits right before its
//!     parent, where step A4 can merge it.
//! A3. `(parent, c) ←` Algorithm E on `G` under `π` (the tree and the
//!     column counts); `nchild[j]` the number of children of `j`.
//! A3′. [Switch.] `φ = Σ_j c_j + Σ_j c_j(c_j+1)/2` (the divisions and
//!     multiply-subtracts of the `LDLᵀ`) and `ℓ = Σ_j c_j`; if
//!     `φ ≤ supernodal_switch · ℓ` ([`SupernodalOptions`], default 40 as
//!     CHOLMOD's `supernodal_switch` and faer's `AUTO`) the matrix is
//!     *simplicial*: A5 keeps the fundamental supernodes and Algorithm N
//!     takes the simplicial path everywhere — on such matrices dense
//!     blocks only add explicit zeros and copies (measured in L1.6a:
//!     fit2p's augmented system, 16 525 columns in 13 500 supernodes,
//!     factorized 2× slower with relaxed blocks than column by column).
//! A4. [Fundamental supernodes.] Column `j` continues the supernode of
//!     `j − 1` iff `parent[j − 1] = j`, `nchild[j] = 1` and
//!     `c_{j−1} = c_j + 1` (Liu, Ng & Peyton 1993); otherwise it starts a
//!     new one. Each fundamental supernode `f` has `size(f)` columns and
//!     `deg(f) = c_{last} − 1`; its parent in the *supernodal* tree is the
//!     supernode of `parent[last]`.
//! A5. [Relaxed amalgamation.] Walk the fundamental supernodes in order,
//!     keeping a *group* that starts as `{f}` and whose top is `f`. The
//!     group (of `size` columns, top degree `deg(f)`, `zeros` explicit
//!     zeros so far) is merged into `f + 1` when `f + 1` is the parent of
//!     `f` — the only child adjacent to it, by A2′ — and the rule below
//!     accepts; the group then continues with top `f + 1`. Otherwise the
//!     group is closed as one supernode and a new group starts at `f + 1`.
//!     Merging adds `z_new = size · (size(f+1) + deg(f+1) − deg(f))`
//!     zeros (every column of the child gains the rows of the parent's
//!     block it does not have; the child's pattern is contained in the
//!     parent's columns and pattern, so the difference is never negative),
//!     for a merged supernode of `w = size + size(f+1)` columns and
//!     `e = w(w+1)/2 + deg(f+1) · w` stored entries in its lower part. With
//!     `relax_zeros = r` ([`SupernodalOptions`]), the merge is accepted
//!     iff `r > 0` and one of: `z_new = 0` (nothing is lost); `w ≤ r`
//!     (small supernodes always merge: CHOLMOD's `nrelax[0]`, faer's
//!     first cutoff); or `(zeros + z_new) / e ≤ ζ(w)` with
//!     `ζ = 0.8` for `w ≤ 16`, `0.1` for `w ≤ 48`, `0.05` beyond
//!     (CHOLMOD's `nrelax[1..]`/`zrelax` defaults, which faer uses too;
//!     [`RELAX_CUTOFFS`], compared in integers). `r = 0` keeps the
//!     fundamental supernodes. The rule is faer's (read, not copied), so
//!     the stored factor of this chapter is the same size as faer's under
//!     the same permutation — a stated acceptance criterion of L1.6a.
//! A6. [Patterns.] For every supernode `s` in order, `R_s` is the union,
//!     restricted to rows `≥ s₁`, of the rows of `G` in the columns of `s`
//!     and of the patterns `R_c` of its children `c` in the supernodal
//!     tree (the children come first in postorder); sorted increasing.
//!     Invariant (L2): `|R_s| = c_{s₁−1} − 1`.
//! A7. [Descendants.] For every supernode `d` and every maximal run of
//!     consecutive rows `R_d[a..b)` that fall in the columns of one
//!     supernode `s`, record `(d, a, b)` in the list of `s`: the updates
//!     `s` receives in Algorithm N come from exactly these `d`, with
//!     `R_d[a..b)` the columns of `s` they touch and `R_d[a..]` the rows.
//! A8. [Scatter map.] For every entry `p` of `L₀` (original indices
//!     `(i, j)`), the supernode of `min(inv[i], inv[j])` and the offset of
//!     the permuted entry in its block: the numeric phase scatters the
//!     values of `K̃_reg` with one indexed store each.
//! A9. [Sizes.] `nnz_l = Σ_s nrows(s) · ncols(s)`, the stored length of
//!     the factor (dense blocks, their explicit zeros and the unused strict
//!     upper triangles of the diagonal blocks included — the convention of
//!     [`crate::FaerSymbolic::nnz_factor`], (F2), so the two backends are
//!     comparable). Invariant (L3): the lower parts of the blocks,
//!     `Σ_s (ncols(s)(ncols(s)+1)/2 + deg(s) · ncols(s))`, equal `Σ_j c_j`
//!     ([`Pattern::cholesky_nnz`]) plus the zeros counted in A5.
//!
//! ## Algorithm N (Numeric factorization, left-looking supernodal)
//! N1. `v ←` the values of `K` on `L₀` (steps K1′–K2 of Chapter 5.14),
//!     equilibrated (K2′), plus the static shifts (K2″): `v` holds `K̃_reg`.
//! N2. Allocate the blocks of `L` (zeros) and scatter `v` into them by the
//!     map of A8. The pivot signs in permuted order are `σ_{fwd[k]}`.
//! N3. For `s = 0, 1, …` (postorder: every descendant before `s`):
//!     N3a. [Map.] `loc[s₀ + i] ← i` for the columns of `s`,
//!          `loc[R_s[q]] ← ncols(s) + q` for its pattern rows.
//!     N3b. [Updates.] For every `(d, a, b)` in the list of `s` (A7),
//!          Algorithm U: `L_s ← L_s − L_d[a.., :] Δ_d L_d[a..b, :]ᵀ` on the
//!          rows and columns `loc` says, lower part only.
//!     N3c. [Factor.] Algorithm D on the block of `s`: the diagonal block
//!          `A₁₁ = L₁₁ Δ_s L₁₁ᵀ` with the bumping of step K3 under the
//!          signs `σ` of its columns, then `L₂₁ ← A₂₁ L₁₁⁻ᵀ Δ_s⁻¹` for the
//!          rows of `R_s`. A zero pivot without dynamic regularization is
//!          [`KktError::ZeroPivot`] at the original row `fwd[k]`; a pivot
//!          that is not finite is [`KktError::NotFinite`] there (an
//!          overflow of the elimination is never bumped).
//!     N3d. [Unmap.] Reset `loc` for the rows touched in N3a.
//! N4. [Unscale.] The factor computed is that of `P K̃_reg Pᵀ`, `K̃ =
//!     S K S`. Since `K_reg = S⁻¹ K̃_reg S⁻¹ = (S⁻¹ L S)(S⁻¹ Δ S⁻¹)(S⁻¹ L
//!     S)ᵀ`, the stored blocks become `L̂ = S⁻¹ L S` (`l̂_ij = l_ij s_j /
//!     s_i`, unit diagonal kept) and `Δ̂ = S⁻¹ Δ S⁻¹` (`d̂_k = d_k /
//!     s_k²`) — exact, the scales being powers of two — so that the
//!     solves work in the caller's units with no scaling pass (the
//!     equilibration is kept, [`NumericLdl::scale`], to read the pivots
//!     of the equilibrated matrix back: `d_k = d̂_k s_k²`).
//! N5. (S10) Every stored value of `L̂` and `Δ̂` is finite, or the call
//!     fails with [`KktError::NotFinite`] at the original row of the first
//!     column holding a NaN or an infinity (step K3′).
//! N6. (S6) The inertia is the count of signs of `Δ̂` (= of `Δ`, step K4;
//!     [`KktError::WrongInertia`] unless it is the form's). With
//!     `τ = Some`, every pivot has its sign `σ_k` by construction and the
//!     check cannot fail. The factor also allocates its work vector for
//!     Algorithm S (`N` scalars).
//!
//! A supernode with fewer than `min_supernode_cols` columns
//! ([`SupernodalOptions`]) takes the *simplicial path*: steps N3b and N3c
//! run column by column with plain loops (the same arithmetic, in the same
//! order as the kernel path up to the rounding of the sums), and so does
//! an update whose descendant `d` has fewer than `min_supernode_cols`
//! columns — a dense kernel on a block one to seven columns wide costs
//! more in copies than it saves. Above the threshold the kernel path
//! gathers the operands into scratch blocks and calls `gemm_sub` and
//! `trsm_lower` ([`DenseKernel`]). The storage is the same on both paths.
//!
//! ## Algorithm U (Update of `s` by a descendant `d`)
//! With `R_d[a..b)` the rows of `d` inside the columns of `s`, `n_m = b − a`,
//! `n_r = |R_d| − a` (the rows of `d` from `a` on: they lie in `s`'s
//! columns or in `R_s`, by A6):
//! U1. `A ← L_d[a.., :]` (`n_r × ncols(d)`), `B ← L_d[a..b, :] Δ_d`
//!     (`n_m × ncols(d)`, the columns scaled by `d_k`), gathered into
//!     scratch blocks (the rows of a column-major block are strided, and a
//!     [`DenseBlock`] has no stride: the gather is the price of the
//!     kernel interface).
//! U2. `C ← 0` (`n_r × n_m`); `C ← C − A Bᵀ` by `gemm_sub`.
//! U3. [Scatter.] For `j < n_m` and `i ≥ j`: `L_s[loc(R_d[a+i]),
//!     R_d[a+j] − s₀] += c_ij` — the lower part of the update, since
//!     `R_d` is sorted so that `i ≥ j` is row at or below column.
//! Simplicial path: for `k < ncols(d)`, `j ∈ [a, b)`, `i ∈ [j, |R_d|)`:
//! `L_s[loc(R_d[i]), R_d[j] − s₀] −= l_{d,ik} · (d_k l_{d,jk})`, column `k`
//! of `L_d` read once, contiguous.
//!
//! ## Algorithm D (Dense `LDLᵀ` of a supernode block, with bumping)
//! On the block `A` of `s` (`n_r` rows, `n_c` columns), panels of
//! [`PANEL`] columns `[p, q)`:
//! D1. [Tile.] Columns `p..q`, rows `p..q`, by the scalar right-looking
//!     elimination: for `k = p, …, q − 1`: `d_k ← a_kk`; if `d_k` is not
//!     finite, fail (N3c); if `τ = Some` and `σ_k d_k ≤ τ`, `d_k ← σ_k β`
//!     and count the bump (step K3); if `τ = None` and `d_k = 0`, fail
//!     (zero pivot); `l_ik ← a_ik / d_k` for `k < i < q`; `a_ij ← a_ij −
//!     l_ik d_k l_jk` for `k < j ≤ i < q`.
//! D2. [Panel below the tile.] If `q < n_r`: `L₂₁ ← A₂₁ L̃⁻ᵀ` with `L̃ =
//!     L₁₁ Δ₁` (the unit lower tile with its columns scaled by the pivots:
//!     lower triangular with diagonal `d_k`), by `trsm_lower(L̃, A₂₁,
//!     transpose)`: row `i` of `A₂₁` solves `L̃ x = a_i`, i.e.
//!     `l_ik = (a_ik − Σ_{j<k} l_ij d_j l_kj) / d_k` — the formula of D1
//!     for the rows below the tile, with the sums in the kernel's order.
//! D3. [Trailing update.] If `q < n_c`: for the remaining columns, in
//!     panels `[c, c + w)`: `A[c.., c..c+w] ← A[c.., c..c+w] − W[c−q..,
//!     :] L₂₁[c−q..c−q+w, :]ᵀ` with `W = L₂₁ Δ₁`, by `gemm_sub` on
//!     gathered blocks — the lower trapezoid of the trailing matrix, one
//!     panel at a time (the strict upper triangle of each `w × w` diagonal
//!     tile is computed and discarded: `w² ncols` wasted flops per panel
//!     against `n_r w ncols` useful ones).
//! For `n_c ≤ PANEL` there is one tile and no trailing update: the whole
//! supernode is D1 on its diagonal block and D2 for its pattern rows. The
//! simplicial path runs D1 with `q = n_r` on all the rows and `n_c`
//! columns (no panels, no kernel).
//!
//! ## Algorithm S (Solve)
//! `y ← P x` gathered into the factor's work vector (`y_k = x_{fwd[k]}`;
//! a second thread solving with the same factor at the same moment uses
//! a temporary vector instead of waiting — the mutex of task L1.5b, D6;
//! an in-place permutation by cycles was measured first and rejected:
//! following a cycle is a chain of dependent loads, twice slower than the
//! gather on the 16 525 unknowns of fit2p's augmented system). Then,
//! supernode by supernode:
//! S1. [Forward, `L̂ y = y`.] For `s` in order, for each column `j` of
//!     `s`: `y_i ← y_i − l̂_ij y_j` for the rows `i` of the diagonal block
//!     below `j` (a contiguous `axpy`) and for the rows `R_s` (a scatter;
//!     column `j` of the block read once).
//! S2. [Diagonal.] `y_k ← y_k / d̂_k`.
//! S3. [Backward, `L̂ᵀ y = y`.] For `s` in reverse order and `j` from the
//!     last column of `s` down: `y_j ← y_j − Σ_{i>j in s} l̂_ij y_i −
//!     Σ_q l̂_{R_s[q], j} y_{R_s[q]}`: two dot products with the same
//!     column, each summed in four independent partial sums (a single
//!     accumulator makes the sweep latency-bound: 2× slower on maros-r7's
//!     normal equations, measured in L1.6a) and combined in a fixed order.
//! A supernode of one column takes the plain loops of a column-oriented
//! simplicial solve. Finally `x_{fwd[k]} ← y_k`. Nothing is allocated
//! while the work vector is free (invariant (L5)); the factor is read only.
//!
//! ## Invariants
//! (L1) The supernodes partition `0..N` into runs in increasing order;
//!      `R_s` is sorted, its rows are `≥ s₁`, and every row of `R_s` lies
//!      in the columns of an ancestor of `s`.
//! (L2) `|R_s| = c_{s₁−1} − 1` (the pattern of the last column, Liu 1990).
//! (L3) `Σ_s (ncols(ncols+1)/2 + deg · ncols) = Σ_j c_j + zeros(A5)`:
//!      the stored lower parts are the structural factor plus the counted
//!      zeros of the relaxation; with `relax_zeros = 0`, exactly
//!      [`Pattern::cholesky_nnz`].
//! (L4) Every `(d, a, b)` of A7 has `d < s`, `R_d[a..b) ⊆ [s₀, s₁)` and
//!      `R_d[b..] ⊆ R_s`.
//! (L5) [`KktSolver::solve`] allocates nothing (its work vector belongs to
//!      the factor) and reads the factor only.
//! (L6) The factorization is a deterministic function of its inputs:
//!      no threads, no hashing, the same operations in the same order —
//!      two factorizations of the same matrix are equal bit for bit.
//! (S6), (S10) as in Chapter 5.14, checked in steps N4–N5 and asserted in
//!      debug builds. [`SymbolicLdl::check_invariants`] checks (L1)–(L4);
//!      the L1.6a harness checks (L5) with a counting allocator (0
//!      allocations in a second solve and in a refined solve with a reused
//!      workspace), the tests check (L6) by comparing two factors.
//!
//! ## What was read
//! CHOLMOD (Chen, Davis, Hager & Rajamanickam 2008: §2.2 analyze, §2.3
//! the supernodal left-looking numeric phase and its relaxed amalgamation
//! parameters `nrelax`, `zrelax`); faer 0.24.4
//! `sparse::linalg::cholesky::supernodal` (the fundamental-supernode test,
//! the merge rule on adjacent children, the storage of a supernode as one
//! column-major block with `Δ` on its diagonal, the per-supernode row
//! structure that drives the left-looking updates) and `amd::postorder`;
//! Liu, Ng & Peyton 1993 (supernodes), Ng & Peyton 1993 (left-looking
//! supernodal), Liu 1992 (multifrontal memory) — read, not copied.
//!
//! ## References
//! See Chapter 5.14; Chen, Davis, Hager & Rajamanickam, "Algorithm 887:
//! CHOLMOD", ACM TOMS 35(3), 2008; Liu, Ng & Peyton, "On finding
//! supernodes for sparse matrix computations", SIAM J. Matrix Anal. Appl.
//! 14(1), 1993; Ng & Peyton, "Block sparse Cholesky algorithms on advanced
//! uniprocessor computers", SIAM J. Sci. Comput. 14(5), 1993; Liu, "The
//! multifrontal method for sparse matrix solution: theory and practice",
//! SIAM Review 34(1), 1992; Liu, "The role of elimination trees in sparse
//! factorization", SIAM J. Matrix Anal. Appl. 11(1), 1990; Davis, *Direct
//! Methods for Sparse Linear Systems* (2006), ch. 4; Ashcraft & Grimes,
//! "The influence of relaxed supernode partitions on the multifrontal
//! method", ACM TOMS 15(4), 1989 (relaxed amalgamation).

use core::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::factor::count_f64;
use crate::kkt::{
    assemble_regularized, check_inertia, check_regularization, count_inertia, lower_with_diagonal,
    pivot_signs,
};
use crate::perm::{etree_counts, idx32};
use crate::real::{Real, is_exact_zero, le};
use crate::{
    CscMatrix, DenseBlock, DenseKernel, Inertia, KktError, KktForm, KktSolver, NaiveDense,
    Ordering, OrderingTarget, Pattern, Perm, Regularization, SparseError,
};

/// "No node".
const NONE: u32 = u32::MAX;

/// Panel width of Algorithm D (a blocking size for the dense kernels, not
/// a numerical threshold).
pub const PANEL: usize = 32;

/// Step A5: the largest fraction of explicit zeros a relaxed supernode of
/// at most the given number of columns may hold, in per mille — CHOLMOD's
/// `nrelax[1..3]`/`zrelax` defaults (`16: 0.8`, `48: 0.1`, else `0.05`),
/// which faer 0.24 uses too. The first cutoff (always merge up to
/// `relax_zeros` columns) is [`SupernodalOptions::relax_zeros`].
pub const RELAX_CUTOFFS: [(usize, usize); 3] = [(16, 800), (48, 100), (usize::MAX, 50)];

/// Options of the supernodal factorization (steps A5 and N3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SupernodalOptions {
    /// Relaxed amalgamation (step A5): a child supernode is merged into
    /// its parent whenever the merged supernode has at most this many
    /// columns, and beyond that when the fraction of explicit zeros it
    /// adds stays under CHOLMOD's cutoffs ([`RELAX_CUTOFFS`]); `0` keeps
    /// the fundamental supernodes (no explicit zero). Default `4`
    /// (CHOLMOD's `nrelax[0]`).
    pub relax_zeros: usize,
    /// Below this many columns a supernode is handled by the simplicial
    /// path (column by column, plain loops) instead of the dense kernel;
    /// an update by a descendant narrower than this takes the plain loops
    /// too. Default `8`.
    pub min_supernode_cols: usize,
    /// The simplicial/supernodal switch (step A3′): when the flop count of
    /// the factorization per stored entry of `L`, `(Σ_j c_j + Σ_j
    /// c_j(c_j+1)/2) / Σ_j c_j`, is at most this, the matrix is too sparse
    /// for dense blocks to pay: the fundamental supernodes are kept (no
    /// relaxation, no explicit zero) and every supernode takes the
    /// simplicial path. `0` forces the supernodal structure, `∞` the
    /// simplicial one. Default `40` (CHOLMOD's `supernodal_switch`, faer's
    /// `AUTO`).
    pub supernodal_switch: f64,
}

impl Default for SupernodalOptions {
    fn default() -> Self {
        Self {
            relax_zeros: 4,
            min_supernode_cols: 8,
            supernodal_switch: 40.0,
        }
    }
}

/// What the solves need of the symbolic phase, shared by the symbolic and
/// the numeric objects (an `Arc`: a [`NumericLdl`] solves after its
/// [`SymbolicLdl`] is dropped, and never with another one's structure).
#[derive(Debug)]
struct Structure {
    n: usize,
    /// `π` (step A2′): `fwd[k]` the original row eliminated `k`-th.
    perm: Perm,
    /// Supernode boundaries (`n_supernodes + 1` entries).
    supernodes: Vec<u32>,
    /// `R_s` for every supernode, CSR.
    row_ptr: Vec<usize>,
    row_idx: Vec<u32>,
}

impl Structure {
    #[inline]
    fn n_supernodes(&self) -> usize {
        self.supernodes.len() - 1
    }

    #[inline]
    fn cols(&self, s: usize) -> (usize, usize) {
        (self.supernodes[s] as usize, self.supernodes[s + 1] as usize)
    }

    #[inline]
    fn pattern(&self, s: usize) -> &[u32] {
        &self.row_idx[self.row_ptr[s]..self.row_ptr[s + 1]]
    }
}

/// The symbolic factor of [`SupernodalLdl`] (Algorithm A): permutation,
/// elimination tree, supernode partition and the row pattern of every
/// supernode, plus (private) the scatter map of `K` and the descendant
/// lists of the left-looking numeric phase.
#[derive(Clone, Debug)]
pub struct SymbolicLdl {
    /// The fill-reducing permutation used (`fwd[k]` = the original row
    /// eliminated `k`-th): the ordering's, postordered (step A2′).
    pub perm: Perm,
    /// Parent of each column (permuted numbering) in the elimination tree;
    /// `u32::MAX` for a root.
    pub etree: Vec<u32>,
    /// Supernode boundaries over the permuted columns: supernode `s` is
    /// the columns `supernodes[s]..supernodes[s + 1]` (`n_supernodes + 1`
    /// entries).
    pub supernodes: Vec<u32>,
    /// Row pattern `R_s` of each supernode (the rows below its diagonal
    /// block, permuted numbering, increasing): `row_idx[row_ptr[s]..row_ptr[s + 1]]`.
    pub row_ptr: Vec<usize>,
    /// See [`SymbolicLdl::row_ptr`].
    pub row_idx: Vec<u32>,
    /// Stored length of the factor: `Σ_s nrows(s) · ncols(s)` (step A9).
    pub nnz_l: usize,
    /// The form the pattern was analyzed for.
    pub form: KktForm,
    /// `L₀` (step A1), the pattern the numeric phase assembles on.
    lower: Pattern,
    /// Step A8: for entry `p` of `lower`, its supernode and the offset in
    /// the block.
    sc_super: Vec<u32>,
    sc_off: Vec<usize>,
    /// Step A7: the `(d, a, b)` of every supernode, CSR by `s`.
    upd_ptr: Vec<usize>,
    upd_d: Vec<u32>,
    upd_from: Vec<u32>,
    upd_to: Vec<u32>,
    /// The explicit zeros counted in step A5 (invariant (L3)).
    relax_zeros: usize,
    /// The column counts `c_j` of Algorithm E (invariant (L3)).
    counts: Vec<usize>,
    /// Step A3′: the simplicial structure was chosen.
    simplicial: bool,
    shared: Arc<Structure>,
}

impl SymbolicLdl {
    /// Number of supernodes.
    pub fn n_supernodes(&self) -> usize {
        self.shared.n_supernodes()
    }

    /// `true` if the switch of step A3′ chose the simplicial structure
    /// (fundamental supernodes, plain loops), `false` for the supernodal
    /// one (relaxed supernodes, dense kernels).
    pub fn is_simplicial(&self) -> bool {
        self.simplicial
    }

    /// The explicit zeros the relaxed amalgamation added to the lower
    /// parts of the blocks (step A5): `0` with `relax_zeros = 0`.
    pub fn relaxation_zeros(&self) -> usize {
        self.relax_zeros
    }

    /// Checks invariants (L1)–(L4).
    #[allow(clippy::too_many_lines)] // the four invariants in one place
    pub fn check_invariants(&self) -> Result<(), SparseError> {
        let st = &self.shared;
        let n = st.n;
        let ns = st.n_supernodes();
        let bad = |what: String| SparseError::Malformed(what);
        if self.perm.len() != n || st.perm != self.perm {
            return Err(bad("(L1): permutation and structure disagree".into()));
        }
        self.perm.check_invariants()?;
        if st.supernodes.len() != self.supernodes.len()
            || st.supernodes != self.supernodes
            || st.row_ptr != self.row_ptr
            || st.row_idx != self.row_idx
        {
            return Err(bad("(L1): public fields and structure disagree".into()));
        }
        if st.supernodes.first() != Some(&0) || st.supernodes.last().map(|&x| x as usize) != Some(n)
        {
            return Err(bad("(L1): supernodes do not cover 0..N".into()));
        }
        let mut super_of = vec![NONE; n];
        let mut lower_stored = 0_usize;
        let mut stored = 0_usize;
        for s in 0..ns {
            let (s0, s1) = st.cols(s);
            if s1 <= s0 {
                return Err(bad(format!("(L1): supernode {s} is empty or decreasing")));
            }
            super_of[s0..s1].fill(idx32(s));
            let pat = st.pattern(s);
            for (q, &r) in pat.iter().enumerate() {
                if (r as usize) < s1 || (q > 0 && pat[q - 1] >= r) {
                    return Err(bad(format!(
                        "(L1): pattern of supernode {s} not sorted above it"
                    )));
                }
            }
            // (L2)
            if pat.len() + 1 != self.counts[s1 - 1] {
                return Err(bad(format!(
                    "(L2): supernode {s} has {} rows, column count says {}",
                    pat.len(),
                    self.counts[s1 - 1] - 1
                )));
            }
            let nc = s1 - s0;
            lower_stored += nc * (nc + 1) / 2 + pat.len() * nc;
            stored += (nc + pat.len()) * nc;
        }
        // (L1): rows of R_s in ancestors of s.
        for s in 0..ns {
            let (_, s1) = st.cols(s);
            let mut anc = self.etree[s1 - 1];
            let mut pat = st.pattern(s).iter().peekable();
            while let Some(&&r) = pat.peek() {
                if anc == NONE {
                    return Err(bad(format!(
                        "(L1): row {r} of supernode {s} is in no ancestor"
                    )));
                }
                let a = super_of[anc as usize] as usize;
                let (a0, a1) = st.cols(a);
                while let Some(&&r) = pat.peek() {
                    if (r as usize) < a1 {
                        if (r as usize) < a0 {
                            return Err(bad(format!(
                                "(L1): row {r} of supernode {s} is in no ancestor"
                            )));
                        }
                        pat.next();
                    } else {
                        break;
                    }
                }
                anc = self.etree[a1 - 1];
            }
        }
        // (L3)
        let structural: usize = self.counts.iter().sum();
        if lower_stored != structural + self.relax_zeros || stored != self.nnz_l {
            return Err(bad(format!(
                "(L3): stored lower {lower_stored} ≠ structural {structural} + zeros {}",
                self.relax_zeros
            )));
        }
        // (L4)
        for s in 0..ns {
            let (s0, s1) = st.cols(s);
            for u in self.upd_ptr[s]..self.upd_ptr[s + 1] {
                let d = self.upd_d[u] as usize;
                let (a, b) = (self.upd_from[u] as usize, self.upd_to[u] as usize);
                let pd = st.pattern(d);
                let pat_s = st.pattern(s);
                if d >= s
                    || a >= b
                    || b > pd.len()
                    || pd[a..b]
                        .iter()
                        .any(|&r| (r as usize) < s0 || (r as usize) >= s1)
                    || pd[b..].iter().any(|&r| pat_s.binary_search(&r).is_err())
                {
                    return Err(bad(format!(
                        "(L4): update ({d}, {a}, {b}) of supernode {s}"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// The numeric factor of [`SupernodalLdl`] (Algorithm N): `P K_reg Pᵀ =
/// L̂ Δ̂ L̂ᵀ` with `L̂` by supernodal blocks (column-major, `Δ̂` on the
/// diagonal, the unit diagonal of `L̂` implicit) and `Δ̂` as `1×1` pivots,
/// in the caller's units (step N4); `pivot_2x2` is all `false` in this
/// edition (the `2×2` pivots of `LBLᵀ` are task L1.6b).
#[derive(Clone, Debug)]
pub struct NumericLdl<T: Real = f64> {
    /// One block per supernode: `nrows(s) × ncols(s)`, column-major.
    pub blocks: Vec<DenseBlock<T>>,
    /// `Δ̂` in pivot order (`diag[k]` = the pivot of permuted column `k`,
    /// in the caller's units: step N4).
    pub diag: Vec<T>,
    /// `2×2` pivot marks (none in `LDLᵀ`; kept for the `LBLᵀ` of L1.6b).
    pub pivot_2x2: Vec<bool>,
    /// The inertia of the factor (S6).
    pub inertia: Inertia,
    /// Pivots that dynamic regularization bumped (step K3), of both signs,
    /// counted as they are bumped.
    pub bumped: usize,
    /// `s` of step K2′ (powers of two), original numbering.
    scale: Vec<T>,
    shared: Arc<Structure>,
    /// The work vector of Algorithm S (`N` scalars), allocated by the
    /// factorization.
    scratch: SolveScratch<T>,
}

/// The work vector of [`KktSolver::solve`], allocated once by the
/// factorization; a mutex so that a shared factor stays `Sync` (a second
/// thread solving with the same factor at the same moment gets a
/// temporary vector instead of waiting).
struct SolveScratch<T>(Mutex<Vec<T>>);

impl<T: Clone> Clone for SolveScratch<T> {
    fn clone(&self) -> Self {
        let v = match self.0.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        Self(Mutex::new(v))
    }
}

impl<T> core::fmt::Debug for SolveScratch<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SolveScratch")
    }
}

impl<T: Real> NumericLdl<T> {
    /// The order `N` of the factor.
    pub fn order(&self) -> usize {
        self.shared.n
    }

    /// Number of supernodes.
    pub fn n_supernodes(&self) -> usize {
        self.shared.n_supernodes()
    }

    /// The permutation of the factor (`fwd[k]` = the original row of pivot
    /// `k`; step A2′).
    pub fn perm(&self) -> &Perm {
        &self.shared.perm
    }

    /// The equilibration `s` of step K2′ (powers of two, original
    /// numbering): the stored factor is of `P K_reg Pᵀ`, the pivots of the
    /// equilibrated matrix being `d_k · s_{fwd[k]}²`.
    pub fn scale(&self) -> &[T] {
        &self.scale
    }
}

/// Our own supernodal `LDLᵀ` (D10; Chapter 5.17): analyze (ordering,
/// elimination tree, relaxed supernodes, pattern of `L`), factorize
/// (left-looking supernodal with the dense kernel `K` on the diagonal and
/// update blocks, pivot bumping with the expected signs, regularization
/// policy of Chapter 5.14), solve (block forward/diagonal/backward sweeps
/// without allocation). Generic in the scalar `T`; `K` is [`NaiveDense`]
/// by default and `FaerDense` with the `faer` feature.
///
/// ```
/// use xingo_sparse::{Amd, Col, CscMatrix, KktForm, KktSolver, Refinement, Regularization, Row, SupernodalLdl};
/// // K = [−D, Aᵀ; A, 0] with D = diag(1, 2), A = [1 1]: lower triangle.
/// let k = CscMatrix::from_triplets(3, 3, &[
///     (Row(0), Col(0), -1.0), (Row(2), Col(0), 1.0),
///     (Row(1), Col(1), -2.0), (Row(2), Col(1), 1.0),
/// ])
/// .unwrap();
/// let form = KktForm::Augmented { n_primal: 2, n_dual: 1 };
/// let solver = SupernodalLdl::<f64>::default(); // NaiveDense kernels, no dependency
/// let sym = solver.analyze(&k.pattern(), form, &Amd::default()).unwrap();
/// let (num, inertia) = solver.factorize(&sym, &k, Regularization::default()).unwrap();
/// assert_eq!(inertia, form.expected_inertia(3)); // (S6): 1 positive, 2 negative
/// let b = [1.0, 0.0, 3.0];
/// let mut x = [0.0; 3];
/// let report = solver.solve_refined(&num, &k, &b, &mut x, Refinement::default());
/// assert!(report.converged && report.residual <= 1e-12);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SupernodalLdl<T: Real = f64, K: DenseKernel<T> = NaiveDense> {
    /// Amalgamation and path-selection options.
    pub opts: SupernodalOptions,
    kernel: K,
    _t: PhantomData<T>,
}

impl<T: Real, K: DenseKernel<T> + Default> Default for SupernodalLdl<T, K> {
    fn default() -> Self {
        Self::new(K::default(), SupernodalOptions::default())
    }
}

impl<T: Real, K: DenseKernel<T>> SupernodalLdl<T, K> {
    /// A solver with the given dense kernel and options.
    pub fn new(kernel: K, opts: SupernodalOptions) -> Self {
        Self {
            opts,
            kernel,
            _t: PhantomData,
        }
    }

    /// The dense kernel.
    pub fn kernel(&self) -> &K {
        &self.kernel
    }
}

/// Step A5: accept the merge of a child group into its parent supernode?
fn accept_merge(
    relax_zeros: usize,
    child_size: usize,
    child_degree: usize,
    child_zeros: usize,
    parent_size: usize,
    parent_degree: usize,
) -> Option<usize> {
    if relax_zeros == 0 {
        return None;
    }
    let extra = (parent_size + parent_degree)
        .checked_sub(child_degree)
        .expect("(L1): a child's pattern lies in its parent's columns and pattern");
    let new_zeros = child_size * extra;
    let total = child_zeros + new_zeros;
    let w = child_size + parent_size;
    if new_zeros == 0 || w <= relax_zeros {
        return Some(total);
    }
    let entries = w * (w + 1) / 2 + parent_degree * w;
    let permille = RELAX_CUTOFFS
        .iter()
        .find(|&&(cols, _)| w <= cols)
        .map_or(0, |&(_, z)| z);
    (total * 1000 <= permille * entries).then_some(total)
}

/// Scratch blocks of Algorithms U and D, reused across supernodes (grown,
/// never shrunk, inside one factorization).
struct Scratch<T: Real> {
    a: DenseBlock<T>,
    b: DenseBlock<T>,
    c: DenseBlock<T>,
    d: DenseBlock<T>,
}

impl<T: Real> Scratch<T> {
    fn new() -> Self {
        Self {
            a: DenseBlock::zeros(0, 0),
            b: DenseBlock::zeros(0, 0),
            c: DenseBlock::zeros(0, 0),
            d: DenseBlock::zeros(0, 0),
        }
    }
}

/// Sizes `blk` as `rows × cols` (new entries zero, old ones kept: the
/// caller overwrites what it reads).
fn fit<T: Real>(blk: &mut DenseBlock<T>, rows: usize, cols: usize) {
    blk.rows = rows;
    blk.cols = cols;
    blk.data.resize(rows * cols, T::ZERO);
}

/// A pivot failure inside a supernode, at a local column.
enum PivotError {
    Zero(usize),
    NotFinite(usize),
}

/// Step D1's pivot: the value to store as `d_k`, bumped if asked (step K3).
/// `Err(true)`: exactly zero without bumping; `Err(false)`: not finite.
#[inline]
fn pivot<T: Real>(d: T, sigma: i8, tau: Option<T>, beta: T, bumped: &mut usize) -> Result<T, bool> {
    if !d.is_finite() {
        return Err(false);
    }
    let sd = if sigma < 0 { -d } else { d };
    match tau {
        Some(t) => {
            if le(sd, t, T::ZERO) {
                *bumped += 1;
                return Ok(if sigma < 0 { -beta } else { beta });
            }
        }
        None => {
            if is_exact_zero(d) {
                return Err(true);
            }
        }
    }
    Ok(d)
}

/// Step D1 on the block `a`: columns `p..q`, rows up to `rows_end` (`q`
/// for a tile, `a.rows` for the simplicial path), the trailing columns
/// `k + 1..q` updated. `signs[k]`, `diag[k]` are indexed by the block's
/// column.
#[allow(clippy::too_many_arguments)] // the pieces of step D1, as in the chapter
fn ldl_scalar<T: Real>(
    a: &mut DenseBlock<T>,
    p: usize,
    q: usize,
    rows_end: usize,
    signs: &[i8],
    tau: Option<T>,
    beta: T,
    diag: &mut [T],
    bumped: &mut usize,
) -> Result<(), PivotError> {
    let nr = a.rows;
    let data = &mut a.data;
    for k in p..q {
        let ck = k * nr;
        let d = match pivot(data[ck + k], signs[k], tau, beta, bumped) {
            Ok(d) => d,
            Err(true) => return Err(PivotError::Zero(k)),
            Err(false) => return Err(PivotError::NotFinite(k)),
        };
        data[ck + k] = d;
        diag[k] = d;
        for x in &mut data[ck + k + 1..ck + rows_end] {
            *x = *x / d;
        }
        for j in k + 1..q {
            let t = data[ck + j] * d;
            let cj = j * nr;
            // a_ij -= l_ik (d l_jk) for i in j..rows_end.
            let (lo, hi) = data.split_at_mut(cj);
            let lk = &lo[ck + j..ck + rows_end];
            let aj = &mut hi[j..rows_end];
            for (x, &l) in aj.iter_mut().zip(lk) {
                *x -= l * t;
            }
        }
    }
    Ok(())
}

/// Copies rows `r0..r1` of column `col` of the block `a` (leading dimension
/// `a.rows`) into `dst`.
#[inline]
fn gather_col<T: Real>(a: &DenseBlock<T>, col: usize, r0: usize, r1: usize, dst: &mut [T]) {
    dst.copy_from_slice(&a.data[r0 + col * a.rows..r1 + col * a.rows]);
}

/// Writes `src` into rows `r0..r1` of column `col` of the block `a`.
#[inline]
fn scatter_col<T: Real>(a: &mut DenseBlock<T>, col: usize, r0: usize, r1: usize, src: &[T]) {
    a.data[r0 + col * a.rows..r1 + col * a.rows].copy_from_slice(src);
}

/// Algorithm D on the block of one supernode (`signs`, `diag` indexed by
/// the block's columns).
#[allow(clippy::too_many_arguments)] // steps D1–D3 in one place
fn factor_block<T: Real, K: DenseKernel<T>>(
    a: &mut DenseBlock<T>,
    signs: &[i8],
    tau: Option<T>,
    beta: T,
    diag: &mut [T],
    bumped: &mut usize,
    kernel_path: bool,
    ws: &mut Scratch<T>,
    kernel: &K,
) -> Result<(), PivotError> {
    let (nr, nc) = (a.rows, a.cols);
    if !kernel_path {
        return ldl_scalar(a, 0, nc, nr, signs, tau, beta, diag, bumped);
    }
    let mut p = 0;
    while p < nc {
        let q = (p + PANEL).min(nc);
        let w = q - p;
        // D1.
        ldl_scalar(a, p, q, q, signs, tau, beta, diag, bumped)?;
        if q < nr {
            // D2: L̃ = L₁₁ Δ₁ (lower, diagonal d_k), A₂₁ gathered, solved, written back.
            let tile = &mut ws.b;
            fit(tile, w, w);
            tile.data.fill(T::ZERO);
            for j in 0..w {
                let d = diag[p + j];
                tile.data[j + j * w] = d;
                for i in j + 1..w {
                    tile.data[i + j * w] = a.data[(p + i) + (p + j) * nr] * d;
                }
            }
            let m = nr - q;
            let below = &mut ws.a;
            fit(below, m, w);
            for j in 0..w {
                gather_col(a, p + j, q, nr, &mut below.data[j * m..(j + 1) * m]);
            }
            kernel.trsm_lower(tile, below, true);
            for j in 0..w {
                scatter_col(a, p + j, q, nr, &below.data[j * m..(j + 1) * m]);
            }
            if q < nc {
                // D3: W = L₂₁ Δ₁ (the rows q..nr), then the trailing block by panels.
                let wblk = &mut ws.b;
                fit(wblk, m, w);
                for j in 0..w {
                    let d = diag[p + j];
                    for (x, &l) in wblk.data[j * m..(j + 1) * m]
                        .iter_mut()
                        .zip(&a.data[q + (p + j) * nr..(p + j + 1) * nr])
                    {
                        *x = l * d;
                    }
                }
                let mut c0 = q;
                while c0 < nc {
                    let c1 = (c0 + PANEL).min(nc);
                    let wc = c1 - c0;
                    let rows = nr - c0;
                    // A ← W[c0 − q.., :] (rows × w).
                    let ablk = &mut ws.a;
                    fit(ablk, rows, w);
                    for j in 0..w {
                        ablk.data[j * rows..(j + 1) * rows]
                            .copy_from_slice(&wblk.data[(c0 - q) + j * m..(j + 1) * m]);
                    }
                    // B ← L₂₁[c0 − q..c1 − q, :] (wc × w), from the block itself.
                    let bblk = &mut ws.c;
                    fit(bblk, wc, w);
                    for j in 0..w {
                        gather_col(a, p + j, c0, c1, &mut bblk.data[j * wc..(j + 1) * wc]);
                    }
                    // C ← A[c0.., c0..c1] (rows × wc), updated, written back.
                    let cblk = &mut ws.d;
                    fit(cblk, rows, wc);
                    for j in 0..wc {
                        gather_col(a, c0 + j, c0, nr, &mut cblk.data[j * rows..(j + 1) * rows]);
                    }
                    kernel.gemm_sub(ablk, bblk, cblk);
                    for j in 0..wc {
                        scatter_col(a, c0 + j, c0, nr, &cblk.data[j * rows..(j + 1) * rows]);
                    }
                    c0 = c1;
                }
            }
        }
        p = q;
    }
    Ok(())
}

/// Algorithm U, kernel path: `L_s ← L_s − L_d[a.., :] Δ_d L_d[a..b, :]ᵀ`
/// (lower part), `loc` mapping permuted rows to the rows of `s`'s block.
#[allow(clippy::too_many_arguments)] // steps U1–U3
fn update_kernel<T: Real, K: DenseKernel<T>>(
    ld: &DenseBlock<T>,
    dd: &[T],
    pat_d: &[u32],
    a: usize,
    b: usize,
    ls: &mut DenseBlock<T>,
    s0: usize,
    loc: &[u32],
    ws: &mut Scratch<T>,
    kernel: &K,
) {
    let (nrd, ncd) = (ld.rows, ld.cols);
    let e = nrd - ncd;
    let (nm, nr) = (b - a, e - a);
    // U1.
    let ab = &mut ws.a;
    fit(ab, nr, ncd);
    for k in 0..ncd {
        gather_col(ld, k, ncd + a, nrd, &mut ab.data[k * nr..(k + 1) * nr]);
    }
    let bb = &mut ws.b;
    fit(bb, nm, ncd);
    for (k, &dk) in dd.iter().enumerate() {
        for (x, &l) in bb.data[k * nm..(k + 1) * nm]
            .iter_mut()
            .zip(&ld.data[k * nrd + ncd + a..k * nrd + ncd + b])
        {
            *x = l * dk;
        }
    }
    // U2.
    let cb = &mut ws.c;
    fit(cb, nr, nm);
    cb.data.fill(T::ZERO);
    kernel.gemm_sub(ab, bb, cb);
    // U3.
    let nrs = ls.rows;
    for j in 0..nm {
        let cj = pat_d[a + j] as usize - s0;
        let col = &cb.data[j * nr..(j + 1) * nr];
        let dst = &mut ls.data[cj * nrs..(cj + 1) * nrs];
        for i in j..nr {
            dst[loc[pat_d[a + i] as usize] as usize] += col[i];
        }
    }
}

/// Algorithm U, simplicial path (plain loops, column `k` of `L_d` read
/// once).
#[allow(clippy::too_many_arguments)] // the same operands as the kernel path
fn update_scalar<T: Real>(
    ld: &DenseBlock<T>,
    dd: &[T],
    pat_d: &[u32],
    a: usize,
    b: usize,
    ls: &mut DenseBlock<T>,
    s0: usize,
    loc: &[u32],
) {
    let (nrd, ncd) = (ld.rows, ld.cols);
    let e = nrd - ncd;
    let nrs = ls.rows;
    for (k, &dk) in dd.iter().enumerate() {
        let col = &ld.data[k * nrd + ncd..(k + 1) * nrd];
        for j in a..b {
            let t = col[j] * dk;
            let cj = pat_d[j] as usize - s0;
            let dst = &mut ls.data[cj * nrs..(cj + 1) * nrs];
            for i in j..e {
                dst[loc[pat_d[i] as usize] as usize] -= col[i] * t;
            }
        }
    }
}

/// `Σ a_i b_i` with four independent partial sums (the dependent chain of a
/// single accumulator bounds the backward sweep of Algorithm S by the
/// latency of the fused multiply-add, not by memory); the order of the
/// additions is fixed, so the result is deterministic.
#[inline]
fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    let mut acc = [T::ZERO; 4];
    let (ca, ra) = a.as_chunks::<4>();
    let (cb, rb) = b.as_chunks::<4>();
    for (pa, pb) in ca.iter().zip(cb) {
        for k in 0..4 {
            acc[k] += pa[k] * pb[k];
        }
    }
    let mut tail = T::ZERO;
    for (&p, &q) in ra.iter().zip(rb) {
        tail += p * q;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// `Σ_q l_q x_{idx[q] − base}`, as [`dot`] with a gathered second operand.
#[inline]
fn gather_dot<T: Real>(l: &[T], idx: &[u32], x: &[T], base: usize) -> T {
    let mut acc = [T::ZERO; 4];
    let (cl, rl) = l.as_chunks::<4>();
    let (ci, ri) = idx.as_chunks::<4>();
    for (pl, pi) in cl.iter().zip(ci) {
        for k in 0..4 {
            acc[k] += pl[k] * x[pi[k] as usize - base];
        }
    }
    let mut tail = T::ZERO;
    for (&p, &i) in rl.iter().zip(ri) {
        tail += p * x[i as usize - base];
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

impl<T: Real, K: DenseKernel<T>> KktSolver<T> for SupernodalLdl<T, K> {
    type Symbolic = SymbolicLdl;
    type Numeric = NumericLdl<T>;

    #[allow(clippy::too_many_lines)] // steps A1–A10 in order, as in the chapter
    fn analyze(
        &self,
        pattern: &Pattern,
        form: KktForm,
        ordering: &dyn Ordering,
    ) -> Result<SymbolicLdl, KktError> {
        // A1.
        let lower = lower_with_diagonal(pattern, form)?;
        let n = lower.n_cols();
        let full = lower.symmetric()?;
        // A2.
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
        let (parent, counts) = etree_counts(&full, &perm);
        let none = usize::MAX;
        let mut nchild = vec![0_u32; n];
        for &p in &parent {
            if p != none {
                nchild[p] += 1;
            }
        }
        // A3′.
        let lnz: usize = counts.iter().sum();
        let flops: usize = counts.iter().map(|&c| c + c * (c + 1) / 2).sum();
        let simplicial = le(
            count_f64(flops),
            self.opts.supernodal_switch * count_f64(lnz),
            0.0,
        );
        let relax = if simplicial { 0 } else { self.opts.relax_zeros };
        // A4.
        let mut fs: Vec<usize> = Vec::with_capacity(n + 1);
        for j in 0..n {
            if j == 0 || !(parent[j - 1] == j && nchild[j] == 1 && counts[j - 1] == counts[j] + 1) {
                fs.push(j);
            }
        }
        fs.push(n);
        let nf = fs.len() - 1;
        let mut fs_of = vec![0_usize; n];
        for f in 0..nf {
            fs_of[fs[f]..fs[f + 1]].fill(f);
        }
        let fdeg = |f: usize| counts[fs[f + 1] - 1] - 1;
        let fparent = |f: usize| {
            let p = parent[fs[f + 1] - 1];
            if p == none { none } else { fs_of[p] }
        };
        // A5.
        let mut bounds: Vec<u32> = vec![0];
        let mut zeros_total = 0_usize;
        let mut g_size = if nf > 0 { fs[1] - fs[0] } else { 0 };
        let mut g_zeros = 0_usize;
        for f in 0..nf {
            let merged = if f + 1 < nf && fparent(f) == f + 1 {
                accept_merge(
                    relax,
                    g_size,
                    fdeg(f),
                    g_zeros,
                    fs[f + 2] - fs[f + 1],
                    fdeg(f + 1),
                )
            } else {
                None
            };
            if let Some(z) = merged {
                g_size += fs[f + 2] - fs[f + 1];
                g_zeros = z;
            } else {
                bounds.push(idx32(fs[f + 1]));
                zeros_total += g_zeros;
                if f + 1 < nf {
                    g_size = fs[f + 2] - fs[f + 1];
                    g_zeros = 0;
                }
            }
        }
        let ns = bounds.len() - 1;
        let cols = |s: usize| (bounds[s] as usize, bounds[s + 1] as usize);
        let mut super_of = vec![NONE; n];
        for s in 0..ns {
            let (s0, s1) = cols(s);
            super_of[s0..s1].fill(idx32(s));
        }
        let sparent: Vec<u32> = (0..ns)
            .map(|s| {
                let p = parent[cols(s).1 - 1];
                if p == none { NONE } else { super_of[p] }
            })
            .collect();
        let mut child_ptr = vec![0_usize; ns + 1];
        for &p in &sparent {
            if p != NONE {
                child_ptr[p as usize + 1] += 1;
            }
        }
        for s in 0..ns {
            child_ptr[s + 1] += child_ptr[s];
        }
        let mut child_idx = vec![0_u32; child_ptr[ns]];
        let mut next = child_ptr.clone();
        for (c, &p) in sparent.iter().enumerate() {
            if p != NONE {
                child_idx[next[p as usize]] = idx32(c);
                next[p as usize] += 1;
            }
        }
        // A6.
        let mut row_ptr = vec![0_usize; ns + 1];
        let mut row_idx: Vec<u32> = Vec::new();
        let mut mark = vec![NONE; n];
        for s in 0..ns {
            let (s0, s1) = cols(s);
            let start = row_idx.len();
            let tag = idx32(s);
            for j in s0..s1 {
                for &i in full.col(perm.fwd[j] as usize) {
                    let r = perm.inv[i as usize] as usize;
                    if r >= s1 && mark[r] != tag {
                        mark[r] = tag;
                        row_idx.push(idx32(r));
                    }
                }
            }
            for &c in &child_idx[child_ptr[s]..child_ptr[s + 1]] {
                let c = c as usize;
                for q in row_ptr[c]..row_ptr[c + 1] {
                    let r = row_idx[q];
                    if r as usize >= s1 && mark[r as usize] != tag {
                        mark[r as usize] = tag;
                        row_idx.push(r);
                    }
                }
            }
            row_idx[start..].sort_unstable();
            row_ptr[s + 1] = row_idx.len();
            debug_assert_eq!(row_idx.len() - start + 1, counts[s1 - 1], "(L2)");
        }
        // A7.
        let mut upd_ptr = vec![0_usize; ns + 1];
        for d in 0..ns {
            let mut q = row_ptr[d];
            while q < row_ptr[d + 1] {
                let s = super_of[row_idx[q] as usize] as usize;
                let s1 = bounds[s + 1];
                while q < row_ptr[d + 1] && row_idx[q] < s1 {
                    q += 1;
                }
                upd_ptr[s + 1] += 1;
            }
        }
        for s in 0..ns {
            upd_ptr[s + 1] += upd_ptr[s];
        }
        let nu = upd_ptr[ns];
        let (mut upd_d, mut upd_from, mut upd_to) =
            (vec![0_u32; nu], vec![0_u32; nu], vec![0_u32; nu]);
        let mut next = upd_ptr.clone();
        for d in 0..ns {
            let mut q = row_ptr[d];
            while q < row_ptr[d + 1] {
                let s = super_of[row_idx[q] as usize] as usize;
                let s1 = bounds[s + 1];
                let a = q - row_ptr[d];
                while q < row_ptr[d + 1] && row_idx[q] < s1 {
                    q += 1;
                }
                let u = next[s];
                upd_d[u] = idx32(d);
                upd_from[u] = idx32(a);
                upd_to[u] = idx32(q - row_ptr[d]);
                next[s] += 1;
            }
        }
        // A8.
        let nrows = |s: usize| {
            let (s0, s1) = cols(s);
            s1 - s0 + row_ptr[s + 1] - row_ptr[s]
        };
        let nnz0 = lower.nnz();
        let (mut sc_super, mut sc_off) = (vec![0_u32; nnz0], vec![0_usize; nnz0]);
        for j in 0..n {
            let b = perm.inv[j] as usize;
            for p in lower.col_ptr[j]..lower.col_ptr[j + 1] {
                let a = perm.inv[lower.row_idx[p] as usize] as usize;
                let (col, row) = (a.min(b), a.max(b));
                let s = super_of[col] as usize;
                let (s0, s1) = cols(s);
                let local_row = if row < s1 {
                    row - s0
                } else {
                    let pat = &row_idx[row_ptr[s]..row_ptr[s + 1]];
                    s1 - s0
                        + pat
                            .binary_search(&idx32(row))
                            .expect("(L1): the row is in R_s")
                };
                sc_super[p] = idx32(s);
                sc_off[p] = local_row + (col - s0) * nrows(s);
            }
        }
        // A9.
        let nnz_l: usize = (0..ns).map(|s| nrows(s) * (cols(s).1 - cols(s).0)).sum();
        let etree: Vec<u32> = parent
            .iter()
            .map(|&p| if p == none { NONE } else { idx32(p) })
            .collect();
        let shared = Arc::new(Structure {
            n,
            perm: perm.clone(),
            supernodes: bounds.clone(),
            row_ptr: row_ptr.clone(),
            row_idx: row_idx.clone(),
        });
        let sym = SymbolicLdl {
            perm,
            etree,
            supernodes: bounds,
            row_ptr,
            row_idx,
            nnz_l,
            form,
            lower,
            sc_super,
            sc_off,
            upd_ptr,
            upd_d,
            upd_from,
            upd_to,
            relax_zeros: zeros_total,
            counts,
            simplicial,
            shared,
        };
        debug_assert_eq!(sym.check_invariants(), Ok(()));
        Ok(sym)
    }

    #[allow(clippy::too_many_lines)] // steps N1–N5 in order, as in the chapter
    fn factorize(
        &self,
        symbolic: &SymbolicLdl,
        k: &CscMatrix<T>,
        reg: Regularization<T>,
    ) -> Result<(NumericLdl<T>, Inertia), KktError> {
        check_regularization(&reg)?;
        let st = &symbolic.shared;
        let n = st.n;
        let ns = st.n_supernodes();
        // N1.
        let (v, scale) = assemble_regularized(&symbolic.lower, symbolic.form, k, &reg)?;
        // N2.
        let mut blocks: Vec<DenseBlock<T>> = (0..ns)
            .map(|s| {
                let (s0, s1) = st.cols(s);
                DenseBlock::zeros(s1 - s0 + st.pattern(s).len(), s1 - s0)
            })
            .collect();
        for (p, &x) in v.iter().enumerate() {
            blocks[symbolic.sc_super[p] as usize].data[symbolic.sc_off[p]] = x;
        }
        let signs_orig = pivot_signs(symbolic.form, n);
        let fwd = &st.perm.fwd;
        let signs: Vec<i8> = (0..n).map(|k| signs_orig[fwd[k] as usize]).collect();
        let mut diag = vec![T::ZERO; n];
        let mut loc = vec![NONE; n];
        let mut ws = Scratch::new();
        let mut bumped = 0_usize;
        let min_cols = if symbolic.simplicial {
            usize::MAX
        } else {
            self.opts.min_supernode_cols
        };
        // N3.
        for s in 0..ns {
            let (s0, s1) = st.cols(s);
            let nc = s1 - s0;
            let pat = st.pattern(s);
            for c in 0..nc {
                loc[s0 + c] = idx32(c);
            }
            for (q, &r) in pat.iter().enumerate() {
                loc[r as usize] = idx32(nc + q);
            }
            let kernel_path = nc >= min_cols;
            let (head, tail) = blocks.split_at_mut(s);
            let ls = &mut tail[0];
            let (dhead, dtail) = diag.split_at_mut(s0);
            // N3b.
            for u in symbolic.upd_ptr[s]..symbolic.upd_ptr[s + 1] {
                let d = symbolic.upd_d[u] as usize;
                let (a, b) = (symbolic.upd_from[u] as usize, symbolic.upd_to[u] as usize);
                let (d0, d1) = st.cols(d);
                let (ld, dd, pat_d) = (&head[d], &dhead[d0..d1], st.pattern(d));
                if kernel_path && d1 - d0 >= min_cols {
                    update_kernel(ld, dd, pat_d, a, b, ls, s0, &loc, &mut ws, &self.kernel);
                } else {
                    update_scalar(ld, dd, pat_d, a, b, ls, s0, &loc);
                }
            }
            // N3c.
            if let Err(e) = factor_block(
                ls,
                &signs[s0..s1],
                reg.pivot_tol,
                reg.bump,
                &mut dtail[..nc],
                &mut bumped,
                kernel_path,
                &mut ws,
                &self.kernel,
            ) {
                return Err(match e {
                    PivotError::Zero(c) => KktError::ZeroPivot {
                        position: fwd[s0 + c] as usize,
                    },
                    PivotError::NotFinite(c) => KktError::NotFinite {
                        position: fwd[s0 + c] as usize,
                    },
                });
            }
            // N3d.
            for c in 0..nc {
                loc[s0 + c] = NONE;
            }
            for &r in pat {
                loc[r as usize] = NONE;
            }
        }
        // N4: L̂ = S⁻¹ L S, Δ̂ = S⁻¹ Δ S⁻¹ (exact, powers of two): the factor
        // of P K_reg Pᵀ in the caller's units, so that the solves need no
        // scaling pass.
        for (s, blk) in blocks.iter_mut().enumerate() {
            let (s0, s1) = st.cols(s);
            let nc = s1 - s0;
            let nr = blk.rows;
            let pat = st.pattern(s);
            for c in 0..nc {
                let sc = scale[fwd[s0 + c] as usize];
                let col = &mut blk.data[c * nr..(c + 1) * nr];
                col[c] = col[c] / (sc * sc);
                diag[s0 + c] = col[c];
                for i in c + 1..nc {
                    col[i] = col[i] * sc / scale[fwd[s0 + i] as usize];
                }
                for (x, &r) in col[nc..].iter_mut().zip(pat) {
                    *x = *x * sc / scale[fwd[r as usize] as usize];
                }
            }
        }
        // N5 (step K3′, (S10)): the lower parts of the blocks.
        for (s, blk) in blocks.iter().enumerate() {
            let (s0, _) = st.cols(s);
            for c in 0..blk.cols {
                if blk.data[c + c * blk.rows..(c + 1) * blk.rows]
                    .iter()
                    .any(|x| !x.is_finite())
                {
                    return Err(KktError::NotFinite {
                        position: fwd[s0 + c] as usize,
                    });
                }
            }
        }
        // N6 (S6).
        let found = count_inertia(diag.iter().copied(), core::iter::empty::<(T, T, T)>());
        let inertia = check_inertia(found, symbolic.form.expected_inertia(n))?;
        debug_assert_eq!(inertia, symbolic.form.expected_inertia(n), "(S6)");
        Ok((
            NumericLdl {
                blocks,
                diag,
                pivot_2x2: vec![false; n],
                inertia,
                bumped,
                scale,
                shared: Arc::clone(st),
                scratch: SolveScratch(Mutex::new(vec![T::ZERO; n])),
            },
            inertia,
        ))
    }

    fn solve(&self, numeric: &NumericLdl<T>, x: &mut [T]) {
        let st = &numeric.shared;
        let n = st.n;
        assert_eq!(x.len(), n, "solve: x has the wrong length");
        // The factor's own work vector; a temporary one only if another
        // thread holds it (or a panic poisoned it).
        let mut held = numeric.scratch.0.try_lock().ok();
        let mut temp = Vec::new();
        let y: &mut [T] = match held.as_deref_mut() {
            Some(v) if v.len() == n => v.as_mut_slice(),
            _ => {
                temp.resize(n, T::ZERO);
                temp.as_mut_slice()
            }
        };
        let fwd = &st.perm.fwd;
        for (yk, &i) in y.iter_mut().zip(fwd) {
            *yk = x[i as usize];
        }
        let ns = st.n_supernodes();
        // S1.
        for s in 0..ns {
            let (s0, s1) = st.cols(s);
            let nc = s1 - s0;
            let blk = &numeric.blocks[s];
            let nr = blk.rows;
            let pat = st.pattern(s);
            if nc == 1 {
                let xj = y[s0];
                for (&r, &l) in pat.iter().zip(&blk.data[1..]) {
                    y[r as usize] -= l * xj;
                }
                continue;
            }
            for j in 0..nc {
                let xj = y[s0 + j];
                let col = &blk.data[j * nr..(j + 1) * nr];
                let (ys, rest) = y.split_at_mut(s1);
                for (yi, &l) in ys[s0 + j + 1..].iter_mut().zip(&col[j + 1..nc]) {
                    *yi -= l * xj;
                }
                for (&r, &l) in pat.iter().zip(&col[nc..]) {
                    rest[r as usize - s1] -= l * xj;
                }
            }
        }
        // S2.
        for (yk, &dk) in y.iter_mut().zip(&numeric.diag) {
            *yk = *yk / dk;
        }
        // S3.
        for s in (0..ns).rev() {
            let (s0, s1) = st.cols(s);
            let nc = s1 - s0;
            let blk = &numeric.blocks[s];
            let nr = blk.rows;
            let pat = st.pattern(s);
            if nc == 1 {
                let mut acc = T::ZERO;
                for (&r, &l) in pat.iter().zip(&blk.data[1..]) {
                    acc += l * y[r as usize];
                }
                y[s0] -= acc;
                continue;
            }
            for j in (0..nc).rev() {
                let col = &blk.data[j * nr..(j + 1) * nr];
                let (ys, rest) = y.split_at(s1);
                let inner = dot(&col[j + 1..nc], &ys[s0 + j + 1..]);
                let outer = gather_dot(&col[nc..], pat, rest, s1);
                y[s0 + j] -= inner + outer;
            }
        }
        for (&yk, &i) in y.iter().zip(fwd) {
            x[i as usize] = yk;
        }
    }

    fn nnz_factor(&self, numeric: &NumericLdl<T>) -> usize {
        numeric.blocks.iter().map(|b| b.data.len()).sum()
    }

    fn name(&self) -> &'static str {
        "supernodal-ldl"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kkt::sym_lower_matvec;
    use crate::real::{approx_eq, le};
    use crate::tests::kkt_gen::{kkt, rhs};
    use crate::{Amd, Col, Natural, Refinement, Row, ShiftScale};
    use proptest::prelude::*;

    type Naive = SupernodalLdl<f64, NaiveDense>;

    /// The three option sets the tests exercise: the defaults, every
    /// supernode fundamental and on the kernel path, every supernode on the
    /// simplicial path.
    fn variants() -> [SupernodalOptions; 3] {
        [
            SupernodalOptions::default(),
            SupernodalOptions {
                relax_zeros: 0,
                min_supernode_cols: 1,
                supernodal_switch: 0.0,
            },
            SupernodalOptions {
                relax_zeros: 4,
                min_supernode_cols: 1,
                supernodal_switch: 0.0,
            },
        ]
    }

    /// Factorizes `k` with `reg` and refines against the unregularized `K`
    /// with the right-hand side `K x*` for `x* = b`; returns the inertia,
    /// the refined residual, the bump count and the solution.
    fn solve_ok<K: DenseKernel<f64>>(
        s: &SupernodalLdl<f64, K>,
        k: &CscMatrix,
        form: KktForm,
        reg: Regularization,
        b: &[f64],
    ) -> (Inertia, f64, usize, Vec<f64>) {
        let sym = s.analyze(&k.pattern(), form, &Amd::default()).unwrap();
        assert_eq!(sym.check_invariants(), Ok(()));
        let (num, inertia) = s.factorize(&sym, k, reg).unwrap();
        let mut rhs = vec![0.0; b.len()];
        sym_lower_matvec(k, b, &mut rhs);
        let mut x = vec![0.0; b.len()];
        let rep = s.solve_refined(&num, k, &rhs, &mut x, Refinement::default());
        (inertia, rep.residual, num.bumped, x)
    }

    #[test]
    fn augmented_and_normal_systems_solve_with_the_expected_inertia() {
        for (seed, spread) in [(1, 0.0), (2, 4.0), (3, 8.0)] {
            let (aug, normal, form) = kkt(12, 30, spread, seed);
            for opts in variants() {
                let s = Naive::new(NaiveDense, opts);
                let (i, res, _, _) =
                    solve_ok(&s, &aug, form, Regularization::default(), &rhs(42, seed));
                assert_eq!(i, form.expected_inertia(42));
                assert!(le(res, 1e-10, 0.0), "aug seed {seed} {opts:?}: {res}");
                let (i, res, _, _) = solve_ok(
                    &s,
                    &normal,
                    KktForm::NormalEquations,
                    Regularization::default(),
                    &rhs(12, seed),
                );
                assert_eq!(i, KktForm::NormalEquations.expected_inertia(12));
                assert!(le(res, 1e-10, 0.0), "normal seed {seed} {opts:?}: {res}");
            }
        }
    }

    /// (L3) and (L6): the stored sizes against `Pattern::cholesky_nnz`, and
    /// two factorizations equal bit for bit; the kernel and the simplicial
    /// paths agree to rounding.
    #[test]
    fn sizes_determinism_and_paths() {
        let (aug, _, form) = kkt(20, 60, 3.0, 11);
        let full = aug.pattern().symmetric().unwrap();
        let mut sols = Vec::new();
        for opts in variants() {
            let s = Naive::new(NaiveDense, opts);
            let sym = s.analyze(&aug.pattern(), form, &Amd::default()).unwrap();
            let structural = full.cholesky_nnz(&sym.perm).unwrap();
            let lower: usize = (0..sym.n_supernodes())
                .map(|t| {
                    let nc = (sym.supernodes[t + 1] - sym.supernodes[t]) as usize;
                    let deg = sym.row_ptr[t + 1] - sym.row_ptr[t];
                    nc * (nc + 1) / 2 + deg * nc
                })
                .sum();
            assert_eq!(lower, structural + sym.relaxation_zeros());
            if opts.relax_zeros == 0 {
                assert_eq!(sym.relaxation_zeros(), 0);
                assert_eq!(lower, structural);
            }
            let (n1, _) = s.factorize(&sym, &aug, Regularization::default()).unwrap();
            let (n2, _) = s.factorize(&sym, &aug, Regularization::default()).unwrap();
            assert_eq!(n1.blocks, n2.blocks);
            assert_eq!(s.nnz_factor(&n1), sym.nnz_l);
            let b = rhs(80, 11);
            let mut x = b.clone();
            s.solve(&n1, &mut x);
            let mut y = b.clone();
            s.solve(&n2, &mut y);
            assert_eq!(x, y);
            sols.push(x);
        }
        for x in &sols[1..] {
            assert!(
                x.iter().zip(&sols[0]).all(|(&a, &b)| approx_eq(a, b, 1e-8)),
                "the paths disagree"
            );
        }
    }

    #[test]
    fn bumps_are_counted_exactly_with_both_signs() {
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
        for opts in variants() {
            let s = Naive::new(NaiveDense, opts);
            let sym = s.analyze(&k.pattern(), form, &Natural).unwrap();
            let (num, inertia) = s.factorize(&sym, &k, reg).unwrap();
            assert_eq!(inertia, form.expected_inertia(2));
            assert_eq!(num.bumped, 1, "{opts:?}");
            let sym = s
                .analyze(&z.pattern(), KktForm::NormalEquations, &Natural)
                .unwrap();
            let (num, _) = s.factorize(&sym, &z, reg).unwrap();
            assert_eq!(num.bumped, 2, "{opts:?}"); // pivots 1 (zero row) and 2 (1 − 1 = 0)
        }
        // Many bumps of both signs: an augmented system with the primal
        // block zeroed out (every primal pivot is bumped to −β) and a dual
        // block whose Schur complement has pivots of both signs; the count
        // equals the number of pivots equal to ±β exactly (β chosen so that
        // no genuine pivot can hit it) on both paths.
        let (aug, _, form) = kkt(15, 40, 0.0, 21);
        let mut t = Vec::new();
        for j in 0..55 {
            for p in aug.col_ptr[j]..aug.col_ptr[j + 1] {
                if aug.row_idx[p] as usize != j {
                    t.push((Row(aug.row_idx[p]), Col::from_usize(j), aug.val[p]));
                }
            }
        }
        let zeroed = CscMatrix::from_triplets(55, 55, &t).unwrap();
        let reg = Regularization {
            primal: 0.0,
            dual: 0.0,
            pivot_tol: Some(1e-3),
            bump: 3.0e-7,
            scale: ShiftScale::Equilibrated,
        };
        for opts in variants() {
            let s = Naive::new(NaiveDense, opts);
            let sym = s.analyze(&zeroed.pattern(), form, &Amd::default()).unwrap();
            let (num, inertia) = s.factorize(&sym, &zeroed, reg).unwrap();
            assert_eq!(inertia, form.expected_inertia(55));
            let by_value = num
                .diag
                .iter()
                .enumerate()
                .filter(|&(k, &d)| {
                    let sk = num.scale()[num.perm().fwd[k] as usize];
                    approx_eq((d * sk * sk).abs(), 3.0e-7, 0.0)
                })
                .count();
            assert_eq!(num.bumped, by_value, "{opts:?}");
            assert!(num.bumped >= 40, "{opts:?}: {}", num.bumped);
        }
    }

    #[test]
    fn s6_errors_zero_pivot_and_wrong_inertia() {
        let s = Naive::default();
        // K = [−1, 0; 0, 0]: the dual pivot is exactly zero.
        let k = CscMatrix::from_triplets(2, 2, &[(Row(0), Col(0), -1.0)]).unwrap();
        let form = KktForm::Augmented {
            n_primal: 1,
            n_dual: 1,
        };
        let sym = s.analyze(&k.pattern(), form, &Natural).unwrap();
        assert_eq!(
            s.factorize(&sym, &k, Regularization::none()).err(),
            Some(KktError::ZeroPivot { position: 1 })
        );
        let (_, inertia) = s.factorize(&sym, &k, Regularization::default()).unwrap();
        assert_eq!(inertia, form.expected_inertia(2));
        // Indefinite "normal equations": wrong inertia without bumping,
        // the expected one (with a bump) with it.
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
        assert_eq!(num.bumped, 1);
        // A pivot equal to δ is not a bump.
        let (aug, _, form) = kkt(3, 5, 0.0, 4);
        let sym = s.analyze(&aug.pattern(), form, &Natural).unwrap();
        let (num, _) = s.factorize(&sym, &aug, Regularization::default()).unwrap();
        assert_eq!(num.bumped, 0);
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
        for opts in variants() {
            let s = Naive::new(NaiveDense, opts);
            let sym = s
                .analyze(&k.pattern(), KktForm::NormalEquations, &Natural)
                .unwrap();
            assert!(
                matches!(
                    s.factorize(&sym, &k, reg),
                    Err(KktError::NotFinite { position: 1 | 2 })
                ),
                "{opts:?}"
            );
            // A NaN in K never reaches the factorization (step K1′).
            let mut nan = t(1.0);
            nan.val[0] = f64::NAN;
            assert!(matches!(
                s.factorize(&sym, &nan, Regularization::default()),
                Err(KktError::Structure(_))
            ));
        }
    }

    #[test]
    fn structural_errors() {
        let s = Naive::default();
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
        // analyzed one; the empty system is fine.
        let (aug, _, form) = kkt(4, 8, 1.0, 5);
        let sym = s.analyze(&aug.pattern(), form, &Amd::default()).unwrap();
        let mut thin = aug.clone();
        thin.row_idx.remove(1);
        thin.val.remove(1);
        for p in &mut thin.col_ptr[1..] {
            *p -= 1;
        }
        assert!(s.factorize(&sym, &thin, Regularization::default()).is_ok());
        let empty = CscMatrix::<f64>::zeros(0, 0);
        let sym = s
            .analyze(&empty.pattern(), KktForm::NormalEquations, &Amd::default())
            .unwrap();
        let (num, _) = s
            .factorize(&sym, &empty, Regularization::default())
            .unwrap();
        let mut x: [f64; 0] = [];
        s.solve(&num, &mut x);
        assert_eq!(s.nnz_factor(&num), 0);
    }

    /// The generic code with a second scalar: `f32` (the test-only
    /// instance of Chapter 5.6).
    #[test]
    fn a_second_scalar_type_factorizes_and_solves() {
        let (aug, normal, form) = kkt(8, 20, 1.0, 7);
        let to32 = |k: &CscMatrix| {
            CscMatrix::<f32>::from_raw_parts(
                k.n_rows,
                k.col_ptr.clone(),
                k.row_idx.clone(),
                #[allow(clippy::cast_possible_truncation)]
                k.val.iter().map(|&v| v as f32).collect(),
            )
            .unwrap()
        };
        for opts in variants() {
            let s = SupernodalLdl::<f32, NaiveDense>::new(NaiveDense, opts);
            for (k, form) in [
                (to32(&aug), form),
                (to32(&normal), KktForm::NormalEquations),
            ] {
                let n = k.n_rows;
                let sym = s.analyze(&k.pattern(), form, &Amd::default()).unwrap();
                let (num, inertia) = s
                    .factorize(&sym, &k, Regularization::<f32>::default())
                    .unwrap();
                assert_eq!(inertia, form.expected_inertia(n));
                let xs: Vec<f32> = (0..n)
                    .map(|i| if i % 2 == 0 { 1.0 } else { -0.5 })
                    .collect();
                let mut b = vec![0.0_f32; n];
                sym_lower_matvec(&k, &xs, &mut b);
                let mut x = vec![0.0_f32; n];
                let rep = s.solve_refined(
                    &num,
                    &k,
                    &b,
                    &mut x,
                    Refinement {
                        max_iters: 10,
                        target_residual: 1e-5,
                    },
                );
                assert!(le(rep.residual, 1e-4, 0.0), "{opts:?} {form:?}: {rep:?}");
            }
        }
    }

    /// The same systems through `FaerLdl` (forced supernodal) and through
    /// `SupernodalLdl<f64, FaerDense>`: the same stored size (the same
    /// amalgamation rule under the same permutation, step A5), the same
    /// inertia, solutions equal to rounding.
    #[cfg(feature = "faer")]
    #[test]
    fn agrees_with_faer() {
        use crate::{FaerDense, FaerLdl};
        for (seed, spread, m, n) in [(1, 0.0, 12, 30), (2, 4.0, 25, 60), (3, 8.0, 40, 90)] {
            let (aug, normal, form) = kkt(m, n, spread, seed);
            for (k, form) in [(aug, form), (normal, KktForm::NormalEquations)] {
                let nn = k.n_rows;
                let faer = FaerLdl {
                    supernodal: Some(true),
                };
                let ours = SupernodalLdl::<f64, FaerDense>::new(
                    FaerDense,
                    SupernodalOptions {
                        supernodal_switch: 0.0,
                        ..SupernodalOptions::default()
                    },
                );
                let fs = faer.analyze(&k.pattern(), form, &Amd::default()).unwrap();
                let os = ours.analyze(&k.pattern(), form, &Amd::default()).unwrap();
                assert_eq!(fs.perm(), &os.perm);
                assert_eq!(fs.nnz_factor(), os.nnz_l, "seed {seed} {form:?}");
                let (fn_, fi) = faer.factorize(&fs, &k, Regularization::default()).unwrap();
                let (on, oi) = ours.factorize(&os, &k, Regularization::default()).unwrap();
                assert_eq!(fi, oi);
                assert_eq!(fn_.bumped(), on.bumped);
                let b = rhs(nn, seed);
                let mut x = b.clone();
                let mut y = b.clone();
                faer.solve(&fn_, &mut x);
                ours.solve(&on, &mut y);
                let scale = x.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
                assert!(
                    x.iter()
                        .zip(&y)
                        .all(|(&a, &b)| approx_eq(a, b, 1e-9 * scale)),
                    "seed {seed} {form:?}"
                );
                // And the naive kernel agrees with faer's on the kernel path.
                let naive = Naive::new(
                    NaiveDense,
                    SupernodalOptions {
                        min_supernode_cols: 1,
                        supernodal_switch: 0.0,
                        ..SupernodalOptions::default()
                    },
                );
                let ns = naive.analyze(&k.pattern(), form, &Amd::default()).unwrap();
                let (nn_, _) = naive.factorize(&ns, &k, Regularization::default()).unwrap();
                let mut z = b.clone();
                naive.solve(&nn_, &mut z);
                assert!(
                    y.iter()
                        .zip(&z)
                        .all(|(&a, &b)| approx_eq(a, b, 1e-9 * scale))
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        /// (S6) and the refined residual on random KKT systems in the three
        /// IPM regimes, with every option set.
        #[test]
        fn random_kkt_systems(m in 1_usize..15, extra in 0_usize..20, regime in 0_usize..3, v in 0_usize..3, seed in any::<u64>()) {
            let n = m + extra;
            let spread = [0.0, 4.0, 9.0][regime];
            let s = Naive::new(NaiveDense, variants()[v]);
            let (aug, normal, form) = kkt(m, n, spread, seed);
            let (i, res, _, _) = solve_ok(&s, &aug, form, Regularization::default(), &rhs(n + m, seed));
            prop_assert_eq!(i, form.expected_inertia(n + m));
            prop_assert!(le(res, 1e-10, 0.0), "augmented: {}", res);
            let (i, _, _, _) = solve_ok(&s, &normal, KktForm::NormalEquations, Regularization::default(), &rhs(m, seed));
            prop_assert_eq!(i, KktForm::NormalEquations.expected_inertia(m));
        }
    }
}
