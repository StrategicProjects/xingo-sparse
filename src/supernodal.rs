//! # Chapter 5.17 — `SupernodalLdl`: the own supernodal `LDLᵀ`
//!
//! *New with task L1.6a (PLAN.md trilha L; the contract's §5.7, decision
//! D10 and D-0098). Part 1 (L1.6a): `1×1` pivots for the two regularized
//! systems. Part 2 (L1.6b): Algorithm M, the `LBLᵀ` with Bunch–Kaufman
//! `1×1`/`2×2` pivots inside each supernode and delayed pivots, for the
//! unregularized augmented system.*
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
//! ## Algorithm M (`LBLᵀ` of the unregularized augmented system)
//! *Task L1.6b.* An augmented system without regularization (`ρ = 0` or
//! `δ = 0`, and `τ = None`: the rule of Chapter 5.15, step N2) is
//! indefinite but not quasi-definite: its dual diagonal is zero, so a dual
//! column that is a leaf of the elimination tree has a pivot that is
//! exactly zero, and a column whose rows depend on the pivots already
//! taken inside its supernode has one that is zero up to rounding. The two
//! regularized systems take Algorithm N as before (they are quasi-definite
//! — Vanderbei 1995 — and need no pivoting; that path is untouched, bit
//! for bit); the unregularized augmented one takes this algorithm, which
//! replaces steps N3b–N3c and the solve.
//!
//! The pivoting is *intranodal* — the candidates for a pivot are the
//! columns of the supernode being factorized, the "fully summed" variables
//! of the multifrontal method, whose columns the block holds complete —
//! and the search is Bunch–Kaufman's (Algorithm B of Chapter 5.16,
//! [`DenseKernel::sytrf`]) on the diagonal block. A column that cannot be
//! pivoted there is *delayed* to the parent supernode (Duff & Reid 1983;
//! MA57, Duff 2004), where its remaining entries lie in any case ((L1):
//! the pattern of a child is contained in the columns and the pattern of
//! its parent), and tries again with more candidates; a column delayed
//! out of a root is a zero pivot of a singular matrix,
//! [`KktError::ZeroPivot`]. The structure of Algorithm A does not change:
//! a delayed column adds one column and one row to the block of the
//! parent (explicit zeros where it has no entry) and leaves its row of
//! `L` behind in the block it came from.
//!
//! For `s = 0, 1, …` in postorder, with `nc` columns and `deg = |R_s|`:
//! M1. [Receive.] The `nd` columns delayed into `s` by its children (step
//!     M5) are appended to its columns: the block becomes
//!     `(nc + nd + deg) × (nc + nd)`, the own entries keep their places
//!     (the pattern rows move down by `nd`), and each delayed column `k`
//!     brings its diagonal, its entries at the columns delayed after it by
//!     the same block (cross terms) and its entries at the pattern rows of
//!     that block, scattered by `loc`. A delayed column needs no update
//!     from the descendants of `s`: it arrives with the updates of its own
//!     subtree and of the pivots of the block that delayed it.
//! M2. [Updates.] Step N3b with Algorithm U, where `B ← L_d[a..b, :] B_d`
//!     has `B_d` block diagonal: for a `2×2` block `E = [e₁₁ e₂₁; e₂₁ e₂₂]`
//!     at the columns `k, k+1` of `d`, `B[:, k] = e₁₁ l_k + e₂₁ l_{k+1}` and
//!     `B[:, k+1] = e₂₁ l_k + e₂₂ l_{k+1}`; the rows of `d` are read past
//!     its own delayed rows (M6). The largest update any descendant makes
//!     to the diagonal of a column of `s` is kept as that column's
//!     reference scale for M3.
//! M3. [Pivot.] With `A₁₁` the diagonal block of the `w = nc + nd` active
//!     columns: `P A₁₁ Pᵀ = L₁₁ B₁ L₁₁ᵀ` by `sytrf` on a copy. A `1×1`
//!     pivot `d` with `|d| ≤ ζ m_j`, or a `2×2` block whose three entries
//!     are all `≤ ζ max(m_j, m_{j'})` (`ζ` = [`SupernodalOptions::zero_pivot_tol`],
//!     default `10⁻¹⁴`: a relative test, so that a pivot that is zero up to
//!     the rounding of the sum that produced it is caught, and a pivot
//!     that is merely small — the `10⁻¹⁰` of a primal column at the end
//!     of an interior-point method — is kept, as in Algorithm N), is a
//!     *zero pivot*: its columns are removed from the active set and M3 is
//!     repeated on the rest (a zero column of the Schur complement
//!     contributes nothing to the other pivots: the repetition only drops
//!     it). `w = 1` is the direct case, `d = a₀₀` against the scale it
//!     carries. Without a parent, a zero pivot is [`KktError::ZeroPivot`]
//!     at the original row of its column: nothing is left to pair it with,
//!     and the matrix is singular to working precision (measured on the
//!     Netlib systems: taking such a pivot instead gave factors with the
//!     expected inertia and residuals of `10¹⁵`, a silent failure); a pivot
//!     that is not finite is [`KktError::NotFinite`] there. The reference scale `m_j`
//!     is the largest term of the sums that produced the pivot — its
//!     diagonal as the block holds it and the largest update a descendant
//!     made to it (M2: a dual column coupled only to pairs below it
//!     receives updates that cancel to rounding noise, and the noise
//!     must not pass for a pivot); for a column delayed into
//!     this block, the scale it was flagged with where it came from and
//!     the terms of the Schur complement step M5 formed for it (a
//!     carried diagonal that is already rounding noise must not be judged
//!     against itself); and the terms `l_jk b_k l_jk` of the pivots `k`
//!     before it in the block. Measured on the random systems of the
//!     tests: without the carried scale a dual leaf whose diagonal arrived
//!     as `3·10⁻¹⁶` was taken as a pivot and the factor was wrong by
//!     `10⁻²`. The scale is a bound on the rounding noise of the sum, not
//!     an uncertainty propagated through the pivots: a version that let a
//!     pivot hand its own scale to what it updates compounded the scales
//!     to `10⁶⁵` on stair (Netlib) and flagged a pivot of `1.27` as zero;
//!     the one case that needed the propagation — a pivot of `5·10⁻¹⁹`
//!     beside terms of `10⁻¹⁸`, the residue of a `2×2` block with one
//!     noise diagonal — is caught by the growth test of M4 instead.
//! M4. [Rows.] The rows below the diagonal block — the `nz` columns
//!     removed in M3, as rows, then `R_s` — are gathered in the order of
//!     `P` and solved: `U ← A₂₁ L₁₁⁻ᵀ` by `trsm_lower` against the unit
//!     `L₁₁` (its diagonal written as ones, the slots of `e₂₁` as zeros),
//!     then `L₂₁ ← U B₁⁻¹` column by column: `/ d` for a `1×1` pivot,
//!     `[x y] E⁻¹` for a `2×2` one. [Growth.] A pivot whose column of
//!     `L₂₁` has an entry above `1/u` (`u` =
//!     [`SupernodalOptions::pivot_threshold`], default `10⁻⁸`) is removed
//!     like a zero pivot and M3 repeats without it: the row of the large
//!     entry is a candidate partner in the parent, where it is fully
//!     summed — the threshold partial pivoting of MA57 among the fully
//!     summed variables, with Ipopt's lax `u`. Without it a pivot small
//!     inside its block but not zero (a `2×2` block with a small `e₂₁`,
//!     stable by Bunch–Kaufman's test on the block alone) made `L₂₁` grow
//!     to `10⁷` and every pivot below it rounding noise (ganges, Netlib:
//!     a pivot of `0.70` with an error of `0.07`); `u = 10⁻²` (MA57's
//!     default) delays so much in the middle of an interior-point method
//!     that one Netlib system (woodw, augmented) did not finish in six
//!     minutes, and `u = 10⁻⁴` sits in between (tables in the decision
//!     log).
//! M5. [Delay.] Each removed column `k` goes to the parent with its
//!     reference scale — raised to `|a_kk|` and to the largest term of
//!     `l_k · u_k`: the pivots the block took *after* `k` was flagged may
//!     cancel in that sum (two singleton rows of `A` on one column: the
//!     second, flagged with an exact zero and scale `0`, got `b² − (ab)²/a²
//!     = 4·10⁻¹⁶` from the pivots of the first and was taken at the root
//!     with the expected inertia; 0.3.1) —
//!     and its Schur complement after the active pivots: `a_kk − l_k · u_k`, the cross
//!     terms `a_{k'k} − l_{k'} · u_k` with the removed columns `k'` after
//!     it, and `a_{rk} − l_r · u_k` for `r ∈ R_s` (`l` the rows of `L₂₁`,
//!     `u` the rows of `U`: `l_r B₁ l_kᵀ = l_r · u_k`). Its row `l_k` stays
//!     in the block of `s` for the solve.
//! M6. [Record.] The block of `s` is `(w' + nz + deg) × w'` (`w' = w − nz`
//!     pivots in the order of `P`, then the `nz` delayed rows, then
//!     `R_s`): `L₁₁` with `B₁` on its diagonal and `e₂₁` at `(k+1, k)` —
//!     the slot where `L` has an implicit zero, the convention of
//!     [`DenseKernel::sytrf`] — and `L₂₁` below. The pivots take the next
//!     `w'` positions of the *pivot order*: `fwd′`, `diag` and `pivot_2x2`
//!     (`true` on both columns of a `2×2` block) are indexed by it, and
//!     `pos[j]` records where the analyzed column `j` was eliminated.
//! Then steps N4–N6 as before (`e₂₁` is an entry of `Δ`: `ê₂₁ = e₂₁ /
//! (s_k s_{k+1})`; the inertia counts `+1 −1` for a `2×2` block with
//! `det E < 0`, step K4), and the rows of every block are remapped by
//! `pos` for the solve. [`NumericLdl::perm`] is `fwd′`,
//! [`NumericLdl::supernodes`] the block boundaries in pivot order,
//! [`NumericLdl::rows`] the rows of a block below its diagonal block,
//! [`NumericLdl::delayed`] the number of delays.
//!
//! The simplicial structure takes the same steps: a supernode of one
//! column is the direct case of M3, and its pivot is zero iff exactly
//! zero — then it is delayed, never ignored (the simplicial `LBLᵀ` of faer
//! accepts a zero pivot in silence and its solve returns infinities,
//! D-0094; here the unregularized factorization of a nonsingular matrix
//! either succeeds or fails with the row named). Why the leaf is not merged
//! with its parent in the symbolic phase instead: the symbolic phase does
//! not know the regularization, the regularized systems would pay the
//! explicit zeros, and a leaf is adjacent to its parent only when it is
//! the heaviest child; the numeric delay costs one row and one column of
//! a block per delayed pivot, only when it happens.
//!
//! *Growth and the end of an interior-point method.* The threshold test
//! of M4 delays every primal pivot below `u` relative to its column — near
//! the end of the method, where `D` reaches `10⁻¹⁰`, that is half of them,
//! and the delays pile up at the root (up to 15 352 in one Netlib system).
//! There the unregularized augmented system is singular to working
//! precision anyway (κ > 10¹² in 93 of 93 Netlib systems of the end
//! regime, by the scipy oracle of L1.5), and the regularized path, which
//! lives with the growth and whose refinement handles it (Chapter 5.14,
//! L1.5b), is the one to use. The refinement of Algorithm F is the
//! safeguard here too, and (S10) still refuses a factor with an infinity.
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
//! ## Algorithm S′ (Solve with the `LBLᵀ` factor)
//! Algorithm S with: `y ← P′ x` by `fwd′`; the blocks walked by their
//! pivot ranges and their remapped rows (delayed rows, then `R_s`); in S1
//! and S3 the slot `(k+1, k)` of a `2×2` block is skipped (it holds
//! `e₂₁`, not an entry of `L`); S2 runs block by block right after S1 (the
//! pivots of a block are final once its forward step is done), and a
//! `2×2` block solves `E [y_k; y_{k+1}] = [b_k; b_{k+1}]` by Cramer's rule.
//! Nothing is allocated (L5); the pivot pairs never cross a block (L8).
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
//! (L7) In the `LBLᵀ` path the pivot ranges of the blocks partition `0..N`,
//!      `fwd′` is a permutation, and every analyzed column is eliminated
//!      exactly once (`pos` is a permutation of `0..N`).
//! (L8) A `2×2` block lies inside one block of pivots (intranodal):
//!      `pivot_2x2[k] = true` comes in pairs `(k, k+1)`, `k+1` in the same
//!      block.
//! (L9) A delayed column is received by the parent supernode of the block
//!      that delayed it and nowhere else; its row of `L` stays in that
//!      block.
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
//! For Algorithm M (L1.6b): Duff & Reid 1983 (delayed pivots in the
//! multifrontal method), Duff 2004 (MA57: the threshold test among the
//! fully summed variables, the treatment of delays), Vanderbei 1995
//! (quasi-definite matrices factorize with `1×1` pivots in any order:
//! why the regularized path needs none), faer 0.24.4
//! `linalg::cholesky::bunch_kaufman::factor` (read to confirm that a
//! column with no off-diagonal entry is taken as a `1×1` pivot without
//! elimination, which is what M3 relies on from `sytrf`) — read, not
//! copied.
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
//! method", ACM TOMS 15(4), 1989 (relaxed amalgamation); Duff & Reid, "The
//! multifrontal solution of indefinite sparse symmetric linear systems",
//! ACM TOMS 9(3), 1983; Duff, "MA57 — a code for the solution of sparse
//! symmetric definite and indefinite systems", ACM TOMS 30(2), 2004;
//! Vanderbei, "Symmetric quasi-definite matrices", SIAM J. Optim. 5(1),
//! 1995; Bunch & Kaufman, "Some stable methods for calculating inertia
//! and solving symmetric linear systems", Math. Comp. 31, 1977.

use core::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::factor::count_f64;
use crate::kkt::{
    assemble_regularized, check_inertia, check_regularization, count_inertia, lower_with_diagonal,
    pivot_signs,
};
use crate::perm::{etree_counts, idx32};
use crate::real::{Real, is_exact_zero, le, le_exact, lt};
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
    /// `LBLᵀ` path only (Algorithm M, step M3): a pivot whose magnitude is
    /// at most this fraction of the largest entry of its column in the
    /// diagonal block of its supernode is a zero pivot and is delayed to
    /// the parent supernode (a `2×2` block: its determinant against the
    /// product of the two columns' largest entries). Relative, so that a
    /// pivot that is zero up to rounding is caught and a merely small one
    /// is kept. Default `10⁻¹⁴`.
    pub zero_pivot_tol: f64,
    /// `LBLᵀ` path only (Algorithm M, step M4): the threshold `u` of the
    /// growth test. A pivot whose column of `L` below the diagonal block
    /// has an entry above `1/u` is delayed to the parent supernode, where
    /// the large entry's row is a candidate partner (the threshold partial
    /// pivoting of MA57 among the fully summed variables). Default `10⁻⁸`
    /// (Ipopt's setting for MA57 on interior-point systems: it stops the
    /// growth that turns the pivots below into rounding noise and leaves
    /// the merely small pivots of a barrier method alone).
    pub pivot_threshold: f64,
}

impl Default for SupernodalOptions {
    fn default() -> Self {
        Self {
            relax_zeros: 4,
            min_supernode_cols: 8,
            supernodal_switch: 40.0,
            zero_pivot_tol: 1e-14,
            pivot_threshold: 1e-8,
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

/// The numeric factor of [`SupernodalLdl`]: `P K_reg Pᵀ = L̂ Δ̂ L̂ᵀ` with
/// `L̂` by blocks (column-major, `Δ̂` on the diagonal, the unit diagonal of
/// `L̂` implicit), in the caller's units (step N4). For the `LDLᵀ` of
/// Algorithm N (both regularized systems) `Δ̂` is diagonal, the blocks are
/// the symbolic supernodes and `P` is [`SymbolicLdl::perm`]. For the
/// `LBLᵀ` of Algorithm M (the unregularized augmented system) `Δ̂` has
/// `2×2` blocks — `pivot_2x2` true on both their columns, `e₂₁` stored in
/// the block at the slot `(k + 1, k)` where `L̂` has an implicit zero —,
/// `P` is the pivot order [`NumericLdl::perm`], and the blocks' pivot
/// ranges and rows are [`NumericLdl::supernodes`] and [`NumericLdl::rows`]
/// (delayed pivots move columns between blocks).
#[derive(Clone, Debug)]
pub struct NumericLdl<T: Real = f64> {
    /// One block per supernode: `nrows(s) × ncols(s)`, column-major.
    pub blocks: Vec<DenseBlock<T>>,
    /// `Δ̂` in pivot order (`diag[k]` = the pivot of permuted column `k`,
    /// in the caller's units: step N4).
    pub diag: Vec<T>,
    /// `2×2` pivot marks, in pivot order: `true` on both columns of a
    /// `2×2` block (`LBLᵀ`); all `false` for `LDLᵀ`.
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
    /// `Some` for an `LBLᵀ` factor (Algorithm M).
    lblt: Option<Lblt>,
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
    /// `k`): the symbolic one (step A2′) for `LDLᵀ`, the pivot order `fwd′`
    /// of Algorithm M for `LBLᵀ`.
    pub fn perm(&self) -> &Perm {
        match &self.lblt {
            Some(l) => &l.perm,
            None => &self.shared.perm,
        }
    }

    /// `true` for an `LBLᵀ` factor (Algorithm M: `2×2` pivots and delayed
    /// pivots possible — the unregularized augmented system), `false` for
    /// the `LDLᵀ` of Algorithm N.
    pub fn is_pivoted(&self) -> bool {
        self.lblt.is_some()
    }

    /// Pivots delayed to a parent supernode in Algorithm M (step M5),
    /// counted once per delay; `0` for `LDLᵀ`.
    pub fn delayed(&self) -> usize {
        self.lblt.as_ref().map_or(0, |l| l.delayed)
    }

    /// The blocks' boundaries in pivot order (`n_supernodes + 1` entries):
    /// `blocks[s]` holds the pivots `supernodes()[s]..supernodes()[s + 1]`.
    /// The symbolic supernodes for `LDLᵀ`; with delayed pivots a block may
    /// hold more or fewer columns than its supernode.
    pub fn supernodes(&self) -> &[u32] {
        match &self.lblt {
            Some(l) => &l.starts,
            None => &self.shared.supernodes,
        }
    }

    /// The rows of `blocks[s]` below its diagonal block, in pivot order:
    /// row `ncols + q` of the block is the pivot `rows(s)[q]` (for `LBLᵀ`,
    /// the delayed rows of the block come first, then its pattern).
    pub fn rows(&self, s: usize) -> &[u32] {
        match &self.lblt {
            Some(l) => &l.row_idx[l.row_ptr[s]..l.row_ptr[s + 1]],
            None => self.shared.pattern(s),
        }
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

/// What the solves of Algorithm S′ need of an `LBLᵀ` factor beyond the
/// shared structure (step M6).
#[derive(Clone, Debug)]
struct Lblt {
    /// `fwd′`: the pivot order (`fwd′[k]` = the original row of pivot `k`).
    perm: Perm,
    /// Pivot ranges of the blocks (`n_supernodes + 1` entries).
    starts: Vec<u32>,
    /// The rows of every block below its diagonal block (its delayed rows,
    /// then `R_s`), in pivot order; CSR by block.
    row_ptr: Vec<usize>,
    row_idx: Vec<u32>,
    /// `true` at the first column of every `2×2` block.
    first: Vec<bool>,
    /// Delays of step M5.
    delayed: usize,
}

/// A column delayed from a block to its parent (step M5).
struct DelayedCol<T> {
    /// The analyzed column.
    sym: u32,
    /// Its Schur complement diagonal.
    diag: T,
    /// The reference scale of that diagonal (step M3): the largest term of
    /// the sums that produced it, carried along so that the parent judges
    /// the pivot against the magnitudes that made it, not against its own.
    scale: T,
    /// Cross terms with the columns delayed after it from the same block.
    cross: Vec<T>,
    /// Its entries at the pattern rows of the block that delayed it.
    pattern: Vec<T>,
    /// That block.
    src: u32,
}

/// Work arrays of steps M3–M6, reused across supernodes.
struct PivotWork<T: Real> {
    tile: DenseBlock<T>,
    unit: DenseBlock<T>,
    panel: DenseBlock<T>,
    next: DenseBlock<T>,
    u: Vec<T>,
    m: Vec<T>,
    perm: Perm,
    two: Vec<bool>,
    active: Vec<usize>,
    removed: Vec<(usize, T)>,
    bad: Vec<usize>,
    cols: Vec<u32>,
    refs: Vec<T>,
}

impl<T: Real> PivotWork<T> {
    fn new() -> Self {
        Self {
            tile: DenseBlock::zeros(0, 0),
            unit: DenseBlock::zeros(0, 0),
            panel: DenseBlock::zeros(0, 0),
            next: DenseBlock::zeros(0, 0),
            u: Vec::new(),
            m: Vec::new(),
            perm: Perm::identity(0),
            two: Vec::new(),
            active: Vec::new(),
            removed: Vec::new(),
            bad: Vec::new(),
            cols: Vec::new(),
            refs: Vec::new(),
        }
    }
}

/// `Σ_i x_i y_i` over two strided rows of column-major blocks (step M5).
#[inline]
fn strided_dot<T: Real>(
    x: &[T],
    x0: usize,
    xs: usize,
    y: &[T],
    y0: usize,
    ys: usize,
    len: usize,
) -> T {
    let mut acc = T::ZERO;
    for i in 0..len {
        acc += x[x0 + i * xs] * y[y0 + i * ys];
    }
    acc
}

/// `max_i |x_i y_i|` over two strided rows of column-major blocks: the
/// largest term of the sum [`strided_dot`] forms (step M5).
#[inline]
fn strided_term_max<T: Real>(
    x: &[T],
    x0: usize,
    xs: usize,
    y: &[T],
    y0: usize,
    ys: usize,
    len: usize,
) -> T {
    let mut big = T::ZERO;
    for i in 0..len {
        big = big.max((x[x0 + i * xs] * y[y0 + i * ys]).abs());
    }
    big
}

/// Steps M3–M6 on the block `a` of one supernode: `w` active columns
/// (rows `0..w` their diagonal block, rows `w..` the pattern rows `R_s`),
/// `cols` their analyzed columns (`w` entries). On success `a` is the
/// block of step M6 (`w'` pivots in pivot order, then the delayed rows,
/// then `R_s`), `cols` holds the `w'` pivots in pivot order, `diag` and
/// `two` (`w'` entries written) the pivots and the `2×2` marks, and the
/// removed columns are returned for the parent (`src` left for the
/// caller); with `root`, a zero pivot is an error instead. `refs` (`w`
/// entries) are the reference
/// scales the columns bring (`0` for the supernode's own columns, the
/// carried scale of a delayed one). Errors name a local column of the
/// block as it was on entry.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // steps M3–M6 in one place
fn pivot_block<T: Real, K: DenseKernel<T>>(
    a: &mut DenseBlock<T>,
    w: usize,
    cols: &mut Vec<u32>,
    refs: &[T],
    zero_tol: T,
    growth_cap: T,
    root: bool,
    diag: &mut [T],
    two: &mut [bool],
    pw: &mut PivotWork<T>,
    kernel: &K,
) -> Result<Vec<DelayedCol<T>>, PivotError> {
    let nr0 = a.rows;
    let deg = nr0 - w;
    // The lower-triangle entry (i, j) of the diagonal block.
    let at = |a: &DenseBlock<T>, i: usize, j: usize| -> T {
        if i >= j {
            a.data[i + j * nr0]
        } else {
            a.data[j + i * nr0]
        }
    };
    // The one-column block: the direct case of M3 and M4 in place, with
    // no scratch — the common case of the simplicial structure (16 512
    // such blocks in fit2p's augmented system). The general path below
    // takes over when the pivot is zero or would make `L` grow.
    if w == 1 {
        let d = a.data[0];
        if !d.is_finite() {
            return Err(PivotError::NotFinite(0));
        }
        let scale = refs[0].max(d.abs());
        if !le_exact(d.abs(), zero_tol * scale)
            && !a.data[1..].iter().any(|&x| {
                let q = x / d;
                !q.is_finite() || lt(growth_cap, q.abs())
            })
        {
            for x in &mut a.data[1..] {
                *x = *x / d;
            }
            diag[0] = d;
            two[0] = false;
            return Ok(Vec::new());
        }
    }
    pw.active.clear();
    pw.active.extend(0..w);
    pw.removed.clear();
    // M3 and M4, repeated while a pivot is delayed.
    loop {
        let wa = pw.active.len();
        pw.m.clear();
        pw.m.resize(wa, T::ZERO);
        if wa == 0 {
            fit(&mut pw.tile, 0, 0);
            pw.perm = Perm::identity(0);
            pw.two.clear();
        } else if wa == 1 {
            let j = pw.active[0];
            let d = a.data[j + j * nr0];
            if !d.is_finite() {
                return Err(PivotError::NotFinite(j));
            }
            let scale = refs[j].max(d.abs());
            pw.m[0] = scale;
            if le_exact(d.abs(), zero_tol * scale) {
                if root {
                    return Err(PivotError::Zero(j));
                }
                pw.removed.push((j, scale));
                pw.active.clear();
                continue;
            }
            fit(&mut pw.tile, 1, 1);
            pw.tile.data[0] = d;
            pw.perm = Perm::identity(1);
            pw.two.clear();
            pw.two.push(false);
        } else {
            let tile = &mut pw.tile;
            fit(tile, wa, wa);
            tile.data.fill(T::ZERO);
            for (jj, &j) in pw.active.iter().enumerate() {
                for (ii, &i) in pw.active.iter().enumerate().skip(jj) {
                    // `active` is increasing: i ≥ j, the lower triangle.
                    tile.data[ii + jj * wa] = a.data[i + j * nr0];
                }
            }
            kernel
                .sytrf(tile, &mut pw.perm, &mut pw.two)
                .expect("sytrf on a square tile");
            // The reference scale `m_i` of pivot i (step M3): its diagonal as
            // the block holds it, the scale it carries, and the terms
            // `l_ij b_j l_ij` of the pivots before it.
            for (i, r) in pw.m.iter_mut().enumerate() {
                let p = pw.active[pw.perm.fwd[i] as usize];
                *r = a.data[p + p * nr0].abs().max(refs[p]);
            }
            {
                let mut j = 0;
                while j < wa {
                    let pair = pw.two[j];
                    let e11 = tile.data[j + j * wa].abs();
                    let (e21, e22) = if pair {
                        (
                            tile.data[j + 1 + j * wa].abs(),
                            tile.data[j + 1 + (j + 1) * wa].abs(),
                        )
                    } else {
                        (T::ZERO, T::ZERO)
                    };
                    for i in j + 1 + usize::from(pair)..wa {
                        let x = tile.data[i + j * wa].abs();
                        let mut t = x * x * e11;
                        if pair {
                            let y = tile.data[i + (j + 1) * wa].abs();
                            t = t.max(x * y * e21).max(y * y * e22);
                        }
                        pw.m[i] = pw.m[i].max(t);
                    }
                    j += 1 + usize::from(pair);
                }
            }
            pw.bad.clear();
            pw.refs.clear();
            pw.refs.resize(wa, T::ZERO);
            let mut i = 0;
            while i < wa {
                let p = pw.perm.fwd[i] as usize;
                if pw.two[i] {
                    let q = pw.perm.fwd[i + 1] as usize;
                    let (e11, e21, e22) = (
                        tile.data[i + i * wa],
                        tile.data[i + 1 + i * wa],
                        tile.data[i + 1 + (i + 1) * wa],
                    );
                    if !(e11.is_finite() && e21.is_finite() && e22.is_finite()) {
                        return Err(PivotError::NotFinite(pw.active[p]));
                    }
                    let big = e11.abs().max(e21.abs()).max(e22.abs());
                    if le_exact(big, zero_tol * pw.m[i].max(pw.m[i + 1])) {
                        pw.bad.push(p);
                        pw.bad.push(q);
                        pw.refs[p] = pw.m[i].max(pw.m[i + 1]);
                        pw.refs[q] = pw.refs[p];
                    }
                    i += 2;
                } else {
                    let d = tile.data[i + i * wa];
                    if !d.is_finite() {
                        return Err(PivotError::NotFinite(pw.active[p]));
                    }
                    if le_exact(d.abs(), zero_tol * pw.m[i]) {
                        pw.bad.push(p);
                        pw.refs[p] = pw.m[i];
                    }
                    i += 1;
                }
            }
            if !pw.bad.is_empty() {
                if root {
                    // Nothing is left to pair a zero pivot with: the matrix is
                    // singular to working precision.
                    return Err(PivotError::Zero(pw.active[pw.bad[0]]));
                }
                pw.bad.sort_unstable();
                for &b in pw.bad.iter().rev() {
                    let j = pw.active.remove(b);
                    pw.removed.push((j, pw.refs[b]));
                }
                continue;
            }
        }
        // M4: the rows below the diagonal block, in pivot order (the
        // delayed rows in increasing order of their column: step M5 stores
        // the cross terms of each with those after it).
        pw.removed.sort_unstable_by_key(|&(j, _)| j);
        let wa = pw.active.len();
        let nz = pw.removed.len();
        let nb = nz + deg;
        let panel = &mut pw.panel;
        fit(panel, nb, wa);
        for i in 0..wa {
            let j = pw.active[pw.perm.fwd[i] as usize];
            let col = &mut panel.data[i * nb..(i + 1) * nb];
            for (r, &(k, _)) in pw.removed.iter().enumerate() {
                col[r] = at(a, k, j);
            }
            col[nz..].copy_from_slice(&a.data[w + j * nr0..nr0 + j * nr0]);
        }
        pw.u.clear();
        if wa > 0 && nb > 0 {
            if wa > 1 {
                let unit = &mut pw.unit;
                fit(unit, wa, wa);
                unit.data.fill(T::ZERO);
                let mut i = 0;
                while i < wa {
                    let pair = pw.two[i];
                    unit.data[i + i * wa] = T::ONE;
                    for ii in i + 1 + usize::from(pair)..wa {
                        unit.data[ii + i * wa] = pw.tile.data[ii + i * wa];
                    }
                    if pair {
                        unit.data[i + 1 + (i + 1) * wa] = T::ONE;
                        for ii in i + 2..wa {
                            unit.data[ii + (i + 1) * wa] = pw.tile.data[ii + (i + 1) * wa];
                        }
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                kernel.trsm_lower(unit, panel, true);
            }
            // U, the rows of the removed columns before B₁⁻¹ (step M5).
            for i in 0..wa {
                pw.u.extend_from_slice(&panel.data[i * nb..i * nb + nz]);
            }
            // L₂₁ ← U B₁⁻¹.
            let mut i = 0;
            while i < wa {
                if pw.two[i] {
                    let (e11, e21, e22) = (
                        pw.tile.data[i + i * wa],
                        pw.tile.data[i + 1 + i * wa],
                        pw.tile.data[i + 1 + (i + 1) * wa],
                    );
                    let det = e11 * e22 - e21 * e21;
                    let (lo, hi) = panel.data.split_at_mut((i + 1) * nb);
                    let cx = &mut lo[i * nb..];
                    let cy = &mut hi[..nb];
                    for (x, y) in cx.iter_mut().zip(cy.iter_mut()) {
                        let (vx, vy) = (*x, *y);
                        *x = (vx * e22 - vy * e21) / det;
                        *y = (vy * e11 - vx * e21) / det;
                    }
                    i += 2;
                } else {
                    let d = pw.tile.data[i + i * wa];
                    for x in &mut panel.data[i * nb..(i + 1) * nb] {
                        *x = *x / d;
                    }
                    i += 1;
                }
            }
            // The growth test (step M4): a pivot whose column of L below the
            // block exceeds 1/u is delayed; its large entry's row is a
            // candidate partner higher up.
            pw.bad.clear();
            pw.refs.clear();
            pw.refs.resize(wa, T::ZERO);
            let grown = |i: usize| -> bool {
                panel.data[i * nb..(i + 1) * nb]
                    .iter()
                    .any(|x| !x.is_finite() || lt(growth_cap, x.abs()))
            };
            let mut i = 0;
            while i < wa {
                let p = pw.perm.fwd[i] as usize;
                if pw.two[i] {
                    let q = pw.perm.fwd[i + 1] as usize;
                    if grown(i) || grown(i + 1) {
                        pw.bad.push(p);
                        pw.bad.push(q);
                        pw.refs[p] = pw.m[i].max(pw.m[i + 1]);
                        pw.refs[q] = pw.refs[p];
                    }
                    i += 2;
                } else {
                    if grown(i) {
                        pw.bad.push(p);
                        pw.refs[p] = pw.m[i];
                    }
                    i += 1;
                }
            }
            if !pw.bad.is_empty() {
                if root {
                    // A root has no pattern rows; only delayed rows can grow,
                    // and nothing is left to pair them with.
                    return Err(PivotError::Zero(pw.active[pw.bad[0]]));
                }
                pw.bad.sort_unstable();
                for &b in pw.bad.iter().rev() {
                    let j = pw.active.remove(b);
                    pw.removed.push((j, pw.refs[b]));
                }
                continue;
            }
        }
        break;
    }
    let wa = pw.active.len();
    let nz = pw.removed.len();
    let nb = nz + deg;
    let panel = &mut pw.panel;
    // M5: the delayed columns, with their Schur complement after the
    // active pivots: `a − l_row · u_k`.
    let mut out = Vec::with_capacity(nz);
    for (r, &(k, scale)) in pw.removed.iter().enumerate() {
        let schur = |row: usize| strided_dot(&panel.data, row, nb, &pw.u, r, nz, wa);
        let diag_k = at(a, k, k) - schur(r);
        // The terms of this sum join the scale (step M5): pivots chosen
        // after the column was flagged may cancel in it.
        let scale = scale.max(at(a, k, k).abs()).max(strided_term_max(
            &panel.data,
            r,
            nb,
            &pw.u,
            r,
            nz,
            wa,
        ));
        let cross: Vec<T> = (r + 1..nz)
            .map(|r2| at(a, pw.removed[r2].0, k) - schur(r2))
            .collect();
        let pattern: Vec<T> = (0..deg)
            .map(|q| a.data[w + q + k * nr0] - schur(nz + q))
            .collect();
        out.push(DelayedCol {
            sym: cols[k],
            diag: diag_k,
            scale,
            cross,
            pattern,
            src: NONE,
        });
    }
    // M6: the new block.
    let nrn = wa + nb;
    let next = &mut pw.next;
    fit(next, nrn, wa);
    next.data.fill(T::ZERO);
    for i in 0..wa {
        let col = &mut next.data[i * nrn..(i + 1) * nrn];
        col[i..wa].copy_from_slice(&pw.tile.data[i + i * wa..(i + 1) * wa]);
        col[wa..].copy_from_slice(&panel.data[i * nb..(i + 1) * nb]);
        diag[i] = pw.tile.data[i + i * wa];
        two[i] = pw.two[i];
    }
    core::mem::swap(&mut a.data, &mut next.data);
    a.rows = nrn;
    a.cols = wa;
    pw.cols.clear();
    pw.cols
        .extend((0..wa).map(|i| cols[pw.active[pw.perm.fwd[i] as usize]]));
    cols.clear();
    cols.extend_from_slice(&pw.cols);
    Ok(out)
}

/// Algorithm U for the `LBLᵀ` path (step M2), kernel path: as
/// [`update_kernel`], with `B ← L_d[a..b, :] B_d` for the block-diagonal
/// `B_d` (`dd`, `td`: the pivots and the `2×2` marks of `d` in pivot
/// order; `e₂₁` read from the block) and the rows of `d` read past its
/// `off` delayed rows.
#[allow(clippy::too_many_arguments)] // the operands of step M2
fn update_lblt_kernel<T: Real, K: DenseKernel<T>>(
    ld: &DenseBlock<T>,
    dd: &[T],
    td: &[bool],
    off: usize,
    pat_d: &[u32],
    a: usize,
    b: usize,
    ls: &mut DenseBlock<T>,
    s0: usize,
    loc: &[u32],
    dscale: &mut [T],
    ws: &mut Scratch<T>,
    kernel: &K,
) {
    let (nrd, ncd) = (ld.rows, ld.cols);
    let base = ncd + off;
    let e = nrd - base;
    let (nm, nr) = (b - a, e - a);
    let ab = &mut ws.a;
    fit(ab, nr, ncd);
    for k in 0..ncd {
        gather_col(ld, k, base + a, nrd, &mut ab.data[k * nr..(k + 1) * nr]);
    }
    let bb = &mut ws.b;
    fit(bb, nm, ncd);
    let mut k = 0;
    while k < ncd {
        let lk = &ld.data[k * nrd + base + a..k * nrd + base + b];
        if td[k] {
            let lk1 = &ld.data[(k + 1) * nrd + base + a..(k + 1) * nrd + base + b];
            let (e11, e21, e22) = (dd[k], ld.data[k + 1 + k * nrd], dd[k + 1]);
            for j in 0..nm {
                bb.data[k * nm + j] = e11 * lk[j] + e21 * lk1[j];
                bb.data[(k + 1) * nm + j] = e21 * lk[j] + e22 * lk1[j];
            }
            k += 2;
        } else {
            let dk = dd[k];
            for (x, &l) in bb.data[k * nm..(k + 1) * nm].iter_mut().zip(lk) {
                *x = l * dk;
            }
            k += 1;
        }
    }
    let cb = &mut ws.c;
    fit(cb, nr, nm);
    cb.data.fill(T::ZERO);
    kernel.gemm_sub(ab, bb, cb);
    let nrs = ls.rows;
    for j in 0..nm {
        let cj = pat_d[a + j] as usize - s0;
        let col = &cb.data[j * nr..(j + 1) * nr];
        dscale[cj] = dscale[cj].max(col[j].abs());
        let dst = &mut ls.data[cj * nrs..(cj + 1) * nrs];
        for i in j..nr {
            dst[loc[pat_d[a + i] as usize] as usize] += col[i];
        }
    }
}

/// Algorithm U for the `LBLᵀ` path, simplicial path (plain loops).
#[allow(clippy::too_many_arguments)] // the same operands as the kernel path
fn update_lblt_scalar<T: Real>(
    ld: &DenseBlock<T>,
    dd: &[T],
    td: &[bool],
    off: usize,
    pat_d: &[u32],
    a: usize,
    b: usize,
    ls: &mut DenseBlock<T>,
    s0: usize,
    loc: &[u32],
    dscale: &mut [T],
) {
    let (nrd, ncd) = (ld.rows, ld.cols);
    let base = ncd + off;
    let e = nrd - base;
    let nrs = ls.rows;
    let mut k = 0;
    while k < ncd {
        let col = &ld.data[k * nrd + base..(k + 1) * nrd];
        if td[k] {
            let col1 = &ld.data[(k + 1) * nrd + base..(k + 2) * nrd];
            let (e11, e21, e22) = (dd[k], ld.data[k + 1 + k * nrd], dd[k + 1]);
            for j in a..b {
                let (t0, t1) = (e11 * col[j] + e21 * col1[j], e21 * col[j] + e22 * col1[j]);
                let cj = pat_d[j] as usize - s0;
                dscale[cj] = dscale[cj].max((col[j] * t0 + col1[j] * t1).abs());
                let dst = &mut ls.data[cj * nrs..(cj + 1) * nrs];
                for i in j..e {
                    dst[loc[pat_d[i] as usize] as usize] -= col[i] * t0 + col1[i] * t1;
                }
            }
            k += 2;
        } else {
            let dk = dd[k];
            for j in a..b {
                let t = col[j] * dk;
                let cj = pat_d[j] as usize - s0;
                dscale[cj] = dscale[cj].max((col[j] * t).abs());
                let dst = &mut ls.data[cj * nrs..(cj + 1) * nrs];
                for i in j..e {
                    dst[loc[pat_d[i] as usize] as usize] -= col[i] * t;
                }
            }
            k += 1;
        }
    }
}

impl<T: Real, K: DenseKernel<T>> SupernodalLdl<T, K> {
    /// Algorithm M: the `LBLᵀ` of the unregularized augmented system.
    #[allow(clippy::too_many_lines)] // steps M1–M6 and N4–N6 in order, as in the chapter
    fn factorize_lblt(
        &self,
        symbolic: &SymbolicLdl,
        k: &CscMatrix<T>,
        reg: &Regularization<T>,
    ) -> Result<(NumericLdl<T>, Inertia), KktError> {
        let st = &symbolic.shared;
        let n = st.n;
        let ns = st.n_supernodes();
        // N1.
        let (v, scale) = assemble_regularized(&symbolic.lower, symbolic.form, k, reg)?;
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
        let fwd = &st.perm.fwd;
        let mut super_of = vec![NONE; n];
        for s in 0..ns {
            let (s0, s1) = st.cols(s);
            super_of[s0..s1].fill(idx32(s));
        }
        let sparent = |s: usize| -> u32 {
            let p = symbolic.etree[st.cols(s).1 - 1];
            if p == NONE {
                NONE
            } else {
                super_of[p as usize]
            }
        };
        let zero_tol = T::from_f64(self.opts.zero_pivot_tol);
        let growth_cap = T::from_f64(self.opts.pivot_threshold).recip();
        let min_cols = if symbolic.simplicial {
            usize::MAX
        } else {
            self.opts.min_supernode_cols
        };
        let mut diag = vec![T::ZERO; n];
        let mut two = vec![false; n];
        let mut fwd_new = vec![0_u32; n];
        let mut pos = vec![NONE; n];
        let mut starts = vec![0_u32; ns + 1];
        let mut drow_ptr = vec![0_usize; ns + 1];
        let mut drows: Vec<u32> = Vec::new();
        let mut inbox: Vec<Vec<DelayedCol<T>>> = (0..ns).map(|_| Vec::new()).collect();
        let mut loc = vec![NONE; n];
        let mut ws = Scratch::new();
        let mut pw = PivotWork::new();
        let mut cols: Vec<u32> = Vec::new();
        let mut refs: Vec<T> = Vec::new();
        let mut dscale = vec![T::ZERO; n];
        let mut delayed = 0_usize;
        let mut next = 0_usize;
        for s in 0..ns {
            let (s0, s1) = st.cols(s);
            let nc = s1 - s0;
            let pat = st.pattern(s);
            let deg = pat.len();
            let inb = core::mem::take(&mut inbox[s]);
            let nd = inb.len();
            let w = nc + nd;
            cols.clear();
            cols.extend((s0..s1).map(idx32));
            cols.extend(inb.iter().map(|d| d.sym));
            // N3a.
            for c in 0..nc {
                loc[s0 + c] = idx32(c);
            }
            for (q, &r) in pat.iter().enumerate() {
                loc[r as usize] = idx32(w + q);
            }
            let (head, tail) = blocks.split_at_mut(s);
            let ls = &mut tail[0];
            // M1.
            if nd > 0 {
                let nr = w + deg;
                let old = core::mem::replace(ls, DenseBlock::zeros(nr, w));
                let nr_old = old.rows;
                for c in 0..nc {
                    let src = &old.data[c * nr_old..(c + 1) * nr_old];
                    let dst = &mut ls.data[c * nr..(c + 1) * nr];
                    dst[..nc].copy_from_slice(&src[..nc]);
                    dst[w..].copy_from_slice(&src[nc..]);
                }
                for (i, d) in inb.iter().enumerate() {
                    let c = nc + i;
                    ls.data[c + c * nr] = d.diag;
                    for (i2, &x) in d.cross.iter().enumerate() {
                        ls.data[c + 1 + i2 + c * nr] = x;
                    }
                    for (&r, &x) in st.pattern(d.src as usize).iter().zip(&d.pattern) {
                        let lr = loc[r as usize] as usize;
                        debug_assert_ne!(lr, NONE as usize, "(L9)");
                        if lr < c {
                            ls.data[c + lr * nr] = x;
                        } else {
                            ls.data[lr + c * nr] = x;
                        }
                    }
                }
            }
            let kernel_path = w >= min_cols;
            // M2.
            for u in symbolic.upd_ptr[s]..symbolic.upd_ptr[s + 1] {
                let d = symbolic.upd_d[u] as usize;
                let (a, b) = (symbolic.upd_from[u] as usize, symbolic.upd_to[u] as usize);
                let ld = &head[d];
                let (d0, d1) = (starts[d] as usize, starts[d + 1] as usize);
                let off = drow_ptr[d + 1] - drow_ptr[d];
                let (dd, td, pat_d) = (&diag[d0..d1], &two[d0..d1], st.pattern(d));
                if kernel_path && ld.cols >= min_cols {
                    update_lblt_kernel(
                        ld,
                        dd,
                        td,
                        off,
                        pat_d,
                        a,
                        b,
                        ls,
                        s0,
                        &loc,
                        &mut dscale[s0..s1],
                        &mut ws,
                        &self.kernel,
                    );
                } else {
                    update_lblt_scalar(
                        ld,
                        dd,
                        td,
                        off,
                        pat_d,
                        a,
                        b,
                        ls,
                        s0,
                        &loc,
                        &mut dscale[s0..s1],
                    );
                }
            }
            // M3–M6; the reference scales the columns bring (step M3): the
            // largest update a descendant made to the diagonal of an own
            // column, the carried scale of a delayed one.
            refs.clear();
            refs.extend_from_slice(&dscale[s0..s1]);
            refs.extend(inb.iter().map(|d| d.scale));
            let parent = sparent(s);
            let out = pivot_block(
                ls,
                w,
                &mut cols,
                &refs,
                zero_tol,
                growth_cap,
                parent == NONE,
                &mut diag[next..],
                &mut two[next..],
                &mut pw,
                &self.kernel,
            )
            .map_err(|e| match e {
                PivotError::Zero(c) => KktError::ZeroPivot {
                    position: fwd[cols[c] as usize] as usize,
                },
                PivotError::NotFinite(c) => KktError::NotFinite {
                    position: fwd[cols[c] as usize] as usize,
                },
            })?;
            let wp = ls.cols;
            for (i, &c) in cols.iter().enumerate() {
                fwd_new[next + i] = fwd[c as usize];
                pos[c as usize] = idx32(next + i);
            }
            next += wp;
            starts[s + 1] = idx32(next);
            drow_ptr[s + 1] = drow_ptr[s] + out.len();
            drows.extend(out.iter().map(|d| d.sym));
            delayed += out.len();
            if !out.is_empty() {
                debug_assert_ne!(parent, NONE, "(L9)");
                let p = parent as usize;
                for mut d in out {
                    d.src = idx32(s);
                    inbox[p].push(d);
                }
            }
            // N3d.
            for c in 0..nc {
                loc[s0 + c] = NONE;
            }
            for &r in pat {
                loc[r as usize] = NONE;
            }
        }
        debug_assert_eq!(next, n, "(L7)");
        debug_assert!(pos.iter().all(|&p| p != NONE), "(L7)");
        // (L8) and the first columns of the 2×2 blocks (Algorithm S′).
        let mut first = vec![false; n];
        for s in 0..ns {
            let (b0, b1) = (starts[s] as usize, starts[s + 1] as usize);
            let mut c = b0;
            while c < b1 {
                if two[c] {
                    debug_assert!(c + 1 < b1 && two[c + 1], "(L8)");
                    first[c] = true;
                    c += 2;
                } else {
                    c += 1;
                }
            }
        }
        // N4: L̂ = S⁻¹ L S, Δ̂ = S⁻¹ Δ S⁻¹ (ê₂₁ = e₂₁ / (s_k s_{k+1})), in the
        // caller's units.
        for (s, blk) in blocks.iter_mut().enumerate() {
            let (b0, b1) = (starts[s] as usize, starts[s + 1] as usize);
            let wp = b1 - b0;
            let nr = blk.rows;
            let dr = &drows[drow_ptr[s]..drow_ptr[s + 1]];
            let pat = st.pattern(s);
            for c in 0..wp {
                let sc = scale[fwd_new[b0 + c] as usize];
                // The slot (c + 1, c) of the first column of a pair holds e₂₁.
                let pair_first = first[b0 + c];
                let col = &mut blk.data[c * nr..(c + 1) * nr];
                col[c] = col[c] / (sc * sc);
                diag[b0 + c] = col[c];
                for i in c + 1..wp {
                    let si = scale[fwd_new[b0 + i] as usize];
                    col[i] = if pair_first && i == c + 1 {
                        col[i] / (sc * si)
                    } else {
                        col[i] * sc / si
                    };
                }
                for (x, &r) in col[wp..wp + dr.len()].iter_mut().zip(dr) {
                    *x = *x * sc / scale[fwd[r as usize] as usize];
                }
                for (x, &r) in col[wp + dr.len()..].iter_mut().zip(pat) {
                    *x = *x * sc / scale[fwd[r as usize] as usize];
                }
            }
        }
        // N5 (step K3′, (S10)).
        for (s, blk) in blocks.iter().enumerate() {
            let b0 = starts[s] as usize;
            for c in 0..blk.cols {
                if blk.data[c + c * blk.rows..(c + 1) * blk.rows]
                    .iter()
                    .any(|x| !x.is_finite())
                {
                    return Err(KktError::NotFinite {
                        position: fwd_new[b0 + c] as usize,
                    });
                }
            }
        }
        // N6 (S6), with the 2×2 blocks (step K4).
        let mut ones = Vec::with_capacity(n);
        let mut twos = Vec::new();
        for s in 0..ns {
            let (b0, b1) = (starts[s] as usize, starts[s + 1] as usize);
            let blk = &blocks[s];
            let mut c = b0;
            while c < b1 {
                if first[c] {
                    let local = c - b0;
                    twos.push((diag[c], blk.data[local + 1 + local * blk.rows], diag[c + 1]));
                    c += 2;
                } else {
                    ones.push(diag[c]);
                    c += 1;
                }
            }
        }
        let found = count_inertia(ones.into_iter(), twos.into_iter());
        let inertia = check_inertia(found, symbolic.form.expected_inertia(n))?;
        // The rows of every block in pivot order (Algorithm S′).
        let mut row_ptr = vec![0_usize; ns + 1];
        let mut row_idx: Vec<u32> = Vec::with_capacity(drows.len() + st.row_idx.len());
        for s in 0..ns {
            row_idx.extend(
                drows[drow_ptr[s]..drow_ptr[s + 1]]
                    .iter()
                    .map(|&c| pos[c as usize]),
            );
            row_idx.extend(st.pattern(s).iter().map(|&r| pos[r as usize]));
            row_ptr[s + 1] = row_idx.len();
        }
        let perm = Perm::from_fwd(fwd_new).map_err(KktError::Structure)?;
        Ok((
            NumericLdl {
                blocks,
                diag,
                pivot_2x2: two,
                inertia,
                bumped: 0,
                scale,
                shared: Arc::clone(st),
                scratch: SolveScratch(Mutex::new(vec![T::ZERO; n])),
                lblt: Some(Lblt {
                    perm,
                    starts,
                    row_ptr,
                    row_idx,
                    first,
                    delayed,
                }),
            },
            inertia,
        ))
    }

    /// Algorithm S′: the solve with an `LBLᵀ` factor.
    fn solve_lblt(numeric: &NumericLdl<T>, lb: &Lblt, x: &mut [T]) {
        let n = numeric.shared.n;
        let mut held = numeric.scratch.0.try_lock().ok();
        let mut temp = Vec::new();
        let y: &mut [T] = match held.as_deref_mut() {
            Some(v) if v.len() == n => v.as_mut_slice(),
            _ => {
                temp.resize(n, T::ZERO);
                temp.as_mut_slice()
            }
        };
        let fwd = &lb.perm.fwd;
        for (yk, &i) in y.iter_mut().zip(fwd) {
            *yk = x[i as usize];
        }
        let ns = lb.starts.len() - 1;
        // S1 with S2 block by block.
        for s in 0..ns {
            let (s0, s1) = (lb.starts[s] as usize, lb.starts[s + 1] as usize);
            let nc = s1 - s0;
            let blk = &numeric.blocks[s];
            let nr = blk.rows;
            let rows = &lb.row_idx[lb.row_ptr[s]..lb.row_ptr[s + 1]];
            let (ys, rest) = y.split_at_mut(s1);
            let mut j = 0;
            while j < nc {
                let pair = lb.first[s0 + j];
                let skip = usize::from(pair);
                {
                    let xj = ys[s0 + j];
                    let col = &blk.data[j * nr..(j + 1) * nr];
                    for (yi, &l) in ys[s0 + j + 1 + skip..]
                        .iter_mut()
                        .zip(&col[j + 1 + skip..nc])
                    {
                        *yi -= l * xj;
                    }
                    for (&r, &l) in rows.iter().zip(&col[nc..]) {
                        rest[r as usize - s1] -= l * xj;
                    }
                }
                if pair {
                    let xj = ys[s0 + j + 1];
                    let col = &blk.data[(j + 1) * nr..(j + 2) * nr];
                    for (yi, &l) in ys[s0 + j + 2..].iter_mut().zip(&col[j + 2..nc]) {
                        *yi -= l * xj;
                    }
                    for (&r, &l) in rows.iter().zip(&col[nc..]) {
                        rest[r as usize - s1] -= l * xj;
                    }
                    // S2 for the 2×2 block, by Cramer's rule.
                    let (e11, e21, e22) = (
                        numeric.diag[s0 + j],
                        blk.data[j + 1 + j * nr],
                        numeric.diag[s0 + j + 1],
                    );
                    let det = e11 * e22 - e21 * e21;
                    let (b0, b1) = (ys[s0 + j], ys[s0 + j + 1]);
                    ys[s0 + j] = (e22 * b0 - e21 * b1) / det;
                    ys[s0 + j + 1] = (e11 * b1 - e21 * b0) / det;
                    j += 2;
                } else {
                    ys[s0 + j] = ys[s0 + j] / numeric.diag[s0 + j];
                    j += 1;
                }
            }
        }
        // S3.
        for s in (0..ns).rev() {
            let (s0, s1) = (lb.starts[s] as usize, lb.starts[s + 1] as usize);
            let nc = s1 - s0;
            let blk = &numeric.blocks[s];
            let nr = blk.rows;
            let rows = &lb.row_idx[lb.row_ptr[s]..lb.row_ptr[s + 1]];
            let (ys, rest) = y.split_at_mut(s1);
            let mut j = nc;
            while j > 0 {
                j -= 1;
                let col = &blk.data[j * nr..(j + 1) * nr];
                let outer = gather_dot(&col[nc..], rows, rest, s1);
                let inner = dot(&col[j + 1..nc], &ys[s0 + j + 1..]);
                ys[s0 + j] -= inner + outer;
                if j > 0 && lb.first[s0 + j - 1] {
                    // The first column of the pair: the slot at row j is e₂₁.
                    j -= 1;
                    let col = &blk.data[j * nr..(j + 1) * nr];
                    let outer = gather_dot(&col[nc..], rows, rest, s1);
                    let inner = dot(&col[j + 2..nc], &ys[s0 + j + 2..]);
                    ys[s0 + j] -= inner + outer;
                }
            }
        }
        for (&yk, &i) in y.iter().zip(fwd) {
            x[i as usize] = yk;
        }
    }
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
        // Algorithm M for the unregularized augmented system (the rule of
        // Chapter 5.15, step N2); Algorithm N otherwise.
        if matches!(symbolic.form, KktForm::Augmented { .. })
            && !reg.is_quasi_definite()
            && reg.pivot_tol.is_none()
        {
            return self.factorize_lblt(symbolic, k, &reg);
        }
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
                lblt: None,
            },
            inertia,
        ))
    }

    fn solve(&self, numeric: &NumericLdl<T>, x: &mut [T]) {
        let st = &numeric.shared;
        let n = st.n;
        assert_eq!(x.len(), n, "solve: x has the wrong length");
        if let Some(lb) = &numeric.lblt {
            return Self::solve_lblt(numeric, lb, x);
        }
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
        for (s, blk) in numeric.blocks.iter().enumerate().take(ns) {
            let (s0, s1) = st.cols(s);
            let nc = s1 - s0;
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
                ..SupernodalOptions::default()
            },
            SupernodalOptions {
                relax_zeros: 4,
                min_supernode_cols: 1,
                supernodal_switch: 0.0,
                ..SupernodalOptions::default()
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

    /// A fixed permutation as an `Ordering` (the tests of the delayed
    /// pivots need the dual rows eliminated first).
    struct Fixed(Vec<u32>);
    impl Ordering for Fixed {
        fn order(&self, _: &Pattern, _: OrderingTarget) -> Result<Perm, SparseError> {
            Perm::from_fwd(self.0.clone())
        }
        fn name(&self) -> &'static str {
            "fixed"
        }
    }

    /// The dual rows first, then the primal columns.
    fn duals_first(n_primal: usize, n: usize) -> Fixed {
        Fixed((n_primal..n).chain(0..n_primal).map(idx32).collect())
    }

    /// `true` if the constraint block `A` of the augmented `k` (its rows
    /// `n_primal..`) has full row rank: Gaussian elimination with partial
    /// pivoting on the dense `A`, relative tolerance `10⁻¹⁰`. The random
    /// generator guarantees a nonzero per row, not independent rows (two
    /// singleton rows on one column happen): a rank-deficient `A` makes the
    /// unregularized `K` singular, and an explicit error (`ZeroPivot`, or
    /// `WrongInertia` when the zero pivot reaches the root as rounding
    /// noise of either sign) is then the right answer.
    fn full_row_rank(k: &CscMatrix, n_primal: usize) -> bool {
        let n = k.n_rows;
        let m = n - n_primal;
        let mut a = vec![0.0_f64; m * n_primal];
        for j in 0..n_primal {
            for p in k.col_ptr[j]..k.col_ptr[j + 1] {
                let i = k.row_idx[p] as usize;
                if i >= n_primal {
                    a[(i - n_primal) + j * m] = k.val[p];
                }
            }
        }
        let big = a.iter().fold(0.0_f64, |x, v| x.max(v.abs()));
        let mut rank = 0;
        let mut row_used = vec![false; m];
        for j in 0..n_primal {
            let mut best = (0.0_f64, usize::MAX);
            for i in 0..m {
                if !row_used[i] && a[i + j * m].abs() > best.0 {
                    best = (a[i + j * m].abs(), i);
                }
            }
            if best.1 == usize::MAX || le(best.0, 1e-10 * big, 0.0) {
                continue;
            }
            let r = best.1;
            row_used[r] = true;
            rank += 1;
            for i in 0..m {
                if i != r && !row_used[i] {
                    let f = a[i + j * m] / a[r + j * m];
                    for c in j..n_primal {
                        a[i + c * m] -= f * a[r + c * m];
                    }
                }
            }
        }
        rank == m
    }

    /// `‖P K Pᵀ − L B Lᵀ‖_max` from the public fields of a factor (dense;
    /// small systems).
    fn reconstruction_error(num: &NumericLdl<f64>, k: &CscMatrix) -> f64 {
        let n = num.order();
        let fwd = &num.perm().fwd;
        let (mut l, mut b) = (vec![0.0; n * n], vec![0.0; n * n]);
        let starts = num.supernodes();
        for s in 0..num.n_supernodes() {
            let (s0, s1) = (starts[s] as usize, starts[s + 1] as usize);
            let blk = &num.blocks[s];
            let (nc, nr) = (s1 - s0, blk.rows);
            let rows = num.rows(s);
            assert_eq!(nr, nc + rows.len());
            let mut j = 0;
            while j < nc {
                let pair = num.pivot_2x2[s0 + j];
                for (jj, skip) in [(j, usize::from(pair)), (j + 1, 0)] {
                    if jj == j + 1 && !pair {
                        break;
                    }
                    let col = &blk.data[jj * nr..(jj + 1) * nr];
                    let g = s0 + jj;
                    l[g + g * n] = 1.0;
                    b[g + g * n] = col[jj];
                    for i in jj + 1 + skip..nc {
                        l[s0 + i + g * n] = col[i];
                    }
                    for (q, &r) in rows.iter().enumerate() {
                        l[r as usize + g * n] = col[nc + q];
                    }
                }
                if pair {
                    assert!(num.pivot_2x2[s0 + j + 1], "(L8)");
                    b[s0 + j + 1 + (s0 + j) * n] = blk.data[j + 1 + j * nr];
                    b[s0 + j + (s0 + j + 1) * n] = blk.data[j + 1 + j * nr];
                    j += 2;
                } else {
                    j += 1;
                }
            }
        }
        let mut kd = vec![0.0; n * n];
        for c in 0..n {
            for p in k.col_ptr[c]..k.col_ptr[c + 1] {
                let r = k.row_idx[p] as usize;
                kd[r + c * n] = k.val[p];
                kd[c + r * n] = k.val[p];
            }
        }
        let mut err = 0.0_f64;
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0;
                for p in 0..n {
                    for q in 0..n {
                        s += l[i + p * n] * b[p + q * n] * l[j + q * n];
                    }
                }
                let (pi, pj) = (fwd[i] as usize, fwd[j] as usize);
                err = err.max((s - kd[pi + pj * n]).abs());
            }
        }
        err
    }

    /// Algorithm M: the unregularized augmented system factorizes with the
    /// expected inertia, `P K Pᵀ = L B Lᵀ` holds, the refined residual is
    /// small, and two factorizations agree bit for bit (L6).
    #[test]
    fn unregularized_augmented_systems_take_the_lblt_path() {
        // Spreads up to 10^±4: beyond that the random systems of the
        // generator are often numerically singular (κ > 10¹⁶, measured with
        // numpy on a failing seed), and the only correct answers are a zero
        // pivot or the wrong inertia, which the proptest below accepts.
        for (seed, spread, m, n) in [
            (1, 0.0, 6, 14),
            (2, 4.0, 12, 30),
            (3, 3.0, 20, 45),
            (4, 2.0, 9, 9),
        ] {
            let (aug, _, form) = kkt(m, n, spread, seed);
            let nn = n + m;
            let big = aug.val.iter().fold(0.0_f64, |x, v| x.max(v.abs()));
            let singular = !full_row_rank(&aug, n);
            for opts in variants() {
                let s = Naive::new(NaiveDense, opts);
                for ord in [&Amd::default() as &dyn Ordering, &duals_first(n, nn)] {
                    let sym = s.analyze(&aug.pattern(), form, ord).unwrap();
                    let factor = s.factorize(&sym, &aug, Regularization::none());
                    if singular {
                        // A zero pivot, or a rounding-level pivot at the root
                        // whose sign is noise (S6): explicit either way.
                        assert!(
                            matches!(
                                factor,
                                Err(KktError::ZeroPivot { .. } | KktError::WrongInertia { .. })
                            ),
                            "seed {seed} {opts:?}: singular K, got {:?}",
                            factor.map(|(_, i)| i)
                        );
                        continue;
                    }
                    let (num, inertia) = factor
                        .unwrap_or_else(|e| panic!("seed {seed} {opts:?} {}: {e:?}", ord.name()));
                    assert_eq!(inertia, form.expected_inertia(nn), "seed {seed} {opts:?}");
                    assert!(num.is_pivoted() && num.bumped == 0);
                    assert_eq!(num.perm().check_invariants(), Ok(()));
                    assert_eq!(num.supernodes().len(), num.n_supernodes() + 1);
                    assert_eq!(*num.supernodes().last().unwrap() as usize, nn);
                    let err = reconstruction_error(&num, &aug);
                    assert!(le(err, 1e-9 * big, 0.0), "seed {seed} {opts:?}: {err}");
                    let b = rhs(nn, seed);
                    let mut rhs_ = vec![0.0; nn];
                    sym_lower_matvec(&aug, &b, &mut rhs_);
                    let mut x = vec![0.0; nn];
                    let rep = s.solve_refined(&num, &aug, &rhs_, &mut x, Refinement::default());
                    assert!(
                        le(rep.residual, 1e-10, 0.0),
                        "seed {seed} {opts:?}: {rep:?}"
                    );
                    let (num2, _) = s.factorize(&sym, &aug, Regularization::none()).unwrap();
                    assert_eq!(num.blocks, num2.blocks);
                    assert_eq!(num.pivot_2x2, num2.pivot_2x2);
                    assert_eq!(num.delayed(), num2.delayed());
                }
            }
            // The regularized path is the LDLᵀ of Algorithm N.
            let s = Naive::default();
            let sym = s.analyze(&aug.pattern(), form, &Amd::default()).unwrap();
            let (num, _) = s.factorize(&sym, &aug, Regularization::default()).unwrap();
            assert!(!num.is_pivoted() && num.delayed() == 0);
            assert!(num.pivot_2x2.iter().all(|&t| !t));
            assert_eq!(num.supernodes(), &sym.supernodes[..]);
        }
    }

    /// Step M5: dual leaves of the elimination tree are delayed to their
    /// parent and pivoted there (`A = [1 1 0; 1 0 1]`, dual rows first:
    /// both are leaves under the first primal column, which can pair with
    /// only one of them, so the other is delayed twice); two identical
    /// dual rows make the matrix singular and the zero pivot is reported at
    /// one of them.
    #[test]
    fn dual_leaves_are_delayed_never_ignored() {
        let aug = |rows: &[(usize, usize)]| {
            let mut t = vec![
                (Row(0), Col(0), -1.0),
                (Row(1), Col(1), -2.0),
                (Row(2), Col(2), -0.5),
            ];
            for &(i, j) in rows {
                t.push((Row::from_usize(3 + i), Col::from_usize(j), 1.0));
            }
            CscMatrix::from_triplets(5, 5, &t).unwrap()
        };
        let form = KktForm::Augmented {
            n_primal: 3,
            n_dual: 2,
        };
        let k = aug(&[(0, 0), (0, 1), (1, 0), (1, 2)]);
        for opts in variants() {
            let s = Naive::new(NaiveDense, opts);
            let sym = s.analyze(&k.pattern(), form, &duals_first(3, 5)).unwrap();
            let (num, inertia) = s.factorize(&sym, &k, Regularization::none()).unwrap();
            assert_eq!(inertia, form.expected_inertia(5));
            assert!(num.delayed() >= 1, "{opts:?}: {}", num.delayed());
            assert!(le(reconstruction_error(&num, &k), 1e-12, 0.0));
            let b = [1.0, 2.0, 3.0, 4.0, 5.0];
            let mut x = [0.0; 5];
            let rep = s.solve_refined(&num, &k, &b, &mut x, Refinement::default());
            assert!(le(rep.residual, 1e-12, 0.0), "{rep:?}");
            // Singular: identical dual rows.
            let z = aug(&[(0, 0), (1, 0)]);
            let sym = s.analyze(&z.pattern(), form, &duals_first(3, 5)).unwrap();
            assert!(
                matches!(
                    s.factorize(&sym, &z, Regularization::none()),
                    Err(KktError::ZeroPivot { position: 3 | 4 })
                ),
                "{opts:?}"
            );
            // And with the other orderings too (the leaf may sit anywhere).
            let sym = s.analyze(&z.pattern(), form, &Amd::default()).unwrap();
            assert!(matches!(
                s.factorize(&sym, &z, Regularization::none()),
                Err(KktError::ZeroPivot { .. })
            ));
        }
    }

    /// The `f32` instance through Algorithm M as well.
    #[test]
    fn a_second_scalar_type_takes_the_lblt_path() {
        let (aug, _, form) = kkt(8, 20, 1.0, 7);
        let k = CscMatrix::<f32>::from_raw_parts(
            aug.n_rows,
            aug.col_ptr.clone(),
            aug.row_idx.clone(),
            #[allow(clippy::cast_possible_truncation)]
            aug.val.iter().map(|&v| v as f32).collect(),
        )
        .unwrap();
        for opts in variants() {
            let s = SupernodalLdl::<f32, NaiveDense>::new(NaiveDense, opts);
            let sym = s.analyze(&k.pattern(), form, &duals_first(20, 28)).unwrap();
            let (num, inertia) = s
                .factorize(&sym, &k, Regularization::<f32>::none())
                .unwrap();
            assert_eq!(inertia, form.expected_inertia(28));
            assert!(num.is_pivoted());
            let xs: Vec<f32> = (0..28)
                .map(|i| if i % 2 == 0 { 1.0 } else { -0.5 })
                .collect();
            let mut b = vec![0.0_f32; 28];
            sym_lower_matvec(&k, &xs, &mut b);
            let mut x = vec![0.0_f32; 28];
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
            assert!(le(rep.residual, 1e-4, 0.0), "{opts:?}: {rep:?}");
        }
    }

    /// `FaerLdl` without regularization (its intranodal `LBLᵀ`, supernodal
    /// structure) and `SupernodalLdl<f64, FaerDense>` on the same systems:
    /// the same inertia and solutions equal to rounding.
    #[cfg(feature = "faer")]
    #[test]
    fn lblt_agrees_with_faer() {
        use crate::{FaerDense, FaerLdl};
        for (seed, spread, m, n) in [(1, 0.0, 12, 30), (2, 4.0, 25, 60), (3, 8.0, 40, 90)] {
            let (aug, _, form) = kkt(m, n, spread, seed);
            let nn = n + m;
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
            let fs = faer.analyze(&aug.pattern(), form, &Amd::default()).unwrap();
            let os = ours.analyze(&aug.pattern(), form, &Amd::default()).unwrap();
            let Ok((fn_, fi)) = faer.factorize(&fs, &aug, Regularization::none()) else {
                // faer's intranodal pivoting has no delays: a dual leaf
                // alone in a supernode defeats it (that is the point of
                // step M5); nothing to compare on this seed.
                continue;
            };
            let (on, oi) = ours.factorize(&os, &aug, Regularization::none()).unwrap();
            assert_eq!(fi, oi);
            assert!(fn_.is_pivoted() && on.is_pivoted());
            let b = rhs(nn, seed);
            let (mut x, mut y) = (b.clone(), b.clone());
            faer.solve_refined(&fn_, &aug, &b, &mut x, Refinement::default());
            ours.solve_refined(&on, &aug, &b, &mut y, Refinement::default());
            let scale = x.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            assert!(
                x.iter()
                    .zip(&y)
                    .all(|(&a, &b)| approx_eq(a, b, 1e-8 * scale)),
                "seed {seed}"
            );
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
            if regime == 2 {
                // Near the end of an interior-point method the unregularized
                // system may be numerically singular: an explicit error or a
                // finite solution, never a silent NaN.
                let sym = s.analyze(&aug.pattern(), form, &Amd::default()).unwrap();
                if let Ok((num, _)) = s.factorize(&sym, &aug, Regularization::none()) {
                    let mut x = rhs(n + m, seed);
                    s.solve(&num, &mut x);
                    prop_assert!(x.iter().all(|v| v.is_finite()));
                }
            } else if full_row_rank(&aug, n) {
                let (i, res, _, _) = solve_ok(&s, &aug, form, Regularization::none(), &rhs(n + m, seed));
                prop_assert_eq!(i, form.expected_inertia(n + m));
                prop_assert!(le(res, 1e-10, 0.0), "augmented, LBLᵀ: {}", res);
            } else {
                let sym = s.analyze(&aug.pattern(), form, &Amd::default()).unwrap();
                let singular = s.factorize(&sym, &aug, Regularization::none());
                prop_assert!(matches!(singular, Err(KktError::ZeroPivot { .. } | KktError::WrongInertia { .. })), "rank-deficient A: {:?}", singular.map(|(_, i)| i));
            }
            let (i, _, _, _) = solve_ok(&s, &normal, KktForm::NormalEquations, Regularization::default(), &rhs(m, seed));
            prop_assert_eq!(i, KktForm::NormalEquations.expected_inertia(m));
        }
    }
}
