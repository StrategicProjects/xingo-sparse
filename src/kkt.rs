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
//! A [`Regularization`] `(ρ, δ, τ, β, scale)` = (`primal`, `dual`,
//! `pivot_tol`, `bump`, `scale`) defines the matrix the backend
//! factorizes. With `S = diag(s)` the symmetric equilibration of `K`
//! (step K2′ below; powers of two) and `Σ = diag(−ρ I_n, δ I_m)` for the
//! augmented system (`Σ = δ I_m` for the normal equations),
//! ```text
//!   K_reg = K + S⁻¹ Σ S⁻¹    (ShiftScale::Equilibrated, the default),
//!   K_reg = K + Σ            (ShiftScale::Absolute),
//! ```
//! so the default shifts are taken *in the equilibrated matrix*
//! `K̃ = S K S`, whose rows have their largest entry near 1: `δ = 10⁻⁸` is
//! then relative to the magnitude of each row, whatever the spread of
//! `D`. The *pivot signs* are `σ_k = −1` for the primal rows, `+1` for the
//! dual rows (all `+1` for the normal equations). The *inertia* of a symmetric
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
//! K1′. [Finite input.] Every stored value of the lower triangle of `K`
//!     is finite ([`KktError::Structure`] otherwise): a NaN or an infinity
//!     in `K` would otherwise surface as a NaN solution.
//! K2. [Assemble.] The lower triangle of `K` on the analyzed pattern, which
//!     has a full diagonal (a dual diagonal entry absent from `K` is `0`).
//! K2′. [Symmetric equilibration (Ruiz 2001).] Scale `K` to
//!     `K̃ = S K S`, `S = diag(s)` with powers of two: start from
//!     `s = 1`; in each pass compute, for every row `i` of the *scaled*
//!     matrix (both triangles), `r_i = max_j |k̃_ij|` and multiply `s_i` by
//!     `⌊1/√r_i⌉₂` (rows without entries keep `s_i`); stop after a pass
//!     that changes no `s_i` — exact, the factors being powers of two — or
//!     after [`EQUILIBRATION_PASSES`] passes. Every row of `K̃` then has
//!     its largest entry in `[2^{-1}, 2^{1}]` roughly, whatever the spread
//!     of `D`. The scaling is a congruence, so the inertia of `K̃` is that
//!     of `K` (Sylvester). Being by powers of two it is exact, and an
//!     `LDLᵀ` without pivoting *commutes* with it: the factor of `K̃` is
//!     `(S L S⁻¹)(S Δ S)(S L S⁻¹)ᵀ` bit for bit, barring overflow and
//!     underflow. So K2′ by itself changes exactly two things: the
//!     thresholds `τ`, `β` of K3 become *relative* to the magnitude of
//!     each row, and the intermediate values stay in range.
//! K2″. [Static regularization.] Add the shifts on the diagonal of `K̃`:
//!     `Σ` itself with [`ShiftScale::Equilibrated`] (`K̃_reg = K̃ + Σ`),
//!     `S Σ S` with [`ShiftScale::Absolute`] (`K̃_reg = S (K + Σ) S`): `−ρ`
//!     on the primal diagonal and `+δ` on the dual one (normal equations:
//!     `+δ` everywhere; `ρ` does not apply, the caller folds it into
//!     `D`). The backend factorizes `K̃_reg` and solves with `S F̃⁻¹ S`;
//!     `S⁻¹ K̃_reg S⁻¹` is the `K_reg` of the notation above.
//! K3. [Dynamic regularization.] If `τ = pivot_tol` is `Some`, the backend
//!     replaces every pivot `d_k` with `σ_k d_k ≤ τ` (too small, or of the
//!     wrong sign) by `σ_k β` while it eliminates — the dynamic part of
//!     Altman & Gondzio's (1999) primal-dual regularization, in the form
//!     faer's `LdltRegularization` applies — and counts the bumped pivots. Bumping
//!     needs the sign `σ_k`, which only the form knows — the policy is
//!     here; the elimination loop that applies it is the backend's.
//! K3′. [Finite factor (S10).] Every stored value of the factor is
//!     finite; otherwise the call fails with [`KktError::NotFinite`] at
//!     the first column (original numbering) holding a NaN or an infinity
//!     — an overflow of the elimination is reported, never handed to the
//!     solves (where it would become a NaN solution).
//! K4. [Inertia (S6).] From the pivots of the factor, count the inertia.
//!     If it is not `(N, 0, 0)` (normal) or `(m, n, 0)` (augmented),
//!     return [`KktError::WrongInertia`]. With `τ = Some`, every pivot has
//!     the sign `σ_k` by K3, so the check cannot fail: the factor is that
//!     of a perturbation of `K_reg` with the right inertia, and refinement
//!     (Algorithm F) measures how far it is from `K`.
//!
//! ## Choosing the regularization
//! Three facts, measured on the 558 Netlib systems of L1.5/L1.5b (93 LPs ×
//! normal and augmented × the start, middle and end regimes of `D`),
//! decide the default.
//!
//! 1. *A shift must be above the rounding level of its row.* A shift
//!    `δ < ε_T · max|k_ij|` changes nothing in floating point. At the end
//!    of an interior-point method `D` spreads to `10^{±10}` and
//!    `max|K| ≈ 10^{10}`, so the absolute `ρ = δ = 10⁻⁸` of L1.5 failed on
//!    32 of the 558 systems (zero pivots after overflow, or NaN
//!    solutions). One shift relative to the whole matrix
//!    ([`Regularization::relative_to`]`(k, 10⁻⁸)`, `δ ≈ 10²` there)
//!    factorizes all 558, but perturbs the rows of small magnitude so much
//!    that refinement against the unregularized `K` stalls (418 of 558 at
//!    or below `10⁻¹⁰` after 50 classical steps). Shifts taken in the
//!    equilibrated matrix (step K2″, [`ShiftScale::Equilibrated`]) are
//!    relative *row by row*.
//! 2. *The product `ρδ` must be well above the unit roundoff.* Eliminating
//!    a primal pivot of size `~ρ` adds terms of size `~1/ρ` to the dual
//!    block, whose pivots must stay above `δ`: with `ρ = δ = 10⁻⁸` in the
//!    equilibrated matrix (`ρδ = 10⁻¹⁶ < ε`) nine augmented systems of the
//!    end regime still got pivots of the wrong sign; with `ρδ ≥ 10⁻¹⁴` none
//!    did (the stability analysis of quasi-definite factorizations of Gill,
//!    Saunders & Shinnerl 1996 bounds the growth by the shifts).
//! 3. *The normal equations see only `δ`*, and every unit of `δ` slows the
//!    refinement on their small eigenvalues: `δ = 10⁻⁶` left 46 systems
//!    above `10⁻¹⁰` after ten steps, `δ = 10⁻⁸` none (with Algorithm F's
//!    GMRES).
//!
//! Hence the default `ρ = 10⁻⁶`, `δ = 10⁻⁸`, equilibrated: all 558 systems
//! with the right inertia, no bump, and a residual against the
//! unregularized `K` at or below `10⁻¹⁰` after the ten steps of
//! [`Refinement::default`] (L1.5b, D1–D3; the bumping of K3 is the safety
//! net, not the mechanism). An interior-point method that wants other
//! values passes them; [`ShiftScale::Absolute`] keeps shifts in the
//! caller's units.
//!
//! *The defaults and the scalar (task L1.8).* The three facts above are
//! statements about the unit roundoff, so the defaults follow it. With
//! `ρ_T = ε_T / ε₆₄` (Rule P of Chapter 5.3): the static shifts and the
//! bump are multiplied by `√ρ_T` — fact 2 constrains the *product* of the
//! shifts, `ρδ ≈ 45 ε_T`, and the bump is of the order of `√ε_T` —, the
//! pivot tolerance `τ` and the target of [`Refinement::default`] by
//! `ρ_T` (magnitudes of rounding noise). `√ρ_T` is computed in `f64`. For
//! `f64` both factors are 1 and the defaults are those measured in
//! L1.5b, bit for bit. In double-double (`√ρ = 2⁻²⁶`, exact): `ρ ≈
//! 1.5·10⁻¹⁴`, `δ ≈ 1.5·10⁻¹⁶`, target `2.2·10⁻²⁸`; on the augmented
//! systems of `xingo-extprec`'s `dd_profile` the plain solve then leaves
//! `10⁻¹¹`–`10⁻¹³` and refinement reaches the target in 2–4 steps where
//! it is attainable (5–8 with the `f64` shifts; D-0111). The budget of ten
//! steps is kept. The two thresholds of the pivoted path,
//! [`crate::SupernodalOptions`]`::{zero_pivot_tol, pivot_threshold}`, do
//! not scale (Chapter 5.17).
//!
//! ## Algorithm F (Refinement by preconditioned GMRES, with the best iterate; (S7))
//! Given a factor `F ≈ K_reg`, the matrix `K` to be solved (the
//! *unregularized* one, or `K_reg` itself — whichever the caller passes),
//! `b` and a [`Refinement`] `(t_max, target)` = (`max_iters`,
//! `target_residual`), where `t_max` bounds the solves with `F` after the
//! first one:
//! F1. `x ← F⁻¹ b`; `r ← b − K x`; `ρ ← res(x)`; `x* ← x`, `ρ* ← ρ`;
//!     `t ← 0`.
//! F2. [Done?] If `ρ* ≤ target`, `ρ` is not finite, or `t = t_max`, go to
//!     F6.
//! F3. [Arnoldi cycle, right preconditioned (Saad 2003, §9.3.2).] `x ← x*`,
//!     `r ← b − K x` (the residual of the best iterate), `β ← ‖r‖₂`,
//!     `v₁ ← r/β`, `g ← β e₁`, `m ← t_max − t`. For `j = 1, 2, …, m`:
//!     `z_j ← F⁻¹ v_j`, `t ← t + 1`; `w ← K z_j`; modified Gram–Schmidt
//!     `h_ij ← ⟨w, v_i⟩`, `w ← w − h_ij v_i` for `i ≤ j`;
//!     `h_{j+1,j} ← ‖w‖₂`; apply the previous Givens rotations to column
//!     `j` of `H`, compute the rotation that annihilates `h_{j+1,j}` and
//!     apply it to `g`. Leave the loop with `k = j` when `h_{j+1,j} = 0`
//!     (the Krylov space is invariant: the solution is in it), when
//!     `|g_{j+1}| ≤ target·(1 + ‖b‖_∞)` (in exact arithmetic `|g_{j+1}|`
//!     is the 2-norm of the residual, which bounds its ∞-norm), or when
//!     `j = m`.
//! F4. [Update.] Solve the `k × k` upper triangular system `R y = g_{1:k}`
//!     (the rotated `H`); `x ← x* + Σ_j y_j z_j`; `r ← b − K x`;
//!     `ρ ← res(x)`. If `ρ < ρ*`, set `x* ← x`, `ρ* ← ρ`.
//! F5. Go to F2 (a new cycle restarts from `x*`).
//! F6. Return `x*` with `iters = t`, `residual = ρ*` and
//!     `converged = (ρ* ≤ target)`.
//!
//! *Why GMRES and not the classical iteration.* The classical refinement
//! `x ← x + F⁻¹ r` of L1.5 contracts the error by the factor
//! `‖I − F⁻¹ K‖ = ‖F⁻¹ (K_reg − K)‖` per step; on the eigenvectors of `K`
//! whose eigenvalues are not much larger than the shifts — the nearly
//! rank-deficient `A D Aᵀ` of the middle of an interior-point method, or
//! an augmented system with `D` spread to `10^{±4}` — that factor is close
//! to 1, and ten steps left 15 of the 558 Netlib systems of L1.5b above
//! `10⁻¹⁰` against the unregularized `K`. Right-preconditioned GMRES
//! minimizes `‖b − K x‖₂` over the same affine Krylov space
//! `x₀ + span{F⁻¹r₀, (F⁻¹K) F⁻¹r₀, …}` that holds the classical
//! iterates, so in exact arithmetic it is never worse, and the few
//! eigenvalues of `F⁻¹ K` far from 1 — one per small eigenvalue of `K`,
//! the regularization being a small perturbation elsewhere — cost one
//! step each. This is GMRES preconditioned by a perturbed (statically
//! pivoted, here statically regularized) `LDLᵀ`, analyzed by Arioli,
//! Duff, Gratton & Pralet (2007), and GMRES-based iterative refinement
//! (Carson & Higham 2017). One cycle with all the budget is the normal
//! case; F5 restarts only when the estimate `|g_{k+1}|` and the computed
//! residual disagree (rounding) or the budget allows another cycle after
//! a breakdown.
//!
//! *Memory.* A cycle of `m` steps keeps `m + 1` Arnoldi vectors `v_j` and
//! `m` preconditioned vectors `z_j` (so F4 needs no further solve), plus
//! `H` and the rotations: `(2m + 3) N + O(m²)` scalars, held by a
//! [`RefineWorkspace`] that [`KktSolver::solve_refined_with`] reuses
//! across calls — after the first call with a given order and budget,
//! refinement allocates nothing (Chapter 5.15 for the solves themselves).
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
//! ## Algorithm H (Condition estimate by solves with the factor; issue #1)
//! *Why.* The unregularized `LBLᵀ` (Algorithm M of Chapter 5.17) refuses
//! every structurally singular `K` (step M7), but a `K` that is singular
//! only numerically — full structural rank, a smallest singular value at
//! the rounding level, like the `A` of etamacro (rank 399 of 400 by SVD) —
//! can be factorized without an error: a backward-stable factor is the
//! exact factor of a nearby nonsingular `K + E`, and no local test on the
//! pivots decides singularity (D-0119 measured four and rejected them,
//! each refusing nonsingular Netlib systems). What remains is a *global*
//! warning: an estimate of the condition of the factored matrix, on
//! request ([`KktSolver::condition_estimate`]), so that no existing path
//! changes and nothing is paid unless asked.
//!
//! *Notation.* `F` is the matrix the solves of a factor apply the inverse
//! of: `K_reg` for a regularized factor, `K + E` for the unregularized
//! `LBLᵀ`; both are symmetric, so `F⁻ᵀ = F⁻¹` and no transposed solve is
//! needed. `‖·‖₁` is the 1-norm (`= ‖·‖_∞` for a symmetric matrix);
//! `sign(y)_i = −1` if `y_i < 0`, else `+1`. The estimate is
//! `rcond = 1 / (‖K‖₁ · est ‖F⁻¹‖₁)`, LAPACK's `xSYCON`.
//!
//! H0. [Norm.] `ν ← ‖K‖₁` from the stored lower triangle of `k` (both
//!     triangles: `|k_ij|` counts in columns `j` and `i` for `i ≠ j`). If
//!     `N = 0`, return `ν = 0`, `est = 0`, no solve.
//! H1. [Start.] `x ← 𝟙/N`; `y ← F⁻¹ x`; `γ ← ‖y‖₁`. If `N = 1`, return
//!     (`γ = |F⁻¹|` exactly). `ξ ← sign(y)`; `z ← F⁻¹ ξ`; `j ← argmax_i |z_i|`
//!     (the first on ties); `k ← 2`.
//! H2. [Unit vector.] `y ← F⁻¹ e_j`; `γ′ ← ‖y‖₁`; `γ ← max(γ, γ′)`. If
//!     `sign(y) = ξ` (the iteration repeats) or `γ′ ≤ γ` held before
//!     (no progress), go to H4. Otherwise `ξ ← sign(y)`.
//! H3. [Gradient.] `z ← F⁻¹ ξ`; `j′ ← j`; `j ← argmax_i |z_i|`. If
//!     `z_{j′} ≠ ‖z‖_∞` (Hager's test `‖z‖_∞ ≤ zᵀx` fails for `x = e_{j′}`)
//!     and `k <` [`CONDITION_ESTIMATE_STEPS`], `k ← k + 1` and go to H2.
//! H4. [Safeguard (Higham 1988).] `x_i ← (−1)^i (1 + i/(N − 1))`,
//!     `y ← F⁻¹ x`; `γ ← max(γ, 2‖y‖₁/(3N))`. Return `ν`, `γ` and the count
//!     of solves.
//!
//! If a solve returns a non-finite vector, the estimate is `+∞` at once
//! (`rcond = 0`). Steps H1–H3 are Hager's (1984) gradient ascent of the
//! convex `‖F⁻¹ x‖₁` over the unit ball of `‖·‖₁`, whose maximum is at a
//! vertex `e_j`, in Higham's (1988) form — LAPACK's `xLACN2`, with at most
//! five iterations and the alternating vector of H4 against the matrices
//! that defeat the ascent; unlike `xLACN2`, `γ` keeps the largest value
//! seen (H2 does not overwrite it with a smaller one). Solves: `1` for
//! `N = 1`, otherwise between `4` (H1, one H2, H4) and
//! `2·CONDITION_ESTIMATE_STEPS + 1 = 11`.
//!
//! *What it is.* (H1′) Every candidate is `‖F̂⁻¹ x‖₁ / ‖x‖₁` for a nonzero
//! `x` (`‖𝟙/N‖₁ = ‖e_j‖₁ = 1`, and the vector of H4 has `‖x‖₁ = 3N/2`),
//! with `F̂⁻¹` the operator the solves apply: so `γ ≤ ‖F̂⁻¹‖₁` up to the
//! rounding of the sums — a *lower* bound, never an upper one, and
//! `rcond` is never too small for `F̂`. Higham (2002, §15.3) reports it
//! within a factor 3 of `‖F⁻¹‖₁` in nearly every case, and exhibits
//! matrices where it is arbitrarily far below; measured here on random
//! KKT systems (tests of this module). (H2′) Only [`KktSolver::solve`]
//! reads the factor: the estimate never changes it, and every backend
//! inherits it.
//!
//! *What it is not.* Not a proof: it can underestimate `‖F⁻¹‖₁`, so a
//! large `rcond` does not certify anything (a rigorous bound is
//! `xingo-extprec`'s business). And it measures `F`, not `K`: for a
//! regularized factor `‖F⁻¹‖` is bounded by the shifts, and a singular `K`
//! shows as a merely large `1/rcond`. It is meant for the unregularized
//! `LBLᵀ` (`Regularization::none()`), where `F = K + E` with `E` at the
//! rounding level, so that `rcond ≈ 1/κ₁(K)` as long as `κ₁(K) u ≪ 1`, and
//! `rcond` of the order of `u` or below says "singular to working
//! precision". Which multiple of `u` to call singular is the caller's
//! choice: on the 279 unregularized Netlib systems of the calibration
//! `rcond ≤ 10³ u` separated the singular from the nonsingular ones — a
//! measured result, not a default; the table is in the documentation of
//! [`KktSolver::condition_estimate`].
//!
//! *Cost.* `ν` in `O(nnz(K))`; between 4 and 11 solves (`O(nnz(F))` each,
//! about one to five steps of Algorithm F); three vectors of order `N`.
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
//!      iterate (step F4). Asserted in debug builds.
//! (S10) A factor returned by [`KktSolver::factorize`] holds only finite
//!      values (step K3′); [`KktSolver::solve_refined`] reports a
//!      non-finite solution explicitly: its `residual` is NaN or infinite
//!      exactly when the returned `x` has a non-finite entry, and then
//!      `converged = false` (the input `K`, `b` being finite).
//!
//! (K2) `Regularization::<f64>::default()` and `Refinement::<f64>::default()`
//!      are the values of the second edition (`10⁻⁶`, `10⁻⁸`, `10⁻¹³`,
//!      `2·10⁻⁷`; ten steps, `10⁻¹²`), bit for bit; for another scalar
//!      they are those values times `√ρ_T` or `ρ_T` as stated in
//!      "Choosing the regularization". Tested.
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
//! solver); Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
//! SIAM 2003, §6.5 and §9.3 (GMRES, right preconditioning, Givens
//! rotations); Arioli, Duff, Gratton & Pralet, "A note on GMRES
//! preconditioned by a perturbed `LDLᵀ` decomposition with static
//! pivoting", SIAM J. Sci. Comput. 29(5), 2007; Carson & Higham, "A new
//! analysis of iterative refinement and its application to accurate
//! solution of ill-conditioned sparse linear systems", SIAM J. Sci.
//! Comput. 39(6), 2017; Hager, "Condition estimates", SIAM J. Sci. Stat.
//! Comput. 5(2), 1984; Higham, "FORTRAN codes for estimating the one-norm
//! of a real or complex matrix, with applications to condition
//! estimation", ACM TOMS 14(4), 1988 (Algorithm H; LAPACK's `xLACN2` and
//! `xSYCON` — known from the literature, not copied); Higham (2002), ch.
//! 15; Saunders & Tomlin, "Solving regularized linear
//! programs using barrier methods and KKT systems", SOL 96-4, 1996 (the
//! product `ρδ` above the rounding level for a stable quasi-definite
//! factorization).

use crate::factor::count_f64;
use crate::real::{Real, eps_ratio, is_exact_zero, is_nan, le, lt};
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

/// Where the static shifts `ρ`, `δ` of a [`Regularization`] are applied
/// (step K2″ of Algorithm K).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShiftScale {
    /// On the diagonal of the equilibrated matrix `K̃ = S K S` (step K2′),
    /// whose rows have their largest entry near 1: the shifts are
    /// relative to the magnitude of each row, `K_reg = K + S⁻¹ Σ S⁻¹`.
    /// The default.
    #[default]
    Equilibrated,
    /// On the diagonal of `K` itself, in the caller's units:
    /// `K_reg = K + Σ`.
    Absolute,
}

/// Regularization of a quasi-definite KKT matrix (Altman & Gondzio 1999;
/// Algorithm K).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Regularization<T: Real = f64> {
    /// `ρ ≥ 0`, added to the primal block (which becomes `−D − ρI`, in
    /// the units [`Regularization::scale`] says); ignored for the normal
    /// equations.
    pub primal: T,
    /// `δ ≥ 0`, added to the dual block (`δI`), or to the whole diagonal
    /// of the normal equations (`A D Aᵀ + δI`), in the units
    /// [`Regularization::scale`] says.
    pub dual: T,
    /// Below this magnitude (or with the wrong sign) a pivot of the
    /// equilibrated matrix is bumped to `±bump` with the sign the inertia
    /// expects (dynamic regularization, step K3); `None` disables it.
    pub pivot_tol: Option<T>,
    /// The magnitude `β > 0` of a bumped pivot (of the equilibrated
    /// matrix).
    pub bump: T,
    /// Whether `ρ`, `δ` are taken in the equilibrated matrix (the default)
    /// or in `K` itself (step K2″).
    pub scale: ShiftScale,
}

impl<T: Real> Default for Regularization<T> {
    /// For `f64`: `ρ = 10⁻⁶`, `δ = 10⁻⁸` *in the equilibrated matrix*
    /// ([`ShiftScale::Equilibrated`]: relative to each row), and pivots of
    /// the wrong sign or of magnitude at most `τ = 10⁻¹³` bumped to
    /// `±β = ±2·10⁻⁷`. `ρδ = 10⁻¹⁴`, about 50 units of roundoff, keeps the
    /// regularized augmented matrix numerically quasi-definite; `δ`
    /// alone, small, is what the normal equations see (Chapter 5.14,
    /// "Choosing the regularization", measured in L1.5b). `τ`, `β` are of
    /// the order of the defaults of current interior-point codes
    /// (Clarabel's documented dynamic regularization; not tuned here).
    ///
    /// For another scalar `T` (task L1.8): `ρ`, `δ` and `β` times `√ρ_T`,
    /// `τ` times `ρ_T`, with `ρ_T = ε_T / ε₆₄` (Chapter 5.14, "The
    /// defaults and the scalar"; invariant (K2)).
    fn default() -> Self {
        let rho = eps_ratio::<T>();
        let root = rho.sqrt();
        Self {
            primal: T::from_f64(1e-6 * root),
            dual: T::from_f64(1e-8 * root),
            pivot_tol: Some(T::from_f64(1e-13 * rho)),
            bump: T::from_f64(2e-7 * root),
            scale: ShiftScale::Equilibrated,
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
            scale: ShiftScale::Equilibrated,
        }
    }

    /// Absolute static shifts relative to the largest entry of `k`:
    /// `ρ = δ = factor · max|k_ij|` with [`ShiftScale::Absolute`], and the
    /// dynamic part of [`Regularization::default`] (the L1.5 variant).
    ///
    /// One shift for the whole matrix: at the end of an interior-point
    /// method, where `max|k_ij| ≈ 10^{10}`, `factor = 10⁻⁸` gives
    /// `δ ≈ 10²`, a large perturbation of the rows of small magnitude —
    /// refinement against the unregularized `K` then converges slowly
    /// (Chapter 5.14, "Choosing the regularization"). The default
    /// ([`ShiftScale::Equilibrated`]) is relative row by row instead.
    #[must_use]
    pub fn relative_to(k: &CscMatrix<T>, factor: T) -> Self {
        let big = inf_norm(&k.val);
        Self {
            primal: factor * big,
            dual: factor * big,
            scale: ShiftScale::Absolute,
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
    /// Ten steps, target `10⁻¹² ρ_T` (`10⁻¹²` for `f64`; `ρ_T = ε_T /
    /// ε₆₄`, Chapter 5.14, "The defaults and the scalar").
    fn default() -> Self {
        Self {
            max_iters: 10,
            target_residual: T::from_f64(1e-12 * eps_ratio::<T>()),
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

/// Most iterations `k` of Algorithm H (steps H2–H3; Higham's `xLACN2` uses
/// five): a cost bound, not a threshold.
pub const CONDITION_ESTIMATE_STEPS: usize = 5;

/// An estimate of the 1-norm condition of a factored symmetric matrix
/// (Algorithm H of Chapter 5.14: Hager 1984, Higham 1988; the shape of
/// LAPACK's `xSYCON`), returned by [`KktSolver::condition_estimate`].
///
/// [`ConditionEstimate::inverse_norm`] is a *lower* bound of `‖F⁻¹‖₁`,
/// `F` the matrix the factor's solves invert (`K_reg`, or `K + E` for the
/// unregularized `LBLᵀ`), in practice within a factor 3 of it but not
/// always (H1′); so [`ConditionEstimate::rcond`] can only be too large,
/// never too small. Not a certificate.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ConditionEstimate<T: Real = f64> {
    /// `‖K‖₁` of the matrix passed (both triangles of the stored lower one;
    /// step H0).
    pub norm: T,
    /// The estimate of `‖F⁻¹‖₁` (`+∞` if a solve returned a non-finite
    /// vector).
    pub inverse_norm: T,
    /// Solves with the factor: `0` for `N = 0`, `1` for `N = 1`, otherwise
    /// from 4 to `2 ·` [`CONDITION_ESTIMATE_STEPS`] `+ 1`.
    pub solves: usize,
}

impl<T: Real> ConditionEstimate<T> {
    /// `1 / (‖K‖₁ · est ‖F⁻¹‖₁)`, clipped to `[0, 1]`: `0` when either norm
    /// is zero or the estimate is not finite (singular to working
    /// precision, as `xSYCON` returns), `1` for the empty matrix.
    pub fn rcond(&self) -> T {
        let (nu, gamma) = (self.norm, self.inverse_norm);
        if self.solves == 0 {
            return T::ONE;
        }
        if !nu.is_finite() || !gamma.is_finite() || !lt(T::ZERO, nu) || !lt(T::ZERO, gamma) {
            return T::ZERO;
        }
        (T::ONE / gamma / nu).min(T::ONE)
    }
}

/// Analyze/factorize/solve of symmetric systems with a fixed pattern (the
/// CHOLMOD shape, read, not copied). Regularization (its policy,
/// Algorithm K) and refinement (Algorithm F) are shared by every backend.
///
/// ```
/// use xingo_sparse::{
///     Amd, Col, CscMatrix, KktForm, KktSolver, Refinement, Regularization, Row, SupernodalLdl,
/// };
/// // K = [−D, Aᵀ; A, 0] with D = diag(1, 2), A = [1 1]: lower triangle.
/// let k = CscMatrix::from_triplets(3, 3, &[
///     (Row(0), Col(0), -1.0), (Row(2), Col(0), 1.0),
///     (Row(1), Col(1), -2.0), (Row(2), Col(1), 1.0),
/// ])
/// .unwrap();
/// let form = KktForm::Augmented { n_primal: 2, n_dual: 1 };
/// let solver = SupernodalLdl::<f64>::default(); // recommended; FaerLdl is the alternative
/// let sym = solver.analyze(&k.pattern(), form, &Amd::default()).unwrap();
/// let (num, inertia) = solver.factorize(&sym, &k, Regularization::default()).unwrap();
/// assert_eq!(inertia, form.expected_inertia(3)); // (S6): 1 positive, 2 negative
/// let b = [1.0, 0.0, 3.0];
/// let mut x = [0.0; 3];
/// let report = solver.solve_refined(&num, &k, &b, &mut x, Refinement::default());
/// assert!(report.converged && report.residual <= 1e-12);
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
    /// Allocates its work vectors; [`KktSolver::solve_refined_with`]
    /// reuses them.
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
        let mut work = RefineWorkspace::new();
        self.solve_refined_with(numeric, k, b, x, refinement, &mut work)
    }

    /// [`KktSolver::solve_refined`] with caller-owned work vectors: once
    /// `work` has served a call with the same order and budget, the
    /// refinement allocates nothing.
    ///
    /// # Panics
    /// If `b`, `x` and `k` do not have the order of the factor.
    #[allow(clippy::too_many_lines)] // one step of Algorithm F per block, as in the chapter
    fn solve_refined_with(
        &self,
        numeric: &Self::Numeric,
        k: &CscMatrix<T>,
        b: &[T],
        x: &mut [T],
        refinement: Refinement<T>,
        work: &mut RefineWorkspace<T>,
    ) -> RefineReport<T> {
        let n = b.len();
        assert_eq!(x.len(), n, "solve_refined: x has the wrong length");
        assert!(
            k.n_rows == n && k.n_cols() == n,
            "solve_refined: k has the wrong order"
        );
        let budget = refinement.max_iters;
        work.reserve(n, budget);
        let scale = T::ONE + inf_norm(b);
        let tol = refinement.target_residual * scale;
        // F1.
        x.copy_from_slice(b);
        self.solve(numeric, x);
        let RefineWorkspace {
            r,
            cur,
            v,
            z,
            h,
            g,
            cs,
            sn,
            y,
        } = work;
        residual(k, x, b, r);
        let first = inf_norm(r) / scale;
        let mut best = first;
        let mut t = 0;
        // `r` holds b − K x* at the top of every cycle.
        loop {
            // F2.
            if !best.is_finite() || le(best, refinement.target_residual, T::ZERO) || t >= budget {
                break;
            }
            // F3.
            let m = budget - t;
            let beta = norm2(r);
            if !beta.is_finite() || is_exact_zero(beta) {
                break;
            }
            for (vi, &ri) in v[..n].iter_mut().zip(r.iter()) {
                *vi = ri / beta;
            }
            g.fill(T::ZERO);
            g[0] = beta;
            let mut steps = 0;
            for j in 0..m {
                let (zj, vj) = (&mut z[j * n..(j + 1) * n], &v[j * n..(j + 1) * n]);
                zj.copy_from_slice(vj);
                self.solve(numeric, zj);
                t += 1;
                steps = j + 1;
                let (done, rest) = v.split_at_mut((j + 1) * n);
                let w = &mut rest[..n];
                sym_lower_matvec(k, &z[j * n..(j + 1) * n], w);
                // Column j of H, (m + 1) rows, column-major.
                let col = &mut h[j * (m + 1)..(j + 1) * (m + 1)];
                for i in 0..=j {
                    let vi = &done[i * n..(i + 1) * n];
                    let hij = dot(w, vi);
                    col[i] = hij;
                    for (wk, &vk) in w.iter_mut().zip(vi) {
                        *wk -= hij * vk;
                    }
                }
                let hnext = norm2(w);
                col[j + 1] = hnext;
                if !hnext.is_finite() {
                    steps = j;
                    break;
                }
                // Previous rotations on column j.
                for i in 0..j {
                    let (a, c) = (col[i], col[i + 1]);
                    col[i] = cs[i] * a + sn[i] * c;
                    col[i + 1] = cs[i] * c - sn[i] * a;
                }
                // New rotation annihilating h_{j+1,j}.
                let (a, c) = (col[j], col[j + 1]);
                let rr = norm2(&[a, c]);
                if is_exact_zero(rr) {
                    // R singular in this column: drop the step.
                    steps = j;
                    break;
                }
                cs[j] = a / rr;
                sn[j] = c / rr;
                col[j] = rr;
                col[j + 1] = T::ZERO;
                g[j + 1] = -sn[j] * g[j];
                g[j] *= cs[j];
                if is_exact_zero(hnext) || le(g[j + 1].abs(), tol, T::ZERO) {
                    break;
                }
                for wk in w.iter_mut() {
                    *wk = *wk / hnext;
                }
            }
            if steps == 0 {
                break;
            }
            // F4: back substitution R y = g.
            for i in (0..steps).rev() {
                let mut acc = g[i];
                for l in i + 1..steps {
                    acc -= h[l * (m + 1) + i] * y[l];
                }
                y[i] = acc / h[i * (m + 1) + i];
            }
            cur.copy_from_slice(x);
            for (l, &yl) in y[..steps].iter().enumerate() {
                for (c, &zl) in cur.iter_mut().zip(&z[l * n..(l + 1) * n]) {
                    *c += yl * zl;
                }
            }
            residual(k, cur, b, r);
            let res = inf_norm(r) / scale;
            if res.is_finite() && lt(res, best) {
                // `r` is the residual of the new x*.
                x.copy_from_slice(cur);
                best = res;
            } else {
                // F5 would restart from the same x*, whose cycle is the
                // one just done (deterministic): stop.
                break;
            }
        }
        // F6.
        debug_assert!(
            !best.is_finite() || !first.is_finite() || le(best, first, T::ZERO),
            "(S7): the returned residual exceeds the first one"
        );
        RefineReport {
            iters: t,
            residual: best,
            converged: best.is_finite() && le(best, refinement.target_residual, T::ZERO),
        }
    }

    /// Algorithm H (Chapter 5.14): a Hager–Higham estimate of `‖F⁻¹‖₁` by
    /// solves with the factor (`F` symmetric: no transposed solve), and
    /// `‖k‖₁` of the matrix passed (its lower triangle, both triangles
    /// counted). Only [`KktSolver::solve`] reads the factor, which is never
    /// changed (H2′); costs between 4 and 11 solves and allocates three
    /// vectors of order `N`; nothing is computed unless this is called.
    ///
    /// Not a proof: [`ConditionEstimate::inverse_norm`] is a lower bound of
    /// `‖F⁻¹‖₁` and can underestimate it (H1′). It measures the factored
    /// `F`: with a regularization, the shifts bound `‖F⁻¹‖`, and a singular
    /// `K` only shows as a large `1/rcond`; it is meant for the
    /// unregularized `LBLᵀ` (`Regularization::none()`), where a
    /// numerically singular `K` that factorizes without an error (issue #1)
    /// shows `rcond` of the order of the unit roundoff or below.
    ///
    /// *Measured (issue #1).* On the 279 unregularized augmented Netlib
    /// systems of the calibration (93 LPs, raw `A`, the start, middle and
    /// end regimes of `D`; SciPy's `κ₁(K) ≤ 10¹²` separates 49 nonsingular
    /// from 230 singular ones), the `LBLᵀ` of [`crate::SupernodalLdl`]
    /// (default kernel) factorizes the 49 and 17 of the 230 (the rest fail
    /// with an error). On the 49, `rcond` lies between `3.0·10⁻¹²`
    /// (`≈ 1.3·10⁴ u`) and `2.4·10⁻²`, and `rcond · κ₁(K)` between 1.0 and
    /// 2.9; on the 17, it is at most `1.7·10⁻¹³` (`≈ 780 u`), and 14 of
    /// them, all from the end regime, are below `3·10⁻²⁰`.
    ///
    /// | threshold `rcond ≤ c u` | singular flagged (of 17) | nonsingular flagged (of 49) |
    /// |---|---:|---:|
    /// | `c = 1, 10, 10²` | 14 | 0 |
    /// | `c = 10³` (`≈ 2.2·10⁻¹³`) | 17 | 0 |
    /// | `c = 10⁴` (`≈ 2.2·10⁻¹²`) | 17 | 0 |
    ///
    /// On this calibration, `rcond ≤ 10³ u` separated the two groups
    /// (1.3× above the largest singular `rcond`, 13× below the smallest
    /// nonsingular one). That is a measured result, not a behaviour or a
    /// default: this method applies no threshold, and which `rcond` to call
    /// singular is the caller's decision (the set is small and its `D` are
    /// synthetic).
    ///
    /// ```
    /// use xingo_sparse::{
    ///     Amd, Col, CscMatrix, KktForm, KktSolver, Regularization, Row, SupernodalLdl,
    /// };
    /// // K = [−1, 1; 1, ε]: an indefinite 2×2 with ‖K⁻¹‖₁ = 2/(1+ε) ≈ 2.
    /// let eps = 1e-3;
    /// let k = CscMatrix::from_triplets(2, 2, &[
    ///     (Row(0), Col(0), -1.0), (Row(1), Col(0), 1.0), (Row(1), Col(1), eps),
    /// ])
    /// .unwrap();
    /// let form = KktForm::Augmented { n_primal: 1, n_dual: 1 };
    /// let solver = SupernodalLdl::<f64>::default();
    /// let sym = solver.analyze(&k.pattern(), form, &Amd::default()).unwrap();
    /// let (num, _) = solver.factorize(&sym, &k, Regularization::none()).unwrap();
    /// let c = solver.condition_estimate(&num, &k);
    /// assert_eq!(c.norm, 2.0);
    /// assert!((c.inverse_norm - 2.0 / (1.0 + eps)).abs() < 1e-12);
    /// assert!((c.rcond() - (1.0 + eps) / 4.0).abs() < 1e-12);
    /// ```
    ///
    /// # Panics
    /// If `k` is not square, or not of the order of the factor (in
    /// [`KktSolver::solve`]).
    fn condition_estimate(
        &self,
        numeric: &Self::Numeric,
        k: &CscMatrix<T>,
    ) -> ConditionEstimate<T> {
        hager_higham(k, |x| self.solve(numeric, x))
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

/// `‖v‖₁` (NaN or `+∞` if an entry is not finite).
fn norm1<T: Real>(v: &[T]) -> T {
    let mut s = T::ZERO;
    for &x in v {
        s += x.abs();
    }
    s
}

/// The first index of the largest `|v_i|` (`v` finite and nonempty).
fn argmax_abs<T: Real>(v: &[T]) -> usize {
    let mut j = 0;
    for (i, &x) in v.iter().enumerate() {
        if lt(v[j].abs(), x.abs()) {
            j = i;
        }
    }
    j
}

/// `ξ ← sign(y)` (`−1` for `y_i < 0`, else `+1`); returns whether `ξ`
/// was already that (step H2's repetition test).
fn set_signs<T: Real>(y: &[T], xi: &mut [T]) -> bool {
    let mut same = true;
    for (s, &v) in xi.iter_mut().zip(y) {
        let neg = lt(v, T::ZERO);
        same &= neg == lt(*s, T::ZERO);
        *s = if neg { -T::ONE } else { T::ONE };
    }
    same
}

/// Step H0: `‖K‖₁` from the lower triangle of `k` (both triangles), with
/// `col` (of order `N`) as the work array of the column sums.
fn sym_lower_norm1<T: Real>(k: &CscMatrix<T>, col: &mut [T]) -> T {
    let n = k.n_cols();
    col.fill(T::ZERO);
    for j in 0..n {
        for p in k.col_ptr[j]..k.col_ptr[j + 1] {
            let i = k.row_idx[p] as usize;
            if i < j {
                continue;
            }
            let v = k.val[p].abs();
            col[j] += v;
            if i != j {
                col[i] += v;
            }
        }
    }
    let mut m = T::ZERO;
    for &c in col.iter() {
        if is_nan(c) {
            return c;
        }
        m = m.max(c);
    }
    m
}

/// Algorithm H with `solve(x)`: `x ← F⁻¹ x` (Chapter 5.14).
fn hager_higham<T: Real>(
    k: &CscMatrix<T>,
    mut solve: impl FnMut(&mut [T]),
) -> ConditionEstimate<T> {
    let n = k.n_rows;
    assert_eq!(k.n_cols(), n, "condition_estimate: k is not square");
    // The three vectors of order N; `z` is first the work array of H0.
    let mut y = vec![T::ZERO; n];
    let mut xi = vec![T::ZERO; n];
    let mut z = vec![T::ZERO; n];
    // H0.
    let norm = sym_lower_norm1(k, &mut z);
    let mut out = ConditionEstimate {
        norm,
        inverse_norm: T::ZERO,
        solves: 0,
    };
    if n == 0 {
        return out;
    }
    let infinite = T::from_f64(f64::INFINITY);
    let nt = T::from_f64(count_f64(n));
    y.fill(T::ONE / nt);
    // One solve in place; `false` (and the estimate +∞) if not finite.
    let mut apply = |v: &mut [T], out: &mut ConditionEstimate<T>| -> Option<T> {
        solve(v);
        out.solves += 1;
        let s = norm1(v);
        if s.is_finite() {
            Some(s)
        } else {
            out.inverse_norm = infinite;
            None
        }
    };
    // H1.
    let Some(mut gamma) = apply(&mut y, &mut out) else {
        return out;
    };
    if n == 1 {
        out.inverse_norm = gamma;
        return out;
    }
    set_signs(&y, &mut xi);
    z.copy_from_slice(&xi);
    if apply(&mut z, &mut out).is_none() {
        return out;
    }
    let mut j = argmax_abs(&z);
    let mut iter = 2;
    loop {
        // H2.
        y.fill(T::ZERO);
        y[j] = T::ONE;
        let Some(g) = apply(&mut y, &mut out) else {
            return out;
        };
        let progress = lt(gamma, g);
        gamma = gamma.max(g);
        if set_signs(&y, &mut xi) || !progress {
            break;
        }
        // H3.
        z.copy_from_slice(&xi);
        if apply(&mut z, &mut out).is_none() {
            return out;
        }
        let last = j;
        j = argmax_abs(&z);
        if is_exact_zero(z[last] - z[j].abs()) || iter >= CONDITION_ESTIMATE_STEPS {
            break;
        }
        iter += 1;
    }
    // H4.
    let denom = T::from_f64(count_f64(n - 1));
    for (i, v) in y.iter_mut().enumerate() {
        let a = T::ONE + T::from_f64(count_f64(i)) / denom;
        *v = if i % 2 == 0 { a } else { -a };
    }
    let Some(g) = apply(&mut y, &mut out) else {
        return out;
    };
    let alt = (g + g) / (T::from_f64(3.0) * nt);
    out.inverse_norm = gamma.max(alt);
    debug_assert!(
        out.solves <= 2 * CONDITION_ESTIMATE_STEPS + 1,
        "Algorithm H: {} solves",
        out.solves
    );
    out
}

/// `‖v‖₂`, scaled against overflow (NaN propagates).
fn norm2<T: Real>(v: &[T]) -> T {
    let big = inf_norm(v);
    if !big.is_finite() || is_exact_zero(big) {
        return big;
    }
    let mut s = T::ZERO;
    for &x in v {
        let q = x / big;
        s += q * q;
    }
    big * s.sqrt()
}

/// `⟨u, v⟩`.
fn dot<T: Real>(u: &[T], v: &[T]) -> T {
    let mut s = T::ZERO;
    for (&a, &b) in u.iter().zip(v) {
        s += a * b;
    }
    s
}

/// Work vectors of Algorithm F, reusable across calls of
/// [`KktSolver::solve_refined_with`]: `(2m + 3) N + O(m²)` scalars for a
/// budget of `m = max_iters` solves and order `N`. Grows on demand, never
/// shrinks.
#[derive(Clone, Debug, Default)]
pub struct RefineWorkspace<T: Real = f64> {
    r: Vec<T>,
    cur: Vec<T>,
    v: Vec<T>,
    z: Vec<T>,
    h: Vec<T>,
    g: Vec<T>,
    cs: Vec<T>,
    sn: Vec<T>,
    y: Vec<T>,
}

impl<T: Real> RefineWorkspace<T> {
    /// An empty workspace (it grows at the first call).
    #[must_use]
    pub fn new() -> Self {
        Self {
            r: Vec::new(),
            cur: Vec::new(),
            v: Vec::new(),
            z: Vec::new(),
            h: Vec::new(),
            g: Vec::new(),
            cs: Vec::new(),
            sn: Vec::new(),
            y: Vec::new(),
        }
    }

    /// Sizes the vectors for order `n` and budget `m` (allocates only when
    /// they grow).
    fn reserve(&mut self, n: usize, m: usize) {
        fn fit<T: Real>(v: &mut Vec<T>, len: usize) {
            if v.len() != len {
                v.resize(len, T::ZERO);
            }
        }
        fit(&mut self.r, n);
        fit(&mut self.cur, n);
        fit(&mut self.v, (m + 1) * n);
        fit(&mut self.z, m * n);
        fit(&mut self.h, (m + 1) * m);
        fit(&mut self.g, m + 1);
        fit(&mut self.cs, m);
        fit(&mut self.sn, m);
        fit(&mut self.y, m);
    }
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
pub(crate) fn pivot_signs(form: KktForm, n: usize) -> Vec<i8> {
    match form {
        KktForm::NormalEquations => vec![1; n],
        KktForm::Augmented { n_primal, .. } => {
            (0..n).map(|k| if k < n_primal { -1 } else { 1 }).collect()
        }
    }
}

/// Steps K1′–K2″: scatters the lower triangle of `k` into the values of
/// the analyzed lower pattern `l` (which has a full diagonal), equilibrates
/// it (K2′) and adds the static shifts (K2″); returns the values of
/// `K̃_reg` and the scaling `s`. Fails (K1, K1′) if `k` does not satisfy
/// (S1), has the wrong order, has a lower entry outside `l`, or a
/// non-finite lower entry.
pub(crate) fn assemble_regularized<T: Real>(
    l: &Pattern,
    form: KktForm,
    k: &CscMatrix<T>,
    reg: &Regularization<T>,
) -> Result<(Vec<T>, Vec<T>), KktError> {
    let mut val = assemble(l, k)?;
    let s = equilibrate_symmetric(l, &mut val);
    for (j, &sj) in s.iter().enumerate() {
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
            // The diagonal is the first entry of column j of l.
            let p = l.col_ptr[j];
            val[p] += match reg.scale {
                ShiftScale::Equilibrated => shift,
                ShiftScale::Absolute => shift * sj * sj,
            };
        }
    }
    Ok((val, s))
}

/// Steps K1′–K2: the lower triangle of `k` on the analyzed pattern `l`.
fn assemble<T: Real>(l: &Pattern, k: &CscMatrix<T>) -> Result<Vec<T>, KktError> {
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
            // K1′.
            if !k.val[p].is_finite() {
                return Err(KktError::Structure(SparseError::InvalidValue {
                    what: "KKT matrix entry (not finite)",
                    index: p,
                    value: k.val[p].to_f64(),
                }));
            }
            val[base + q] = k.val[p];
        }
        // The diagonal is the first entry of column j of l.
        debug_assert_eq!(lrows[0] as usize, j);
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
pub(crate) fn check_inertia(found: Inertia, expected: Inertia) -> Result<Inertia, KktError> {
    if found == expected {
        Ok(found)
    } else {
        Err(KktError::WrongInertia { found, expected })
    }
}

#[cfg(test)]
// Exact equality is the property under test here (the assembly and the
// scaling are products and sums of powers of two, bit for bit); clippy 1.90
// flags it, 1.99 does not.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::{Col, Row};

    /// Invariant (K2): the `f64` defaults of the second edition bit for
    /// bit, and for the test-only `f32` (`ρ = 2²⁹`) the same values times
    /// `√ρ` or `ρ`, computed in `f64`.
    #[test]
    #[allow(clippy::float_cmp, clippy::cast_possible_truncation)] // exact identities
    fn defaults_follow_the_scalar() {
        let r = Regularization::<f64>::default();
        assert_eq!(
            (r.primal.to_bits(), r.dual.to_bits(), r.bump.to_bits()),
            (1e-6_f64.to_bits(), 1e-8_f64.to_bits(), 2e-7_f64.to_bits())
        );
        assert_eq!(r.pivot_tol.map(f64::to_bits), Some(1e-13_f64.to_bits()));
        assert_eq!(r.scale, ShiftScale::Equilibrated);
        let f = Refinement::<f64>::default();
        assert_eq!(
            (f.max_iters, f.target_residual.to_bits()),
            (10, 1e-12_f64.to_bits())
        );
        let rho = f64::from(f32::EPSILON) / f64::EPSILON;
        let root = rho.sqrt();
        let r = Regularization::<f32>::default();
        assert_eq!(r.primal, (1e-6 * root) as f32);
        assert_eq!(r.dual, (1e-8 * root) as f32);
        assert_eq!(r.pivot_tol, Some((1e-13 * rho) as f32));
        assert_eq!(r.bump, (2e-7 * root) as f32);
        let f = Refinement::<f32>::default();
        assert_eq!((f.max_iters, f.target_residual), (10, (1e-12 * rho) as f32));
    }

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
        let mut reg = Regularization::<f64> {
            primal: 0.5,
            dual: 0.25,
            pivot_tol: None,
            bump: 0.0,
            scale: ShiftScale::Absolute,
        };
        // S⁻¹ K̃_reg S⁻¹ = K + Σ exactly (powers of two), whatever S.
        let unscale = |(v, s): (Vec<f64>, Vec<f64>)| -> Vec<f64> {
            let mut u = Vec::new();
            for j in 0..l.n_cols() {
                for p in l.col_ptr[j]..l.col_ptr[j + 1] {
                    u.push(v[p] / (s[l.row_idx[p] as usize] * s[j]));
                }
            }
            u
        };
        let v = assemble_regularized(&l, form, &k, &reg).unwrap();
        assert!(v.1.iter().all(|&x| x > 0.0 && x.log2().fract() == 0.0));
        assert_eq!(unscale(v), vec![-1.5, 1.0, -2.5, 0.25]);
        // Equilibrated: K + S⁻¹ Σ S⁻¹.
        reg.scale = ShiftScale::Equilibrated;
        let (v, s) = assemble_regularized(&l, form, &k, &reg).unwrap();
        let u = unscale((v, s.clone()));
        assert_eq!(u[0], -1.0 - 0.5 / (s[0] * s[0]));
        assert_eq!(u[1], 1.0);
        assert_eq!(u[2], -2.0 - 0.5 / (s[1] * s[1]));
        assert_eq!(u[3], 0.25 / (s[2] * s[2]));
        // A non-finite entry is refused (K1′).
        let mut nan = k.clone();
        nan.val[1] = f64::NAN;
        assert!(matches!(
            assemble_regularized(&l, form, &nan, &reg),
            Err(KktError::Structure(_))
        ));
        let other = CscMatrix::from_triplets(3, 3, &[(Row(2), Col(1), 1.0)]).unwrap();
        assert!(matches!(
            assemble_regularized(&l, form, &other, &reg),
            Err(KktError::Structure(_))
        ));
    }

    /// `‖M‖₁` and the operator `x ← M x` of a dense row-major `M`.
    fn dense_norm1(m: &[Vec<f64>]) -> f64 {
        (0..m.len())
            .map(|j| m.iter().map(|r| r[j].abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    fn apply_dense(m: &[Vec<f64>], x: &mut [f64]) {
        let y: Vec<f64> = m
            .iter()
            .map(|r| r.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
            .collect();
        x.copy_from_slice(&y);
    }

    /// A lower-triangle `k` of order `n` with a diagonal of ones: Algorithm
    /// H reads its norm only, the operator is the closure.
    fn identity(n: usize) -> CscMatrix {
        let t: Vec<_> = (0..n)
            .map(|i| (Row::from_usize(i), Col::from_usize(i), 1.0))
            .collect();
        CscMatrix::from_triplets(n, n, &t).unwrap()
    }

    /// Algorithm H, (H1′) and the factor of Higham (1988), on explicit
    /// symmetric operators: random dense symmetric `M` of order 2–30 with
    /// entries of mixed signs and magnitudes. The estimate never exceeds
    /// `‖M‖₁` (beyond the rounding of the sums), it is within a factor 10 in
    /// at least 99 % of the cases (measured: see the counts below), and the
    /// number of solves stays in `[4, 2·STEPS + 1]`.
    #[test]
    fn hager_higham_is_a_lower_bound_within_a_small_factor() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut unit = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            let u = (state >> 11) as f64 / (1_u64 << 53) as f64;
            u
        };
        let (mut total, mut within3, mut within10) = (0, 0, 0);
        for case in 0..2000 {
            let n = 2 + case % 29;
            let mut m = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..=i {
                    let mag = 10f64.powf(4.0 * unit() - 2.0);
                    let v = if unit() < 0.5 { -mag } else { mag };
                    let v = if unit() < 0.3 { 0.0 } else { v };
                    m[i][j] = v;
                    m[j][i] = v;
                }
            }
            let exact = dense_norm1(&m);
            if exact == 0.0 {
                continue;
            }
            let c = hager_higham(&identity(n), |x| apply_dense(&m, x));
            assert!(
                c.inverse_norm <= exact * (1.0 + 1e-12),
                "case {case}: estimate {} above ‖M‖₁ = {exact}",
                c.inverse_norm
            );
            assert!(
                (4..=2 * CONDITION_ESTIMATE_STEPS + 1).contains(&c.solves),
                "{} solves",
                c.solves
            );
            total += 1;
            within3 += usize::from(c.inverse_norm * 3.0 >= exact);
            within10 += usize::from(c.inverse_norm * 10.0 >= exact);
        }
        // Measured (task I05): 1 999 of the 1 999 nonzero cases within a
        // factor 3; the assertion is the documented 99 % within 10.
        assert!(
            within10 * 100 >= total * 99,
            "{within10} of {total} within 10"
        );
        assert!(within3 <= within10);
    }

    /// Algorithm H on the edge cases: `N = 0` (no solve, `rcond = 1`),
    /// `N = 1` (one solve, exact), a diagonal operator (exact: the maximum
    /// is found at the first unit vector), a non-finite solve (`+∞`,
    /// `rcond = 0`), a zero matrix (`rcond = 0`).
    #[test]
    fn hager_higham_edge_cases() {
        let empty = CscMatrix::<f64>::zeros(0, 0);
        let c = hager_higham(&empty, |_| unreachable!());
        assert_eq!((c.solves, c.norm, c.inverse_norm), (0, 0.0, 0.0));
        assert_eq!(c.rcond(), 1.0);
        let k = CscMatrix::from_triplets(1, 1, &[(Row(0), Col(0), -4.0)]).unwrap();
        let c = hager_higham(&k, |x| x[0] /= -4.0);
        assert_eq!((c.solves, c.norm, c.inverse_norm), (1, 4.0, 0.25));
        assert_eq!(c.rcond(), 1.0);
        let d = [3.0, -0.5, 8.0, 0.25, -2.0];
        let t: Vec<_> = d
            .iter()
            .enumerate()
            .map(|(i, &v)| (Row::from_usize(i), Col::from_usize(i), v))
            .collect();
        let k = CscMatrix::from_triplets(5, 5, &t).unwrap();
        let c = hager_higham(&k, |x| {
            for (xi, di) in x.iter_mut().zip(d) {
                *xi /= di;
            }
        });
        assert_eq!((c.norm, c.inverse_norm), (8.0, 4.0));
        assert_eq!(c.rcond(), 1.0 / 32.0);
        let c = hager_higham(&k, |x| x[2] = f64::NAN);
        assert_eq!((c.solves, c.inverse_norm), (1, f64::INFINITY));
        assert_eq!(c.rcond(), 0.0);
        let mut calls = 0;
        let c = hager_higham(&k, |x| {
            calls += 1;
            if calls == 3 {
                x[0] = f64::INFINITY;
            }
        });
        assert_eq!((c.solves, c.inverse_norm), (3, f64::INFINITY));
        let zero = CscMatrix::<f64>::zeros(3, 3);
        let c = hager_higham(&zero, |_| {});
        assert_eq!(c.norm, 0.0);
        assert_eq!(c.rcond(), 0.0);
    }

    /// Step H0: `‖K‖₁` counts both triangles of the stored lower one and
    /// ignores entries above the diagonal.
    #[test]
    fn condition_norm_reads_the_lower_triangle_only() {
        let k = CscMatrix::from_triplets(
            2,
            2,
            &[
                (Row(0), Col(0), 2.0),
                (Row(1), Col(0), -3.0),
                (Row(0), Col(1), 99.0),
                (Row(1), Col(1), 4.0),
            ],
        )
        .unwrap();
        let mut work = [0.0; 2];
        assert_eq!(sym_lower_norm1(&k, &mut work), 7.0);
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
