# Changelog

All notable changes to `xingo-sparse` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate
follows [Semantic Versioning](https://semver.org/); before 1.0, a minor
version may change the public API.

## [0.4.0] — 2026-11-17

Scalars other than `f64` (task L1.8 of the Xingó plan, D-0113): the
defaults follow the scalar, the solves of `SupernodalLdl` can run in the
dense kernel, and the Forrest–Tomlin update bounds the multipliers of its
row etas. In `f64` every solver path is unchanged where a default is
unchanged; see "Breaking" for what is not.

### Added

* `LuOptions::for_scalar::<T>()`: the defaults for the scalar `T` (Rule P,
  Chapter 5.3) — `drop_tol`, `abs_pivot_tol` and `update_stability_tol`,
  the three thresholds that are magnitudes of rounding noise, multiplied by
  `ε_T / ε₆₄`; the identity for `f64` (invariant (O1)).
* `LuOptions::update_multiplier_limit` (default `1e6`) and step U4′ of the
  Forrest–Tomlin update (Chapter 5.5): an update whose row eta needs a
  multiplier `|μ_k|` above the limit is not applied and returns
  `RefactorNeeded(RefactorReason::Unstable)`. The stability test of step
  U4 compares two orders of summation of the same terms and does not see
  large multipliers: measured in `f64` with the 0.3.x defaults, updated
  factors up to `5.5·10⁹` times less accurate than fresh ones
  (`4·10¹⁶` in double-double); with the default limit `3.6·10⁷` and
  `8.7·10⁷` on the same grid (the experiment is in the documentation of
  the field). `f64::INFINITY` restores the 0.3.x update exactly.
* `UpdateStats::max_multiplier`: the largest `|μ_k|` stored since the last
  factorization (invariant (F4)).
* `DenseKernel::{solve_min_rows, solve_unit_lower, solve_unit_lower_t}`,
  provided methods (Algorithms V and W of Chapter 5.16): the two halves of
  a supernode's solve, which `SupernodalLdl` sends through the kernel for
  the blocks of at least `solve_min_rows()` rows (step S0′ of Chapter
  5.17; `usize::MAX`, the default, keeps the loops). The provided methods
  return the bits of the 0.3.x solve (invariant (D4)); a kernel that
  overrides them should too. `NaiveDense` and `FaerDense` do not override
  them. Supernodes of one column of the `LDLᵀ` path keep their loops.

### Changed

* `Lu::<T>::default()` uses `LuOptions::for_scalar::<T>()` (for `f64`,
  `LuOptions::default()` as before). `LuOptions::default()` itself is
  unchanged and stays calibrated for `f64`.
* `Regularization::<T>::default()` and `Refinement::<T>::default()` follow
  the scalar (Chapter 5.14): the shifts and the bump times `√(ε_T/ε₆₄)`,
  the pivot tolerance and the refinement target times `ε_T/ε₆₄`. The
  `f64` values are those of 0.3.x, bit for bit (invariant (K2)).
  `SupernodalOptions` does not change (Chapter 5.17).
* `NaiveDense::potrf` runs by columns (left-looking, Algorithm P of
  Chapter 5.16: `a_ij ← a_ij − l_ik · l_jk` term by term instead of a sum
  formed apart): vectorizable with the same bits. Its factor differs from
  0.3.x's in rounding from the third column on (and possibly in the sign
  of a zero); no bound changes. Nothing in the crate calls `potrf`, so no
  factorization or solve of the crate changes.
* The work vector of `SupernodalLdl`'s solve has `N + max_s n_r(s)`
  entries (allocated by the factorization, as before: the solve still
  allocates nothing).
* Documentation: requirement (R1) of `Real` asks for operations with a
  relative error of at most `c_T · ε_T`, `c_T ≤ 4`, deterministic — not
  correctly rounded ones (the double-double of `xingo-extprec` is faithful
  with proven bounds; D-0109, E2).

### Breaking

* `LuOptions` and `UpdateStats` have a new public field each
  (`update_multiplier_limit`, `max_multiplier`): a struct literal without
  `..Default::default()` and an exhaustive destructuring no longer compile.
* The default multiplier limit changes the behaviour of `Lu` in `f64`: an
  update with a multiplier above `10⁶` now asks for a refactorization
  instead of being applied. A caller that needs the 0.3.x sequence of
  updates sets `update_multiplier_limit: f64::INFINITY` (the simplex method
  of `xingo-simplex` does: on 7 of the 93 Netlib problems the limit would
  refuse updates, 126 in all, and every refusal changes its path).
* For a scalar other than `f64`, `Lu::default()`, `Regularization::default()`
  and `Refinement::default()` return other values than in 0.3.x.

### Fixed

* `SupernodalLdl` with `Regularization::none()` (the `LBLᵀ` path, Algorithm
  M of Chapter 5.17): on the kernel path of step M2 (supernodes of at least
  `SupernodalOptions::min_supernode_cols` columns) the reference scale of
  a diagonal was the *sum* of each descendant's update, not its largest
  term as on the simplicial path. A pair of a primal column and a dual
  leaf delayed into its block cancels exactly on every other dual row
  coupled to the primal, the sum is then rounding noise, and a singular
  matrix (rank-deficient `A`) was factorized with the expected inertia
  instead of returning `KktError::ZeroPivot`. The kernel path now records
  the same terms as the simplicial path (invariant (L10)); the factor of
  every matrix the old scale did not misjudge is unchanged bit for bit.
  Present since 0.3.0; found by the crate's property test (`m = 10`,
  `n = 10`, seed `15906914706389704250`, every supernode on the kernel
  path), kept as a deterministic test with a `5 × 5` test of the structure.

* `SupernodalLdl`, `LBLᵀ` path: a structurally singular `K` (no perfect
  matching on the nonzero entries — for an augmented system, `A` of
  structural rank below its row count) is refused with
  `KktError::ZeroPivot` even when the numerical test of step M3 let every
  pivot through (new step M7 of Chapter 5.17, after N6; an error the
  numerical steps report keeps its kind and its row, and a factor that
  passes is unchanged bit for bit). All 557 singular systems of 20 000
  random cases of the property test are structurally singular; before this
  step 11 of them were factorized with the expected inertia. The saved
  regression line `cc b6aceb…` (L1.9; relaxed supernodes) is in the
  regression file, and 16 seeds are a deterministic test.

### Known limitations

* The `LBLᵀ` path can still factorize a singular matrix without an error
  when `K` is structurally nonsingular and numerically singular (the
  Netlib `A` of etamacro: full structural rank, rank 399 of 400 by SVD),
  the zero pivot comes out as rounding noise above
  `SupernodalOptions::zero_pivot_tol` times its reference scale and with
  the sign the inertia expects. This is not a defect of the scale that a
  better scale removes: a backward-stable `LBLᵀ` is the exact factor of
  `K + E` with `E` on the pattern of the factor, and `K + E` is in general
  nonsingular (measured: a computed factor that is, to four digits, the
  exact factor of a nonsingular matrix `0.95 u` away from `K` in one dual
  diagonal entry; task L1.10). Four stronger scales (first-order error
  bounds through the rows of `L⁻¹`, the uncertainty of a noise diagonal of
  a `2×2` block, rook pivoting, a stricter `pivot_threshold`) catch more
  but each makes nonsingular Netlib systems fail with `ZeroPivot`, and
  were not adopted. The default regularization is not affected.

## [0.3.2] — 2026-11-14

### Changed

* Documentation only: the README (and the landing page) gain the section
  "Authorship, credits and disclaimer" — who writes the crate, the authors
  of the algorithms it implements and of the projects studied as
  references, and that this is an independent project provided "as is".
  No change to the code; 0.3.1 was released on GitHub and Zenodo but not
  on crates.io, so this is the version that carries its fix there.

## [0.3.1] — 2026-11-14

### Fixed

* `SupernodalLdl`, `LBLᵀ` path (`Regularization::none()`): a delayed
  column now carries, in its reference scale, the terms of the Schur
  complement formed when it is handed to the parent supernode (Chapter
  5.17, step M5). The pivots a supernode takes *after* flagging a column
  could cancel in that sum, and the rounding noise left (`4·10⁻¹⁶` beside
  terms of order one) was judged against a scale of zero and taken as a
  pivot: on a singular augmented system — two singleton rows of `A` on the
  same column — the factorization returned the expected inertia instead of
  `KktError::ZeroPivot`, for one of the three option sets the tests
  exercise. Found by the random KKT systems of the test suite; the seed is
  now a saved regression case. Factors of nonsingular systems are
  unchanged.

### Changed

* Tests only: the final oracle check of `hypersparse_solves_match_dense`
  follows the growth the updates added (the tolerance its loop already
  used), never below `10⁻⁹` — after 22 Forrest–Tomlin updates with growth
  `8.5·10⁶`, below `growth_limit`, an error of `1.6·10⁻⁹ ≈ ε · growth` on a
  basis with `κ ≈ 10³` is the expected accuracy, not a defect.
* `CITATION.cff` lists the two licenses as CFF 1.2.0 requires and a
  `.zenodo.json` carries the archive metadata (releases 0.2.0 and 0.3.0
  were not archived by Zenodo: "Citation metadata load failed").

## [0.3.0] — 2026-11-10

### Added

* `SupernodalLdl` factorizes the **unregularized augmented system**
  (`Regularization::none()`) with an `LBLᵀ` (Chapter 5.17, Algorithm M):
  Bunch–Kaufman `1×1`/`2×2` pivots inside each supernode through
  `DenseKernel::sytrf`, and *delayed pivots* — a column with no acceptable
  pivot in its supernode (a dual leaf of the elimination tree, or a pivot
  that is rounding noise or would make `L` grow past `1/u`) moves to the
  parent supernode and is pivoted there, as in MA57. A zero pivot at a root
  means the matrix is singular to working precision: `KktError::ZeroPivot`,
  never a silent infinity. New options `SupernodalOptions::zero_pivot_tol`
  (`10⁻¹⁴`, relative to the largest term that produced the pivot) and
  `SupernodalOptions::pivot_threshold` (`u = 10⁻⁸`); new accessors
  `NumericLdl::{is_pivoted, delayed, supernodes, rows}`, and
  `NumericLdl::perm` returns the pivot order of the factor. `pivot_2x2` is
  `true` on both columns of a `2×2` block; its off-diagonal is stored in
  the block at `(k + 1, k)`. The regularized systems take the `LDLᵀ` of
  0.2.0 unchanged (bit for bit on the 558 Netlib systems with `FaerDense`).
* `NaiveDense::{gemm_sub, trsm_lower, syrk_sub}` run contiguous blocked
  loops (`GEMM_BLOCK_ROWS`, `GEMM_BLOCK_K`) instead of per-entry sums:
  `SupernodalLdl<f64, NaiveDense>` factorizes the ten largest Netlib
  systems in 1.37× the time of `FaerLdl` (was 3.3×; worst 3.3×,
  was 14.5×). The rounding of the naive kernels changes; the faer kernels
  do not.

### Measured, not changed

* Fill of our `Amd` against METIS nested dissection (external tool) on 40
  KKT systems: 0.97× on average (D11 of the solver's plan closed without
  a nested dissection of our own).
* A `DenseKernel` with a leading dimension (to avoid the scratch copies of
  the supernodal kernel path) and `PANEL = 64` were measured and not
  adopted: the scratch copies of the kernel path take 22–28 % of the factorization time with `FaerDense` on dfl001, fit2p and pilot87 (13 % with `NaiveDense`), an upper bound on what a strided kernel could save, and the change is a new public method of `DenseKernel`; left for L2 with the measurement. `PANEL = 64` gains 17 % on fit2p with faer's kernels and loses 11 % with the naive ones.

## [0.2.0] — 2026-11-10

`SupernodalLdl` becomes the recommended KKT backend; `FaerLdl` stays as the
alternative and the oracle of the test suite.

### Added

* `SupernodalLdl<T, K>`, our own supernodal LDLᵀ behind `KktSolver`, with
  no dependency: elimination tree and relaxed supernodes (CHOLMOD's
  amalgamation rule, `SupernodalOptions::relax_zeros`), a left-looking
  supernodal factorization with `1×1` pivots on the dense kernels
  (`NaiveDense` always, `FaerDense` with the `faer` feature), the pivot
  bumping of the regularization policy with the sign each block expects
  and an **exact** count of the bumped pivots of both signs, the
  simplicial/supernodal switch of CHOLMOD (`supernodal_switch`), solves
  without allocation, generic in the scalar `T`. The symbolic
  (`SymbolicLdl`) and numeric (`NumericLdl`) factors expose the
  permutation, the supernode partition, the row patterns and the blocks.
  `FaerLdl` stays as the optional backend and is the oracle of the tests.
* The `KktSolver` documentation and the `kkt_ipm_step` example run on
  `SupernodalLdl` (and, with `faer`, on `FaerLdl` too); the example no
  longer requires the `faer` feature.
* `CITATION.cff` and the Zenodo DOI (concept 10.5281/zenodo.24095312;
  0.1.0 is 10.5281/zenodo.24095313) in the README.

### Changed

* The approximate-minimum-degree engine behind `Amd` and `Colamd` keeps
  its lists in one garbage-collected workspace (as SuiteSparse's AMD does)
  instead of one vector per node, and the orderings hand it compressed
  lists: no allocation per node. The output is unchanged bit for bit
  (checked by a hash of the permutations on the 93 Netlib LPs, three
  patterns each); the ordering time is reported in the decision log.

### Known limitations

* `SupernodalLdl` factorizes the two *regularized* systems (`1×1`
  pivots); the Bunch–Kaufman `2×2` pivots for an unregularized augmented
  system are the next step (`pivot_2x2` is all `false`).

## [0.1.0] — 2026-11-09

The first release: the sparse linear algebra of the Xingó solver, split
out as a crate that depends on nothing in the solver.

### Added

* Containers generic in the scalar `T: Real` (`f64` by default, `u32`
  indices): `CscMatrix`, `CsrMatrix`, the work vector `SparseVec` with an
  explicit pattern, the coordinate builder `Triplets` (duplicates summed in
  insertion order), the value-free `Pattern`, `BlockPartition`, and the
  typed indices `Col`/`Row`.
* The dynamic LU `Lu<T, U>` behind `LuFactor`: singleton phase and
  threshold Markowitz pivoting (Suhl & Suhl 1990), dense and hypersparse
  FTRAN/BTRAN chosen per call by a density policy (Gilbert & Peierls 1988;
  Hall & McKinnon 2005), the Forrest–Tomlin update as the `BasisUpdate`
  policy, transactional updates and a refactorization policy that only
  *asks* (`UpdateOutcome::RefactorNeeded`).
* A reusable pivot sequence `SymbolicLu` with
  `LuFactor::refactorize_same_pattern` (left-looking replay with the
  threshold test; fallback to a fresh factorization), an optional COLAMD
  pre-order of the bump (`Lu::set_pre_order`) and a replay threshold
  (`Lu::set_refactor_pivot_tol`).
* Fill-reducing orderings behind `Ordering`: `Natural`, `Amd` (Amestoy,
  Davis & Duff 1996) and `Colamd` (Davis, Gilbert, Larimore & Ng 2004),
  on one approximate-minimum-degree engine over a quotient graph; the
  elimination-tree postorder (heaviest child last); `Pattern::cholesky_nnz`
  to measure fill.
* Row/column scaling `Scaling` by the policies of `Equilibration`
  (geometric in powers of two, Curtis–Reid, ∞-norm).
* The KKT interface `KktSolver` (analyze/factorize/solve) with the
  regularization policy (static shifts in the equilibrated matrix, dynamic
  pivot bumping, inertia check) and iterative refinement by
  right-preconditioned GMRES as provided methods, with reusable work
  vectors (`RefineWorkspace`); the backend `FaerLdl` over faer 0.24's
  sparse LDLᵀ/LBLᵀ (feature `faer`, default); dense kernels `DenseKernel`
  with `NaiveDense` and `FaerDense`.
* Three runnable examples (`basis_lu`, `ordering`, `kkt_ipm_step`) and the
  crate README.

### Known limitations

See "Known limitations of 0.1.0" in the README: `FaerLdl` is `f64`-only
and pivots only in the unregularized augmented case; the default
regularization was calibrated on Netlib; our AMD is ~1.25× slower than
faer's with the same fill; supernodal bump counts may overcount by value;
the LU's absolute pivot tolerance declares unscaled tiny columns singular;
`solve` allocates once per factor on first use; `Triplets::to_csc` does
not store exact-zero sums.

[0.3.2]: https://github.com/StrategicProjects/xingo-sparse/releases/tag/v0.3.2
[0.3.1]: https://github.com/StrategicProjects/xingo-sparse/releases/tag/v0.3.1
[0.3.0]: https://github.com/StrategicProjects/xingo-sparse/releases/tag/v0.3.0
[0.2.0]: https://github.com/StrategicProjects/xingo-sparse/releases/tag/v0.2.0
[0.1.0]: https://github.com/StrategicProjects/xingo-sparse/releases/tag/v0.1.0
