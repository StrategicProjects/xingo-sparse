# Changelog

All notable changes to `xingo-sparse` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate
follows [Semantic Versioning](https://semver.org/); before 1.0, a minor
version may change the public API.

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

[0.3.0]: https://github.com/StrategicProjects/xingo-sparse/releases/tag/v0.3.0
[0.2.0]: https://github.com/StrategicProjects/xingo-sparse/releases/tag/v0.2.0
[0.1.0]: https://github.com/StrategicProjects/xingo-sparse/releases/tag/v0.1.0
