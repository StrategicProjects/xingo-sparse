# Changelog

All notable changes to `xingo-sparse` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate
follows [Semantic Versioning](https://semver.org/); before 1.0, a minor
version may change the public API.

## [Unreleased]

### Added

* `CITATION.cff` and the Zenodo DOI (concept 10.5281/zenodo.24095312;
  0.1.0 is 10.5281/zenodo.24095313) in the README.

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

[0.1.0]: https://github.com/StrategicProjects/xingo/tree/main/crates/sparse
