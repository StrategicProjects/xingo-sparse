# xingo-sparse

[![crates.io](https://img.shields.io/crates/v/xingo-sparse.svg)](https://crates.io/crates/xingo-sparse)
[![docs.rs](https://docs.rs/xingo-sparse/badge.svg)](https://docs.rs/xingo-sparse)
![MSRV 1.90](https://img.shields.io/badge/MSRV-1.90-blue.svg)
![license: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24095312.svg)](https://doi.org/10.5281/zenodo.24095312)

**Website:** <https://strategicprojects.github.io/xingo-sparse/> (overview, measured results, API reference).

Sparse linear algebra **for optimization solvers**, in pure Rust: the
factorizations a simplex method and an interior-point method need, over
compressed sparse containers generic in the scalar type. It is the linear
algebra layer of [Xingó](https://github.com/StrategicProjects/xingo), a
linear and mixed-integer programming solver, and depends on nothing in it:
the solver depends on this crate, never the other way round.

It is **not** a general sparse or dense linear algebra library. There is no
sparse matrix product, no QR or SVD, no eigenvalues, no iterative solvers,
no complex scalars (see "Scope" below); [faer](https://crates.io/crates/faer),
[nalgebra](https://crates.io/crates/nalgebra) and
[ndarray](https://crates.io/crates/ndarray) cover that ground. What is here
is what those do not offer and an optimization solver cannot do without.

## Three entry points

**1. The basis of a simplex method** — a dynamic LU with threshold Markowitz
pivoting (Suhl & Suhl 1990), **Forrest–Tomlin updates** when one column of
the basis changes, and **hypersparse** FTRAN/BTRAN (Gilbert–Peierls reach,
Hall–McKinnon density policy). The basis is never formed: it is a list of
references into `[A | ±I]` plus the constraint matrix `A`.

```rust
use xingo_sparse::{Col, ColumnRef, Lu, LuFactor, Row, Sign, SparseVec, Triplets, UpdateOutcome};

// A 2 × 3 constraint matrix by coordinates, then its CSC form.
let mut t = Triplets::new(2, 3);
for (i, j, v) in [(0, 0, 2.0), (0, 2, 1.0), (1, 1, 1.0), (1, 2, 3.0)] {
    t.push(i, j, v);
}
let a = t.to_csc()?;

// A basis: column a_0 at position 0, the slack of row 1 at position 1.
let mut heading = vec![ColumnRef::Structural(Col(0)), ColumnRef::Logical(Row(1), Sign::Plus)];
let mut lu = Lu::<f64>::default(); // Forrest–Tomlin is the default update policy
lu.factorize(&(heading.as_slice(), &a))?;

// FTRAN: x = B⁻¹ b (row-indexed in, position-indexed out).
let mut x = SparseVec::from_dense(&[4.0, 5.0]);
lu.ftran(&mut x);
assert_eq!(x.values(), &[2.0, 5.0]);

// Exchange: a_2 enters at position 1. `ftran_for_update` records the spike.
let mut alpha = SparseVec::new(2);
ColumnRef::Structural(Col(2)).for_each_entry(&a, |i, v| alpha.set(i, v));
lu.ftran_for_update(&mut alpha);
assert_eq!(lu.update(1, alpha.get(1))?, UpdateOutcome::Updated);
heading[1] = ColumnRef::Structural(Col(2));

// The factor never refactorizes on its own; it asks (`RefactorNeeded`) and
// the caller, who owns the columns, calls `factorize` again.
# Ok::<(), Box<dyn std::error::Error>>(())
```

**2. Fill-reducing orderings** — our own `Amd` (Amestoy, Davis & Duff 1996)
and `Colamd` (Davis, Gilbert, Larimore & Ng 2004) behind the `Ordering`
trait, on a value-free `Pattern`; `Pattern::cholesky_nnz` counts the fill an
ordering yields. Fill within 1.05× of SuiteSparse's on Netlib and
Harwell–Boeing (331 comparisons, worst 1.029×).

```rust
use xingo_sparse::{Amd, Natural, Ordering, OrderingTarget, Triplets};

// The arrowhead: a dense first row and column plus the diagonal.
let n = 20;
let mut t = Triplets::new(n, n);
for i in 0..n as u32 {
    t.push(i, i, 4.0);
    if i > 0 {
        t.push(0, i, -1.0);
        t.push(i, 0, -1.0);
    }
}
let s = t.to_csc()?.pattern();
let band = s.cholesky_nnz(&Natural.order(&s, OrderingTarget::Symmetric)?)?;
let amd = s.cholesky_nnz(&Amd::default().order(&s, OrderingTarget::Symmetric)?)?;
assert_eq!(band, n * (n + 1) / 2); // pivoting on the hub first fills everything
assert_eq!(amd, 2 * n - 1);        // minimum degree leaves it for last
# Ok::<(), Box<dyn std::error::Error>>(())
```

**3. The KKT systems of an interior-point method** — the `KktSolver` trait
in the CHOLMOD shape (analyze once, factorize many, solve), with the
**regularization** policy (static shifts `ρ, δ` in the equilibrated matrix
plus dynamic pivot bumping, Altman & Gondzio 1999) and **iterative
refinement** (right-preconditioned GMRES, best iterate kept) in the trait's
provided methods, so every backend shares them. The **recommended
backend is our own `SupernodalLdl`** — a supernodal LDLᵀ (elimination
tree, relaxed supernodes, left-looking blocks on dense kernels, exact bump
counts), generic in the scalar and with no dependency; `FaerLdl`, faer's
sparse LDLᵀ/LBLᵀ (feature `faer`), remains as an alternative and as the
oracle of the test suite. Both are ordered by our AMD.

```rust
use xingo_sparse::{Amd, BlockPartition, KktForm, KktSolver, Refinement, Regularization, SupernodalLdl, Triplets};

// K = [−D Aᵀ; A δI] for A = [1 1 0; 0 1 1], D = diag(1, 2, 3): lower triangle.
let (n, m) = (3, 2);
let part = BlockPartition::from_sizes(&[n, m], &[n, m])?; // primal block, then dual
let dual = part.row_block(1).start as u32;
let mut t = Triplets::new(n + m, n + m);
for (j, d) in [1.0, 2.0, 3.0].into_iter().enumerate() {
    t.push(j as u32, j as u32, -d);
}
for (i, j) in [(0, 0), (0, 1), (1, 1), (1, 2)] {
    t.push(dual + i, j, 1.0);
}
let k = t.to_csc()?;
let form = KktForm::Augmented { n_primal: n, n_dual: m };

let solver = SupernodalLdl::<f64>::default(); // the recommended backend; `FaerLdl` is the alternative
let sym = solver.analyze(&k.pattern(), form, &Amd::default())?; // once per pattern
let (num, inertia) = solver.factorize(&sym, &k, Regularization::default())?; // every iteration
assert_eq!(inertia, form.expected_inertia(n + m)); // m positive, n negative pivots
let b = [1.0, 1.0, 1.0, -1.0, -1.0];
let mut x = [0.0; 5];
// Refine against the unregularized K: the shifts' bias is removed.
let report = solver.solve_refined(&num, &k, &b, &mut x, Refinement::default());
assert!(report.residual <= 1e-12);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The three programs in [`examples/`](examples/) run these end to end
(`cargo run -p xingo-sparse --example basis_lu`, `ordering`,
`kkt_ipm_step`), each ending in a check of its result.

## Features

* `faer` (default) — brings in [faer](https://crates.io/crates/faer) 0.24
  (pure Rust, MIT; no `rayon`: the factorizations run sequentially and the
  results do not depend on threads) for the KKT backend `FaerLdl` and the
  dense kernel `FaerDense`.
* Without it (`default-features = false`) the crate has no dependency at
  all: the containers, the LU with its updates and solves, the orderings,
  the scaling, the `KktSolver` trait with its regularization and
  refinement, the own backend `SupernodalLdl` on the reference dense
  kernel `NaiveDense` — a complete KKT solver — remain. The Xingó solver
  builds it this way: the simplex side never touches faer.

## Scope

In: CSC/CSR containers and a sparse work vector, generic in `T: Real`
(`f64` today; double-double and quad-double are planned in a sibling
crate); the coordinate builder `Triplets` and `BlockPartition`; row/column
scaling (geometric in powers of two, Curtis–Reid, ∞-norm); the dynamic LU
with a reusable pivot sequence (`SymbolicLu`) for refactorizations of the
same pattern; Forrest–Tomlin updates; AMD and COLAMD; the regularized,
refined KKT solver with the own supernodal LDLᵀ and the faer backend.

Out, by decision: a complete dense library; sparse matrix products, QR,
SVD, eigenvalues; iterative solvers and preconditioners beyond the
refinement loop; complex scalars; METIS or any C binding (nested dissection
was measured against METIS as an external tool and not adopted: our AMD's
fill is 0.97× METIS's on average over 40 KKT systems); GPU.
Bartels–Golub updates are specified in the crate's contract and scheduled
next.

## Known limitations

Measured, not guessed (the decision log of the solver records each):

1. `SupernodalLdl` factorizes the two *regularized* systems with `1×1`
   pivots (they are quasi-definite and need none) and an *unregularized*
   augmented system (`Regularization::none()`) with the `LBLᵀ` of
   Chapter 5.17, Algorithm M: Bunch–Kaufman `1×1`/`2×2` pivots inside each
   supernode and *delayed pivots* to the parent supernode when a column has
   no acceptable pivot (MA57's scheme). Its zero test (`zero_pivot_tol`) and
   growth threshold (`pivot_threshold`, `u = 10⁻⁸`) were calibrated on the
   Netlib LPs: of the 279 unregularized augmented systems, 49 are
   nonsingular (κ ≤ 10¹² by scipy) and all 49 factorize with the expected
   inertia and a residual ≤ 10⁻¹⁰; the others are singular to working
   precision (raw Netlib has dependent and empty rows) and get an explicit
   `ZeroPivot`/`WrongInertia` or a small residual, never a silent infinity.
   The delays pile up where the system is singular anyway (the end of an
   interior-point method without regularization: up to 15 352 delayed
   pivots in one system); the regularized path, which a running method
   should use there, is untouched bit for bit. `FaerLdl`'s intranodal
   Bunch–Kaufman has no delays and fails on 246 of the 279 systems.
2. The default regularization (`ρ = 10⁻⁶`, `δ = 10⁻⁸` in the equilibrated
   matrix) was calibrated on the 93 Netlib LPs at three stages of an
   interior-point method (558 systems, plus 3 × 558 with other random
   diagonals): residual against the unregularized `K` at most `10⁻¹⁰` with
   the default refinement in all but one case (that one at `1.6·10⁻¹⁰`). A
   running interior-point method will review `ρ, δ`.
3. Our `Amd` takes about 1.15× the time of faer's AMD (geometric mean on the
   ten largest Netlib systems), with the same fill. Against METIS's nested
   dissection (`ndmetis` 5.1.0, external tool) its fill is 0.97× on
   average over 40 KKT systems (ten largest Netlib, five Kennington, five
   MIPLIB 2017 root LPs, normal and augmented), from 0.43× to
   1.63× (pds-20's normal equations); no nested dissection is planned
   unless an interior-point method meets more cases like that one.
8. `SupernodalLdl<f64, NaiveDense>` (no dependency) factorizes the ten
   largest Netlib systems in 1.37× the time of `FaerLdl` (geometric
   mean; worst 3.3×): the dense kernels are plain blocked loops
   with no SIMD. With `FaerDense` the ratio is 1.00×.
4. `FaerLdl`'s count of bumped pivots is exact for the simplicial
   structure; for the supernodal one, negative-sign bumps are counted by
   value and may be overcounted (faer counts only the positive ones;
   reported upstream). `SupernodalLdl` counts both signs exactly.
5. The LU declares singular a column whose entries are all at or below the
   absolute pivot tolerance (`10⁻¹⁰`), without scaling; the Harwell–Boeing
   matrix sherman3 needs `Scaling` first (or a lower `abs_pivot_tol`).
   Refactorization with a reused pivot sequence speeds up dense factors
   less than 1.5× (it saves the analysis, not the arithmetic). The replay
   threshold defaults to the Markowitz threshold `u`; `u/10` is available.
6. A `KktSolver::solve` allocates once per factor, at its first call (the
   lazy initialization of a mutex on macOS); `solve_refined_with` with a
   reused `RefineWorkspace` allocates nothing afterwards.
7. `Triplets::to_csc` does not store an entry whose sum is exactly zero
   (the containers forbid explicit zeros); zeros are accepted as input.

## Design notes

* **Literate.** Every module is a chapter: notation, the algorithm in
  numbered steps, invariants (each a `debug_assert!` or a
  `check_invariants()`), references. The crate documentation *is* the
  text; start at the crate root on docs.rs, or read Chapter 5 of the
  solver's book, which includes these sources verbatim.
* **Read, not copied.** HiGHS (`HFactor`, `HVector`), SuiteSparse (AMD,
  COLAMD, CHOLMOD) and faer were studied as references and oracles; the
  code here is written from the papers. No GPL or LGPL code is used or
  linked; faer (MIT) is the only dependency, and only with the `faer`
  feature.
* **Deterministic.** Same input, same options ⇒ same result, bit for bit:
  no hashing, no threads, no allocation-order dependence anywhere in the
  numerical paths.
* **Errors, not panics**, on data: malformed containers, singular bases,
  wrong inertia and non-finite factors are reported with what the caller
  needs to recover. Panics are reserved for contract violations (a vector
  of the wrong length).
* **No `unsafe`** in this crate (faer has its own, audited, kernels).
* **Two KKT backends, one policy.** `SupernodalLdl` (ours, generic, no
  dependency; recommended) and `FaerLdl` (faer; alternative and oracle)
  share the regularization and the refinement of the trait; on the 558
  Netlib systems both give the expected inertia and a residual against the
  unregularized `K` at most `10⁻¹⁰`. Under the same ordering,
  `SupernodalLdl` stores exactly faer's factor on the 84 supernodal
  systems and a smaller one on the 474 simplicial ones, and factorizes in
  1.02× of faer's time (geometric mean, ten largest Netlib) with faer's
  dense kernels.

## Repository, issues and citation

The crate is developed inside the Xingó monorepo
([StrategicProjects/xingo](https://github.com/StrategicProjects/xingo),
directory `crates/sparse`), where every change is validated together with
the solver that uses it. This stand-alone repository,
[StrategicProjects/xingo-sparse](https://github.com/StrategicProjects/xingo-sparse),
is a read-only mirror of that directory (`git subtree split`, full history),
updated at every release; it exists so that the library can be archived and
cited on its own. Issues and pull requests about the crate are welcome in
either repository. Each release of the mirror is archived on Zenodo: the
concept DOI [10.5281/zenodo.24095312](https://doi.org/10.5281/zenodo.24095312)
always resolves to the latest version, each release has its own DOI on its
release page, and `CITATION.cff` has the citation metadata.

## Authorship, credits and disclaimer

`xingo-sparse` is written and maintained by **André Leite** as the linear
algebra layer of the [Xingó](https://github.com/StrategicProjects/xingo)
solver, developed at [StrategicProjects](https://github.com/StrategicProjects).

The algorithms are those of their authors, implemented here from the
published papers: Suhl & Suhl (LU for simplex bases), Forrest & Tomlin
(basis updates), Gilbert & Peierls and Hall & McKinnon (hypersparse solves),
Amestoy, Davis & Duff (AMD), Davis, Gilbert, Larimore & Ng (COLAMD), Liu
(elimination trees), Bunch & Kaufman and Duff & Reid (symmetric indefinite
pivoting and delayed pivots; MA57), Altman & Gondzio and Arioli, Duff,
Gratton & Pralet (regularization and refinement). [HiGHS](https://highs.dev),
[SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) (AMD,
COLAMD, CHOLMOD) and [faer](https://crates.io/crates/faer) were studied as
references and used as oracles in the tests; faer (MIT) is also the optional
dense-kernel dependency. Each chapter of the sources lists its references.

This is an independent project: it is not affiliated with, maintained by or
endorsed by the HiGHS, SuiteSparse, HSL or faer authors, and it includes no
code from them (faer is linked only through the optional `faer` feature).
All names and trademarks belong to their owners. The software is provided
"as is", without warranty of any kind, under the licenses below.

## Minimum supported Rust version

1.90 (edition 2024), tested in CI together with stable.

## License

Licensed under either of the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
or the [MIT license](https://opensource.org/licenses/MIT), at your option.
