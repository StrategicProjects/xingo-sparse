//! # Chapter 5 — `xingo-sparse`: sparse linear algebra for optimization
//!
//! *Second edition (PLAN.md trilha L, task L1.2; D-0080 and its two
//! addenda).* The first edition of this chapter was the crate
//! `xingo-linalg`: the LU factorization of the simplex basis with
//! Forrest–Tomlin updates and hypersparse solves. This edition moves it,
//! together with the compressed containers of `xingo-core` (formerly
//! Chapter 1.2), into a crate that does not depend on the solver and can
//! be published on its own. **The migration has no numerical effect**:
//! instantiated with `f64`, every operation below performs the same
//! floating-point operations in the same order as the first edition, so
//! the solver's results are bit for bit those of before (checked on the
//! MIPLIB anchors and the `nl9` set, D-0080 addendum 2).
//!
//! ## 5.A The layer inversion
//! Before: `xingo-linalg → xingo-core` (the factorization borrowed the
//! containers, the typed indices, the error type and the tolerances of
//! the solver's core). Now: `xingo-core → xingo-sparse`, and
//! `xingo-sparse` depends on nothing in the workspace. Each borrowed item
//! received a home here:
//!
//! | first edition (in `xingo-core`)          | second edition (here)                         |
//! |------------------------------------------|-----------------------------------------------|
//! | `CscMatrix`, `CsrMatrix`, `SparseVec`     | [`CscMatrix<T>`], [`CsrMatrix<T>`], [`SparseVec<T>`] (§5.8), `T = f64` by default |
//! | `Col`, `Row` (Chapter 1.1)                | [`Col`], [`Row`] (§5.7), the same newtypes      |
//! | `CoreError`                               | [`SparseError`] (containers) and [`LuError`] (factorization), §5.0 |
//! | `Tolerances::{markowitz, zero}`           | [`LuOptions::rel_pivot_tol`], [`LuOptions::drop_tol`] |
//! | `Tolerances::{lt, approx_eq, is_zero, …}` | the comparison helpers of §5.6 (crate-private)  |
//!
//! `xingo-core` re-exports the containers at their old paths as the `f64`
//! instances (`xingo_core::CscMatrix = xingo_sparse::CscMatrix<f64>`, and
//! so on), re-exports [`Col`] and [`Row`], and converts [`SparseError`] and
//! [`LuError`] into its `CoreError`; the solver maps its `Tolerances` into
//! [`LuOptions`] when it creates a factor.
//!
//! ## 5.B Scalars and genericity
//! Every container and the factorization are generic in the scalar
//! `T: `[`Real`] (§5.6), with `T = f64` the default type parameter, so
//! `CscMatrix` written without parameters is `CscMatrix<f64>`. [`Real`] is
//! the small trait a threshold-pivoting LU needs — the field operations, a
//! total order on the values that occur (`PartialOrd`, NaN excluded by
//! contract), `abs`, `max`, `is_finite`, conversion from and to `f64` —
//! and is implemented here for `f64` only; `xingo-extprec` (trilha L2)
//! will implement it for double-double and quad-double. Intervals are not
//! `Real` (no total order, no single rounding; D-0080, Q2).
//!
//! *Genericity costs nothing* (D-0080, addendum 2): the code is generic
//! by monomorphization only — no trait objects in the kernels — and every
//! method of [`Real`] for `f64` is an `#[inline]` call of the inherent
//! `f64` method, so `Lu<f64>` compiles to the same machine operations as
//! the first edition's `Factor`. Options and statistics stay in `f64`
//! (they are thresholds and counters, not data); a threshold enters the
//! kernels through [`Real::from_f64`], the identity for `f64`.
//!
//! ## 5.C Comparisons
//! The rule of the solver — "never compare `f64` outside
//! `xingo_core::Tolerances`" — cannot hold in a crate below `xingo-core`.
//! Its equivalent here: **every comparison of floating-point values goes
//! through the helpers of §5.6** (`lt`, `le`, `approx_eq`, `is_zero`,
//! `is_exact_zero`, the same definitions as `Tolerances`, generic in `T`),
//! and every *threshold* of the factorization is a field of [`LuOptions`]
//! — never a bare literal in a kernel. The two constants that remain are
//! policy, documented where they are defined: the clearing fraction of
//! [`SparseVec::clear`] (`0.3`, Chapter 5.8.3) and the moving-average
//! weight [`HYPER_HISTORY_WEIGHT`] (Chapter 5.4).
//!
//! ## 5.D Contract and implementation
//! The approved contract (`contracts/xingo-sparse/src/lib.rs`, L1.1) is the
//! specification of the whole v0.1. This edition realizes the part listed
//! for L1.2; its divergences from the contract are recorded in the cycle's
//! decisions and listed for ratification.
//!
//! | contract                         | here                                                    |
//! |----------------------------------|---------------------------------------------------------|
//! | `Real`                           | [`Real`] (adds `Sum`, needed for bit-identical sums)     |
//! | `CscMatrix<T>`, `CsrMatrix<T>`, `SparseVec<T>` | the same layouts; first-edition methods kept |
//! | `SparseError`, `LuError`         | [`SparseError`], [`LuError`] (variants of the first edition, see §5.0) |
//! | `ColumnRef`, `Sign`, `ColumnSource<T>` | [`ColumnRef`], [`Sign`], [`ColumnSource`] (static dispatch) |
//! | `LuOptions`, `LuStats`, `SolveStats`, `UpdateStats` | [`LuOptions`] (+ `drop_tol`), [`LuStats`], [`SolveStats`], [`UpdateStats`] |
//! | `Lu<T, U = ForrestTomlin<T>>`, `LuFactor<T>`, `Factors<T>` | [`Lu`], [`LuFactor`], [`Factors`] |
//! | `BasisUpdate<T>`, `ForrestTomlin<T>` | [`BasisUpdate`], [`ForrestTomlin`]                  |
//! | `UpdateOutcome`, `RefactorReason` | [`UpdateOutcome`], [`RefactorReason`] (unchanged)      |
//! | `Pattern`, `Perm`, `CscMatrix::{pattern, symmetric_pattern}` | [`Pattern`] (+ `transpose`, `symmetric`, `ata`, `cholesky_nnz`), [`Perm`] (L1.3) |
//! | `Ordering`, `OrderingTarget`, `Natural`, `Amd`, `Colamd` | the same (L1.3; `Amd::default()` has aggressive absorption on) |
//! | `PreOrder`, `LuOptions::pre_order`, `SymbolicLu`, `LuStats::symbolic_reuses` | [`PreOrder`], `Lu::{set_pre_order, pre_order}` (not an `LuOptions` field: see [`Lu::set_pre_order`]), [`SymbolicLu`] (+ `singletons`), `LuStats::{symbolic_reuses, symbolic_fallbacks}` (L1.3) |
//! | `LuFactor::{refactorize_same_pattern, symbolic}` | the same (L1.3)                       |
//! | `Equilibration`, `Scaling<T>` | [`Equilibration`], [`Scaling`] (+ `check_invariants`; §5.13, L1.5) |
//! | `Inertia`, `Regularization<T>`, `Refinement<T>`, `RefineReport<T>`, `KktForm`, `KktSolver<T>`, `KktError` | the same (§5.14, L1.5), plus `Default` for the option structs, `Regularization::{none, relative_to, is_quasi_definite}`, `KktForm::expected_inertia`, `KktError::Structure`, [`EQUILIBRATION_PASSES`]; the augmented inertia reads `{positive: m, negative: n}` |
//! | `FaerLdl`, `FaerSymbolic`, `FaerNumeric<T>` (feature `faer`) | the same (§5.15, L1.5), plus read-only accessors |
//! | `DenseBlock<T>`, `DenseKernel<T>`, `NaiveDense`, `FaerDense`, `DenseError` | the same (§5.16, L1.5); `sytrf` also returns its permutation and `2×2` marks |
//!
//! ## 5.E What the tasks add
//! * **L1.3** (done) — [`Perm`] and [`Pattern`] (§5.9), the orderings
//!   [`Natural`], [`Amd`] and [`Colamd`] behind [`Ordering`] (§5.10, our
//!   own; their engine, Algorithm Q, in §5.11), the reusable pivot
//!   sequence [`SymbolicLu`] with [`LuFactor::refactorize_same_pattern`]
//!   and the COLAMD pre-order of the bump [`Lu::set_pre_order`] (§5.12,
//!   step L5′ of §5.3). With `pre_order = None` (the default) Algorithm L
//!   is unchanged, bit for bit; the simplex uses none of it yet.
//! * **L1.4** (after P3.9) — `BartelsGolub<T>`, the second
//!   [`BasisUpdate`]; the trait already separates the update policy from
//!   the stored factors ([`Factors`]), so the simplex will not change.
//! * **L1.5** (done) — row/column scaling [`Scaling`] by the policies of
//!   [`Equilibration`] (§5.13); the KKT interface [`KktSolver`] with the
//!   regularization policy and the iterative refinement shared by every
//!   backend (§5.14); its v0.1 backend [`FaerLdl`] over faer's sparse
//!   `LDLᵀ`/`LBLᵀ` ordered by our [`Amd`] (§5.15, feature `faer`); and the
//!   dense kernels [`DenseKernel`], [`NaiveDense`] and `FaerDense`
//!   (§5.16). `Triplets` and `BlockPartition` wait for a consumer
//!   ([`CscMatrix::from_triplets`] covers the builder today). The simplex
//!   uses none of it, and faer is not in the solver's build: the workspace
//!   depends on this crate without default features.
//! * **L1.6** — the own `SupernodalLdl<T, K>`, after the interior-point
//!   method has measured faer.
//!
//! ## Purpose (the first edition, unchanged)
//! The revised simplex method (Chapter 6) works with the basis matrix `B`
//! only through three operations: factorize `B`, solve with `B` or `Bᵀ`,
//! and replace one column of `B`. This crate provides them
//! (`PLAN.md` §5 items 1–2, tasks P1.1–P1.2):
//!
//! * §5.0 [`error`] — [`SparseError`], [`LuError`], [`KktError`] and
//!   [`DenseError`];
//! * §5.1 [`basis`] — how the basis is named: [`ColumnRef`] into `[A | ±I]`,
//!   and the general [`ColumnSource`];
//! * §5.2 (private) — dynamic row/column files and count lists;
//! * §5.3 [`factor`] — [`Lu`] and [`LuFactor::factorize`], Algorithm L:
//!   singleton (triangular) phase, then Markowitz pivoting with a relative
//!   threshold on the bump (Suhl & Suhl 1990);
//! * §5.4 [`solve`] — [`LuFactor::ftran`] and [`LuFactor::btran`]: the
//!   dense Algorithms F and B and the hypersparse Algorithms HF and HB
//!   (Gilbert–Peierls reach), chosen per call by the density policy of
//!   Hall & McKinnon (task P1.4);
//! * §5.5 [`update`] — [`LuFactor::update`] through the [`BasisUpdate`]
//!   policy, the Forrest–Tomlin update [`ForrestTomlin`] (Algorithm U), and
//!   the refactorization policy ([`UpdateOutcome`]);
//! * §5.6 [`real`] — the scalar trait [`Real`] and the comparisons;
//! * §5.7 [`index`] — the typed indices [`Col`] and [`Row`];
//! * §5.8 [`compressed`], [`csc`], [`csr`], [`vector`] — the containers
//!   (Algorithms T, C and S; Chapter 1.2 of the first edition);
//! * §5.9 [`perm`] — [`Perm`] and [`Pattern`], and the Cholesky count of an
//!   ordering (Algorithm E);
//! * §5.10 [`ordering`] — [`Ordering`], [`Natural`], [`Amd`] (Algorithm A),
//!   [`Colamd`] (Algorithm C);
//! * §5.11 (private) — approximate minimum degree on a quotient graph
//!   (Algorithm Q), the engine of both;
//! * §5.12 [`symbolic`] — [`SymbolicLu`] and Algorithm R
//!   ([`LuFactor::refactorize_same_pattern`]);
//! * §5.13 [`scaling`] — [`Equilibration`] and [`Scaling`] (Algorithms G,
//!   R and I);
//! * §5.14 [`kkt`] — [`KktSolver`] and its shared policy: regularization
//!   (Algorithm K) and iterative refinement (Algorithm F);
//! * §5.15 `faer_ldl` (feature `faer`) — [`FaerLdl`] (Algorithms A, N, S);
//! * §5.16 [`dense`] — [`DenseKernel`], [`NaiveDense`], `FaerDense`
//!   (Algorithms P, B, T, G, S).
//!
//! ## Notation
//! `B ∈ R^{m×m}`; *rows* `i` and basis *positions* `k` both range over
//! `0..m` but are distinct index spaces. `P B Q = L U` with `P`, `Q`
//! permutations, `L` unit lower and `U` upper triangular; after `t`
//! updates `B = Pᵀ L R₁⁻¹ ⋯ R_t⁻¹ U Qᵀ` with row etas `R_e` and updated
//! `P`, `Q`, `U`. Identifiers follow the chapters: `l_ic`, `u_pj`, `mu`
//! (`μ`), `alpha_r` (`α_r`), `tau` (`τ`), `sigma` (`σ`). Indices are `u32`
//! in the stored arrays, pointers `usize` (PLAN.md §3.2; D-0080, Q3).
//!
//! ## Contract with the simplex method
//! 1. `lu.factorize(&(&heading[..], &a))` (a [`ColumnSource`]); on
//!    [`LuError::Singular`] replace each `cols[k]` by
//!    `ColumnRef::Logical(rows[k], _)` and factorize again.
//! 2. `ftran(&mut v)` maps a row-indexed `v` to the position-indexed
//!    `B⁻¹ v`; `btran(&mut v)` maps a position-indexed `v` to the
//!    row-indexed `B⁻ᵀ v`. Both take `&mut self`: the factor owns the work
//!    arrays of the solves and the density statistics that choose between
//!    the dense and the hypersparse algorithms. Give the right-hand side in
//!    sparse mode with a correct pattern to allow the hypersparse
//!    algorithms; the result comes back in sparse mode, its pattern a
//!    superset of its support (Chapter 5.4, invariant (V1)).
//! 3. To pivot position `r` out for column `a_q`:
//!    `ftran_for_update(&mut a_q)` (giving `α_q`), then
//!    `update(r, α_q[r])`. On [`UpdateOutcome::RefactorNeeded`] (or any
//!    error) refactorize with the new heading; the factor never does so on
//!    its own (D-0016).
//!
//! ## Safety
//! Per D-0006 this is the one crate where `unsafe` is permitted (audited,
//! `// SAFETY:` comments, Miri where executable). This edition, like the
//! first, uses none; optimized kernels wait for measurements (task P1.10).
//! The `faer` feature (L1.5) brings faer's own `unsafe` kernels in as a
//! dependency; the code of this crate stays free of `unsafe`.
//!
//! ## References
//! Suhl & Suhl (1990); Forrest & Tomlin (1972); Maros, *Computational
//! Techniques of the Simplex Method* (2003), ch. 5 and 8; Davis, *Direct
//! Methods for Sparse Linear Systems* (2006); Duff, Erisman & Reid (1986);
//! Koberstein (2005), ch. 5; Gilbert & Peierls (1988); Hall & McKinnon
//! (2005); Gustavson (1978); Amestoy, Davis & Duff (1996); Davis, Gilbert,
//! Larimore & Ng (2004); the contract of the crate
//! (`contracts/xingo-sparse`, D-0080).

// Without `faer`, the chapter text still names FaerLdl/FaerDense.
#![cfg_attr(not(feature = "faer"), allow(rustdoc::broken_intra_doc_links))]

pub mod basis;
pub mod compressed;
pub mod csc;
pub mod csr;
pub mod dense;
pub mod error;
pub mod factor;
#[cfg(feature = "faer")]
pub mod faer_ldl;
pub mod index;
pub mod kkt;
pub mod ordering;
pub mod perm;
mod qgraph;
pub mod real;
pub mod scaling;
pub mod solve;
mod store;
pub mod symbolic;
pub mod update;
pub mod vector;

#[cfg(test)]
mod tests;

pub use basis::{ColumnRef, ColumnSource, Sign};
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
#[cfg(feature = "faer")]
pub use dense::FaerDense;
pub use dense::{DenseBlock, DenseKernel, NaiveDense};
pub use error::{DenseError, KktError, LuError, SparseError};
pub use factor::{Factors, Lu, LuFactor, LuOptions, LuStats, PreOrder};
#[cfg(feature = "faer")]
pub use faer_ldl::{FaerLdl, FaerNumeric, FaerSymbolic};
pub use index::{Col, Row};
pub use kkt::{
    EQUILIBRATION_PASSES, Inertia, KktForm, KktSolver, RefineReport, Refinement, Regularization,
};
pub use ordering::{Amd, Colamd, Natural, Ordering, OrderingTarget};
pub use perm::{Pattern, Perm};
pub use real::Real;
pub use scaling::{Equilibration, Scaling};
pub use solve::{HYPER_HISTORY_WEIGHT, SolveStats};
pub use symbolic::SymbolicLu;
pub use update::{BasisUpdate, ForrestTomlin, RefactorReason, UpdateOutcome, UpdateStats};
pub use vector::SparseVec;
