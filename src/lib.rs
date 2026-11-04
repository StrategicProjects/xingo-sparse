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
//!
//! ## 5.E What the next tasks add
//! * **L1.3** — `Perm`, `Pattern`, `Ordering` with `Natural`, `Amd` and
//!   `Colamd` (our own), the reusable pivot sequence `SymbolicLu` with
//!   `LuFactor::refactorize_same_pattern`, and `LuOptions::pre_order`.
//! * **L1.4** (after P3.9) — `BartelsGolub<T>`, the second
//!   [`BasisUpdate`]; the trait already separates the update policy from
//!   the stored factors ([`Factors`]), so the simplex will not change.
//! * **L1.5** — `KktSolver` with `FaerLdl`, regularization, iterative
//!   refinement, `Equilibration`/`Scaling`, and the dense kernels
//!   (`DenseKernel`, `NaiveDense`, `FaerDense`); `Triplets` and
//!   `BlockPartition` when a consumer needs them.
//! * **L1.6** — the own `SupernodalLdl<T, K>`, after the interior-point
//!   method has measured faer.
//!
//! ## Purpose (the first edition, unchanged)
//! The revised simplex method (Chapter 6) works with the basis matrix `B`
//! only through three operations: factorize `B`, solve with `B` or `Bᵀ`,
//! and replace one column of `B`. This crate provides them
//! (`PLAN.md` §5 items 1–2, tasks P1.1–P1.2):
//!
//! * §5.0 [`error`] — [`SparseError`] and [`LuError`];
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
//!   (Algorithms T, C and S; Chapter 1.2 of the first edition).
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
//!
//! ## References
//! Suhl & Suhl (1990); Forrest & Tomlin (1972); Maros, *Computational
//! Techniques of the Simplex Method* (2003), ch. 5 and 8; Davis, *Direct
//! Methods for Sparse Linear Systems* (2006); Duff, Erisman & Reid (1986);
//! Koberstein (2005), ch. 5; Gilbert & Peierls (1988); Hall & McKinnon
//! (2005); Gustavson (1978); the contract of the crate
//! (`contracts/xingo-sparse`, D-0080).

pub mod basis;
pub mod compressed;
pub mod csc;
pub mod csr;
pub mod error;
pub mod factor;
pub mod index;
pub mod real;
pub mod solve;
mod store;
pub mod update;
pub mod vector;

#[cfg(test)]
mod tests;

pub use basis::{ColumnRef, ColumnSource, Sign};
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
pub use error::{LuError, SparseError};
pub use factor::{Factors, Lu, LuFactor, LuOptions, LuStats};
pub use index::{Col, Row};
pub use real::Real;
pub use solve::{HYPER_HISTORY_WEIGHT, SolveStats};
pub use update::{BasisUpdate, ForrestTomlin, RefactorReason, UpdateOutcome, UpdateStats};
pub use vector::SparseVec;
