//! # Chapter 5.6 — Scalars and comparisons
//!
//! ## Purpose
//! The containers and the factorization are written once, for any scalar
//! type `T` that behaves like a real number with rounding: `f64` today,
//! double-double and quad-double from `xingo-extprec` (trilha L2). This
//! chapter fixes that interface, [`Real`], and the *only* functions of the
//! crate that compare scalar values (Chapter 5, §5.C).
//!
//! ## Notation
//! `T` is the scalar type, `ε_T` its machine epsilon ([`Real::EPSILON`]).
//! For finite `a`, `b` and a tolerance `ε ≥ 0` let `s(a, b) = max(1, |a|,
//! |b|)` (the scale of `xingo_core::Tolerances`, Chapter 1.3). Then
//!
//! * `lt(a, b)` — `a < b` exactly, written `b − a > 0·s(a, b)` for finite
//!   values and `a < b` otherwise: the definition of
//!   `Tolerances::lt(a, b, 0.0)`, kept *as an expression*, so that the
//!   `f64` instance performs the very operations of the first edition;
//! * `le(a, b, ε)` — `¬ lt_ε(b, a)`; `approx_eq(a, b, ε)` — `|a − b| ≤
//!   ε·s(a, b)`, exact when a value is infinite;
//! * `is_zero(x, ε)` — `|x| ≤ ε` (a drop test); `is_exact_zero(x)` —
//!   `x = 0` (a structural test: is the entry stored?).
//!
//! ## Requirements on `T`
//! (R1) The field operations are the correctly rounded ones of `T` (no
//!      fused or reassociated arithmetic is assumed anywhere).
//! (R2) `PartialOrd` is a total order on the values that occur: NaN never
//!      enters a factorization (debug builds assert it in the comparisons).
//! (R3) `Sum` adds in iteration order starting from the type's additive
//!      identity for sums — for `f64`, the standard library's
//!      `Iterator::sum`, which the first edition used; requiring it keeps
//!      the order of every dot product of the solves unchanged.
//! (R4) `from_f64(to_f64(x))` is `x` for `f64`; for the extended types
//!      `to_f64` is the nearest `f64` and `from_f64` exact.
//!
//! ## Why a trait of this size
//! A threshold LU needs the field operations, `abs`, `max` and a total
//! order (pivot search and tests), `is_finite` (input validation), and the
//! conversion of `f64` thresholds into `T`. `sqrt`, `recip` and `mul_add`
//! are in the contract for the KKT solver (L1.5) and are unused by the LU;
//! they are part of the trait so that implementors see the whole
//! interface once. faer's `RealField` is the model, without its SIMD
//! surface (D-0080, Q1).
//!
//! ## References
//! Knuth, TAOCP vol. 2, §4.2.2; Higham, *Accuracy and Stability of
//! Numerical Algorithms* (2002), ch. 1–2; faer's `RealField` (read, not
//! copied); Chapter 1.3 of the solver (`Tolerances`).

use core::fmt::Debug;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// The scalar type of every container and factorization of the crate.
///
/// Implemented for `f64`; `xingo-extprec` implements it for its extended
/// types (trilha L2). A total order is required (pivot search and pivot
/// tests), so intervals are not `Real` (D-0080, Q2). See the module
/// documentation for the requirements (R1)–(R4).
pub trait Real:
    Copy
    + Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Sum
    + 'static
{
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;
    /// Machine epsilon of the type (`2⁻⁵²` for `f64`).
    const EPSILON: Self;
    /// The value of an `f64` (exact for `f64` and the extended types).
    fn from_f64(x: f64) -> Self;
    /// The nearest `f64` (lossy for the extended types).
    fn to_f64(self) -> f64;
    /// `|x|`.
    #[must_use]
    fn abs(self) -> Self;
    /// `√x`.
    #[must_use]
    fn sqrt(self) -> Self;
    /// `1 / x`.
    #[must_use]
    fn recip(self) -> Self;
    /// `self · a + b` with a single rounding where the type allows.
    #[must_use]
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// `true` iff the value is neither infinite nor NaN.
    fn is_finite(self) -> bool;
    /// The larger of two values, with the NaN rule of `f64::max`.
    #[must_use]
    fn max(self, other: Self) -> Self;
    /// The smaller of two values, with the NaN rule of `f64::min`.
    #[must_use]
    fn min(self, other: Self) -> Self;
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f64::EPSILON;
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn recip(self) -> Self {
        f64::recip(self)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
}

/// A second scalar for the tests of the generic code (Chapter 5.17 among
/// others): `f32`, with the inherent methods, as `f64` above. Not part of
/// the public API — the extended types of `xingo-extprec` are the intended
/// second instance.
#[cfg(test)]
impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f32::EPSILON;
    #[inline]
    #[allow(clippy::cast_possible_truncation)] // the nearest f32 is the contract (R4)
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline]
    fn recip(self) -> Self {
        f32::recip(self)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
}

// The comparisons of the crate. This is the one place where scalar values
// meet `<`, `>` or `==` (Chapter 5, §5.C); the definitions are those of
// `xingo_core::Tolerances` (Chapter 1.3), generic in `T`.

/// The scale `s(a, b) = max(1, |a|, |b|)`.
#[inline]
fn scale<T: Real>(a: T, b: T) -> T {
    T::ONE.max(a.abs()).max(b.abs())
}

/// `a < b` exactly (`Tolerances::lt(a, b, 0.0)`).
#[inline]
pub(crate) fn lt<T: Real>(a: T, b: T) -> bool {
    lt_eps(a, b, T::ZERO)
}

/// Strict tolerant order `a <_ε b`: `b − a > ε·s(a, b)`; exact when a
/// value is infinite.
#[inline]
pub(crate) fn lt_eps<T: Real>(a: T, b: T, eps: T) -> bool {
    debug_assert!(!is_nan(a) && !is_nan(b), "lt called with NaN");
    if a.is_finite() && b.is_finite() {
        b - a > eps * scale(a, b)
    } else {
        a < b
    }
}

/// Tolerant order `a ≤_ε b`, defined as `¬(b <_ε a)`.
#[inline]
#[cfg_attr(not(test), allow(dead_code))] // the tests' oracles use it
pub(crate) fn le<T: Real>(a: T, b: T, eps: T) -> bool {
    !lt_eps(b, a, eps)
}

/// `a ≤ b` exactly, without the scale (for ratios in `[0, 1]` such as
/// densities, where the two definitions coincide).
#[inline]
pub(crate) fn le_exact<T: Real>(a: T, b: T) -> bool {
    debug_assert!(!is_nan(a) && !is_nan(b), "le_exact called with NaN");
    a <= b
}

/// `|a − b| ≤ ε·s(a, b)`; exact when a value is infinite.
#[inline]
pub(crate) fn approx_eq<T: Real>(a: T, b: T, eps: T) -> bool {
    debug_assert!(!is_nan(a) && !is_nan(b), "approx_eq called with NaN");
    if a.is_finite() && b.is_finite() {
        (a - b).abs() <= eps * scale(a, b)
    } else {
        a == b
    }
}

/// `|x| ≤ ε`: `x` is negligible for the drop tolerance `ε`.
#[inline]
pub(crate) fn is_zero<T: Real>(x: T, eps: T) -> bool {
    debug_assert!(!is_nan(x), "is_zero called with NaN");
    x.abs() <= eps
}

/// `x = 0` exactly (or `−0`): a structural test.
#[inline]
pub(crate) fn is_exact_zero<T: Real>(x: T) -> bool {
    x == T::ZERO
}

/// `x` is NaN (it is not equal to itself).
#[inline]
#[allow(clippy::eq_op)] // the NaN test is x ≠ x
pub(crate) fn is_nan<T: Real>(x: T) -> bool {
    x != x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparisons_match_their_definitions() {
        assert!(lt(1.0, 2.0) && !lt(2.0, 2.0) && !lt(3.0, 2.0));
        assert!(lt(f64::NEG_INFINITY, -1e308) && !lt(f64::INFINITY, f64::INFINITY));
        assert!(le(2.0, 2.0, 0.0) && le(2.0 + 1e-13, 2.0, 1e-12));
        assert!(approx_eq(1e6, 1e6 + 1e-7, 1e-12));
        assert!(approx_eq(f64::INFINITY, f64::INFINITY, 0.0));
        assert!(!approx_eq(1e308, f64::INFINITY, 1e-9));
        assert!(is_zero(1e-13, 1e-12) && !is_zero(1e-11, 1e-12));
        assert!(is_exact_zero(-0.0) && !is_exact_zero(f64::MIN_POSITIVE));
        assert!(is_nan(f64::NAN) && !is_nan(1.0));
    }

    #[test]
    fn f64_is_real() {
        fn roundtrip<T: Real>(x: f64) -> f64 {
            T::from_f64(x).to_f64()
        }
        assert!(approx_eq(roundtrip::<f64>(0.1), 0.1, 0.0));
        assert!(approx_eq(Real::abs(-2.0_f64), 2.0, 0.0));
        assert!(approx_eq(Real::mul_add(2.0_f64, 3.0, 1.0), 7.0, 0.0));
        assert!(approx_eq(Real::sqrt(4.0_f64), 2.0, 0.0));
        assert!(approx_eq(Real::recip(4.0_f64), 0.25, 0.0));
        assert!(approx_eq(Real::min(4.0_f64, 1.0), 1.0, 0.0));
        assert!(approx_eq(<f64 as Real>::EPSILON, f64::EPSILON, 0.0));
        let s: f64 = [1.0, 2.0, 3.0].into_iter().sum();
        assert!(approx_eq(s, 6.0, 0.0));
    }
}
