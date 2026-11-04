//! # Chapter 5.7 — Typed indices
//!
//! ## Purpose
//! Columns and rows of a matrix are both numbered from zero, and passing a
//! row number where a column number was expected is the classic silent
//! bug of sparse codes. [`Col`] and [`Row`] are distinct newtypes, so the
//! compiler rejects that confusion at no runtime cost. They lived in
//! `xingo-core` (Chapter 1.1) until task L1.2; the containers (Chapter 5.8)
//! and the basis references (Chapter 5.1) speak them, so they moved down
//! with the containers and `xingo-core` re-exports them unchanged.
//!
//! ## Notation
//! A matrix has `n` columns, indexed by `j ∈ {0, …, n−1}`, and `m` rows,
//! indexed by `i ∈ {0, …, m−1}`. `Col(j)` names column `j` and `Row(i)`
//! names row `i`. In the solver a column is a variable and a row a
//! constraint.
//!
//! ## Representation
//! Both are `u32` wrappers (`PLAN.md` §3.2; D-0080, Q3): 2³² − 1 entities
//! is ample, and halving the width of `row_idx` compared with `usize`
//! saves memory bandwidth in every sparse kernel. Offsets into nonzero
//! arrays (`col_ptr`, `row_ptr`) are `usize`, because `nnz` may exceed
//! 2³².
//!
//! ## Invariants
//! (I1) Conversion from `usize` never truncates: [`Col::from_usize`] and
//!      [`Row::from_usize`] panic beyond `u32::MAX`.
//! (I2) Range validity (`j < n`, `i < m`) is a property of the structure
//!      that stores the index and is checked by its `check_invariants()`.

use std::fmt;

use crate::SparseError;

macro_rules! typed_index {
    ($(#[$doc:meta])* $name:ident, $what:literal, $prefix:literal) => {
        $(#[$doc])*
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
        pub struct $name(pub u32);

        impl $name {
            #[doc = concat!("Builds a ", $what, " index from a `usize`.")]
            ///
            /// # Panics
            /// If `i > u32::MAX` (invariant I1).
            #[inline]
            pub fn from_usize(i: usize) -> Self {
                Self(u32::try_from(i).expect(concat!($what, " index exceeds u32::MAX")))
            }

            /// Returns the index as a `usize`, ready for slice indexing.
            #[inline]
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl From<u32> for $name {
            #[inline]
            fn from(i: u32) -> Self {
                Self(i)
            }
        }

        impl From<$name> for usize {
            #[inline]
            fn from(i: $name) -> usize {
                i.index()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "{}"), self.0)
            }
        }
    };
}

typed_index!(
    /// Index of a column (in the solver: a variable), `0 ≤ j < n`.
    Col,
    "column",
    "c"
);

typed_index!(
    /// Index of a row (in the solver: a constraint), `0 ≤ i < m`.
    Row,
    "row",
    "r"
);

/// Converts a dimension to `u32`, reporting [`SparseError::TooLarge`] on
/// overflow.
pub(crate) fn dim_to_u32(what: &'static str, value: usize) -> Result<u32, SparseError> {
    u32::try_from(value).map_err(|_| SparseError::TooLarge { what, value })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_display() {
        let j = Col::from_usize(7);
        assert_eq!(j.index(), 7);
        assert_eq!(usize::from(j), 7);
        assert_eq!(j.to_string(), "c7");
        assert_eq!(Row::from(3).to_string(), "r3");
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "exceeds u32::MAX")]
    fn from_usize_rejects_overflow() {
        let _ = Row::from_usize(1 << 33);
    }

    #[test]
    fn dim_to_u32_reports_overflow() {
        assert_eq!(dim_to_u32("rows", 5), Ok(5));
        #[cfg(target_pointer_width = "64")]
        assert!(matches!(
            dim_to_u32("rows", 1 << 33),
            Err(SparseError::TooLarge { .. })
        ));
    }
}
