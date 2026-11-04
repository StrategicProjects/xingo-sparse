//! # Chapter 5.0 — Errors
//!
//! ## Purpose
//! Two kinds of failure are reported, never panicked on. The containers
//! (Chapter 5.8) validate outside data — triplets, raw compressed arrays —
//! and report the first violation found as a [`SparseError`], which names
//! the offending array, position and value; every `check_invariants()` of
//! the crate returns the same type. The factorization reports a singular
//! basis, an inconsistent basis description, or an update called without
//! its spike, as an [`LuError`] carrying what the caller needs to recover.
//!
//! `xingo-core` converts both into its `CoreError` (`From` impls there):
//! a [`SparseError`] maps variant by variant onto the `CoreError` that the
//! first edition reported, so the messages the solver prints are
//! unchanged.
//!
//! ## Invariants
//! (E1) `Singular { rank_deficiency, cols, rows }` has
//!      `rank_deficiency = cols.len() = rows.len() ≥ 1`. Replacing, for
//!      every `k`, the basis column at position `cols[k]` by the logical
//!      column of row `rows[k]` yields a basis whose factorization succeeds
//!      (tested in Chapter 5.3); this is the classic basis repair of the
//!      simplex method (Koberstein 2005, §8.2.3; HiGHS
//!      `HFactor::buildSimple` rank deficiency handling).
//! (E2) Errors are reported, never repaired silently: a constructor either
//!      returns a structure satisfying all of its invariants or an error.

use std::fmt;

use crate::Row;

/// Structural error of the containers: invariants (C1)–(C4) of Chapter 5.8
/// or (V1)–(V3) of Chapter 5.8.3 violated by the input, or the basis
/// references of Chapter 5.1 out of range.
///
/// The `what` fields are short static descriptions of the offending array
/// or quantity (for instance `"row index"` or `"value array"`).
#[derive(Clone, Debug, PartialEq)]
pub enum SparseError {
    /// Two lengths that must agree do not.
    Dimension {
        /// Which array or dimension has the wrong length.
        what: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        found: usize,
    },
    /// An index is not smaller than the dimension it refers to.
    IndexOutOfRange {
        /// What kind of index was out of range.
        what: &'static str,
        /// The offending index.
        index: usize,
        /// The exclusive upper bound it violated.
        bound: usize,
    },
    /// A stored value is not allowed (non-finite, or an explicit zero
    /// where (C3) forbids it).
    InvalidValue {
        /// What the value represents.
        what: &'static str,
        /// Position of the value.
        index: usize,
        /// The offending value, as the nearest `f64`.
        value: f64,
    },
    /// A dimension does not fit in the `u32` index space.
    TooLarge {
        /// What dimension overflowed.
        what: &'static str,
        /// The offending dimension.
        value: usize,
    },
    /// Pointers not nondecreasing, indices not sorted, a pattern
    /// inconsistent with its marks…; the message names the invariant.
    Malformed(String),
}

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfRange { what, index, bound } => {
                write!(f, "{what} {index} out of bounds (must be < {bound})")
            }
            Self::Dimension {
                what,
                expected,
                found,
            } => {
                write!(f, "{what} has length {found}, expected {expected}")
            }
            Self::InvalidValue { what, index, value } => {
                write!(f, "invalid {what} at position {index}: {value}")
            }
            Self::TooLarge { what, value } => {
                write!(f, "{what} = {value} exceeds the u32 index space")
            }
            Self::Malformed(msg) => write!(f, "invariant violated: {msg}"),
        }
    }
}

impl std::error::Error for SparseError {}

/// Error raised by [`crate::LuFactor::factorize`] and
/// [`crate::LuFactor::update`].
#[derive(Clone, Debug, PartialEq)]
pub enum LuError {
    /// The basis is singular: no acceptable pivot (magnitude above
    /// [`crate::LuOptions::abs_pivot_tol`]) exists for `rank_deficiency`
    /// columns. `cols` lists their basis positions and `rows` the rows
    /// left without a pivot, both in increasing order (invariant E1).
    ///
    /// After a failed factorization the factor is invalid until the next
    /// successful one; after a failed update it still represents the
    /// previous basis (Chapter 5.5).
    Singular {
        /// Number of dependent columns, `= cols.len() = rows.len()`.
        rank_deficiency: usize,
        /// Basis positions of the dependent columns.
        cols: Vec<usize>,
        /// Rows left without a pivot.
        rows: Vec<Row>,
    },
    /// The basis references do not fit the matrix (Chapter 5.1, B1), or a
    /// position is out of range.
    Structure(SparseError),
    /// [`crate::LuFactor::update`] was called without a spike recorded by
    /// [`crate::LuFactor::ftran_for_update`] since the last change of the
    /// factor.
    NoSpike,
    /// A solve or an update was requested from a factor that does not hold
    /// a valid factorization (never factorized, or the last factorization
    /// failed).
    NotFactorized,
}

impl fmt::Display for LuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Singular {
                rank_deficiency,
                cols,
                ..
            } => write!(
                f,
                "singular basis: rank deficiency {rank_deficiency} (positions {cols:?})"
            ),
            Self::Structure(e) => write!(f, "invalid basis: {e}"),
            Self::NoSpike => write!(f, "update called without a spike from ftran_for_update"),
            Self::NotFactorized => write!(f, "the factor holds no valid factorization"),
        }
    }
}

impl std::error::Error for LuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Structure(e) => Some(e),
            _ => None,
        }
    }
}

impl From<SparseError> for LuError {
    fn from(e: SparseError) -> Self {
        Self::Structure(e)
    }
}
