//! # Chapter 5.1 — Basis columns
//!
//! ## Purpose
//! The simplex method never forms its basis matrix `B` explicitly: `B` is a
//! list of `m` references into the computational form `[A | ±I]`. This
//! chapter fixes how the factorization receives that list, so that the
//! simplex (Chapter 6) can call [`crate::LuFactor::factorize`] with its
//! basis heading and the constraint matrix `A`, without copying a column.
//!
//! The factorization reads `B` through the trait [`ColumnSource`] (the
//! contract's name): the pair `(&[ColumnRef], &CscMatrix)` of the simplex
//! is one source, a square [`CscMatrix`] (`B` given explicitly) another.
//! The trait is used by static dispatch only — `factorize` is generic in
//! the source and the visitor of the entries is a generic closure — so the
//! load of the factorization (step L1) costs what it cost in the first
//! edition, where it was hard-wired to the pair.
//!
//! ## Notation
//! `A ∈ R^{m×n}` is a [`CscMatrix`]. The basis is `B = [b_0 | … | b_{m−1}]`,
//! and `k ∈ 0..m` is the *position* of a basic variable (the index of `x_B`
//! it occupies). Each column is either
//!
//! * structural, `b_k = a_j = A e_j` ([`ColumnRef::Structural`]), or
//! * logical, `b_k = σ e_i` with `σ ∈ {+1, −1}` ([`ColumnRef::Logical`]) —
//!   the column of the slack of row `i`; the sign lets the simplex use either
//!   convention (`A x − s = 0` gives `σ = −1`, `A x + s = b` gives `σ = +1`).
//!
//! Positions and rows are different index spaces even though both range over
//! `0..m`: FTRAN maps a vector indexed by rows to one indexed by positions,
//! BTRAN the other way round (Chapter 5.4).
//!
//! ## Invariants
//! (B1) Structural indices are `< n`, logical rows are `< m`, and `A` has
//!      exactly `m` rows; checked by [`ColumnSource::validate`], which
//!      [`crate::LuFactor::factorize`] calls first, reporting a violation as
//!      [`crate::LuError::Structure`].
//! Repeated references (the same column twice) are *not* an input error:
//! they make `B` singular, which the factorization diagnoses.
//!
//! ## References
//! Maros (2003), §5.1 (computational form and logical variables);
//! Koberstein (2005), §2.2.

use crate::real::Real;
use crate::{Col, CscMatrix, Row, SparseError};

/// Sign `σ` of a logical column `σ e_i`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Sign {
    /// `σ = +1`.
    Plus,
    /// `σ = −1`.
    Minus,
}

impl Sign {
    /// The value `σ` as a float.
    #[inline]
    pub fn value(self) -> f64 {
        match self {
            Sign::Plus => 1.0,
            Sign::Minus => -1.0,
        }
    }

    /// The value `σ` in the scalar type `T`.
    #[inline]
    pub fn of<T: Real>(self) -> T {
        T::from_f64(self.value())
    }
}

/// A column of the basis: a structural column of `A` or a logical `σ e_i`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ColumnRef {
    /// Column `j` of `A`.
    Structural(Col),
    /// The logical column `σ e_i` of row `i`.
    Logical(Row, Sign),
}

impl ColumnRef {
    /// Calls `f(i, b_ik)` for every stored entry of the column (in increasing
    /// row order for structural columns).
    ///
    /// # Panics
    /// If a structural index is out of range for `a`.
    #[inline]
    pub fn for_each_entry<T: Real>(self, a: &CscMatrix<T>, mut f: impl FnMut(u32, T)) {
        match self {
            ColumnRef::Structural(j) => {
                let (rows, vals) = a.col(j);
                for (&i, &v) in rows.iter().zip(vals) {
                    f(i, v);
                }
            }
            ColumnRef::Logical(i, s) => f(i.0, s.of()),
        }
    }

    /// Checks (B1) for this column of a basis of `A`.
    pub(crate) fn validate<T: Real>(self, a: &CscMatrix<T>) -> Result<(), SparseError> {
        match self {
            ColumnRef::Structural(j) if j.index() >= a.n_cols() => {
                Err(SparseError::IndexOutOfRange {
                    what: "structural basis column",
                    index: j.index(),
                    bound: a.n_cols(),
                })
            }
            ColumnRef::Logical(i, _) if i.index() >= a.n_rows() => {
                Err(SparseError::IndexOutOfRange {
                    what: "logical basis row",
                    index: i.index(),
                    bound: a.n_rows(),
                })
            }
            _ => Ok(()),
        }
    }
}

/// Where the factorization reads the columns of `B` from (the contract's
/// `ColumnSource`). Static dispatch only: [`crate::LuFactor::factorize`] is
/// generic in the source, and [`ColumnSource::for_each_entry`] takes a
/// generic visitor.
pub trait ColumnSource<T: Real> {
    /// Order `m` of `B` (its number of columns).
    fn m(&self) -> usize;
    /// Calls `f(i, b_ik)` for every stored entry of column `k`, rows
    /// increasing for structural columns.
    fn for_each_entry<F: FnMut(u32, T)>(&self, k: usize, f: F);
    /// Upper bound on the nonzeros of `B` (an allocation hint).
    fn nnz_hint(&self) -> usize;
    /// Checks that the source describes a square `m × m` matrix whose
    /// references are in range (invariant (B1)).
    fn validate(&self) -> Result<(), SparseError>;
}

/// The simplex basis: the heading `cols` over `[A | ±I]`.
impl<T: Real> ColumnSource<T> for (&[ColumnRef], &CscMatrix<T>) {
    #[inline]
    fn m(&self) -> usize {
        self.0.len()
    }
    #[inline]
    fn for_each_entry<F: FnMut(u32, T)>(&self, k: usize, f: F) {
        self.0[k].for_each_entry(self.1, f);
    }
    fn nnz_hint(&self) -> usize {
        self.0
            .iter()
            .map(|c| match *c {
                ColumnRef::Structural(j) => {
                    self.1.col_ptr[j.index() + 1] - self.1.col_ptr[j.index()]
                }
                ColumnRef::Logical(..) => 1,
            })
            .sum()
    }
    fn validate(&self) -> Result<(), SparseError> {
        let (basis, a) = *self;
        if a.n_rows() != basis.len() {
            return Err(SparseError::Dimension {
                what: "basis (must have one column per row of A)",
                expected: a.n_rows(),
                found: basis.len(),
            });
        }
        for c in basis {
            c.validate(a)?;
        }
        Ok(())
    }
}

/// An explicit square matrix `B`.
impl<T: Real> ColumnSource<T> for CscMatrix<T> {
    #[inline]
    fn m(&self) -> usize {
        self.n_cols()
    }
    #[inline]
    fn for_each_entry<F: FnMut(u32, T)>(&self, k: usize, mut f: F) {
        let (rows, vals) = self.col(Col::from_usize(k));
        for (&i, &v) in rows.iter().zip(vals) {
            f(i, v);
        }
    }
    fn nnz_hint(&self) -> usize {
        self.nnz()
    }
    fn validate(&self) -> Result<(), SparseError> {
        if self.n_rows() == self.n_cols() {
            Ok(())
        } else {
            Err(SparseError::Dimension {
                what: "matrix B (must be square)",
                expected: self.n_rows(),
                found: self.n_cols(),
            })
        }
    }
}
