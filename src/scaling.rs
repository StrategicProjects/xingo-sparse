//! # Chapter 5.13 — Row and column scaling
//!
//! *New with task L1.5 (PLAN.md trilha L; the contract's §5.4).*
//!
//! ## Purpose
//! The entries of a constraint matrix often span many orders of magnitude
//! (`1e-6` next to `1e6` in the same row), and every factorization pays
//! for it: pivot tests compare magnitudes, growth is measured against
//! `max|a|`, and the condition of `A D Aᵀ` in an interior-point method
//! is that of `A` squared. *Equilibration* replaces `A` by
//! `A' = R A C` with positive diagonal `R = diag(row)`, `C = diag(col)`
//! chosen so that the nonzeros of `A'` are close to one in magnitude;
//! solutions are mapped back afterwards. This chapter computes such
//! factors by three classic policies ([`Equilibration`]) and applies or
//! removes them ([`Scaling`]).
//!
//! ## Notation
//! `A ∈ R^{m×n}` with nonzeros `a_ij`, `(i, j) ∈ 𝒩`; `r_i`, `c_j > 0` the
//! row and column factors; `a'_ij = r_i a_ij c_j`. The *spread* of a
//! matrix is `max |a'_ij| / min |a'_ij|` over `𝒩` (one for a matrix
//! without nonzeros). For the logarithmic policies `v_ij = log₂|a_ij|`,
//! `ρ_i = log₂ r_i`, `γ_j = log₂ c_j`; `n_i` and `n_j` count the nonzeros
//! of row `i` and column `j`. Rounding to a power of two is
//! `⌊x⌉₂ = 2^{round(log₂ x)}`.
//!
//! ## Algorithm G (Geometric mean iteration; Maros 2003, §6.2)
//! G1. Set `r_i ← 1`, `c_j ← 1`.
//! G2. Repeat `max_iters` times:
//!     (a) for every nonempty row `i`, `r_i ← 1 / √(max_j |a_ij| c_j ·
//!         min_j |a_ij| c_j)`;
//!     (b) for every nonempty column `j`, `c_j ← 1 / √(max_i r_i |a_ij| ·
//!         min_i r_i |a_ij|)`.
//! G3. If `power_of_two`, replace every factor `f` by `⌊f⌉₂`.
//!
//! Each half-step makes the largest and the smallest entry of a row (a
//! column) reciprocal, so the spread of every row and column is reduced
//! towards its square root; a few passes suffice in practice (Maros
//! recommends 4–8).
//!
//! ## Algorithm R (Curtis & Reid 1972, least squares)
//! Find `ρ`, `γ` minimizing `Σ_{(i,j)∈𝒩} (v_ij + ρ_i + γ_j)²`, the sum of
//! squared logarithms of the scaled entries. The normal equations are
//! ```text
//!   n_i ρ_i + Σ_{j:(i,j)∈𝒩} γ_j = −σ_i,   σ_i = Σ_j v_ij   (rows)
//!   Σ_{i:(i,j)∈𝒩} ρ_i + n_j γ_j = −τ_j,   τ_j = Σ_i v_ij   (columns)
//! ```
//! a singular but consistent positive semidefinite system (adding `t` to
//! every `ρ_i` and `−t` to every `γ_j` changes nothing).
//! R1. Form `σ`, `τ`, `n_i`, `n_j`.
//! R2. Solve the normal equations by conjugate gradients preconditioned by
//!     their diagonal `diag(n_i, n_j)` (Curtis & Reid's own iteration is
//!     this method), from `ρ = γ = 0`, for at most `max_iters` steps or
//!     until the preconditioned residual `rᵀ P⁻¹ r` falls to `ε_{f64}`
//!     times its initial value (the precision of the computation, not a
//!     tuning constant). Empty rows and columns keep `ρ_i = 0`, `γ_j = 0`.
//! R3. `r_i ← ⌊2^{ρ_i}⌉₂`, `c_j ← ⌊2^{γ_j}⌉₂`: Curtis & Reid round to
//!     powers of two, and so do we (exact unscaling, (S8)).
//!
//! ## Algorithm I (Infinity norm, once)
//! I1. `r_i ← 1 / max_j |a_ij|` for every nonempty row.
//! I2. `c_j ← 1 / max_i r_i |a_ij|` for every nonempty column.
//! After it every row and column of `A'` has infinity norm at most one and
//! every column exactly one. The factors are not rounded.
//!
//! ## Applying and removing
//! [`Scaling::apply`] maps each stored value to `(a_ij · r_i) · c_j`,
//! [`Scaling::unapply`] to `(a'_ij / r_i) / c_j`. The factors are
//! computed in `f64` (they are a policy, not data) and enter `T` through
//! [`Real::from_f64`].
//!
//! ## Invariants
//! (S8) (the contract's) *Round trip.* When every factor is a power of two
//!      — always for Algorithm R, for Algorithm G with `power_of_two` —
//!      `unapply(apply(A)) = A` bit for bit, provided no product leaves the
//!      normal range of `T` (no overflow, no subnormal result): a
//!      multiplication or division by `2^k` only changes the exponent.
//!      Otherwise the round trip holds to working precision (two
//!      roundings per entry each way). Checked by the tests of Chapter
//!      5.13 and by a property test.
//! (S9) Every factor is finite and positive; a row or column without
//!      nonzeros has factor one. Asserted in debug builds by
//!      [`Scaling::compute`] and checked by [`Scaling::check_invariants`].
//!
//! ## References
//! Curtis & Reid, "On the automatic scaling of matrices for Gaussian
//! elimination", J. Inst. Maths Applics 10, 1972; Maros, *Computational
//! Techniques of the Simplex Method* (2003), ch. 6; Duff, Erisman & Reid,
//! *Direct Methods for Sparse Matrices* (2017), §6.10; HiGHS
//! `HighsLpUtils` scaling routines (read, not copied).

use crate::real::{Real, approx_eq, is_exact_zero, le_exact, lt};
use crate::{CscMatrix, SparseError};

/// Row/column equilibration policies (Algorithms G, R and I).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Equilibration {
    /// Geometric mean iteration (Maros 2003, ch. 6), Algorithm G, rounded
    /// to powers of two when `power_of_two` (exact unscaling, (S8)).
    Geometric {
        /// Passes over rows and columns.
        max_iters: usize,
        /// Round the factors to powers of two.
        power_of_two: bool,
    },
    /// Curtis & Reid (1972) least-squares scaling, Algorithm R: at most
    /// `max_iters` conjugate-gradient steps; factors are powers of two.
    CurtisReid {
        /// Conjugate-gradient steps.
        max_iters: usize,
    },
    /// Rows then columns by the infinity norm, once (Algorithm I).
    InfinityNorm,
}

/// Row and column factors: `A' = diag(row) · A · diag(col)` (invariants
/// (S8), (S9)).
#[derive(Clone, Debug, PartialEq)]
pub struct Scaling<T: Real = f64> {
    /// `r_i`, one per row.
    pub row: Vec<T>,
    /// `c_j`, one per column.
    pub col: Vec<T>,
}

impl<T: Real> Scaling<T> {
    /// The factors of `a` under `policy` (Algorithms G, R, I).
    ///
    /// ```
    /// use xingo_sparse::{CscMatrix, Col, Equilibration, Row, Scaling};
    /// let a = CscMatrix::from_triplets(2, 2, &[
    ///     (Row(0), Col(0), 1e-3), (Row(1), Col(0), 4.0), (Row(1), Col(1), 1e3),
    /// ])
    /// .unwrap();
    /// let policy = Equilibration::Geometric { max_iters: 4, power_of_two: true };
    /// let s = Scaling::compute(&a, policy);
    /// assert!(s.spread(&a) < 1e6); // the spread of `a` itself is 1e3 / 1e-3
    /// let mut b = a.clone();
    /// s.apply(&mut b);
    /// s.unapply(&mut b);
    /// assert_eq!(a, b); // (S8): powers of two round-trip exactly
    /// ```
    #[must_use]
    pub fn compute(a: &CscMatrix<T>, policy: Equilibration) -> Self {
        let (row, col) = match policy {
            Equilibration::Geometric {
                max_iters,
                power_of_two,
            } => geometric(a, max_iters, power_of_two),
            Equilibration::CurtisReid { max_iters } => curtis_reid(a, max_iters),
            Equilibration::InfinityNorm => infinity_norm(a),
        };
        let s = Self {
            row: row.into_iter().map(T::from_f64).collect(),
            col: col.into_iter().map(T::from_f64).collect(),
        };
        debug_assert_eq!(s.check_invariants(), Ok(()));
        debug_assert!(
            !matches!(
                policy,
                Equilibration::CurtisReid { .. }
                    | Equilibration::Geometric {
                        power_of_two: true,
                        ..
                    }
            ) || s.is_power_of_two(),
            "(S8): the policy rounds to powers of two"
        );
        s
    }

    /// `A ← diag(row) · A · diag(col)` in place.
    ///
    /// # Panics
    /// If the factor lengths do not match the dimensions of `a`.
    pub fn apply(&self, a: &mut CscMatrix<T>) {
        self.assert_fits(a);
        for j in 0..a.n_cols() {
            let cj = self.col[j];
            for p in a.col_ptr[j]..a.col_ptr[j + 1] {
                let ri = self.row[a.row_idx[p] as usize];
                a.val[p] = a.val[p] * ri * cj;
            }
        }
    }

    /// `A ← diag(row)⁻¹ · A · diag(col)⁻¹` in place: the inverse of
    /// [`Scaling::apply`], exactly so for powers of two ((S8)).
    ///
    /// # Panics
    /// If the factor lengths do not match the dimensions of `a`.
    pub fn unapply(&self, a: &mut CscMatrix<T>) {
        self.assert_fits(a);
        for j in 0..a.n_cols() {
            let cj = self.col[j];
            for p in a.col_ptr[j]..a.col_ptr[j + 1] {
                let ri = self.row[a.row_idx[p] as usize];
                a.val[p] = a.val[p] / ri / cj;
            }
        }
    }

    /// The spread `max|a'| / min|a'|` over the nonzeros of the scaled
    /// matrix `a' = diag(row) · a · diag(col)` (`a` itself is not
    /// modified); one when `a` has no nonzeros.
    ///
    /// # Panics
    /// If the factor lengths do not match the dimensions of `a`.
    pub fn spread(&self, a: &CscMatrix<T>) -> T {
        self.assert_fits(a);
        let mut big = T::ZERO;
        let mut small = T::ZERO;
        let mut any = false;
        for j in 0..a.n_cols() {
            for p in a.col_ptr[j]..a.col_ptr[j + 1] {
                let v = (a.val[p] * self.row[a.row_idx[p] as usize] * self.col[j]).abs();
                if any {
                    big = big.max(v);
                    small = small.min(v);
                } else {
                    (big, small, any) = (v, v, true);
                }
            }
        }
        if any && !is_exact_zero(small) {
            big / small
        } else {
            T::ONE
        }
    }

    /// Checks (S9): every factor finite and positive.
    pub fn check_invariants(&self) -> Result<(), SparseError> {
        for (what, f) in [
            ("row scale factor", &self.row),
            ("column scale factor", &self.col),
        ] {
            for (k, &x) in f.iter().enumerate() {
                if !x.is_finite() || !lt(T::ZERO, x) {
                    return Err(SparseError::InvalidValue {
                        what,
                        index: k,
                        value: x.to_f64(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Every factor is a power of two (the premise of the exact case of
    /// (S8)).
    fn is_power_of_two(&self) -> bool {
        self.row
            .iter()
            .chain(&self.col)
            .all(|&x| approx_eq(x.to_f64(), round_pow2(x.to_f64()), 0.0))
    }

    fn assert_fits(&self, a: &CscMatrix<T>) {
        assert_eq!(self.row.len(), a.n_rows, "scaling: row factors do not fit");
        assert_eq!(
            self.col.len(),
            a.n_cols(),
            "scaling: column factors do not fit"
        );
    }
}

/// `⌊x⌉₂ = 2^{round(log₂ x)}` for a finite `x > 0`.
fn round_pow2(x: f64) -> f64 {
    x.log2().round().exp2()
}

/// Algorithm G.
fn geometric<T: Real>(
    a: &CscMatrix<T>,
    max_iters: usize,
    power_of_two: bool,
) -> (Vec<f64>, Vec<f64>) {
    let (m, n) = (a.n_rows, a.n_cols());
    // G1.
    let mut row = vec![1.0_f64; m];
    let mut col = vec![1.0_f64; n];
    let mut rmax = vec![0.0_f64; m];
    let mut rmin = vec![f64::INFINITY; m];
    for _ in 0..max_iters {
        // G2 (a): row extremes of |a_ij| c_j, gathered column by column.
        rmax.fill(0.0);
        rmin.fill(f64::INFINITY);
        for (j, &cj) in col.iter().enumerate() {
            for p in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[p] as usize;
                let v = a.val[p].to_f64().abs() * cj;
                rmax[i] = rmax[i].max(v);
                rmin[i] = rmin[i].min(v);
            }
        }
        for i in 0..m {
            if lt(0.0, rmax[i]) {
                row[i] = (rmax[i] * rmin[i]).sqrt().recip();
            }
        }
        // G2 (b).
        for (j, cj) in col.iter_mut().enumerate() {
            let (mut big, mut small) = (0.0_f64, f64::INFINITY);
            for p in a.col_ptr[j]..a.col_ptr[j + 1] {
                let v = row[a.row_idx[p] as usize] * a.val[p].to_f64().abs();
                big = big.max(v);
                small = small.min(v);
            }
            if lt(0.0, big) {
                *cj = (big * small).sqrt().recip();
            }
        }
    }
    // G3.
    if power_of_two {
        for f in row.iter_mut().chain(col.iter_mut()) {
            *f = round_pow2(*f);
        }
    }
    (row, col)
}

/// Algorithm R: preconditioned conjugate gradients on the normal
/// equations of the Curtis–Reid least-squares problem.
#[allow(clippy::cast_precision_loss)] // counts of nonzeros, far below 2⁵³
fn curtis_reid<T: Real>(a: &CscMatrix<T>, max_iters: usize) -> (Vec<f64>, Vec<f64>) {
    let (m, n) = (a.n_rows, a.n_cols());
    // R1.
    let mut cnt = vec![0.0_f64; m + n];
    let mut g = vec![0.0_f64; m + n];
    for j in 0..n {
        for p in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[p] as usize;
            let v = a.val[p].to_f64().abs().log2();
            g[i] += v;
            g[m + j] += v;
            cnt[i] += 1.0;
            cnt[m + j] += 1.0;
        }
    }
    // The operator z ↦ M z of the normal equations.
    let apply_m = |z: &[f64], out: &mut [f64]| {
        for (o, (&c, &zk)) in out.iter_mut().zip(cnt.iter().zip(z)) {
            *o = c * zk;
        }
        for j in 0..n {
            for p in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[p] as usize;
                out[i] += z[m + j];
                out[m + j] += z[i];
            }
        }
    };
    // Diagonal preconditioner; empty rows/columns are excluded (zero).
    let pinv: Vec<f64> = cnt
        .iter()
        .map(|&c| if lt(0.0, c) { c.recip() } else { 0.0 })
        .collect();
    let dot = |x: &[f64], y: &[f64]| -> f64 { x.iter().zip(y).map(|(a, b)| a * b).sum() };
    // R2. z = 0, residual = rhs = −g.
    let mut z = vec![0.0_f64; m + n];
    let mut res: Vec<f64> = g.iter().map(|&v| -v).collect();
    let mut s: Vec<f64> = res.iter().zip(&pinv).map(|(r, p)| r * p).collect();
    let mut dir = s.clone();
    let mut q = vec![0.0_f64; m + n];
    let mut rs = dot(&res, &s);
    let stop = rs * f64::EPSILON;
    for _ in 0..max_iters {
        if le_exact(rs, stop) {
            break;
        }
        apply_m(&dir, &mut q);
        let dq = dot(&dir, &q);
        if !lt(0.0, dq) {
            break;
        }
        let alpha = rs / dq;
        for k in 0..m + n {
            z[k] += alpha * dir[k];
            res[k] -= alpha * q[k];
            s[k] = res[k] * pinv[k];
        }
        let rs_new = dot(&res, &s);
        let beta = rs_new / rs;
        for k in 0..m + n {
            dir[k] = s[k] + beta * dir[k];
        }
        rs = rs_new;
    }
    // R3.
    let row = z[..m].iter().map(|&x| x.round().exp2()).collect();
    let col = z[m..].iter().map(|&x| x.round().exp2()).collect();
    (row, col)
}

/// Algorithm I.
fn infinity_norm<T: Real>(a: &CscMatrix<T>) -> (Vec<f64>, Vec<f64>) {
    let (m, n) = (a.n_rows, a.n_cols());
    let mut rmax = vec![0.0_f64; m];
    for j in 0..n {
        for p in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[p] as usize;
            rmax[i] = rmax[i].max(a.val[p].to_f64().abs());
        }
    }
    // I1.
    let row: Vec<f64> = rmax
        .iter()
        .map(|&v| if lt(0.0, v) { v.recip() } else { 1.0 })
        .collect();
    // I2.
    let col = (0..n)
        .map(|j| {
            let big = (a.col_ptr[j]..a.col_ptr[j + 1])
                .map(|p| row[a.row_idx[p] as usize] * a.val[p].to_f64().abs())
                .fold(0.0_f64, f64::max);
            if lt(0.0, big) { big.recip() } else { 1.0 }
        })
        .collect();
    (row, col)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Col, Row};
    use proptest::prelude::*;

    fn wide() -> CscMatrix {
        CscMatrix::from_triplets(
            3,
            3,
            &[
                (Row(0), Col(0), 1e-4),
                (Row(0), Col(2), 3e2),
                (Row(1), Col(0), 2.0),
                (Row(1), Col(1), -7e5),
                (Row(2), Col(1), 5e-3),
                (Row(2), Col(2), 1.5),
            ],
        )
        .unwrap()
    }

    const POLICIES: [Equilibration; 4] = [
        Equilibration::Geometric {
            max_iters: 6,
            power_of_two: true,
        },
        Equilibration::Geometric {
            max_iters: 6,
            power_of_two: false,
        },
        Equilibration::CurtisReid { max_iters: 50 },
        Equilibration::InfinityNorm,
    ];

    #[test]
    fn every_policy_reduces_the_spread() {
        let a = wide();
        let unit = Scaling {
            row: vec![1.0; 3],
            col: vec![1.0; 3],
        };
        let before = unit.spread(&a);
        for policy in POLICIES {
            let s = Scaling::compute(&a, policy);
            assert_eq!(s.check_invariants(), Ok(()));
            assert!(
                lt(s.spread(&a), before),
                "{policy:?}: {} vs {before}",
                s.spread(&a)
            );
        }
    }

    #[test]
    fn s8_round_trip() {
        let a = wide();
        for policy in POLICIES {
            let s = Scaling::compute(&a, policy);
            let mut b = a.clone();
            s.apply(&mut b);
            s.unapply(&mut b);
            if s.is_power_of_two() {
                assert_eq!(a, b, "{policy:?}");
            } else {
                for (x, y) in a.val.iter().zip(&b.val) {
                    assert!(approx_eq(*x, *y, 4.0 * f64::EPSILON), "{policy:?}");
                }
            }
        }
    }

    #[test]
    fn infinity_norm_makes_columns_unit() {
        let a = wide();
        let s = Scaling::compute(&a, Equilibration::InfinityNorm);
        let mut b = a.clone();
        s.apply(&mut b);
        for j in 0..3 {
            let (_, v) = b.col(Col::from_usize(j));
            let big = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            assert!(approx_eq(big, 1.0, 4.0 * f64::EPSILON));
        }
    }

    #[test]
    fn empty_rows_and_columns_keep_unit_factors() {
        let a = CscMatrix::from_triplets(3, 3, &[(Row(0), Col(0), 8.0)]).unwrap();
        for policy in POLICIES {
            let s = Scaling::compute(&a, policy);
            assert!(approx_eq(s.row[1], 1.0, 0.0) && approx_eq(s.col[2], 1.0, 0.0));
            assert!(approx_eq(s.spread(&a), 1.0, 0.0));
        }
        let e = CscMatrix::<f64>::zeros(2, 2);
        assert!(approx_eq(
            Scaling::compute(&e, POLICIES[0]).spread(&e),
            1.0,
            0.0
        ));
    }

    #[test]
    fn curtis_reid_is_exact_on_a_rank_one_magnitude_pattern() {
        // |a_ij| = 2^{p_i + q_j}: the least-squares optimum makes every
        // scaled entry ±1.
        let p = [3, -5, 0];
        let q = [-2, 7, 1];
        let mut t = Vec::new();
        for (i, &pi) in p.iter().enumerate() {
            for (j, &qj) in q.iter().enumerate() {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                t.push((
                    Row::from_usize(i),
                    Col::from_usize(j),
                    sign * f64::from(pi + qj).exp2(),
                ));
            }
        }
        let a = CscMatrix::from_triplets(3, 3, &t).unwrap();
        let s = Scaling::compute(&a, Equilibration::CurtisReid { max_iters: 20 });
        assert!(approx_eq(s.spread(&a), 1.0, 0.0));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        /// (S8) on random matrices whose magnitudes span 2^±40.
        #[test]
        fn s8_round_trip_is_exact_for_powers_of_two(
            m in 1_usize..12, n in 1_usize..12,
            entries in proptest::collection::vec((0_usize..12, 0_usize..12, -40_i32..40, 0.5_f64..1.0), 1..60),
        ) {
            let t: Vec<_> = entries
                .iter()
                .filter(|e| e.0 < m && e.1 < n)
                .map(|&(i, j, e, f)| (Row::from_usize(i), Col::from_usize(j), f * f64::from(e).exp2()))
                .collect();
            let a = CscMatrix::from_triplets(m, n, &t).unwrap();
            for policy in POLICIES {
                let s = Scaling::compute(&a, policy);
                let mut b = a.clone();
                s.apply(&mut b);
                // Algorithm I makes every column's largest entry one.
                if policy == Equilibration::InfinityNorm {
                    for j in 0..n {
                        let big = b.col(Col::from_usize(j)).1.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                        prop_assert!(is_exact_zero(big) || approx_eq(big, 1.0, 4.0 * f64::EPSILON));
                    }
                }
                s.unapply(&mut b);
                if s.is_power_of_two() {
                    prop_assert_eq!(&a, &b);
                }
            }
        }
    }
}
