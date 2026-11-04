//! # Chapter 5.8.3 — The sparse work vector
//!
//! ## Purpose
//! The simplex kernels (FTRAN, BTRAN, pivot row, updates) produce vectors
//! whose density ranges from a few entries to completely dense, often within
//! the same solve. [`SparseVec`] serves both regimes, in the spirit of
//! HiGHS's `HVector`: values live in a *dense work array* of length `size`,
//! so any entry is read or written in `O(1)`, and an optional *pattern* lists
//! the positions that may be nonzero, so that sparse operations cost
//! `O(count)` instead of `O(size)`.
//!
//! ## Notation
//! `y ∈ R^size` is the vector, stored densely in `val`. When the vector is in
//! *sparse mode*, `P = idx[0..count]` is its pattern; in *dense mode* no
//! pattern is kept and `count = size` by convention. A packed vector is a
//! pair of slices `(ι, ν)` meaning `Σ_k ν_k e_{ι_k}`.
//!
//! Operations (each with a sparse path, used when the relevant pattern is
//! known, and a dense path over all `size` positions):
//!
//! * `clear`: `y ← 0`;
//! * `scatter(α, ι, ν)`: `y ← y + α Σ_k ν_k e_{ι_k}` (packed → work);
//! * `axpy(α, x)`: `y ← y + α x` for another work vector `x`;
//! * `gather`: work → packed list of the nonzero entries;
//! * `tidy`: zero out entries with `|y_i| ≤ ε` (a drop tolerance) and drop
//!   them from `P`.
//!
//! ## Algorithm S (Sparse update of the work vector)
//! To add `δ` to `y_i`:
//!
//! S1. [Record.] If the vector is in sparse mode and `mark[i]` is false, set
//!     `mark[i] ← true` and append `i` to the pattern.
//! S2. [Update.] Set `val[i] ← val[i] + δ`.
//!
//! `axpy` applies S to every `i` with `x_i ≠ 0`, enumerating `i` over the
//! pattern of `x` if it has one and over `0..size` otherwise; the two
//! enumerations visit the same set of positions and perform, at each, the
//! identical floating-point operation, so the sparse and dense paths give
//! bit-for-bit equal values. `clear` resets only the pattern positions when
//! `count ≤ 0.3·size` (the HiGHS threshold, a policy constant of this
//! chapter) and the whole array otherwise.
//!
//! The `mark` array makes membership in `P` an `O(1)` test that does not
//! depend on `val[i] ≠ 0`: an entry that cancels to exactly zero stays in
//! the pattern and is never duplicated. HiGHS instead keeps cancelled entries
//! at a tiny sentinel value; the explicit mark costs one byte per position
//! and keeps exact zeros exact.
//!
//! ## Invariants
//! (V1) `val.len() = mark.len() = size`.
//! (V2) Sparse mode: the entries of `P` are distinct and `< size`;
//!      `mark[i] ⇔ i ∈ P`; and `val[i] ≠ 0 ⇒ i ∈ P` (the pattern is a
//!      superset of the support; it may contain explicit zeros).
//! (V3) Dense mode: `P` is empty and every `mark[i]` is false.
//!
//! Checked by [`SparseVec::check_invariants`].
//!
//! ## References
//! Hall & McKinnon (2005); HiGHS `HVector` (`src/simplex/HVector*`), studied,
//! not copied; Gilbert & Peierls (1988) for why the pattern matters.

use crate::SparseError;
use crate::real::{Real, is_exact_zero, is_zero, le_exact};

/// Fraction of `size` above which `clear` zeroes the whole array (a policy
/// constant, not a numerical tolerance: Chapter 5, §5.C).
const CLEAR_DENSE_FRACTION: f64 = 0.3;

/// Work vector with a dense value array and an optional nonzero pattern.
///
/// See the module documentation for the representation and invariants.
#[derive(Clone, Debug)]
pub struct SparseVec<T: Real = f64> {
    size: u32,
    idx: Vec<u32>,
    val: Vec<T>,
    mark: Vec<bool>,
    sparse: bool,
}

impl<T: Real> SparseVec<T> {
    /// The zero vector of length `size`, in sparse mode with an empty
    /// pattern.
    ///
    /// # Panics
    /// If `size > u32::MAX`.
    pub fn new(size: usize) -> Self {
        let size32 = u32::try_from(size).expect("SparseVec size exceeds u32::MAX");
        Self {
            size: size32,
            idx: Vec::with_capacity(size),
            val: vec![T::ZERO; size],
            mark: vec![false; size],
            sparse: true,
        }
    }

    /// A vector equal to `values`, in sparse mode with the exact support as
    /// pattern (in increasing order).
    pub fn from_dense(values: &[T]) -> Self {
        let mut v = Self::new(values.len());
        v.val.copy_from_slice(values);
        v.sparse = false;
        v.rebuild_pattern();
        v
    }

    /// Length of the vector.
    #[inline]
    pub fn size(&self) -> usize {
        self.size as usize
    }

    /// Number of positions that may be nonzero: `|P|` in sparse mode, `size`
    /// in dense mode.
    #[inline]
    pub fn count(&self) -> usize {
        if self.sparse {
            self.idx.len()
        } else {
            self.size()
        }
    }

    /// Fraction `count / size` (zero for an empty vector).
    pub fn density(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        // count ≤ size ≤ u32::MAX, so the conversion is exact.
        let count = u32::try_from(self.count()).unwrap_or(self.size);
        f64::from(count) / f64::from(self.size)
    }

    /// `true` iff the vector is in sparse mode (it keeps a pattern).
    #[inline]
    pub fn is_sparse(&self) -> bool {
        self.sparse
    }

    /// The pattern `P` in sparse mode (in insertion order), `None` in dense
    /// mode.
    #[inline]
    pub fn pattern(&self) -> Option<&[u32]> {
        self.sparse.then_some(self.idx.as_slice())
    }

    /// The dense value array.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.val
    }

    /// Mutable access to the dense value array. Since arbitrary writes
    /// invalidate the pattern, this switches the vector to dense mode; call
    /// [`SparseVec::rebuild_pattern`] afterwards to return to sparse mode.
    pub fn values_mut(&mut self) -> &mut [T] {
        self.make_dense();
        &mut self.val
    }

    /// Entry `y_i`.
    #[inline]
    pub fn get(&self, i: u32) -> T {
        self.val[i as usize]
    }

    /// Algorithm S, step S1: records `i` in the pattern if needed.
    #[inline]
    fn touch(&mut self, i: u32) {
        let iu = i as usize;
        if self.sparse && !self.mark[iu] {
            self.mark[iu] = true;
            self.idx.push(i);
        }
    }

    /// Sets `y_i ← v`.
    #[inline]
    pub fn set(&mut self, i: u32, v: T) {
        self.touch(i);
        self.val[i as usize] = v;
    }

    /// Algorithm S: `y_i ← y_i + δ`.
    #[inline]
    pub fn add(&mut self, i: u32, delta: T) {
        self.touch(i);
        self.val[i as usize] += delta;
    }

    /// Sets `y ← 0` and returns to sparse mode with an empty pattern.
    /// Costs `O(count)` when `count ≤ 0.3·size`, `O(size)` otherwise.
    pub fn clear(&mut self) {
        let partial = self.sparse && le_exact(self.density(), CLEAR_DENSE_FRACTION);
        if partial {
            for &i in &self.idx {
                self.val[i as usize] = T::ZERO;
            }
        } else {
            self.val.fill(T::ZERO);
        }
        self.unmark_pattern();
        self.sparse = true;
    }

    /// Scatter: `y ← y + α Σ_k ν_k e_{ι_k}` for the packed vector
    /// `(idx, val)`. Repeated indices accumulate.
    ///
    /// # Panics
    /// If the slices differ in length or an index is `≥ size`.
    pub fn scatter(&mut self, alpha: T, idx: &[u32], val: &[T]) {
        assert_eq!(
            idx.len(),
            val.len(),
            "scatter: index and value lengths differ"
        );
        for (&i, &v) in idx.iter().zip(val) {
            self.add(i, alpha * v);
        }
    }

    /// `y ← y + α x`. Uses the pattern of `x` when it has one, and a dense
    /// sweep of `x` otherwise; both paths update exactly the positions with
    /// `x_i ≠ 0`, so the results are identical. The pattern of `y` is kept
    /// if `y` is in sparse mode.
    ///
    /// # Panics
    /// If the sizes differ.
    pub fn axpy(&mut self, alpha: T, x: &SparseVec<T>) {
        assert_eq!(self.size, x.size, "axpy: size mismatch");
        if x.sparse {
            for &i in &x.idx {
                let xi = x.val[i as usize];
                if !is_exact_zero(xi) {
                    self.add(i, alpha * xi);
                }
            }
        } else {
            for (i, &xi) in (0..x.size).zip(&x.val) {
                if !is_exact_zero(xi) {
                    self.add(i, alpha * xi);
                }
            }
        }
    }

    /// Gather: writes the nonzero entries as a packed vector into `idx` and
    /// `val` (both cleared first), in pattern order in sparse mode and in
    /// increasing index order in dense mode.
    pub fn gather(&self, idx: &mut Vec<u32>, val: &mut Vec<T>) {
        idx.clear();
        val.clear();
        let mut push = |i: u32| {
            let v = self.val[i as usize];
            if !is_exact_zero(v) {
                idx.push(i);
                val.push(v);
            }
        };
        if self.sparse {
            self.idx.iter().copied().for_each(&mut push);
        } else {
            (0..self.size).for_each(push);
        }
    }

    /// Inner product `yᵀ z` with a dense vector `z`.
    ///
    /// # Panics
    /// If `z.len() != size`.
    pub fn dot(&self, z: &[T]) -> T {
        assert_eq!(z.len(), self.size(), "dot: size mismatch");
        if self.sparse {
            self.idx
                .iter()
                .map(|&i| self.val[i as usize] * z[i as usize])
                .sum()
        } else {
            self.val.iter().zip(z).map(|(&a, &b)| a * b).sum()
        }
    }

    /// Drops negligible entries: every `y_i` with `|y_i| ≤ drop_tol` is set
    /// to exactly zero and, in sparse mode, removed from the pattern (the
    /// solver passes `Tolerances::zero`).
    pub fn tidy(&mut self, drop_tol: T) {
        let Self {
            idx,
            val,
            mark,
            sparse,
            ..
        } = self;
        if *sparse {
            idx.retain(|&i| {
                let iu = i as usize;
                if is_zero(val[iu], drop_tol) {
                    val[iu] = T::ZERO;
                    mark[iu] = false;
                    false
                } else {
                    true
                }
            });
        } else {
            for v in val.iter_mut() {
                if is_zero(*v, drop_tol) {
                    *v = T::ZERO;
                }
            }
        }
    }

    /// The entries `(i, y_i)` of the pattern in sparse mode (in insertion
    /// order, explicit zeros included), of every position in dense mode.
    pub fn iter(&self) -> impl Iterator<Item = (u32, T)> + '_ {
        let all = if self.sparse { 0 } else { self.size };
        self.idx
            .iter()
            .copied()
            .chain(0..all)
            .map(move |i| (i, self.val[i as usize]))
    }

    /// Switches to dense mode, discarding the pattern (`O(count)`).
    pub fn make_dense(&mut self) {
        self.unmark_pattern();
        self.sparse = false;
    }

    /// Switches to sparse mode with the pattern set to the exact support of
    /// `y`, in increasing order (`O(size)`).
    pub fn rebuild_pattern(&mut self) {
        self.unmark_pattern();
        for (i, &v) in (0..self.size).zip(&self.val) {
            if !is_exact_zero(v) {
                self.mark[i as usize] = true;
                self.idx.push(i);
            }
        }
        self.sparse = true;
    }

    /// Empties the pattern and clears its marks, leaving values untouched.
    fn unmark_pattern(&mut self) {
        for &i in &self.idx {
            self.mark[i as usize] = false;
        }
        self.idx.clear();
    }

    /// Checks invariants (V1)–(V3).
    pub fn check_invariants(&self) -> Result<(), SparseError> {
        let n = self.size();
        if self.val.len() != n {
            return Err(SparseError::Dimension {
                what: "SparseVec values",
                expected: n,
                found: self.val.len(),
            });
        }
        if self.mark.len() != n {
            return Err(SparseError::Dimension {
                what: "SparseVec marks",
                expected: n,
                found: self.mark.len(),
            });
        }
        let marked = self.mark.iter().filter(|&&b| b).count();
        if !self.sparse {
            if !self.idx.is_empty() || marked != 0 {
                return Err(SparseError::Malformed(
                    "dense SparseVec keeps a pattern".into(),
                ));
            }
            return Ok(());
        }
        for &i in &self.idx {
            if i as usize >= n {
                return Err(SparseError::IndexOutOfRange {
                    what: "SparseVec pattern entry",
                    index: i as usize,
                    bound: n,
                });
            }
            if !self.mark[i as usize] {
                return Err(SparseError::Malformed(format!(
                    "pattern entry {i} is not marked"
                )));
            }
        }
        if marked != self.idx.len() {
            return Err(SparseError::Malformed(format!(
                "{marked} marks for {} pattern entries (duplicates or stale marks)",
                self.idx.len()
            )));
        }
        if let Some(i) = (0..n).find(|&i| !self.mark[i] && !is_exact_zero(self.val[i])) {
            return Err(SparseError::Malformed(format!(
                "nonzero entry {i} missing from the pattern"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::{approx_eq, le};
    use proptest::prelude::*;

    /// The drop tolerance of the tests (`Tolerances::zero` of the solver).
    const DROP: f64 = 1e-12;

    #[test]
    fn ex_1_2_cancellation_keeps_pattern_unique() {
        let mut y = SparseVec::new(5);
        y.add(3, 2.0);
        y.add(3, -2.0); // cancels to an exact zero, stays in the pattern
        y.add(3, 1.0);
        assert_eq!(y.pattern(), Some(&[3_u32][..]));
        assert!(y.check_invariants().is_ok());
        y.tidy(DROP);
        assert_eq!(y.count(), 1);
        y.add(1, 1e-15);
        y.tidy(DROP);
        assert_eq!(y.pattern(), Some(&[3_u32][..]));
        let (mut i, mut v) = (Vec::new(), Vec::new());
        y.gather(&mut i, &mut v);
        assert_eq!((i, v), (vec![3], vec![1.0]));
        y.clear();
        assert_eq!(y.count(), 0);
        assert!(y.values().iter().all(|&v| is_exact_zero(v)));
        assert!(y.check_invariants().is_ok());
    }

    #[test]
    fn modes_and_density() {
        let mut y = SparseVec::from_dense(&[0.0, 2.0, 0.0, 4.0]);
        assert_eq!(y.pattern(), Some(&[1_u32, 3][..]));
        assert!(approx_eq(y.density(), 0.5, 0.0));
        assert!(approx_eq(y.dot(&[1.0, 1.0, 1.0, 1.0]), 6.0, 0.0));
        y.values_mut()[0] = 1.0;
        assert!(!y.is_sparse());
        assert_eq!(y.count(), 4);
        assert!(y.check_invariants().is_ok());
        y.rebuild_pattern();
        assert_eq!(y.pattern(), Some(&[0_u32, 1, 3][..]));
        assert!(approx_eq(SparseVec::<f64>::new(0).density(), 0.0, 0.0));
        let it: Vec<(u32, f64)> = y.iter().collect();
        assert_eq!(it, vec![(0, 1.0), (1, 2.0), (3, 4.0)]);
        y.make_dense();
        assert_eq!(y.iter().count(), 4);
    }

    /// One operation of the equivalence test.
    #[derive(Clone, Debug)]
    enum Op {
        Set(u32, f64),
        Add(u32, f64),
        Scatter(f64, Vec<(u32, f64)>),
        /// `y ← y + α x` with `x` given densely; the flag says whether `x`
        /// is presented in sparse mode.
        Axpy(f64, Vec<f64>, bool),
        Tidy,
        Clear,
    }

    /// A value that is often zero, sometimes tiny (exercises `tidy`).
    fn value() -> impl Strategy<Value = f64> {
        prop_oneof![
            3 => Just(0.0),
            1 => Just(1e-14),
            6 => -100.0..100.0_f64,
        ]
    }

    fn ops() -> impl Strategy<Value = (usize, Vec<Op>)> {
        (1_usize..30).prop_flat_map(|n| {
            let n32 = u32::try_from(n).unwrap();
            let op = prop_oneof![
                2 => (0..n32, value()).prop_map(|(i, v)| Op::Set(i, v)),
                2 => (0..n32, value()).prop_map(|(i, v)| Op::Add(i, v)),
                2 => (value(), proptest::collection::vec((0..n32, value()), 0..8))
                    .prop_map(|(a, p)| Op::Scatter(a, p)),
                3 => (value(), proptest::collection::vec(value(), n), any::<bool>())
                    .prop_map(|(a, x, s)| Op::Axpy(a, x, s)),
                1 => Just(Op::Tidy),
                1 => Just(Op::Clear),
            ];
            (Just(n), proptest::collection::vec(op, 0..25))
        })
    }

    fn apply(y: &mut SparseVec, op: &Op, drop: f64) {
        match op {
            Op::Set(i, v) => y.set(*i, *v),
            Op::Add(i, v) => y.add(*i, *v),
            Op::Scatter(a, p) => {
                let (i, v): (Vec<u32>, Vec<f64>) = p.iter().copied().unzip();
                y.scatter(*a, &i, &v);
            }
            Op::Axpy(a, x, sparse) => {
                let mut xv = SparseVec::from_dense(x);
                if !sparse {
                    xv.make_dense();
                }
                y.axpy(*a, &xv);
            }
            Op::Tidy => y.tidy(drop),
            Op::Clear => y.clear(),
        }
    }

    /// Plain dense oracle for the same operation.
    fn apply_oracle(y: &mut [f64], op: &Op, drop: f64) {
        match op {
            Op::Set(i, v) => y[*i as usize] = *v,
            Op::Add(i, v) => y[*i as usize] += *v,
            Op::Scatter(a, p) => {
                for &(i, v) in p {
                    y[i as usize] += a * v;
                }
            }
            Op::Axpy(a, x, _) => {
                for (yi, &xi) in y.iter_mut().zip(x) {
                    if !is_exact_zero(xi) {
                        *yi += a * xi;
                    }
                }
            }
            Op::Tidy => {
                for v in y.iter_mut() {
                    if is_zero(*v, drop) {
                        *v = 0.0;
                    }
                }
            }
            Op::Clear => y.fill(0.0),
        }
    }

    fn gathered(y: &SparseVec) -> Vec<(u32, u64)> {
        let (mut i, mut v) = (Vec::new(), Vec::new());
        y.gather(&mut i, &mut v);
        let mut g: Vec<(u32, u64)> = i.into_iter().zip(v.iter().map(|x| x.to_bits())).collect();
        g.sort_unstable();
        g
    }

    /// Bitwise equality up to the sign of zero.
    fn same(a: &[f64], b: &[f64]) -> bool {
        a.iter().zip(b).all(|(&x, &y)| approx_eq(x, y, 0.0))
    }

    proptest! {
        /// A random operation sequence gives the same values on a vector kept
        /// in sparse mode, on one forced to dense mode after every operation,
        /// and on a plain `Vec<f64>`; the gathered nonzeros coincide and the
        /// invariants (V1)–(V3) hold throughout.
        #[test]
        fn sparse_equals_dense((n, ops) in ops()) {
            let tol = DROP;
            let mut ys = SparseVec::new(n);
            let mut yd = SparseVec::new(n);
            yd.make_dense();
            let mut oracle = vec![0.0; n];
            for op in &ops {
                apply(&mut ys, op, tol);
                apply(&mut yd, op, tol);
                yd.make_dense();
                apply_oracle(&mut oracle, op, tol);
                prop_assert!(ys.is_sparse());
                prop_assert!(ys.check_invariants().is_ok(), "{:?}", ys.check_invariants());
                prop_assert!(yd.check_invariants().is_ok(), "{:?}", yd.check_invariants());
                prop_assert!(same(ys.values(), yd.values()));
                prop_assert!(same(ys.values(), &oracle));
                prop_assert_eq!(gathered(&ys), gathered(&yd));
                // The two `dot`s sum bitwise-identical values in different
                // orders (pattern order vs. index order), so they agree only
                // up to reordering error, bounded by n·ε·Σ|y_i|; the final
                // dot may cancel to a small value, so the tolerance must
                // scale with the ℓ1 mass, not with the result.
                let ones = vec![1.0; n];
                let l1: f64 = yd.values().iter().map(|v| v.abs()).sum();
                prop_assert!(le((ys.dot(&ones) - yd.dot(&ones)).abs(), 1e-12 * l1.max(1.0), 0.0));
            }
            // Rebuilding the pattern of the dense copy yields the exact support.
            yd.rebuild_pattern();
            prop_assert!(yd.check_invariants().is_ok());
            prop_assert_eq!(yd.count(), gathered(&ys).len());
        }
    }
}
