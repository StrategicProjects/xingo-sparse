//! # Chapter 5.9 — Permutations and sparsity patterns
//!
//! *New with task L1.3 (PLAN.md trilha L).*
//!
//! ## Purpose
//! A fill-reducing ordering (Chapter 5.10) looks only at *where* the
//! nonzeros of a matrix are, never at their values, and answers with a
//! permutation. This chapter fixes both objects: [`Pattern`], the
//! structure of a compressed matrix without its values, and [`Perm`], a
//! permutation stored together with its inverse. It also measures what an
//! ordering is for — the number of nonzeros of the Cholesky factor of the
//! permuted pattern (Algorithm E) — which is the yardstick of every test
//! and of the acceptance tables of the orderings.
//!
//! ## Notation
//! A [`Pattern`] `S` with `n_rows = m` and `n = n_cols` columns stores, as
//! a [`CscMatrix`] does, the row indices of column `j` in
//! `row_idx[col_ptr[j]..col_ptr[j + 1]]`. A [`Perm`] `π` of `0..n` is the
//! pair of arrays
//!
//! * `fwd[k] = i` — the *old* index `i` that comes `k`-th in the new order
//!   (the `k`-th pivot of an elimination, the `k`-th column of `A Q`);
//! * `inv[i] = k` — the new position of the old index `i`.
//!
//! As matrices: `P` with `(P x)_k = x_{fwd[k]}`, so `P A Pᵀ` has entry
//! `(k, l) = a_{fwd[k], fwd[l]}`.
//!
//! For a square pattern `S` let `G(S)` be the undirected graph on `0..n`
//! with an edge `{i, j}` (`i ≠ j`) whenever `s_ij` or `s_ji` is stored: the
//! graph of `S + Sᵀ` without its diagonal. `L` is the Cholesky factor of a
//! symmetric positive definite matrix with the pattern of `P (S + Sᵀ) Pᵀ`
//! (and a full diagonal), `nnz(L)` its number of entries *including* the
//! diagonal — the *fill* of the ordering in the tables (no numerical
//! cancellation is assumed: the structural count).
//!
//! ## Algorithm S (Symmetrization `S + Sᵀ`)
//! S1. Compute `Sᵀ` by Algorithm T of Chapter 5.8 (indices only).
//! S2. For each column `j`, merge the sorted lists of column `j` of `S` and
//!     of `Sᵀ`, keeping each index once. The result has sorted columns.
//!
//! ## Algorithm E (Elimination tree and Cholesky counts, by row subtrees)
//! Given the symmetric graph `G` and the permutation `π`; let `C = P G Pᵀ`.
//! Row `k` of `L` is the set of `l ≤ k` reached from the entries `c_kl`,
//! `l < k`, by walking up the elimination tree of the leading `k × k`
//! block (the *row subtree*, Liu 1986).
//!
//! E1. Set `parent[k] ← ∅`, `mark[k] ← ∅` for all `k`, `count ← n` (the
//!     diagonal).
//! E2. For `k = 0, 1, …, n − 1`: set `mark[k] ← k`; for every neighbour `i`
//!     of `fwd[k]` with `l = inv[i] < k`, walk `x ← l, parent[x], …` while
//!     `mark[x] ≠ k`: set `mark[x] ← k`, `count ← count + 1` (the entry
//!     `l_kx`), and if `parent[x] = ∅` set `parent[x] ← k` and stop.
//! E3. Return `count`.
//!
//! The walk of E2 from `l` stops at the first node already in row `k`'s
//! subtree, so every node of the row subtree is visited once: the cost is
//! `O(nnz(L) + nnz(S))`, no `log` and no sort. Setting `parent` the first
//! time a root is reached is Liu's elimination-tree algorithm without path
//! compression.
//!
//! ## Algorithm G (The pattern of `SᵀS`)
//! For COLAMD (Chapter 5.10) the relevant symmetric graph is that of `SᵀS`,
//! whose Cholesky factor bounds the `U` of an LU with partial pivoting and
//! is the `R` of a QR factorization (George & Ng 1987). Column `j` of `SᵀS`
//! is the union, over the rows `i` of column `j` of `S`, of the columns of
//! row `i` (rows read from `Sᵀ`): mark-and-collect, then sort each column.
//! Cost `O(Σ_i r_i²)` with `r_i` the row counts — which is why COLAMD itself
//! never forms it.
//!
//! ## Invariants
//! (P1) (= the contract's (S3)) A [`Perm`] of length `n` has `fwd` and `inv`
//!      of length `n`, entries `< n`, and `inv[fwd[k]] = k` for every `k`
//!      (so both are bijections). [`Perm::check_invariants`] checks it and
//!      every constructor asserts it in debug builds.
//! (P2) A [`Pattern`] satisfies (C1), (C2) and (C4) of Chapter 5.8 (there
//!      are no values, so (C3) is void): pointers non-decreasing from 0 to
//!      `nnz`, row indices strictly increasing inside a column and smaller
//!      than `n_rows`. Checked by [`Pattern::check_invariants`].
//! (P3) The outputs of Algorithms S and G are symmetric patterns that
//!      satisfy (P2).
//!
//! ## References
//! Liu, "A compact row storage scheme for Cholesky factors using
//! elimination trees", ACM TOMS 12(2), 1986, and "The role of elimination
//! trees in sparse factorization", SIAM J. Matrix Anal. Appl. 11(1), 1990;
//! George & Ng, "Symbolic factorization for sparse Gaussian elimination
//! with partial pivoting", SIAM J. Sci. Stat. Comput. 8(6), 1987; Davis,
//! *Direct Methods for Sparse Linear Systems* (2006), §4.1 and §4.4 (read,
//! not copied); the contract of the crate (`Pattern`, `Perm`, (S3)).

use crate::index::dim_to_u32;
use crate::real::Real;
use crate::{CscMatrix, SparseError};

/// A permutation of `0..n` with its inverse (invariant (P1), the
/// contract's (S3)).
///
/// `fwd[k]` is the old index placed at new position `k`; `inv[i]` is the
/// new position of the old index `i`. The fields are public, as in the
/// contract; a value built by hand should be checked with
/// [`Perm::check_invariants`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Perm {
    /// `fwd[k]`: the old index at new position `k`.
    pub fwd: Vec<u32>,
    /// `inv[i]`: the new position of the old index `i`.
    pub inv: Vec<u32>,
}

impl Perm {
    /// The identity permutation of `0..n`.
    ///
    /// # Panics
    /// If `n` exceeds `u32::MAX`.
    pub fn identity(n: usize) -> Self {
        assert!(dim_to_u32("permutation", n).is_ok(), "n exceeds u32::MAX");
        let fwd: Vec<u32> = (0..n).map(idx32).collect();
        Self {
            inv: fwd.clone(),
            fwd,
        }
    }

    /// The permutation whose forward array is `fwd`, with its inverse;
    /// an error unless `fwd` is a bijection of `0..fwd.len()`.
    pub fn from_fwd(fwd: Vec<u32>) -> Result<Self, SparseError> {
        let n = fwd.len();
        dim_to_u32("permutation", n)?;
        let mut inv = vec![u32::MAX; n];
        for (k, &i) in fwd.iter().enumerate() {
            let i = i as usize;
            if i >= n {
                return Err(SparseError::IndexOutOfRange {
                    what: "permutation entry",
                    index: i,
                    bound: n,
                });
            }
            if inv[i] != u32::MAX {
                return Err(SparseError::Malformed(format!(
                    "(P1): index {i} appears twice in the permutation"
                )));
            }
            inv[i] = idx32(k);
        }
        let p = Self { fwd, inv };
        debug_assert_eq!(p.check_invariants(), Ok(()));
        Ok(p)
    }

    /// The length `n`.
    #[inline]
    pub fn len(&self) -> usize {
        self.fwd.len()
    }

    /// `true` iff `n = 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.fwd.is_empty()
    }

    /// The inverse permutation (`fwd` and `inv` exchanged).
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            fwd: self.inv.clone(),
            inv: self.fwd.clone(),
        }
    }

    /// `true` iff this is the identity.
    pub fn is_identity(&self) -> bool {
        self.fwd.iter().enumerate().all(|(k, &i)| i as usize == k)
    }

    /// Checks invariant (P1).
    pub fn check_invariants(&self) -> Result<(), SparseError> {
        let n = self.fwd.len();
        if self.inv.len() != n {
            return Err(SparseError::Dimension {
                what: "inverse permutation",
                expected: n,
                found: self.inv.len(),
            });
        }
        for (k, &i) in self.fwd.iter().enumerate() {
            let i = i as usize;
            if i >= n {
                return Err(SparseError::IndexOutOfRange {
                    what: "permutation entry",
                    index: i,
                    bound: n,
                });
            }
            if self.inv[i] as usize != k {
                return Err(SparseError::Malformed(format!(
                    "(P1): inv[fwd[{k}]] = {} ≠ {k}",
                    self.inv[i]
                )));
            }
        }
        Ok(())
    }
}

/// The structure of a compressed sparse column matrix without its values
/// (invariant (P2)): what the orderings of Chapter 5.10 consume.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pattern {
    /// Number of rows `m`.
    pub n_rows: usize,
    /// Column pointers, length `n + 1`.
    pub col_ptr: Vec<usize>,
    /// Row indices, strictly increasing inside each column.
    pub row_idx: Vec<u32>,
}

impl Pattern {
    /// The pattern with the given arrays, validated ((P2)).
    pub fn from_raw_parts(
        n_rows: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<u32>,
    ) -> Result<Self, SparseError> {
        let p = Self {
            n_rows,
            col_ptr,
            row_idx,
        };
        p.check_invariants()?;
        Ok(p)
    }

    /// The empty `m × n` pattern.
    ///
    /// # Panics
    /// If `m` or `n` exceeds `u32::MAX`.
    pub fn empty(n_rows: usize, n_cols: usize) -> Self {
        assert!(dim_to_u32("rows", n_rows).is_ok() && dim_to_u32("columns", n_cols).is_ok());
        Self {
            n_rows,
            col_ptr: vec![0; n_cols + 1],
            row_idx: Vec::new(),
        }
    }

    /// Number of rows `m`.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns `n`.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.col_ptr.len().saturating_sub(1)
    }

    /// Number of stored entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }

    /// The row indices of column `j`.
    #[inline]
    pub fn col(&self, j: usize) -> &[u32] {
        &self.row_idx[self.col_ptr[j]..self.col_ptr[j + 1]]
    }

    /// `Sᵀ` (Algorithm T of Chapter 5.8, indices only).
    #[must_use]
    pub fn transpose(&self) -> Self {
        // T1–T2. Counts and prefix sums.
        let mut ptr = vec![0_usize; self.n_rows + 1];
        for &i in &self.row_idx {
            ptr[i as usize + 1] += 1;
        }
        for i in 0..self.n_rows {
            ptr[i + 1] += ptr[i];
        }
        // T3. Scatter by increasing column (sorted output).
        let mut next = ptr[..self.n_rows].to_vec();
        let mut idx = vec![0_u32; self.nnz()];
        for j in 0..self.n_cols() {
            for &i in self.col(j) {
                idx[next[i as usize]] = idx32(j);
                next[i as usize] += 1;
            }
        }
        let t = Self {
            n_rows: self.n_cols(),
            col_ptr: ptr,
            row_idx: idx,
        };
        debug_assert_eq!(t.check_invariants(), Ok(()));
        t
    }

    /// `S + Sᵀ` (Algorithm S); the diagonal entries of `S` are kept, none
    /// is added. An error unless `S` is square.
    pub fn symmetric(&self) -> Result<Self, SparseError> {
        let n = self.n_cols();
        if self.n_rows != n {
            return Err(SparseError::Dimension {
                what: "pattern (must be square to symmetrize)",
                expected: n,
                found: self.n_rows,
            });
        }
        let t = self.transpose();
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::with_capacity(2 * self.nnz());
        col_ptr.push(0);
        for j in 0..n {
            // S2. Merge two sorted lists.
            let (a, b) = (self.col(j), t.col(j));
            let (mut p, mut q) = (0, 0);
            while p < a.len() || q < b.len() {
                let x = a.get(p).copied().unwrap_or(u32::MAX);
                let y = b.get(q).copied().unwrap_or(u32::MAX);
                let next = x.min(y);
                p += usize::from(x == next);
                q += usize::from(y == next);
                row_idx.push(next);
            }
            col_ptr.push(row_idx.len());
        }
        let s = Self {
            n_rows: n,
            col_ptr,
            row_idx,
        };
        debug_assert_eq!(s.check_invariants(), Ok(()));
        Ok(s)
    }

    /// The pattern of `SᵀS` (Algorithm G), diagonal included: an `n × n`
    /// symmetric pattern.
    #[must_use]
    pub fn ata(&self) -> Self {
        let n = self.n_cols();
        let t = self.transpose();
        let mut mark = vec![usize::MAX; n];
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
        for j in 0..n {
            let start = row_idx.len();
            for &i in self.col(j) {
                for &k in t.col(i as usize) {
                    if mark[k as usize] != j {
                        mark[k as usize] = j;
                        row_idx.push(k);
                    }
                }
            }
            row_idx[start..].sort_unstable();
            col_ptr.push(row_idx.len());
        }
        let s = Self {
            n_rows: n,
            col_ptr,
            row_idx,
        };
        debug_assert_eq!(s.check_invariants(), Ok(()));
        s
    }

    /// `nnz(L)`, diagonal included, of the Cholesky factor of
    /// `P (S + Sᵀ) Pᵀ` for the permutation `perm` (Algorithm E): the fill
    /// an ordering yields. An error unless `S` is square with `perm` of the
    /// same order.
    pub fn cholesky_nnz(&self, perm: &Perm) -> Result<usize, SparseError> {
        let n = self.n_cols();
        if self.n_rows != n || perm.len() != n {
            return Err(SparseError::Dimension {
                what: "cholesky_nnz (square pattern and permutation of its order)",
                expected: n,
                found: if self.n_rows == n {
                    perm.len()
                } else {
                    self.n_rows
                },
            });
        }
        perm.check_invariants()?;
        let g = self.symmetric()?;
        // E1.
        let none = usize::MAX;
        let mut parent = vec![none; n];
        let mut mark = vec![none; n];
        let mut count = n;
        // E2.
        for k in 0..n {
            mark[k] = k;
            for &i in g.col(perm.fwd[k] as usize) {
                let mut x = perm.inv[i as usize] as usize;
                if x >= k {
                    continue;
                }
                while mark[x] != k {
                    mark[x] = k;
                    count += 1;
                    if parent[x] == none {
                        parent[x] = k;
                        break;
                    }
                    x = parent[x];
                }
            }
        }
        // E3.
        Ok(count)
    }

    /// Checks invariant (P2).
    pub fn check_invariants(&self) -> Result<(), SparseError> {
        dim_to_u32("rows", self.n_rows)?;
        dim_to_u32("columns", self.n_cols())?;
        if self.col_ptr.is_empty() {
            return Err(SparseError::Malformed("(P2): empty pointer array".into()));
        }
        if self.col_ptr[0] != 0 || self.col_ptr[self.n_cols()] != self.row_idx.len() {
            return Err(SparseError::Malformed(format!(
                "(P2): pointers must run from 0 to nnz = {}",
                self.row_idx.len()
            )));
        }
        for j in 0..self.n_cols() {
            if self.col_ptr[j] > self.col_ptr[j + 1] {
                return Err(SparseError::Malformed(format!(
                    "(P2): pointer of column {j} decreases"
                )));
            }
            let col = self.col(j);
            for (p, &i) in col.iter().enumerate() {
                if i as usize >= self.n_rows {
                    return Err(SparseError::IndexOutOfRange {
                        what: "pattern row index",
                        index: i as usize,
                        bound: self.n_rows,
                    });
                }
                if p > 0 && col[p - 1] >= i {
                    return Err(SparseError::Malformed(format!(
                        "(P2): rows of column {j} not strictly increasing"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl<T: Real> CscMatrix<T> {
    /// The structure without values, consumed by the orderings (Chapter
    /// 5.9).
    pub fn pattern(&self) -> Pattern {
        Pattern {
            n_rows: self.n_rows,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
        }
    }

    /// The pattern of `A + Aᵀ` (Algorithm S) for the symmetric orderings;
    /// an error unless `A` is square.
    pub fn symmetric_pattern(&self) -> Result<Pattern, SparseError> {
        self.pattern().symmetric()
    }
}

/// Converts an index below `u32::MAX` to `u32`.
#[inline]
pub(crate) fn idx32(i: usize) -> u32 {
    u32::try_from(i).expect("index exceeds u32::MAX")
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// A random `m × n` pattern from a list of `(i, j)` (duplicates
    /// collapse).
    fn pattern_of(m: usize, n: usize, entries: &[(usize, usize)]) -> Pattern {
        let mut cols: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(i, j) in entries {
            cols[j % n].push(idx32(i % m));
        }
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        for mut c in cols {
            c.sort_unstable();
            c.dedup();
            row_idx.extend(c);
            col_ptr.push(row_idx.len());
        }
        Pattern::from_raw_parts(m, col_ptr, row_idx).unwrap()
    }

    /// Brute force: `nnz(L)` by eliminating the dense graph of `P G Pᵀ`.
    fn cholesky_nnz_dense(s: &Pattern, perm: &Perm) -> usize {
        let n = s.n_cols();
        let mut adj = vec![vec![false; n]; n];
        for j in 0..n {
            for &i in s.col(j) {
                let (a, b) = (perm.inv[i as usize] as usize, perm.inv[j] as usize);
                if a != b {
                    adj[a][b] = true;
                    adj[b][a] = true;
                }
            }
        }
        let mut count = n;
        for k in 0..n {
            let later: Vec<usize> = (k + 1..n).filter(|&l| adj[k][l]).collect();
            count += later.len();
            for &a in &later {
                for &b in &later {
                    if a != b {
                        adj[a][b] = true;
                    }
                }
            }
        }
        count
    }

    #[test]
    fn perm_constructors_and_errors() {
        let p = Perm::identity(4);
        assert!(p.is_identity() && p.check_invariants().is_ok() && p.len() == 4);
        let q = Perm::from_fwd(vec![2, 0, 3, 1]).unwrap();
        assert_eq!(q.inv, vec![1, 3, 0, 2]);
        assert_eq!(q.inverse().inverse(), q);
        assert!(!q.is_identity() && !q.is_empty());
        assert!(Perm::from_fwd(vec![0, 0]).is_err());
        assert!(Perm::from_fwd(vec![0, 2]).is_err());
        let bad = Perm {
            fwd: vec![0, 1],
            inv: vec![1, 0],
        };
        assert!(bad.check_invariants().is_err());
        assert!(Perm::identity(0).is_empty());
    }

    #[test]
    fn pattern_validation_and_access() {
        assert!(Pattern::from_raw_parts(2, vec![0, 2], vec![1, 0]).is_err());
        assert!(Pattern::from_raw_parts(2, vec![0, 1], vec![2]).is_err());
        assert!(Pattern::from_raw_parts(2, vec![0, 2, 1], vec![0, 1]).is_err());
        let p = Pattern::from_raw_parts(3, vec![0, 2, 2, 3], vec![0, 2, 1]).unwrap();
        assert_eq!((p.n_rows(), p.n_cols(), p.nnz()), (3, 3, 3));
        assert_eq!(p.col(0), &[0, 2]);
        assert!(p.col(1).is_empty());
        assert!(Pattern::empty(2, 3).check_invariants().is_ok());
        let a = CscMatrix::<f64>::from_raw_parts(
            3,
            vec![0, 2, 2, 3],
            vec![0, 2, 1],
            vec![1.0, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(a.pattern(), p);
        let s = a.symmetric_pattern().unwrap();
        assert_eq!(s.col(0), &[0, 2]);
        assert_eq!(s.col(1), &[2]);
        assert_eq!(s.col(2), &[0, 1]);
        assert!(Pattern::empty(2, 3).symmetric().is_err());
        assert!(p.cholesky_nnz(&Perm::identity(2)).is_err());
    }

    proptest! {
        /// (P2)/(P3): transposition is an involution, `S + Sᵀ` is symmetric
        /// and contains `S`, `SᵀS` is symmetric with its diagonal.
        #[test]
        fn pattern_algebra(m in 1_usize..12, n in 1_usize..12,
                           e in prop::collection::vec((0_usize..12, 0_usize..12), 0..60)) {
            let s = pattern_of(m, n, &e);
            prop_assert_eq!(s.transpose().transpose(), s.clone());
            let g = s.ata();
            prop_assert_eq!(g.transpose(), g.clone());
            for j in 0..n {
                if !s.col(j).is_empty() {
                    prop_assert!(g.col(j).contains(&idx32(j)));
                }
            }
            let q = pattern_of(n, n, &e);
            let sym = q.symmetric().unwrap();
            prop_assert_eq!(sym.transpose(), sym.clone());
            for j in 0..n {
                for &i in q.col(j) {
                    prop_assert!(sym.col(j).contains(&i));
                }
            }
        }

        /// Algorithm E against the dense elimination, under random
        /// permutations.
        #[test]
        fn cholesky_count_matches_dense_elimination(
            n in 1_usize..25,
            e in prop::collection::vec((0_usize..25, 0_usize..25), 0..80),
            keys in prop::collection::vec(any::<u32>(), 25),
        ) {
            let s = pattern_of(n, n, &e);
            let mut fwd: Vec<u32> = (0..n).map(idx32).collect();
            fwd.sort_by_key(|&i| keys[i as usize]);
            let perm = Perm::from_fwd(fwd).unwrap();
            prop_assert_eq!(s.cholesky_nnz(&perm).unwrap(), cholesky_nnz_dense(&s, &perm));
        }
    }
}
