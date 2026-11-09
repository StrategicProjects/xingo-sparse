//! # Chapter 5.10 — Fill-reducing orderings: [`Natural`], [`Amd`], [`Colamd`]
//!
//! *New with task L1.3 (PLAN.md trilha L; D-0080, decision D10: the
//! orderings are our own from v0.1, no C, no GPL/LGPL).*
//!
//! ## Purpose
//! The cost of a sparse factorization is decided before the first flop, by
//! the order of elimination. An [`Ordering`] looks at a [`Pattern`]
//! (Chapter 5.9) and returns a [`Perm`]; it never sees values. Two targets
//! ([`OrderingTarget`]):
//!
//! * [`OrderingTarget::Symmetric`] — a symmetric order `P (S + Sᵀ) Pᵀ` for
//!   Cholesky or `LDLᵀ` (the KKT systems of the interior-point method,
//!   L1.5): the permutation of the rows *and* columns;
//! * [`OrderingTarget::Columns`] — a column order `S Q` for an LU with row
//!   pivoting (or a QR): it reduces the fill of the Cholesky factor of
//!   `QᵀSᵀSQ`, which bounds the `U` of every row-pivoted LU of `S Q`
//!   (George & Ng 1987), without forming `SᵀS`.
//!
//! Three orderings, stateless policies:
//!
//! * [`Natural`] — the identity (the reference of every table);
//! * [`Amd`] — approximate minimum degree on the graph of `S + Sᵀ`
//!   (Amestoy, Davis & Duff 1996), with aggressive absorption, mass
//!   elimination, supervariables and dense-row detection; for the
//!   `Columns` target it orders the graph of `SᵀS`, formed explicitly;
//! * [`Colamd`] — column approximate minimum degree (Davis, Gilbert,
//!   Larimore & Ng 2004): the same process on the quotient graph of `SᵀS`
//!   started from the rows of `S`, never forming `SᵀS`; for the `Symmetric`
//!   target, the SYMAMD construction (every off-diagonal pair `{i, j}` of
//!   `S + Sᵀ` is a row with two entries, whose `MᵀM` has the pattern of
//!   `S + Sᵀ`).
//!
//! Both run Algorithm Q of Chapter 5.11; this chapter is their setup.
//!
//! ## Notation
//! `S` is `m × n`; `r_i` and `c_j` are its row and column counts.
//! `κ(α, N) = max(16, ⌊α √N⌋)` is the *dense threshold* (SuiteSparse's
//! rule); a negative `α` disables dense detection. Excluded (dense or
//! empty) indices are ordered last, in increasing index order.
//!
//! ## Algorithm A (AMD)
//! A1. [Graph.] `G ← S + Sᵀ` without its diagonal (Algorithm S of Chapter
//!     5.9; for `Columns`, the graph of `SᵀS` by Algorithm G). An error
//!     unless `S` is square for `Symmetric`.
//! A2. [Dense rows.] With `δ = κ(dense, n)` (`dense` default 10), every
//!     node of degree `> δ` is excluded: ordering it in the middle would
//!     cost as much as it fills, and its presence would spoil every
//!     degree it touches.
//! A3. [Order.] Algorithm Q with `A_i = G`'s adjacency, no initial
//!     elements, the bound `min(d̄_i, δ_i)`, AMD's conventions (largest
//!     index first among equal degrees, the postorder Q8 as output);
//!     aggressive absorption as asked (default on).
//! A4. Append the excluded nodes. Return the [`Perm`] (`fwd` = pivot order).
//!
//! ## Algorithm C (COLAMD)
//! C1. [Dense columns.] With `κ_c = κ(dense_col, min(m, n))` (default
//!     `dense_col = 10`), exclude every column with `c_j > κ_c`.
//! C2. [Dense rows.] With `κ_r = κ(dense_row, n)` (default 10), drop every
//!     row with more than `κ_r` entries in non-excluded columns: a dense row
//!     makes `SᵀS` dense and says nothing useful about the order.
//! C3. [Null columns.] Exclude every column left without a row (they are
//!     ordered last, before the dense columns, as COLAMD does).
//! C4. [Order.] Algorithm Q with no `A_i`, the surviving rows as initial
//!     elements, COLAMD's score `δ_i` (no `min` with the previous score) and
//!     COLAMD's conventions (smallest index first among ties, initial rows
//!     before pivot rows in Q3, the pivot order Q8′ as output); aggressive
//!     absorption on.
//! C5. Append the null columns, then the dense ones.
//!
//! For `Symmetric` (SYMAMD): exclude the nodes of `S + Sᵀ` of degree
//! `> κ(dense_col, n)`; every edge `{i, j}` (`i < j`) between the others is
//! an initial element `{i, j}`; then C4–C5 with the excluded nodes.
//!
//! ## Invariants
//! (O1) Every ordering returns a [`Perm`] of `n` (the number of columns)
//!      satisfying (P1) of Chapter 5.9, and fails only on an input error
//!      (a non-square pattern for `Symmetric`, (P2) violated).
//! (O2) [`Natural`] is the identity; the others are deterministic
//!      functions of the pattern.
//!
//! ## Defaults and their source
//! `Amd { dense: None (= 10), aggressive: true }` and `Colamd { dense_row:
//! None (= 10), dense_col: None (= 10) }` are SuiteSparse's `AMD_DENSE`,
//! `AMD_AGGRESSIVE`, `COLAMD_DENSE_ROW` and `COLAMD_DENSE_COL` defaults
//! (AMD 2.x and COLAMD 2.x user guides).
//!
//! ## References
//! Amestoy, Davis & Duff, SIAM J. Matrix Anal. Appl. 17(4), 1996, and
//! Algorithm 837 (AMD), ACM TOMS 30(3), 2004; Davis, Gilbert, Larimore &
//! Ng, "A column approximate minimum degree ordering algorithm" and
//! Algorithm 836 (COLAMD), ACM TOMS 30(3), 2004; George & Ng (1987); the
//! contract of the crate (`Ordering`, `OrderingTarget`, `Natural`, `Amd`,
//! `Colamd`).

use crate::factor::count_f64;
use crate::perm::idx32;
use crate::qgraph::{self, Input};
use crate::real::lt;
use crate::{Pattern, Perm, SparseError};

/// What an ordering is asked to reduce fill for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderingTarget {
    /// A symmetric order of a square pattern (Cholesky, `LDLᵀ`): rows and
    /// columns are permuted alike.
    Symmetric,
    /// A column order for an LU with row pivoting (or a QR): the pattern
    /// of `SᵀS`, without forming it (COLAMD).
    Columns,
}

/// Fill-reducing orderings. Implementations are stateless policies
/// (invariants (O1)–(O2)).
pub trait Ordering {
    /// A permutation of the columns of `pattern` (for `Symmetric`, of its
    /// rows and columns alike).
    fn order(&self, pattern: &Pattern, target: OrderingTarget) -> Result<Perm, SparseError>;
    /// A short name for logs and tables.
    fn name(&self) -> &'static str;
}

/// The identity ordering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Natural;

/// Approximate minimum degree (Amestoy, Davis & Duff 1996), Algorithm A:
/// quotient graph with element absorption, approximate external degrees,
/// mass elimination of indistinguishable nodes and dense-row detection.
/// Our own implementation (D10); SuiteSparse AMD is the oracle in tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Amd {
    /// Nodes of degree above `max(16, dense · √n)` are ordered last
    /// (`None`: AMD's default `10`; negative: no dense detection).
    pub dense: Option<f64>,
    /// Aggressive absorption of elements (AMD's default: on).
    pub aggressive: bool,
}

impl Default for Amd {
    fn default() -> Self {
        Self {
            dense: None,
            aggressive: true,
        }
    }
}

/// Column approximate minimum degree (Davis, Gilbert, Larimore & Ng 2004),
/// Algorithm C: the AMD idea applied to the pattern of `SᵀS` without
/// forming it, for an LU with row pivoting. Our own implementation (D10).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Colamd {
    /// Rows with more than `max(16, dense_row · √n)` entries are ignored
    /// (`None`: COLAMD's default `10`; negative: none is).
    pub dense_row: Option<f64>,
    /// Columns with more than `max(16, dense_col · √min(m, n))` entries
    /// are ordered last (`None`: COLAMD's default `10`; negative: none
    /// is).
    pub dense_col: Option<f64>,
}

/// The default dense knob of AMD and COLAMD.
const DENSE_DEFAULT: f64 = 10.0;

/// The least dense threshold (SuiteSparse's `16`).
const DENSE_MIN: usize = 16;

/// `κ(α, N) = max(16, ⌊α √N⌋)`; `usize::MAX` (never dense) for `α < 0`.
fn dense_threshold(alpha: Option<f64>, big_n: usize) -> usize {
    let a = alpha.unwrap_or(DENSE_DEFAULT);
    if lt(a, 0.0) {
        return usize::MAX;
    }
    floor_usize(a * count_f64(big_n).sqrt()).max(DENSE_MIN)
}

/// `⌊x⌋` for a finite `x ≥ 0` (saturating, as `as` is).
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // x ≥ 0, saturating
fn floor_usize(x: f64) -> usize {
    x as usize
}

/// A square check for the `Symmetric` target.
fn square(pattern: &Pattern) -> Result<usize, SparseError> {
    let n = pattern.n_cols();
    if pattern.n_rows == n {
        Ok(n)
    } else {
        Err(SparseError::Dimension {
            what: "pattern (a symmetric ordering needs a square one)",
            expected: n,
            found: pattern.n_rows,
        })
    }
}

/// The off-diagonal adjacency of a symmetric pattern.
fn adjacency(g: &Pattern) -> Vec<Vec<u32>> {
    (0..g.n_cols())
        .map(|j| {
            g.col(j)
                .iter()
                .copied()
                .filter(|&i| i as usize != j)
                .collect()
        })
        .collect()
}

/// The off-diagonal adjacency of `S + Sᵀ` for a square `S` (Algorithm S of
/// Chapter 5.9 straight into the lists the engine takes: one transpose
/// and one merge per column, no intermediate pattern).
fn sym_adjacency(s: &Pattern) -> Vec<Vec<u32>> {
    let t = s.transpose();
    (0..s.n_cols())
        .map(|j| {
            let (a, b) = (s.col(j), t.col(j));
            let jj = idx32(j);
            let mut out = Vec::with_capacity(a.len() + b.len());
            let (mut p, mut q) = (0, 0);
            while p < a.len() && q < b.len() {
                let (x, y) = (a[p], b[q]);
                let next = x.min(y);
                p += usize::from(x == next);
                q += usize::from(y == next);
                if next != jj {
                    out.push(next);
                }
            }
            out.extend(a[p..].iter().chain(&b[q..]).copied().filter(|&i| i != jj));
            out
        })
        .collect()
}

/// `fwd` = `ordered`, then the excluded indices of each group in order.
fn finish(n: usize, mut fwd: Vec<u32>, groups: &[&[bool]]) -> Perm {
    for g in groups {
        fwd.extend((0..n).filter(|&j| g[j]).map(idx32));
    }
    debug_assert_eq!(fwd.len(), n);
    let p = Perm::from_fwd(fwd).expect("(O1): the ordering is a permutation");
    debug_assert_eq!(p.check_invariants(), Ok(()));
    p
}

impl Ordering for Natural {
    fn order(&self, pattern: &Pattern, target: OrderingTarget) -> Result<Perm, SparseError> {
        pattern.check_invariants()?;
        let n = match target {
            OrderingTarget::Symmetric => square(pattern)?,
            OrderingTarget::Columns => pattern.n_cols(),
        };
        Ok(Perm::identity(n))
    }
    fn name(&self) -> &'static str {
        "natural"
    }
}

impl Ordering for Amd {
    fn order(&self, pattern: &Pattern, target: OrderingTarget) -> Result<Perm, SparseError> {
        pattern.check_invariants()?;
        // A1.
        let adj = match target {
            OrderingTarget::Symmetric => {
                square(pattern)?;
                sym_adjacency(pattern)
            }
            OrderingTarget::Columns => adjacency(&pattern.ata()),
        };
        let n = adj.len();
        // A2.
        let delta = dense_threshold(self.dense, n);
        let excluded: Vec<bool> = adj.iter().map(|a| a.len() > delta).collect();
        // A3.
        let fwd = qgraph::order(Input {
            n,
            adj,
            elems: Vec::new(),
            excluded: excluded.clone(),
            aggressive: self.aggressive,
            previous_degree: true,
            column_order: false,
        });
        // A4.
        Ok(finish(n, fwd, &[excluded.as_slice()]))
    }
    fn name(&self) -> &'static str {
        "amd"
    }
}

impl Ordering for Colamd {
    fn order(&self, pattern: &Pattern, target: OrderingTarget) -> Result<Perm, SparseError> {
        pattern.check_invariants()?;
        match target {
            OrderingTarget::Columns => Ok(self.columns(pattern)),
            OrderingTarget::Symmetric => {
                square(pattern)?;
                Ok(self.symmetric(&pattern.symmetric()?))
            }
        }
    }
    fn name(&self) -> &'static str {
        "colamd"
    }
}

impl Colamd {
    /// Algorithm C.
    fn columns(self, s: &Pattern) -> Perm {
        let (m, n) = (s.n_rows, s.n_cols());
        // C1.
        let kc = dense_threshold(self.dense_col, m.min(n));
        let dense: Vec<bool> = (0..n).map(|j| s.col(j).len() > kc).collect();
        // C2.
        let t = s.transpose();
        let kr = dense_threshold(self.dense_row, n);
        let mut elems = Vec::with_capacity(m);
        let mut has_row = vec![false; n];
        for i in 0..m {
            let row: Vec<u32> = t
                .col(i)
                .iter()
                .copied()
                .filter(|&j| !dense[j as usize])
                .collect();
            if row.is_empty() || row.len() > kr {
                continue;
            }
            for &j in &row {
                has_row[j as usize] = true;
            }
            elems.push(row);
        }
        // C3.
        let null: Vec<bool> = (0..n).map(|j| !dense[j] && !has_row[j]).collect();
        let excluded: Vec<bool> = (0..n).map(|j| dense[j] || null[j]).collect();
        // C4.
        let fwd = qgraph::order(Input {
            n,
            adj: vec![Vec::new(); n],
            elems,
            excluded,
            aggressive: true,
            previous_degree: false,
            column_order: true,
        });
        // C5.
        finish(n, fwd, &[null.as_slice(), dense.as_slice()])
    }

    /// SYMAMD: Algorithm C on the edge rows of the symmetric pattern `g`.
    fn symmetric(self, g: &Pattern) -> Perm {
        let n = g.n_cols();
        let adj = adjacency(g);
        let kc = dense_threshold(self.dense_col, n);
        let dense: Vec<bool> = adj.iter().map(|a| a.len() > kc).collect();
        let mut elems = Vec::new();
        for (j, a) in adj.iter().enumerate() {
            if dense[j] {
                continue;
            }
            for &i in a {
                if (i as usize) < j && !dense[i as usize] {
                    elems.push(vec![i, idx32(j)]);
                }
            }
        }
        let fwd = qgraph::order(Input {
            n,
            adj: vec![Vec::new(); n],
            elems,
            excluded: dense.clone(),
            aggressive: true,
            previous_degree: false,
            column_order: true,
        });
        finish(n, fwd, &[dense.as_slice()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// xorshift64*.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            usize::try_from(self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) % (n as u64)).unwrap()
        }
        fn shuffle(&mut self, v: &mut [u32]) {
            for i in (1..v.len()).rev() {
                let j = self.below(i + 1);
                v.swap(i, j);
            }
        }
    }

    /// A pattern from `(i, j)` pairs (duplicates collapse).
    fn pat(m: usize, n: usize, e: &[(usize, usize)]) -> Pattern {
        let mut cols: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(i, j) in e {
            cols[j].push(idx32(i));
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

    /// The symmetric pattern of an undirected graph given by its edges,
    /// relabelled by `label`.
    fn graph(n: usize, edges: &[(usize, usize)], label: &[u32]) -> Pattern {
        let mut e: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        for &(a, b) in edges {
            let (a, b) = (label[a] as usize, label[b] as usize);
            e.push((a, b));
            e.push((b, a));
        }
        pat(n, n, &e)
    }

    fn random_label(rng: &mut Rng, n: usize) -> Vec<u32> {
        let mut l: Vec<u32> = (0..n).map(idx32).collect();
        rng.shuffle(&mut l);
        l
    }

    /// A random tree on `n` nodes.
    fn tree_edges(rng: &mut Rng, n: usize) -> Vec<(usize, usize)> {
        (1..n).map(|i| (i, rng.below(i))).collect()
    }

    /// The `k × k` grid graph in its natural (row-major) numbering.
    fn grid_edges(k: usize) -> Vec<(usize, usize)> {
        let mut e = Vec::new();
        for r in 0..k {
            for c in 0..k {
                let v = r * k + c;
                if c + 1 < k {
                    e.push((v, v + 1));
                }
                if r + 1 < k {
                    e.push((v, v + k));
                }
            }
        }
        e
    }

    /// The edge–node incidence pattern (`|E| × n`): its `MᵀM` is the graph.
    fn incidence(n: usize, edges: &[(usize, usize)], label: &[u32]) -> Pattern {
        let mut e = Vec::new();
        for (r, &(a, b)) in edges.iter().enumerate() {
            e.push((r, label[a] as usize));
            e.push((r, label[b] as usize));
        }
        pat(edges.len().max(1), n, &e)
    }

    fn fill(s: &Pattern, p: &Perm) -> usize {
        s.cholesky_nnz(p).unwrap()
    }

    // -----------------------------------------------------------------
    // Naive references.
    // -----------------------------------------------------------------

    /// Exact minimum degree on the explicit elimination graph (ties: the
    /// smallest index) — the classical algorithm AMD approximates.
    fn exact_minimum_degree(s: &Pattern) -> Perm {
        let n = s.n_cols();
        let g = s.symmetric().unwrap();
        let mut adj: Vec<Vec<bool>> = vec![vec![false; n]; n];
        // `g` is symmetric: column j lists the neighbours of j.
        for (j, row) in adj.iter_mut().enumerate() {
            for &i in g.col(j) {
                if i as usize != j {
                    row[i as usize] = true;
                }
            }
        }
        let mut alive = vec![true; n];
        let mut fwd = Vec::with_capacity(n);
        for _ in 0..n {
            let degree = |v: usize| (0..n).filter(|&w| alive[w] && adj[v][w]).count();
            let p = (0..n)
                .filter(|&v| alive[v])
                .min_by_key(|&v| (degree(v), v))
                .unwrap();
            let nb: Vec<usize> = (0..n).filter(|&w| alive[w] && adj[p][w]).collect();
            for &a in &nb {
                for &b in &nb {
                    if a != b {
                        adj[a][b] = true;
                    }
                }
            }
            alive[p] = false;
            fwd.push(idx32(p));
        }
        Perm::from_fwd(fwd).unwrap()
    }

    /// The least fill over all `n!` orders (brute force, `n ≤ 7`).
    fn optimal_fill(s: &Pattern) -> usize {
        fn rec(k: usize, v: &mut Vec<u32>, s: &Pattern, best: &mut usize) {
            if k == v.len() {
                let f = s.cholesky_nnz(&Perm::from_fwd(v.clone()).unwrap()).unwrap();
                *best = (*best).min(f);
                return;
            }
            for i in k..v.len() {
                v.swap(k, i);
                rec(k + 1, v, s, best);
                v.swap(k, i);
            }
        }
        let mut v: Vec<u32> = (0..s.n_cols()).map(idx32).collect();
        let mut best = usize::MAX;
        rec(0, &mut v, s, &mut best);
        best
    }

    fn random_graph(rng: &mut Rng, n: usize, pct: usize) -> Pattern {
        let mut e = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                if rng.below(100) < pct {
                    e.push((a, b));
                }
            }
        }
        let id: Vec<u32> = (0..n).map(idx32).collect();
        graph(n, &e, &id)
    }

    // -----------------------------------------------------------------
    // Examples and properties.
    // -----------------------------------------------------------------

    #[test]
    fn natural_and_errors() {
        let s = pat(2, 3, &[(0, 0), (1, 2)]);
        assert!(
            Natural
                .order(&s, OrderingTarget::Columns)
                .unwrap()
                .is_identity()
        );
        assert!(Natural.order(&s, OrderingTarget::Symmetric).is_err());
        assert!(Amd::default().order(&s, OrderingTarget::Symmetric).is_err());
        assert!(
            Colamd::default()
                .order(&s, OrderingTarget::Symmetric)
                .is_err()
        );
        assert_eq!(
            Amd::default()
                .order(&s, OrderingTarget::Columns)
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            Colamd::default()
                .order(&s, OrderingTarget::Columns)
                .unwrap()
                .len(),
            3
        );
        let empty = Pattern::empty(0, 0);
        for o in [
            &Natural as &dyn Ordering,
            &Amd::default(),
            &Colamd::default(),
        ] {
            assert!(
                o.order(&empty, OrderingTarget::Symmetric)
                    .unwrap()
                    .is_empty()
            );
            assert_ne!(o.name(), "");
        }
        assert_eq!(dense_threshold(None, 100), 100);
        assert_eq!(dense_threshold(Some(1.0), 4), DENSE_MIN);
        assert_eq!(dense_threshold(Some(-1.0), 4), usize::MAX);
    }

    /// A path long enough that the flag of step Q4 (`n + 1` per pivot, in
    /// `u32`) restarts (Chapter 5.11, "Constant factors"): the order is
    /// still perfect (`nnz(L) = 2n − 1`).
    #[test]
    fn long_path_restarts_the_flag_and_stays_perfect() {
        let n = 100_000;
        let edges: Vec<(usize, usize)> = (1..n).map(|i| (i, i - 1)).collect();
        let s = pat(n, n, &edges).symmetric().unwrap();
        for o in [&Amd::default() as &dyn Ordering, &Colamd::default()] {
            let p = o.order(&s, OrderingTarget::Symmetric).unwrap();
            assert_eq!(s.cholesky_nnz(&p).unwrap(), 2 * n - 1, "{}", o.name());
        }
    }

    /// Example 5.10.1: the arrowhead with its hub first fills completely
    /// in the natural order; AMD and COLAMD order the hub last (by degree —
    /// mass-eliminated with the last leaf — for small `n`, as a dense node
    /// for large `n`) and fill nothing.
    #[test]
    fn ex_5_10_1_arrowhead() {
        for n in [8_usize, 30, 400] {
            let edges: Vec<(usize, usize)> = (1..n).map(|i| (0, i)).collect();
            let id: Vec<u32> = (0..n).map(idx32).collect();
            let s = graph(n, &edges, &id);
            let nat = fill(&s, &Natural.order(&s, OrderingTarget::Symmetric).unwrap());
            assert_eq!(nat, n * (n + 1) / 2);
            for o in [&Amd::default() as &dyn Ordering, &Colamd::default()] {
                let p = o.order(&s, OrderingTarget::Symmetric).unwrap();
                assert_eq!(fill(&s, &p), 2 * n - 1, "{} n = {n}", o.name());
                // Last, or mass-eliminated with the last leaf (Q5).
                assert!(
                    p.inv[0] as usize + 2 >= n,
                    "{}: hub at {}",
                    o.name(),
                    p.inv[0]
                );
            }
            let m = incidence(n, &edges, &id);
            let p = Colamd::default()
                .order(&m, OrderingTarget::Columns)
                .unwrap();
            assert_eq!(fill(&s, &p), 2 * n - 1);
        }
    }

    /// Example 5.10.2: on a `k × k` grid, AMD (on a random relabelling)
    /// fills less than the natural band order.
    #[test]
    fn ex_5_10_2_grids_beat_the_band() {
        let mut rng = Rng::new(7);
        for k in [6_usize, 10, 20, 30] {
            let n = k * k;
            let e = grid_edges(k);
            let id: Vec<u32> = (0..n).map(idx32).collect();
            let band = fill(&graph(n, &e, &id), &Perm::identity(n));
            let label = random_label(&mut rng, n);
            let s = graph(n, &e, &label);
            let amd = fill(
                &s,
                &Amd::default().order(&s, OrderingTarget::Symmetric).unwrap(),
            );
            let sym = fill(
                &s,
                &Colamd::default()
                    .order(&s, OrderingTarget::Symmetric)
                    .unwrap(),
            );
            let m = incidence(n, &e, &label);
            let col = fill(
                &s,
                &Colamd::default()
                    .order(&m, OrderingTarget::Columns)
                    .unwrap(),
            );
            assert!(
                amd <= band && sym <= band && col <= band,
                "k {k}: band {band}, amd {amd}, symamd {sym}, colamd {col}"
            );
        }
    }

    /// Brute force: on every small random graph the fill of AMD is at
    /// least the optimum (a check of the counter) and, summed over the
    /// sample, within 10 % of exact minimum degree and of 1.25× the
    /// optimum; the same for COLAMD on the incidence pattern.
    #[test]
    fn brute_force_against_minimum_degree_and_optimum() {
        let mut rng = Rng::new(2026);
        let (mut amd_s, mut col_s, mut md_s, mut opt_s) = (0, 0, 0, 0);
        for case in 0..300 {
            let n = 2 + case % 6;
            let s = random_graph(&mut rng, n, 20 + case % 50);
            let opt = optimal_fill(&s);
            let amd = fill(
                &s,
                &Amd::default().order(&s, OrderingTarget::Symmetric).unwrap(),
            );
            let mut edges = Vec::new();
            for j in 0..n {
                for &i in s.col(j) {
                    if (i as usize) < j {
                        edges.push((i as usize, j));
                    }
                }
            }
            let id: Vec<u32> = (0..n).map(idx32).collect();
            let m = incidence(n, &edges, &id);
            let col = fill(
                &s,
                &Colamd::default()
                    .order(&m, OrderingTarget::Columns)
                    .unwrap(),
            );
            let md = fill(&s, &exact_minimum_degree(&s));
            assert!(amd >= opt && col >= opt && md >= opt);
            amd_s += amd;
            col_s += col;
            md_s += md;
            opt_s += opt;
        }
        let r = |a: usize, b: usize| count_f64(a) / count_f64(b);
        assert!(r(amd_s, md_s) <= 1.10, "AMD/MD {}", r(amd_s, md_s));
        assert!(r(col_s, md_s) <= 1.10, "COLAMD/MD {}", r(col_s, md_s));
        assert!(r(amd_s, opt_s) <= 1.25 && r(col_s, opt_s) <= 1.25);
    }

    /// Larger graphs: AMD within 10 % of exact minimum degree in total.
    #[test]
    fn amd_tracks_exact_minimum_degree() {
        let mut rng = Rng::new(11);
        let (mut amd_s, mut md_s) = (0, 0);
        for case in 0..40 {
            let n = 20 + case;
            let s = random_graph(&mut rng, n, 3 + case % 8);
            amd_s += fill(
                &s,
                &Amd::default().order(&s, OrderingTarget::Symmetric).unwrap(),
            );
            md_s += fill(&s, &exact_minimum_degree(&s));
        }
        assert!(
            count_f64(amd_s) <= 1.10 * count_f64(md_s),
            "AMD {amd_s} MD {md_s}"
        );
    }

    proptest! {
        /// (O1): every ordering is a bijection, for both targets, on random
        /// rectangular patterns with empty and dense lines.
        #[test]
        fn orderings_are_permutations(m in 1_usize..40, n in 1_usize..40,
                                      e in prop::collection::vec((0_usize..40, 0_usize..40), 0..300),
                                      dense_line in 0_usize..3, knob in -1.0_f64..2.0) {
            let mut e: Vec<(usize, usize)> = e.into_iter().map(|(i, j)| (i % m, j % n)).collect();
            if dense_line == 1 {
                e.extend((0..n).map(|j| (0, j)));
            } else if dense_line == 2 {
                e.extend((0..m).map(|i| (i, 0)));
            }
            let s = pat(m, n, &e);
            let sq = pat(n, n, &e.iter().map(|&(i, j)| (i % n, j)).collect::<Vec<_>>());
            let orderings: [&dyn Ordering; 5] = [
                &Natural,
                &Amd::default(),
                &Amd { dense: Some(knob), aggressive: false },
                &Colamd::default(),
                &Colamd { dense_row: Some(knob), dense_col: Some(knob) },
            ];
            for o in orderings {
                let p = o.order(&s, OrderingTarget::Columns).unwrap();
                prop_assert_eq!(p.len(), n);
                prop_assert!(p.check_invariants().is_ok());
                let q = o.order(&sq, OrderingTarget::Symmetric).unwrap();
                prop_assert_eq!(q.len(), n);
                prop_assert!(q.check_invariants().is_ok());
                prop_assert_eq!(o.order(&sq, OrderingTarget::Symmetric).unwrap(), q);
            }
        }

        /// Trees (relabelled at random) are eliminated without fill by AMD
        /// and SYMAMD, and COLAMD on their incidence pattern; the natural
        /// order of the relabelled tree is never better.
        #[test]
        fn trees_have_no_fill(n in 1_usize..120, seed in any::<u64>()) {
            let mut rng = Rng::new(seed);
            let e = tree_edges(&mut rng, n);
            let label = random_label(&mut rng, n);
            let s = graph(n, &e, &label);
            let none = 2 * n - 1;
            prop_assert!(fill(&s, &Perm::identity(n)) >= none);
            prop_assert_eq!(fill(&s, &Amd::default().order(&s, OrderingTarget::Symmetric).unwrap()), none);
            prop_assert_eq!(fill(&s, &Colamd::default().order(&s, OrderingTarget::Symmetric).unwrap()), none);
            let m = incidence(n, &e, &label);
            prop_assert_eq!(fill(&s, &Colamd::default().order(&m, OrderingTarget::Columns).unwrap()), none);
            prop_assert_eq!(fill(&s, &Amd::default().order(&m, OrderingTarget::Columns).unwrap()), none);
        }

        /// Structured matrices: a block-arrow (dense border of width `w`
        /// first, then independent diagonal blocks) relabelled at random —
        /// the orderings never fill more than the natural order of the
        /// unrelabelled matrix, which puts the border first and so turns
        /// all the blocks into one clique. (With one block the natural
        /// order has no fill at all, and a minimum-degree tie can cost one
        /// entry: the case is excluded, it is not a property; so are borders
        /// whose nodes have no more neighbours than the block nodes.)
        #[test]
        fn block_arrows_beat_the_natural_order(blocks in 2_usize..8, size in 2_usize..6,
                                               w in 1_usize..4, seed in any::<u64>()) {
            // The block nodes must be the ones of least degree
            // (`size − 1 + w < blocks · size`), else minimum degree may
            // start on the border as the natural order does.
            prop_assume!(size + w <= blocks * size);
            let mut rng = Rng::new(seed);
            let n = w + blocks * size;
            let mut e = Vec::new();
            for b in 0..blocks {
                let base = w + b * size;
                for x in 0..size {
                    for y in x + 1..size {
                        e.push((base + x, base + y));
                    }
                    for c in 0..w {
                        e.push((c, base + x));
                    }
                }
            }
            let id: Vec<u32> = (0..n).map(idx32).collect();
            let nat = fill(&graph(n, &e, &id), &Perm::identity(n));
            let label = random_label(&mut rng, n);
            let s = graph(n, &e, &label);
            for o in [&Amd::default() as &dyn Ordering, &Colamd::default()] {
                let f = fill(&s, &o.order(&s, OrderingTarget::Symmetric).unwrap());
                prop_assert!(f <= nat, "{}: {} > natural {}", o.name(), f, nat);
            }
        }
    }
}
