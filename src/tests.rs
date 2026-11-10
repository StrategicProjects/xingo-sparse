//! Tests of Chapter 5: dense oracles, generators of simplex-like bases and
//! the property tests of tasks P1.1, P1.2 and L1.3 (step L5′, Algorithm R).

use proptest::prelude::*;

use crate::real::{approx_eq, is_exact_zero, le, lt};
use crate::{
    Col, ColumnRef, CscMatrix, Lu, LuError, LuFactor, LuOptions, PreOrder, RefactorReason, Row,
    Sign, SparseVec, UpdateOutcome,
};

/// The factor of these tests: `f64`, Forrest–Tomlin.
type LuF64 = Lu<f64>;

/// Factorizes the basis `basis` of `a` (the first edition's
/// `Factor::factorize(&basis, &a)`).
fn fz(f: &mut LuF64, basis: &[ColumnRef], a: &CscMatrix) -> Result<(), LuError> {
    f.factorize(&(basis, a))
}

// ---------------------------------------------------------------------------
// Dense oracle (Gaussian elimination with partial pivoting).
// ---------------------------------------------------------------------------

type Dense = Vec<Vec<f64>>;

/// `B` as a dense row-major matrix.
fn dense_basis(basis: &[ColumnRef], a: &CscMatrix) -> Dense {
    let m = basis.len();
    let mut b = vec![vec![0.0; m]; m];
    for (k, c) in basis.iter().enumerate() {
        c.for_each_entry(a, |i, v| b[i as usize][k] += v);
    }
    b
}

fn transpose(b: &Dense) -> Dense {
    let m = b.len();
    (0..m).map(|j| (0..m).map(|i| b[i][j]).collect()).collect()
}

/// Solves `B x = rhs` by Gaussian elimination with partial pivoting; `None`
/// if a pivot is below `1e-9 · max |b_ij|`.
fn dense_solve(b: &Dense, rhs: &[f64]) -> Option<Vec<f64>> {
    let m = b.len();
    let mut a: Dense = b.clone();
    let mut x = rhs.to_vec();
    let scale = a
        .iter()
        .flatten()
        .fold(0.0_f64, |g, v| g.max(v.abs()))
        .max(1.0);
    for k in 0..m {
        let piv = (k..m).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if le(a[piv][k].abs(), 1e-9 * scale, 0.0) {
            return None;
        }
        a.swap(k, piv);
        x.swap(k, piv);
        let row_k = a[k].clone();
        for i in k + 1..m {
            let f = a[i][k] / row_k[k];
            if is_exact_zero(f) {
                continue;
            }
            for (a_ij, a_kj) in a[i][k..].iter_mut().zip(&row_k[k..]) {
                *a_ij -= f * a_kj;
            }
            x[i] -= f * x[k];
        }
    }
    for k in (0..m).rev() {
        let s: f64 = (k + 1..m).map(|j| a[k][j] * x[j]).sum();
        x[k] = (x[k] - s) / a[k][k];
    }
    Some(x)
}

/// `B⁻¹` by Gauss–Jordan elimination with partial pivoting (`None` if a
/// pivot is below `1e-12 · max |b_ij|`).
fn dense_inverse(b: &Dense) -> Option<Dense> {
    let m = b.len();
    let mut a: Dense = b.clone();
    let mut inv: Dense = (0..m)
        .map(|i| (0..m).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let scale = a
        .iter()
        .flatten()
        .fold(0.0_f64, |g, v| g.max(v.abs()))
        .max(1.0);
    for k in 0..m {
        let piv = (k..m).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if le(a[piv][k].abs(), 1e-12 * scale, 0.0) {
            return None;
        }
        a.swap(k, piv);
        inv.swap(k, piv);
        let d = a[k][k];
        for j in 0..m {
            a[k][j] /= d;
            inv[k][j] /= d;
        }
        let (row_a, row_i) = (a[k].clone(), inv[k].clone());
        for i in (0..m).filter(|&i| i != k) {
            let f = a[i][k];
            if is_exact_zero(f) {
                continue;
            }
            for j in 0..m {
                a[i][j] -= f * row_a[j];
                inv[i][j] -= f * row_i[j];
            }
        }
    }
    Some(inv)
}

/// Skeel's condition numbers of `B` for the solves with `B` and with `Bᵀ`:
/// `max(‖ |B⁻¹| |B| ‖∞, ‖ |B| |B⁻¹| ‖₁)`, the factor by which a
/// perturbation of the rounding order of the same elimination can move
/// the solution, relative to `m ε` (Higham 2002, §7.2 and Thm. 8.5).
fn skeel_condition(b: &Dense) -> f64 {
    let m = b.len();
    let Some(inv) = dense_inverse(b) else {
        return f64::INFINITY;
    };
    let abs = |d: &Dense| -> Dense {
        d.iter()
            .map(|r| r.iter().map(|v| v.abs()).collect())
            .collect()
    };
    let (ai, ab) = (abs(&inv), abs(b));
    let left = mat_mul(&ai, &ab); // |B⁻¹| |B|: rows for the ∞-norm
    let right = mat_mul(&ab, &ai); // |B| |B⁻¹|: columns for the 1-norm
    let norm_inf = left
        .iter()
        .map(|r| r.iter().sum::<f64>())
        .fold(0.0, f64::max);
    let norm_one = (0..m)
        .map(|j| (0..m).map(|i| right[i][j]).sum::<f64>())
        .fold(0.0, f64::max);
    norm_inf.max(norm_one).max(1.0)
}

/// Number of proptest cases: `default`, or `PROPTEST_CASES` when set (for
/// long stress runs).
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn count(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap())
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |g, x| g.max(x.abs()))
}

/// Normwise backward error `‖B x − b‖∞ / ((‖B‖∞ ‖x‖∞ + ‖b‖∞) m)`,
/// independent of the conditioning of `B`.
fn backward_err(b: &Dense, x: &[f64], rhs: &[f64]) -> f64 {
    let r: Vec<f64> = b
        .iter()
        .zip(rhs)
        .map(|(row, &bi)| row.iter().zip(x).map(|(a, xi)| a * xi).sum::<f64>() - bi)
        .collect();
    let norm_b = b
        .iter()
        .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    inf_norm(&r) / ((norm_b * inf_norm(x) + inf_norm(rhs)).max(1.0) * count(b.len().max(1)))
}

/// `‖x − y‖∞ / max(1, ‖y‖∞)`.
fn rel_err(x: &[f64], y: &[f64]) -> f64 {
    let d: Vec<f64> = x.iter().zip(y).map(|(a, b)| a - b).collect();
    inf_norm(&d) / inf_norm(y).max(1.0)
}

fn ftran_vec(f: &mut LuF64, b: &[f64]) -> Vec<f64> {
    let mut v = SparseVec::from_dense(b);
    f.ftran(&mut v);
    assert!(v.check_invariants().is_ok());
    v.values().to_vec()
}

fn btran_vec(f: &mut LuF64, c: &[f64]) -> Vec<f64> {
    let mut v = SparseVec::from_dense(c);
    f.btran(&mut v);
    v.values().to_vec()
}

/// Dense `L` and `U` in step order, and `P B Q`, from the stored factors.
fn dense_lu(f: &LuF64, b: &Dense) -> (Dense, Dense, Dense) {
    let f = &f.factors;
    let m = f.m;
    let mut l = vec![vec![0.0; m]; m];
    let mut u = vec![vec![0.0; m]; m];
    for (s, row) in l.iter_mut().enumerate() {
        row[s] = 1.0;
    }
    for (e, &p) in f.l_pivot.iter().enumerate() {
        let s = f.row_step[p as usize];
        for q in f.l_start[e]..f.l_start[e + 1] {
            l[f.row_step[f.l_index[q] as usize]][s] = f.l_value[q];
        }
    }
    for i in 0..m {
        let s = f.row_step[i];
        u[s][s] = f.diag[i];
        let (idx, val) = f.u_rows.line(i);
        for (&j, &v) in idx.iter().zip(val) {
            u[s][f.col_step[j as usize]] = v;
        }
    }
    let pbq: Dense = (0..m)
        .map(|s| {
            (0..m)
                .map(|t| b[f.step_row[s] as usize][f.step_col[t] as usize])
                .collect()
        })
        .collect();
    (l, u, pbq)
}

fn mat_mul(a: &Dense, b: &Dense) -> Dense {
    let m = a.len();
    (0..m)
        .map(|i| {
            (0..m)
                .map(|j| (0..m).map(|k| a[i][k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}

/// Max-norm of `L U − P B Q` relative to `max(1, max |b_ij|, max |u_ij|)`,
/// divided by `m` — the backward error of Gaussian elimination is bounded
/// by a small multiple of `m ε` times the size of the factors (Higham 2002,
/// Thm. 9.3), so this should be a modest multiple of `ε`.
fn lu_residual(f: &LuF64, b: &Dense) -> f64 {
    let (l, u, pbq) = dense_lu(f, b);
    let lu = mat_mul(&l, &u);
    let scale = b
        .iter()
        .flatten()
        .fold(1.0_f64, |g, v| g.max(v.abs()))
        .max(f.stats().max_u)
        * count(f.m().max(1));
    lu.iter()
        .flatten()
        .zip(pbq.iter().flatten())
        .fold(0.0_f64, |g, (x, y)| g.max((x - y).abs()))
        / scale
}

// ---------------------------------------------------------------------------
// Generators of simplex-like bases.
// ---------------------------------------------------------------------------

/// xorshift64* — the proptest seed drives a deterministic generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % (n as u64)).unwrap()
    }
    /// `true` with probability `num / den`.
    fn chance(&mut self, num: usize, den: usize) -> bool {
        self.below(den) < num
    }
    /// A nonzero integer in `[-k, k]`.
    fn nonzero(&mut self, k: i32) -> f64 {
        let k = usize::try_from(k).unwrap();
        let v = i32::try_from(self.below(k) + 1).unwrap();
        f64::from(if self.chance(1, 2) { v } else { -v })
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
    fn dense_ints(&mut self, m: usize) -> Vec<f64> {
        (0..m)
            .map(|_| {
                if self.chance(1, 4) {
                    0.0
                } else {
                    self.nonzero(5)
                }
            })
            .collect()
    }
}

/// Shape of a generated basis.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// Probability (in percent) of an entry below the (permuted) diagonal.
    lower: usize,
    /// Probability (in percent) of an entry above it.
    upper: usize,
    /// Percentage of logical columns.
    logical: usize,
    /// Diagonal dominance (well-conditioned) or not (triangular tests).
    dominant: bool,
}

/// A basis `B = P M Qᵀ` mixing logicals and structurals, plus `extra`
/// additional sparse structural columns for the update tests. Logical
/// columns sit on rows `S`; the structural block on the other rows is
/// diagonally dominant by rows and columns when `shape.dominant`, so `B`
/// is block triangular with well-conditioned blocks, as simplex bases are.
fn gen_basis(rng: &mut Rng, m: usize, shape: Shape, extra: usize) -> (CscMatrix, Vec<ColumnRef>) {
    let mut rows: Vec<usize> = (0..m).collect();
    rng.shuffle(&mut rows);
    let n_log = m * shape.logical / 100;
    let (log_rows, str_rows) = rows.split_at(n_log);
    let s = str_rows.len();
    // Structural column c has its diagonal at row str_rows[c].
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); s];
    let mut row_sum = vec![0.0_f64; m];
    for (c, col) in cols.iter_mut().enumerate() {
        for (r, &i) in str_rows.iter().enumerate() {
            let pct = if r > c { shape.lower } else { shape.upper };
            if r != c && rng.chance(pct, 100) {
                let v = rng.nonzero(3);
                col.push((i, v));
                row_sum[i] += v.abs();
            }
        }
        for &i in log_rows {
            if rng.chance(shape.lower, 100) {
                col.push((i, rng.nonzero(3)));
            }
        }
    }
    for (c, col) in cols.iter_mut().enumerate() {
        let i = str_rows[c];
        let col_sum: f64 = col
            .iter()
            .filter(|(r, _)| !log_rows.contains(r))
            .map(|(_, v)| v.abs())
            .sum();
        let d = if shape.dominant {
            1.0 + row_sum[i].max(col_sum) + rng.nonzero(2).abs()
        } else {
            rng.nonzero(4)
        };
        col.push((i, if rng.chance(1, 2) { d } else { -d }));
    }
    // Extra columns: sparse integers with one larger entry.
    for _ in 0..extra {
        let mut col = Vec::new();
        let big = rng.below(m);
        for i in 0..m {
            if i == big {
                col.push((i, rng.nonzero(3) * 4.0));
            } else if rng.chance(15, 100) {
                col.push((i, rng.nonzero(2)));
            }
        }
        cols.push(col);
    }
    let triplets: Vec<(Row, Col, f64)> = cols
        .iter()
        .enumerate()
        .flat_map(|(j, col)| {
            col.iter()
                .map(move |&(i, v)| (Row::from_usize(i), Col::from_usize(j), v))
        })
        .collect();
    let a = CscMatrix::from_triplets(m, cols.len(), &triplets).unwrap();
    let mut basis: Vec<ColumnRef> = (0..s)
        .map(|j| ColumnRef::Structural(Col::from_usize(j)))
        .collect();
    for &i in log_rows {
        let sign = if rng.chance(1, 2) {
            Sign::Plus
        } else {
            Sign::Minus
        };
        basis.push(ColumnRef::Logical(Row::from_usize(i), sign));
    }
    rng.shuffle(&mut basis);
    (a, basis)
}

const SPARSE: Shape = Shape {
    lower: 25,
    upper: 25,
    logical: 30,
    dominant: true,
};
const DENSE: Shape = Shape {
    lower: 70,
    upper: 70,
    logical: 0,
    dominant: true,
};
const QUASI_TRIANGULAR: Shape = Shape {
    lower: 30,
    upper: 4,
    logical: 40,
    dominant: true,
};
const SLACK: Shape = Shape {
    lower: 0,
    upper: 0,
    logical: 100,
    dominant: true,
};
const TRIANGULAR: Shape = Shape {
    lower: 40,
    upper: 0,
    logical: 20,
    dominant: false,
};

fn shape_of(k: usize) -> Shape {
    [SPARSE, DENSE, QUASI_TRIANGULAR, SLACK][k % 4]
}

fn factor_ok(basis: &[ColumnRef], a: &CscMatrix) -> LuF64 {
    let mut f = LuF64::default();
    fz(&mut f, basis, a).unwrap();
    assert!(f.check_invariants().is_ok(), "{:?}", f.check_invariants());
    f
}

// ---------------------------------------------------------------------------
// P1.1 — factorization and solves.
// ---------------------------------------------------------------------------

#[test]
fn ex_5_1_slack_basis_is_free() {
    let m = 6;
    let a = CscMatrix::zeros(m, 0);
    let basis: Vec<ColumnRef> = (0..m)
        .map(|i| {
            let sign = if i % 2 == 0 { Sign::Plus } else { Sign::Minus };
            ColumnRef::Logical(Row::from_usize((i + 2) % m), sign)
        })
        .collect();
    let mut f = factor_ok(&basis, &a);
    let s = f.stats();
    assert_eq!(
        (s.nnz_l, s.nnz_u, s.markowitz_pivots, s.col_singletons),
        (0, m, 0, m)
    );
    assert!(approx_eq(s.growth, 1.0, 0.0));
    let b: Vec<f64> = (0..m)
        .map(|i| f64::from(u32::try_from(i).unwrap()) + 1.0)
        .collect();
    let x = ftran_vec(&mut f, &b);
    for (k, c) in basis.iter().enumerate() {
        let ColumnRef::Logical(i, sign) = *c else {
            unreachable!()
        };
        assert!(approx_eq(x[k], sign.value() * b[i.index()], 0.0));
    }
}

#[test]
fn ex_5_2_small_bump_by_hand() {
    // B = [[4, 1, 0], [1, 4, 1], [0, 1, 4]]: no singletons, a 3×3 bump.
    let t = [
        (0, 0, 4.0),
        (1, 0, 1.0),
        (0, 1, 1.0),
        (1, 1, 4.0),
        (2, 1, 1.0),
        (1, 2, 1.0),
        (2, 2, 4.0),
    ]
    .map(|(i, j, v)| (Row(i), Col(j), v));
    let a = CscMatrix::from_triplets(3, 3, &t).unwrap();
    let basis = [0, 1, 2].map(|j| ColumnRef::Structural(Col(j)));
    let mut f = factor_ok(&basis, &a);
    assert_eq!(f.stats().markowitz_pivots, 3);
    let b = dense_basis(&basis, &a);
    assert!(lu_residual(&f, &b) < 1e-15);
    let x = ftran_vec(&mut f, &[5.0, 6.0, 5.0]);
    for v in x {
        assert!(approx_eq(v, 1.0, 1e-14));
    }
}

#[test]
fn input_errors_are_reported() {
    let a = CscMatrix::zeros(2, 1);
    let mut f = LuF64::default();
    let e = fz(&mut f, &[ColumnRef::Structural(Col(0))], &a);
    assert!(matches!(e, Err(LuError::Structure(_))));
    let e = fz(
        &mut f,
        &[ColumnRef::Structural(Col(5)), ColumnRef::Structural(Col(0))],
        &a,
    );
    assert!(matches!(e, Err(LuError::Structure(_))));
    let e = fz(
        &mut f,
        &[
            ColumnRef::Logical(Row(2), Sign::Plus),
            ColumnRef::Structural(Col(0)),
        ],
        &a,
    );
    assert!(matches!(e, Err(LuError::Structure(_))));
    assert!(!f.is_valid());
    assert_eq!(f.update(0, 1.0), Err(LuError::NotFactorized));
    // m = 0 is a valid (empty) basis.
    fz(&mut f, &[], &CscMatrix::zeros(0, 0)).unwrap();
    assert!(f.is_valid());
}

/// Checks that `rows` repair the singular basis (invariant E1).
fn assert_repairable(basis: &[ColumnRef], a: &CscMatrix, e: &LuError, expected_def: usize) {
    let LuError::Singular {
        rank_deficiency,
        cols,
        rows,
    } = e
    else {
        panic!("expected Singular, got {e:?}");
    };
    assert_eq!(*rank_deficiency, expected_def);
    assert_eq!((cols.len(), rows.len()), (expected_def, expected_def));
    let mut repaired = basis.to_vec();
    for (&k, &i) in cols.iter().zip(rows) {
        repaired[k] = ColumnRef::Logical(i, Sign::Plus);
    }
    let f = factor_ok(&repaired, a);
    let b = dense_basis(&repaired, a);
    assert!(lu_residual(&f, &b) < 1e-15);
}

#[test]
fn ex_5_3_singular_bases_are_diagnosed() {
    let mut rng = Rng::new(7);
    let (a, basis) = gen_basis(&mut rng, 12, SPARSE, 0);
    let structural: Vec<usize> = (0..12)
        .filter(|&k| matches!(basis[k], ColumnRef::Structural(_)))
        .collect();
    let mut f = LuF64::default();

    // Duplicate column.
    let mut dup = basis.clone();
    dup[structural[0]] = dup[structural[1]];
    let e = fz(&mut f, &dup, &a).unwrap_err();
    assert_repairable(&dup, &a, &e, 1);

    // Empty column and zero row: B with a structurally zero column.
    let t = [(Row(0), Col(0), 1.0), (Row(1), Col(0), 2.0)];
    let z = CscMatrix::from_triplets(3, 2, &t).unwrap();
    let zb = [
        ColumnRef::Structural(Col(0)),
        ColumnRef::Structural(Col(1)),
        ColumnRef::Logical(Row(0), Sign::Minus),
    ];
    let e = fz(&mut f, &zb, &z).unwrap_err();
    assert_repairable(&zb, &z, &e, 1);

    // Two dependent combinations: b_k = Σ of two others, twice.
    let b = dense_basis(&basis, &a);
    let n = a.n_cols();
    let mut trip: Vec<(Row, Col, f64)> = Vec::new();
    for j in 0..n {
        let (rs, vs) = a.col(Col::from_usize(j));
        trip.extend(
            rs.iter()
                .zip(vs)
                .map(|(&i, &v)| (Row(i), Col::from_usize(j), v)),
        );
    }
    let (k0, k1, k2, k3) = (structural[0], structural[1], structural[2], structural[3]);
    for (extra, (x, y)) in [(k0, k1), (k2, k3)].into_iter().enumerate() {
        for (i, b_i) in b.iter().enumerate() {
            let v = b_i[x] + 2.0 * b_i[y];
            if !is_exact_zero(v) {
                trip.push((Row::from_usize(i), Col::from_usize(n + extra), v));
            }
        }
    }
    let a2 = CscMatrix::from_triplets(12, n + 2, &trip).unwrap();
    let mut dep = basis.clone();
    dep[structural[4]] = ColumnRef::Structural(Col::from_usize(n));
    dep[structural[5]] = ColumnRef::Structural(Col::from_usize(n + 1));
    let e = fz(&mut f, &dep, &a2).unwrap_err();
    assert_repairable(&dep, &a2, &e, 2);
    assert!(!f.is_valid());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(96)))]

    /// P1.1: `L·U = P·B·Q` on random sparse, dense, quasi-triangular and
    /// slack bases; no growth above the threshold on these well-conditioned
    /// bases; FTRAN/BTRAN agree with the dense oracle.
    #[test]
    fn lu_equals_pbq_and_solves_match_oracle(m in 1_usize..40, kind in 0_usize..4, seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let (a, basis) = gen_basis(&mut rng, m, shape_of(kind), 0);
        let mut f = factor_ok(&basis, &a);
        let b = dense_basis(&basis, &a);
        prop_assert!(lu_residual(&f, &b) < 1e-15, "residual {}", lu_residual(&f, &b));
        let s = *f.stats();
        prop_assert!(!s.growth_exceeded && s.growth < 1e6, "growth {}", s.growth);
        prop_assert_eq!(s.col_singletons + s.row_singletons + s.markowitz_pivots, m);
        prop_assert_eq!(s.fill_in, (s.nnz_l + s.nnz_u).saturating_sub(s.nnz_basis));

        let rhs = rng.dense_ints(m);
        let x = ftran_vec(&mut f, &rhs);
        let x_ref = dense_solve(&b, &rhs).unwrap();
        prop_assert!(rel_err(&x, &x_ref) < 1e-9, "ftran error {}", rel_err(&x, &x_ref));
        let tol_bw = 1e-14 * s.growth.max(1.0);
        prop_assert!(backward_err(&b, &x, &rhs) < tol_bw, "ftran residual {}", backward_err(&b, &x, &rhs));
        let y = btran_vec(&mut f, &rhs);
        let bt = transpose(&b);
        let y_ref = dense_solve(&bt, &rhs).unwrap();
        prop_assert!(rel_err(&y, &y_ref) < 1e-9, "btran error {}", rel_err(&y, &y_ref));
        prop_assert!(backward_err(&bt, &y, &rhs) < tol_bw, "btran residual {}", backward_err(&bt, &y, &rhs));
    }

    /// Singleton test: a basis that is triangular up to permutations is
    /// factorized by the triangular phase alone (no Markowitz pivot), and
    /// still satisfies `L·U = P·B·Q`.
    #[test]
    fn triangular_basis_needs_no_markowitz_pivot(m in 1_usize..50, seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let (a, basis) = gen_basis(&mut rng, m, TRIANGULAR, 0);
        let f = factor_ok(&basis, &a);
        prop_assert_eq!(f.stats().markowitz_pivots, 0);
        let b = dense_basis(&basis, &a);
        prop_assert!(lu_residual(&f, &b) < 1e-15);
    }
}

// ---------------------------------------------------------------------------
// P1.2 — Forrest–Tomlin updates.
// ---------------------------------------------------------------------------

/// Performs one valid exchange: enters candidate column `q` of `a` at a
/// position `r` with `|α_r| ≥ ½ ‖α‖∞`; returns `(r, outcome)`.
fn exchange(
    f: &mut LuF64,
    rng: &mut Rng,
    a: &CscMatrix,
    q: ColumnRef,
) -> (usize, Result<UpdateOutcome, LuError>) {
    let m = f.m();
    let mut col = vec![0.0; m];
    q.for_each_entry(a, |i, v| col[i as usize] = v);
    let mut v = SparseVec::from_dense(&col);
    f.ftran_for_update(&mut v);
    let alpha = v.values().to_vec();
    let amax = inf_norm(&alpha);
    let ok: Vec<usize> = (0..m)
        .filter(|&k| le(0.5 * amax, alpha[k].abs(), 0.0))
        .collect();
    let r = ok[rng.below(ok.len())];
    (r, f.update(r, alpha[r]))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(64)))]

    /// P1.2: after k random valid exchanges, FTRAN and BTRAN of the updated
    /// factor agree with a fresh factorization of the same basis and with
    /// the dense oracle (relative error ≤ 1e-9).
    #[test]
    fn updates_match_refactorization(m in 2_usize..30, kind in 0_usize..4, k in 1_usize..25, seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let (a, mut basis) = gen_basis(&mut rng, m, shape_of(kind), m);
        let opts = LuOptions { refactor_interval: 1000, fill_factor: 1e6, ..LuOptions::default() };
        let mut f = LuF64::new(opts);
        fz(&mut f, &basis, &a).unwrap();
        for step in 0..k {
            // Candidate: an extra column, or a logical, not in the basis.
            let cand = if rng.chance(3, 4) {
                ColumnRef::Structural(Col::from_usize(a.n_cols() - m + rng.below(m)))
            } else {
                ColumnRef::Logical(Row::from_usize(rng.below(m)), Sign::Plus)
            };
            if basis.iter().any(|c| match (c, &cand) {
                (ColumnRef::Logical(i, _), ColumnRef::Logical(j, _)) => i == j,
                _ => *c == cand,
            }) {
                continue;
            }
            let (r, out) = exchange(&mut f, &mut rng, &a, cand);
            let out = out.unwrap();
            prop_assert_eq!(out, UpdateOutcome::Updated, "step {}", step);
            basis[r] = cand;
            prop_assert!(f.check_invariants().is_ok(), "{:?}", f.check_invariants());

            let mut fresh = factor_ok(&basis, &a);
            let b = dense_basis(&basis, &a);
            let rhs = rng.dense_ints(m);
            let x = ftran_vec(&mut f, &rhs);
            let x_fresh = ftran_vec(&mut fresh, &rhs);
            let x_ref = dense_solve(&b, &rhs).unwrap();
            prop_assert!(rel_err(&x, &x_fresh) < 1e-9, "ftran vs fresh {}", rel_err(&x, &x_fresh));
            prop_assert!(rel_err(&x, &x_ref) < 1e-9, "ftran vs oracle {}", rel_err(&x, &x_ref));
            let y = btran_vec(&mut f, &rhs);
            let y_fresh = btran_vec(&mut fresh, &rhs);
            let y_ref = dense_solve(&transpose(&b), &rhs).unwrap();
            prop_assert!(rel_err(&y, &y_fresh) < 1e-9, "btran vs fresh {}", rel_err(&y, &y_fresh));
            prop_assert!(rel_err(&y, &y_ref) < 1e-9, "btran vs oracle {}", rel_err(&y, &y_ref));
        }
        prop_assert_eq!(f.update_stats().n_etas <= f.update_stats().updates, true);
    }

    /// P1.2: an exchange that makes the basis singular is detected, and the
    /// factor still represents the old basis.
    #[test]
    fn singular_update_is_detected(m in 2_usize..25, kind in 0_usize..4, seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let (a, basis) = gen_basis(&mut rng, m, shape_of(kind), 0);
        let mut f = factor_ok(&basis, &a);
        let b = dense_basis(&basis, &a);
        // a_q = B v with v_r = 0: a_q lies in the span of the other columns.
        let r = rng.below(m);
        let mut v = rng.dense_ints(m);
        v[r] = 0.0;
        let aq: Vec<f64> = (0..m).map(|i| (0..m).map(|k| b[i][k] * v[k]).sum()).collect();
        let mut w = SparseVec::from_dense(&aq);
        f.ftran_for_update(&mut w);
        let alpha_r = w.values()[r];
        let e = f.update(r, alpha_r).unwrap_err();
        let is_singular = matches!(e, LuError::Singular { rank_deficiency: 1, ref cols, .. } if cols == &vec![r]);
        prop_assert!(is_singular, "{:?}", e);
        // The factor still represents B: it solves exactly as a fresh
        // factorization does (the oracle itself differs by rounding, up to
        // ~1e-12 on these bases — a stress run found 1.1e-12 for both).
        let rhs = rng.dense_ints(m);
        let x = ftran_vec(&mut f, &rhs);
        let x_fresh = ftran_vec(&mut factor_ok(&basis, &a), &rhs);
        prop_assert!(rel_err(&x, &x_fresh) < 1e-12, "vs fresh {}", rel_err(&x, &x_fresh));
        prop_assert!(rel_err(&x, &dense_solve(&b, &rhs).unwrap()) < 1e-10);
    }
}

#[test]
fn unstable_update_is_refused_and_leaves_factor_intact() {
    let mut rng = Rng::new(11);
    let m = 10;
    let (a, basis) = gen_basis(&mut rng, m, SPARSE, m);
    let mut f = factor_ok(&basis, &a);
    let b = dense_basis(&basis, &a);
    let mut col = vec![0.0; m];
    ColumnRef::Structural(Col::from_usize(m)).for_each_entry(&a, |i, v| col[i as usize] = v);
    let mut v = SparseVec::from_dense(&col);
    f.ftran_for_update(&mut v);
    let r = (0..m)
        .max_by(|&i, &j| v.values()[i].abs().total_cmp(&v.values()[j].abs()))
        .unwrap();
    // A wrong α_r (as if FTRAN had been inaccurate) fails the stability test.
    let out = f.update(r, 2.0 * v.values()[r]).unwrap();
    assert_eq!(out, UpdateOutcome::RefactorNeeded(RefactorReason::Unstable));
    assert_eq!(f.update_stats().updates, 0);
    let rhs = rng.dense_ints(m);
    assert!(rel_err(&ftran_vec(&mut f, &rhs), &dense_solve(&b, &rhs).unwrap()) < 1e-12);
    // The spike is still recorded: the correct α_r now succeeds.
    assert!(f.has_spike());
    assert_eq!(f.update(r, v.values()[r]).unwrap(), UpdateOutcome::Updated);
    assert!(!f.has_spike());
    assert_eq!(f.update(r, 1.0), Err(LuError::NoSpike));
}

#[test]
fn refactor_interval_is_reported() {
    let mut rng = Rng::new(5);
    let m = 15;
    let (a, mut basis) = gen_basis(&mut rng, m, SPARSE, m);
    let opts = LuOptions {
        refactor_interval: 3,
        fill_factor: 1e6,
        ..LuOptions::default()
    };
    let mut f = LuF64::new(opts);
    fz(&mut f, &basis, &a).unwrap();
    let mut outcomes = Vec::new();
    for q in 0..3 {
        let cand = ColumnRef::Structural(Col::from_usize(m + q));
        let (r, out) = exchange(&mut f, &mut rng, &a, cand);
        basis[r] = cand;
        outcomes.push(out.unwrap());
    }
    assert_eq!(
        outcomes,
        vec![
            UpdateOutcome::Updated,
            UpdateOutcome::Updated,
            UpdateOutcome::RefactorNeeded(RefactorReason::UpdateLimit)
        ]
    );
    // The third update was applied: the factor matches the new basis.
    let b = dense_basis(&basis, &a);
    let rhs = rng.dense_ints(m);
    assert!(rel_err(&ftran_vec(&mut f, &rhs), &dense_solve(&b, &rhs).unwrap()) < 1e-10);
    assert_eq!(f.update_stats().updates, 3);
    // Refactorizing resets the counters.
    fz(&mut f, &basis, &a).unwrap();
    let u = f.update_stats();
    assert_eq!(
        (u.updates, u.n_etas, u.nnz_eta, f.stats().factorizations),
        (0, 0, 0, 2)
    );
}

#[test]
fn fill_growth_is_reported() {
    let mut rng = Rng::new(9);
    let m = 20;
    // A slack basis has nnz(L)+nnz(U) = m; dense entering columns fill U fast.
    let (a, mut basis) = gen_basis(&mut rng, m, SLACK, m);
    let opts = LuOptions {
        fill_factor: 1.5,
        ..LuOptions::default()
    };
    let mut f = LuF64::new(opts);
    fz(&mut f, &basis, &a).unwrap();
    let mut seen = false;
    for q in 0..m {
        let cand = ColumnRef::Structural(Col::from_usize(q));
        if basis.contains(&cand) {
            continue;
        }
        let (r, out) = exchange(&mut f, &mut rng, &a, cand);
        basis[r] = cand;
        if out.unwrap() == UpdateOutcome::RefactorNeeded(RefactorReason::FillGrowth) {
            seen = true;
            break;
        }
    }
    assert!(seen);
    let b = dense_basis(&basis, &a);
    let rhs = rng.dense_ints(m);
    assert!(rel_err(&ftran_vec(&mut f, &rhs), &dense_solve(&b, &rhs).unwrap()) < 1e-10);
}

/// The growth trigger measures the growth added by the updates (Chapter
/// 5.5): a factor whose fresh factorization is already above the limit
/// does not ask for a refactorization at every update (the loop that
/// stalled MAROS-R7), while updates that blow `U` up still do.
#[test]
fn growth_is_measured_from_the_fresh_factorization() {
    let m = 3;
    // Columns 0 and 1 form the bump [[1, 1], [1, −1]] (no singletons):
    // eliminating it gives u = −2, a fresh growth of 2 over max |b| = 1.
    let a = CscMatrix::from_triplets(
        m,
        5,
        &[
            (Row(0), Col(0), 1.0),
            (Row(1), Col(0), 1.0),
            (Row(0), Col(1), 1.0),
            (Row(1), Col(1), -1.0),
            (Row(2), Col(2), 1.0),
            (Row(0), Col(3), 0.5),
            (Row(2), Col(3), 1.0),
            (Row(0), Col(4), 1e6),
            (Row(2), Col(4), 1.0),
        ],
    )
    .unwrap();
    let structural = |j: usize| ColumnRef::Structural(Col::from_usize(j));
    let basis = vec![structural(0), structural(1), structural(2)];
    // A limit of 1.5 flags the fresh factor, as an unscaled MAROS-R7 basis
    // is flagged at 1e10.
    let opts = LuOptions {
        growth_limit: 1.5,
        fill_factor: 1e6,
        ..LuOptions::default()
    };
    let mut f = LuF64::new(opts);
    fz(&mut f, &basis, &a).unwrap();
    let s = *f.stats();
    assert!(s.growth_exceeded, "fresh growth {}", s.growth);
    assert!(approx_eq(s.growth, 2.0, 1e-12));
    assert!(le(f.update_stats().update_growth, 1.0, 0.0));
    // An exchange that does not grow U: column 3 replaces column 2.
    let mut v = SparseVec::new(m);
    let (rows, vals) = a.col(Col(3));
    v.scatter(1.0, rows, vals);
    f.ftran_for_update(&mut v);
    let r = 2;
    let out = f.update(r, v.get(2)).unwrap();
    assert_eq!(out, UpdateOutcome::Updated, "stats {:?}", f.stats());
    // An exchange that grows U by 5e5 relative to the fresh factor: the
    // default limit accepts it, a limit of 1e3 flags it.
    for (limit, want) in [
        (1e10, UpdateOutcome::Updated),
        (1e3, UpdateOutcome::RefactorNeeded(RefactorReason::Growth)),
    ] {
        let opts = LuOptions {
            growth_limit: limit,
            fill_factor: 1e6,
            ..LuOptions::default()
        };
        let mut f = LuF64::new(opts);
        fz(&mut f, &basis, &a).unwrap();
        assert!(!f.stats().growth_exceeded);
        let mut v = SparseVec::new(m);
        let (rows, vals) = a.col(Col(4));
        v.scatter(1.0, rows, vals);
        f.ftran_for_update(&mut v);
        let r = 2;
        let out = f.update(r, v.get(2)).unwrap();
        assert_eq!(out, want, "limit {limit}: stats {:?}", f.stats());
        assert!(lt(1e5, f.update_stats().update_growth));
    }
}

// ---------------------------------------------------------------------------
// P1.4 — hypersparse solves (Chapter 5.4).
// ---------------------------------------------------------------------------

/// Very sparse bases: the reach of a sparse right-hand side is a small part
/// of `0..m`.
const HYPER: Shape = Shape {
    lower: 3,
    upper: 2,
    logical: 50,
    dominant: true,
};

/// Options forcing the hypersparse algorithms (for a right-hand side in
/// sparse mode) or the dense ones.
fn mode(opts: LuOptions, hyper: bool) -> LuOptions {
    let d = if hyper { 2.0 } else { -1.0 };
    LuOptions {
        hyper_rhs_density: d,
        hyper_result_density: d,
        ..opts
    }
}

/// A right-hand side in sparse mode with `k` random nonzeros (and, now and
/// then, an explicit zero in the pattern).
fn sparse_rhs(rng: &mut Rng, m: usize, k: usize) -> SparseVec {
    let mut v = SparseVec::new(m);
    for _ in 0..k {
        let i = u32::try_from(rng.below(m)).unwrap();
        let x = if rng.chance(1, 10) {
            0.0
        } else {
            rng.nonzero(5)
        };
        v.set(i, x);
    }
    v
}

/// One solve of each kind in mode `hyper`; returns the dense values and
/// checks the invariants of the result.
fn solve_in(f: &mut LuF64, v: &SparseVec, hyper: bool, transposed: bool) -> Vec<f64> {
    let opts = *f.options();
    f.set_options(mode(opts, hyper));
    let mut w = v.clone();
    if transposed {
        f.btran(&mut w);
    } else {
        f.ftran(&mut w);
    }
    f.set_options(opts);
    assert!(w.is_sparse());
    assert!(w.check_invariants().is_ok(), "{:?}", w.check_invariants());
    w.values().to_vec()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(64)))]

    /// P1.4: on fresh and updated factors of sparse and dense bases, the
    /// hypersparse FTRAN and BTRAN (Algorithms HF, HB) agree with the dense
    /// ones (Algorithms F, B) within `1e-12` relative (times `growth/100`
    /// when the growth factor exceeds 100, and times 10 on the 70%-dense
    /// bases) or, when that fixed tolerance is exceeded, within the
    /// first-order bound `2 m ε κ(B)` of a reordering of the same roundings
    /// (`κ` Skeel's condition number of the current basis; D-0087
    /// pendency (i): `m = 56`, `κ ≈ 1.2·10⁴`, error `1.2·10⁻¹²`), for
    /// right-hand sides
    /// of every density; the result is in sparse mode with a pattern that
    /// covers its support. The updates themselves record their spike with
    /// a randomly chosen algorithm, and the updated factor still agrees
    /// with a fresh factorization and the dense oracle.
    #[test]
    fn hypersparse_solves_match_dense(
        m in 1_usize..120,
        kind in 0_usize..5,
        k in 0_usize..30,
        seed in any::<u64>(),
    ) {
        let mut rng = Rng::new(seed);
        let shape = [HYPER, SPARSE, QUASI_TRIANGULAR, SLACK, DENSE][kind];
        let (a, mut basis) = gen_basis(&mut rng, m, shape, m);
        let opts = LuOptions { refactor_interval: 1000, fill_factor: 1e6, ..LuOptions::default() };
        let mut f = LuF64::new(opts);
        fz(&mut f, &basis, &a).unwrap();
        for _ in 0..=k {
            // The two algorithms differ only in the order of the roundings,
            // which the growth of the updated factors (and, on the 70%-dense
            // bases, the conditioning) amplifies; both stay equally close to
            // the dense oracle.
            let dense_shape = if kind == 4 { 10.0 } else { 1.0 };
            let tol_eq = 1e-12 * (f.stats().growth / 100.0).max(1.0) * dense_shape;
            // Solves of every density, both directions.
            for nnz in [1, 2, 1 + m / 20, 1 + m / 4, m] {
                let v = sparse_rhs(&mut rng, m, nnz);
                for transposed in [false, true] {
                    let xh = solve_in(&mut f, &v, true, transposed);
                    let xd = solve_in(&mut f, &v, false, transposed);
                    let err = rel_err(&xh, &xd);
                    // Above the fixed tolerance, the bound that follows the
                    // conditioning of the current basis decides (computed
                    // only then: it costs a dense inverse).
                    let tol = if le(err, tol_eq, 0.0) {
                        tol_eq
                    } else {
                        let kappa = skeel_condition(&dense_basis(&basis, &a));
                        tol_eq.max(2.0 * count(m) * f64::EPSILON * kappa)
                    };
                    prop_assert!(le(err, tol, 0.0),
                        "transposed {transposed}, nnz {nnz}: error {err} > {tol}");
                }
            }
            // One exchange with the spike recorded by either algorithm.
            let cand = if rng.chance(3, 4) {
                ColumnRef::Structural(Col::from_usize(a.n_cols() - m + rng.below(m)))
            } else {
                ColumnRef::Logical(Row::from_usize(rng.below(m)), Sign::Plus)
            };
            if basis.iter().any(|c| match (c, &cand) {
                (ColumnRef::Logical(i, _), ColumnRef::Logical(j, _)) => i == j,
                _ => *c == cand,
            }) {
                continue;
            }
            let mut col = SparseVec::new(m);
            cand.for_each_entry(&a, |i, v| col.add(i, v));
            let hyper = rng.chance(1, 2);
            f.set_options(mode(opts, hyper));
            f.ftran_for_update(&mut col);
            f.set_options(opts);
            prop_assert!(col.check_invariants().is_ok());
            let alpha = col.values().to_vec();
            let amax = inf_norm(&alpha);
            let ok: Vec<usize> = (0..m)
                .filter(|&r| le(0.5 * amax, alpha[r].abs(), 0.0))
                .collect();
            let r = ok[rng.below(ok.len())];
            prop_assert_eq!(f.update(r, alpha[r]).unwrap(), UpdateOutcome::Updated);
            basis[r] = cand;
            prop_assert!(f.check_invariants().is_ok(), "{:?}", f.check_invariants());
        }
        // The updated factor against a fresh one and the oracle, both modes.
        let mut fresh = factor_ok(&basis, &a);
        let b = dense_basis(&basis, &a);
        let v = sparse_rhs(&mut rng, m, 1 + m / 10);
        let rhs = v.values().to_vec();
        let x_ref = dense_solve(&b, &rhs).unwrap();
        let y_ref = dense_solve(&transpose(&b), &rhs).unwrap();
        for hyper in [false, true] {
            let x = solve_in(&mut f, &v, hyper, false);
            let y = solve_in(&mut f, &v, hyper, true);
            prop_assert!(rel_err(&x, &x_ref) < 1e-9, "ftran vs oracle {}", rel_err(&x, &x_ref));
            prop_assert!(rel_err(&y, &y_ref) < 1e-9, "btran vs oracle {}", rel_err(&y, &y_ref));
            let xf = solve_in(&mut fresh, &v, hyper, false);
            prop_assert!(rel_err(&x, &xf) < 1e-9, "ftran vs fresh {}", rel_err(&x, &xf));
        }
    }
}

/// The alternation policy: sparse right-hand sides of a very sparse basis
/// go to the hypersparse algorithms, dense-mode ones to the dense
/// algorithms, and the statistics count both.
#[test]
fn alternation_policy_and_statistics() {
    let mut rng = Rng::new(3);
    let m = 400;
    let very_sparse = Shape {
        lower: 1,
        upper: 0,
        logical: 80,
        dominant: true,
    };
    let (a, basis) = gen_basis(&mut rng, m, very_sparse, 0);
    let mut f = factor_ok(&basis, &a);
    for _ in 0..10 {
        let mut v = sparse_rhs(&mut rng, m, 2);
        f.ftran(&mut v);
        let mut v = sparse_rhs(&mut rng, m, 2);
        f.btran(&mut v);
    }
    let s = *f.stats();
    assert_eq!((s.ftran.calls, s.btran.calls), (10, 10));
    assert_eq!((s.ftran.hyper_calls, s.btran.hyper_calls), (10, 10));
    assert!(s.ftran.mean_rhs_density() <= 2.0 / 400.0);
    assert!(s.ftran.mean_result_density() < 0.1 && s.btran.mean_result_density() < 0.1);
    // A right-hand side in dense mode always takes the dense path.
    let mut v = SparseVec::from_dense(&rng.dense_ints(m));
    v.make_dense();
    f.ftran(&mut v);
    assert_eq!(f.stats().ftran.hyper_calls, 10);
    assert!(approx_eq(
        f.stats().ftran.hyper_fraction(),
        10.0 / 11.0,
        1e-15
    ));
    // The statistics survive a refactorization.
    fz(&mut f, &basis, &a).unwrap();
    assert_eq!(f.stats().ftran.calls, 11);
    // Dense results push the expected density over the threshold.
    let (a2, basis2) = gen_basis(&mut rng, 50, DENSE, 0);
    let mut g = factor_ok(&basis2, &a2);
    for _ in 0..10 {
        let mut v = sparse_rhs(&mut rng, 50, 1);
        g.ftran(&mut v);
    }
    let s = g.stats().ftran;
    assert!(s.hyper_calls < s.calls, "{s:?}");
    assert!(g.expected_densities()[0] > 0.1);
}

// ---------------------------------------------------------------------------
// L1.3 — the COLAMD pre-order of the bump (step L5′) and Algorithm R.
// ---------------------------------------------------------------------------

/// A factor with the COLAMD pre-order of the bump.
fn colamd_lu() -> LuF64 {
    let mut f = LuF64::default();
    f.set_pre_order(Some(PreOrder::Colamd));
    assert_eq!(f.pre_order(), Some(PreOrder::Colamd));
    f
}

/// `a` with every value multiplied by a factor in `[1 − ρ, 1 + ρ]`: the
/// same pattern, new values.
fn perturbed(rng: &mut Rng, a: &CscMatrix, rho: f64) -> CscMatrix {
    let mut b = a.clone();
    for v in &mut b.val {
        let t = f64::from(u32::try_from(rng.below(2001)).unwrap()) / 1000.0 - 1.0;
        *v *= 1.0 + rho * t;
    }
    b
}

/// The factor is valid and agrees with the dense oracle.
fn assert_factor_matches(f: &mut LuF64, basis: &[ColumnRef], a: &CscMatrix, rng: &mut Rng) {
    assert!(f.check_invariants().is_ok(), "{:?}", f.check_invariants());
    let b = dense_basis(basis, a);
    let res = lu_residual(f, &b);
    assert!(res < 1e-14, "residual {res}");
    let rhs = rng.dense_ints(basis.len());
    let x = ftran_vec(f, &rhs);
    let x_ref = dense_solve(&b, &rhs).unwrap();
    assert!(
        rel_err(&x, &x_ref) < 1e-9,
        "ftran error {}",
        rel_err(&x, &x_ref)
    );
    let y = btran_vec(f, &rhs);
    let y_ref = dense_solve(&transpose(&b), &rhs).unwrap();
    assert!(
        rel_err(&y, &y_ref) < 1e-9,
        "btran error {}",
        rel_err(&y, &y_ref)
    );
}

#[test]
fn pre_order_defaults_to_none() {
    let f = LuF64::default();
    assert_eq!(f.pre_order(), None);
    assert_eq!(f.refactor_pivot_tol(), None);
    assert!(f.symbolic().is_none());
    assert_eq!(f.stats().symbolic_reuses + f.stats().symbolic_fallbacks, 0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(96)))]

    /// Step L5′: with the COLAMD pre-order the factorization satisfies
    /// `L U = P B Q` and solves like the dense oracle; its pivot sequence
    /// is recorded (invariant (R3)).
    #[test]
    fn preordered_factorization_is_valid(m in 1_usize..40, kind in 0_usize..4, seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let (a, basis) = gen_basis(&mut rng, m, shape_of(kind), 0);
        let mut f = colamd_lu();
        fz(&mut f, &basis, &a).unwrap();
        assert_factor_matches(&mut f, &basis, &a, &mut rng);
        let s = *f.stats();
        prop_assert_eq!(s.col_singletons + s.row_singletons + s.markowitz_pivots, m);
        let sym = f.symbolic().unwrap().clone();
        prop_assert!(crate::Perm::from_fwd(sym.pivot_rows.clone()).is_ok());
        prop_assert!(crate::Perm::from_fwd(sym.pivot_cols.clone()).is_ok());
        prop_assert_eq!(sym.singletons, s.col_singletons + s.row_singletons);
        prop_assert_eq!((sym.nnz_l, sym.nnz_u), (s.nnz_l, s.nnz_u));
    }

    /// Algorithm R: the same values replay the sequence (never a
    /// fallback); new values on the same pattern replay it or fall back,
    /// and in every case the factor is valid and solves correctly;
    /// Forrest–Tomlin updates work on a refactorized factor.
    #[test]
    fn refactorization_reuses_the_sequence(m in 1_usize..40, kind in 0_usize..4, pre in any::<bool>(),
                                            rho in 0.0_f64..0.5, loose in any::<bool>(), seed in any::<u64>()) {
        let mut rng = Rng::new(seed);
        let (a, basis) = gen_basis(&mut rng, m, shape_of(kind), 4);
        let mut f = if pre { colamd_lu() } else { LuF64::default() };
        // A looser replay threshold (step R5) keeps every factor valid.
        if loose {
            f.set_refactor_pivot_tol(Some(1e-3));
        }
        fz(&mut f, &basis, &a).unwrap();
        let sym = f.symbolic().unwrap().clone();
        f.refactorize_same_pattern(&(&basis[..], &a)).unwrap();
        prop_assert_eq!((f.stats().symbolic_reuses, f.stats().symbolic_fallbacks), (1, 0));
        prop_assert_eq!(f.symbolic().unwrap().pivot_rows.clone(), sym.pivot_rows.clone());
        prop_assert_eq!(f.symbolic().unwrap().pivot_cols.clone(), sym.pivot_cols.clone());
        prop_assert_eq!(f.stats().factorizations, 2);
        assert_factor_matches(&mut f, &basis, &a, &mut rng);

        let a2 = perturbed(&mut rng, &a, rho);
        f.refactorize_same_pattern(&(&basis[..], &a2)).unwrap();
        let s = *f.stats();
        prop_assert_eq!(s.symbolic_reuses + s.symbolic_fallbacks, 2);
        assert_factor_matches(&mut f, &basis, &a2, &mut rng);

        // Updates on top of a replayed factor.
        let n = a2.n_cols();
        for step in 0..3.min(m) {
            let q = ColumnRef::Structural(Col::from_usize(n - 1 - step % 4));
            let (_, out) = exchange(&mut f, &mut rng, &a2, q);
            if !matches!(out, Ok(UpdateOutcome::Updated)) {
                break;
            }
            prop_assert!(f.check_invariants().is_ok());
        }
    }
}

/// Algorithm R falls back to Algorithm L without a sequence, with a
/// sequence of another order, and when a pivot fails the threshold test;
/// a singular basis is diagnosed as by `factorize` and drops the sequence.
#[test]
fn refactorization_falls_back() {
    let mut rng = Rng::new(17);
    let (a, basis) = gen_basis(&mut rng, 30, SPARSE, 0);
    let mut f = LuF64::default();
    // No sequence yet.
    f.refactorize_same_pattern(&(&basis[..], &a)).unwrap();
    assert_eq!(
        (f.stats().symbolic_reuses, f.stats().symbolic_fallbacks),
        (0, 1)
    );
    assert_factor_matches(&mut f, &basis, &a, &mut rng);
    // Another order.
    let (a3, basis3) = gen_basis(&mut rng, 12, DENSE, 0);
    f.refactorize_same_pattern(&(&basis3[..], &a3)).unwrap();
    assert_eq!(f.stats().symbolic_fallbacks, 2);
    assert_factor_matches(&mut f, &basis3, &a3, &mut rng);
    // A refused pivot: shrink the pivot entry of a bump step.
    let (a4, basis4) = gen_basis(&mut rng, 25, DENSE, 0);
    fz(&mut f, &basis4, &a4).unwrap();
    let sym = f.symbolic().unwrap().clone();
    assert!(sym.singletons < 25);
    let s = sym.singletons;
    let (p, k) = (sym.pivot_rows[s], sym.pivot_cols[s] as usize);
    let ColumnRef::Structural(j) = basis4[k] else {
        panic!("DENSE bases are structural");
    };
    let mut a5 = a4.clone();
    let r = a5.col_ptr[j.index()]..a5.col_ptr[j.index() + 1];
    let at = r
        .clone()
        .find(|&q| a5.row_idx[q] == p)
        .expect("the pivot is stored");
    a5.val[at] = 1e-9;
    f.refactorize_same_pattern(&(&basis4[..], &a5)).unwrap();
    assert_eq!(f.stats().symbolic_fallbacks, 3);
    assert_factor_matches(&mut f, &basis4, &a5, &mut rng);
    // A singular basis.
    let mut dep = basis4.clone();
    dep[1] = dep[0];
    let e = f.refactorize_same_pattern(&(&dep[..], &a4)).unwrap_err();
    assert!(matches!(e, LuError::Singular { .. }));
    assert!(f.symbolic().is_none() && !f.is_valid());
}

/// [`Lu::set_refactor_pivot_tol`] refuses a threshold outside `(0, 1]`.
#[test]
#[should_panic(expected = "outside (0, 1]")]
fn refactor_pivot_tol_is_checked() {
    LuF64::default().set_refactor_pivot_tol(Some(0.0));
}

// ---------------------------------------------------------------------------
// Generators of KKT systems (Chapters 5.15 and 5.17).
// ---------------------------------------------------------------------------

/// Random KKT systems for the tests of the backends (task L1.6a; the
/// generator of the `FaerLdl` tests of L1.5, shared).
pub(crate) mod kkt_gen {
    use crate::real::is_exact_zero;
    use crate::{Col, CscMatrix, KktForm, Row};

    /// xorshift in `[0, 1)`.
    pub(crate) struct Rng(pub(crate) u64);
    impl Rng {
        pub(crate) fn unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            #[allow(clippy::cast_precision_loss)]
            let u = (self.0 >> 11) as f64 / (1_u64 << 53) as f64;
            u
        }
    }

    /// A random sparse `m × n` matrix with a nonzero in every row (full
    /// row rank with probability one) and the KKT systems built from it
    /// with `D = 10^{±spread·u}`: the lower triangles of the augmented
    /// system and of the normal equations, and the augmented form.
    pub(crate) fn kkt(
        m: usize,
        n: usize,
        spread: f64,
        seed: u64,
    ) -> (CscMatrix, CscMatrix, KktForm) {
        let mut r = Rng(seed | 1);
        let mut a = Vec::new();
        for i in 0..m {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let j = (r.unit() * n as f64) as usize % n;
            a.push((i, j, 1.0 + r.unit()));
            for jj in 0..n {
                if jj != j && r.unit() < 0.2 {
                    a.push((i, jj, 2.0 * r.unit() - 1.0));
                }
            }
        }
        let d: Vec<f64> = (0..n)
            .map(|_| 10f64.powf(spread * (2.0 * r.unit() - 1.0)))
            .collect();
        let mut t = Vec::new();
        for (j, &dj) in d.iter().enumerate() {
            t.push((Row::from_usize(j), Col::from_usize(j), -dj));
        }
        for &(i, j, v) in &a {
            t.push((Row::from_usize(n + i), Col::from_usize(j), v));
        }
        let aug = CscMatrix::from_triplets(n + m, n + m, &t).unwrap();
        let mut dense = vec![vec![0.0; m]; m];
        for &(i, j, v) in &a {
            for &(p, q, w) in &a {
                if q == j && p <= i {
                    dense[i][p] += v * d[j] * w;
                }
            }
        }
        let mut tn = Vec::new();
        for (i, row) in dense.iter().enumerate() {
            for (p, &v) in row.iter().enumerate() {
                if !is_exact_zero(v) {
                    tn.push((Row::from_usize(i), Col::from_usize(p), v));
                }
            }
        }
        let normal = CscMatrix::from_triplets(m, m, &tn).unwrap();
        (
            aug,
            normal,
            KktForm::Augmented {
                n_primal: n,
                n_dual: m,
            },
        )
    }

    /// A random vector in `[-1, 1)^n`.
    pub(crate) fn rhs(n: usize, seed: u64) -> Vec<f64> {
        let mut r = Rng(seed | 3);
        (0..n).map(|_| 2.0 * r.unit() - 1.0).collect()
    }
}
