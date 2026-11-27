//! Tests of the indefinite form (issue #8; Chapter 5.14, "The indefinite
//! form", and invariants (S6′), (L11), (F3)).
//!
//! Oracles, from the strongest:
//! 1. (L11): with `Regularization::none()`, the factor of
//!    `KktForm::Indefinite` is the factor of `Augmented { n_primal:
//!    i.negative, n_dual: i.positive }` bit for bit (`i` the inertia found);
//!    every stored array of `SupernodalLdl` is compared. (F3): `FaerLdl`
//!    refuses the form (task I08), checked wherever it was an oracle.
//! 2. Exact inertia by construction: `K = P L₀ Δ₀ L₀ᵀ Pᵀ` with `L₀` unit
//!    lower of small integers and `Δ₀` of `±1`, `±2` and `[0 1; 1 0]`
//!    blocks — every entry an exact integer in `f64`, the inertia that of
//!    `Δ₀` (Sylvester); and quasi-definite `[−H, Aᵀ; A, G]`, symmetrically
//!    permuted, of inertia `(m, n, 0)` (Vanderbei 1995).
//! 3. Dense eigenvalues (cyclic Jacobi, written here) of small random
//!    symmetric matrices, when no eigenvalue is near zero.

use proptest::prelude::*;

use crate::kkt::{first_zero_pivot, sym_lower_matvec};
use crate::real::{is_exact_zero, le, lt};
use crate::tests::kkt_gen::{Rng, kkt, rhs};
use crate::{
    Amd, Col, CscMatrix, DenseKernel, Inertia, KktError, KktForm, KktSolver, NaiveDense,
    NumericLdl, Refinement, Regularization, Row, SparseError, SupernodalLdl, SupernodalOptions,
};

/// The option sets of the `SupernodalLdl` tests: the defaults, every
/// supernode fundamental and on the kernel path, relaxed on the kernel
/// path (as `supernodal::tests::variants`).
fn variants() -> [SupernodalOptions; 3] {
    [
        SupernodalOptions::default(),
        SupernodalOptions {
            relax_zeros: 0,
            min_supernode_cols: 1,
            supernodal_switch: 0.0,
            ..SupernodalOptions::default()
        },
        SupernodalOptions {
            relax_zeros: 4,
            min_supernode_cols: 1,
            supernodal_switch: 0.0,
            ..SupernodalOptions::default()
        },
    ]
}

/// The lower triangle of a dense symmetric matrix (exact zeros dropped).
fn lower(dense: &[Vec<f64>]) -> CscMatrix {
    let n = dense.len();
    let mut t = Vec::new();
    for (i, row) in dense.iter().enumerate() {
        for (j, &v) in row.iter().enumerate().take(i + 1) {
            if !is_exact_zero(v) {
                t.push((Row::from_usize(i), Col::from_usize(j), v));
            }
        }
    }
    CscMatrix::from_triplets(n, n, &t).unwrap()
}

/// `0..n` shuffled (Fisher–Yates).
fn shuffle(n: usize, r: &mut Rng) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let j = (r.unit() * (i + 1) as f64) as usize % (i + 1);
        p.swap(i, j);
    }
    p
}

/// `P M Pᵀ` for a dense `M` and a permutation `p` (`(PMPᵀ)_{ij} = m_{p_i p_j}`).
fn permute(m: &[Vec<f64>], p: &[usize]) -> Vec<Vec<f64>> {
    p.iter()
        .map(|&pi| p.iter().map(|&pj| m[pi][pj]).collect())
        .collect()
}

/// A small integer in `{−2, −1, 1, 2}`.
fn small(r: &mut Rng) -> i64 {
    [-2, -1, 1, 2][usize::from(lt(0.5, r.unit())) + 2 * usize::from(lt(0.5, r.unit()))]
}

/// Oracle 2: `K = P L₀ Δ₀ L₀ᵀ Pᵀ` of order `n` with exact integer entries
/// and its inertia, that of `Δ₀`. With `singular`, one `1×1` block of `Δ₀`
/// is `0` (then `K` is singular and the inertia has `zero = 1`).
fn ldl_product(n: usize, density: f64, singular: bool, seed: u64) -> (CscMatrix, Inertia) {
    let mut r = Rng(seed | 1);
    let mut delta = vec![vec![0_i64; n]; n];
    let mut inertia = Inertia::default();
    // The blocks of Δ₀: `true` for a 2×2 block starting there.
    let mut starts = Vec::new();
    let mut k = 0;
    while k < n {
        let pair = k + 1 < n && lt(r.unit(), 0.3);
        starts.push((k, pair));
        k += if pair { 2 } else { 1 };
    }
    if singular && starts.iter().all(|&(_, pair)| pair) {
        // Only 2×2 blocks: split the last one into two 1×1 blocks.
        let (k, _) = starts.pop().expect("n ≥ 1");
        starts.push((k, false));
        starts.push((k + 1, false));
    }
    let mut zero_pending = singular;
    for &(k, pair) in &starts {
        if pair {
            // [0 1; 1 0]: one eigenvalue of each sign.
            delta[k][k + 1] = 1;
            delta[k + 1][k] = 1;
            inertia.positive += 1;
            inertia.negative += 1;
        } else if zero_pending {
            zero_pending = false;
            inertia.zero += 1;
        } else {
            let d = small(&mut r);
            delta[k][k] = d;
            if d > 0 {
                inertia.positive += 1;
            } else {
                inertia.negative += 1;
            }
        }
    }
    let mut l = vec![vec![0_i64; n]; n];
    for (i, row) in l.iter_mut().enumerate() {
        row[i] = 1;
        for x in row.iter_mut().take(i) {
            if lt(r.unit(), density) {
                *x = small(&mut r);
            }
        }
    }
    // K = L Δ Lᵀ in exact integer arithmetic.
    let mut ld = vec![vec![0_i64; n]; n];
    for i in 0..n {
        for j in 0..n {
            ld[i][j] = (0..n).map(|q| l[i][q] * delta[q][j]).sum();
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let kd: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let v: i64 = (0..n).map(|q| ld[i][q] * l[j][q]).sum();
                    assert!(v.abs() < 1 << 50, "exact in f64");
                    v as f64
                })
                .collect()
        })
        .collect();
    let p = shuffle(n, &mut r);
    (lower(&permute(&kd, &p)), inertia)
}

/// Oracle 2, quasi-definite: `[−H, Aᵀ; A, G]` with `H`, `G` positive
/// diagonal and `A` random sparse, symmetrically permuted (so that it is
/// not in the primal-first order `Augmented` needs); inertia `(m, n, 0)`.
fn quasi_definite(n: usize, m: usize, seed: u64) -> (CscMatrix, Inertia) {
    let mut r = Rng(seed | 5);
    let nn = n + m;
    let mut kd = vec![vec![0.0; nn]; nn];
    for (j, row) in kd.iter_mut().enumerate() {
        let d = 0.25 + 4.0 * r.unit();
        row[j] = if j < n { -d } else { d };
    }
    for i in 0..m {
        for j in 0..n {
            if lt(r.unit(), 0.3) {
                let v = 2.0 * r.unit() - 1.0;
                kd[n + i][j] = v;
                kd[j][n + i] = v;
            }
        }
    }
    let p = shuffle(nn, &mut r);
    (
        lower(&permute(&kd, &p)),
        Inertia {
            positive: m,
            negative: n,
            zero: 0,
        },
    )
}

/// Every stored array of two `SupernodalLdl` factors, bit for bit.
fn same_ldl(a: &NumericLdl<f64>, b: &NumericLdl<f64>) -> bool {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    a.inertia == b.inertia
        && a.is_pivoted() == b.is_pivoted()
        && a.delayed() == b.delayed()
        && a.perm() == b.perm()
        && a.supernodes() == b.supernodes()
        && (0..a.n_supernodes()).all(|s| a.rows(s) == b.rows(s))
        && a.pivot_2x2 == b.pivot_2x2
        && bits(&a.diag) == bits(&b.diag)
        && bits(a.scale()) == bits(b.scale())
        && a.blocks.len() == b.blocks.len()
        && a.blocks
            .iter()
            .zip(&b.blocks)
            .all(|(x, y)| x.rows == y.rows && x.cols == y.cols && bits(&x.data) == bits(&y.data))
}

/// Factorizes `k` with `form` and `Regularization::none()`.
fn factor<S: KktSolver<f64>>(
    s: &S,
    k: &CscMatrix,
    form: KktForm,
) -> Result<(S::Numeric, Inertia), KktError> {
    let sym = s.analyze(&k.pattern(), form, &Amd::default())?;
    s.factorize(&sym, k, Regularization::none())
}

/// The solves of two factors agree bit for bit on two right-hand sides.
fn same_solves<S: KktSolver<f64>>(s: &S, a: &S::Numeric, b: &S::Numeric, n: usize) -> bool {
    (0..2_u64).all(|t| {
        let mut x = rhs(n, 17 + t);
        let mut y = x.clone();
        s.solve(a, &mut x);
        s.solve(b, &mut y);
        x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits())
    })
}

/// What the indefinite form may do (S6′): succeed with `zero = 0` and the
/// order counted, or fail with a zero pivot or an overflow — never
/// `WrongInertia`, never a structural or backend error on a valid `K`.
fn check_s6_prime(r: &Result<Inertia, KktError>, n: usize) -> Result<(), String> {
    match r {
        Ok(i) if i.zero == 0 && i.positive + i.negative == n => Ok(()),
        Err(KktError::ZeroPivot { position } | KktError::NotFinite { position })
            if *position < n =>
        {
            Ok(())
        }
        other => Err(format!("(S6′) violated: {other:?}")),
    }
}

/// Oracle 1 on `SupernodalLdl`, plus (S6′) and the refined residual;
/// returns the indefinite outcome.
fn oracle_supernodal<K: DenseKernel<f64>>(
    s: &SupernodalLdl<f64, K>,
    k: &CscMatrix,
) -> Result<Result<Inertia, KktError>, String> {
    let n = k.n_cols();
    let ind = factor(s, k, KktForm::Indefinite);
    let out = ind.as_ref().map(|(_, i)| *i).map_err(Clone::clone);
    check_s6_prime(&out, n)?;
    if let Ok((num, i)) = &ind {
        if !num.is_pivoted() {
            return Err("Indefinite did not take Algorithm M".into());
        }
        let aug = KktForm::Augmented {
            n_primal: i.negative,
            n_dual: i.positive,
        };
        let (num_a, ia) = factor(s, k, aug).map_err(|e| format!("(L11): Augmented failed: {e}"))?;
        if ia != *i || !same_ldl(num, &num_a) || !same_solves(s, num, &num_a, n) {
            return Err("(L11): the factors differ".into());
        }
        let x_star = rhs(n, 99);
        let mut b = vec![0.0; n];
        sym_lower_matvec(k, &x_star, &mut b);
        let mut x = vec![0.0; n];
        let rep = s.solve_refined(num, k, &b, &mut x, Refinement::default());
        if !le(rep.residual, 1e-10, 0.0) {
            return Err(format!("refined residual {:e}", rep.residual));
        }
    }
    Ok(out)
}

/// (F3): `FaerLdl` refuses the indefinite form with `KktError::Backend` on
/// every structure (task I08).
#[cfg(feature = "faer")]
fn faer_refuses_indefinite(k: &CscMatrix) -> Result<(), String> {
    for supernodal in [Some(true), Some(false), None] {
        let s = crate::FaerLdl { supernodal };
        match factor(&s, k, KktForm::Indefinite) {
            Err(KktError::Backend(_)) => {}
            other => {
                return Err(format!(
                    "(F3): FaerLdl {{ supernodal: {supernodal:?} }} did not refuse: {:?}",
                    other.map(|(_, i)| i)
                ));
            }
        }
    }
    Ok(())
}

/// Oracle 3: the eigenvalues of a dense symmetric matrix by cyclic Jacobi
/// rotations (Golub & Van Loan, Algorithm 8.5.3), to convergence of the
/// off-diagonal mass.
fn jacobi_eigenvalues(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let all: f64 = a.iter().flatten().map(|x| x * x).sum();
        if le(off, 1e-30 * all, 0.0) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if is_exact_zero(a[p][q]) {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in &mut a {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
                for j in 0..n {
                    let (x, y) = (a[p][j], a[q][j]);
                    a[p][j] = c * x - s * y;
                    a[q][j] = s * x + c * y;
                }
            }
        }
    }
    (0..n).map(|i| a[i][i]).collect()
}

/// A random dense symmetric matrix of order `n`: entries in `[−1, 1)` with
/// probability `density`, diagonal entries absent with probability 0.3.
fn random_symmetric(n: usize, density: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut r = Rng(seed | 7);
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let keep = if i == j {
                !lt(r.unit(), 0.3)
            } else {
                lt(r.unit(), density)
            };
            if keep {
                let v = 2.0 * r.unit() - 1.0;
                a[i][j] = v;
                a[j][i] = v;
            }
        }
    }
    a
}

#[test]
fn expected_inertia_of_every_form() {
    assert_eq!(
        KktForm::NormalEquations.expected_inertia(4),
        Some(Inertia {
            positive: 4,
            negative: 0,
            zero: 0
        })
    );
    assert_eq!(
        KktForm::Augmented {
            n_primal: 3,
            n_dual: 2
        }
        .expected_inertia(5),
        Some(Inertia {
            positive: 2,
            negative: 3,
            zero: 0
        })
    );
    for n in [0, 1, 7] {
        assert_eq!(KktForm::Indefinite.expected_inertia(n), None);
    }
}

#[test]
fn first_zero_pivot_finds_ones_and_twos() {
    assert_eq!(first_zero_pivot(&[1.0, -2.0, 3.0], |_| None), None);
    assert_eq!(first_zero_pivot(&[1.0, 0.0, 0.0], |_| None), Some(1));
    // [1 1; 1 1] at (1, 2) is singular; [0 1; 1 0] at (0, 1) is not.
    let pairs = |c: usize| match c {
        0 | 2 => Some(1.0),
        _ => None,
    };
    assert_eq!(first_zero_pivot(&[0.0, 0.0, 1.0, 1.0], pairs), Some(2));
    assert_eq!(first_zero_pivot(&[0.0, 0.0, 1.0, 2.0], pairs), None);
}

#[test]
fn indefinite_refuses_any_regularization() {
    let (k, _) = ldl_product(6, 0.3, false, 3);
    let reg_cases = [
        Regularization::default(),
        Regularization {
            primal: 1e-8,
            ..Regularization::none()
        },
        Regularization {
            dual: 1e-8,
            ..Regularization::none()
        },
        Regularization {
            pivot_tol: Some(1e-13),
            bump: 1e-7,
            ..Regularization::none()
        },
    ];
    let s = SupernodalLdl::<f64>::default();
    let sym = s
        .analyze(&k.pattern(), KktForm::Indefinite, &Amd::default())
        .unwrap();
    for reg in reg_cases {
        let e = s.factorize(&sym, &k, reg).err();
        assert!(
            matches!(
                e,
                Some(KktError::Structure(SparseError::InvalidValue { .. }))
            ),
            "{reg:?}: {e:?}"
        );
    }
    #[cfg(feature = "faer")]
    {
        let f = crate::FaerLdl {
            supernodal: Some(true),
        };
        let sym = f
            .analyze(&k.pattern(), KktForm::Indefinite, &Amd::default())
            .unwrap();
        for reg in reg_cases {
            assert!(matches!(
                f.factorize(&sym, &k, reg).err(),
                Some(KktError::Structure(SparseError::InvalidValue { .. }))
            ));
        }
    }
    // The bump and the scale are not read without `τ`.
    let reg = Regularization {
        bump: 1.0,
        scale: crate::ShiftScale::Absolute,
        ..Regularization::none()
    };
    assert!(s.factorize(&sym, &k, reg).is_ok());
}

/// The singular cases of step K4 and M7, on every backend.
#[test]
fn singular_matrices_are_zero_pivots() {
    // [1 1; 1 1]: numerically singular, full structural rank (M3 at the root).
    let ones = lower(&[vec![1.0, 1.0], vec![1.0, 1.0]]);
    // diag(1, 2, 3) − 2I: the middle entry becomes an exact zero and is not
    // stored, so K is structurally singular (M7) at row 1.
    let shifted = lower(&[
        vec![-1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    // An exact zero pivot inside an otherwise coupled matrix: L₀ Δ₀ L₀ᵀ
    // with a zero block.
    for opts in variants() {
        let s = SupernodalLdl::<f64, NaiveDense>::new(NaiveDense, opts);
        assert!(matches!(
            factor(&s, &ones, KktForm::Indefinite).err(),
            Some(KktError::ZeroPivot { .. })
        ));
        assert_eq!(
            factor(&s, &shifted, KktForm::Indefinite).err(),
            Some(KktError::ZeroPivot { position: 1 })
        );
    }
    #[cfg(feature = "faer")]
    {
        assert_eq!(faer_refuses_indefinite(&ones), Ok(()));
        assert_eq!(faer_refuses_indefinite(&shifted), Ok(()));
    }
}

/// `FaerLdl` refuses the indefinite form on every structure, before
/// factorizing (task I08; on a simplicial one since D-0094) — even on a
/// quasi-definite matrix, which its intranodal Bunch–Kaufman would
/// factorize in any order.
#[cfg(feature = "faer")]
#[test]
fn faer_ldl_refuses_the_indefinite_form() {
    let (k, _) = quasi_definite(6, 4, 11);
    for supernodal in [Some(false), Some(true), None] {
        let f = crate::FaerLdl { supernodal };
        assert!(matches!(
            factor(&f, &k, KktForm::Indefinite).err(),
            Some(KktError::Backend(_))
        ));
    }
}

#[test]
fn the_empty_matrix_has_the_empty_inertia() {
    let k = CscMatrix::from_triplets(0, 0, &[]).unwrap();
    let s = SupernodalLdl::<f64>::default();
    assert_eq!(
        factor(&s, &k, KktForm::Indefinite).map(|(_, i)| i),
        Ok(Inertia::default())
    );
}

/// The doc example of `KktForm::Indefinite` on every backend, with the
/// solve checked too.
#[test]
fn shift_invert_example_counts_the_eigenvalues_below_the_shift() {
    // A = [1 .5 0; .5 2 0; 0 0 3], σ = 2.5: the eigenvalues of A − σI are
    // −1 ± √0.5 (both negative) and 0.5.
    let k = lower(&[
        vec![-1.5, 0.5, 0.0],
        vec![0.5, -0.5, 0.0],
        vec![0.0, 0.0, 0.5],
    ]);
    let want = Inertia {
        positive: 1,
        negative: 2,
        zero: 0,
    };
    for opts in variants() {
        let s = SupernodalLdl::<f64, NaiveDense>::new(NaiveDense, opts);
        assert_eq!(oracle_supernodal(&s, &k), Ok(Ok(want)));
    }
    #[cfg(feature = "faer")]
    assert_eq!(faer_refuses_indefinite(&k), Ok(()));
}

/// The unregularized augmented systems of the existing tests, as
/// `Indefinite`: where `Augmented` succeeds, the same factor (L11); where it
/// fails with `WrongInertia { found }`, `found` (or the M7 zero pivot);
/// where it fails with a zero pivot, the same error.
#[test]
fn indefinite_agrees_with_augmented_on_the_kkt_generator() {
    let mut ok = 0;
    for seed in 0..60_u64 {
        for (m, n) in [(3, 5), (6, 10), (10, 12), (12, 30)] {
            let (aug, _, form) = kkt(m, n, 2.0, seed);
            for opts in variants() {
                let s = SupernodalLdl::<f64, NaiveDense>::new(NaiveDense, opts);
                let ind = oracle_supernodal(&s, &aug).unwrap();
                let decl = factor(&s, &aug, form).map(|(_, i)| i);
                match (&decl, &ind) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(a, b);
                        ok += 1;
                    }
                    (Err(KktError::WrongInertia { found, .. }), r) => {
                        assert!(
                            *r == Ok(*found) || matches!(r, Err(KktError::ZeroPivot { .. })),
                            "seed {seed}: {decl:?} vs {r:?}"
                        );
                    }
                    (Err(e), Err(f)) => assert_eq!(e, f, "seed {seed}"),
                    _ => panic!("seed {seed}: {decl:?} vs {ind:?}"),
                }
            }
            #[cfg(feature = "faer")]
            faer_refuses_indefinite(&aug).unwrap();
        }
    }
    assert!(ok > 500, "{ok}");
}

/// Oracle 3 on small random symmetric matrices, every backend.
#[test]
fn inertia_matches_dense_eigenvalues() {
    let (mut checked, mut skipped) = (0, 0);
    for seed in 0..400_u64 {
        #[allow(clippy::cast_possible_truncation)]
        let n = 1 + (seed % 8) as usize;
        let a = random_symmetric(n, 0.5, seed);
        let lam = jacobi_eigenvalues(a.clone());
        let big = lam.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let small = lam.iter().fold(f64::INFINITY, |m, x| m.min(x.abs()));
        let k = lower(&a);
        let separated = lt(1e-6 * big, small);
        let want = Inertia {
            positive: lam.iter().filter(|&&x| lt(0.0, x)).count(),
            negative: lam.iter().filter(|&&x| lt(x, 0.0)).count(),
            zero: 0,
        };
        let mut outcomes = Vec::new();
        for opts in variants() {
            let s = SupernodalLdl::<f64, NaiveDense>::new(NaiveDense, opts);
            outcomes.push(oracle_supernodal(&s, &k).unwrap());
        }
        #[cfg(feature = "faer")]
        outcomes.push(
            oracle_supernodal(&SupernodalLdl::<f64, crate::FaerDense>::default(), &k).unwrap(),
        );
        #[cfg(feature = "faer")]
        faer_refuses_indefinite(&k).unwrap();
        if separated {
            for o in &outcomes {
                assert_eq!(*o, Ok(want), "seed {seed}: {lam:?}");
            }
            checked += 1;
        } else {
            skipped += 1;
        }
    }
    assert!(checked > 250, "checked {checked}, skipped {skipped}");
}

/// The nonsingular `L₀ Δ₀ L₀ᵀ` (κ from 5·10² to 2·10⁶) that Algorithm M
/// with the `FaerDense` kernel refused in I07 (`ZeroPivot` or `NotFinite`
/// in release builds, with `Augmented` and the right inertia too).
const FAER_DENSE_REFUSALS: [(usize, f64, u64); 5] = [
    (24, 0.318_199_627_921_231_7, 2_927_192_055_965_926_386),
    (17, 0.232_310_126_225_435_43, 749_334_245_793_736_704),
    (19, 0.280_040_349_628_252_9, 995_591_194_406_609_536),
    (27, 0.121_355_513_803_929_24, 248_665_748_007_482_208),
    (11, 0.210_804_671_947_220_47, 293_172_923_017_713_792),
];

/// The `NaiveDense` kernel factorizes [`FAER_DENSE_REFUSALS`] with their
/// inertia, on every option set.
#[test]
fn faer_dense_refusals_factorize_with_the_naive_kernel() {
    for (n, density, seed) in FAER_DENSE_REFUSALS {
        let (k, inertia) = ldl_product(n, density, false, seed);
        for opts in variants() {
            let s = SupernodalLdl::<f64, NaiveDense>::new(NaiveDense, opts);
            assert_eq!(
                oracle_supernodal(&s, &k),
                Ok(Ok(inertia)),
                "n {n} seed {seed}"
            );
        }
    }
}

/// Reproducer of a defect of the `FaerDense` kernel on the `LBLᵀ` path,
/// found in I07 and not a matter of the indefinite form (`Augmented` with
/// the right inertia failed alike): on [`FAER_DENSE_REFUSALS`],
/// `FaerDense::sytrf` produced a NaN in the diagonal block (a debug build
/// panicked in the NaN assertion of `real::lt`, from `block_inertia`; a
/// release build reported `ZeroPivot` or `NotFinite` for these
/// nonsingular, well-conditioned matrices). The cause was faer's
/// Bunch–Kaufman test with `λ` and `σ` exchanged (Algorithm B′ of Chapter
/// 5.16, task I08); it now asserts what the `NaiveDense` kernel does, on
/// every option set and with `Augmented` of the right inertia too.
#[cfg(feature = "faer")]
#[test]
fn faer_dense_kernel_reproducer() {
    for (n, density, seed) in FAER_DENSE_REFUSALS {
        let (k, inertia) = ldl_product(n, density, false, seed);
        for opts in variants() {
            let s = SupernodalLdl::<f64, crate::FaerDense>::new(crate::FaerDense, opts);
            assert_eq!(
                oracle_supernodal(&s, &k),
                Ok(Ok(inertia)),
                "n {n} seed {seed}"
            );
            let aug = KktForm::Augmented {
                n_primal: inertia.negative,
                n_dual: inertia.positive,
            };
            assert_eq!(factor(&s, &k, aug).map(|(_, i)| i), Ok(inertia));
        }
    }
}

/// The smallest order among the matrices the `FaerDense` kernel refused
/// in the 3 000 `L₀ Δ₀ L₀ᵀ` of task I08 (`NotFinite` in a release build
/// before the fix: faer took a singular `2×2` pivot, Algorithm B′ of
/// Chapter 5.16): every option set gives its inertia, with `Indefinite`
/// and with `Augmented` of that inertia.
#[cfg(feature = "faer")]
#[test]
fn smallest_faer_dense_refusal_factorizes() {
    let (k, inertia) = ldl_product(
        8,
        0.232_845_212_516_157_03,
        false,
        6_738_647_053_308_660_863,
    );
    assert_eq!(
        inertia,
        Inertia {
            positive: 5,
            negative: 3,
            zero: 0
        }
    );
    for opts in variants() {
        let s = SupernodalLdl::<f64, crate::FaerDense>::new(crate::FaerDense, opts);
        assert_eq!(oracle_supernodal(&s, &k), Ok(Ok(inertia)));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// Oracle 2 (non-quasi-definite: zero diagonals, `2×2` blocks) and
    /// oracle 1, every backend.
    #[test]
    fn ldl_products_give_their_inertia(
        n in 1_usize..28,
        density in 0.0_f64..0.35,
        seed in any::<u64>(),
    ) {
        let (k, inertia) = ldl_product(n, density, false, seed);
        for opts in variants() {
            let s = SupernodalLdl::<f64, NaiveDense>::new(NaiveDense, opts);
            let r = oracle_supernodal(&s, &k).map_err(TestCaseError::fail)?;
            prop_assert_eq!(r, Ok(inertia));
        }
        #[cfg(feature = "faer")]
        {
            let s = SupernodalLdl::<f64, crate::FaerDense>::default();
            let r = oracle_supernodal(&s, &k).map_err(TestCaseError::fail)?;
            prop_assert_eq!(r, Ok(inertia));
            faer_refuses_indefinite(&k).map_err(TestCaseError::fail)?;
        }
    }

    /// Oracle 2 on permuted quasi-definite matrices, and oracle 1.
    #[test]
    fn quasi_definite_matrices_give_their_inertia(
        n in 1_usize..20,
        m in 0_usize..20,
        seed in any::<u64>(),
    ) {
        let (k, inertia) = quasi_definite(n, m, seed);
        for opts in variants() {
            let s = SupernodalLdl::<f64, NaiveDense>::new(NaiveDense, opts);
            let r = oracle_supernodal(&s, &k).map_err(TestCaseError::fail)?;
            prop_assert_eq!(r, Ok(inertia));
        }
        #[cfg(feature = "faer")]
        faer_refuses_indefinite(&k).map_err(TestCaseError::fail)?;
    }

    /// Exactly singular `L₀ Δ₀ L₀ᵀ`: (S6′) always — a zero pivot, or (the
    /// limit of step M3, "What the scale cannot do") a factor of a
    /// nonsingular neighbour, with its inertia; never `WrongInertia`, never
    /// a count with `zero > 0`.
    #[test]
    fn singular_ldl_products_keep_s6_prime(
        n in 1_usize..24,
        density in 0.0_f64..0.35,
        seed in any::<u64>(),
    ) {
        let (k, _) = ldl_product(n, density, true, seed);
        for opts in variants() {
            let s = SupernodalLdl::<f64, NaiveDense>::new(NaiveDense, opts);
            let r = factor(&s, &k, KktForm::Indefinite).map(|(_, i)| i);
            check_s6_prime(&r, n).map_err(TestCaseError::fail)?;
        }
        #[cfg(feature = "faer")]
        faer_refuses_indefinite(&k).map_err(TestCaseError::fail)?;
    }
}
