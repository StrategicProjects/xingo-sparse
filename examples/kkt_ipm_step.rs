//! The linear algebra of one interior-point iteration (Chapters 5.14–5.15):
//! assemble the augmented system `K = [−D  Aᵀ; A  δI]` of a toy LP with
//! `Triplets` and a `BlockPartition`, analyze its pattern once
//! (`FaerLdl::analyze`, ordered by our AMD), factorize with the default
//! regularization (Algorithm K), solve with refinement against the
//! *unregularized* `K` (Algorithm F, GMRES), and print the inertia, the
//! bumped pivots and the residual. Then a second factorization with the
//! `D` of a later iteration on the same symbolic phase — the pattern is
//! fixed along the method, only the values change.
//!
//! Needs the `faer` feature (on by default). Run with
//! `cargo run -p xingo-sparse --example kkt_ipm_step`.

use xingo_sparse::{
    Amd, BlockPartition, Col, CscMatrix, FaerLdl, FaerNumeric, FaerSymbolic, KktForm, KktSolver,
    Refinement, Regularization, Triplets,
};

/// The lower triangle of `K = [−D  Aᵀ; A  0]` for `A` of `m × n` and the
/// primal diagonal `d` (`δI` is the regularization's business, Algorithm
/// K). `part` says where the dual block starts.
fn assemble(a: &CscMatrix, d: &[f64], part: &BlockPartition) -> CscMatrix {
    let dual = part.row_block(1).start;
    let mut t = Triplets::new(part.n_rows(), part.n_cols());
    for (j, &dj) in d.iter().enumerate() {
        let jj = u32::try_from(j).expect("fits u32");
        t.push(jj, jj, -dj);
    }
    for j in 0..a.n_cols() {
        let (rows, vals) = a.col(Col::from_usize(j));
        for (&i, &v) in rows.iter().zip(vals) {
            t.push(
                u32::try_from(dual + i as usize).expect("fits u32"),
                u32::try_from(j).expect("fits u32"),
                v,
            );
        }
    }
    t.to_csc().expect("valid triplets")
}

/// `‖K x − b‖∞ / (1 + ‖b‖∞)` with `K` symmetric, given by its lower
/// triangle: an independent check of the refinement's own report.
fn residual(k_lower: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let mut r: Vec<f64> = b.iter().map(|v| -v).collect();
    for j in 0..k_lower.n_cols() {
        let (rows, vals) = k_lower.col(Col::from_usize(j));
        for (&i, &v) in rows.iter().zip(vals) {
            let i = i as usize;
            r[i] += v * x[j];
            if i != j {
                r[j] += v * x[i];
            }
        }
    }
    let norm = |v: &[f64]| v.iter().fold(0.0_f64, |acc, t| acc.max(t.abs()));
    norm(&r) / (1.0 + norm(b))
}

/// One iteration's factorize-and-solve on the analyzed pattern.
fn step(
    solver: FaerLdl,
    sym: &FaerSymbolic,
    a: &CscMatrix,
    d: &[f64],
    part: &BlockPartition,
    b: &[f64],
    label: &str,
) -> f64 {
    let form = sym.form();
    let k = assemble(a, d, part);
    let (num, inertia): (FaerNumeric<f64>, _) = solver
        .factorize(sym, &k, Regularization::default())
        .expect("quasi-definite after regularization");
    let expected = form.expected_inertia(k.n_cols());
    println!(
        "{label}: D = {d:?}\n  inertia (+, −, 0) = ({}, {}, {}), expected ({}, {}, {}); {} pivot(s) bumped; nnz(L) = {}",
        inertia.positive,
        inertia.negative,
        inertia.zero,
        expected.positive,
        expected.negative,
        expected.zero,
        num.bumped(),
        solver.nnz_factor(&num)
    );
    assert_eq!(inertia, expected, "(S6)");
    let mut x = vec![0.0; b.len()];
    let report = solver.solve_refined(&num, &k, b, &mut x, Refinement::default());
    let independent = residual(&k, &x, b);
    println!(
        "  refinement: {} GMRES step(s), residual {:.2e} (reported) / {independent:.2e} (recomputed), converged = {}",
        report.iters, report.residual, report.converged
    );
    independent
}

fn main() {
    // The toy LP: min cᵀx, A x = b, x ≥ 0 with A of 3 × 5 (full row rank).
    let (m, n) = (3, 5);
    let mut t = Triplets::new(m, n);
    for (i, j, v) in [
        (0, 0, 1.0),
        (0, 1, 1.0),
        (0, 2, 1.0),
        (1, 0, 2.0),
        (1, 3, 1.0),
        (2, 1, -1.0),
        (2, 2, 3.0),
        (2, 4, 1.0),
    ] {
        t.push(i, j, v);
    }
    let a = t.to_csc().expect("valid triplets");

    // The augmented system has the primal block first, then the dual one:
    // the partition records the two boundaries once, for the assembly and
    // for reading the solution back.
    let part = BlockPartition::from_sizes(&[n, m], &[n, m]).expect("nonempty blocks");
    let form = KktForm::Augmented {
        n_primal: n,
        n_dual: m,
    };
    println!(
        "augmented system of order {} = {} primal + {} dual",
        part.n_rows(),
        part.row_block(0).len(),
        part.row_block(1).len()
    );

    // Analyze once. Any `Ordering` serves; our AMD is the default choice.
    let solver = FaerLdl::default();
    let pattern = assemble(&a, &vec![1.0; n], &part).pattern();
    let sym = solver
        .analyze(&pattern, form, &Amd::default())
        .expect("symmetric lower pattern");
    println!(
        "analyzed: {} structure, nnz(L) = {}",
        if sym.is_supernodal() {
            "supernodal"
        } else {
            "simplicial"
        },
        sym.nnz_factor()
    );

    // The right-hand side: a dual residual in the primal block and a
    // primal residual in the dual block.
    let b: Vec<f64> = (0..part.n_rows())
        .map(|i| if part.row_block_of(i) == 0 { 1.0 } else { -2.0 })
        .collect();

    // Iteration 1: a moderate D = X S⁻¹. Iteration 2: the end of the
    // method, where D splits towards 0 (variables at their bound) and ∞
    // (basic variables); the default regularization is what keeps the
    // factorization quasi-definite there, and the refinement against the
    // original K what removes its bias.
    let r1 = step(
        solver,
        &sym,
        &a,
        &[1.0, 0.5, 2.0, 1e-3, 1e3],
        &part,
        &b,
        "iteration 1",
    );
    let r2 = step(
        solver,
        &sym,
        &a,
        &[1e-8, 1e8, 1e-7, 1e7, 1.0],
        &part,
        &b,
        "iteration 2",
    );

    assert!(
        r1 <= 1e-10 && r2 <= 1e-10,
        "residuals against the unregularized K"
    );
    println!("ok");
}
