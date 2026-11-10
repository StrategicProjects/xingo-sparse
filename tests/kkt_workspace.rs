//! Chapters 5.15 and 5.17, Algorithm S, and Chapter 5.14, Algorithm F
//! (tasks L1.5b and L1.6a): a reused `RefineWorkspace` gives the same
//! refined solution as a fresh one, call after call, with every backend.
//! (That the reuse allocates nothing was measured with a counting
//! allocator in the L1.5b/L1.6a harnesses, outside the crate: a global
//! allocator needs `unsafe`, which D-0006 keeps out of tests.)

use xingo_sparse::{
    Amd, Col, CscMatrix, KktForm, KktSolver, RefineWorkspace, Refinement, Regularization, Row,
    SupernodalLdl, SupernodalOptions,
};

/// `[−D, Aᵀ; A, 0]` (lower triangle) for a banded `A` (m × n) and `D` with
/// a spread of `10^{±spread}`, deterministic.
fn augmented(m: usize, n: usize, spread: f64) -> CscMatrix {
    let mut t = Vec::new();
    for j in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let u = (j as f64 * 0.618_033_988_75).fract();
        t.push((
            Row::from_usize(j),
            Col::from_usize(j),
            -(10f64.powf(spread * (2.0 * u - 1.0))),
        ));
        for i in [j % m, (j + 1) % m] {
            #[allow(clippy::cast_precision_loss)]
            let v = 1.0 + (i + j) as f64 / 7.0;
            t.push((Row::from_usize(n + i), Col::from_usize(j), v));
        }
    }
    CscMatrix::from_triplets(n + m, n + m, &t).unwrap()
}

/// The check for one backend: two refined solves with the same workspace
/// agree exactly and converge.
fn reuses<S: KktSolver<f64>>(solver: &S, k: &CscMatrix, form: KktForm, label: &str) {
    let n = k.n_rows;
    let sym = solver.analyze(&k.pattern(), form, &Amd::default()).unwrap();
    let (num, _) = solver
        .factorize(&sym, k, Regularization::default())
        .unwrap();
    let b: Vec<f64> = (0..n)
        .map(|i| f64::from(u32::try_from((i * 37) % 11).unwrap()) - 5.0)
        .collect();
    let mut x = vec![0.0; n];
    let mut work = RefineWorkspace::new();
    let first = solver.solve_refined_with(&num, k, &b, &mut x, Refinement::default(), &mut work);
    let x1 = x.clone();
    let again = solver.solve_refined_with(&num, k, &b, &mut x, Refinement::default(), &mut work);
    assert_eq!(first, again);
    assert_eq!(x, x1);
    assert!(first.converged, "{label}: {first:?}");
    // The same without a shared workspace.
    let mut y = vec![0.0; n];
    let fresh = solver.solve_refined(&num, k, &b, &mut y, Refinement::default());
    assert_eq!(fresh, first);
    assert_eq!(y, x1);
}

#[test]
fn refinement_reuses_its_workspace() {
    let (m, n) = (40, 120);
    let k = augmented(m, n, 4.0);
    let form = KktForm::Augmented {
        n_primal: n,
        n_dual: m,
    };
    for switch in [0.0, f64::INFINITY, 40.0] {
        let own = SupernodalLdl::<f64>::new(
            xingo_sparse::NaiveDense,
            SupernodalOptions {
                supernodal_switch: switch,
                ..SupernodalOptions::default()
            },
        );
        reuses(&own, &k, form, "supernodal-ldl");
    }
    #[cfg(feature = "faer")]
    for supernodal in [Some(false), Some(true)] {
        let solver = xingo_sparse::FaerLdl { supernodal };
        reuses(&solver, &k, form, "faer-ldl");
    }
}
