//! Chapter 5.15, Algorithm S and Chapter 5.14, Algorithm F (task L1.5b):
//! a reused `RefineWorkspace` gives the same refined solution as a fresh
//! one, call after call. (That the reuse allocates nothing was measured
//! with a counting allocator in the L1.5b harness, outside the crate: a
//! global allocator needs `unsafe`, which D-0006 keeps out of tests.)
#![cfg(feature = "faer")]

use xingo_sparse::{
    Amd, Col, CscMatrix, FaerLdl, KktForm, KktSolver, RefineWorkspace, Refinement, Regularization,
    Row,
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

#[test]
fn refinement_reuses_its_workspace() {
    let (m, n) = (40, 120);
    let k = augmented(m, n, 4.0);
    let form = KktForm::Augmented {
        n_primal: n,
        n_dual: m,
    };
    for supernodal in [Some(false), Some(true)] {
        let solver = FaerLdl { supernodal };
        let sym = solver.analyze(&k.pattern(), form, &Amd::default()).unwrap();
        let (num, _) = solver
            .factorize(&sym, &k, Regularization::default())
            .unwrap();
        let b: Vec<f64> = (0..n + m)
            .map(|i| f64::from(u32::try_from((i * 37) % 11).unwrap()) - 5.0)
            .collect();
        let mut x = vec![0.0; n + m];
        let mut work = RefineWorkspace::new();
        let first =
            solver.solve_refined_with(&num, &k, &b, &mut x, Refinement::default(), &mut work);
        let x1 = x.clone();
        let again =
            solver.solve_refined_with(&num, &k, &b, &mut x, Refinement::default(), &mut work);
        assert_eq!(first, again);
        assert_eq!(x, x1);
        assert!(first.converged, "{supernodal:?}: {first:?}");
        // The allocating convenience gives the same answer.
        let mut z = vec![0.0; n + m];
        assert_eq!(
            solver.solve_refined(&num, &k, &b, &mut z, Refinement::default()),
            first
        );
        assert_eq!(z, x);
    }
}
