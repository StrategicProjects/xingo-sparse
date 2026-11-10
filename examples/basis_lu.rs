//! A simplex basis through the dynamic LU (Chapters 5.1, 5.3–5.5):
//! factorize `B`, solve with `B` (FTRAN) and `Bᵀ` (BTRAN), exchange one
//! column with the Forrest–Tomlin update, refactorize the new basis and
//! compare.
//!
//! The basis is never formed: it is a list of references into `[A | ±I]`
//! (`ColumnRef`) plus the constraint matrix `A`, exactly what the simplex
//! method holds. Run with `cargo run -p xingo-sparse --example basis_lu`.

use xingo_sparse::{
    Col, ColumnRef, CscMatrix, Lu, LuFactor, Row, Sign, SparseVec, Triplets, UpdateOutcome,
};

/// `B v` for the basis `heading` of `a`; `v` is indexed by position, the
/// result by row.
fn basis_times(heading: &[ColumnRef], a: &CscMatrix, v: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.n_rows()];
    for (k, c) in heading.iter().enumerate() {
        c.for_each_entry(a, |i, b_ik| y[i as usize] += b_ik * v[k]);
    }
    y
}

/// `Bᵀ w`; `w` is indexed by row, the result by position.
fn basis_transpose_times(heading: &[ColumnRef], a: &CscMatrix, w: &[f64]) -> Vec<f64> {
    heading
        .iter()
        .map(|c| {
            let mut s = 0.0;
            c.for_each_entry(a, |i, b_ik| s += b_ik * w[i as usize]);
            s
        })
        .collect()
}

/// `‖x − y‖∞`.
fn distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max)
}

fn main() {
    // 1. The constraint matrix A (4 × 6) of a small LP, by coordinates.
    let (m, n) = (4, 6);
    let mut t = Triplets::new(m, n);
    for (i, j, v) in [
        (0, 0, 2.0),
        (0, 1, 1.0),
        (0, 3, 1.0),
        (1, 1, 1.0),
        (1, 2, 3.0),
        (1, 4, -1.0),
        (2, 0, 1.0),
        (2, 2, 1.0),
        (2, 5, 2.0),
        (3, 3, 4.0),
        (3, 4, 1.0),
        (3, 5, 1.0),
    ] {
        t.push(i, j, v);
    }
    let a: CscMatrix = t.to_csc().expect("valid triplets");
    println!("A: {} × {}, {} nonzeros", a.n_rows(), a.n_cols(), a.nnz());

    // 2. A basis heading: position k holds column b_k of B. Four structural
    //    columns of A; row 1 is a singleton (only a_2 has an entry there),
    //    the other three rows form a cycle that the Markowitz search
    //    eliminates (the "bump").
    let mut heading = vec![
        ColumnRef::Structural(Col(0)),
        ColumnRef::Structural(Col(2)),
        ColumnRef::Structural(Col(5)),
        ColumnRef::Structural(Col(3)),
    ];

    // 3. Factorize (Algorithm L: singletons first, then Markowitz
    //    threshold pivoting on the bump). The default update policy of
    //    `Lu` is Forrest–Tomlin.
    let mut lu = Lu::<f64>::default();
    lu.factorize(&(heading.as_slice(), &a))
        .expect("the basis is nonsingular");
    let s = lu.stats();
    println!(
        "factorized B ({} nonzeros): nnz(L) = {}, nnz(U) = {}, singletons {} + {}, bump of {} pivot(s), growth {:.2}",
        s.nnz_basis,
        s.nnz_l,
        s.nnz_u,
        s.col_singletons,
        s.row_singletons,
        s.markowitz_pivots,
        s.growth
    );

    // 4. FTRAN: x = B⁻¹ b (a row-indexed right-hand side in, the
    //    position-indexed solution out).
    let b = [1.0, 2.0, 3.0, 4.0];
    let mut x = SparseVec::from_dense(&b);
    lu.ftran(&mut x);
    let res_ftran = distance(&basis_times(&heading, &a, x.values()), &b);
    println!("FTRAN: x = {:?}, ‖B x − b‖∞ = {res_ftran:.1e}", x.values());

    // 5. BTRAN: y = B⁻ᵀ c_B (position-indexed in, row-indexed out): the
    //    duals of the simplex method.
    let c_b = [1.0, -1.0, 0.5, 2.0];
    let mut y = SparseVec::from_dense(&c_b);
    lu.btran(&mut y);
    let res_btran = distance(&basis_transpose_times(&heading, &a, y.values()), &c_b);
    println!(
        "BTRAN: y = {:?}, ‖Bᵀ y − c_B‖∞ = {res_btran:.1e}",
        y.values()
    );

    // 6. Exchange: the logical (slack) column of row 1 enters at position
    //    r = 1 (a_2 leaves). The entering column goes through
    //    `ftran_for_update`, which computes α = B⁻¹ e_1 and records the
    //    spike; `update(r, α_r)` then applies the Forrest–Tomlin update
    //    (Algorithm U). The factor never refactorizes on its own: it only
    //    asks (`RefactorNeeded`).
    let r = 1;
    let entering = ColumnRef::Logical(Row(1), Sign::Plus);
    let mut alpha = SparseVec::new(m);
    entering.for_each_entry(&a, |i, v| alpha.set(i, v));
    lu.ftran_for_update(&mut alpha);
    let alpha_r = alpha.get(u32::try_from(r).expect("position fits u32"));
    let outcome = lu
        .update(r, alpha_r)
        .expect("a stable, nonsingular exchange");
    heading[r] = entering;
    let u = lu.update_stats();
    println!(
        "update at position {r} (α_r = {alpha_r:.4}): {outcome:?}; {} update(s), {} row eta(s) with {} multiplier(s)",
        u.updates, u.n_etas, u.nnz_eta
    );
    assert_eq!(outcome, UpdateOutcome::Updated);

    // The updated factor solves with the new basis.
    let mut x_updated = SparseVec::from_dense(&b);
    lu.ftran(&mut x_updated);
    let res_updated = distance(&basis_times(&heading, &a, x_updated.values()), &b);
    println!(
        "FTRAN after the update: x = {:?}, ‖B′ x − b‖∞ = {res_updated:.1e}",
        x_updated.values()
    );

    // 7. Refactorize the new basis from scratch and compare the solutions.
    lu.factorize(&(heading.as_slice(), &a))
        .expect("the new basis is nonsingular");
    let mut x_fresh = SparseVec::from_dense(&b);
    lu.ftran(&mut x_fresh);
    let res_fresh = distance(&basis_times(&heading, &a, x_fresh.values()), &b);
    let gap = distance(x_updated.values(), x_fresh.values());
    println!(
        "refactorized: {} factorization(s) so far, ‖B′ x − b‖∞ = {res_fresh:.1e}, updated vs fresh ‖Δx‖∞ = {gap:.1e}",
        lu.stats().factorizations
    );

    let tol = 1e-12;
    assert!(
        res_ftran <= tol && res_btran <= tol,
        "solves with the factorized basis"
    );
    assert!(res_updated <= tol, "solve with the updated factor");
    assert!(
        res_fresh <= tol && gap <= tol,
        "updated and fresh factors agree"
    );
    println!("ok");
}
