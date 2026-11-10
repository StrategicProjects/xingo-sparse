//! Fill-reducing orderings (Chapters 5.9–5.10) against the natural order,
//! with the fill counted before and after by `Pattern::cholesky_nnz`
//! (Algorithm E, the structural `nnz(L)` of the Cholesky factor):
//!
//! * `Amd` (and `Colamd` as SYMAMD) on a symmetric pattern — the 5-point
//!   Laplacian of a `k × k` grid, the textbook case where the band order
//!   fills `O(k³)` and a minimum-degree order much less;
//! * `Colamd` on a rectangular pattern for an LU with row pivoting (or a
//!   QR): a matrix with one dense column, which the natural order
//!   eliminates first — filling everything — and COLAMD eliminates last.
//!
//! Run with `cargo run -p xingo-sparse --example ordering`.

use xingo_sparse::{Amd, Colamd, CscMatrix, Natural, Ordering, OrderingTarget, Pattern, Triplets};

/// The 5-point Laplacian of a `k × k` grid: node `(r, c)` is `r k + c`,
/// `4` on the diagonal, `−1` to the four neighbours.
fn grid_laplacian(k: usize) -> CscMatrix {
    let n = k * k;
    let node = |r: usize, c: usize| u32::try_from(r * k + c).expect("grid fits u32");
    let mut t = Triplets::new(n, n);
    for r in 0..k {
        for c in 0..k {
            t.push(node(r, c), node(r, c), 4.0);
            if r + 1 < k {
                t.push(node(r, c), node(r + 1, c), -1.0);
                t.push(node(r + 1, c), node(r, c), -1.0);
            }
            if c + 1 < k {
                t.push(node(r, c), node(r, c + 1), -1.0);
                t.push(node(r, c + 1), node(r, c), -1.0);
            }
        }
    }
    t.to_csc().expect("valid triplets")
}

/// An `m × n` matrix (`m ≥ n ≥ 2`) whose column 0 is dense and whose
/// column `j ≥ 1` has entries at rows `j − 1` and `j`: `SᵀS` is a hub
/// (column 0) joined to a path.
fn dense_column_first(m: usize, n: usize) -> CscMatrix {
    let mut t = Triplets::new(m, n);
    for i in 0..m {
        t.push(u32::try_from(i).expect("fits u32"), 0, 1.0);
    }
    for j in 1..n {
        let jj = u32::try_from(j).expect("fits u32");
        t.push(jj - 1, jj, 1.0);
        t.push(jj, jj, -1.0);
    }
    t.to_csc().expect("valid triplets")
}

/// `nnz(L)` of the Cholesky factor of the symmetric graph that `target`
/// implies — `S + Sᵀ` for `Symmetric`, `SᵀS` for `Columns` — under the
/// ordering `o`.
fn fill(o: &dyn Ordering, s: &Pattern, target: OrderingTarget) -> usize {
    let perm = o
        .order(s, target)
        .expect("a square or rectangular pattern as the target requires");
    let graph = match target {
        OrderingTarget::Symmetric => s.clone(),
        OrderingTarget::Columns => s.ata(),
    };
    graph
        .cholesky_nnz(&perm)
        .expect("square graph, permutation of its order")
}

fn report(
    title: &str,
    s: &Pattern,
    target: OrderingTarget,
    orderings: &[&dyn Ordering],
) -> Vec<usize> {
    println!(
        "{title}: {} × {}, {} entries",
        s.n_rows(),
        s.n_cols(),
        s.nnz()
    );
    let fills: Vec<usize> = orderings.iter().map(|o| fill(*o, s, target)).collect();
    let natural = fills[0];
    for (o, &f) in orderings.iter().zip(&fills) {
        #[allow(clippy::cast_precision_loss)] // counts far below 2^52
        let ratio = f as f64 / natural as f64;
        println!(
            "  {:<8} nnz(L) = {f:>6}  ({ratio:.2}× the natural order)",
            o.name()
        );
    }
    fills
}

fn main() {
    // 1. Symmetric: AMD and SYMAMD on the grid Laplacian.
    let k = 12;
    let grid = grid_laplacian(k).pattern();
    let symmetric = report(
        &format!("5-point Laplacian of a {k} × {k} grid"),
        &grid,
        OrderingTarget::Symmetric,
        &[&Natural, &Amd::default(), &Colamd::default()],
    );

    // 2. Columns: COLAMD on the matrix with a dense first column. The
    //    natural order pivots on the dense column first and fills the
    //    whole triangle, n(n + 1)/2 entries; a good column order leaves it
    //    for last.
    let (m, n) = (40, 30);
    let arrow = dense_column_first(m, n).pattern();
    let columns = report(
        "dense column first",
        &arrow,
        OrderingTarget::Columns,
        &[&Natural, &Colamd::default()],
    );

    assert!(
        symmetric[1] < symmetric[0],
        "AMD beats the band order on the grid"
    );
    assert!(
        symmetric[2] < symmetric[0],
        "SYMAMD beats the band order on the grid"
    );
    assert_eq!(
        columns[0],
        n * (n + 1) / 2,
        "the natural order fills the whole triangle"
    );
    assert!(
        columns[1] < columns[0],
        "COLAMD moves the dense column last"
    );
    println!("ok");
}
