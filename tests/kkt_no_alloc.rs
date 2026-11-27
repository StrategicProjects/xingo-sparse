//! Chapters 5.15 and 5.17, invariant (L5), and Chapter 5.14, Algorithm S
//! (task L1.5b, L1.6a; issue #7 of the mirror): with every backend, a
//! [`KktSolver::solve`] allocates nothing — the **first** call with a
//! fresh factor included — and a refined solve with a reused
//! [`RefineWorkspace`] allocates nothing after the first.
//!
//! The count is made by a global allocator that forwards to [`System`] and
//! counts, per thread, the calls to `alloc`, `alloc_zeroed` and `realloc`
//! (the other tests of this binary run on other threads and do not
//! disturb it). This is the one `unsafe` of the crate's tests: a
//! `GlobalAlloc` cannot be implemented without it (D-0006).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use xingo_sparse::{
    Amd, Col, CscMatrix, KktForm, KktSolver, RefineWorkspace, Refinement, Regularization, Row,
    SupernodalLdl, SupernodalOptions,
};

thread_local! {
    /// Allocations made by this thread (a `const` initializer and no
    /// destructor: reading it never allocates).
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

/// [`System`], counting.
struct Counting;

#[allow(unsafe_code)]
// SAFETY: every method forwards its arguments unchanged to `System`, which
// upholds the `GlobalAlloc` contract; the only extra work is a `Cell`
// increment in a const-initialized thread-local without destructor, which
// neither allocates nor unwinds (`try_with` ignores a thread being torn
// down).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: the caller's contract for `alloc` is passed on as is.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was returned by `System` (every allocation of this
        // allocator is `System`'s) with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        // SAFETY: as for `dealloc`; `new_size` is the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn bump() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocations this thread makes while running `f`.
fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCS.with(Cell::get);
    f();
    ALLOCS.with(Cell::get) - before
}

/// `[−D, Aᵀ; A, 0]` (lower triangle) for a banded `A` (m × n) and `D` with
/// a spread of `10^{±spread}`, deterministic (as in `kkt_workspace.rs`).
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

/// A fresh factor of `k` and a right-hand side.
fn factor<S: KktSolver<f64>>(solver: &S, k: &CscMatrix, form: KktForm) -> (S::Numeric, Vec<f64>) {
    let sym = solver.analyze(&k.pattern(), form, &Amd::default()).unwrap();
    let (num, _) = solver
        .factorize(&sym, k, Regularization::default())
        .unwrap();
    let b = (0..k.n_rows)
        .map(|i| f64::from(u32::try_from((i * 37) % 11).unwrap()) - 5.0)
        .collect();
    (num, b)
}

/// The check for one backend, on a fresh factor: the first solve, a second
/// solve and a second refined solve with the same workspace allocate
/// nothing, and the two solves agree bit for bit.
fn allocates_nothing<S: KktSolver<f64>>(solver: &S, k: &CscMatrix, form: KktForm, label: &str) {
    let n = k.n_rows;
    let (num, b) = factor(solver, k, form);
    let mut x = b.clone();
    let first = allocations(|| solver.solve(&num, &mut x));
    assert_eq!(first, 0, "{label}: the first solve allocated {first} times");
    let mut y = b.clone();
    let second = allocations(|| solver.solve(&num, &mut y));
    assert_eq!(
        second, 0,
        "{label}: the second solve allocated {second} times"
    );
    assert_eq!(x, y, "{label}");
    // Refinement: the workspace grows at its first call, then nothing.
    let mut w = RefineWorkspace::new();
    let mut r = vec![0.0; n];
    let one = solver.solve_refined_with(&num, k, &b, &mut r, Refinement::default(), &mut w);
    let mut s = vec![0.0; n];
    let mut two = None;
    let again = allocations(|| {
        two = Some(solver.solve_refined_with(&num, k, &b, &mut s, Refinement::default(), &mut w));
    });
    assert_eq!(
        again, 0,
        "{label}: a refined solve with a reused workspace allocated {again} times"
    );
    assert_eq!(two, Some(one), "{label}");
    assert_eq!(r, s, "{label}");
}

#[test]
fn kkt_solves_allocate_nothing_from_the_first_call() {
    let (m, n) = (40, 120);
    let k = augmented(m, n, 4.0);
    let form = KktForm::Augmented {
        n_primal: n,
        n_dual: m,
    };
    // Simplicial (0), supernodal on every block (∞) and the mixed switch;
    // the kernel path of S0′ through `NaiveDense`.
    for switch in [0.0, f64::INFINITY, 40.0] {
        let own = SupernodalLdl::<f64>::new(
            xingo_sparse::NaiveDense,
            SupernodalOptions {
                supernodal_switch: switch,
                ..SupernodalOptions::default()
            },
        );
        allocates_nothing(&own, &k, form, "supernodal-ldl");
        // A clone of the factor (`NumericLdl: Clone`) solves without
        // allocating from its first call too, with the same bits.
        let (num, b) = factor(&own, &k, form);
        let copy = num.clone();
        let (mut x, mut z) = (b.clone(), b);
        own.solve(&num, &mut x);
        let cloned = allocations(|| own.solve(&copy, &mut z));
        assert_eq!(cloned, 0, "a solve with a clone allocated {cloned} times");
        assert_eq!(x, z);
    }
    #[cfg(feature = "faer")]
    for supernodal in [Some(false), Some(true)] {
        let solver = xingo_sparse::FaerLdl { supernodal };
        allocates_nothing(&solver, &k, form, "faer-ldl");
    }
}

/// Chapter 5.14, Algorithm H (issue #1): a condition estimate allocates
/// its three vectors of order `N` and nothing else (the solves allocate
/// nothing, (L5)), with every backend.
#[test]
fn condition_estimate_allocates_three_vectors() {
    fn three<S: KktSolver<f64>>(solver: &S, k: &CscMatrix, form: KktForm, label: &str) {
        let (num, _) = factor(solver, k, form);
        let mut c = None;
        let count = allocations(|| c = Some(solver.condition_estimate(&num, k)));
        assert_eq!(count, 3, "{label}: {count} allocations");
        let c = c.unwrap();
        assert!(c.solves >= 4 && c.rcond() > 0.0, "{label}: {c:?}");
    }
    let (m, n) = (40, 120);
    let k = augmented(m, n, 4.0);
    let form = KktForm::Augmented {
        n_primal: n,
        n_dual: m,
    };
    three(&SupernodalLdl::<f64>::default(), &k, form, "supernodal-ldl");
    #[cfg(feature = "faer")]
    for supernodal in [Some(false), Some(true)] {
        three(&xingo_sparse::FaerLdl { supernodal }, &k, form, "faer-ldl");
    }
}
