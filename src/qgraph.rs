//! # Chapter 5.11 — Approximate minimum degree on a quotient graph
//!
//! *New with task L1.3. Private: the engine behind [`crate::Amd`] and
//! [`crate::Colamd`] (Chapter 5.10).*
//!
//! ## Purpose
//! Minimum degree orders the unknowns of a symmetric sparse matrix one at
//! a time, always eliminating the one that has the fewest neighbours in
//! the graph of the remaining Schur complement. Forming that graph is too
//! expensive (it fills in); the *quotient graph* represents it implicitly
//! in the space of the original matrix, and the *approximate* degree of
//! Amestoy, Davis & Duff (1996) replaces the exact degree, whose update
//! dominates the cost of classical minimum degree, by an upper bound that
//! costs only a scan of the lists already touched. COLAMD (Davis, Gilbert,
//! Larimore & Ng 2004) is the same process on the graph of `AᵀA`, whose
//! quotient graph can be *started* from `A` itself: every row of `A` is a
//! clique of `AᵀA`, i.e. an *element*. One engine, two initializations.
//!
//! ## Notation
//! The nodes are *variables* (the unknowns still to be ordered, `0..n`)
//! and *elements* (cliques of the eliminated graph). A variable `i` has
//!
//! * `E_i` — the elements adjacent to it;
//! * `A_i` — the variables adjacent to it by an original edge not yet
//!   covered by an element;
//! * `|i|` — its weight: the number of variables it represents (a
//!   *supervariable* collects indistinguishable variables, whose
//!   adjacencies coincide);
//! * `d̄_i` — its approximate external degree (an upper bound of the
//!   weighted number of variables adjacent to `i`, `i` excluded).
//!
//! An element `e` has its variable list `L_e` and its weight `|L_e| =
//! Σ_{j ∈ L_e} |j|`. When variable `p` is eliminated it *becomes* the
//! element `p` (same index), with `L_p = (A_p ∪ ⋃_{e ∈ E_p} L_e) ∖ {p}`;
//! the elements of `E_p` are *absorbed* (contained in `L_p`, they are no
//! longer needed). The exact external degree of `i` is
//! `|(A_i ∪ ⋃_{e ∈ E_i} L_e) ∖ {i}|`. `n_el` is the weighted number of
//! variables already eliminated (or excluded), so `n − n_el` remain.
//! In the code the list of a node `x` is `iw[pe[x]..pe[x] + len[x]]` inside
//! one workspace `iw` (see "Storage" below): for a variable `i` it holds
//! `A_i` followed by `E_i` (its last `elen[i]` entries, the newest element
//! *last*, so that step Q5 updates the list in place and appends `p`), for
//! an element `e` it holds `L_e`.
//!
//! ## Algorithm Q (approximate minimum degree on a quotient graph)
//! Given `n` variables with original adjacencies `A_i` (AMD) and/or
//! initial elements `L_e` (COLAMD: the rows of `A`), and a set of excluded
//! variables (dense ones, ordered last by the caller).
//!
//! Q1. [Initialize.] `|i| ← 1`; `E_i ←` the initial elements containing
//!     `i`; `d̄_i ← min(n − 1, |A_i| + Σ_{e ∈ E_i} (|L_e| − 1))` (exact when
//!     there are no initial elements; COLAMD's initial score otherwise).
//!     Excluded variables count as eliminated. Put every other variable in
//!     the degree list of `d̄_i`.
//! Q2. [Select.] Take `p` from the nonempty degree list of least degree.
//!     `n_el ← n_el + |p|`.
//! Q3. [New element.] Build `L_p = (A_p ∪ ⋃_{e ∈ E_p} L_e) ∖ {p}` from the
//!     live principal variables, removing each from its degree list;
//!     absorb every `e ∈ E_p` into `p`; `p` becomes an element of weight
//!     `|L_p|`.
//! Q4. [Set differences.] For every `i ∈ L_p` and every live `e ∈ E_i`,
//!     accumulate `w(e) = |L_e ∖ L_p| = |L_e| − Σ_{i ∈ L_e ∩ L_p} |i|`
//!     (the first visit sets `w(e) ← |L_e|`).
//! Q5. [Degree update.] For every `i ∈ L_p`: drop the dead elements of
//!     `E_i`; with *aggressive absorption*, absorb into `p` every `e` with
//!     `w(e) = 0` (then `L_e ⊆ L_p`); prune from `A_i` the variables that
//!     are dead or in `L_p` (element `p` now covers those edges); add `p`
//!     to `E_i`. Let `δ_i = |A_i| + Σ_{e ∈ E_i ∖ p} w(e)`. If `E_i = {p}`
//!     and `A_i = ∅`, `i` is indistinguishable from `p`: *mass
//!     elimination* — `i` is eliminated with `p` (`n_el ← n_el + |i|`,
//!     `|L_p| ← |L_p| − |i|`). Otherwise keep `δ_i` (AMD: `min(d̄_i, δ_i)`,
//!     the bound `d̄_i^{old} + |L_p ∖ i|` of ADD 1996, eq. (4); COLAMD:
//!     `δ_i`) and hash `i` by the sum of the indices of `E_i ∖ p` and
//!     `A_i`.
//! Q6. [Supervariables.] Among the variables of `L_p` with equal hash,
//!     merge every pair with identical `E_i` and `A_i` (equal lengths,
//!     marked comparison): `|i| ← |i| + |j|`, `j` dies, its parent is `i`.
//! Q7. [Finalize.] For every principal `i ∈ L_p`:
//!     `d̄_i ← min(δ_i + |L_p ∖ i|, n − n_el − |i|)`; put `i` back in its
//!     degree list. `L_p` keeps its principal variables only. If `n_el < n`
//!     go to Q2.
//! Q8. [Output, AMD.] The *assembly tree* has an edge from every absorbed
//!     element to the element that absorbed it and from every merged or
//!     mass-eliminated variable to its principal or pivot. Its postorder
//!     (children before parents; absorbed elements' subtrees before the
//!     merged variables, which come right before their principal) lists
//!     every non-excluded variable once: that is AMD's ordering (as
//!     SuiteSparse's `amd_postorder`).
//! Q8′. [Output, COLAMD.] The pivots in order of elimination, each
//!     followed by the variables merged into it or mass-eliminated with it
//!     (as `colamd.c`'s `order_children`, which keeps supercolumns
//!     contiguous).
//!
//! *Measured, not proved:* the postorder of the assembly tree would have
//! exactly the fill of the pivot order if the assembly tree were the
//! elimination tree of the pivot order (Liu 1990). It is not quite (an
//! element absorbed aggressively hangs below the absorbing pivot, while
//! its elimination-tree parent is an ancestor of that pivot), and the two
//! outputs differ by a fraction of a percent on Netlib and Matrix Market
//! (the decisions of task L1.3, "rejected, measured"); each ordering keeps the
//! convention of its reference implementation, which is what makes the
//! oracle comparison tight.
//!
//! ## Invariants
//! Checked after every pivot by `check` when the graph is small (`n ≤
//! PARANOID_N`) in debug builds, and in the tests:
//! (Q1) every live principal variable is in exactly one degree list, that
//!      of `d̄_i`; no other node is;
//! (Q2) with aggressive absorption, every live element has
//!      `|L_e| = Σ |j|` over its live principal variables (a variable of
//!      `L_e` dies only by becoming a pivot adjacent to `e`, which absorbs
//!      `e`, or by mass elimination, which happens only when every other
//!      element of the variable had `w(e) = 0` and was absorbed);
//! (Q3) `Σ |i|` over the live principal variables plus `n_el` is `n`;
//! (Q4) `d̄_i` is at least the exact external degree of `i` (ADD 1996,
//!      Theorem 4.1: each term of the minimum is an upper bound).
//! The output is a permutation (Chapter 5.9, (P1)).
//!
//! ## Complexity
//! Q3–Q7 cost `O(Σ_{i ∈ L_p} (|E_i| + |A_i|))` per pivot, plus the
//! quadratic comparison of a hash bucket (short in practice). No exact
//! degree is ever computed: this is the gain of ADD 1996 over multiple
//! minimum degree.
//!
//! *Constant factors (task L1.5b).* The scans of Q4 and Q5 are the whole
//! cost (≈ 85 % of the samples on maros-r7's augmented system), so they
//! touch one array per entry, as SuiteSparse's AMD does: `w[e]` holds
//! `w(e)` offset by a flag that grows by `n + 1` per pivot (`w[e] ≥ flag`:
//! seen in this pivot; `0`: a dead element; restarted before it would
//! overflow), and `vw[j]` is `|j|` for a live principal variable outside
//! `L_p`, `0` for any other — the test of Q5's scan of `A_i`. Q5 compacts
//! the list of `i` in place (hence `A_i` before `E_i`, the newest element
//! last). On maros-r7 the engine scans the same number of list entries as
//! faer's port of SuiteSparse AMD (14.76 M in Q4 and 18.36 M in Q5,
//! against 14.78 M and 18.38 M); what was left of the time difference was
//! the storage of the lists, one `Vec` per node, replaced in task L1.6a by
//! the workspace below.
//!
//! ## Storage (task L1.6a): one workspace, garbage collected
//! Every list lives in one array `iw`, node `x` owning `iw[pe[x]..pe[x] +
//! len[x]]`; `pfree` is the first unused position. Lists only ever shrink
//! in place (step Q5 drops the dead elements and the covered variables and
//! appends `p` in a slot one of them freed — there always is one: `i ∈
//! L_p` because `p ∈ A_i`, or because an element now absorbed was in
//! `E_i`, and both are dropped), die (`len[x] ← 0`: absorbed elements,
//! merged and mass-eliminated variables, the pivot's own list), or are
//! created at `pfree` (the new element `L_p`, written there in step Q3 and
//! compacted to its principal variables in Q7). What they leave behind is
//! garbage. Before step Q3 the engine checks that `pfree` plus an upper
//! bound of `|L_p|` (`|A_p| + Σ_{e ∈ E_p} |L_e|`) fits; when it does not,
//! **garbage collection** compacts every live list to the front of `iw`
//! in one pass — the first entry of each list is replaced by a flagged
//! copy of its owner (`x | 2³¹`) and saved in `pe[x]`, so the pass can
//! tell where each list starts without a sort — and the workspace grows
//! only if the compaction still leaves too little room. Node indices must
//! therefore stay below `2³¹` (asserted). This is the scheme of
//! SuiteSparse's AMD (read, not copied). The order of the entries inside
//! every list is exactly that of the one-vector-per-list engine of
//! L1.3–L1.5b, so the pivot sequence and the output permutation are
//! unchanged (checked by a hash of the permutations of AMD and COLAMD on
//! the 93 Netlib LPs, three patterns each, before and after). The input
//! lists arrive in compressed form ([`Input`]) and the arrays of the graph
//! are three allocations sliced into fields ([`Arena`]) for the same
//! reason: no allocation per node anywhere in the ordering. Measured in
//! L1.6a on the ten largest Netlib KKT patterns (medians of 5, load
//! gate): 1.25× faer's AMD before, 1.15× after (geometric mean; 0.87–1.51);
//! what is left is in the scans themselves (Q4 + Q5 are 82 % of the time
//! on maros-r7's augmented system, with the same number of entries as
//! faer's port), of which the bounds checks of the random accesses are
//! about 11 % (measured once with unchecked indexing, not kept: the crate
//! has no `unsafe`).
//!
//! ## References
//! Amestoy, Davis & Duff, "An approximate minimum degree ordering
//! algorithm", SIAM J. Matrix Anal. Appl. 17(4), 1996 (Algorithm 2, the
//! approximate degree, aggressive absorption, mass elimination and
//! supervariable hashing); Davis, Gilbert, Larimore & Ng, "A column
//! approximate minimum degree ordering algorithm", ACM TOMS 30(3), 2004
//! (rows as initial elements, the column score); George & Liu, "The
//! evolution of the minimum degree ordering algorithm", SIAM Review 31(1),
//! 1989 (quotient graphs, indistinguishable nodes); Liu (1990) for the
//! postorder. SuiteSparse AMD/COLAMD (BSD-3) are the oracles of the tests,
//! read, not copied.

/// "No node".
const NONE: u32 = u32::MAX;

/// Garbage collection marks the first entry of a live list with its owner
/// and this bit (node indices are below it; see "Storage").
const FLIP: u32 = 1 << 31;

/// Graphs at most this large are checked after every pivot in debug
/// builds (the check costs `O(n²)`).
const PARANOID_N: usize = 64;

/// What the engine is asked to order. The lists are compressed (`ptr`,
/// `idx`), as a [`crate::Pattern`] stores its columns.
pub(crate) struct Input {
    /// Number of variables `n`.
    pub(crate) n: usize,
    /// `A_i` for each variable in `adj_idx[adj_ptr[i]..adj_ptr[i + 1]]`:
    /// original neighbours, no self loop, no duplicate (`adj_ptr` all
    /// zero for COLAMD).
    pub(crate) adj_ptr: Vec<usize>,
    pub(crate) adj_idx: Vec<u32>,
    /// The initial elements `L_e` in `elem_idx[elem_ptr[e]..elem_ptr[e +
    /// 1]]` (none for AMD), without duplicates or excluded variables.
    pub(crate) elem_ptr: Vec<usize>,
    pub(crate) elem_idx: Vec<u32>,
    /// Variables left out of the ordering (the caller orders them last).
    pub(crate) excluded: Vec<bool>,
    /// Aggressive absorption in step Q5.
    pub(crate) aggressive: bool,
    /// AMD's bound `min(d̄_i, δ_i)` in Q5 (`false`: COLAMD's score `δ_i`).
    pub(crate) previous_degree: bool,
    /// COLAMD's conventions (`false`: AMD's): among equal degrees the
    /// smallest index first (AMD: the largest); in Q3 the initial elements
    /// of `E_p` before the newer ones, oldest first (AMD: newest first);
    /// the pivot order as output instead of the postorder of Q8.
    pub(crate) column_order: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// A live variable (principal iff `nv > 0`).
    Var,
    /// A live element.
    Elem,
    /// Dead: absorbed, merged, eliminated, or excluded.
    Gone,
}

/// The arrays of a [`Graph`], three allocations (one per element type)
/// sliced into the fields: no allocation per array ("Storage").
struct Arena {
    us: Vec<usize>,
    u: Vec<u32>,
    b: Vec<bool>,
}

impl Arena {
    /// Sized for `n` variables and `total` nodes.
    fn new(n: usize, total: usize) -> Self {
        Self {
            us: vec![0; 4 * total + 4 * n],
            u: vec![NONE; 2 * total + 4 * n + 1 + n.max(1)],
            b: vec![false; 2 * n],
        }
    }
}

/// Takes the first `k` entries of `rest`.
#[allow(clippy::mut_mut)] // a cursor over the arena: the reference is advanced
fn take<'a, T>(rest: &mut &'a mut [T], k: usize) -> &'a mut [T] {
    let r = core::mem::take(rest);
    let (a, b) = r.split_at_mut(k);
    *rest = b;
    a
}

#[allow(clippy::struct_excessive_bools)] // independent switches of one algorithm
struct Graph<'a> {
    n: usize,
    /// The workspace of every list ("Storage").
    iw: Vec<u32>,
    /// `pe[x]`, `len[x]`: the list of node `x` is `iw[pe[x]..pe[x] + len[x]]`.
    pe: &'a mut [usize],
    len: &'a mut [usize],
    /// First unused position of `iw`.
    pfree: usize,
    /// Garbage collections so far (statistics of the tests).
    ncmpa: usize,
    /// `|E_i|`, the tail of the list of variable `i`.
    elen: &'a mut [usize],
    kind: Vec<Kind>,
    /// `|i|` of principal variables (`0` otherwise).
    nv: &'a mut [usize],
    /// Variables: `d̄_i`; elements: `|L_e|`.
    deg: &'a mut [usize],
    parent: &'a mut [u32],
    is_pivot: &'a mut [bool],
    head: &'a mut [u32],
    next: &'a mut [u32],
    prev: &'a mut [u32],
    mindeg: usize,
    nel: usize,
    in_lp: &'a mut [bool],
    /// Step Q4's `w(e)`, offset: `w[e] ≥ wflg` marks an element seen in
    /// the current pivot, whose `w(e)` is `w[e] − wflg` (one array for the
    /// mark and the value, as in SuiteSparse's AMD).
    w: &'a mut [u32],
    wflg: u32,
    /// `|j|` of a live principal variable outside `L_p`, `0` for every
    /// other variable: the one-array test of step Q5's scan of `A_i`.
    vw: &'a mut [usize],
    mark: &'a mut [usize],
    mark_cur: usize,
    hhead: &'a mut [u32],
    hnext: &'a mut [u32],
    hof: &'a mut [usize],
    aggressive: bool,
    previous_degree: bool,
    column_order: bool,
    /// Pivots in order of elimination.
    pivots: Vec<u32>,
    paranoid: bool,
}

/// Algorithm Q: the order of the non-excluded variables (old indices,
/// first pivot first).
pub(crate) fn order(input: Input) -> Vec<u32> {
    let mut arena = Arena::new(input.n, input.n + input.elem_ptr.len() - 1);
    let mut g = Graph::new(input, &mut arena);
    while g.nel < g.n {
        let p = g.select();
        g.eliminate(p);
        if g.paranoid {
            debug_assert_eq!(g.check(), Ok(()));
        }
    }
    if g.column_order {
        g.pivot_order()
    } else {
        g.postorder()
    }
}

#[inline]
fn u(i: u32) -> usize {
    i as usize
}

#[inline]
fn idx32(i: usize) -> u32 {
    u32::try_from(i).expect("index exceeds u32::MAX")
}

impl<'a> Graph<'a> {
    /// Step Q1.
    #[allow(clippy::too_many_lines)] // step Q1 and its bookkeeping, as in the chapter
    fn new(input: Input, arena: &'a mut Arena) -> Self {
        let Input {
            n,
            adj_ptr,
            adj_idx,
            elem_ptr,
            elem_idx,
            excluded,
            aggressive,
            previous_degree,
            column_order,
        } = input;
        debug_assert_eq!(adj_ptr.len(), n + 1);
        debug_assert_eq!(excluded.len(), n);
        let ne = elem_ptr.len() - 1;
        let total = n + ne;
        assert!(total < u(FLIP), "the quotient graph has too many nodes");
        let (mut us, mut uu, mut bb) = (&mut arena.us[..], &mut arena.u[..], &mut arena.b[..]);
        let pe = take(&mut us, total);
        let len = take(&mut us, total);
        let elen = take(&mut us, n);
        let nv = take(&mut us, n);
        let deg = take(&mut us, total);
        let vw = take(&mut us, n);
        let mark = take(&mut us, total);
        let hof = take(&mut us, n);
        let parent = take(&mut uu, total);
        let head = take(&mut uu, n + 1);
        let next = take(&mut uu, n);
        let prev = take(&mut uu, n);
        let w = take(&mut uu, total);
        let hhead = take(&mut uu, n.max(1));
        let hnext = take(&mut uu, n);
        let is_pivot = take(&mut bb, n);
        let in_lp = take(&mut bb, n);
        nv.fill(1);
        vw.fill(1);
        w.fill(1);
        // Layout of iw: variable i = A_i then E_i; element e = L_e.
        for &v in &elem_idx {
            debug_assert!(!excluded[u(v)], "excluded variable in an element");
            elen[u(v)] += 1;
        }
        let mut pfree = 0;
        for i in 0..n {
            pe[i] = pfree;
            len[i] = adj_ptr[i + 1] - adj_ptr[i] + elen[i];
            pfree += len[i];
        }
        for e in 0..ne {
            pe[n + e] = pfree;
            len[n + e] = elem_ptr[e + 1] - elem_ptr[e];
            pfree += len[n + e];
        }
        let iwlen = pfree + pfree / 5 + n + 1;
        let mut iw = vec![0_u32; iwlen];
        for i in 0..n {
            let a = adj_ptr[i]..adj_ptr[i + 1];
            iw[pe[i]..pe[i] + a.len()].copy_from_slice(&adj_idx[a]);
        }
        // E_i oldest... newest = the initial elements in decreasing order
        // (the traversal of Q3 runs from the newest, i.e. increasing);
        // `hof` serves as the fill counter until the graph is built.
        for e in (0..ne).rev() {
            for &v in &elem_idx[elem_ptr[e]..elem_ptr[e + 1]] {
                let v = u(v);
                iw[pe[v] + len[v] - elen[v] + hof[v]] = idx32(n + e);
                hof[v] += 1;
            }
        }
        hof.fill(0);
        let mut kind = vec![Kind::Var; total];
        for e in 0..ne {
            let l = elem_ptr[e]..elem_ptr[e + 1];
            iw[pe[n + e]..pe[n + e] + l.len()].copy_from_slice(&elem_idx[l.clone()]);
            deg[n + e] = l.len();
            if l.is_empty() {
                kind[n + e] = Kind::Gone;
                w[n + e] = 0;
            } else {
                kind[n + e] = Kind::Elem;
            }
        }
        let mut g = Self {
            n,
            iw,
            pe,
            len,
            pfree,
            ncmpa: 0,
            elen,
            kind,
            nv,
            deg,
            parent,
            is_pivot,
            head,
            next,
            prev,
            mindeg: 0,
            nel: 0,
            in_lp,
            // 0 marks a dead element (any other value below `wflg`: not
            // yet seen in the current pivot).
            w,
            wflg: 0,
            vw,
            mark,
            mark_cur: 0,
            hhead,
            hnext,
            hof,
            aggressive,
            previous_degree,
            column_order,
            pivots: Vec::with_capacity(n),
            paranoid: cfg!(debug_assertions) && n <= PARANOID_N,
        };
        for (i, &ex) in excluded.iter().enumerate() {
            if ex {
                g.kind[i] = Kind::Gone;
                g.nv[i] = 0;
                g.vw[i] = 0;
                g.nel += 1;
                continue;
            }
            let ei = g.elen[i];
            let from_elems: usize = g
                .elems(i)
                .iter()
                .map(|&e| g.deg[u(e)].saturating_sub(1))
                .sum();
            let d = g.len[i] - ei + from_elems;
            g.deg[i] = d.min(n - 1);
        }
        let insert = |g: &mut Self, i: usize| {
            if g.kind[i] == Kind::Var {
                g.list_insert(i, g.deg[i]);
            }
        };
        if column_order {
            for i in (0..n).rev() {
                insert(&mut g, i);
            }
        } else {
            for i in 0..n {
                insert(&mut g, i);
            }
        }
        g.mindeg = 0;
        if g.paranoid {
            debug_assert_eq!(g.check(), Ok(()));
        }
        g
    }

    /// The list of node `x`.
    #[inline]
    fn list(&self, x: usize) -> &[u32] {
        &self.iw[self.pe[x]..self.pe[x] + self.len[x]]
    }

    /// `E_i`, oldest first.
    #[inline]
    fn elems(&self, i: usize) -> &[u32] {
        let l = self.list(i);
        &l[l.len() - self.elen[i]..]
    }

    /// `A_i`.
    #[inline]
    fn vars(&self, i: usize) -> &[u32] {
        let l = self.list(i);
        &l[..l.len() - self.elen[i]]
    }

    fn list_insert(&mut self, i: usize, d: usize) {
        let h = self.head[d];
        self.next[i] = h;
        self.prev[i] = NONE;
        if h != NONE {
            self.prev[u(h)] = idx32(i);
        }
        self.head[d] = idx32(i);
    }

    fn list_remove(&mut self, i: usize) {
        let (p, x) = (self.prev[i], self.next[i]);
        if p == NONE {
            self.head[self.deg[i]] = x;
        } else {
            self.next[u(p)] = x;
        }
        if x != NONE {
            self.prev[u(x)] = p;
        }
        self.next[i] = NONE;
        self.prev[i] = NONE;
    }

    /// Step Q2.
    fn select(&mut self) -> usize {
        while self.head[self.mindeg] == NONE {
            self.mindeg += 1;
        }
        let p = u(self.head[self.mindeg]);
        self.list_remove(p);
        self.nel += self.nv[p];
        p
    }

    /// Makes room for `need` more entries at `pfree`: garbage collection
    /// first ("Storage"), growth of the workspace only if that is not
    /// enough.
    fn ensure(&mut self, need: usize) {
        if self.pfree + need <= self.iw.len() {
            return;
        }
        self.collect_garbage();
        if self.pfree + need > self.iw.len() {
            let target = self.pfree + need;
            self.iw.resize(target + target / 5 + self.n + 1, 0);
        }
    }

    /// Garbage collection: every live list moved to the front of `iw`,
    /// its entries in the same order.
    fn collect_garbage(&mut self) {
        self.ncmpa += 1;
        for x in 0..self.pe.len() {
            if self.len[x] > 0 {
                let p = self.pe[x];
                self.pe[x] = u(self.iw[p]);
                self.iw[p] = idx32(x) | FLIP;
            }
        }
        let (mut psrc, mut pdst) = (0, 0);
        while psrc < self.pfree {
            let v = self.iw[psrc];
            psrc += 1;
            if v & FLIP != 0 {
                let x = u(v & !FLIP);
                self.iw[pdst] = idx32(self.pe[x]);
                self.pe[x] = pdst;
                pdst += 1;
                let rest = self.len[x] - 1;
                self.iw.copy_within(psrc..psrc + rest, pdst);
                psrc += rest;
                pdst += rest;
            }
        }
        self.pfree = pdst;
    }

    /// Steps Q3–Q7 for the pivot `p`.
    fn eliminate(&mut self, p: usize) {
        self.is_pivot[p] = true;
        self.pivots.push(idx32(p));
        self.in_lp[p] = true;
        self.vw[p] = 0;
        let (pme1, pme2, mut dk) = self.new_element(p);
        self.set_differences(pme1, pme2);
        let mut nvp = self.nv[p];
        for q in pme1..pme2 {
            let i = u(self.iw[q]);
            if let Some(w) = self.update_degree(i, p) {
                // Mass elimination of i with p.
                dk -= w;
                nvp += w;
                self.nel += w;
            }
        }
        self.deg[p] = dk;
        self.supervariables(pme1, pme2);
        // Q7. Finalize: L_p compacted in place to its principal variables.
        let mut kept = pme1;
        for q in pme1..pme2 {
            let i = u(self.iw[q]);
            self.in_lp[i] = false;
            if self.kind[i] == Kind::Var && self.nv[i] > 0 {
                let d = (self.deg[i] + dk - self.nv[i]).min(self.n - self.nel - self.nv[i]);
                self.deg[i] = d;
                self.list_insert(i, d);
                self.mindeg = self.mindeg.min(d);
                self.vw[i] = self.nv[i];
                self.iw[kept] = idx32(i);
                kept += 1;
            }
        }
        self.in_lp[p] = false;
        self.nv[p] = nvp;
        if kept == pme1 {
            self.kind[p] = Kind::Gone;
            self.w[p] = 0;
        }
        self.pe[p] = pme1;
        self.len[p] = kept - pme1;
        self.pfree = kept;
    }

    /// Step Q3: `L_p` written at `pfree` (returned as `iw[pme1..pme2]`)
    /// and its weight; absorbs `E_p`.
    fn new_element(&mut self, p: usize) -> (usize, usize, usize) {
        // Room for the upper bound of |L_p| (garbage collection moves the
        // lists, so the positions are read afterwards).
        let bound = self.len[p] - self.elen[p]
            + self
                .elems(p)
                .iter()
                .filter(|&&e| self.kind[u(e)] == Kind::Elem)
                .map(|&e| self.len[u(e)])
                .sum::<usize>();
        self.ensure(bound);
        let pme1 = self.pfree;
        let (lp0, lp1) = (self.pe[p], self.pe[p] + self.len[p]);
        let split = lp1 - self.elen[p];
        let mut dk = 0;
        // E_p holds the newest elements last (Q5 appends p): AMD visits
        // them newest first; COLAMD visits the initial rows first (in
        // increasing order), then the pivot rows oldest first.
        let n = self.n;
        if self.column_order {
            for q in (split..lp1).rev() {
                let e = self.iw[q];
                if u(e) >= n {
                    dk += self.absorb(u(e), p);
                }
            }
            for q in split..lp1 {
                let e = self.iw[q];
                if u(e) < n {
                    dk += self.absorb(u(e), p);
                }
            }
        } else {
            for q in (split..lp1).rev() {
                let e = self.iw[q];
                dk += self.absorb(u(e), p);
            }
        }
        for q in lp0..split {
            let j = self.iw[q];
            dk += self.add_to_lp(u(j));
        }
        self.kind[p] = Kind::Elem;
        self.elen[p] = 0;
        self.len[p] = 0;
        (pme1, self.pfree, dk)
    }

    /// Absorbs the element `e` into `p` (its variables into `L_p`);
    /// returns the weight added.
    fn absorb(&mut self, e: usize, p: usize) -> usize {
        if self.kind[e] != Kind::Elem {
            return 0;
        }
        let mut dk = 0;
        for q in self.pe[e]..self.pe[e] + self.len[e] {
            let j = self.iw[q];
            dk += self.add_to_lp(u(j));
        }
        self.kind[e] = Kind::Gone;
        self.w[e] = 0;
        self.parent[e] = idx32(p);
        self.len[e] = 0;
        dk
    }

    /// Adds the variable `j` to `L_p` (at `pfree`) if it is a live
    /// principal variable not yet there; returns the weight added.
    fn add_to_lp(&mut self, j: usize) -> usize {
        // `vw[j] > 0` iff `j` is a live principal variable not in `L_p`.
        if j >= self.n || self.vw[j] == 0 {
            return 0;
        }
        debug_assert!(self.kind[j] == Kind::Var && self.nv[j] > 0 && !self.in_lp[j]);
        debug_assert!(self.pfree < self.iw.len(), "room was ensured");
        self.in_lp[j] = true;
        self.vw[j] = 0;
        self.iw[self.pfree] = idx32(j);
        self.pfree += 1;
        self.list_remove(j);
        self.nv[j]
    }

    /// Step Q4 over `L_p = iw[pme1..pme2]`.
    fn set_differences(&mut self, pme1: usize, pme2: usize) {
        // Every w(e) is at most n: a stride of n + 1 leaves all the old
        // marks below the new flag.
        let stride = idx32(self.n + 1);
        if self.wflg >= u32::MAX - 2 * stride {
            // Restart the flag (SuiteSparse's clear_flag): every mark
            // falls back to "not seen" (1), the dead stay 0.
            for x in self.w.iter_mut() {
                if *x != 0 {
                    *x = 1;
                }
            }
            self.wflg = 1;
        }
        self.wflg += stride;
        let wflg = self.wflg;
        // Local slices: the scans are the whole cost of the algorithm, and
        // field accesses through `self` are reloaded at every store.
        let (iw, pe, len, elen, nv, deg, w) = (
            &self.iw[..],
            &self.pe[..],
            &self.len[..],
            &self.elen[..],
            &self.nv[..],
            &self.deg[..],
            &mut self.w[..],
        );
        for &i in &iw[pme1..pme2] {
            let i = u(i);
            let nvi = idx32(nv[i]);
            let end = pe[i] + len[i];
            for &e in &iw[end - elen[i]..end] {
                let e = u(e);
                let mut we = w[e];
                if we == 0 {
                    continue; // dead
                }
                debug_assert_eq!(self.kind[e], Kind::Elem);
                if we < wflg {
                    we = idx32(deg[e]) + wflg;
                }
                w[e] = we.saturating_sub(nvi).max(wflg);
            }
        }
    }

    /// Step Q5 for `i ∈ L_p`; returns `Some(|i|)` if `i` was
    /// mass-eliminated with `p`.
    fn update_degree(&mut self, i: usize, p: usize) -> Option<usize> {
        // Compacted in place: A_i first, then E_i (old order), then p.
        let base = self.pe[i];
        let old_len = self.len[i];
        let split = old_len - self.elen[i];
        let mut d = 0_usize;
        let mut h = 0_usize;
        let mut w = 0;
        let n_vars;
        {
            let (list, vw, we_) = (
                &mut self.iw[base..base + old_len],
                &self.vw[..],
                &mut self.w[..],
            );
            for r in 0..split {
                let j = list[r];
                let ju = u(j);
                debug_assert!(ju < self.n, "A_i holds variables only");
                let wj = vw[ju];
                debug_assert_eq!(
                    wj > 0,
                    self.kind[ju] == Kind::Var && self.nv[ju] > 0 && !self.in_lp[ju]
                );
                if wj > 0 {
                    d += wj;
                    list[w] = j;
                    w += 1;
                    h = h.wrapping_add(ju);
                }
            }
            n_vars = w;
            let wflg = self.wflg;
            let aggressive = self.aggressive;
            for r in split..old_len {
                let e = list[r];
                let eu = u(e);
                let we = we_[eu];
                // Every live element of E_i was seen in Q4; below the flag:
                // dead.
                if we < wflg {
                    debug_assert_ne!(self.kind[eu], Kind::Elem);
                    continue;
                }
                let dext = u(we - wflg);
                if dext > 0 || !aggressive {
                    d += dext;
                    list[w] = e;
                    w += 1;
                    h = h.wrapping_add(eu);
                } else {
                    self.kind[eu] = Kind::Gone;
                    we_[eu] = 0;
                    self.parent[eu] = idx32(p);
                }
            }
        }
        if w == 0 {
            // Only p: indistinguishable from the pivot.
            let weight = self.nv[i];
            self.kind[i] = Kind::Gone;
            self.parent[i] = idx32(p);
            self.nv[i] = 0;
            self.len[i] = 0;
            return Some(weight);
        }
        // p takes a slot that p itself (in A_i) or an absorbed element (in
        // E_i) freed ("Storage"); the relocation below is never taken and
        // only keeps the engine total.
        debug_assert!(
            w < old_len,
            "(Storage): no slot freed for p in the list of {i}"
        );
        if w == old_len {
            self.ensure(w + 1);
            let (src, dst) = (self.pe[i], self.pfree);
            self.iw.copy_within(src..src + w, dst);
            self.pe[i] = dst;
            self.pfree += w + 1;
        }
        self.iw[self.pe[i] + w] = idx32(p);
        self.len[i] = w + 1;
        self.deg[i] = if self.previous_degree {
            self.deg[i].min(d)
        } else {
            d
        };
        self.elen[i] = w + 1 - n_vars;
        let b = h % self.n;
        self.hof[i] = b;
        self.hnext[i] = self.hhead[b];
        self.hhead[b] = idx32(i);
        None
    }

    /// Step Q6 over `L_p = iw[pme1..pme2]`.
    fn supervariables(&mut self, pme1: usize, pme2: usize) {
        for q in pme1..pme2 {
            let i = u(self.iw[q]);
            if self.kind[i] != Kind::Var || self.nv[i] == 0 {
                continue;
            }
            let b = self.hof[i];
            let mut x = self.hhead[b];
            if x == NONE {
                continue;
            }
            self.hhead[b] = NONE;
            while x != NONE && self.hnext[u(x)] != NONE {
                let xu = u(x);
                self.mark_cur += 1;
                let (px, lx) = (self.pe[xu], self.len[xu]);
                for r in px..px + lx - 1 {
                    let v = u(self.iw[r]);
                    self.mark[v] = self.mark_cur;
                }
                let mut last = xu;
                let mut y = self.hnext[xu];
                while y != NONE {
                    let yu = u(y);
                    let (py, ly) = (self.pe[yu], self.len[yu]);
                    let same = ly == lx
                        && self.elen[yu] == self.elen[xu]
                        && self.iw[py..py + ly - 1]
                            .iter()
                            .all(|&v| self.mark[u(v)] == self.mark_cur);
                    let after = self.hnext[yu];
                    if same {
                        self.nv[xu] += self.nv[yu];
                        self.nv[yu] = 0;
                        self.kind[yu] = Kind::Gone;
                        self.parent[yu] = x;
                        self.len[yu] = 0;
                        self.hnext[last] = after;
                    } else {
                        last = yu;
                    }
                    y = after;
                }
                x = self.hnext[xu];
            }
        }
    }

    /// Step Q8′ (COLAMD): the pivots in order of elimination, each
    /// followed by the variables merged into it or eliminated with it.
    fn pivot_order(&self) -> Vec<u32> {
        let n = self.n;
        let mut time = vec![usize::MAX; n];
        for (t, &p) in self.pivots.iter().enumerate() {
            time[u(p)] = t;
        }
        let mut keyed: Vec<(usize, bool, u32)> = Vec::with_capacity(n);
        for i in 0..n {
            if !self.is_pivot[i] && self.parent[i] == NONE {
                continue; // excluded
            }
            let mut a = i;
            while !self.is_pivot[a] {
                a = u(self.parent[a]);
            }
            keyed.push((time[a], !self.is_pivot[i], idx32(i)));
        }
        keyed.sort_unstable();
        keyed.into_iter().map(|k| k.2).collect()
    }

    /// Step Q8 (the degree lists `head`/`next`, dead by now, hold the
    /// child lists).
    fn postorder(&mut self) -> Vec<u32> {
        let n = self.n;
        let first = &mut self.head[..n];
        let sibling = &mut self.next[..];
        first.fill(NONE);
        let mut push = |c: usize, p: u32, first: &mut [u32]| {
            sibling[c] = first[u(p)];
            first[u(p)] = idx32(c);
        };
        // Merged and mass-eliminated variables first (they end up after
        // the elements in each child list), then the absorbed elements.
        for c in (0..n).rev() {
            if !self.is_pivot[c] && self.parent[c] != NONE {
                push(c, self.parent[c], first);
            }
        }
        for c in (0..n).rev() {
            if self.is_pivot[c] && self.parent[c] != NONE {
                push(c, self.parent[c], first);
            }
        }
        let mut out = Vec::with_capacity(n);
        let mut stack = Vec::new();
        for r in 0..n {
            if !self.is_pivot[r] || self.parent[r] != NONE {
                continue;
            }
            stack.push(r);
            while let Some(&x) = stack.last() {
                let c = first[x];
                if c == NONE {
                    stack.pop();
                    out.push(idx32(x));
                } else {
                    first[x] = sibling[u(c)];
                    stack.push(u(c));
                }
            }
        }
        out
    }

    /// Invariants (Q1)–(Q4), and the lists inside the workspace.
    fn check(&self) -> Result<(), String> {
        let n = self.n;
        // The workspace: every live list inside [0, pfree), disjoint.
        let mut owner = vec![NONE; self.pfree];
        for x in 0..self.pe.len() {
            if self.len[x] == 0 {
                continue;
            }
            if self.pe[x] + self.len[x] > self.pfree {
                return Err(format!("(Storage): list of {x} beyond pfree"));
            }
            for slot in &mut owner[self.pe[x]..self.pe[x] + self.len[x]] {
                if *slot != NONE {
                    return Err(format!("(Storage): lists of {} and {x} overlap", *slot));
                }
                *slot = idx32(x);
            }
        }
        // (Q1)
        let mut listed = vec![false; n];
        for (d, &h) in self.head.iter().enumerate() {
            let mut x = h;
            while x != NONE {
                let xu = u(x);
                if listed[xu] || self.kind[xu] != Kind::Var || self.nv[xu] == 0 {
                    return Err(format!("(Q1): node {xu} wrongly listed"));
                }
                if self.deg[xu] != d {
                    return Err(format!(
                        "(Q1): node {xu} in list {d}, degree {}",
                        self.deg[xu]
                    ));
                }
                listed[xu] = true;
                x = self.next[xu];
            }
        }
        let mut live = 0;
        for (i, &is_listed) in listed.iter().enumerate() {
            let principal = self.kind[i] == Kind::Var && self.nv[i] > 0;
            if principal != is_listed {
                return Err(format!(
                    "(Q1): node {i} principal {principal}, listed {is_listed}"
                ));
            }
            if principal {
                live += self.nv[i];
            }
        }
        // (Q3)
        if live + self.nel != n {
            return Err(format!("(Q3): {live} live + {} eliminated ≠ {n}", self.nel));
        }
        let alive = |j: usize| j < n && self.kind[j] == Kind::Var && self.nv[j] > 0;
        // (Q2)
        if self.aggressive {
            for e in 0..self.kind.len() {
                if self.kind[e] == Kind::Elem {
                    let w: usize = self
                        .list(e)
                        .iter()
                        .filter(|&&j| alive(u(j)))
                        .map(|&j| self.nv[u(j)])
                        .sum();
                    if w != self.deg[e] {
                        return Err(format!("(Q2): element {e} weight {} ≠ {w}", self.deg[e]));
                    }
                }
            }
        }
        // (Q4)
        let mut seen = vec![false; n];
        for i in (0..n).filter(|&i| alive(i)) {
            seen.fill(false);
            seen[i] = true;
            let mut exact = 0;
            let mut visit = |j: usize, seen: &mut Vec<bool>| {
                if alive(j) && !seen[j] {
                    seen[j] = true;
                    exact += self.nv[j];
                }
            };
            for &e in self.elems(i) {
                if self.kind[u(e)] == Kind::Elem {
                    for &j in self.list(u(e)) {
                        visit(u(j), &mut seen);
                    }
                }
            }
            for &j in self.vars(i) {
                visit(u(j), &mut seen);
            }
            if self.deg[i] < exact {
                return Err(format!("(Q4): d̄_{i} = {} < exact {exact}", self.deg[i]));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path whose lists are built so that the workspace must be garbage
    /// collected several times (tight initial slack: the engine grows only
    /// after compacting), with the order checked to be perfect.
    #[test]
    fn garbage_collection_keeps_the_order_perfect() {
        let n = 2000;
        let mut adj_ptr = vec![0_usize];
        let mut adj_idx = Vec::new();
        for i in 0..n {
            if i > 0 {
                adj_idx.push(idx32(i - 1));
            }
            if i + 1 < n {
                adj_idx.push(idx32(i + 1));
            }
            adj_ptr.push(adj_idx.len());
        }
        let mut arena = Arena::new(n, n);
        let mut g = Graph::new(
            Input {
                n,
                adj_ptr,
                adj_idx,
                elem_ptr: vec![0],
                elem_idx: Vec::new(),
                excluded: vec![false; n],
                aggressive: true,
                previous_degree: true,
                column_order: false,
            },
            &mut arena,
        );
        // Shrink the slack so that the first new elements already run out.
        g.iw.truncate(g.pfree + 2);
        while g.nel < g.n {
            let p = g.select();
            g.eliminate(p);
            assert_eq!(g.check(), Ok(()));
        }
        assert!(g.ncmpa > 0, "no garbage collection happened");
        let post = g.postorder();
        let perm = crate::Perm::from_fwd(post).unwrap();
        let mut e = Vec::new();
        for i in 1..n {
            e.push((i - 1, i));
        }
        let mut cols: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(a, b) in &e {
            cols[a].push(idx32(b));
            cols[b].push(idx32(a));
        }
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        for mut c in cols {
            c.sort_unstable();
            row_idx.extend(c);
            col_ptr.push(row_idx.len());
        }
        let s = crate::Pattern::from_raw_parts(n, col_ptr, row_idx).unwrap();
        assert_eq!(s.cholesky_nnz(&perm).unwrap(), 2 * n - 1);
    }
}
