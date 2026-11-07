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
//! In the code `list[i]` holds `E_i` (its first `elen[i]` entries) followed
//! by `A_i`, and `list[e]` holds `L_e`.
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
//! minimum degree. Lists are separate vectors (no garbage collection).
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

use std::mem;

/// "No node".
const NONE: u32 = u32::MAX;

/// Graphs at most this large are checked after every pivot in debug
/// builds (the check costs `O(n²)`).
const PARANOID_N: usize = 64;

/// What the engine is asked to order.
pub(crate) struct Input {
    /// Number of variables `n`.
    pub(crate) n: usize,
    /// `A_i` for each variable: original neighbours, no self loop, no
    /// duplicate (empty for COLAMD).
    pub(crate) adj: Vec<Vec<u32>>,
    /// The initial elements `L_e` (empty for AMD), without duplicates or
    /// excluded variables.
    pub(crate) elems: Vec<Vec<u32>>,
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

#[allow(clippy::struct_excessive_bools)] // independent switches of one algorithm
struct Graph {
    n: usize,
    list: Vec<Vec<u32>>,
    elen: Vec<usize>,
    kind: Vec<Kind>,
    /// `|i|` of principal variables (`0` otherwise).
    nv: Vec<usize>,
    /// Variables: `d̄_i`; elements: `|L_e|`.
    deg: Vec<usize>,
    parent: Vec<u32>,
    is_pivot: Vec<bool>,
    head: Vec<u32>,
    next: Vec<u32>,
    prev: Vec<u32>,
    mindeg: usize,
    nel: usize,
    in_lp: Vec<bool>,
    stamp: Vec<usize>,
    diff: Vec<usize>,
    cur: usize,
    mark: Vec<usize>,
    mark_cur: usize,
    hhead: Vec<u32>,
    hnext: Vec<u32>,
    hof: Vec<usize>,
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
    let mut g = Graph::new(input);
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

impl Graph {
    /// Step Q1.
    fn new(input: Input) -> Self {
        let Input {
            n,
            adj,
            elems,
            excluded,
            aggressive,
            previous_degree,
            column_order,
        } = input;
        debug_assert_eq!(adj.len(), n);
        debug_assert_eq!(excluded.len(), n);
        let ne = elems.len();
        let total = n + ne;
        let mut einc: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut deg = vec![0_usize; total];
        let mut kind = vec![Kind::Var; total];
        let mut list: Vec<Vec<u32>> = Vec::with_capacity(total);
        for (e, l) in elems.iter().enumerate() {
            for &v in l {
                debug_assert!(!excluded[u(v)], "excluded variable in an element");
                einc[u(v)].push(idx32(n + e));
            }
        }
        let mut elen = vec![0_usize; n];
        for (i, a) in adj.into_iter().enumerate() {
            elen[i] = einc[i].len();
            let mut l = mem::take(&mut einc[i]);
            l.extend(a);
            list.push(l);
        }
        for (e, l) in elems.into_iter().enumerate() {
            deg[n + e] = l.len();
            kind[n + e] = if l.is_empty() { Kind::Gone } else { Kind::Elem };
            list.push(l);
        }
        let mut g = Self {
            n,
            list,
            elen,
            kind,
            nv: vec![1; n],
            deg,
            parent: vec![NONE; total],
            is_pivot: vec![false; n],
            head: vec![NONE; n + 1],
            next: vec![NONE; n],
            prev: vec![NONE; n],
            mindeg: 0,
            nel: 0,
            in_lp: vec![false; n],
            stamp: vec![0; total],
            diff: vec![0; total],
            cur: 0,
            mark: vec![0; total],
            mark_cur: 0,
            hhead: vec![NONE; n.max(1)],
            hnext: vec![NONE; n],
            hof: vec![0; n],
            aggressive,
            previous_degree,
            column_order,
            pivots: Vec::new(),
            paranoid: cfg!(debug_assertions) && n <= PARANOID_N,
        };
        for (i, &ex) in excluded.iter().enumerate() {
            if ex {
                g.kind[i] = Kind::Gone;
                g.nv[i] = 0;
                g.nel += 1;
                continue;
            }
            let ei = g.elen[i];
            let from_elems: usize = g.list[i][..ei]
                .iter()
                .map(|&e| g.deg[u(e)].saturating_sub(1))
                .sum();
            let d = g.list[i].len() - ei + from_elems;
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

    /// Steps Q3–Q7 for the pivot `p`.
    fn eliminate(&mut self, p: usize) {
        self.is_pivot[p] = true;
        self.pivots.push(idx32(p));
        self.in_lp[p] = true;
        let (lp, mut dk) = self.new_element(p);
        self.set_differences(&lp);
        let mut nvp = self.nv[p];
        for &i in &lp {
            if let Some(w) = self.update_degree(u(i), p) {
                // Mass elimination of i with p.
                dk -= w;
                nvp += w;
                self.nel += w;
            }
        }
        self.deg[p] = dk;
        self.supervariables(&lp);
        // Q7. Finalize.
        let mut kept = Vec::with_capacity(lp.len());
        for &i in &lp {
            let i = u(i);
            self.in_lp[i] = false;
            if self.kind[i] == Kind::Var && self.nv[i] > 0 {
                let d = (self.deg[i] + dk - self.nv[i]).min(self.n - self.nel - self.nv[i]);
                self.deg[i] = d;
                self.list_insert(i, d);
                self.mindeg = self.mindeg.min(d);
                kept.push(idx32(i));
            }
        }
        self.in_lp[p] = false;
        self.nv[p] = nvp;
        if kept.is_empty() {
            self.kind[p] = Kind::Gone;
        }
        self.list[p] = kept;
    }

    /// Step Q3: `L_p` and its weight; absorbs `E_p`.
    fn new_element(&mut self, p: usize) -> (Vec<u32>, usize) {
        let lst = mem::take(&mut self.list[p]);
        let ep = self.elen[p];
        let mut lp = Vec::new();
        let mut dk = 0;
        // E_p holds the newest elements first (Q5 puts p in front); COLAMD
        // visits the initial rows first, then the pivot rows oldest first.
        let order: Vec<u32> = if self.column_order {
            let (n, e) = (self.n, &lst[..ep]);
            e.iter()
                .copied()
                .filter(|&x| u(x) >= n)
                .chain(e.iter().rev().copied().filter(|&x| u(x) < n))
                .collect()
        } else {
            lst[..ep].to_vec()
        };
        for &e in &order {
            let e = u(e);
            if self.kind[e] != Kind::Elem {
                continue;
            }
            let le = mem::take(&mut self.list[e]);
            for &j in &le {
                dk += self.add_to_lp(u(j), &mut lp);
            }
            self.kind[e] = Kind::Gone;
            self.parent[e] = idx32(p);
        }
        for &j in &lst[ep..] {
            dk += self.add_to_lp(u(j), &mut lp);
        }
        self.kind[p] = Kind::Elem;
        self.elen[p] = 0;
        (lp, dk)
    }

    /// Adds the variable `j` to `L_p` if it is a live principal variable
    /// not yet there; returns the weight added.
    fn add_to_lp(&mut self, j: usize, lp: &mut Vec<u32>) -> usize {
        if j >= self.n || self.kind[j] != Kind::Var || self.nv[j] == 0 || self.in_lp[j] {
            return 0;
        }
        self.in_lp[j] = true;
        lp.push(idx32(j));
        self.list_remove(j);
        self.nv[j]
    }

    /// Step Q4.
    fn set_differences(&mut self, lp: &[u32]) {
        self.cur += 1;
        for &i in lp {
            let i = u(i);
            let w = self.nv[i];
            for &e in &self.list[i][..self.elen[i]] {
                let e = u(e);
                if self.kind[e] != Kind::Elem {
                    continue;
                }
                if self.stamp[e] != self.cur {
                    self.stamp[e] = self.cur;
                    self.diff[e] = self.deg[e];
                }
                self.diff[e] = self.diff[e].saturating_sub(w);
            }
        }
    }

    /// Step Q5 for `i ∈ L_p`; returns `Some(|i|)` if `i` was
    /// mass-eliminated with `p`.
    fn update_degree(&mut self, i: usize, p: usize) -> Option<usize> {
        let old = mem::take(&mut self.list[i]);
        let ei = self.elen[i];
        let mut new = Vec::with_capacity(old.len() + 1);
        new.push(idx32(p));
        let mut d = 0_usize;
        let mut h = 0_usize;
        for &e in &old[..ei] {
            let eu = u(e);
            if self.kind[eu] != Kind::Elem {
                continue;
            }
            let dext = self.diff[eu];
            if dext > 0 || !self.aggressive {
                d += dext;
                new.push(e);
                h = h.wrapping_add(eu);
            } else {
                self.kind[eu] = Kind::Gone;
                self.parent[eu] = idx32(p);
            }
        }
        let n_elems = new.len();
        for &j in &old[ei..] {
            let ju = u(j);
            if ju < self.n && self.kind[ju] == Kind::Var && self.nv[ju] > 0 && !self.in_lp[ju] {
                d += self.nv[ju];
                new.push(j);
                h = h.wrapping_add(ju);
            }
        }
        if new.len() == 1 {
            let w = self.nv[i];
            self.kind[i] = Kind::Gone;
            self.parent[i] = idx32(p);
            self.nv[i] = 0;
            return Some(w);
        }
        self.deg[i] = if self.previous_degree {
            self.deg[i].min(d)
        } else {
            d
        };
        self.elen[i] = n_elems;
        self.list[i] = new;
        let b = h % self.n;
        self.hof[i] = b;
        self.hnext[i] = self.hhead[b];
        self.hhead[b] = idx32(i);
        None
    }

    /// Step Q6.
    fn supervariables(&mut self, lp: &[u32]) {
        for &i in lp {
            let i = u(i);
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
                for &v in &self.list[xu][1..] {
                    self.mark[u(v)] = self.mark_cur;
                }
                let mut last = xu;
                let mut y = self.hnext[xu];
                while y != NONE {
                    let yu = u(y);
                    let same = self.list[yu].len() == self.list[xu].len()
                        && self.elen[yu] == self.elen[xu]
                        && self.list[yu][1..]
                            .iter()
                            .all(|&v| self.mark[u(v)] == self.mark_cur);
                    let after = self.hnext[yu];
                    if same {
                        self.nv[xu] += self.nv[yu];
                        self.nv[yu] = 0;
                        self.kind[yu] = Kind::Gone;
                        self.parent[yu] = x;
                        self.list[yu] = Vec::new();
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

    /// Step Q8.
    fn postorder(&self) -> Vec<u32> {
        let n = self.n;
        let mut first = vec![NONE; n];
        let mut sibling = vec![NONE; n];
        let mut push = |c: usize, p: u32, first: &mut Vec<u32>| {
            sibling[c] = first[u(p)];
            first[u(p)] = idx32(c);
        };
        // Merged and mass-eliminated variables first (they end up after
        // the elements in each child list), then the absorbed elements.
        for c in (0..n).rev() {
            if !self.is_pivot[c] && self.parent[c] != NONE {
                push(c, self.parent[c], &mut first);
            }
        }
        for c in (0..n).rev() {
            if self.is_pivot[c] && self.parent[c] != NONE {
                push(c, self.parent[c], &mut first);
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

    /// Invariants (Q1)–(Q4).
    fn check(&self) -> Result<(), String> {
        let n = self.n;
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
                    let w: usize = self.list[e]
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
            for &e in &self.list[i][..self.elen[i]] {
                if self.kind[u(e)] == Kind::Elem {
                    for &j in &self.list[u(e)] {
                        visit(u(j), &mut seen);
                    }
                }
            }
            for &j in &self.list[i][self.elen[i]..] {
                visit(u(j), &mut seen);
            }
            if self.deg[i] < exact {
                return Err(format!("(Q4): d̄_{i} = {} < exact {exact}", self.deg[i]));
            }
        }
        Ok(())
    }
}
