//! # Chapter 5.2 — Dynamic sparse storage for the factorization
//!
//! ## Purpose
//! Both the Markowitz elimination (Chapter 5.3) and the Forrest–Tomlin update
//! (Chapter 5.5) change sparse matrices *in place*: entries disappear when a
//! row or column is pivoted out, fill-in appears in arbitrary lines, and a
//! whole column of `U` is replaced by a spike. A static compressed format
//! (Chapter 5.8) cannot absorb that, so this module provides the two classic
//! dynamic structures of sparse direct solvers:
//!
//! * [`LineStore`] — a set of sparse *lines* (rows or columns) sharing one
//!   pair of arrays, each line owning a contiguous segment with spare room
//!   (the "row file"/"column file" of MA28 and of Suhl & Suhl);
//! * [`CountLists`] — doubly linked lists bucketing lines by their current
//!   nonzero count, so that the Markowitz search visits the lines of smallest
//!   count first in `O(1)` per step.
//!
//! ## Notation
//! A [`LineStore`] holds `n` lines. Line `ℓ` occupies `idx[s_ℓ .. s_ℓ + len_ℓ]`
//! and `val[…]` with capacity `cap_ℓ ≥ len_ℓ`; positions inside a line are
//! relative (`0 .. len_ℓ`) and survive relocation. `live = Σ_ℓ len_ℓ` and
//! `garbage` counts the array slots abandoned by relocated lines.
//!
//! ## Algorithm G (Growing a full line)
//! To append to line `ℓ` when `len_ℓ = cap_ℓ`:
//!
//! G1. [At the end?] If the segment of `ℓ` is the last one of the arrays,
//!     extend the arrays to `cap_ℓ ← max(4, 2 cap_ℓ)` and stop.
//! G2. [Compact?] If `garbage > max(64, live)`, run Algorithm K; if `ℓ` now
//!     has room, stop.
//! G3. [Relocate.] Append a segment of capacity `max(4, 2 len_ℓ)` at the end
//!     of the arrays, copy the line there and add the old capacity to
//!     `garbage`.
//!
//! ## Algorithm K (Compaction)
//! K1. Rebuild the arrays visiting the lines in index order, giving each
//!     line the capacity `len_ℓ + max(2, len_ℓ / 2)`; set `garbage ← 0`.
//!
//! Doubling makes the cost of appends amortized `O(1)`; compaction is
//! triggered only when the abandoned space exceeds the live data, so the
//! arrays never hold more than about three times `live` slots.
//!
//! ## Cross-linked files (task P3.37)
//! The Markowitz elimination keeps the active submatrix twice, in a column
//! file `C` (with the values) and a row file `R` (the pattern), and every
//! pivot deletes entries from both. Finding the entry `(i, j)` in line `i`
//! of `R` by a linear search costs `O(r_i)`; summed over an elimination it
//! is `O(Σ_i r_i²)`, which dominates on bases with dense rows (task P3.37:
//! on `hypothyroid-k1`, `m = 5 258`, 52% of a factorization was that
//! search). A *linked* store therefore carries, for every entry, the
//! relative position of its twin in the other file:
//!
//! * the entry at position `q` of line `j` of `C` is `(i, s_ij)` with
//!   `link_C(j, q) = r`, the position of `j` in line `i` of `R`;
//! * the entry at position `r` of line `i` of `R` is `j` with
//!   `link_R(i, r) = q`.
//!
//! A link is a *hint*: the twin of an entry of line `ℓ` with index `k` is
//! looked for at the position its link names, and is there iff the index
//! stored there is `ℓ` (a line holds an index at most once, so a match
//! cannot be another entry); otherwise the line of the twin is searched.
//! Keeping every hint exact would cost a write at a random place of the
//! other file at every removal, which on the many short lines of a
//! simplex basis costs more than the searches it saves; so the hints are
//! kept exact only for the entries of *long* lines, `len ≥ L_link`
//! ([`LINK_MIN`] = 16), the only ones whose search is expensive.
//!
//! ### Algorithm T (Removal through the twin links)
//! To find the entry with index `k` in line `ℓ` from the hint `t`: if
//! `t < len_ℓ` and the entry at `t` has index `k`, it is at `t`; otherwise
//! search the line ([`LineStore::find`]). To remove the entry at position
//! `q` of line `ℓ` of a linked store `X` whose twin file is `Y`:
//!
//! T1. [Move the last.] As in a plain removal, move the last entry of line
//!     `ℓ` (index `k`, link `t`) into position `q` and shorten the line.
//! T2. [Fix its twin.] If an entry moved (`q` was not the last position)
//!     and line `ℓ` still has at least `L_link` entries, find the twin of
//!     the moved entry in line `k` of `Y` from its hint `t` (exact or not)
//!     and set its link to `q`.
//!
//! Removing an entry from both files is T1–T2 on `C` at `(j, q)` and then,
//! with the twin found from the hint `link_C(j, q)` read *before* T1, on
//! `R`. Removing it from one file only is allowed when the line of its twin
//! is about to be cleared as a whole (a pivot column or row, a dead
//! column). A removal costs `O(1)` when the hint is exact, which it is in
//! every line that was long at its last moves, and a search of a short
//! line otherwise. Either way it leaves every line in *exactly* the order
//! the search-based removal would — the position found is the same unique
//! position — so the elimination makes the same choices, bit for bit
//! (task P3.37, *path identity*).
//!
//! ## Invariants
//! (S1) Segments `[s_ℓ, s_ℓ + cap_ℓ)` of lines with `cap_ℓ > 0` are pairwise
//!      disjoint and inside the arrays; `len_ℓ ≤ cap_ℓ`.
//! (S2) `live = Σ_ℓ len_ℓ`.
//! (S3) For a linked pair `(C, R)` between two operations of the
//!      elimination: both files hold the same entries — `(i, j)` is in
//!      line `j` of `C` iff `j` is in line `i` of `R` — and the hint of an
//!      entry of a line of `C` (of `R`) with at least `L_link` entries
//!      since its last move is exact ([`check_links`] checks the first part
//!      and the exactness of the hints into lines that never were short).
//! (L1) In a [`CountLists`], an element is in at most one bucket, the bucket
//!      of its recorded count, and `prev`/`next` are mutually consistent.
//!
//! ## References
//! Duff, Erisman & Reid, *Direct Methods for Sparse Matrices* (1986), §2.10
//! and §9.4 (row/column files, compaction); Suhl & Suhl, "Computing sparse
//! LU factorizations for large-scale linear programming bases", ORSA J.
//! Comput. 2(4), 1990, §3 (count lists); Maros, *Computational Techniques of
//! the Simplex Method* (2003), §5.4. The cross links are the usual device
//! of storing with each entry of one file a pointer to its twin in the
//! other (no section of these references is followed literally here).

use crate::real::Real;

/// Sentinel "no element" for the linked lists and scatter maps.
pub(crate) const NIL: usize = usize::MAX;

/// `L_link` of Algorithm T: the hints of the twins of the entries that move
/// in a line are kept exact only while the line has at least this many
/// entries (shorter lines are searched).
pub(crate) const LINK_MIN: usize = 16;

/// A set of sparse lines with in-place growth (Algorithms G and K),
/// optionally *linked* to a twin file (Algorithm T).
#[derive(Clone, Debug)]
pub(crate) struct LineStore<T: Real> {
    start: Vec<usize>,
    len: Vec<usize>,
    cap: Vec<usize>,
    idx: Vec<u32>,
    val: Vec<T>,
    /// `link[p]`: position of the twin of the entry in slot `p` within its
    /// line of the twin file; empty for an unlinked store.
    link: Vec<u32>,
    /// The store is one file of a linked pair (Algorithm T).
    linked: bool,
    /// The store keeps values (`false`: a pattern, `val` stays empty).
    has_val: bool,
    live: usize,
    garbage: usize,
}

impl<T: Real> Default for LineStore<T> {
    fn default() -> Self {
        Self {
            start: Vec::new(),
            len: Vec::new(),
            cap: Vec::new(),
            idx: Vec::new(),
            val: Vec::new(),
            link: Vec::new(),
            linked: false,
            has_val: false,
            live: 0,
            garbage: 0,
        }
    }
}

impl<T: Real> LineStore<T> {
    /// `caps.len()` empty lines; line `ℓ` gets the initial capacity `caps[ℓ]`.
    pub(crate) fn new(caps: &[usize]) -> Self {
        let mut start = Vec::with_capacity(caps.len());
        let mut s = 0;
        for &c in caps {
            start.push(s);
            s += c;
        }
        Self {
            start,
            len: vec![0; caps.len()],
            cap: caps.to_vec(),
            idx: vec![0; s],
            val: vec![T::ZERO; s],
            link: Vec::new(),
            linked: false,
            has_val: true,
            live: 0,
            garbage: 0,
        }
    }

    /// Like [`LineStore::new`], for one file of a linked pair (Algorithm T):
    /// fill it with [`LineStore::push_linked`] only.
    pub(crate) fn new_linked(caps: &[usize]) -> Self {
        let mut s = Self::new(caps);
        s.link = vec![0; s.idx.len()];
        s.linked = true;
        s
    }

    /// Like [`LineStore::new_linked`], for a file that holds a pattern only
    /// (no values: [`LineStore::indices`] reads it, the values given to
    /// [`LineStore::push_linked`] are dropped).
    pub(crate) fn new_pattern_linked(caps: &[usize]) -> Self {
        let mut s = Self::new_linked(caps);
        s.val = Vec::new();
        s.has_val = false;
        s
    }

    /// Indices of line `l`.
    #[inline]
    pub(crate) fn indices(&self, l: usize) -> &[u32] {
        &self.idx[self.start[l]..self.start[l] + self.len[l]]
    }

    /// The link of the entry at relative position `pos` of line `l`.
    #[inline]
    pub(crate) fn link(&self, l: usize, pos: usize) -> usize {
        debug_assert!(pos < self.len[l]);
        self.link[self.start[l] + pos] as usize
    }

    /// Sets the link of the entry at relative position `pos` of line `l`.
    #[inline]
    fn set_link(&mut self, l: usize, pos: usize, t: usize) {
        debug_assert!(pos < self.len[l]);
        self.link[self.start[l] + pos] = u32::try_from(t).expect("position exceeds u32::MAX");
    }

    /// Index of the entry at relative position `pos` of line `l`.
    #[inline]
    pub(crate) fn index_at(&self, l: usize, pos: usize) -> u32 {
        debug_assert!(pos < self.len[l]);
        self.idx[self.start[l] + pos]
    }

    /// Value of the entry at relative position `pos` of line `l`.
    #[inline]
    pub(crate) fn value_at(&self, l: usize, pos: usize) -> T {
        debug_assert!(pos < self.len[l] && self.has_val);
        self.val[self.start[l] + pos]
    }

    /// The position of index `i` in line `l`, looked for at the hint `t`
    /// first (Algorithm T), else by a search; `None` if absent.
    #[inline]
    pub(crate) fn find(&self, l: usize, i: u32, t: usize) -> Option<usize> {
        if t < self.len[l] && self.idx[self.start[l] + t] == i {
            Some(t)
        } else {
            self.position(l, i)
        }
    }

    /// Appends `(i, v)` with the link `t` to line `l` of a linked store
    /// (the caller guarantees `i` is absent and appends the twin).
    pub(crate) fn push_linked(&mut self, l: usize, i: u32, v: T, t: usize) {
        debug_assert!(self.linked);
        debug_assert!(
            self.position(l, i).is_none(),
            "duplicate index {i} in line {l}"
        );
        if self.len[l] == self.cap[l] {
            self.grow(l);
        }
        let p = self.start[l] + self.len[l];
        self.idx[p] = i;
        if self.has_val {
            self.val[p] = v;
        }
        self.link[p] = u32::try_from(t).expect("position exceeds u32::MAX");
        self.len[l] += 1;
        self.live += 1;
    }

    /// Total number of stored entries, `live`.
    #[inline]
    pub(crate) fn live(&self) -> usize {
        self.live
    }

    /// Number of entries of line `l`.
    #[inline]
    pub(crate) fn len(&self, l: usize) -> usize {
        self.len[l]
    }

    /// Indices and values of line `l` (of a store with values).
    #[inline]
    pub(crate) fn line(&self, l: usize) -> (&[u32], &[T]) {
        debug_assert!(self.has_val);
        let r = self.start[l]..self.start[l] + self.len[l];
        (&self.idx[r.clone()], &self.val[r])
    }

    /// Mutable values of line `l`.
    #[inline]
    pub(crate) fn values_mut(&mut self, l: usize) -> &mut [T] {
        debug_assert!(self.has_val);
        let r = self.start[l]..self.start[l] + self.len[l];
        &mut self.val[r]
    }

    /// Relative position of index `i` in line `l`, by linear search.
    #[inline]
    pub(crate) fn position(&self, l: usize, i: u32) -> Option<usize> {
        self.indices(l).iter().position(|&x| x == i)
    }

    /// Value of index `i` in line `l`, if stored.
    #[inline]
    pub(crate) fn get(&self, l: usize, i: u32) -> Option<T> {
        self.position(l, i).map(|p| self.line(l).1[p])
    }

    /// Appends `(i, v)` to line `l` (the caller guarantees `i` is absent).
    pub(crate) fn push(&mut self, l: usize, i: u32, v: T) {
        debug_assert!(
            self.position(l, i).is_none(),
            "duplicate index {i} in line {l}"
        );
        if self.len[l] == self.cap[l] {
            self.grow(l);
        }
        let p = self.start[l] + self.len[l];
        self.idx[p] = i;
        if self.has_val {
            self.val[p] = v;
        }
        self.len[l] += 1;
        self.live += 1;
    }

    /// Removes the entry at relative position `pos` of line `l` by moving
    /// the last entry of the line into its place; returns the removed pair.
    pub(crate) fn swap_remove(&mut self, l: usize, pos: usize) -> (u32, T) {
        debug_assert!(pos < self.len[l]);
        let s = self.start[l];
        let p = s + pos;
        let last = s + self.len[l] - 1;
        let out = (
            self.idx[p],
            if self.has_val { self.val[p] } else { T::ZERO },
        );
        self.idx[p] = self.idx[last];
        if self.has_val {
            self.val[p] = self.val[last];
        }
        if self.linked {
            self.link[p] = self.link[last];
        }
        self.len[l] -= 1;
        self.live -= 1;
        out
    }

    /// Removes index `i` from line `l`, returning its value if it was stored.
    pub(crate) fn remove(&mut self, l: usize, i: u32) -> Option<T> {
        self.position(l, i).map(|p| self.swap_remove(l, p).1)
    }

    /// Empties line `l` (its capacity is kept).
    pub(crate) fn clear(&mut self, l: usize) {
        self.live -= self.len[l];
        self.len[l] = 0;
    }

    /// Algorithm G.
    fn grow(&mut self, l: usize) {
        let end = self.idx.len();
        // G1. The line is the last segment: extend the arrays in place.
        if self.cap[l] > 0 && self.start[l] + self.cap[l] == end {
            let new_cap = (2 * self.cap[l]).max(4);
            self.idx.resize(self.start[l] + new_cap, 0);
            if self.has_val {
                self.val.resize(self.start[l] + new_cap, T::ZERO);
            }
            if self.linked {
                self.link.resize(self.start[l] + new_cap, 0);
            }
            self.cap[l] = new_cap;
            return;
        }
        // G2. Too much garbage: compact, which leaves room in every line.
        if self.garbage > self.live.max(64) {
            self.compact();
            if self.len[l] < self.cap[l] {
                return;
            }
        }
        // G3. Relocate the line to a fresh segment at the end.
        let new_cap = (2 * self.len[l]).max(4);
        let new_start = self.idx.len();
        self.idx.resize(new_start + new_cap, 0);
        let s = self.start[l];
        self.idx.copy_within(s..s + self.len[l], new_start);
        if self.has_val {
            self.val.resize(new_start + new_cap, T::ZERO);
            self.val.copy_within(s..s + self.len[l], new_start);
        }
        if self.linked {
            self.link.resize(new_start + new_cap, 0);
            self.link.copy_within(s..s + self.len[l], new_start);
        }
        self.garbage += self.cap[l];
        self.start[l] = new_start;
        self.cap[l] = new_cap;
    }

    /// Algorithm K.
    fn compact(&mut self) {
        let total: usize = self.len.iter().map(|&n| n + (n / 2).max(2)).sum();
        let mut idx = Vec::with_capacity(total);
        let mut val = Vec::with_capacity(if self.has_val { total } else { 0 });
        let mut link = Vec::with_capacity(if self.linked { total } else { 0 });
        for l in 0..self.start.len() {
            let (s, n) = (self.start[l], self.len[l]);
            let cap = n + (n / 2).max(2);
            self.start[l] = idx.len();
            self.cap[l] = cap;
            idx.extend_from_slice(&self.idx[s..s + n]);
            idx.resize(idx.len() + cap - n, 0);
            if self.has_val {
                val.extend_from_slice(&self.val[s..s + n]);
                val.resize(val.len() + cap - n, T::ZERO);
            }
            if self.linked {
                link.extend_from_slice(&self.link[s..s + n]);
                link.resize(link.len() + cap - n, 0);
            }
        }
        self.idx = idx;
        self.val = val;
        self.link = link;
        self.garbage = 0;
    }

    /// Checks (S1)–(S2); used by the tests.
    #[cfg(test)]
    pub(crate) fn check(&self) -> bool {
        let mut segs: Vec<(usize, usize)> = (0..self.start.len())
            .filter(|&l| self.cap[l] > 0)
            .map(|l| (self.start[l], self.start[l] + self.cap[l]))
            .collect();
        segs.sort_unstable();
        let disjoint = segs.windows(2).all(|w| w[0].1 <= w[1].0);
        let inside = segs.last().is_none_or(|s| s.1 <= self.idx.len());
        let lens = (0..self.start.len()).all(|l| self.len[l] <= self.cap[l]);
        disjoint && inside && lens && self.live == self.len.iter().sum::<usize>()
    }
}

/// Algorithm T: removes the entry at relative position `pos` of line `l` of
/// the linked store `x` (T1) and, if line `l` is still long, repoints the
/// twin of the entry that moved into its place (T2) in the twin file `y`.
/// Returns the index of the removed entry and its link (the hint of its own
/// twin in `y`, which the caller removes too unless that line is about to
/// be cleared).
pub(crate) fn remove_linked<T: Real>(
    x: &mut LineStore<T>,
    y: &mut LineStore<T>,
    l: usize,
    pos: usize,
) -> (u32, usize) {
    debug_assert!(x.linked && y.linked);
    let t = x.link(l, pos);
    let last = x.len[l] - 1;
    let (i, _) = x.swap_remove(l, pos);
    if pos != last && last >= LINK_MIN {
        // T2. The entry of the last slot moved to `pos`; its twin is found
        // from its hint (exact or not) and pointed at the new position.
        let (k, tk) = (x.index_at(l, pos) as usize, x.link(l, pos));
        let l32 = u32::try_from(l).expect("line fits in u32");
        let tk = y
            .find(k, l32, tk)
            .expect("the twin of a linked entry exists");
        y.set_link(k, tk, pos);
    }
    (i, t)
}

/// Checks invariant (S3) of the linked pair `(c, r)` (Algorithm T): the
/// same entries in both files, and — with `exact` (no line ever shorter
/// than [`LINK_MIN`] at a move, e.g. right after the load) — exact hints.
/// Used by the tests and by debug builds.
pub(crate) fn check_links<T: Real>(
    c: &LineStore<T>,
    r: &LineStore<T>,
    exact: bool,
) -> Result<(), String> {
    if !c.linked || !r.linked {
        return Err("store not linked".into());
    }
    if c.live != r.live {
        return Err(format!("{} entries by columns, {} by rows", c.live, r.live));
    }
    for j in 0..c.start.len() {
        for q in 0..c.len[j] {
            let i = c.index_at(j, q) as usize;
            let j32 = u32::try_from(j).expect("line fits in u32");
            if i >= r.start.len() {
                return Err(format!("entry ({j}, {q}) names row {i}, out of range"));
            }
            let Some(t) = r.find(i, j32, c.link(j, q)) else {
                return Err(format!("entry ({i}, {j}) has no twin"));
            };
            if exact && (t != c.link(j, q) || r.link(i, t) != q) {
                return Err(format!("entry ({i}, {j}): hints not exact"));
            }
        }
    }
    Ok(())
}

/// Elements bucketed by count in doubly linked lists (Suhl & Suhl 1990).
#[derive(Clone, Debug, Default)]
pub(crate) struct CountLists {
    head: Vec<usize>,
    next: Vec<usize>,
    prev: Vec<usize>,
    count: Vec<usize>,
    size: usize,
}

impl CountLists {
    /// `n` elements, none in a list; counts range over `0..=max_count`.
    pub(crate) fn new(n: usize, max_count: usize) -> Self {
        Self {
            head: vec![NIL; max_count + 1],
            next: vec![NIL; n],
            prev: vec![NIL; n],
            count: vec![NIL; n],
            size: 0,
        }
    }

    /// Number of elements currently in some list.
    #[inline]
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Inserts `e` (not in any list) at the head of bucket `c`.
    pub(crate) fn insert(&mut self, e: usize, c: usize) {
        debug_assert_eq!(self.count[e], NIL, "element {e} already listed");
        let h = self.head[c];
        self.count[e] = c;
        self.prev[e] = NIL;
        self.next[e] = h;
        if h != NIL {
            self.prev[h] = e;
        }
        self.head[c] = e;
        self.size += 1;
    }

    /// Removes `e` from its bucket.
    pub(crate) fn remove(&mut self, e: usize) {
        let c = self.count[e];
        debug_assert_ne!(c, NIL, "element {e} not listed");
        let (p, n) = (self.prev[e], self.next[e]);
        if p == NIL {
            self.head[c] = n;
        } else {
            self.next[p] = n;
        }
        if n != NIL {
            self.prev[n] = p;
        }
        self.count[e] = NIL;
        self.size -= 1;
    }

    /// Moves the listed element `e` to bucket `c`.
    pub(crate) fn set(&mut self, e: usize, c: usize) {
        if self.count[e] != c {
            self.remove(e);
            self.insert(e, c);
        }
    }

    /// `true` iff `e` is in some list.
    #[inline]
    pub(crate) fn contains(&self, e: usize) -> bool {
        self.count[e] != NIL
    }

    /// First element of bucket `c`.
    #[inline]
    pub(crate) fn first(&self, c: usize) -> Option<usize> {
        let h = self.head.get(c).copied().unwrap_or(NIL);
        (h != NIL).then_some(h)
    }

    /// Element after `e` in its bucket.
    #[inline]
    pub(crate) fn next(&self, e: usize) -> Option<usize> {
        let n = self.next[e];
        (n != NIL).then_some(n)
    }

    /// Checks (L1); used by the tests.
    #[cfg(test)]
    pub(crate) fn check(&self) -> bool {
        let mut seen = 0;
        for (c, &h) in self.head.iter().enumerate() {
            let mut e = h;
            let mut p = NIL;
            while e != NIL {
                if self.count[e] != c || self.prev[e] != p {
                    return false;
                }
                seen += 1;
                p = e;
                e = self.next[e];
            }
        }
        seen == self.size && self.count.iter().filter(|&&c| c != NIL).count() == self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::approx_eq;

    #[test]
    fn line_store_grows_relocates_and_compacts() {
        let mut s = LineStore::<f64>::new(&[1, 0, 2]);
        // Many pushes force relocations of every line and compactions.
        for k in 0..500_u32 {
            let l = (k % 3) as usize;
            s.push(l, k, f64::from(k));
            if k % 7 == 0 {
                let (i, _) = s.swap_remove(l, 0);
                assert!(s.position(l, i).is_none());
            }
            assert!(s.check());
        }
        for l in 0..3 {
            let (idx, val) = s.line(l);
            for (&i, &v) in idx.iter().zip(val) {
                assert_eq!(i % 3, u32::try_from(l).unwrap());
                assert!(approx_eq(v, f64::from(i), 0.0));
            }
        }
        let n1 = s.len(1);
        s.clear(1);
        assert_eq!(s.live(), s.len(0) + s.len(2));
        assert!(n1 > 0 && s.check());
        assert_eq!(s.remove(0, 1), None);
        assert!(s.get(0, 498).is_some());
    }

    /// Task P3.37: removal through the twin links (Algorithm T) leaves both
    /// files exactly as the search-based removal of before does — same
    /// entries, same order — through growth, relocation and compaction,
    /// with lines on both sides of `L_link` (exact hints and searches), and
    /// keeps invariant (S3).
    #[test]
    fn ex_5_2_twin_links_match_search_removal() {
        const N: usize = 40;
        // A deterministic pseudo-random sequence (no external crate).
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = |k: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            usize::try_from(seed % (k as u64)).unwrap()
        };
        let caps = vec![1; N];
        let (mut c, mut r) = (
            LineStore::<f64>::new_linked(&caps),
            LineStore::<f64>::new_linked(&caps),
        );
        let (mut c0, mut r0) = (LineStore::<f64>::new(&caps), LineStore::<f64>::new(&caps));
        let (mut removals, mut exact, mut searched) = (0, 0, 0);
        for step in 0..30_000 {
            // Lines fill up to about half of N, then drain and refill.
            let (i, j) = (next(N), next(N));
            let (i32_, j32) = (u32::try_from(i).unwrap(), u32::try_from(j).unwrap());
            match c0.position(j, i32_) {
                None => {
                    let v = f64::from(u32::try_from(step).unwrap());
                    let (q, t) = (c.len(j), r.len(i));
                    c.push_linked(j, i32_, v, t);
                    r.push_linked(i, j32, 0.0, q);
                    c0.push(j, i32_, v);
                    r0.push(i, j32, 0.0);
                }
                Some(q0) => {
                    // Both files through the links: the column entry from
                    // the hint its twin holds, then the twin ...
                    let t0 = r0.position(i, j32).unwrap();
                    let q = c.find(j, i32_, r.link(i, t0)).unwrap();
                    assert_eq!(q, q0);
                    let (_, t) = remove_linked(&mut c, &mut r, j, q);
                    let tf = r.find(i, j32, t).unwrap();
                    if tf == t {
                        exact += 1;
                    } else {
                        searched += 1;
                    }
                    remove_linked(&mut r, &mut c, i, tf);
                    // ... and by search, as the factorization did before.
                    c0.remove(j, i32_);
                    r0.remove(i, j32);
                    removals += 1;
                }
            }
            if step % 97 == 0 {
                assert_eq!(check_links(&c, &r, false), Ok(()));
            }
            for l in 0..N {
                assert_eq!(c.line(l).0, c0.line(l).0, "column {l} at step {step}");
                assert_eq!(r.line(l).0, r0.line(l).0, "row {l} at step {step}");
                let same = c.line(l).1.iter().zip(c0.line(l).1);
                assert!(same.into_iter().all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
        assert!(
            removals > 5000 && exact > 1000 && searched > 100,
            "{removals} {exact} {searched}"
        );
        assert!(c.check() && r.check());
        assert_eq!(check_links(&c, &r, false), Ok(()));
        // One-sided removal before clearing a whole line (a pivot column).
        let twins: Vec<(usize, usize)> = (0..c.len(3))
            .map(|q| {
                let i = c.index_at(3, q) as usize;
                (i, r.find(i, 3, c.link(3, q)).unwrap())
            })
            .collect();
        c.clear(3);
        for (i, t) in twins {
            remove_linked(&mut r, &mut c, i, t);
        }
        assert_eq!(check_links(&c, &r, false), Ok(()));
        // Right after a load every hint is exact.
        let (mut c1, mut r1) = (
            LineStore::<f64>::new_linked(&caps),
            LineStore::<f64>::new_linked(&caps),
        );
        for j in 0..N {
            for i in (j % 3..N).step_by(3) {
                let (q, t) = (c1.len(j), r1.len(i));
                c1.push_linked(j, u32::try_from(i).unwrap(), 1.0, t);
                r1.push_linked(i, u32::try_from(j).unwrap(), 0.0, q);
            }
        }
        assert_eq!(check_links(&c1, &r1, true), Ok(()));
    }

    /// Task P3.37: removals that keep every line long (step T2 of
    /// Algorithm T always repoints the twin of the moved entry) keep every
    /// hint exact.
    #[test]
    fn ex_5_2_twin_links_stay_exact_in_long_lines() {
        const N: usize = 40;
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |k: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            usize::try_from(seed % (k as u64)).unwrap()
        };
        let caps = vec![1; N];
        let (mut c2, mut r2) = (
            LineStore::<f64>::new_linked(&caps),
            LineStore::<f64>::new_linked(&caps),
        );
        for j in 0..N {
            for i in 0..N {
                let (q, t) = (c2.len(j), r2.len(i));
                c2.push_linked(j, u32::try_from(i).unwrap(), 1.0, t);
                r2.push_linked(i, u32::try_from(j).unwrap(), 0.0, q);
            }
        }
        let mut removed = 0;
        for _ in 0..5_000 {
            let (i, j) = (next(N), next(N));
            let j32 = u32::try_from(j).unwrap();
            let Some(q) = c2.position(j, u32::try_from(i).unwrap()) else {
                continue;
            };
            if c2.len(j) <= LINK_MIN + 1 || r2.len(i) <= LINK_MIN + 1 {
                continue;
            }
            let (_, t) = remove_linked(&mut c2, &mut r2, j, q);
            assert_eq!(r2.find(i, j32, t), Some(t), "an exact hint");
            remove_linked(&mut r2, &mut c2, i, t);
            removed += 1;
        }
        assert!(removed > 300, "{removed}");
        assert_eq!(check_links(&c2, &r2, true), Ok(()));
    }

    #[test]
    fn count_lists_move_elements() {
        let mut c = CountLists::new(5, 4);
        for e in 0..5 {
            c.insert(e, e % 3);
        }
        assert!(c.check());
        c.set(0, 4);
        c.remove(3);
        assert!(c.check());
        assert_eq!(c.size(), 4);
        assert_eq!(c.first(4), Some(0));
        let mut b1 = vec![];
        let mut e = c.first(1);
        while let Some(x) = e {
            b1.push(x);
            e = c.next(x);
        }
        assert_eq!(b1, vec![4, 1]);
    }
}
