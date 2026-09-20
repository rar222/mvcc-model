#pragma once
//
// A flat, mutable hash set: open addressing, linear probing, backward-shift
// deletion. One contiguous array of keys, no per-key allocation.
//
// This is the counterpart of persistent_map.h for state no reader ever sees.
// The HAMT there path-copies so that published versions share structure; that
// sharing is what a flat table cannot offer, so a FlatSet must never be part
// of published state (CLAUDE.md invariant 3). It is for writer-private
// structures mutated in place under commit_mu_, such as the bucket of a target
// with many referrers in Model::referrers_.
//
// Not thread-safe: callers serialize access themselves.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace model::fset {

/// A set of K, with Hash a stateless functor returning a 64-bit value. The
/// value-initialized K{} marks an empty cell, so it can never be a member:
/// insert() asserts on it, contains() and erase() report it absent. model::Id
/// fits, since Id{} is the null id. K must be default-constructible, copyable
/// and comparable with operator==.
///
/// Terms. The table is an array of cells, each holding one key or empty. A
/// key's home cell is where its hash points. A lookup starts at the home cell
/// and steps forward one cell at a time, wrapping at the end, until it finds
/// the key or reaches an empty cell; those steps are the probe run. This is
/// linear probing. The table is a power of two in size and never more than
/// 3/4 full, so every probe run ends at an empty cell.
///
/// The hash is scrambled internally before it selects a home cell, so an
/// identity-style Hash (model::IdHash) is fine.
///
/// insert(), erase(), reserve() and shrink_to_fit() invalidate any
/// for_each() in progress.
template <class K, class Hash>
class FlatSet {
public:
    FlatSet() = default;
    FlatSet(const FlatSet&) = default;
    FlatSet& operator=(const FlatSet&) = default;

    FlatSet(FlatSet&& o) noexcept : cells_(std::move(o.cells_)), size_(std::exchange(o.size_, 0)) {
        o.cells_.clear();
    }
    FlatSet& operator=(FlatSet&& o) noexcept {
        // Takes the other set's table, leaving it empty.
        if (this != &o) {
            cells_ = std::move(o.cells_);
            size_ = std::exchange(o.size_, 0);
            o.cells_.clear();
        }
        return *this;
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    /// Number of cells in the table, empty or not; 0 before the first insert,
    /// otherwise a power of two of at least kMinCells.
    std::size_t cell_count() const noexcept { return cells_.size(); }

    /// True iff `key` was not already present.
    bool insert(const K& key) {
        assert(!(key == K{}) && "K{} marks an empty cell and cannot be stored");
        // Looks for the key, and for the empty cell that would hold it, in the existing table.
        if (!cells_.empty()) {
            const std::size_t mask = cells_.size() - 1;
            std::size_t i = home_of(key, mask);
            // Walks the probe run to its first empty cell.
            while (!(cells_[i] == K{})) {
                if (cells_[i] == key) return false;
                i = (i + 1) & mask;
            }
            // Stores in place when the table stays at most 3/4 full.
            if ((size_ + 1) * 4 <= cells_.size() * 3) {
                cells_[i] = key;
                ++size_;
                return true;
            }
        }
        rehash(cells_.empty() ? kMinCells : cells_.size() * 2);
        place(key);
        ++size_;
        return true;
    }

    /// True iff `key` was present.
    bool erase(const K& key) {
        if (cells_.empty() || key == K{}) return false;
        const std::size_t mask = cells_.size() - 1;
        std::size_t i = home_of(key, mask);
        // Walks the probe run to the key, giving up at the first empty cell.
        while (!(cells_[i] == key)) {
            if (cells_[i] == K{}) return false;
            i = (i + 1) & mask;
        }
        std::size_t j = i;
        // Backward shift: refills the hole from later keys in the run that may legally move into it.
        for (;;) {
            j = (j + 1) & mask;
            if (cells_[j] == K{}) break;
            const std::size_t h = home_of(cells_[j], mask);
            if (i <= j ? (i < h && h <= j) : (i < h || h <= j)) continue;
            cells_[i] = cells_[j];
            i = j;
        }
        cells_[i] = K{};
        --size_;
        return true;
    }

    bool contains(const K& key) const {
        if (cells_.empty() || key == K{}) return false;
        const std::size_t mask = cells_.size() - 1;
        // Walks the probe run until the key or an empty cell.
        for (std::size_t i = home_of(key, mask);; i = (i + 1) & mask) {
            if (cells_[i] == key) return true;
            if (cells_[i] == K{}) return false;
        }
    }

    /// Ensures room for `n` keys in total without another rehash. Never shrinks.
    void reserve(std::size_t n) {
        const std::size_t want = cells_for(n);
        if (want > cells_.size()) rehash(want);
    }

    /// Reallocates to the smallest table that holds the current keys; frees
    /// the table entirely when empty. erase() alone never gives space back.
    void shrink_to_fit() {
        // An empty set releases its table entirely.
        if (size_ == 0) {
            std::vector<K>().swap(cells_);
            return;
        }
        const std::size_t want = cells_for(size_);
        if (want < cells_.size()) rehash(want);
    }

    /// Visits every key once, in table order (not insertion or sorted order).
    template <class F>
    void for_each(F&& f) const {
        // Visits each occupied cell.
        for (const K& k : cells_)
            if (!(k == K{})) f(k);
    }

    /// `f(key)` returns bool (true = keep going). Returns false iff `f` stopped the walk.
    template <class F>
    bool for_each_short_circuit(F&& f) const {
        // Visits each occupied cell until `f` says stop.
        for (const K& k : cells_)
            if (!(k == K{}) && !f(k)) return false;
        return true;
    }

private:
    static constexpr std::size_t kMinCells = 8;

    static std::size_t cells_for(std::size_t n) noexcept {
        std::size_t c = kMinCells;
        while (n * 4 > c * 3) c *= 2;
        return c;
    }

    // MurmurHash3's 64-bit finalizer.
    static std::uint64_t scramble(std::uint64_t x) noexcept {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ull;
        x ^= x >> 33;
        return x;
    }

    static std::size_t home_of(const K& key, std::size_t mask) {
        const std::uint64_t h = scramble(Hash{}(key));
        return h & mask;
    }

    // Stores a key known to be absent, into a table known to have room.
    void place(const K& key) {
        const std::size_t mask = cells_.size() - 1;
        std::size_t i = home_of(key, mask);
        while (!(cells_[i] == K{})) i = (i + 1) & mask;
        cells_[i] = key;
    }

    void rehash(std::size_t new_cell_count) {
        std::vector<K> old = std::move(cells_);
        cells_.assign(new_cell_count, K{});
        // Re-places every key, since each home cell depends on the table size.
        for (const K& k : old)
            if (!(k == K{})) place(k);
    }

    std::vector<K> cells_;
    std::size_t size_ = 0;
};

}  // namespace model::fset
