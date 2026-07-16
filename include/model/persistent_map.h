#pragma once
//
// A minimal persistent hash map (HAMT: hash array mapped trie).
//
// Why this exists: the secondary index (external key -> Id) must be part of every
// published Root, and Roots are immutable and cheap to derive from one another. A
// std::unordered_map forces an O(n) copy per commit -- the exact size-proportional
// cost the chunked object store is designed to avoid. A HAMT shares structure
// between versions, so producing a new version path-copies only O(log32 n) nodes.
//
// This is a from-scratch, dependency-free implementation (no network to pull
// immer). It supports exactly what the model needs: set, erase, get, and cheap
// persistent derivation. It is NOT a general-purpose container.
//
// Layout: a 32-ary trie keyed on 5-bit slices of the key's hash. Each internal
// node holds a 32-bit bitmap and a densely packed array of children; a child is
// either another node or a leaf bucket. Collisions (full 64-bit hash equal, or
// deeper than the hash provides) live together in a leaf bucket and are compared
// by the actual key.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace model::pmap {

template <class V>
class PersistentMap {
    struct Leaf {
        std::uint64_t hash;
        std::vector<std::pair<std::string, V>> kvs;  // usually 1; >1 only on collision
    };

    struct Node {
        std::uint32_t bitmap = 0;
        // Each present bit has a child: either a Node or a Leaf.
        std::vector<std::shared_ptr<const Node>> children;
        std::vector<std::shared_ptr<const Leaf>> leaves;
        // children[k]/leaves[k] correspond positionally; exactly one is non-null
        // per occupied slot. Kept as two parallel vectors for simplicity.
    };

    static std::uint64_t hash_key(const std::string& k) noexcept {
        // FNV-1a, 64-bit. Adequate for dispersing string keys across the trie.
        std::uint64_t h = 1469598103934665603ull;
        for (unsigned char c : k) {
            h ^= c;
            h *= 1099511628211ull;
        }
        return h;
    }

    static std::uint32_t slice(std::uint64_t hash, int shift) noexcept {
        return static_cast<std::uint32_t>((hash >> shift) & 0x1f);  // 5 bits
    }
    static std::uint32_t bit(std::uint32_t idx) noexcept { return 1u << idx; }
    static std::uint32_t popcount_below(std::uint32_t bitmap, std::uint32_t idx) noexcept {
        return static_cast<std::uint32_t>(__builtin_popcount(bitmap & (bit(idx) - 1)));
    }

    std::shared_ptr<const Node> root_;
    std::size_t size_ = 0;

    explicit PersistentMap(std::shared_ptr<const Node> r, std::size_t n)
        : root_(std::move(r)), size_(n) {}

    // Return a new node equal to `n` but with slot `pos` replaced/inserted.
    static std::shared_ptr<Node> clone_node(const Node* n) {
        auto c = std::make_shared<Node>();
        if (n) {
            c->bitmap = n->bitmap;
            c->children = n->children;
            c->leaves = n->leaves;
        }
        return c;
    }

    static const V* leaf_get(const Leaf* lf, const std::string& key) {
        for (auto& kv : lf->kvs)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }

    // ---- set ----
    // Returns the new subtree root for the node at `shift`, and reports whether
    // the key was newly inserted (vs replaced) via `added`.
    static std::shared_ptr<const Node> set_in(const Node* n, std::uint64_t hash,
                                              int shift, const std::string& key,
                                              const V& val, bool& added) {
        const std::uint32_t idx = slice(hash, shift);
        const std::uint32_t b = bit(idx);

        if (!n || !(n->bitmap & b)) {
            // Empty slot: insert a fresh single-entry leaf.
            auto nn = clone_node(n);
            const std::uint32_t pos = popcount_below(nn->bitmap, idx);
            auto lf = std::make_shared<Leaf>();
            lf->hash = hash;
            lf->kvs.push_back({key, val});
            nn->bitmap |= b;
            nn->children.insert(nn->children.begin() + pos, nullptr);
            nn->leaves.insert(nn->leaves.begin() + pos, lf);
            added = true;
            return nn;
        }

        const std::uint32_t pos = popcount_below(n->bitmap, idx);
        auto nn = clone_node(n);

        if (n->children[pos]) {
            // Slot holds a subtree: recurse.
            nn->children[pos] = set_in(n->children[pos].get(), hash, shift + 5, key, val, added);
            return nn;
        }

        // Slot holds a leaf.
        const Leaf* lf = n->leaves[pos].get();
        if (lf->hash == hash) {
            // Same hash: replace-or-append within the bucket (true collision or
            // same key).
            auto nl = std::make_shared<Leaf>(*lf);
            bool replaced = false;
            for (auto& kv : nl->kvs)
                if (kv.first == key) {
                    kv.second = val;
                    replaced = true;
                    break;
                }
            if (!replaced) {
                nl->kvs.push_back({key, val});
                added = true;
            }
            nn->leaves[pos] = nl;
            return nn;
        }

        // Different hash sharing this slot: push the existing leaf down into a
        // new subtree, then insert the new key beside it.
        if (shift + 5 >= 64) {
            // Ran out of hash bits (astronomically unlikely with distinct hashes,
            // but handle it): merge into one collision bucket.
            auto nl = std::make_shared<Leaf>(*lf);
            nl->kvs.push_back({key, val});
            nn->leaves[pos] = nl;
            added = true;
            return nn;
        }
        auto sub = std::make_shared<Node>();
        const std::uint32_t exist_idx = slice(lf->hash, shift + 5);
        sub->bitmap = bit(exist_idx);
        sub->children.push_back(nullptr);
        sub->leaves.push_back(n->leaves[pos]);
        auto sub2 = set_in(sub.get(), hash, shift + 5, key, val, added);
        nn->children[pos] = sub2;
        nn->leaves[pos] = nullptr;
        return nn;
    }

    // ---- erase ----
    static std::shared_ptr<const Node> erase_in(const Node* n, std::uint64_t hash,
                                                int shift, const std::string& key,
                                                bool& removed) {
        if (!n) return nullptr;
        const std::uint32_t idx = slice(hash, shift);
        const std::uint32_t b = bit(idx);
        if (!(n->bitmap & b)) return copy_shared(n);  // not present

        const std::uint32_t pos = popcount_below(n->bitmap, idx);
        auto nn = clone_node(n);

        if (n->children[pos]) {
            auto sub = erase_in(n->children[pos].get(), hash, shift + 5, key, removed);
            if (sub && sub->bitmap != 0) {
                nn->children[pos] = sub;
                return nn;
            }
            // Subtree emptied: drop this slot.
            nn->bitmap &= ~b;
            nn->children.erase(nn->children.begin() + pos);
            nn->leaves.erase(nn->leaves.begin() + pos);
            return nn;
        }

        // Leaf slot.
        const Leaf* lf = n->leaves[pos].get();
        const V* hit = leaf_get(lf, key);
        if (!hit) return copy_shared(n);  // key absent

        removed = true;
        if (lf->kvs.size() > 1) {
            auto nl = std::make_shared<Leaf>(*lf);
            for (std::size_t i = 0; i < nl->kvs.size(); ++i)
                if (nl->kvs[i].first == key) {
                    nl->kvs.erase(nl->kvs.begin() + i);
                    break;
                }
            nn->leaves[pos] = nl;
            return nn;
        }
        // Single-entry leaf: remove the slot entirely.
        nn->bitmap &= ~b;
        nn->children.erase(nn->children.begin() + pos);
        nn->leaves.erase(nn->leaves.begin() + pos);
        return nn;
    }

    static std::shared_ptr<const Node> copy_shared(const Node* n) {
        // The node is unchanged; reuse it. We only have a raw pointer here, so we
        // must reconstruct a shared_ptr owner -- but every caller already holds
        // one, so instead of copying we clone. (Called only on the not-present
        // paths, which are rare; correctness over cleverness.)
        return clone_node(n);
    }

    static const V* get_in(const Node* n, std::uint64_t hash, int shift,
                           const std::string& key) {
        while (n) {
            const std::uint32_t idx = slice(hash, shift);
            const std::uint32_t b = bit(idx);
            if (!(n->bitmap & b)) return nullptr;
            const std::uint32_t pos = popcount_below(n->bitmap, idx);
            if (n->children[pos]) {
                n = n->children[pos].get();
                shift += 5;
                continue;
            }
            return leaf_get(n->leaves[pos].get(), key);
        }
        return nullptr;
    }

    template <class F>
    static void each_in(const Node* n, F& f) {
        if (!n) return;
        for (std::size_t i = 0; i < n->children.size(); ++i) {
            if (n->children[i])
                each_in(n->children[i].get(), f);
            else if (n->leaves[i])
                for (auto& kv : n->leaves[i]->kvs) f(kv.first, kv.second);
        }
    }

public:
    PersistentMap() = default;

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    /// Returns a new map with key=val. O(log32 n) node allocations; shares the
    /// rest of the structure with `*this`.
    PersistentMap set(const std::string& key, const V& val) const {
        bool added = false;
        auto r = set_in(root_.get(), hash_key(key), 0, key, val, added);
        return PersistentMap(r, size_ + (added ? 1 : 0));
    }

    /// Returns a new map without `key` (or an equal map if absent).
    PersistentMap erase(const std::string& key) const {
        bool removed = false;
        auto r = erase_in(root_.get(), hash_key(key), 0, key, removed);
        return PersistentMap(r, size_ - (removed ? 1 : 0));
    }

    const V* get(const std::string& key) const {
        return get_in(root_.get(), hash_key(key), 0, key);
    }

    template <class F>
    void for_each(F&& f) const {
        each_in(root_.get(), f);
    }
};

}  // namespace model::pmap
