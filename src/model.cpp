#include "model/model.h"

#include <chrono>

#include <algorithm>
#include <cassert>
#include <cstdlib>

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

namespace model {

namespace detail {
std::string demangle_type_name(const std::type_info& ti) {
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;
    char* demangled = abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);
        std::free(demangled);
        return result;
    }
#endif
    return ti.name();  // fallback: mangled (GCC/Clang demangle failed) or already readable (MSVC)
}
}  // namespace detail

namespace {
/// The one integrity failure with no real Id to classify against the
/// transaction's base -- always CommitStatus::Invalid, never a Conflict.
constexpr const char* kUnmappedLocalMsg =
    "Ref<>/Opt<> points at a local id that was never created, or was removed "
    "within the same transaction before it was ever committed";

}  // namespace

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

const ObjectBase* Snapshot::find_by_key_raw(const void* field, const std::string& key) const {
    if (!root_) return nullptr;
    auto it = root_->by_field.find(field);
    if (it == root_->by_field.end()) return nullptr;
    const Id* id = it->second.get(key);
    return id ? find_raw(*id) : nullptr;
}

struct Snapshot::Lease {
    Model* m;
    std::uint64_t v;

    Lease(Model* mm, std::uint64_t vv) noexcept : m(mm), v(vv) {}

    // A copy would deregister the version early. This bit an earlier version of
    // the code: make_shared<Lease>(Lease{...}) built a temporary whose
    // destructor took the model's mutex while a commit was still holding it.
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    ~Lease() { m->release_version(v); }
};

// ---------------------------------------------------------------------------
// Subscription
// ---------------------------------------------------------------------------

bool Subscription::wait(Update& out) {
    std::unique_lock lk(m_);
    cv_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
}

bool Subscription::try_drain(Update& out) {
    std::lock_guard lk(m_);
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
}

void Subscription::push(Update u) {
    std::lock_guard lk(m_);
    if (q_.size() >= cap_) collapse(std::move(u));
    else q_.push_back(std::move(u));
    cv_.notify_one();
}

void Subscription::collapse(Update tail) {
    // tag never changes across merges: a given Id (generation included) names
    // one object for its whole life, so whichever Change first put it in the
    // map already carries the right tag.
    std::unordered_map<Id, ChangeKind, IdHash> merged;
    std::unordered_map<Id, TypeTag, IdHash> tags;

    auto apply = [&](const Change& c) {
        auto it = merged.find(c.id);
        if (it == merged.end()) {
            merged.emplace(c.id, c.kind);
            tags.emplace(c.id, c.tag);
            return;
        }
        switch (c.kind) {
            case ChangeKind::Updated:
                // Created + Updated stays Created.
                break;
            case ChangeKind::Deleted:
                // Created + Deleted cancels out entirely. This is only correct
                // because the generation is part of the Id, so a recycled slot
                // is a genuinely different Id and cannot be confused with this one.
                if (it->second == ChangeKind::Created) {
                    merged.erase(it);
                    tags.erase(c.id);
                } else {
                    it->second = ChangeKind::Deleted;
                }
                break;
            case ChangeKind::Created:
                it->second = ChangeKind::Created;
                break;
        }
    };

    for (auto& u : q_)
        for (auto& c : u.changes) apply(c);
    for (auto& c : tail.changes) apply(c);

    Update out;
    out.snapshot = std::move(tail.snapshot);
    out.coalesced = true;
    out.changes.reserve(merged.size());
    for (auto& [id, kind] : merged) out.changes.push_back({id, kind, tags.at(id)});

    q_.clear();  // drops the intermediate snapshots -- the whole point
    q_.push_back(std::move(out));
}

void Subscription::close() {
    std::lock_guard lk(m_);
    closed_ = true;
    cv_.notify_all();
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

Model::Model() {
    auto r = std::make_shared<Root>();
    r->version = 1;
    root_.store(r, std::memory_order_release);
    version_ = 1;
    reaper_ = std::thread([this] { reaper_loop(); });
}

Model::~Model() {
    // Stop the reaper and join it.
    {
        std::lock_guard lk(reap_mu_);
        reaper_stop_ = true;
    }
    reap_cv_.notify_all();
    if (reaper_.joinable()) reaper_.join();

    // Free whatever the reaper couldn't (objects still pinned at shutdown, now
    // safe because all readers are gone), plus everything still live in the
    // spine, plus any uncommitted-then-abandoned retirees.
    for (auto& [v, p] : reap_queue_) delete p;
    for (auto& [v, p] : retired_) delete p;
    for (auto& ch : spine_)
        for (std::uint32_t i = 0; i < kChunkSize; ++i) delete ch->obj[i];
}

void Model::reaper_loop() {
    std::unique_lock lk(reap_mu_);
    for (;;) {
        reap_cv_.wait(lk, [&] { return reaper_stop_ || dirty_reap_; });
        if (reaper_stop_ && reap_queue_.empty()) return;
        dirty_reap_ = false;

        std::uint64_t min_live;
        {
            std::lock_guard vl(ver_mu_);
            min_live = live_.empty() ? UINT64_MAX : live_.begin()->first;
        }

        // Free everything no live snapshot can still see; keep the rest in the
        // shared queue so a barrier (wait_for_reclamation) can observe exactly
        // what remains pinned.
        std::size_t freed = 0;
        auto it = std::remove_if(reap_queue_.begin(), reap_queue_.end(), [&](auto& e) {
            if (min_live < e.first) return false;  // still visible somewhere
            delete e.second;
            ++freed;
            return true;
        });
        reap_queue_.erase(it, reap_queue_.end());
        if (freed) retired_pending_.fetch_sub(freed, std::memory_order_relaxed);

        reap_done_round_++;
        reap_done_cv_.notify_all();

        if (reaper_stop_ && reap_queue_.empty()) return;
    }
}

void Model::enqueue_retired(std::vector<std::pair<std::uint64_t, const ObjectBase*>> batch) {
    if (batch.empty()) return;
    retired_pending_.fetch_add(batch.size(), std::memory_order_relaxed);
    {
        std::lock_guard lk(reap_mu_);
        for (auto& e : batch) reap_queue_.push_back(e);
        dirty_reap_ = true;
    }
    reap_cv_.notify_one();
}

std::size_t Model::wait_for_reclamation() {
    // Nudge the reaper and wait for it to complete at least one full round after
    // this point, so anything reclaimable at the current watermark is freed.
    std::unique_lock lk(reap_mu_);
    const std::uint64_t target = reap_done_round_ + 1;
    dirty_reap_ = true;
    reap_cv_.notify_one();
    reap_done_cv_.wait(lk, [&] { return reap_done_round_ >= target; });
    return reap_queue_.size();  // whatever is still pinned by live snapshots
}

Snapshot Model::snapshot() {
    Snapshot s;
    // The root load and the version registration must be atomic with respect to
    // reclamation: otherwise the reaper could free a version in the window
    // between our load and our registration, and we'd hold a snapshot over freed
    // objects. We take ver_mu_ across both.
    //
    // This is a far smaller critical section than a model-wide lock would be:
    // ver_mu_ guards only the version map -- never a commit's apply/traversal
    // work, never a reader's own reads -- so readers and committers never
    // serialize against each other's real work. Never takes commit_mu_, which
    // is what makes begin() (a thin wrapper over this) fully contention-free
    // with respect to try_commit().
    std::lock_guard lk(ver_mu_);
    s.root_ = root_.load(std::memory_order_acquire);
    live_[s.root_->version]++;
    s.lease_ = std::make_shared<Snapshot::Lease>(this, s.root_->version);
    return s;
}

std::shared_ptr<Subscription> Model::subscribe(std::size_t queue_depth) {
    auto s = std::make_shared<Subscription>(queue_depth);
    std::lock_guard lk(subs_mu_);
    subs_.push_back(s);
    return s;
}

void Model::shutdown() {
    std::vector<std::shared_ptr<Subscription>> subs;
    {
        std::lock_guard lk(subs_mu_);
        subs = subs_;
    }
    for (auto& s : subs) s->close();
}

const ObjectBase* Model::peek(Id id) const {
    if (!id) return nullptr;
    const std::uint32_t c = id.index >> kChunkBits;
    const std::uint32_t i = id.index & kChunkMask;
    if (c >= spine_.size()) return nullptr;
    if (spine_[c]->gen[i] != id.gen) return nullptr;
    return spine_[c]->obj[i];
}

Chunk* Model::cow(std::uint32_t c) {
    while (spine_.size() <= c) spine_.push_back(std::make_shared<Chunk>());
    // First touch this try_commit() attempt clones the chunk; subsequent
    // writes hit the clone in place. dirty_ is cleared once the attempt ends.
    if (dirty_.insert(c).second) spine_[c] = std::make_shared<Chunk>(*spine_[c]);
    return const_cast<Chunk*>(spine_[c].get());
}

std::uint32_t Model::alloc_slot() {
    // A recycled slot whose generation is about to wrap to 0 (the null sentinel)
    // must never be reused: a stale handle bearing the old generation would then
    // alias the new object. Such slots are dropped from circulation forever.
    //
    // gen wraps at kGenMax reuses OF ONE SLOT, not across the whole model -- and
    // because free_slots_ is LIFO, reuse concentrates on a few hot slots, so this
    // is far less remote than 2^32 total objects would suggest. Burning one slot
    // per ~4 billion reuses of it costs nothing.
    while (!free_slots_.empty()) {
        const std::uint32_t s = free_slots_.back();
        free_slots_.pop_back();
        log([this, s] { free_slots_.push_back(s); });  // undo the pop
        const std::uint32_t c = s >> kChunkBits, i = s & kChunkMask;
        if (c < spine_.size() && spine_[c]->gen[i] >= kGenMax) {
            ++exhausted_slots_;  // permanently retired; never returned to circulation
            log([this] { --exhausted_slots_; });
            continue;
        }
        return s;
    }
    const std::uint32_t s = next_slot_++;
    log([this] { --next_slot_; });
    // Real slots must never collide with a local (not-yet-committed) id's
    // reserved bit -- see kLocalIdBit. Unreachable at this project's target
    // scale (100k-1M objects); asserting rather than silently misbehaving if
    // it somehow is reached.
    assert(!(s & kLocalIdBit) && "slot allocation reached the reserved local-id bit");
    return s;
}

void Model::set_slot(std::uint32_t slot, const ObjectBase* obj, std::uint32_t gen) {
    const std::uint32_t c = slot >> kChunkBits, i = slot & kChunkMask;
    Chunk* ch = cow(c);
    const ObjectBase* prev_obj = ch->obj[i];
    const std::uint32_t prev_gen = ch->gen[i];
    ch->obj[i] = obj;
    ch->gen[i] = gen;
    log([this, slot, prev_obj, prev_gen] {
        const std::uint32_t cc = slot >> kChunkBits, ii = slot & kChunkMask;
        Chunk* c2 = cow(cc);
        c2->obj[ii] = prev_obj;
        c2->gen[ii] = prev_gen;
    });
}

std::optional<Model::IntegrityError> Model::validate(const ObjectBase* o) const {
    // Referential integrity is enforced *here*, at apply time, against the
    // CURRENT (latest) state -- not the transaction's base. That's what makes
    // "Ref<T> re-validated against latest, not just base" fall out for free,
    // rather than needing a bespoke second check. Returning the violation
    // (rather than asserting) lets try_commit() classify the failure -- did
    // the target exist at the transaction's own base()? -- and turn a
    // concurrent deletion into a Conflict rather than an Invalid rejection.
    // See IntegrityError::bad_target and try_commit()'s error handling.
    std::optional<IntegrityError> err;
    o->each_ref([&](const void* /*field*/, const char* name, Id target, bool nullable) {
        if (err) return;  // keep the FIRST violation; the attempt aborts either way
        if (!target) {
            if (!nullable)
                err = IntegrityError{std::string("null Ref in field ") + name + " of " + o->type(), Id{}};
            return;
        }
        if (!peek(target))
            err = IntegrityError{
                std::string(o->type()) + " field " + name + " references a dead object", target};
    });
    return err;
}

void Model::add_out_refs(const ObjectBase* o) {
    const Id from = o->id;
    o->each_ref([&](const void* field, const char* /*name*/, Id target, bool nullable) {
        if (!target) return;
        referrers_[target.index].push_back(RefEdge{from, field, nullable});
        const std::uint32_t key = target.index;
        log([this, key] {
            auto it = referrers_.find(key);
            if (it != referrers_.end()) {
                it->second.pop_back();  // exact inverse of the push_back above
                if (it->second.empty()) referrers_.erase(it);
            }
        });
    });
}

void Model::drop_out_refs(const ObjectBase* o) {
    const Id from = o->id;
    o->each_ref([&](const void* field, const char* /*name*/, Id target, bool /*nullable*/) {
        if (!target) return;
        auto it = referrers_.find(target.index);
        if (it == referrers_.end()) return;
        auto& v = it->second;
        auto pos = std::find_if(v.begin(), v.end(),
                                [&](const RefEdge& e) { return e.from == from && e.field == field; });
        if (pos == v.end()) return;
        const RefEdge edge = *pos;
        v.erase(pos);
        const std::uint32_t key = target.index;
        log([this, key, edge] { referrers_[key].push_back(edge); });
    });
}

void Model::reconcile_referrer_edges(const ObjectBase* before, const ObjectBase* after) {
    // Per-field diff. `before` and `after` are the same derived type, so
    // each_ref() reports the identical set of fields for both -- only the
    // targets can differ. Fields whose target didn't change are left alone:
    // no redundant erase/insert for a field the caller never touched.
    //
    // Like every other apply-phase mutation, each edit here logs its inverse:
    // a rollback (conflict, veto, or an invalid transaction) must restore the
    // index exactly, or the next cascade resolves against a lie -- an edge
    // erased here but never restored means a later remove() of the old target
    // publishes a dangling Ref.
    const Id id = after->id;
    std::unordered_map<const void*, Id> old_targets;
    before->each_ref([&](const void* field, const char*, Id target, bool) { old_targets[field] = target; });

    after->each_ref([&](const void* field, const char*, Id new_target, bool nullable) {
        const auto it = old_targets.find(field);
        const Id old_target = (it != old_targets.end()) ? it->second : Id{};
        if (new_target == old_target) return;

        if (old_target) {
            auto rit = referrers_.find(old_target.index);
            if (rit != referrers_.end()) {
                auto& v = rit->second;
                auto pos = std::find_if(v.begin(), v.end(), [&](const RefEdge& e) {
                    return e.from == id && e.field == field;
                });
                if (pos != v.end()) {
                    const RefEdge edge = *pos;
                    v.erase(pos);
                    if (v.empty()) referrers_.erase(rit);
                    const std::uint32_t key = old_target.index;
                    log([this, key, edge] { referrers_[key].push_back(edge); });
                }
            }
        }
        if (new_target) {
            referrers_[new_target.index].push_back(RefEdge{id, field, nullable});
            const std::uint32_t key = new_target.index;
            log([this, key] {
                auto rit = referrers_.find(key);
                if (rit != referrers_.end()) {
                    rit->second.pop_back();  // exact inverse of the push_back above
                    if (rit->second.empty()) referrers_.erase(rit);
                }
            });
        }
    });
}

void Model::add_field_keys(const ObjectBase* o) {
    const Id id = o->id;
    o->each_field_key([&](const void* field, std::string key) {
        auto prev = by_field_[field];
        by_field_[field] = prev.set(key, id);
        log([this, field, prev = std::move(prev)]() mutable { by_field_[field] = std::move(prev); });
    });
}

void Model::drop_field_keys(const ObjectBase* o) {
    o->each_field_key([&](const void* field, std::string key) {
        auto prev = by_field_[field];
        by_field_[field] = prev.erase(key);
        log([this, field, prev = std::move(prev)]() mutable { by_field_[field] = std::move(prev); });
    });
}

void Model::reconcile_field_keys(const ObjectBase* before, const ObjectBase* after) {
    const Id id = after->id;
    std::unordered_map<const void*, std::string> old_keys;
    before->each_field_key([&](const void* field, std::string key) { old_keys.emplace(field, std::move(key)); });

    after->each_field_key([&](const void* field, std::string new_key) {
        const auto it = old_keys.find(field);
        if (it != old_keys.end() && it->second == new_key) return;  // unchanged

        // Same undo pattern as add_field_keys/drop_field_keys: capture the
        // whole prior map (cheap -- persistent, structure-shared) and restore
        // it on rollback. Without this, a vetoed/conflicted attempt leaves
        // by_field_ permanently indexing values that never committed.
        auto prev = by_field_[field];
        if (it != old_keys.end()) by_field_[field] = by_field_[field].erase(it->second);
        by_field_[field] = by_field_[field].set(new_key, id);
        log([this, field, prev = std::move(prev)]() mutable { by_field_[field] = std::move(prev); });
    });
}

// The cached-field (multimap) index. Buckets are persistent maps keyed by the
// Id's own bytes -- never flat vectors, which would make each mutation
// O(#duplicates of that value) -- and every mutation captures the prior OUTER
// map for the undo log (a shared_ptr copy, same pattern as add_field_keys):
// a rollback that skipped these would leave the index claiming membership
// that never committed.

void Model::add_cached_fields(const ObjectBase* o) {
    const Id id = o->id;
    o->each_cached_field([&](const void* field, std::string key) {
        auto prev = by_cached_field_[field];
        const pmap::PersistentMap<Id>* bucket = prev.get(key);
        by_cached_field_[field] =
            prev.set(key, (bucket ? *bucket : pmap::PersistentMap<Id>{}).set(detail::id_key(id), id));
        log([this, field, prev = std::move(prev)]() mutable { by_cached_field_[field] = std::move(prev); });
    });
}

void Model::drop_cached_fields(const ObjectBase* o) {
    const Id id = o->id;
    o->each_cached_field([&](const void* field, std::string key) {
        auto prev = by_cached_field_[field];
        const pmap::PersistentMap<Id>* bucket = prev.get(key);
        if (!bucket) return;
        auto nb = bucket->erase(detail::id_key(id));
        // An emptied bucket is dropped outright, so a value with no remaining
        // holders doesn't leave a tombstone entry behind.
        by_cached_field_[field] = nb.empty() ? prev.erase(key) : prev.set(key, nb);
        log([this, field, prev = std::move(prev)]() mutable { by_cached_field_[field] = std::move(prev); });
    });
}

void Model::reconcile_cached_fields(const ObjectBase* before, const ObjectBase* after) {
    const Id id = after->id;
    std::unordered_map<const void*, std::string> old_keys;
    before->each_cached_field([&](const void* field, std::string key) { old_keys.emplace(field, std::move(key)); });

    after->each_cached_field([&](const void* field, std::string new_key) {
        const auto it = old_keys.find(field);
        if (it != old_keys.end() && it->second == new_key) return;  // unchanged

        auto prev = by_cached_field_[field];
        auto cur = prev;
        if (it != old_keys.end()) {
            if (const pmap::PersistentMap<Id>* ob = cur.get(it->second)) {
                auto nb = ob->erase(detail::id_key(id));
                cur = nb.empty() ? cur.erase(it->second) : cur.set(it->second, nb);
            }
        }
        const pmap::PersistentMap<Id>* bucket = cur.get(new_key);
        cur = cur.set(new_key, (bucket ? *bucket : pmap::PersistentMap<Id>{}).set(detail::id_key(id), id));
        by_cached_field_[field] = std::move(cur);
        log([this, field, prev = std::move(prev)]() mutable { by_cached_field_[field] = std::move(prev); });
    });
}

// The cached-reference (reverse multimap) index -- the read-side counterpart
// of referrers_, opt-in per Ref<>/Opt<> field via define_cached_references().
// Same persistent-bucket discipline as the cached-field trio above (never a
// flat vector; every mutation undo-logged), keyed by the TARGET's Id bytes
// instead of a field's value, storing the REFERRER's Id in each bucket.
// each_cached_reference() already filters to just the declared fields, so
// these never do anything for a field not opted in.

void Model::add_cached_references(const ObjectBase* o) {
    const Id id = o->id;
    o->each_cached_reference([&](const void* field, const char*, Id target, bool) {
        if (!target) return;  // Opt<> currently null: nothing to index
        auto prev = by_cached_reference_[field];
        const pmap::PersistentMap<Id>* bucket = prev.get(detail::id_key(target));
        by_cached_reference_[field] = prev.set(
            detail::id_key(target), (bucket ? *bucket : pmap::PersistentMap<Id>{}).set(detail::id_key(id), id));
        log([this, field, prev = std::move(prev)]() mutable { by_cached_reference_[field] = std::move(prev); });
    });
}

void Model::drop_cached_references(const ObjectBase* o) {
    const Id id = o->id;
    o->each_cached_reference([&](const void* field, const char*, Id target, bool) {
        if (!target) return;
        auto prev = by_cached_reference_[field];
        const pmap::PersistentMap<Id>* bucket = prev.get(detail::id_key(target));
        if (!bucket) return;
        auto nb = bucket->erase(detail::id_key(id));
        // An emptied bucket is dropped outright, so a target with no
        // remaining referrers doesn't leave a tombstone entry behind.
        by_cached_reference_[field] =
            nb.empty() ? prev.erase(detail::id_key(target)) : prev.set(detail::id_key(target), nb);
        log([this, field, prev = std::move(prev)]() mutable { by_cached_reference_[field] = std::move(prev); });
    });
}

void Model::reconcile_cached_references(const ObjectBase* before, const ObjectBase* after) {
    const Id id = after->id;
    std::unordered_map<const void*, Id> old_targets;
    before->each_cached_reference(
        [&](const void* field, const char*, Id target, bool) { old_targets[field] = target; });

    after->each_cached_reference([&](const void* field, const char*, Id new_target, bool) {
        const auto it = old_targets.find(field);
        const Id old_target = (it != old_targets.end()) ? it->second : Id{};
        if (new_target == old_target) return;  // unchanged (incl. both still null)

        auto prev = by_cached_reference_[field];
        auto cur = prev;
        if (old_target) {
            if (const pmap::PersistentMap<Id>* ob = cur.get(detail::id_key(old_target))) {
                auto nb = ob->erase(detail::id_key(id));
                cur = nb.empty() ? cur.erase(detail::id_key(old_target)) : cur.set(detail::id_key(old_target), nb);
            }
        }
        if (new_target) {
            const pmap::PersistentMap<Id>* bucket = cur.get(detail::id_key(new_target));
            cur = cur.set(detail::id_key(new_target),
                          (bucket ? *bucket : pmap::PersistentMap<Id>{}).set(detail::id_key(id), id));
        }
        by_cached_reference_[field] = std::move(cur);
        log([this, field, prev = std::move(prev)]() mutable { by_cached_reference_[field] = std::move(prev); });
    });
}

void Model::retire(const ObjectBase* o) {
    // Invisible from version_+1 onward, but snapshots at or below version_ can
    // still see it. Free only once none of those remain.
    retired_.emplace_back(version_ + 1, o);
}

void Model::release_version(std::uint64_t v) {
    {
        std::lock_guard lk(ver_mu_);
        auto it = live_.find(v);
        if (--it->second == 0) live_.erase(it);
    }
    // A dropped snapshot (or a dropped/committed Transaction's base) may have
    // lowered the watermark, unblocking retired objects. Mark a reap round due
    // and wake the reaper.
    {
        std::lock_guard lk(reap_mu_);
        dirty_reap_ = true;
    }
    reap_cv_.notify_one();
}

void Model::debug_set_generation(std::uint32_t slot, std::uint32_t gen) {
    std::lock_guard lk(commit_mu_);  // spine_/dirty_ are commit_mu_-protected now
    Chunk* ch = cow(slot >> kChunkBits);
    ch->gen[slot & kChunkMask] = gen;
    dirty_.clear();  // keep the poke out of the next attempt's dirty set
}

std::size_t Model::debug_changelog_size() const {
    std::lock_guard lk(commit_mu_);
    return changelog_.size();
}

std::vector<std::pair<std::uint64_t, int>> Model::debug_live_versions() const {
    std::lock_guard lk(ver_mu_);
    return {live_.begin(), live_.end()};
}

// ---------------------------------------------------------------------------
// try_commit() internals
// ---------------------------------------------------------------------------

std::optional<Model::IntegrityError> Model::apply_create(std::unique_ptr<ObjectBase> o,
                                                         std::unordered_map<std::uint32_t, Id>& remap) {
    const std::uint32_t local_index = o->id.index;  // still local; the remap key

    // May reference an earlier local create in this txn. Nothing is logged
    // yet, so an unmapped local id can just return; `o` frees itself.
    bool unmapped = false;
    o->remap_refs(RefRemapper{remap, &unmapped});
    if (unmapped) return IntegrityError{kUnmappedLocalMsg, Id{}};

    const std::uint32_t slot = alloc_slot();
    const std::uint32_t i = slot & kChunkMask;
    const std::uint32_t cur_gen = (slot >> kChunkBits) < spine_.size()
                                      ? spine_[slot >> kChunkBits]->gen[i]
                                      : 0;
    const std::uint32_t g = cur_gen + 1;  // recycled slot gets a fresh generation

    o->id = Id{slot, g};
    // On a violation, the slot allocation above is already logged, so
    // rollback_apply() reclaims it; the object is still owned by `o` and is
    // freed by unique_ptr on the early return.
    if (auto err = validate(o.get())) return err;

    const Id id = o->id;
    ObjectBase* raw = o.release();

    set_slot(slot, raw, g);  // logs the inverse (restores prev obj + gen)

    // Every object gets an (internal, Id-keyed) entry in its type's
    // enumeration index, unconditionally -- this is what for_each<T>() scans.
    const TypeTag tag = raw->tag();
    auto prev_sub = by_type_[tag];
    by_type_[tag] = prev_sub.set(detail::id_key(id), id);
    log([this, tag, prev_sub = std::move(prev_sub)]() mutable { by_type_[tag] = std::move(prev_sub); });

    add_out_refs(raw);
    add_field_keys(raw);
    add_cached_fields(raw);
    add_cached_references(raw);

    changes_.push_back({id, ChangeKind::Created, tag});
    log([this] { changes_.pop_back(); });

    // Never published, nothing else owns it: rollback_apply() deletes
    // everything in txn_created_. Deliberately NOT logged as an undo op -- the
    // rollback delete loop consumes this list directly, and a pop-undo would
    // empty it first.
    txn_created_.push_back(raw);

    remap[local_index] = id;
    return std::nullopt;
}

std::optional<Model::IntegrityError> Model::apply_update(std::unique_ptr<ObjectBase> clone,
                                                         std::unordered_map<std::uint32_t, Id>& remap) {
    ObjectBase* raw = clone.release();

    // Log the clone's deletion FIRST (before anything can fail), so in
    // reverse replay it runs LAST -- after set_slot's undo has repointed the
    // slot back at the baseline. Otherwise we'd free `raw` while the slot
    // still referenced it, and an early error return below would leak it.
    log([raw] { delete raw; });

    bool unmapped = false;
    raw->remap_refs(RefRemapper{remap, &unmapped});  // may reference a local create in this same txn
    if (unmapped) return IntegrityError{kUnmappedLocalMsg, Id{}};

    // Against CURRENT peek(), not the transaction's base -- see validate()'s comment.
    if (auto err = validate(raw)) return err;

    const Id id = raw->id;
    const ObjectBase* baseline = peek(id);
    // try_commit()'s conflict check (check_id_overlap) already guarantees
    // `baseline` is non-null and unchanged since txn.base(): if any other
    // commit had touched this id since then, this attempt would have been
    // rejected as a Conflict before ever reaching apply.

    retire(baseline);
    log([this] { retired_.pop_back(); });

    set_slot(id.index, raw, id.gen);  // logs restore of baseline + its generation

    changes_.push_back({id, ChangeKind::Updated, raw->tag()});
    log([this] { changes_.pop_back(); });

    // Unlike the single-writer design this project's sibling uses, `raw` is
    // already fully written by the time we get here (the caller finished
    // writing to it back when building the Transaction) -- so reconciliation
    // runs immediately, not in a later deferred pass. This MUST happen before
    // remove_raw()'s cascade BFS runs (see try_commit()'s phase order): a
    // same-transaction "repoint away from X, then delete X" needs referrers_
    // already updated, or the repointed-away-from object would incorrectly
    // be dragged into X's cascade.
    reconcile_referrer_edges(baseline, raw);
    reconcile_field_keys(baseline, raw);
    reconcile_cached_fields(baseline, raw);
    reconcile_cached_references(baseline, raw);
    return std::nullopt;
}

ObjectBase* Model::clone_for_cascade_null(Id id) {
    // remove_raw()'s cascade BFS calls this for every NULLABLE referrer of a
    // victim: clone, install, and hand back a mutable pointer so the caller
    // can null_ref() the one field that pointed at the victim. Structurally
    // identical to apply_update's install steps, minus remap/validate (a
    // cascade null can't introduce a dangling ref -- it can only clear one)
    // and minus reconciliation, which the caller does AFTER null_ref(), not
    // here: reconciling before the field is actually nulled would compare
    // two identical objects and be a silent no-op, then nothing would ever
    // fix up referrers_ once the field really does change. (This exact
    // ordering mistake -- reconcile before the caller's write lands -- is
    // why the single-writer sibling project defers reconciliation to a
    // separate pass in the first place; here, deferring to "right after
    // null_ref(), same call site" is simpler and just as correct.)
    const ObjectBase* cur = peek(id);
    if (!cur) return nullptr;

    ObjectBase* copy = cur->clone();
    log([copy] { delete copy; });

    retire(cur);
    log([this] { retired_.pop_back(); });

    set_slot(id.index, copy, id.gen);

    changes_.push_back({id, ChangeKind::Updated, copy->tag()});
    log([this] { changes_.pop_back(); });

    return copy;
}

std::vector<Id> Model::remove_raw(Id id) {
    std::vector<Id> killed;
    std::vector<Id> work{id};
    std::unordered_set<std::uint32_t> visited;

    while (!work.empty()) {
        const Id x = work.back();
        work.pop_back();
        if (!peek(x)) continue;
        if (!visited.insert(x.index).second) continue;  // cycles terminate here

        // Copy the referrer list: we are about to mutate it.
        auto it = referrers_.find(x.index);
        const std::vector<RefEdge> edges =
            (it == referrers_.end()) ? std::vector<RefEdge>{} : it->second;

        for (const RefEdge& e : edges) {
            if (!peek(e.from)) continue;
            if (e.nullable) {
                const ObjectBase* baseline = peek(e.from);
                if (ObjectBase* m = clone_for_cascade_null(e.from)) {
                    m->null_ref(e.field);
                    reconcile_referrer_edges(baseline, m);
                    reconcile_field_keys(baseline, m);
                    reconcile_cached_fields(baseline, m);
                    reconcile_cached_references(baseline, m);
                }
            } else {
                work.push_back(e.from);  // dies with its target
            }
        }

        const ObjectBase* victim = peek(x);
        if (!victim) continue;

        drop_out_refs(victim);  // logs re-add of victim's outgoing edges

        // victim.index should have no remaining incoming edges (its referrers
        // were cascaded or nulled above), but if any survive, preserve them for
        // undo. In practice this is empty; capture it to be exact.
        auto rit = referrers_.find(x.index);
        if (rit != referrers_.end()) {
            std::vector<RefEdge> saved = std::move(rit->second);
            referrers_.erase(rit);
            const std::uint32_t key = x.index;
            log([this, key, saved = std::move(saved)]() mutable {
                referrers_[key] = std::move(saved);
            });
        }

        const TypeTag tag = victim->tag();
        auto prev_sub = by_type_[tag];
        by_type_[tag] = prev_sub.erase(detail::id_key(x));
        log([this, tag, prev_sub = std::move(prev_sub)]() mutable { by_type_[tag] = std::move(prev_sub); });

        drop_field_keys(victim);
        drop_cached_fields(victim);
        drop_cached_references(victim);

        set_slot(x.index, nullptr, x.gen);  // logs restore of victim + its gen

        retire(victim);
        log([this] { retired_.pop_back(); });

        free_slots_.push_back(x.index);
        log([this] { free_slots_.pop_back(); });

        changes_.push_back({x, ChangeKind::Deleted, tag});
        log([this] { changes_.pop_back(); });

        killed.push_back(x);
    }
    return killed;
}

std::vector<Id> Model::check_id_overlap(const Transaction& txn) const {
    std::unordered_set<std::uint32_t> written_slots;
    for (const auto& [slot, clone] : txn.local_updated_) {
        (void)clone;
        written_slots.insert(slot);
    }
    for (Id rid : txn.remove_intents_) written_slots.insert(rid.index);

    std::vector<Id> conflicts;
    if (written_slots.empty()) return conflicts;

    for (const auto& entry : changelog_) {
        if (entry.version <= txn.base_version()) continue;
        for (const Change& c : entry.changes) {
            if (written_slots.count(c.id.index)) conflicts.push_back(c.id);
        }
    }
    return conflicts;
}

void Model::prune_changelog() {
    // Safe to discard any entry no open Transaction could still need to
    // compare against: a Transaction's base() is a real, pinned Snapshot, so
    // it already can't let the watermark advance past its own base version.
    // Piggybacking on live_ this way needs no separate retention bookkeeping.
    std::uint64_t watermark;
    {
        std::lock_guard lk(ver_mu_);
        watermark = live_.empty() ? version_ : live_.begin()->first;
    }
    while (!changelog_.empty() && changelog_.front().version <= watermark) changelog_.pop_front();
}

void Model::rollback_apply() {
    // Replay inverses in reverse. Each closure exactly undoes one primitive
    // mutation, so commit-lock-protected state returns to its last committed
    // shape.
    for (auto it = undo_.rbegin(); it != undo_.rend(); ++it) (*it)();
    undo_.clear();

    // Objects created this attempt were never published and are owned by
    // nothing after their slot-writes are undone. Free them.
    for (const ObjectBase* o : txn_created_) delete o;
    txn_created_.clear();

    changes_.clear();  // any survivors were popped by the log; clear defensively
    dirty_.clear();
}

// ---------------------------------------------------------------------------
// begin() / try_commit()
// ---------------------------------------------------------------------------

Transaction Model::begin() { return begin(snapshot()); }

Transaction Model::begin(Snapshot base) { return Transaction(this, std::move(base)); }

Transaction Snapshot::begin() const {
    assert(lease_ && "begin() on a default-constructed Snapshot -- no Model to build against");
    return lease_->m->begin(*this);
}

CommitResult Model::try_commit(Transaction& txn) {
    assert(txn.model_ == this && "Transaction belongs to a different Model");

    if (txn.local_created_.empty() && txn.local_updated_.empty() && txn.remove_intents_.empty())
        return CommitResult{CommitStatus::Committed, txn.base_, {}, std::nullopt, {}, std::nullopt};

    std::lock_guard commit_lk(commit_mu_);

    if (std::vector<Id> overlap = check_id_overlap(txn); !overlap.empty()) {
        return CommitResult{CommitStatus::Conflict, Snapshot{}, {},
                            ConflictInfo{ConflictReason::IdSetOverlap, std::move(overlap)}, {},
                            std::nullopt};
    }

    std::unordered_map<std::uint32_t, Id> remap;  // local Id::index -> real Id, this attempt only

    // Creates first, then updates (reconciled immediately, see apply_update),
    // then deletes resolved last -- in that order, so a same-transaction
    // "repoint away from X, then delete X" sees the repoint already reflected
    // in referrers_ before the cascade BFS runs. The first integrity
    // violation aborts the attempt; everything applied so far is unwound.
    std::optional<IntegrityError> err;
    for (auto& obj : txn.local_created_) {
        if (obj && (err = apply_create(std::move(obj), remap))) break;  // null: cancelled locally
    }
    if (!err) {
        for (auto& [slot, clone] : txn.local_updated_) {
            (void)slot;
            if (clone && (err = apply_update(std::move(clone), remap))) break;
        }
    }
    if (!err) {
        for (Id rid : txn.remove_intents_) remove_raw(rid);
    }

    if (err) {
        rollback_apply();
        // Did the dangling target exist at this transaction's own base()? If
        // so, someone else deleted it concurrently -- a Conflict, not a bug.
        // (bad_target is null for a null non-nullable Ref or an unmapped
        // local id, which have no target to check and are always genuine
        // transaction-building bugs -- those reject as Invalid.)
        if (err->bad_target && txn.base().find_raw(err->bad_target)) {
            return CommitResult{CommitStatus::Conflict, Snapshot{}, {},
                                ConflictInfo{ConflictReason::RefIntegrity, {err->bad_target}}, {},
                                std::nullopt};
        }
        return CommitResult{CommitStatus::Invalid, Snapshot{}, {}, std::nullopt, {}, std::move(err)};
    }

    if (changes_.empty()) {
        // Everything in txn had already been applied by an earlier
        // try_commit() on this same Transaction (or every local create was
        // locally cancelled) -- a no-op success, not a fresh publish.
        return CommitResult{CommitStatus::Committed, snapshot(), {}, std::nullopt, {}, std::nullopt};
    }

    // Runs after apply, not before: it sees the FULLY resolved changeset,
    // including cascade deletes (resolved just above). A false return unwinds
    // everything applied so far, exactly like an integrity violation does.
    // (The hook cannot report failure by throwing -- the project builds with
    // -fno-exceptions, so a throw is std::terminate, not an error path.)
    if (pre_commit_ && !pre_commit_(*this, changes_)) {
        rollback_apply();
        return CommitResult{CommitStatus::Vetoed, Snapshot{}, {}, std::nullopt, {}, std::nullopt};
    }

    ++version_;

    auto r = std::make_shared<Root>();
    r->version = version_;
    r->spine = spine_;        // ~n/kChunkSize shared_ptr copies. Cheap.
    r->by_type = by_type_;    // O(#types): each per-type submap is shared, not copied.
    r->by_field = by_field_;  // O(#indexed fields): same reasoning.
    r->by_cached_field = by_cached_field_;  // O(#cached fields): ditto.
    r->by_cached_reference = by_cached_reference_;  // O(#cached ref fields): ditto.

    Snapshot pub;
    {
        // Publish the new root and register the event payload's version together
        // under ver_mu_, so a concurrent snapshot()/begin() sees a consistent
        // (root, live_) pair and the reaper never frees this version early.
        std::lock_guard lk(ver_mu_);
        root_.store(r, std::memory_order_release);
        pub.root_ = r;
        live_[r->version]++;
        pub.lease_ = std::make_shared<Snapshot::Lease>(this, r->version);
    }

    {
        std::vector<std::shared_ptr<Subscription>> subs;
        {
            std::lock_guard lk(subs_mu_);
            subs = subs_;
        }
        for (auto& s : subs) s->push(Update{pub, changes_, false});
    }

    // Hand this attempt's retired objects to the background reaper rather
    // than freeing them inline: a large cascade must not stall a commit, and
    // destructors should run off both the committing thread and any reader.
    enqueue_retired(std::move(retired_));
    retired_.clear();

    changelog_.push_back({r->version, changes_});
    prune_changelog();

    std::vector<Change> resolved = std::move(changes_);

    changes_.clear();
    dirty_.clear();        // next attempt re-COWs each chunk it touches
    undo_.clear();         // committed: nothing to roll back to
    txn_created_.clear();  // published objects are now owned by the spine

    return CommitResult{CommitStatus::Committed, pub, std::move(resolved), std::nullopt, std::move(remap),
                        std::nullopt};
}

}  // namespace model
