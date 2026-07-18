#include "model/model.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

namespace model {

namespace detail {
// Turns typeid(Derived).name() (e.g. "6Widget" or "N4demo7AccountE") into the
// human-readable "Widget"/"demo::Account" that Object<Derived>::type() hands
// back to callers -- ObjectBase::type() is part of the read-side API (error
// messages, logging), so a mangled name there would be a usability bug, not
// just cosmetic. __cxa_demangle is a GCC/Clang runtime extension: it
// heap-allocates its result with malloc (hence std::free, not delete), and
// signals failure via `status` rather than an exception (this project builds
// -fno-exceptions, so it couldn't throw here even if it wanted to).
std::string demangle_type_name(const std::type_info& ti) {
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;  // 0 == success; nonzero == demangle failed (e.g. not a mangled name)
    char* demangled = abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);  // copy into a std::string we own...
        std::free(demangled);           // ...then free the C-allocated buffer immediately
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
    if (!root_) return nullptr;  // default-constructed Snapshot: nothing to look up
    // by_field maps field-tag (the `field` pointer, see field_tag<>) -> that
    // field's own persistent map of key -> Id. Two lookups, not one flat map,
    // because each field owns an independent keyspace (see field_tag's doc).
    auto it = root_->by_field.find(field);
    if (it == root_->by_field.end()) return nullptr;  // this field was never define_keys()'d
    const Id* id = it->second.get(key);               // no match for this exact key value
    // find_raw() re-checks the generation, so even if the id this key mapped
    // to at commit time has since been recycled (in a LATER snapshot -- this
    // one is immutable), this snapshot still resolves it correctly or not at
    // all; it never aliases the new occupant.
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
    std::unique_lock lk(m_);  // unique_lock, not lock_guard: cv_.wait() must be able to unlock
                              // while blocked and re-lock before returning
    // Woken by either push() (work arrived) or close() (shutdown) -- the
    // predicate re-checks both conditions itself, so a spurious wakeup (or a
    // wakeup that raced another consumer thread and lost) just loops back to
    // sleep instead of returning garbage.
    cv_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;  // woke because closed_, not because work arrived: no more events, ever
    out = std::move(q_.front());   // move out, not copy: an Update carries a Snapshot (pins a version)
                                   // and a full Change vector -- no reason to duplicate either
    q_.pop_front();
    return true;
}

bool Subscription::try_drain(Update& out) {
    std::lock_guard lk(m_);
    if (q_.empty()) return false;  // non-blocking: nothing queued right now, caller decides what to do
    out = std::move(q_.front());
    q_.pop_front();
    return true;
}

void Subscription::push(Update u) {
    std::lock_guard lk(m_);
    // cap_ is the queue depth passed to Subscription's constructor: once the
    // queue is already at capacity, a slow consumer must not be allowed to
    // make it grow further (see CLAUDE.md's "why not let a slow subscriber's
    // queue grow" -- an unbounded queue pins every version behind it and every
    // object retired since). Coalesce into one Update instead of enqueuing a
    // new one.
    if (q_.size() >= cap_)
        collapse(std::move(u));
    else
        q_.push_back(std::move(u));
    cv_.notify_one();  // wakes at most one blocked wait() -- there is always at most one consumer
                       // per queue entry to hand off, so notify_one (not notify_all) is correct
}

void Subscription::collapse(Update tail) {
    // The overflow path for push(): instead of enqueuing `tail` as a new,
    // distinct entry (which would exceed cap_), replay every Change from
    // every Update ALREADY queued plus tail's own Changes through `apply`,
    // producing one Change per Id that reflects only its NET effect across
    // the whole merged window (e.g. Created then Updated collapses to just
    // Created; Created then Deleted cancels out entirely -- see the switch
    // below). The result replaces the entire queue with a single coalesced
    // Update carrying tail's snapshot (the newest one) but a synthesized
    // changeset.
    //
    // Two parallel maps, not one, because Change bundles a kind AND a tag,
    // but the merge logic below only ever needs to branch on `kind` -- tag
    // never changes across merges (a given Id, generation included, names
    // one object for its whole life, so whichever Change first put it in the
    // map already carries the right tag) and is just carried through
    // untouched at the end.
    std::unordered_map<Id, ChangeKind, IdHash> merged;  // Id -> its net ChangeKind so far
    std::unordered_map<Id, TypeTag, IdHash> tags;       // Id -> its (unchanging) TypeTag

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

    // Oldest first: q_'s existing entries (already in arrival order), then
    // tail (the newest, about to overflow the queue) -- so `merged`'s final
    // state reflects the changes in the order they actually happened.
    for (auto& u : q_)
        for (auto& c : u.changes) apply(c);
    for (auto& c : tail.changes) apply(c);

    Update out;
    out.snapshot = std::move(tail.snapshot);  // the newest version -- every older Snapshot in the
                                              // queue is dropped along with q_ below
    out.coalesced = true;                     // tells the consumer this Update skipped intermediate versions
    out.changes.reserve(merged.size());
    for (auto& [id, kind] : merged) out.changes.push_back({id, kind, tags.at(id)});

    q_.clear();  // drops the intermediate snapshots -- the whole point (each one was pinning a
                // version, and everything retired since, alive)
    q_.push_back(std::move(out));
}

void Subscription::close() {
    std::lock_guard lk(m_);
    closed_ = true;      // wait()'s predicate re-checks this: any blocked or future wait() call
                         // returns false instead of hanging forever
    cv_.notify_all();  // notify_all, not notify_one: every blocked consumer thread (there may be
                       // several) must wake up and observe closed_, not just one of them
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

Model::Model() {
    // Version 0 is never used -- the very first published state is version 1,
    // empty (no objects). Building it here rather than lazily on the first
    // commit means root_ is never null: every reader that calls snapshot()
    // before any writer has committed anything still gets a valid, empty
    // Root, not a special-cased nullptr.
    auto r = std::make_shared<Root>();
    r->version = 1;
    root_.store(r, std::memory_order_release);
    version_ = 1;  // the writer's own copy of "latest version" (commit_mu_-protected); try_commit()
                  // increments this, then builds the next Root from it
    reaper_ = std::thread([this] { reaper_loop(); });
}

Model::~Model() {
    // Stop the reaper and join it FIRST, before touching anything it might
    // also be touching (reap_queue_): joining guarantees reaper_loop() has
    // fully returned, so there is no concurrent access to race against below.
    {
        std::lock_guard lk(reap_mu_);
        reaper_stop_ = true;
    }
    reap_cv_.notify_all();
    if (reaper_.joinable()) reaper_.join();

    // Free whatever the reaper couldn't (objects still pinned at shutdown, now
    // safe because all readers are gone), plus everything still live in the
    // spine, plus any uncommitted-then-abandoned retirees.
    for (auto& [v, p] : reap_queue_) delete p;  // v (the retirement version) is irrelevant now --
                                                // no live snapshot can exist past ~Model()
    for (auto& [v, p] : retired_) delete p;     // an in-flight try_commit() attempt's retirees,
                                                // never published (only reached via a leaked
                                                // Transaction -- defensive, not the normal path)
    for (auto& ch : spine_)                     // every object still live in the last-published Root
        for (std::uint32_t i = 0; i < kChunkSize; ++i) delete ch->obj[i];  // null slots: delete(nullptr) is a no-op
}

void Model::reaper_loop() {
    std::unique_lock lk(reap_mu_);  // unique_lock: reap_cv_.wait/reap_done_cv_.wait need to unlock
                                    // while blocked and re-lock on wakeup
    for (;;) {
        // Woken by enqueue_retired() (new work), release_version() (the
        // watermark may have moved, unblocking existing work), or the
        // destructor (reaper_stop_). dirty_reap_ is the "there's a reason to
        // run a pass" flag both producers set; re-check it (not just
        // reaper_stop_) so a spurious wakeup just goes back to sleep.
        reap_cv_.wait(lk, [&] { return reaper_stop_ || dirty_reap_; });
        if (reaper_stop_ && reap_queue_.empty()) return;  // nothing left to drain: exit now
        dirty_reap_ = false;  // this pass is about to consume the reason it was set

        // min_live: the oldest version any live Snapshot (or open
        // Transaction's base()) still needs -- the reclamation watermark.
        // Anything retired at or before this version is invisible to every
        // live reader and safe to free. Taken under ver_mu_, released
        // immediately: this is the reaper's own reap_mu_ -> ver_mu_ ordering
        // (the one exception to the usual commit_mu_ -> ver_mu_ -> reap_mu_
        // chain -- see CLAUDE.md invariant 10).
        std::uint64_t min_live;
        {
            std::lock_guard vl(ver_mu_);
            min_live = live_.empty() ? UINT64_MAX : live_.begin()->first;
        }

        // Free everything no live snapshot can still see; keep the rest in the
        // shared queue so a barrier (wait_for_reclamation) can observe exactly
        // what remains pinned. remove_if partitions reap_queue_ in place
        // (freeable entries deleted and moved to the tail); `it` marks where
        // the surviving (still-pinned) entries end, so erase() below drops
        // exactly the tail that remove_if already deleted through.
        std::size_t freed = 0;
        auto it = std::remove_if(reap_queue_.begin(), reap_queue_.end(), [&](auto& e) {
            if (min_live < e.first) return false;  // still visible somewhere
            delete e.second;
            ++freed;
            return true;
        });
        reap_queue_.erase(it, reap_queue_.end());
        if (freed) retired_pending_.fetch_sub(freed, std::memory_order_relaxed);

        // Completing a pass -- even one that freed nothing -- advances the
        // round counter and wakes wait_for_reclamation(), whose contract is
        // "at least one full pass ran after I asked," not "something got freed."
        reap_done_round_++;
        reap_done_cv_.notify_all();

        if (reaper_stop_ && reap_queue_.empty()) return;  // re-check: this pass may have been the
                                                          // last one needed to drain everything
    }
}

void Model::enqueue_retired(std::vector<std::pair<std::uint64_t, const ObjectBase*>> batch) {
    if (batch.empty()) return;  // nothing to hand off (e.g. a commit that touched no existing object)
    // Counted BEFORE being made visible in reap_queue_ (under reap_mu_
    // below), so retired_pending() -- which adds this atomic to
    // reap_queue_.size() -- never transiently UNDERcounts a batch that's
    // already enqueued but not yet reflected here; the reverse order could.
    retired_pending_.fetch_add(batch.size(), std::memory_order_relaxed);
    {
        std::lock_guard lk(reap_mu_);
        for (auto& e : batch) reap_queue_.push_back(e);
        dirty_reap_ = true;
    }
    reap_cv_.notify_one();  // one reaper thread, so notify_one suffices
}

std::size_t Model::wait_for_reclamation() {
    // Nudge the reaper and wait for it to complete at least one full round after
    // this point, so anything reclaimable at the current watermark is freed.
    std::unique_lock lk(reap_mu_);
    // target: the round counter value that proves a FRESH pass (started after
    // this call, not one already in flight) has finished -- reap_done_round_
    // is incremented once per completed pass, so "current value + 1" is the
    // next pass to complete from here on.
    const std::uint64_t target = reap_done_round_ + 1;
    dirty_reap_ = true;  // force a pass even if nothing new was enqueued since the last one
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
    // Constructed before the lock is taken (Subscription's own constructor
    // touches no Model state), so subs_mu_ is held only for the push_back --
    // a commit publishing an Update takes the same lock just long enough to
    // copy subs_ out (see publish_now()), never while actually pushing to a
    // subscriber's queue.
    auto s = std::make_shared<Subscription>(queue_depth);
    std::lock_guard lk(subs_mu_);
    subs_.push_back(s);  // shared_ptr, not a raw reference: `s` must outlive this call (the caller
                         // holds the other reference) and outlive a subsequent shutdown() too
    return s;
}

void Model::shutdown() {
    // Copy the subscriber list out and release subs_mu_ BEFORE calling
    // close() on each one: close() takes that Subscription's own m_ and
    // notifies its condition variable, which can run arbitrarily long if a
    // consumer thread is slow to react. Holding subs_mu_ across that would
    // block subscribe() (and every commit's publish, which briefly takes
    // subs_mu_ to copy the list) for no reason.
    std::vector<std::shared_ptr<Subscription>> subs;
    {
        std::lock_guard lk(subs_mu_);
        subs = subs_;
    }
    for (auto& s : subs) s->close();
}

const ObjectBase* Model::peek(Id id) const {
    if (!id) return nullptr;  // Id{} (default-constructed): never a valid handle, by construction
    // Slot index splits into (chunk index, offset within chunk) via a shift
    // and a mask, rather than division/modulo, because kChunkSize is a power
    // of two -- see Chunk's own doc comment for why chunking exists at all
    // (bounding a single COW clone's cost).
    const std::uint32_t c = id.index >> kChunkBits;
    const std::uint32_t i = id.index & kChunkMask;
    if (c >= spine_.size()) return nullptr;             // never allocated: chunk doesn't exist yet
    if (spine_[c]->gen[i] != id.gen) return nullptr;    // slot was recycled since this Id was
                                                        // minted -- id.gen names a specific
                                                        // occupant, not just a slot (invariant 5)
    return spine_[c]->obj[i];  // may itself be nullptr if the slot is currently empty (removed,
                              // generation bumped, no create yet) -- caller must still check
}

Chunk* Model::cow(std::uint32_t c) {
    // Grow the spine lazily, one fresh (empty) Chunk at a time, up to and
    // including index c -- this is the only place new Chunks are appended,
    // so alloc_slot() handing out a slot in a chunk that doesn't exist yet
    // is exactly what triggers growth here on the following write.
    while (spine_.size() <= c) spine_.push_back(std::make_shared<Chunk>());
    // First touch this try_commit() attempt clones the chunk; subsequent
    // writes hit the clone in place. dirty_ is cleared once the attempt ends.
    // dirty_.insert(c).second is true only the FIRST time c is touched this
    // attempt (unordered_set::insert reports whether it actually inserted) --
    // that's what makes this a once-per-attempt clone rather than a clone
    // per write, while still leaving the OLD shared_ptr<Chunk> (and every
    // object it points at) untouched for any Snapshot still reading it.
    if (dirty_.insert(c).second) spine_[c] = std::make_shared<Chunk>(*spine_[c]);
    // const_cast: spine_ holds shared_ptr<const Chunk> because published
    // chunks are immutable (invariant 3), but the chunk we just cloned above
    // is this attempt's own private working copy -- not yet published, not
    // yet shared with any reader -- so mutating it through this pointer is
    // safe precisely because cow() is the only path that clones it.
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
    Chunk* ch = cow(c);  // clones the chunk on first touch this attempt; see cow()'s own comment
    // Captured before being overwritten, purely so the undo closure below can
    // restore EXACTLY what was there -- not "the current baseline" (which
    // could itself have changed by the time rollback runs, if this slot were
    // touched more than once in one attempt), but the specific prior value
    // this one write is undoing.
    const ObjectBase* prev_obj = ch->obj[i];
    const std::uint32_t prev_gen = ch->gen[i];
    ch->obj[i] = obj;
    ch->gen[i] = gen;
    // The undo closure re-derives (cc, ii, c2) rather than capturing `ch`/`i`
    // directly: replay can happen after further COW churn this same attempt,
    // so `ch` (a raw Chunk* into a specific shared_ptr<Chunk> generation)
    // could be dangling by then -- re-running cow(cc) gets whatever chunk
    // object is currently live for that index instead.
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
                err = IntegrityError{std::string("null Ref in field ") + name + " of " + o->type(),
                                     Id{}};
            return;
        }
        if (!peek(target))
            err = IntegrityError{
                std::string(o->type()) + " field " + name + " references a dead object", target};
    });
    return err;
}

// referrers_ maintenance: the writer-private reverse index that drives
// cascade delete (invariant 8). Keyed by TARGET slot index (bare, not a full
// Id -- only the live generation of a slot can ever be the target of a
// live Ref, so the generation would be redundant), each bucket is a
// vector<RefEdge> naming every field, on every object, currently pointing at
// that slot. add_out_refs/drop_out_refs add or remove an object's WHOLE
// outgoing edge set at once (create/delete); reconcile_referrer_edges below
// diffs an update's before/after instead, touching only the fields whose
// target actually changed.

void Model::add_out_refs(const ObjectBase* o) {
    const Id from = o->id;  // the object doing the pointing -- becomes RefEdge::from below
    o->each_ref([&](const void* field, const char* /*name*/, Id target, bool nullable) {
        if (!target) return;  // a null Opt<>: nothing to index
        // field is the field-tag pointer (see field_tag<>), captured in the
        // edge so drop_out_refs/reconcile can later find and remove exactly
        // this (from, field) pair without disturbing some other field on the
        // same object that happens to point at the same target.
        referrers_[target.index].push_back(RefEdge{from, field, nullable});
        const std::uint32_t key = target.index;  // copied out for the undo closure below, since
                                                 // `target` itself is a loop-local captured by
                                                 // value into each_ref's lambda, not by the log()
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
        if (it == referrers_.end()) return;  // nothing indexed for this target (shouldn't happen if
                                             // add_out_refs was called for every create, but a
                                             // missing bucket is harmless to tolerate here)
        auto& v = it->second;
        // Find the specific edge this (from, field) pair added -- v may hold
        // edges from many OTHER objects/fields pointing at the same target.
        auto pos = std::find_if(v.begin(), v.end(), [&](const RefEdge& e) {
            return e.from == from && e.field == field;
        });
        if (pos == v.end()) return;
        const RefEdge edge = *pos;  // saved for the undo closure -- `pos` itself won't survive the erase
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
    before->each_ref(
        [&](const void* field, const char*, Id target, bool) { old_targets[field] = target; });

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

// by_field_ maintenance: the define_keys() unique-key index (Root::by_field).
// One persistent map PER FIELD (by_field_[field]), each mapping that field's
// key string -> the single Id currently holding it -- unique, unlike the
// cached-field/cached-reference multimaps below. Every mutation captures the
// prior per-field map as a whole (`prev`; cheap, since PersistentMap sharing
// means this is a handle copy, not a deep copy) so the undo log can restore
// it verbatim on rollback: a missed restore would leave by_field_ answering
// find_by_key() with a value that never actually committed.

void Model::add_field_keys(const ObjectBase* o) {
    const Id id = o->id;
    o->each_field_key([&](const void* field, std::string key) {
        auto prev = by_field_[field];         // this field's map before the insert
        by_field_[field] = prev.set(key, id);  // key is unique per field by construction (see
                                               // define_keys' contract); a collision here is a
                                               // caller bug, not something this layer detects
        log([this, field, prev = std::move(prev)]() mutable {
            by_field_[field] = std::move(prev);
        });
    });
}

void Model::drop_field_keys(const ObjectBase* o) {
    o->each_field_key([&](const void* field, std::string key) {
        auto prev = by_field_[field];
        by_field_[field] = prev.erase(key);
        log([this, field, prev = std::move(prev)]() mutable {
            by_field_[field] = std::move(prev);
        });
    });
}

void Model::reconcile_field_keys(const ObjectBase* before, const ObjectBase* after) {
    const Id id = after->id;
    // old_keys: this object's OWN key per field, as it was before the
    // update -- one snapshot of each_field_key() taken up front, so the
    // `after` pass below can diff against it field by field without a
    // second traversal of `before`.
    std::unordered_map<const void*, std::string> old_keys;
    before->each_field_key(
        [&](const void* field, std::string key) { old_keys.emplace(field, std::move(key)); });

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
        log([this, field, prev = std::move(prev)]() mutable {
            by_field_[field] = std::move(prev);
        });
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
        // Two-level structure: by_cached_field_[field] is the OUTER map (key
        // string -> bucket); `bucket`, if this key already has other
        // holders, is the INNER map (Id-bytes -> Id) collecting every object
        // currently holding that value. `prev` is the outer map's state
        // before this insert, captured whole for the undo log below.
        auto prev = by_cached_field_[field];
        const pmap::PersistentMap<Id>* bucket = prev.get(key);
        by_cached_field_[field] = prev.set(
            key, (bucket ? *bucket : pmap::PersistentMap<Id>{}).set(detail::id_key(id), id));
        log([this, field, prev = std::move(prev)]() mutable {
            by_cached_field_[field] = std::move(prev);
        });
    });
}

void Model::drop_cached_fields(const ObjectBase* o) {
    const Id id = o->id;
    o->each_cached_field([&](const void* field, std::string key) {
        auto prev = by_cached_field_[field];
        const pmap::PersistentMap<Id>* bucket = prev.get(key);
        if (!bucket) return;  // nothing indexed under this key -- nothing to remove
        auto nb = bucket->erase(detail::id_key(id));  // nb: the bucket with just this id removed
        // An emptied bucket is dropped outright, so a value with no remaining
        // holders doesn't leave a tombstone entry behind.
        by_cached_field_[field] = nb.empty() ? prev.erase(key) : prev.set(key, nb);
        log([this, field, prev = std::move(prev)]() mutable {
            by_cached_field_[field] = std::move(prev);
        });
    });
}

void Model::reconcile_cached_fields(const ObjectBase* before, const ObjectBase* after) {
    const Id id = after->id;
    // old_keys: this object's own cached-field value per field, as it was
    // BEFORE the update -- snapshotted up front so the `after` pass can diff
    // against it one field at a time.
    std::unordered_map<const void*, std::string> old_keys;
    before->each_cached_field(
        [&](const void* field, std::string key) { old_keys.emplace(field, std::move(key)); });

    after->each_cached_field([&](const void* field, std::string new_key) {
        const auto it = old_keys.find(field);
        if (it != old_keys.end() && it->second == new_key) return;  // unchanged

        // prev: the OUTER map's state before any edit -- what the undo log
        // restores wholesale on rollback. cur: the outer map as it's built
        // up across the two steps below (old value's bucket shrinks/drops,
        // new value's bucket grows), then installed once at the end.
        auto prev = by_cached_field_[field];
        auto cur = prev;
        if (it != old_keys.end()) {
            // Remove this id from its OLD value's bucket (ob), unless that
            // value was never actually indexed (e.g. this field just started
            // returning a cacheable value).
            if (const pmap::PersistentMap<Id>* ob = cur.get(it->second)) {
                auto nb = ob->erase(detail::id_key(id));  // ob with this id removed
                cur = nb.empty() ? cur.erase(it->second) : cur.set(it->second, nb);
            }
        }
        // Add this id to its NEW value's bucket, creating that bucket if this
        // is the first object ever to hold this particular value.
        const pmap::PersistentMap<Id>* bucket = cur.get(new_key);
        cur = cur.set(new_key,
                      (bucket ? *bucket : pmap::PersistentMap<Id>{}).set(detail::id_key(id), id));
        by_cached_field_[field] = std::move(cur);
        log([this, field, prev = std::move(prev)]() mutable {
            by_cached_field_[field] = std::move(prev);
        });
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
    const Id id = o->id;  // the REFERRER -- the value stored in the bucket, not the bucket's key
    o->each_cached_reference([&](const void* field, const char*, Id target, bool) {
        if (!target) return;  // Opt<> currently null: nothing to index
        auto prev = by_cached_reference_[field];
        const pmap::PersistentMap<Id>* bucket = prev.get(detail::id_key(target));
        by_cached_reference_[field] =
            prev.set(detail::id_key(target),
                     (bucket ? *bucket : pmap::PersistentMap<Id>{}).set(detail::id_key(id), id));
        log([this, field, prev = std::move(prev)]() mutable {
            by_cached_reference_[field] = std::move(prev);
        });
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
        log([this, field, prev = std::move(prev)]() mutable {
            by_cached_reference_[field] = std::move(prev);
        });
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
                cur = nb.empty() ? cur.erase(detail::id_key(old_target))
                                 : cur.set(detail::id_key(old_target), nb);
            }
        }
        if (new_target) {
            const pmap::PersistentMap<Id>* bucket = cur.get(detail::id_key(new_target));
            cur =
                cur.set(detail::id_key(new_target),
                        (bucket ? *bucket : pmap::PersistentMap<Id>{}).set(detail::id_key(id), id));
        }
        by_cached_reference_[field] = std::move(cur);
        log([this, field, prev = std::move(prev)]() mutable {
            by_cached_reference_[field] = std::move(prev);
        });
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
        // live_[v] is a refcount: multiple Snapshots (or Transaction bases)
        // can share the same version, so only the LAST one dropping it
        // actually removes the entry and can move the watermark.
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

std::optional<Model::IntegrityError> Model::apply_create(
    std::unique_ptr<ObjectBase> o, std::unordered_map<std::uint32_t, Id>& remap) {
    const std::uint32_t local_index = o->id.index;  // still local; the remap key

    // May reference an earlier local create in this txn. Nothing is logged
    // yet, so an unmapped local id can just return; `o` frees itself.
    bool unmapped = false;
    o->remap_refs(RefRemapper{remap, &unmapped});
    if (unmapped) return IntegrityError{kUnmappedLocalMsg, Id{}};

    const std::uint32_t slot = alloc_slot();
    const std::uint32_t i = slot & kChunkMask;
    const std::uint32_t cur_gen =
        (slot >> kChunkBits) < spine_.size() ? spine_[slot >> kChunkBits]->gen[i] : 0;
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
    log([this, tag, prev_sub = std::move(prev_sub)]() mutable {
        by_type_[tag] = std::move(prev_sub);
    });

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

std::optional<Model::IntegrityError> Model::apply_update(
    std::unique_ptr<ObjectBase> clone, std::unordered_map<std::uint32_t, Id>& remap) {
    ObjectBase* raw = clone.release();

    // Log the clone's deletion FIRST (before anything can fail), so in
    // reverse replay it runs LAST -- after set_slot's undo has repointed the
    // slot back at the baseline. Otherwise we'd free `raw` while the slot
    // still referenced it, and an early error return below would leak it.
    log([raw] { delete raw; });

    bool unmapped = false;
    raw->remap_refs(
        RefRemapper{remap, &unmapped});  // may reference a local create in this same txn
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
    // Breadth-first cascade delete, resolved here (at apply time, under
    // commit_mu_) and never eagerly (invariant 8). `work` is the BFS
    // frontier -- ids still waiting to be visited, seeded with the one
    // Transaction::remove() intent this call is resolving; `visited` is the
    // set of slot indices already processed, which both terminates cycles
    // (a self- or mutually-referential graph would otherwise loop forever)
    // and prevents processing the same victim twice; `killed` accumulates
    // every id actually deleted, in visitation order, for the caller
    // (try_commit()) to report.
    std::vector<Id> killed;
    std::vector<Id> work{id};
    std::unordered_set<std::uint32_t> visited;

    while (!work.empty()) {
        const Id x = work.back();  // `x`: the id currently being resolved this iteration
        work.pop_back();
        if (!peek(x)) continue;  // already gone (e.g. cascaded in from another branch of the BFS)
        if (!visited.insert(x.index).second) continue;  // cycles terminate here

        // Copy the referrer list: we are about to mutate it (both directly,
        // via the erase below, and indirectly, via reconcile_referrer_edges
        // inside the nullable branch) while iterating what it pointed to.
        auto it = referrers_.find(x.index);
        const std::vector<RefEdge> edges =
            (it == referrers_.end()) ? std::vector<RefEdge>{} : it->second;

        // Every edge currently pointing AT x: for a NULLABLE field, clear
        // just that field (the referrer survives); for a non-nullable one,
        // the referrer cannot exist without x, so it joins the BFS frontier
        // and will itself be visited (and cascade further) in a later
        // iteration of this same loop.
        for (const RefEdge& e : edges) {
            if (!peek(e.from)) continue;  // referrer itself already deleted this same pass
            if (e.nullable) {
                // `baseline` is captured BEFORE clone_for_cascade_null() installs
                // the clone, so reconcile_referrer_edges (etc.) below can diff
                // "before this field was nulled" against "after" -- see
                // clone_for_cascade_null's own comment for why reconciliation
                // must happen here, after null_ref(), not inside that helper.
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
        if (!victim) continue;  // defensive, matching the peek()-then-check style used for every
                                // other id in this BFS (x itself, and each e.from above) --
                                // nothing in the edges loop above installs a null at slot x itself

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
        log([this, tag, prev_sub = std::move(prev_sub)]() mutable {
            by_type_[tag] = std::move(prev_sub);
        });

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
    // written_slots: every EXISTING object this transaction wants to touch --
    // slot indices, not full Ids, because a conflict is about the SLOT (did
    // anyone else touch it since?), not about matching generations; slot is
    // the map key of txn.local_updated_ (an update targets one specific
    // already-committed object) and rid.index for a remove intent. Creates
    // are deliberately absent: they allocate a brand-new slot only once
    // apply_create() actually runs, so there is nothing published yet for
    // another commit to have collided with.
    std::unordered_set<std::uint32_t> written_slots;
    for (const auto& [slot, clone] : txn.local_updated_) {
        (void)clone;  // only the map's key (the slot) matters here, not the pending clone itself
        written_slots.insert(slot);
    }
    for (Id rid : txn.remove_intents_) written_slots.insert(rid.index);

    std::vector<Id> conflicts;
    if (written_slots.empty()) return conflicts;  // nothing to conflict-check (a create-only txn)

    // Scan every changelog entry published strictly after this transaction's
    // OWN base version -- i.e. everything it could not have seen when it was
    // built. Any Change in there whose slot this transaction also wants to
    // touch is a genuine race: someone else committed against the same
    // object first. `conflicts` collects the SPECIFIC ids that collided, for
    // ConflictInfo::ids -- not just a yes/no.
    for (const auto& entry : changelog_) {
        if (entry.version <= txn.base_version()) continue;  // txn's base already reflects this
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

Transaction Model::begin() {
    return begin(snapshot());
}

Transaction Model::begin(Snapshot base) {
    return Transaction(this, std::move(base));
}

Transaction Snapshot::begin() const {
    assert(lease_ && "begin() on a default-constructed Snapshot -- no Model to build against");
    return lease_->m->begin(*this);
}

std::optional<Model::IntegrityError> Model::apply_transaction_contents(
    Transaction& txn, std::unordered_map<std::uint32_t, Id>& remap) {
    // Creates first, then updates (reconciled immediately, see apply_update),
    // then deletes resolved last -- in that order, so a same-transaction
    // "repoint away from X, then delete X" sees the repoint already reflected
    // in referrers_ before the cascade BFS runs. The first integrity
    // violation aborts the attempt; the caller must then rollback (via
    // classify_apply_failure()).
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
    return err;
}

CommitResult Model::classify_apply_failure(IntegrityError err, const Transaction& txn) {
    rollback_apply();
    // Did the dangling target exist at this transaction's own base()? If
    // so, someone else deleted it concurrently -- a Conflict, not a bug.
    // (bad_target is null for a null non-nullable Ref or an unmapped
    // local id, which have no target to check and are always genuine
    // transaction-building bugs -- those reject as Invalid.)
    if (err.bad_target && txn.base().find_raw(err.bad_target)) {
        return CommitResult{CommitStatus::Conflict,
                            Snapshot{},
                            {},
                            ConflictInfo{ConflictReason::RefIntegrity, {err.bad_target}},
                            {},
                            std::nullopt};
    }
    return CommitResult{CommitStatus::Invalid, Snapshot{}, {}, std::nullopt, {}, std::move(err)};
}

std::optional<CommitResult> Model::check_and_apply(Transaction& txn,
                                                   std::unordered_map<std::uint32_t, Id>& remap) {
    if (std::vector<Id> overlap = check_id_overlap(txn); !overlap.empty()) {
        return CommitResult{CommitStatus::Conflict,
                            Snapshot{},
                            {},
                            ConflictInfo{ConflictReason::IdSetOverlap, std::move(overlap)},
                            {},
                            std::nullopt};
    }
    if (auto err = apply_transaction_contents(txn, remap)) {
        return classify_apply_failure(std::move(*err), txn);
    }
    return std::nullopt;  // applied; remap is populated, changes_ may or may not be empty
}

CommitResult Model::publish_now(std::unordered_map<std::uint32_t, Id> remap) {
    ++version_;  // the version this attempt is about to publish -- commit_mu_-protected, so no
                // other thread can be racing this increment

    // r: the new, immutable Root that becomes the published state. Every
    // field is a cheap handle copy (shared_ptr vector / persistent-map
    // handle), not a deep copy -- see the per-field comments below -- so
    // building this costs O(#chunks touched + #indexes), never O(model size).
    auto r = std::make_shared<Root>();
    r->version = version_;
    r->spine = spine_;        // ~n/kChunkSize shared_ptr copies. Cheap.
    r->by_type = by_type_;    // O(#types): each per-type submap is shared, not copied.
    r->by_field = by_field_;  // O(#indexed fields): same reasoning.
    r->by_cached_field = by_cached_field_;          // O(#cached fields): ditto.
    r->by_cached_reference = by_cached_reference_;  // O(#cached ref fields): ditto.

    // pub: this new version's own Snapshot, pinned (via its Lease) in the
    // same critical section that publishes root_ and registers the version
    // in live_ -- so by the time any other thread can observe r as "latest,"
    // this version is already un-freeable, and pub is ready to hand to every
    // subscriber below and to return to try_commit()'s caller.
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
        // subs: a copy of the subscriber list, taken and immediately
        // released from subs_mu_ before any push() -- a slow subscriber's
        // push (which may block on ITS OWN queue-full path, see
        // Subscription::push/collapse) must never hold up subscribe() or
        // another commit's own publish, which only need subs_mu_ briefly.
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

    return CommitResult{CommitStatus::Committed, pub,         std::move(resolved), std::nullopt,
                        std::move(remap),        std::nullopt};
}

CommitResult Model::commit_locked(Transaction& txn) {
    assert(txn.model_ == this && "Transaction belongs to a different Model");

    if (txn.local_created_.empty() && txn.local_updated_.empty() && txn.remove_intents_.empty())
        return CommitResult{CommitStatus::Committed, txn.base_, {}, std::nullopt, {}, std::nullopt};

    std::unordered_map<std::uint32_t, Id> remap;  // local Id::index -> real Id, this attempt only
    if (auto failure = check_and_apply(txn, remap)) return std::move(*failure);

    if (changes_.empty()) {
        // Everything in txn had already been applied by an earlier
        // try_commit() on this same Transaction (or every local create was
        // locally cancelled) -- a no-op success, not a fresh publish.
        return CommitResult{
            CommitStatus::Committed, snapshot(), {}, std::nullopt, {}, std::nullopt};
    }

    return publish_now(std::move(remap));
}

CommitResult Model::run_pre_commit_transaction(Transaction& txn) {
    // Crash, don't return a bad CommitResult: this function does no locking of its own -- it
    // goes straight into commit_locked(), which requires commit_mu_ ALREADY
    // held. Called from anywhere outside the one window try_commit()
    // guarantees the lock is held, it would mutate commit_mu_-protected
    // state (spine_, referrers_, ...) with no synchronization at all --
    // corruption, not a recoverable error. See the doc comment in model.h.
    assert(in_pre_transactions_phase_ &&
           "run_pre_commit_transaction() called outside a running PreTransactionsFn callback");
    assert(!precommit_failed_ &&
           "run_pre_commit_transaction() called again after an earlier pre-transaction this same "
           "attempt already failed -- check the return value and stop");
    assert(txn.model_ == this && "Transaction belongs to a different Model");

    CommitResult result = commit_locked(txn);
    if (result.status != CommitStatus::Committed) {
        precommit_failed_ = true;
        precommit_failure_ = std::make_unique<CommitResult>(result);
    }
    return result;
}

CommitResult Model::try_commit(Transaction& txn) {
    assert(txn.model_ == this && "Transaction belongs to a different Model");

    if (txn.local_created_.empty() && txn.local_updated_.empty() && txn.remove_intents_.empty())
        return CommitResult{CommitStatus::Committed, txn.base_, {}, std::nullopt, {}, std::nullopt};

    std::lock_guard commit_lk(commit_mu_);

    if (pre_transactions_) {
        assert(!in_pre_transactions_phase_ && "pre_transactions_ invoked reentrantly");
        precommit_failed_ = false;
        precommit_failure_.reset();
        in_pre_transactions_phase_ = true;
        pre_transactions_(*this);
        in_pre_transactions_phase_ = false;

        if (precommit_failed_) {
            // txn itself was never touched -- not conflict-checked, not
            // applied -- so it's still fresh and may be retried as-is. See
            // CommitStatus::PrecommitConflict.
            CommitResult failure = std::move(*precommit_failure_);
            precommit_failure_.reset();
            failure.status = CommitStatus::PrecommitConflict;
            return failure;
        }
    }

    // The main transaction gets its conflict-checking against any
    // pre-transaction(s) just published above for free: check_and_apply()'s
    // check_id_overlap() scans the changelog (which now includes them), and
    // apply_transaction_contents()'s Ref<> validation runs against the
    // now-current state -- exactly as if those pre-transactions were made by
    // another writer racing this one. See PreTransactionsFn.
    //
    // Inlined rather than routed through commit_locked(): the main
    // transaction, unlike a pre-transaction (run_pre_commit_transaction()),
    // must still go through the pre_commit_ veto hook below.
    std::unordered_map<std::uint32_t, Id> remap;
    if (auto failure = check_and_apply(txn, remap)) return std::move(*failure);

    if (changes_.empty()) {
        // Everything in txn had already been applied by an earlier
        // try_commit() on this same Transaction (or every local create was
        // locally cancelled) -- a no-op success, not a fresh publish.
        return CommitResult{
            CommitStatus::Committed, snapshot(), {}, std::nullopt, {}, std::nullopt};
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

    return publish_now(std::move(remap));
}

}  // namespace model
