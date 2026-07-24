#include "model/model.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

namespace model {

namespace detail {
// Turns typeid(Derived).name() (e.g. "6Widget" or "N4example7AccountE") into the
// human-readable "Widget"/"example::Account" that Object<Derived>::type() hands
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

/// Total order over Id (index, then gen) -- NOT the same relation as
/// operator==, which is what Id actually needs for its own invariants
/// (different generations of the same slot are different objects). Exists
/// purely as the sort/binary_search key for apply_transaction_contents's
/// `pending` vector, so it stays local to this file rather than becoming
/// part of Id's public interface. A function OBJECT, not a plain function:
/// std::sort/std::binary_search take it by value as a template parameter
/// either way, but a free function decays to a pointer that GCC's -O2
/// measurably failed to inline through here (seen as its own separate frame
/// in basic_record_bench's callgrind profile) -- a distinctly-typed functor
/// gives the template instantiation a unique type to inline instead.
struct IdLess {
    bool operator()(Id a, Id b) const noexcept {
        return a.index != b.index ? a.index < b.index : a.gen < b.gen;
    }
};

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

// Needs Lease complete (the `lease_->m` access) -- see model.h's forward
// declaration of Lease and this function's own doc comment there. `lease_`
// is null only for a default-constructed Snapshot, which none of this
// method's four callers (find_by_scan_field/find_by_cached_field/
// for_each_referrer/find_cached_referrers) can reach without also failing
// their own `!root_` check first -- but checking here too costs nothing and
// doesn't rely on call-site discipline to stay safe.
void Snapshot::record_field_lookup(const std::type_info& type, const void* field,
                                   bool cached) const {
    if (lease_) lease_->m->record_field_lookup(type, field, cached);
}

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
    if (q_.empty())
        return false;  // woke because closed_, not because work arrived: no more events, ever
    out =
        std::move(q_.front());  // move out, not copy: an Update carries a Snapshot (pins a version)
                                // and a full Change vector -- no reason to duplicate either
    q_.pop_front();
    return true;
}

bool Subscription::try_drain(Update& out) {
    std::lock_guard lk(m_);
    if (q_.empty())
        return false;  // non-blocking: nothing queued right now, caller decides what to do
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
        for (auto& c : *u.changes) apply(c);
    for (auto& c : *tail.changes) apply(c);

    Update out;
    out.snapshot = std::move(tail.snapshot);  // the newest version -- every older Snapshot in the
                                              // queue is dropped along with q_ below
    out.coalesced = true;  // tells the consumer this Update skipped intermediate versions
    auto merged_changes = std::make_shared<std::vector<Change>>();
    merged_changes->reserve(merged.size());
    for (auto& [id, kind] : merged) merged_changes->push_back({id, kind, tags.at(id)});
    out.changes = std::move(merged_changes);  // this IS a genuinely new, synthesized changeset --
                                              // nothing to share with q_'s or tail's own changes

    q_.clear();  // drops the intermediate snapshots -- the whole point (each one was pinning a
                 // version, and everything retired since, alive)
    q_.push_back(std::move(out));
}

void Subscription::close() {
    std::lock_guard lk(m_);
    closed_ = true;    // wait()'s predicate re-checks this: any blocked or future wait() call
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
    for (auto& [v, p] : reap_queue_)
        delete p;  // v (the retirement version) is irrelevant now --
                   // no live snapshot can exist past ~Model()
    for (auto& [v, p] : retired_)
        delete p;            // an in-flight try_commit() attempt's retirees,
                             // never published (only reached via a leaked
                             // Transaction -- defensive, not the normal path)
    for (auto& ch : spine_)  // every object still live in the last-published Root
        for (std::uint32_t i = 0; i < kChunkSize; ++i)
            delete ch->obj[i];  // null slots: delete(nullptr) is a no-op
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

        // Free everything no live snapshot can still see. reap_queue_ is
        // ALWAYS sorted ascending by version (see its own comment), so every
        // freeable entry is a prefix of the queue -- pop it from the front
        // instead of re-testing the whole queue every pass. Without this, a
        // single reader lagging a few commits behind turns every subsequent
        // pass into an O(backlog) rescan of entries already known unfreeable.
        std::size_t freed = 0;
        while (!reap_queue_.empty() && reap_queue_.front().first <= min_live) {
            delete reap_queue_.front().second;
            reap_queue_.pop_front();
            ++freed;
        }
        if (freed) reap_backlog_.fetch_sub(freed, std::memory_order_relaxed);

        // Completing a pass -- even one that freed nothing -- advances the
        // round counter and wakes wait_for_reclamation(), whose contract is
        // "at least one full pass ran after I asked," not "something got freed."
        reap_done_round_++;
        reap_done_cv_.notify_all();

        if (reaper_stop_ && reap_queue_.empty())
            return;  // re-check: this pass may have been the
                     // last one needed to drain everything
    }
}

void Model::enqueue_retired(std::vector<std::pair<std::uint64_t, const ObjectBase*>> batch) {
    if (batch.empty())
        return;  // nothing to hand off (e.g. a commit that touched no existing object)
    // Counted BEFORE being made visible in reap_queue_ (under reap_mu_
    // below), so reap_backlog() -- which adds this atomic to
    // reap_queue_.size() -- never transiently UNDERcounts a batch that's
    // already enqueued but not yet reflected here; the reverse order could.
    reap_backlog_.fetch_add(batch.size(), std::memory_order_relaxed);
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

const ObjectBase* Model::peek_raw(Id id) const {
    if (!id) return nullptr;  // Id{} (default-constructed): never a valid handle, by construction
    // Slot index splits into (chunk index, offset within chunk) via a shift
    // and a mask, rather than division/modulo, because kChunkSize is a power
    // of two -- see Chunk's own doc comment for why chunking exists at all
    // (bounding a single COW clone's cost).
    const std::uint32_t c = id.index >> kChunkBits;
    const std::uint32_t i = id.index & kChunkMask;
    if (c >= spine_.size()) return nullptr;  // never allocated: chunk doesn't exist yet
    if (spine_[c]->gen[i] != id.gen)
        return nullptr;        // slot was recycled since this Id was
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
        log(PushFreeSlot{s});  // rolls back the pop
        const std::uint32_t c = s >> kChunkBits, i = s & kChunkMask;
        if (c < spine_.size() && spine_[c]->gen[i] >= kGenMax) {
            ++exhausted_slots_;  // permanently retired; never returned to circulation
            log(DecExhaustedSlots{});
            continue;
        }
        return s;
    }
    const std::uint32_t s = next_slot_++;
    log(DecNextSlot{});
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
    // Captured before being overwritten, purely so the rollback closure below can
    // restore EXACTLY what was there -- not "the current baseline" (which
    // could itself have changed by the time rollback runs, if this slot were
    // touched more than once in one attempt), but the specific prior value
    // this one write is rolling back.
    const ObjectBase* prev_obj = ch->obj[i];
    const std::uint32_t prev_gen = ch->gen[i];
    ch->obj[i] = obj;
    ch->gen[i] = gen;
    // RestoreSlot re-derives (cc, ii, c2) from `slot` rather than storing
    // `ch`/`i` directly: replay can happen after further COW churn this same
    // attempt, so `ch` (a raw Chunk* into a specific shared_ptr<Chunk>
    // generation) could be dangling by then -- re-running cow(cc) gets
    // whatever chunk object is currently live for that index instead. See
    // apply_rollback_op(RestoreSlot).
    log(RestoreSlot{slot, prev_obj, prev_gen});
}

std::optional<Model::IntegrityError> Model::validate(const ObjectBase* o,
                                                      const std::vector<Id>* pending) const {
    // Referential integrity is enforced *here*, at apply time, against the
    // CURRENT (latest) state -- not the transaction's base. That's what makes
    // "Ref<T> re-validated against latest, not just base" fall out for free,
    // rather than needing a bespoke second check. Returning the violation
    // (rather than asserting) lets try_commit() classify the failure -- did
    // the target exist at the transaction's own base()? -- and turn a
    // concurrent deletion into a Conflict rather than an Invalid rejection.
    // See IntegrityError::bad_target and try_commit()'s error handling.
    //
    // A target in `pending` (minted this attempt, slot not necessarily
    // installed yet -- see the pre-mint pass in apply_transaction_contents)
    // passes: it either installs later in this same attempt or the whole
    // attempt rolls back, so it can never be published dangling.
    std::optional<IntegrityError> err;
    o->each_ref([&](const void* /*field*/, const char* name, Id target, bool nullable) {
        if (err) return;  // keep the FIRST violation; the attempt aborts either way
        if (!target) {
            if (!nullable)
                err = IntegrityError{std::string("null Ref in field ") + name + " of " + o->type(),
                                     Id{}};
            return;
        }
        if (!peek_raw(target) &&
            !(pending && std::binary_search(pending->begin(), pending->end(), target, IdLess{})))
            err = IntegrityError{
                std::string(o->type()) + " field " + name + " references a dead object", target};
    });
    return err;
}

std::unordered_map<std::uint32_t, std::vector<std::pair<const void*, std::string>>>
Model::collect_update_baseline_field_keys(const Transaction& txn) const {
    std::unordered_map<std::uint32_t, std::vector<std::pair<const void*, std::string>>> out;
    out.reserve(txn.local_updated_.size());
    for (const auto& [slot, clone] : txn.local_updated_) {
        // peek_raw(), not txn.update_baseline_: this runs from inside
        // apply_transaction_contents, the same "latest, not base" moment
        // apply_update() itself reads baseline from -- see validate()'s
        // comment. try_commit()'s conflict check already guarantees the two
        // agree for every id in local_updated_ (nothing else could have
        // touched it since txn.base()).
        const ObjectBase* baseline = peek_raw(clone->id);
        // A linear-scan vector, not an unordered_map -- see
        // reconcile_out_refs()'s comment for why (a handful of fields at
        // most).
        std::vector<std::pair<const void*, std::string>> old_keys;
        baseline->each_field_key(
            [&](const void* field, std::string key) { old_keys.emplace_back(field, std::move(key)); });
        out.emplace(slot, std::move(old_keys));
    }
    return out;
}

std::optional<Model::IntegrityError> Model::validate_field_key_uniqueness(
    const Transaction& txn,
    const std::unordered_map<std::uint32_t, std::vector<std::pair<const void*, std::string>>>&
        old_keys) const {
    // Per define_keys() field: every (key -> claiming object) this
    // transaction's creates/updates will install, plus the set of keys some
    // update in this same transaction is moving OFF of. `claims` uses the
    // object's own address purely as an identity token (to tell "the same
    // object claimed this key twice, harmlessly" -- impossible, each_field_key
    // reports one value per field -- from "two DIFFERENT objects want it").
    struct FieldPlan {
        std::unordered_map<std::string, const void*> claims;
        std::unordered_set<std::string> vacated;
    };
    std::unordered_map<const void*, FieldPlan> plans;

    for (const auto& obj : txn.local_created_) {
        if (!obj) continue;  // cancelled locally (see remove_raw): nothing to claim
        std::optional<IntegrityError> err;
        obj->each_field_key([&](const void* field, std::string key) {
            if (err) return;  // keep the FIRST violation, same convention as validate()
            auto& plan = plans[field];
            const void* self = obj.get();
            auto [it, inserted] = plan.claims.try_emplace(key, self);
            if (!inserted && it->second != self)
                err = IntegrityError{"duplicate key '" + it->first +
                                     "' claimed by more than one object within the same transaction",
                                     Id{}};
        });
        if (err) return err;
    }

    for (const auto& [slot, clone] : txn.local_updated_) {
        // old_keys[slot] was already collected by
        // collect_update_baseline_field_keys() -- one entry per id in
        // local_updated_, unconditionally -- so this reads it rather than
        // walking baseline->each_field_key() itself; see this function's own
        // doc comment in model.h.
        const auto old_keys_it = old_keys.find(slot);
        assert(old_keys_it != old_keys.end() &&
              "collect_update_baseline_field_keys() covers every id in local_updated_");
        const std::vector<std::pair<const void*, std::string>>& obj_old_keys = old_keys_it->second;

        std::optional<IntegrityError> err;
        clone->each_field_key([&](const void* field, std::string new_key) {
            if (err) return;
            const auto oldit = std::find_if(obj_old_keys.begin(), obj_old_keys.end(),
                                            [&](const auto& p) { return p.first == field; });
            if (oldit != obj_old_keys.end() && oldit->second == new_key) return;  // unchanged

            auto& plan = plans[field];
            if (oldit != obj_old_keys.end()) plan.vacated.insert(oldit->second);
            const void* self = clone.get();
            auto [it, inserted] = plan.claims.try_emplace(new_key, self);
            if (!inserted && it->second != self)
                err = IntegrityError{"duplicate key '" + it->first +
                                     "' claimed by more than one object within the same transaction",
                                     Id{}};
        });
        if (err) return err;
    }

    // Finally, check every claim against the CURRENTLY COMMITTED holder (if
    // any) -- unless that holder is itself vacating this same key in this
    // same transaction, in which case it's not a blocker regardless of which
    // of the two ends up applying first (see reconcile_field_keys()'s own
    // conditional erase for the mutation-order half of this).
    for (const auto& [field, plan] : plans) {
        const auto fit = by_field_.find(field);
        if (fit == by_field_.end()) continue;  // this field has no committed entries at all
        for (const auto& [key, owner] : plan.claims) {
            (void)owner;
            if (plan.vacated.count(key)) continue;
            if (const Id* holder = fit->second.get(key))
                return IntegrityError{"key '" + key + "' is already in use by object " +
                                          std::to_string(holder->index) + ":" +
                                          std::to_string(holder->gen),
                                      Id{}};
        }
    }
    return std::nullopt;
}

// referrers_ maintenance: the writer-private reverse index that drives
// cascade delete (invariant 8). Keyed by TARGET slot index (bare, not a full
// Id -- only the live generation of a slot can ever be the target of a
// live Ref, so the generation would be redundant), each bucket is a
// vector<RefEdge> naming every field, on every object, currently pointing at
// that slot. add_out_refs/drop_out_refs add or remove an object's WHOLE
// outgoing edge set at once (create/delete); reconcile_out_refs below
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
        log(ReferrersPopBack{target.index});  // exact inverse of the push_back above
    });
}

void Model::drop_out_refs(const ObjectBase* o) {
    const Id from = o->id;
    o->each_ref([&](const void* field, const char* /*name*/, Id target, bool /*nullable*/) {
        if (!target) return;
        auto it = referrers_.find(target.index);
        if (it == referrers_.end())
            return;  // nothing indexed for this target (shouldn't happen if
                     // add_out_refs was called for every create, but a
                     // missing bucket is harmless to tolerate here)
        auto& v = it->second;
        // Find the specific edge this (from, field) pair added -- v may hold
        // edges from many OTHER objects/fields pointing at the same target.
        auto pos = std::find_if(v.begin(), v.end(), [&](const RefEdge& e) {
            return e.from == from && e.field == field;
        });
        if (pos == v.end()) return;
        const RefEdge edge = *pos;  // saved for the rollback op -- `pos` itself won't survive the erase
        v.erase(pos);
        log(ReferrersPushEdge{target.index, edge});
    });
}

void Model::reconcile_out_refs(const ObjectBase* before, const ObjectBase* after) {
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
    // Objects have a handful of Ref<>/Opt<> fields at most, so a linear-scan
    // vector beats an unordered_map here: one heap allocation for the whole
    // vector instead of one hash-table node allocation per field.
    std::vector<std::pair<const void*, Id>> old_targets;
    before->each_ref([&](const void* field, const char*, Id target, bool) {
        old_targets.emplace_back(field, target);
    });

    after->each_ref([&](const void* field, const char*, Id new_target, bool nullable) {
        const auto it = std::find_if(old_targets.begin(), old_targets.end(),
                                     [&](const auto& p) { return p.first == field; });
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
                    log(ReferrersPushEdge{old_target.index, edge});
                }
            }
        }
        if (new_target) {
            referrers_[new_target.index].push_back(RefEdge{id, field, nullable});
            log(ReferrersPopBack{new_target.index});  // exact inverse of the push_back above
        }
    });
}

// First-touch-this-attempt rollback capture for the four whole-map-handle
// indexes -- see log_by_type_once()'s doc comment in model.h for why only
// the first touch of a given key needs to log anything.

void Model::log_by_type_once(TypeTag tag) {
    if (!dirty_by_type_.insert(tag).second) return;
    auto prev = by_type_[tag];
    // GenericRollback, not a dedicated Kind: this fires at most once per
    // DISTINCT type touched per attempt (bounded by #types, never by object
    // count), so it was never the cost the TxnRollbackOp conversion targets -- see
    // log()'s own doc comment.
    log(GenericRollback{[this, tag, prev = std::move(prev)]() mutable { by_type_[tag] = std::move(prev); }});
}

void Model::log_by_field_once(const void* field) {
    if (!dirty_by_field_.insert(field).second) return;
    auto prev = by_field_[field];
    log(GenericRollback{
        [this, field, prev = std::move(prev)]() mutable { by_field_[field] = std::move(prev); }});
}

void Model::log_by_cached_field_once(const void* field) {
    if (!dirty_by_cached_field_.insert(field).second) return;
    auto prev = by_cached_field_[field];
    log(GenericRollback{[this, field, prev = std::move(prev)]() mutable {
        by_cached_field_[field] = std::move(prev);
    }});
}

void Model::log_by_cached_reference_once(const void* field) {
    if (!dirty_by_cached_reference_.insert(field).second) return;
    auto prev = by_cached_reference_[field];
    log(GenericRollback{[this, field, prev = std::move(prev)]() mutable {
        by_cached_reference_[field] = std::move(prev);
    }});
}

// by_field_ maintenance: the define_keys() unique-key index (Root::by_field).
// One persistent map PER FIELD (by_field_[field]), each mapping that field's
// key string -> the single Id currently holding it -- unique, unlike the
// cached-field/cached-reference multimaps below. Every mutation captures the
// prior per-field map as a whole (`prev`; cheap, since PersistentMap sharing
// means this is a handle copy, not a deep copy) so the rollback log can restore
// it verbatim on rollback: a missed restore would leave by_field_ answering
// find_by_key() with a value that never actually committed.

void Model::add_field_keys(const ObjectBase* o) {
    const Id id = o->id;
    o->each_field_key([&](const void* field, std::string key) {
        // try_emplace, not operator[], and BEFORE log_by_field_once(): the
        // FIRST time this field is seen, seed it with node_pool_ -- see
        // apply_create's identical by_type_ seeding for why the ordering
        // matters (log_by_field_once's own by_field_[field] read would
        // otherwise default-construct an unpooled entry first, and
        // try_emplace would then find the key already present and silently
        // keep it unpooled).
        auto& sub =
            by_field_.try_emplace(field, pmap::PersistentMap<std::string, Id, pmap::StringHash>(&node_pool_))
                .first->second;
        log_by_field_once(field);  // captures the pre-attempt value once; see its own doc comment
        sub = sub.set(key, id);         // key is unique per field by construction (see
                                        // define_keys' contract); a collision here is a
                                        // caller bug, not something this layer detects
    });
}

void Model::drop_field_keys(const ObjectBase* o) {
    o->each_field_key([&](const void* field, std::string key) {
        log_by_field_once(field);
        auto& sub = by_field_[field];
        sub = sub.erase(key);
    });
}

void Model::reconcile_field_keys(
    const ObjectBase* before, const ObjectBase* after,
    const std::vector<std::pair<const void*, std::string>>* old_keys_hint) {
    const Id id = after->id;
    // old_keys: this object's OWN key per field, as it was before the
    // update. If the caller already computed this (old_keys_hint --
    // apply_update(), forwarding validate_field_key_uniqueness()'s own
    // per-update pass; see that method's old_keys_out doc comment), reuse it
    // instead of walking before->each_field_key() a second time. Every OTHER
    // caller (remove_raw()'s cascade-null branch, reconciling a survivor
    // against a baseline validate_field_key_uniqueness() never saw) computes
    // it here, same as before. A linear-scan vector, not an unordered_map --
    // see reconcile_out_refs()'s comment for why.
    std::vector<std::pair<const void*, std::string>> computed_old_keys;
    if (!old_keys_hint) {
        before->each_field_key([&](const void* field, std::string key) {
            computed_old_keys.emplace_back(field, std::move(key));
        });
        old_keys_hint = &computed_old_keys;
    }
    const std::vector<std::pair<const void*, std::string>>& old_keys = *old_keys_hint;

    after->each_field_key([&](const void* field, std::string new_key) {
        const auto it = std::find_if(old_keys.begin(), old_keys.end(),
                                     [&](const auto& p) { return p.first == field; });
        if (it != old_keys.end() && it->second == new_key) return;  // unchanged

        // Same rollback pattern as add_field_keys/drop_field_keys: capture the
        // whole prior map (cheap -- persistent, structure-shared), once per
        // attempt (log_by_field_once), and restore it on rollback. Without
        // this, a vetoed/conflicted attempt leaves by_field_ permanently
        // indexing values that never committed.
        log_by_field_once(field);
        auto& sub = by_field_[field];
        // Only erase the OLD key if we still hold it. local_updated_ is an
        // unordered_map -- apply order across objects in one transaction is
        // arbitrary -- so a same-transaction swap (this object vacates K
        // while some OTHER object in the same transaction claims K) may
        // already have overwritten this entry with that other object's id
        // by the time this runs. validate_field_key_uniqueness() has already
        // proven the transaction's FINAL state is collision-free; erasing
        // unconditionally here would still wipe out that other object's
        // legitimate, already-applied claim.
        if (it != old_keys.end()) {
            const Id* cur = sub.get(it->second);
            if (cur && *cur == id) sub = sub.erase(it->second);
        }
        sub = sub.set(new_key, id);
    });
}

// The cached-field (multimap) index. Buckets are persistent maps keyed by the
// Id's own bytes -- never flat vectors, which would make each mutation
// O(#duplicates of that value) -- and every mutation captures the prior OUTER
// map for the rollback log (a shared_ptr copy, same pattern as add_field_keys):
// a rollback that skipped these would leave the index claiming membership
// that never committed.

void Model::add_cached_fields(const ObjectBase* o) {
    const Id id = o->id;
    o->each_cached_field([&](const void* field, std::string key) {
        // Two-level structure: by_cached_field_[field] is the OUTER map (key
        // string -> bucket); `bucket`, if this key already has other
        // holders, is the INNER set collecting every object currently
        // holding that value. log_by_cached_field_once captures the outer
        // map's pre-attempt state, once, for the rollback log.
        //
        // try_emplace, not operator[], and BEFORE log_by_cached_field_once():
        // same ordering hazard as add_field_keys -- log_by_cached_field_once's
        // own by_cached_field_[field] read would otherwise default-construct
        // an unpooled entry first. See that function's comment.
        auto& sub = by_cached_field_
                       .try_emplace(field, pmap::PersistentMap<std::string, pmap::PersistentSet<Id, IdHash>,
                                                               pmap::StringHash>(&node_pool_))
                       .first->second;
        log_by_cached_field_once(field);
        const pmap::PersistentSet<Id, IdHash>* bucket = sub.get(key);
        // Pool-seeded empty bucket, not `{}`: the first object ever to hold
        // this particular value would otherwise start an unpooled lineage
        // for that bucket, same as the outer map above.
        sub = sub.set(key, (bucket ? *bucket : pmap::PersistentSet<Id, IdHash>(&node_pool_)).insert(id));
    });
}

void Model::drop_cached_fields(const ObjectBase* o) {
    const Id id = o->id;
    o->each_cached_field([&](const void* field, std::string key) {
        log_by_cached_field_once(field);
        auto& sub = by_cached_field_[field];
        const pmap::PersistentSet<Id, IdHash>* bucket = sub.get(key);
        if (!bucket) return;  // nothing indexed under this key -- nothing to remove
        auto nb = bucket->erase(id);  // nb: the bucket with just this id removed
        // An emptied bucket is dropped outright, so a value with no remaining
        // holders doesn't leave a tombstone entry behind.
        sub = nb.empty() ? sub.erase(key) : sub.set(key, nb);
    });
}

void Model::reconcile_cached_fields(const ObjectBase* before, const ObjectBase* after) {
    const Id id = after->id;
    // old_keys: this object's own cached-field value per field, as it was
    // BEFORE the update -- snapshotted up front so the `after` pass can diff
    // against it one field at a time. A linear-scan vector, not an
    // unordered_map -- see reconcile_out_refs()'s comment for why.
    std::vector<std::pair<const void*, std::string>> old_keys;
    before->each_cached_field(
        [&](const void* field, std::string key) { old_keys.emplace_back(field, std::move(key)); });

    after->each_cached_field([&](const void* field, std::string new_key) {
        const auto it = std::find_if(old_keys.begin(), old_keys.end(),
                                     [&](const auto& p) { return p.first == field; });
        if (it != old_keys.end() && it->second == new_key) return;  // unchanged

        // log_by_cached_field_once captures the OUTER map's pre-attempt state
        // (once) -- what the rollback log restores wholesale on rollback. sub is
        // written through directly across the two steps below (old value's
        // bucket shrinks/drops, new value's bucket grows) -- each
        // reassignment updates the map in place, so there's no separate
        // "cur" to install at the end.
        log_by_cached_field_once(field);
        auto& sub = by_cached_field_[field];
        if (it != old_keys.end()) {
            // Remove this id from its OLD value's bucket (ob), unless that
            // value was never actually indexed (e.g. this field just started
            // returning a cacheable value).
            if (const pmap::PersistentSet<Id, IdHash>* ob = sub.get(it->second)) {
                auto nb = ob->erase(id);  // ob with this id removed
                sub = nb.empty() ? sub.erase(it->second) : sub.set(it->second, nb);
            }
        }
        // Add this id to its NEW value's bucket, creating that bucket if this
        // is the first object ever to hold this particular value -- pool-
        // seeded, not `{}` (see add_cached_fields's identical bucket seeding).
        const pmap::PersistentSet<Id, IdHash>* bucket = sub.get(new_key);
        sub = sub.set(new_key, (bucket ? *bucket : pmap::PersistentSet<Id, IdHash>(&node_pool_)).insert(id));
    });
}

// The cached-reference (reverse multimap) index -- the read-side counterpart
// of referrers_, opt-in per Ref<>/Opt<> field via define_cached_references().
// Same persistent-bucket discipline as the cached-field trio above (never a
// flat vector; every mutation rollback-logged), keyed by the TARGET's Id bytes
// instead of a field's value, storing the REFERRER's Id in each bucket.
// each_cached_reference() already filters to just the declared fields, so
// these never do anything for a field not opted in.

void Model::add_cached_references(const ObjectBase* o) {
    const Id id = o->id;  // the REFERRER -- the value stored in the bucket, not the bucket's key
    o->each_cached_reference([&](const void* field, const char*, Id target, bool) {
        if (!target) return;  // Opt<> currently null: nothing to index
        // try_emplace, not operator[], and BEFORE log_by_cached_reference_once():
        // same ordering hazard as add_cached_fields -- see that function's
        // comment.
        auto& sub =
            by_cached_reference_
                .try_emplace(field, pmap::PersistentMap<Id, pmap::PersistentSet<Id, IdHash>, IdHash>(
                                        &node_pool_))
                .first->second;
        log_by_cached_reference_once(field);
        const pmap::PersistentSet<Id, IdHash>* bucket = sub.get(target);
        // Pool-seeded empty bucket, not `{}` -- see add_cached_fields's
        // identical bucket seeding.
        sub = sub.set(target, (bucket ? *bucket : pmap::PersistentSet<Id, IdHash>(&node_pool_)).insert(id));
    });
}

void Model::drop_cached_references(const ObjectBase* o) {
    const Id id = o->id;
    o->each_cached_reference([&](const void* field, const char*, Id target, bool) {
        if (!target) return;
        log_by_cached_reference_once(field);
        auto& sub = by_cached_reference_[field];
        const pmap::PersistentSet<Id, IdHash>* bucket = sub.get(target);
        if (!bucket) return;
        auto nb = bucket->erase(id);
        // An emptied bucket is dropped outright, so a target with no
        // remaining referrers doesn't leave a tombstone entry behind.
        sub = nb.empty() ? sub.erase(target) : sub.set(target, nb);
    });
}

void Model::reconcile_cached_references(const ObjectBase* before, const ObjectBase* after) {
    const Id id = after->id;
    // A linear-scan vector, not an unordered_map -- see
    // reconcile_out_refs()'s comment for why.
    std::vector<std::pair<const void*, Id>> old_targets;
    before->each_cached_reference([&](const void* field, const char*, Id target, bool) {
        old_targets.emplace_back(field, target);
    });

    after->each_cached_reference([&](const void* field, const char*, Id new_target, bool) {
        const auto it = std::find_if(old_targets.begin(), old_targets.end(),
                                     [&](const auto& p) { return p.first == field; });
        const Id old_target = (it != old_targets.end()) ? it->second : Id{};
        if (new_target == old_target) return;  // unchanged (incl. both still null)

        log_by_cached_reference_once(field);
        auto& sub = by_cached_reference_[field];
        if (old_target) {
            if (const pmap::PersistentSet<Id, IdHash>* ob = sub.get(old_target)) {
                auto nb = ob->erase(id);
                sub = nb.empty() ? sub.erase(old_target) : sub.set(old_target, nb);
            }
        }
        if (new_target) {
            // Pool-seeded empty bucket, not `{}` -- see add_cached_fields's
            // identical bucket seeding.
            const pmap::PersistentSet<Id, IdHash>* bucket = sub.get(new_target);
            sub = sub.set(new_target,
                          (bucket ? *bucket : pmap::PersistentSet<Id, IdHash>(&node_pool_)).insert(id));
        }
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

Model::Diagnostics Model::diagnostics() const {
    Diagnostics diag;
    diag.transactions_begun = next_txn_id_.load(std::memory_order_relaxed) - 1;

    diag.commits_succeeded = commits_succeeded_.load(std::memory_order_relaxed);
    diag.commits_conflicted = commits_conflicted_.load(std::memory_order_relaxed);
    diag.commits_vetoed = commits_vetoed_.load(std::memory_order_relaxed);
    diag.commits_invalid = commits_invalid_.load(std::memory_order_relaxed);
    diag.commits_precommit_conflicted =
        commits_precommit_conflicted_.load(std::memory_order_relaxed);

    {
        std::lock_guard lk(commit_mu_);
        diag.version = version_;

        diag.live_by_type.reserve(by_type_.size());
        for (const auto& [tag, ids] : by_type_) {
            (void)tag;  // TypeTag carries no name of its own -- see TypeCount's doc comment
            Diagnostics::TypeCount tc;
            tc.live_count = ids.size();
            if (tc.live_count > 0) {
                Id sample;
                bool found = false;
                ids.for_each([&](const Id& id) {
                    if (!found) {
                        sample = id;
                        found = true;
                    }
                });
                const ObjectBase* obj = peek_raw(sample);
                tc.type_name = obj ? obj->type() : "<no live object of this type>";
            } else {
                tc.type_name = "<no live object of this type>";
            }
            diag.live_object_count += tc.live_count;
            diag.live_by_type.push_back(std::move(tc));
        }

        diag.chunk_count = spine_.size();
        diag.slots_allocated = next_slot_;
        diag.slots_free = free_slots_.size();
        diag.slots_exhausted = exhausted_slots_;

        diag.key_indexed_fields = by_field_.size();
        diag.cached_value_indexed_fields = by_cached_field_.size();
        diag.cached_reference_indexed_fields = by_cached_reference_.size();

        diag.reverse_index_targets = referrers_.size();
        for (const auto& [slot, edges] : referrers_) {
            (void)slot;
            diag.reverse_index_edges += edges.size();
        }

        diag.pre_transactions_hook_installed = static_cast<bool>(pre_transactions_);
        diag.pre_commit_hook_installed = static_cast<bool>(pre_commit_);

        diag.reap_backlog = reap_backlog_.load(std::memory_order_relaxed) + retired_.size();

        diag.retained_commit_history.reserve(changelog_.size());
        for (const auto& entry : changelog_)
            diag.retained_commit_history.emplace_back(entry.version, entry.changes->size());
    }

    {
        std::lock_guard lk(ver_mu_);
        diag.live_snapshot_versions = live_.size();
        for (const auto& [ver, refcount] : live_) {
            (void)ver;
            diag.live_snapshot_refs += static_cast<std::size_t>(refcount);
        }
        diag.reclamation_watermark = live_.empty() ? diag.version : live_.begin()->first;
    }

    {
        std::lock_guard lk(subs_mu_);
        diag.subscriber_count = subs_.size();
    }

    return diag;
}

// ---------------------------------------------------------------------------
// Field lookup stats
// ---------------------------------------------------------------------------

void Model::record_field_lookup(const std::type_info& type, const void* field,
                                bool cached) const {
    const FieldLookupKey key{&type, field};
    {
        // Steady-state path: every call after the first-ever one for this
        // (type, field) pair on this Model takes only a SHARED lock --
        // concurrent with every other reader, contending only with the
        // (rare) insert path below.
        std::shared_lock rlk(field_lookup_mu_);
        if (auto it = field_lookup_counts_.find(key); it != field_lookup_counts_.end()) {
            (cached ? it->second.cached : it->second.uncached)
                .fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    // First-ever lookup of this field on this Model: the shared lock above
    // found nothing, so take the exclusive lock to insert it. operator[]
    // default-constructs a fresh FieldLookupCounters if another thread lost
    // this same race and inserted first -- either way, `field_lookup_counts_
    // [key]` after this line is THE entry for `key`, inserted exactly once.
    std::unique_lock wlk(field_lookup_mu_);
    FieldLookupCounters& c = field_lookup_counts_[key];
    (cached ? c.cached : c.uncached).fetch_add(1, std::memory_order_relaxed);
}

LookupCounts Model::lookup_stats_raw(const std::type_info& type, const void* field) const {
    std::shared_lock lk(field_lookup_mu_);
    auto it = field_lookup_counts_.find(FieldLookupKey{&type, field});
    if (it == field_lookup_counts_.end()) return {};
    return {it->second.cached.load(std::memory_order_relaxed),
           it->second.uncached.load(std::memory_order_relaxed)};
}

Model::LookupDiagnostics Model::lookup_diagnostics() const {
    LookupDiagnostics report;
    std::shared_lock lk(field_lookup_mu_);
    report.stats.reserve(field_lookup_counts_.size());
    for (const auto& [key, counters] : field_lookup_counts_) {
        report.stats.emplace(key, LookupCounts{counters.cached.load(std::memory_order_relaxed),
                                               counters.uncached.load(std::memory_order_relaxed)});
    }
    return report;
}

std::string Model::LookupDiagnostics::to_string() const {
    // A plain vector copy for sorting -- unordered_map itself can't be
    // sorted in place. This is also the only place demangling `type` (a
    // real string-allocating, non-trivial operation) ever happens: `stats`
    // itself stores nothing but a type_info pointer, a field_tag address,
    // and two integers, so building a LookupDiagnostics never pays for a
    // name nobody asked to print yet.
    std::vector<std::pair<FieldLookupKey, LookupCounts>> sorted(stats.begin(), stats.end());
    std::sort(sorted.begin(), sorted.end(),
             [](const auto& a, const auto& b) {
                 return (a.second.cached_calls + a.second.uncached_calls) >
                       (b.second.cached_calls + b.second.uncached_calls);
             });

    std::ostringstream out;
    out << "Field lookup stats (cached vs uncached), " << sorted.size() << " field(s) recorded:\n";
    if (sorted.empty()) out << "  (none -- no lookup function has been called yet)\n";
    for (const auto& [key, counts] : sorted) {
        const std::uint64_t total = counts.cached_calls + counts.uncached_calls;
        const double cached_pct =
            total > 0 ? 100.0 * static_cast<double>(counts.cached_calls) / static_cast<double>(total)
                     : 0.0;
        // Prefer the real field name (opted in via FieldKeyReader::key()/
        // RefIndexReader::index() -- see detail::register_field_name_once)
        // over the address: a real name is a strictly better disambiguator
        // than a hex address, so once we have one there's no reason to show
        // both. A field never given a name falls back to the address, same
        // as before this option existed.
        const std::string field_name = detail::field_name_of(key.field);
        std::ostringstream label;
        label << detail::demangle_type_name(*key.type);
        if (!field_name.empty()) {
            label << "::" << field_name;
        } else {
            label << " @" << std::hex << std::setw(12) << std::setfill('0')
                  << reinterpret_cast<std::uintptr_t>(key.field) << std::dec << std::setfill(' ');
        }
        out << "  " << std::left << std::setw(34) << label.str() << std::right
            << "  cached=" << std::setw(8) << counts.cached_calls
            << "  uncached=" << std::setw(8) << counts.uncached_calls << "  (" << std::fixed
            << std::setprecision(1) << cached_pct << "% cached)\n";
    }
    return out.str();
}

// ---------------------------------------------------------------------------
// try_commit() internals
// ---------------------------------------------------------------------------

std::optional<Model::IntegrityError> Model::apply_create(std::unique_ptr<ObjectBase> o,
                                                          std::unordered_map<std::uint32_t, Id>& remap,
                                                          const std::vector<Id>& pending,
                                                          bool keep_undo) {
    const std::uint32_t local_index = o->id.index;  // still local; the remap key

    // May reference ANY local create in this txn -- earlier, later, or this
    // very object (the pre-mint pass filled the whole table before the
    // first create applied). Unmapped now means only a cancelled create or
    // a stray id from another transaction. Nothing is logged yet, so an
    // unmapped local id can just return; `o` frees itself.
    bool unmapped = false;
    o->remap_refs(RefRemapper{remap, &unmapped});
    if (unmapped) return IntegrityError{kUnmappedLocalMsg, Id{}};

    const auto minted = remap.find(local_index);
    assert(minted != remap.end() && "the pre-mint pass covers every live create");
    o->id = minted->second;

    // On a violation, this object's slot allocation (pre-mint pass) is
    // already logged, so rollback_apply() reclaims it; the object is still
    // owned by `o` and is freed by unique_ptr on the early return. A target
    // in `pending` -- minted this attempt, possibly not yet installed -- is
    // as good as installed here; see validate()'s doc comment.
    if (auto err = validate(o.get(), &pending)) return err;

    const Id id = o->id;
    ObjectBase* raw = o.release();

    set_slot(id.index, raw, id.gen);  // logs the inverse (restores prev obj + gen)

    // Every object gets an (internal, Id-keyed) entry in its type's
    // enumeration index, unconditionally -- this is what for_each<T>() scans.
    const TypeTag tag = raw->tag();
    // emplace, not operator[], and BEFORE log_by_type_once(): the FIRST time
    // this tag is seen, seed it with node_pool_ so every Node/Leaf this (and
    // every later) insert()/erase() on it allocates from the pool instead of
    // plain new/delete -- see node_pool_'s own comment in model.h. This MUST
    // run before log_by_type_once(), which itself reads by_type_[tag] via
    // operator[] to capture the pre-attempt value for rollback -- doing that
    // first would default-construct an UNPOOLED (mem_ == nullptr) entry
    // first, and try_emplace() then silently keeps that already-present,
    // unpooled value instead of seeding the pool at all (a real bug this
    // ordering fixes: measured, before the fix, as zero pool allocations
    // ever happening). A tag already present (either from an earlier
    // create(), or from log_by_type_once() below) is left untouched by
    // try_emplace() either way -- its existing, already-seeded value carries
    // forward via insert()'s own copy of mem_ regardless.
    auto& sub = by_type_.try_emplace(tag, pmap::PersistentSet<Id, IdHash>(&node_pool_)).first->second;
    log_by_type_once(tag);
    sub = sub.insert(id);

    add_out_refs(raw);
    add_field_keys(raw);
    add_cached_fields(raw);
    add_cached_references(raw);

    changes_.push_back({id, ChangeKind::Created, tag});
    log(PopChanges{});

    // Undo: the inverse of a create is removing it. No snapshot data
    // needed -- unlike pending_undo_actions_'s other two action kinds, there is no
    // pre-image to capture. Not logged for rollback -- see pending_undo_actions_'s
    // own comment: it's cleared wholesale in rollback_apply(), not
    // unwound entry-by-entry like changes_. Skipped entirely (not just
    // discarded later) when keep_undo is false -- see try_commit_without_
    // undo()'s own comment for why that distinction matters.
    if (keep_undo) pending_undo_actions_.push_back({UndoAction::Kind::Remove, id, nullptr});

    // Never published, nothing else owns it: rollback_apply() deletes
    // everything in txn_created_. Deliberately NOT logged as a rollback op -- the
    // rollback delete loop consumes this list directly, and a pop-rollback would
    // empty it first.
    txn_created_.push_back(raw);

    return std::nullopt;  // remap[local_index] was set by the pre-mint pass
}

std::optional<Model::IntegrityError> Model::apply_update(
    std::unique_ptr<ObjectBase> clone, std::unordered_map<std::uint32_t, Id>& remap,
    const std::vector<std::pair<const void*, std::string>>* old_field_keys_hint, bool keep_undo) {
    ObjectBase* raw = clone.release();

    // Log the clone's deletion FIRST (before anything can fail), so in
    // reverse replay it runs LAST -- after set_slot's rollback has repointed the
    // slot back at the baseline. Otherwise we'd free `raw` while the slot
    // still referenced it, and an early error return below would leak it.
    log(DeleteObject{raw});

    bool unmapped = false;
    raw->remap_refs(
        RefRemapper{remap, &unmapped});  // may reference a local create in this same txn
    if (unmapped) return IntegrityError{kUnmappedLocalMsg, Id{}};

    // Against CURRENT peek(), not the transaction's base -- see validate()'s comment.
    if (auto err = validate(raw)) return err;

    const Id id = raw->id;
    const ObjectBase* baseline = peek_raw(id);
    // try_commit()'s conflict check (check_id_overlap) already guarantees
    // `baseline` is non-null and unchanged since txn.base(): if any other
    // commit had touched this id since then, this attempt would have been
    // rejected as a Conflict before ever reaching apply.

    // Undo: capture BEFORE baseline is retired -- this IS the pre-image a
    // RestoreUpdate action needs. Same reasoning as changes_: not logged
    // for rollback, since pending_undo_actions_ is cleared wholesale on failure.
    // The clone() itself -- not just its retention -- is skipped when
    // keep_undo is false; see try_commit_without_undo()'s own comment.
    if (keep_undo)
        pending_undo_actions_.push_back(
            {UndoAction::Kind::RestoreUpdate, id, std::unique_ptr<ObjectBase>(baseline->clone())});

    retire(baseline);
    log(PopRetired{});

    set_slot(id.index, raw, id.gen);  // logs restore of baseline + its generation

    changes_.push_back({id, ChangeKind::Updated, raw->tag()});
    log(PopChanges{});

    // Unlike the single-writer design this project's sibling uses, `raw` is
    // already fully written by the time we get here (the caller finished
    // writing to it back when building the Transaction) -- so reconciliation
    // runs immediately, not in a later deferred pass. This MUST happen before
    // remove_raw()'s cascade BFS runs (see try_commit()'s phase order): a
    // same-transaction "repoint away from X, then delete X" needs referrers_
    // already updated, or the repointed-away-from object would incorrectly
    // be dragged into X's cascade.
    reconcile_out_refs(baseline, raw);
    reconcile_field_keys(baseline, raw, old_field_keys_hint);
    reconcile_cached_fields(baseline, raw);
    reconcile_cached_references(baseline, raw);
    return std::nullopt;
}

ObjectBase* Model::clone_for_cascade_null(Id id, bool keep_undo) {
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
    const ObjectBase* cur = peek_raw(id);
    if (!cur) return nullptr;

    // Undo: same action kind as apply_update's -- "this survivor's field
    // got cascade-nulled" and "this object got explicitly updated" are the
    // same fix from undo's perspective: restore the captured pre-image.
    // Skipped entirely (not just discarded) when keep_undo is false.
    if (keep_undo)
        pending_undo_actions_.push_back(
            {UndoAction::Kind::RestoreUpdate, id, std::unique_ptr<ObjectBase>(cur->clone())});

    ObjectBase* copy = cur->clone();
    log(DeleteObject{copy});

    retire(cur);
    log(PopRetired{});

    set_slot(id.index, copy, id.gen);

    changes_.push_back({id, ChangeKind::Updated, copy->tag()});
    log(PopChanges{});

    return copy;
}

std::vector<Id> Model::remove_raw(std::vector<Id> work, bool keep_undo) {
    // Breadth-first cascade delete, resolved here (at apply time, under
    // commit_mu_) and never eagerly (invariant 8). `work` -- taken by value,
    // so the caller's vector becomes this BFS's own frontier with no extra
    // copy when passed as an rvalue (see model.h's doc comment) -- is seeded
    // with EVERY Transaction::remove() intent this attempt resolves (see
    // this function's own doc comment in model.h for why one shared BFS
    // replaces one call per intent); `visited` is the set of slot indices
    // already processed, which both terminates cycles (a self- or
    // mutually-referential graph would otherwise loop forever) and prevents
    // processing the same victim twice -- including a victim reachable from
    // more than one seed's cascade. `killed` accumulates every id actually
    // deleted, in visitation order, for the caller (try_commit()) to report.
    std::vector<Id> killed;
    std::unordered_set<std::uint32_t> visited;

    while (!work.empty()) {
        const Id x = work.back();  // `x`: the id currently being resolved this iteration
        work.pop_back();
        if (!peek_raw(x)) continue;  // already gone (e.g. cascaded in from another branch of the BFS)
        if (!visited.insert(x.index).second) continue;  // cycles terminate here

        // Copy the referrer list: we are about to mutate it (both directly,
        // via the erase below, and indirectly, via reconcile_out_refs
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
            if (!peek_raw(e.from)) continue;  // referrer itself already deleted this same pass
            if (e.nullable) {
                // `baseline` is captured BEFORE clone_for_cascade_null() installs
                // the clone, so reconcile_out_refs (etc.) below can diff
                // "before this field was nulled" against "after" -- see
                // clone_for_cascade_null's own comment for why reconciliation
                // must happen here, after null_ref(), not inside that helper.
                const ObjectBase* baseline = peek_raw(e.from);
                if (ObjectBase* m = clone_for_cascade_null(e.from, keep_undo)) {
                    m->null_ref(e.field);
                    reconcile_out_refs(baseline, m);
                    reconcile_field_keys(baseline, m);
                    reconcile_cached_fields(baseline, m);
                    reconcile_cached_references(baseline, m);
                }
            } else {
                work.push_back(e.from);  // dies with its target
            }
        }

        const ObjectBase* victim = peek_raw(x);
        if (!victim)
            continue;  // defensive, matching the peek()-then-check style used for every
                       // other id in this BFS (x itself, and each e.from above) --
                       // nothing in the edges loop above installs a null at slot x itself

        // Undo: the inverse of a delete is recreating it -- x's `id` here
        // is the OLD one, permanently stale the moment this BFS finishes
        // (invariant 5: generation never recycles). Captured before
        // anything below touches victim's data. Skipped entirely (not
        // just discarded) when keep_undo is false.
        if (keep_undo)
            pending_undo_actions_.push_back(
                {UndoAction::Kind::Recreate, x, std::unique_ptr<ObjectBase>(victim->clone())});

        drop_out_refs(victim);  // logs re-add of victim's outgoing edges

        // victim.index should have no remaining incoming edges (its referrers
        // were cascaded or nulled above), but if any survive, preserve them for
        // rollback. In practice this is empty; capture it to be exact.
        auto rit = referrers_.find(x.index);
        if (rit != referrers_.end()) {
            std::vector<RefEdge> saved = std::move(rit->second);
            referrers_.erase(rit);
            log(RestoreReferrersBucket{x.index, std::move(saved)});
        }

        const TypeTag tag = victim->tag();
        log_by_type_once(tag);
        auto& sub = by_type_[tag];
        sub = sub.erase(x);

        drop_field_keys(victim);
        drop_cached_fields(victim);
        drop_cached_references(victim);

        set_slot(x.index, nullptr, x.gen);  // logs restore of victim + its gen

        retire(victim);
        log(PopRetired{});

        free_slots_.push_back(x.index);
        log(PopFreeSlot{});

        changes_.push_back({x, ChangeKind::Deleted, tag});
        log(PopChanges{});

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

    // O(this transaction's own write set), not O(everything committed by
    // everyone since its base) -- see last_write_version_'s own comment.
    // `conflicts` collects the SPECIFIC ids that collided, for
    // ConflictInfo::ids -- not just a yes/no.
    for (std::uint32_t slot : written_slots) {
        if (slot < last_write_version_.size() && last_write_version_[slot] > txn.base_version())
            conflicts.push_back(last_write_id_[slot]);
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
    // Replay inverses in reverse. Each op exactly undoes one primitive
    // mutation, so commit-lock-protected state returns to its last committed
    // shape. std::visit + apply_rollback_op's overload set is exhaustive at
    // compile time -- see TxnRollbackOp's own doc comment. Moving *it in (rather
    // than visiting a reference) lets RestoreReferrersBucket/GenericRollback
    // move their payload out instead of copying it; every other alternative
    // is trivially-copyable-sized, so the move costs nothing extra there.
    for (auto it = txn_rollback_log_.rbegin(); it != txn_rollback_log_.rend(); ++it)
        std::visit([this](auto&& op) { apply_rollback_op(std::forward<decltype(op)>(op)); }, std::move(*it));
    txn_rollback_log_.clear();

    // Objects created this attempt were never published and are owned by
    // nothing after their slot-writes are undone. Free them.
    for (const ObjectBase* o : txn_created_) delete o;
    txn_created_.clear();

    changes_.clear();  // any survivors were popped by the log; clear defensively
    pending_undo_actions_.clear();  // not log()-replayed like changes_ -- see its own comment; a blunt
                            // clear is correct either way, since a failed attempt keeps nothing
    dirty_.clear();
    dirty_by_type_.clear();
    dirty_by_field_.clear();
    dirty_by_cached_field_.clear();
    dirty_by_cached_reference_.clear();
}

// ---------------------------------------------------------------------------
// begin() / try_commit()
// ---------------------------------------------------------------------------

Transaction Model::begin(std::string name, std::any data) {
    return begin(snapshot(), std::move(name), std::move(data));
}

Transaction Model::begin(Snapshot base, std::string name, std::any data) {
    return Transaction(this, std::move(base), std::move(name), std::move(data));
}

Transaction Snapshot::begin(std::string name, std::any data) const {
    assert(lease_ && "begin() on a default-constructed Snapshot -- no Model to build against");
    return lease_->m->begin(*this, std::move(name), std::move(data));
}

std::vector<Change> Transaction::estimate_changes_with_cascades() const {
    // Creates and updates are already known exactly -- no estimation needed.
    std::vector<Change> out = pending_changes_;
    if (remove_intents_.empty()) return out;

    // Read-only cascade ESTIMATE, against base() -- mirrors remove_raw()'s
    // real BFS, but referrers_ (the writer's fast reverse index) is
    // commit_mu_-protected and unreachable from here (Transaction building
    // is lock-free -- invariant 10), so this walks base()'s published state
    // directly via for_each_referrer_any(), at O(total live objects) cost
    // per pending remove. See estimate_changes_with_cascades()'s own doc comment for the
    // "estimate, not guarantee" caveats.
    std::unordered_set<std::uint32_t> visited;
    std::vector<Id> work(remove_intents_.begin(), remove_intents_.end());

    while (!work.empty()) {
        const Id x = work.back();
        work.pop_back();
        const ObjectBase* obj = base_.find_raw(x);
        if (!obj) continue;  // already dead even at base() -- nothing to estimate
        if (!visited.insert(x.index).second)
            continue;  // cycles terminate here, same as remove_raw()

        out.push_back({x, ChangeKind::Deleted, obj->tag()});

        base_.for_each_referrer_any(x, [&](Id from, const void*, bool nullable, TypeTag from_tag) {
            if (!base_.find_raw(from)) return;  // already accounted for, or already dead
            if (nullable) {
                out.push_back({from, ChangeKind::Updated, from_tag});  // estimated null-out
            } else {
                work.push_back(from);  // estimated to die with its target
            }
        });
    }
    return out;
}

std::optional<Model::IntegrityError> Model::apply_transaction_contents(
    Transaction& txn, std::unordered_map<std::uint32_t, Id>& remap, bool keep_undo) {
    // publish_now() never appends an entry when max_undo_list_size_ == 0 or
    // max_undo_bytes_ == 0 (see set_max_undo_list_size()/set_max_undo_
    // memory_bytes()'s doc comments) -- it re-checks both caps itself,
    // independently of what's passed here. So folding them into this LOCAL
    // keep_undo, before the apply loop below ever pushes to pending_undo_actions_, is
    // enough to skip the per-object clones (apply_update/remove_raw/
    // clone_for_cascade_null) that would only feed an entry publish_now() is
    // about to discard anyway -- no need to thread the fold back out to
    // either caller's own keep_undo. Both callers (via check_and_apply())
    // hold commit_mu_ already, so max_undo_list_size_/max_undo_bytes_ are
    // safe to read here.
    keep_undo = keep_undo && max_undo_list_size_ != 0 && max_undo_bytes_ != 0;

    // Size hint for every container the apply loop below grows one entry (or
    // more, for changes_/pending_undo_actions_/txn_rollback_log_) at a time -- creates+updates is
    // a lower bound (cascade deletes add more, unpredictably), but reserving
    // even that much upfront avoids the geometric-growth reallocations a
    // large transaction would otherwise pay on remap/pending/changes_/
    // pending_undo_actions_/txn_rollback_log_ every time. Only changes remap/pending/changes_/
    // pending_undo_actions_/txn_rollback_log_'s CAPACITY, never their contents.
    const std::size_t apply_size_hint = txn.local_created_.size() + txn.local_updated_.size();
    remap.reserve(remap.size() + apply_size_hint);
    changes_.reserve(changes_.size() + apply_size_hint);
    if (keep_undo) pending_undo_actions_.reserve(pending_undo_actions_.size() + apply_size_hint);
    txn_rollback_log_.reserve(txn_rollback_log_.size() + apply_size_hint);

    // old_field_keys: per-update baseline define_keys() fields -- collected
    // once, up front, and handed to BOTH validate_field_key_uniqueness()
    // (below) and apply_update() (later in this same function) so neither
    // has to walk a given baseline's each_field_key() itself; see
    // collect_update_baseline_field_keys()'s own doc comment.
    const std::unordered_map<std::uint32_t, std::vector<std::pair<const void*, std::string>>>
        old_field_keys = collect_update_baseline_field_keys(txn);

    // Whole-transaction key-uniqueness check, before ANYTHING mutates (not
    // even the pre-mint pass below) -- see validate_field_key_uniqueness()'s
    // own doc comment for why this can't be folded into the per-object
    // create/update loop the way validate() (Ref<> integrity) is.
    if (auto err = validate_field_key_uniqueness(txn, old_field_keys)) return err;

    // Pre-mint pass: every live create gets its real id BEFORE anything is
    // applied, so the remap table is complete when the first remap_refs()
    // runs -- which is what lets creates in one transaction reference each
    // other in ANY order: backward, forward, mutually, or themselves, the
    // same order-independence commit_bulk_without_undo() gets from minting its whole
    // table up front. A local id left unmapped after this pass can only be
    // a cancelled create (remove() of a local id nulls its entry here) or
    // a stray id from some other transaction -- both still rejected as
    // Invalid by apply_create/apply_update. alloc_slot() is rollback-logged,
    // so a rollback reclaims every slot minted here.
    //
    // `pending` (the minted ids) is what validate() accepts as targets in
    // place of an installed slot during the create loop below: a minted id
    // either installs later this same attempt or the whole attempt rolls
    // back, so integrity holds either way.
    std::vector<Id> pending;
    pending.reserve(txn.local_created_.size());
    for (auto& obj : txn.local_created_) {
        if (!obj) continue;
        const std::uint32_t slot = alloc_slot();
        const std::uint32_t i = slot & kChunkMask;
        const std::uint32_t cur_gen =
            (slot >> kChunkBits) < spine_.size() ? spine_[slot >> kChunkBits]->gen[i] : 0;
        const Id id{slot, cur_gen + 1};  // recycled slot gets a fresh generation
        remap[obj->id.index] = id;
        pending.push_back(id);
    }
    // One allocation for the whole vector (via reserve() above) instead of a
    // hash-node malloc per minted id, at the cost of a single sort here --
    // `pending` is fully built and never mutated again below, so this is the
    // one place a sort can happen. See its type's own doc comment (model.h).
    std::sort(pending.begin(), pending.end(), IdLess{});

    // Creates first, then updates (reconciled immediately, see apply_update),
    // then deletes resolved last -- in that order, so a same-transaction
    // "repoint away from X, then delete X" sees the repoint already reflected
    // in referrers_ before the cascade BFS runs. The first integrity
    // violation aborts the attempt; the caller must then rollback (via
    // classify_apply_failure()).
    std::optional<IntegrityError> err;
    for (auto& obj : txn.local_created_) {
        if (obj && (err = apply_create(std::move(obj), remap, pending, keep_undo)))
            break;  // null: cancelled locally
    }
    if (!err) {
        for (auto& [slot, clone] : txn.local_updated_) {
            // No `pending` needed here: every create installed before the
            // first update applies, so peek() resolves them directly.
            // old_field_keys[slot] was populated for every id in
            // local_updated_ by collect_update_baseline_field_keys() above --
            // see apply_update()'s own old_field_keys_hint doc comment.
            if (!clone) continue;
            const auto hint_it = old_field_keys.find(slot);
            const auto* hint = hint_it != old_field_keys.end() ? &hint_it->second : nullptr;
            if ((err = apply_update(std::move(clone), remap, hint, keep_undo))) break;
        }
    }
    if (!err) {
        // Deferred local removes (see Transaction::remove_raw) and real
        // remove_intents_ are resolved together, in ONE shared BFS call --
        // see Model::remove_raw's own doc comment for why. Each deferred
        // local's create just installed above, so its real id (out of the
        // remap table) takes the exact same cascade BFS a committed id does
        // -- referrers_ already reflects every install and reconcile.
        std::vector<Id> remove_seeds;
        remove_seeds.reserve(txn.local_remove_intents_.size() + txn.remove_intents_.size());
        for (std::uint32_t idx : txn.local_remove_intents_) {
            const auto it = remap.find(kLocalIdBit | idx);
            assert(it != remap.end() && "a deferred-removed create is still live, so it was minted");
            if (it != remap.end()) remove_seeds.push_back(it->second);
        }
        for (Id rid : txn.remove_intents_) remove_seeds.push_back(rid);
        // std::move: remove_raw takes its `work` list by value and uses it
        // AS the BFS frontier directly -- see its own doc comment. Without
        // the move this would copy the vector a second time (having already
        // built it once above) for no reason, right before its only use.
        if (!remove_seeds.empty()) remove_raw(std::move(remove_seeds), keep_undo);
    }
    // Collapse pending_undo_actions_ to at most one action per id -- see
    // collapse_undo_actions()'s own doc comment. Guarded by keep_undo, same
    // as every push into pending_undo_actions_ above (apply_create/apply_update/
    // clone_for_cascade_null/remove_raw): when keep_undo is false the vector
    // never had anything in it to collapse, so this would be a no-op anyway,
    // but gating explicitly keeps that fact grep-able here rather than
    // relying on collapse_undo_actions() noticing the vector is empty.
    if (!err && keep_undo) collapse_undo_actions();
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

void Model::collapse_undo_actions() {
    if (pending_undo_actions_.size() < 2) return;  // nothing to collapse

    // pending_undo_actions_ is index-parallel to changes_: every push site (apply_
    // create/apply_update/clone_for_cascade_null/remove_raw) adds exactly
    // one of each, same id, same order -- see those functions' own
    // pending_undo_actions_.push_back calls, each immediately alongside a
    // changes_.push_back for the same id. changes_ itself is read here but
    // never modified -- see Change's own doc comment for why an id showing
    // up twice there is fine (a deliberate audit trail), unlike here.
    assert(pending_undo_actions_.size() == changes_.size() &&
          "pending_undo_actions_ must be index-parallel with changes_ when non-empty");

    std::unordered_map<Id, std::size_t, IdHash> slot_of;  // Id -> its index in `out`
    std::vector<UndoAction> out;
    std::vector<bool> cancelled;
    out.reserve(pending_undo_actions_.size());
    cancelled.reserve(pending_undo_actions_.size());

    for (std::size_t i = 0; i < changes_.size(); ++i) {
        const Id id = changes_[i].id;
        auto it = slot_of.find(id);
        if (it == slot_of.end()) {
            slot_of.emplace(id, out.size());
            out.push_back(std::move(pending_undo_actions_[i]));
            cancelled.push_back(false);
            continue;
        }
        const std::size_t slot = it->second;
        switch (changes_[i].kind) {
            case ChangeKind::Created:
                break;  // structurally never anything but the first entry for an id
            case ChangeKind::Updated:
                // A second (or later) Updated for this id -- e.g. two
                // cascade-nulls of different fields on the same referrer, or
                // a cascade-null of an object also create()'d this same
                // transaction. out[slot] already holds the FIRST action
                // pushed for this id (a Remove if it originated as a
                // create() this transaction, else a RestoreUpdate holding
                // the TRUE pre-transaction baseline) -- keep it as-is. A
                // later capture would only reflect this same transaction's
                // own earlier edit to this id, not what existed before the
                // transaction started, so it must not overwrite what's kept.
                break;
            case ChangeKind::Deleted:
                if (out[slot].kind == UndoAction::Kind::Remove) {
                    // This id originated as a create() THIS transaction and
                    // is now being cascade-deleted, also this transaction:
                    // nothing was ever visible to a reader. Cancel the
                    // pairing entirely -- resurrecting it on undo would
                    // bring into existence something that never did.
                    cancelled[slot] = true;
                } else {
                    // out[slot] is a RestoreUpdate holding the true
                    // pre-transaction baseline (from the FIRST edit to this
                    // id this transaction). Promote it to a Recreate of
                    // that SAME baseline -- not pending_undo_actions_[i]'s own
                    // Recreate, which captured the value right before
                    // deletion and so already reflects this transaction's
                    // own prior edits. The id is unchanged from Updated
                    // through to Deleted (a slot keeps its generation until
                    // the delete actually zeros it), so `id` is correct for
                    // the promoted Recreate action too.
                    out[slot].kind = UndoAction::Kind::Recreate;
                    out[slot].id = id;
                }
                break;
        }
    }

    pending_undo_actions_.clear();
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (cancelled[i]) continue;
        pending_undo_actions_.push_back(std::move(out[i]));
    }
}

std::optional<CommitResult> Model::check_and_apply(Transaction& txn,
                                                   std::unordered_map<std::uint32_t, Id>& remap,
                                                   bool keep_undo) {
    if (std::vector<Id> overlap = check_id_overlap(txn); !overlap.empty()) {
        return CommitResult{CommitStatus::Conflict,
                            Snapshot{},
                            {},
                            ConflictInfo{ConflictReason::IdSetOverlap, std::move(overlap)},
                            {},
                            std::nullopt};
    }
    if (auto err = apply_transaction_contents(txn, remap, keep_undo)) {
        return classify_apply_failure(std::move(*err), txn);
    }
    return std::nullopt;  // applied; remap is populated, changes_ may or may not be empty
}

CommitResult Model::publish_now(std::unordered_map<std::uint32_t, Id> remap, std::string undo_name,
                                std::any undo_data, bool keep_undo, std::uint64_t undo_txn_id) {
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

    // Built once and shared (not copied) into every subscriber's Update AND
    // the changelog entry below -- changes_ itself is still read a few more
    // times past this point (last_write_version_, the undo list's `touched`
    // set, and the final CommitResult), so this is one copy out of it, same
    // as before; what's eliminated is the PER-SUBSCRIBER copy the old code
    // made by passing changes_ by value into each Update.
    auto shared_changes = std::make_shared<const std::vector<Change>>(changes_);

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
        for (auto& s : subs) s->push(Update{pub, shared_changes, false});
    }

    // Hand this attempt's retired objects to the background reaper rather
    // than freeing them inline: a large cascade must not stall a commit, and
    // destructors should run off both the committing thread and any reader.
    enqueue_retired(std::move(retired_));
    retired_.clear();

    changelog_.push_back({r->version, shared_changes});
    prune_changelog();

    // Feed check_id_overlap()'s slot-indexed conflict index -- see
    // last_write_version_'s own comment. Grown lazily to the highest slot
    // any commit has touched so far; a slot never touched keeps its
    // default (0), which can never exceed any real base_version() (version
    // numbering starts at 1), so an unresized tail slot correctly reads as
    // "never touched" without needing to be pre-sized to spine_'s extent.
    for (const Change& c : changes_) {
        if (c.id.index >= last_write_version_.size()) {
            last_write_version_.resize(c.id.index + 1, 0);
            last_write_id_.resize(c.id.index + 1, Id{});
        }
        last_write_version_[c.id.index] = r->version;
        last_write_id_[c.id.index] = c.id;
    }

    // Undo list: this commit's own touched ids invalidate any existing
    // entry that overlaps them (see undo_list_'s own comment for what
    // "conflicts" means here), whether or not this commit produced undo
    // data of its own; then append this commit's own entry, if it has any
    // (a commit_bulk_without_undo()-style wipe, or an attempt with only
    // no-op-since-cancelled local creates, has none).
    //
    // Skipping this outright when max_undo_list_size_ == 0 is safe -- not
    // just an optimistic guess -- because set_max_undo_list_size(0) clears
    // undo_list_/undo_touch_index_ IMMEDIATELY (see its own doc comment),
    // and nothing can add to either while the cap stays 0 (keep_undo is
    // folded against the cap before apply even runs, in
    // apply_transaction_contents). So "cap is 0" and "undo_list_ is
    // (provably, permanently, until the cap changes) empty" are the same
    // fact here -- there is nothing this block could possibly find to
    // prune or append.
    if (max_undo_list_size_ != 0) {
        // Conflicting entries, via undo_touch_index_ -- O(|changes_|) expected
        // lookups instead of an O(|undo_list_|) scan of every retained
        // entry regardless of whether it could possibly conflict. Walks
        // changes_ directly rather than collecting a deduped `touched` set
        // first -- a duplicate id in changes_ just costs one redundant
        // lookup here, and is deduped anyway by `seen` below (by list-node
        // address), so a set's per-element allocation would buy nothing.
        // Collected into `to_erase` first rather than erased inline, since
        // untrack_undo_entry() below reads an entry's OWN touched list --
        // which may contain ids not in changes_ at all -- to unindex it, and
        // doing that while also iterating changes_ would be iterating one
        // container while mutating another it doesn't own.
        std::vector<std::list<UndoEntry>::iterator> to_erase;
        {
            std::unordered_set<const UndoEntry*> seen;
            for (const Change& c : changes_) {
                auto it = undo_touch_index_.find(c.id);
                if (it == undo_touch_index_.end()) continue;
                if (seen.insert(&*it->second).second) to_erase.push_back(it->second);
            }
        }
        for (auto& it : to_erase) {
            untrack_undo_entry(*it);  // drop every OTHER id this entry indexed under, first
            undo_list_.erase(it);
        }

        // Already inside `max_undo_list_size_ != 0` -- make room first
        // (oldest entries first, same ordering list_undo() promises) so
        // this add never leaves the list over either configured cap.
        if (keep_undo && !pending_undo_actions_.empty()) {
            // UndoEntry::approx_bytes for the entry about to be added --
            // computed once, here, from pending_undo_actions_'s final (collapsed)
            // contents. See that member's own doc comment for exactly what
            // this counts (and doesn't).
            std::size_t new_bytes = 0;
            for (const UndoAction& a : pending_undo_actions_) {
                new_bytes += sizeof(UndoAction) + (a.previous_value ? a.previous_value->byte_size() : 0);
            }

            // Evict oldest-first against WHICHEVER cap the incoming entry
            // would violate -- count, bytes, or both. The loop can only ever
            // empty undo_list_, never refuse to run: the entry being added
            // here is never itself a candidate, so if new_bytes alone
            // exceeds max_undo_bytes_ this still terminates (once
            // undo_list_ is empty) and the add below proceeds anyway,
            // temporarily leaving total_undo_bytes_ over budget -- see
            // set_max_undo_memory_bytes()'s doc comment. An undo entry is
            // never rejected for its own size.
            while (undo_list_.size() >= max_undo_list_size_ ||
                   (!undo_list_.empty() && total_undo_bytes_ + new_bytes > max_undo_bytes_)) {
                untrack_undo_entry(undo_list_.front());
                undo_list_.erase(undo_list_.begin());
            }
            total_undo_bytes_ += new_bytes;

            // Materialized only now that an entry is actually going to be
            // stored -- see UndoEntry::touched's own doc comment for why a
            // plain (possibly-duplicate) vector is enough.
            std::vector<Id> touched;
            touched.reserve(changes_.size());
            for (const Change& c : changes_) touched.push_back(c.id);

            undo_list_.push_back({r->version, std::move(pending_undo_actions_), std::move(touched),
                                  std::move(undo_name), std::move(undo_data), undo_txn_id, new_bytes});
            // Index the just-added entry under every id it touches, so a
            // LATER commit's conflict check (the lookup above) can find it
            // in O(1) instead of scanning for it.
            const auto new_it = std::prev(undo_list_.end());
            for (Id id : new_it->touched) undo_touch_index_[id] = new_it;
        }
    }
    pending_undo_actions_.clear();

    std::vector<Change> resolved = std::move(changes_);

    changes_.clear();
    dirty_.clear();        // next attempt re-COWs each chunk it touches
    dirty_by_type_.clear();
    dirty_by_field_.clear();
    dirty_by_cached_field_.clear();
    dirty_by_cached_reference_.clear();
    txn_rollback_log_.clear();         // committed: nothing to roll back to
    txn_created_.clear();  // published objects are now owned by the spine

    return CommitResult{CommitStatus::Committed, pub,         std::move(resolved), std::nullopt,
                        std::move(remap),        std::nullopt};
}

// ---------------------------------------------------------------------------
// Undo list
// ---------------------------------------------------------------------------

void Model::untrack_undo_entry(const UndoEntry& e) {
    for (Id id : e.touched) undo_touch_index_.erase(id);
    total_undo_bytes_ -= e.approx_bytes;
}

std::vector<Model::UndoSummary> Model::list_undo() const {
    std::lock_guard lk(commit_mu_);
    std::vector<UndoSummary> out;
    out.reserve(undo_list_.size());
    for (const UndoEntry& e : undo_list_)
        out.push_back({e.version, e.actions.size(), e.name, e.data, e.txn_id, e.approx_bytes});
    return out;
}

std::optional<Model::UndoEntry> Model::take_undo(std::uint64_t version) {
    std::lock_guard lk(commit_mu_);
    for (auto it = undo_list_.begin(); it != undo_list_.end(); ++it) {
        if (it->version != version) continue;
        untrack_undo_entry(*it);
        UndoEntry taken = std::move(*it);
        undo_list_.erase(it);
        return taken;
    }
    return std::nullopt;
}

void Model::clear_undo_list() {
    std::lock_guard lk(commit_mu_);
    undo_list_.clear();
    undo_touch_index_.clear();  // every entry it indexed is gone too
    total_undo_bytes_ = 0;
}

CommitResult Model::apply_undo(const UndoEntry& entry) {
    Transaction inv = begin();
    std::unordered_map<Id, Id, IdHash> old_to_new;
    std::vector<Id> local(entry.actions.size());

    // Pass 1: mint every Recreate action's new local id up front (mirrors
    // the pre-mint pass), so pass 2 can remap victim-to-victim edges
    // regardless of which order they appear in `entry.actions`.
    for (std::size_t i = 0; i < entry.actions.size(); ++i) {
        const UndoAction& a = entry.actions[i];
        if (a.kind == UndoAction::Kind::Recreate) {
            local[i] = inv.create_raw(std::unique_ptr<ObjectBase>(a.previous_value->clone()));
            old_to_new[a.id] = local[i];
        }
    }

    const UndoRemapper remapper{old_to_new};
    for (std::size_t i = 0; i < entry.actions.size(); ++i) {
        const UndoAction& a = entry.actions[i];
        switch (a.kind) {
            case UndoAction::Kind::Recreate:
                if (ObjectBase* p = inv.update_raw(local[i])) p->remap_undo_refs(remapper);
                break;
            case UndoAction::Kind::Remove:
                inv.remove_raw(a.id);
                break;
            case UndoAction::Kind::RestoreUpdate: {
                std::unique_ptr<ObjectBase> remapped(a.previous_value->clone());
                remapped->remap_undo_refs(remapper);
                if (ObjectBase* p = inv.update_raw(a.id)) p->assign_from(*remapped);
                break;
            }
        }
    }
    return try_commit(inv);
}

CommitResult Model::commit_pretransaction_locked(Transaction& txn, bool keep_undo) {
    assert(txn.model_ == this && "Transaction belongs to a different Model");

    if (txn.local_created_.empty() && txn.local_updated_.empty() && txn.remove_intents_.empty())
        return CommitResult{CommitStatus::Committed, txn.base_, {}, std::nullopt, {}, std::nullopt};

    std::unordered_map<std::uint32_t, Id> remap;  // local Id::index -> real Id, this attempt only
    if (auto failure = check_and_apply(txn, remap, keep_undo)) return std::move(*failure);

    if (changes_.empty()) {
        // Everything in txn had already been applied by an earlier
        // try_commit() on this same Transaction (or every local create was
        // locally cancelled) -- a no-op success, not a fresh publish.
        return CommitResult{
            CommitStatus::Committed, snapshot(), {}, std::nullopt, {}, std::nullopt};
    }

    return publish_now(std::move(remap), txn.name(), txn.data(), keep_undo, txn.id());
}

CommitResult Model::run_pre_transaction_core(Transaction& txn, bool keep_undo) {
    // Crash, don't return a bad CommitResult: this function does no locking of its own -- it
    // goes straight into commit_pretransaction_locked(), which requires commit_mu_ ALREADY
    // held. Called from anywhere outside the one window try_commit()
    // guarantees the lock is held, it would mutate commit_mu_-protected
    // state (spine_, referrers_, ...) with no synchronization at all --
    // corruption, not a recoverable error. See the doc comment in model.h.
    assert(in_pre_transactions_phase_ &&
           "run_pre_transaction()/run_pre_transaction_without_undo() called outside a running "
           "PreTransactionsFn callback");
    assert(!precommit_failed_ &&
           "run_pre_transaction()/run_pre_transaction_without_undo() called again after an "
           "earlier pre-transaction this same attempt already failed -- check the return value "
           "and stop");
    assert(txn.model_ == this && "Transaction belongs to a different Model");

    CommitResult result = commit_pretransaction_locked(txn, keep_undo);
    if (result.status != CommitStatus::Committed) {
        precommit_failed_ = true;
        precommit_failure_ = std::make_unique<CommitResult>(result);
    }
    return result;
}

CommitResult Model::run_pre_transaction(Transaction& txn) {
    return run_pre_transaction_core(txn, /*keep_undo=*/true);
}

CommitResult Model::run_pre_transaction_without_undo(Transaction& txn) {
    return run_pre_transaction_core(txn, /*keep_undo=*/false);
}

CommitResult Model::commit_main_locked(Transaction& txn, bool keep_undo) {
    assert(txn.model_ == this && "Transaction belongs to a different Model");

    if (pre_transactions_) {
        assert(!in_pre_transactions_phase_ && "pre_transactions_ invoked reentrantly");
        precommit_failed_ = false;
        precommit_failure_.reset();
        in_pre_transactions_phase_ = true;
        pre_transactions_(*this, txn);
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
    // Inlined rather than routed through commit_pretransaction_locked(): the main
    // transaction, unlike a pre-transaction (run_pre_transaction()),
    // must still go through the pre_commit_ veto hook below.
    std::unordered_map<std::uint32_t, Id> remap;
    if (auto failure = check_and_apply(txn, remap, keep_undo)) return std::move(*failure);

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
    if (pre_commit_ && !pre_commit_(*this, txn, changes_)) {
        rollback_apply();
        return CommitResult{CommitStatus::Vetoed, Snapshot{}, {}, std::nullopt, {}, std::nullopt};
    }

    return publish_now(std::move(remap), txn.name(), txn.data(), keep_undo, txn.id());
}

// One counter per CommitStatus (see the Diagnostics::commits_* doc
// comment); noexcept because it's called from try_commit_core()'s hot
// path and must never be a reason a commit attempt could throw under
// -fno-exceptions.
void Model::record_commit_outcome(CommitStatus status) noexcept {
    switch (status) {
        case CommitStatus::Committed: commits_succeeded_.fetch_add(1, std::memory_order_relaxed); break;
        case CommitStatus::Conflict: commits_conflicted_.fetch_add(1, std::memory_order_relaxed); break;
        case CommitStatus::Vetoed: commits_vetoed_.fetch_add(1, std::memory_order_relaxed); break;
        case CommitStatus::Invalid: commits_invalid_.fetch_add(1, std::memory_order_relaxed); break;
        case CommitStatus::PrecommitConflict:
            commits_precommit_conflicted_.fetch_add(1, std::memory_order_relaxed);
            break;
    }
}

CommitResult Model::try_commit_core(Transaction& txn, bool keep_undo) {
    assert(txn.model_ == this && "Transaction belongs to a different Model");

    if (txn.local_created_.empty() && txn.local_updated_.empty() && txn.remove_intents_.empty()) {
        record_commit_outcome(CommitStatus::Committed);
        return CommitResult{CommitStatus::Committed, txn.base_, {}, std::nullopt, {}, std::nullopt};
    }

    // post_commit_copy: post_commit_ read out (a std::function copy, cheap)
    // while commit_mu_ is still held by the lambda below -- that's the only
    // point at which reading the commit_mu_-protected member is safe. It is
    // then invoked below, AFTER the lambda's lock_guard has gone out of
    // scope and released the lock -- see PostCommitFn for why that matters.
    PostCommitFn post_commit_copy;
    CommitResult result = [&] {
        std::lock_guard commit_lk(commit_mu_);
        post_commit_copy = post_commit_;
        return commit_main_locked(txn, keep_undo);
    }();
    record_commit_outcome(result.status);

    if (post_commit_copy) post_commit_copy(*this, txn, result);
    return result;
}

CommitResult Model::try_commit(Transaction& txn) {
    return try_commit_core(txn, /*keep_undo=*/true);
}

CommitResult Model::try_commit_without_undo(Transaction& txn) {
    return try_commit_core(txn, /*keep_undo=*/false);
}

// ---------------------------------------------------------------------------
// begin_bulk() / commit_bulk_without_undo() -- see the section comment on the
// declarations in model.h for the exclusive-access precondition. Everything
// below assumes it holds; there is no way to check it from in here beyond
// the two live_-emptiness asserts.
// ---------------------------------------------------------------------------

BulkTransaction Model::begin_bulk() {
    return BulkTransaction(this);
}

// set_slot() clones via cow() (once per attempt per chunk) and logs the
// EXACT prior (obj, gen) pair so a rollback can restore whatever occupant
// this write is displacing -- including a live one, if the slot was already
// in use. Here, that prior-occupant case cannot happen: commit_bulk_without_undo()'s wipe
// (see its own comment) has already emptied every chunk this attempt could
// possibly touch, via cow()'s own lazy spine growth, so `ch->obj[i]`/
// `ch->gen[i]` are guaranteed to be null/0 before this write -- there is no
// occupant to capture, and (since nothing here is ever undone -- see the
// declaration's doc comment) nothing to restore it for even if there were.
void Model::set_slot_no_log(std::uint32_t slot, const ObjectBase* obj, std::uint32_t gen) {
    const std::uint32_t c = slot >> kChunkBits, i = slot & kChunkMask;
    Chunk* ch = cow(c);  // clones the chunk on first touch this attempt; see cow()'s own comment
    ch->obj[i] = obj;
    ch->gen[i] = gen;
}

// Same referrers_ push as add_out_refs(), minus the pop_back/erase-if-empty
// undo closure: with the whole index freshly cleared by the wipe, this can
// only ever be the first edge for its (from, field) pair -- there is no
// existing bucket state a mistaken push here could corrupt, and (as above)
// nothing this attempt can fail into that would need it undone.
void Model::add_out_refs_no_log(const ObjectBase* o) {
    const Id from = o->id;
    o->each_ref([&](const void* field, const char*, Id target, bool nullable) {
        if (!target) return;
        referrers_[target.index].push_back(RefEdge{from, field, nullable});
    });
}

// Same as add_field_keys(), minus capturing/logging the per-field map's
// prior state. Reads the outer map through a reference (`auto&`, not
// add_field_keys()'s `auto prev = ...` copy) since there is no "prev" to
// hold onto -- one fewer PersistentMap handle copy per field per object.
// Collisions are still the caller's problem, not detected here -- UNLIKE the
// ordinary Transaction path (see validate_field_key_uniqueness()), which
// now rejects a duplicate key as CommitStatus::Invalid instead of
// overwriting it. commit_bulk_without_undo() has no undo log to unwind a
// rejected batch against, so extending the same check here would mean a
// second full validation pass over the whole batch upfront, same idea as
// Pass 1's ref-integrity check just above -- not done today, so a bulk load
// that violates define_keys()'s uniqueness contract still silently
// overwrites the earlier entry.
void Model::add_field_keys_no_log(const ObjectBase* o) {
    const Id id = o->id;
    o->each_field_key([&](const void* field, std::string key) {
        // try_emplace, not operator[]: see add_field_keys's identical seeding
        // for why (no log_by_field_once ordering hazard on this bulk-load
        // path, but still needs to seed the pool on first touch).
        auto& sub =
            by_field_.try_emplace(field, pmap::PersistentMap<std::string, Id, pmap::StringHash>(&node_pool_))
                .first->second;
        sub = sub.set(key, id);
    });
}

// Same two-level (outer map keyed by field, inner bucket keyed by value)
// structure as add_cached_fields(), minus logging the outer map's prior
// state. `sub` is a reference into by_cached_field_, not a copy, for the
// same reason as add_field_keys_no_log() above.
void Model::add_cached_fields_no_log(const ObjectBase* o) {
    const Id id = o->id;
    o->each_cached_field([&](const void* field, std::string key) {
        // try_emplace + pool-seeded bucket: see add_cached_fields's identical
        // seeding (no log_by_cached_field_once ordering hazard on this
        // bulk-load path, but still needs to seed the pool on first touch).
        auto& sub = by_cached_field_
                       .try_emplace(field, pmap::PersistentMap<std::string, pmap::PersistentSet<Id, IdHash>,
                                                               pmap::StringHash>(&node_pool_))
                       .first->second;
        const pmap::PersistentSet<Id, IdHash>* bucket = sub.get(key);
        sub = sub.set(key, (bucket ? *bucket : pmap::PersistentSet<Id, IdHash>(&node_pool_)).insert(id));
    });
}

// Same reverse-multimap structure as add_cached_references() (outer map
// keyed by field, inner bucket keyed by the TARGET's Id, storing the
// referrer), minus logging. Same `auto&`-not-copy reasoning as the two
// helpers above.
void Model::add_cached_references_no_log(const ObjectBase* o) {
    const Id id = o->id;
    o->each_cached_reference([&](const void* field, const char*, Id target, bool) {
        if (!target) return;
        // try_emplace + pool-seeded bucket: see add_cached_references's
        // identical seeding (no log_by_cached_reference_once ordering
        // hazard on this bulk-load path, but still needs to seed the pool
        // on first touch).
        auto& sub =
            by_cached_reference_
                .try_emplace(field, pmap::PersistentMap<Id, pmap::PersistentSet<Id, IdHash>, IdHash>(
                                        &node_pool_))
                .first->second;
        const pmap::PersistentSet<Id, IdHash>* bucket = sub.get(target);
        sub = sub.set(target, (bucket ? *bucket : pmap::PersistentSet<Id, IdHash>(&node_pool_)).insert(id));
    });
}

CommitResult Model::commit_bulk_without_undo(BulkTransaction& txn) {
    assert(txn.model_ == this && "BulkTransaction belongs to a different Model");

    // Checkpoint 1/2: the exclusivity precondition, checked (not enforced --
    // see the doc comment on the declaration) before anything is touched.
    {
        std::lock_guard lk(ver_mu_);
        assert(live_.empty() &&
               "commit_bulk_without_undo() requires exclusive access -- see its declaration's doc comment");
    }

    std::lock_guard commit_lk(commit_mu_);

    // Pass 1: validate the WHOLE batch before mutating anything. There is no
    // undo log here, so every check the ordinary path defers to per-object
    // validate() (interleaved with installation) has to happen upfront,
    // against the batch as a whole: after the wipe below, a non-null ref can
    // only ever resolve to another object in THIS SAME BATCH, by local index.
    for (const auto& obj : txn.objects_) {
        bool bad = false;
        obj->each_ref([&](const void*, const char*, Id target, bool nullable) {
            if (bad) return;
            if (!target) {
                if (!nullable) bad = true;
                return;
            }
            if (!is_local(target) || (target.index & ~kLocalIdBit) >= txn.objects_.size())
                bad = true;
        });
        if (bad) {
            // Nothing mutated yet -- txn still owns every object untouched.
            return CommitResult{
                CommitStatus::Invalid, Snapshot{}, {},
                std::nullopt,          {},         IntegrityError{kUnmappedLocalMsg, Id{}}};
        }
    }

    // Checkpoint 2/2, immediately before the wipe: same assert, same caveat.
    {
        std::lock_guard lk(ver_mu_);
        assert(live_.empty() &&
               "commit_bulk_without_undo() requires exclusive access -- see its declaration's doc comment");
    }

    // Pass 2: wipe. No live Snapshot exists (asserted above), so every
    // currently-installed object can be freed directly -- the retire-and-
    // wait-for-the-reaper protocol exists for readers, and by precondition
    // there are none. Same idiom as ~Model(), which is in exactly the same
    // position (nothing left to observe the spine).
    for (auto& ch : spine_)
        for (std::uint32_t i = 0; i < kChunkSize; ++i) delete ch->obj[i];  // delete(nullptr): no-op
    spine_.clear();
    dirty_.clear();
    dirty_by_type_.clear();
    dirty_by_field_.clear();
    dirty_by_cached_field_.clear();
    dirty_by_cached_reference_.clear();
    by_type_.clear();
    by_field_.clear();
    by_cached_field_.clear();
    by_cached_reference_.clear();
    referrers_.clear();
    free_slots_.clear();
    next_slot_ = 0;
    exhausted_slots_ = 0;
    changelog_.clear();
    last_write_version_.clear();
    last_write_id_.clear();
    undo_list_.clear();
    undo_touch_index_.clear();  // every entry it indexed is gone too
    total_undo_bytes_ = 0;

    // Pass 3: install. next_slot_ is 0 and free_slots_ is empty (just
    // cleared), so real ids are just 0..N-1 in order -- what alloc_slot()
    // would compute anyway, minus its own (otherwise pointless here) undo
    // logging. Minting the whole remap table before installing anything
    // means remap_refs() below always sees every target's real id already,
    // regardless of which local index refers to which -- order doesn't
    // matter the way it does for apply_create()'s incremental remap.
    // Keyed by the FULL local Id::index (kLocalIdBit set), matching what
    // RefRemapper::translate() looks up -- not the bare 0-based position.
    std::unordered_map<std::uint32_t, Id> remap;
    remap.reserve(txn.objects_.size());
    for (std::size_t i = 0; i < txn.objects_.size(); ++i)
        remap[kLocalIdBit | static_cast<std::uint32_t>(i)] =
            Id{next_slot_++, 1};

    for (std::size_t i = 0; i < txn.objects_.size(); ++i) {
        ObjectBase* raw = txn.objects_[i].release();
        bool unmapped = false;
        raw->remap_refs(RefRemapper{remap, &unmapped});
        assert(!unmapped && "pass 1 already validated every ref resolves within the batch");

        const Id id = remap[kLocalIdBit | static_cast<std::uint32_t>(i)];
        raw->id = id;

        set_slot_no_log(id.index, raw, id.gen);
        // try_emplace, not operator[]: see apply_create's identical seeding
        // for why (this path has no log_by_type_once ordering hazard --
        // commit_bulk_without_undo never logs -- but still needs to seed
        // the pool on first touch rather than default-constructing unpooled).
        auto& sub =
            by_type_.try_emplace(raw->tag(), pmap::PersistentSet<Id, IdHash>(&node_pool_)).first->second;
        sub = sub.insert(id);
        add_out_refs_no_log(raw);
        add_field_keys_no_log(raw);
        add_cached_fields_no_log(raw);
        add_cached_references_no_log(raw);

        changes_.push_back({id, ChangeKind::Created, raw->tag()});
    }
    txn.objects_.clear();  // every unique_ptr already release()'d above

    return publish_now(std::move(remap));
}

}  // namespace model
