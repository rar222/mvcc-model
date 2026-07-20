#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "model/model.h"
#include "test_harness.h"
#include "test_helpers.h"

using namespace model;


// Basic roundtrip: create+commit, then find both by Ref/Id and by a
// define_keys()-declared field, and confirm a typed lookup distinguishes
// two types even when one's key string looks like the other's.
TEST(create_and_find_by_id_and_key) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 100);
    const Ref<Order> o = make_order(m, "O1", a, {}, 7);

    Snapshot s = m.snapshot();
    CHECK(s.find(a) != nullptr);
    CHECK(s.find(o) != nullptr);
    CHECK_EQ(s.find(a)->balance, 100);  // typed: no cast
    CHECK_EQ(s.find(o)->qty, 7);

    CHECK(s.find_by_key<&Account::name>("A1") != nullptr);
    CHECK(s.find_by_key<&Order::computed_key>("ord:O1") != nullptr);
    CHECK(s.find_by_key<&Order::computed_key>("ord:nope") == nullptr);
    CHECK(s.find_by_key<&Account::name>("ord:O1") == nullptr);  // right key, wrong type
    CHECK_EQ(s.find_by_key<&Account::name>("A1")->id, a.raw());
}

// Ref<>/Opt<> resolve correctly through a Snapshot: a non-null Opt
// resolves to the right object, a null Opt (the root order's parent)
// resolves to nullptr -- no special-casing needed by the caller.
TEST(refs_resolve_through_snapshot) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> p = make_order(m, "P", a);
    const Ref<Order> c = make_order(m, "C", a, p);

    Snapshot s = m.snapshot();
    const Order* child = s.find(c);

    const Account& acct = s.resolve(child->account);
    CHECK_EQ(acct.id, a.raw());
    CHECK(s.resolve(child->parent) != nullptr);
    CHECK_EQ(s.resolve(child->parent)->id, p.raw());
    CHECK(s.resolve(s.find(p)->parent) == nullptr);
}

// A Transaction's local edits -- including a same-transaction local-id
// create -- are invisible to any Snapshot taken before commit, and only
// become visible via a FRESH Snapshot taken after commit.
TEST(uncommitted_writes_are_invisible_to_readers) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    Snapshot before = m.snapshot();
    Transaction txn = m.begin();
    make_order(m, "placeholder", a);  // committed separately, just to touch `before`'s absence
    CHECK(before.find_by_key<&Order::computed_key>("ord:placeholder") == nullptr);

    Ref<Order> local = txn.create(std::make_unique<Order>());
    txn.update(local)->code = "O1";
    txn.update(local)->account = a;
    CHECK(txn.exists(local));  // visible in the transaction's own working set
    CHECK(before.find_by_key<&Order::computed_key>("ord:O1") == nullptr);  // not committed yet

    commit_ok(m, txn);
    CHECK(before.find_by_key<&Order::computed_key>("ord:O1") ==
          nullptr);  // old snapshot never changes
    CHECK(m.snapshot().find_by_key<&Order::computed_key>("ord:O1") != nullptr);
}


// Snapshot isolation's headline property: a snapshot taken before an
// update keeps reporting the pre-update value even after a later commit
// changes it and publishes a newer version.
TEST(old_snapshot_sees_old_values) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a, {}, 1);

    Snapshot v1 = m.snapshot();
    update_field(m, o, [](Order* p) { p->qty = 999; });
    Snapshot v2 = m.snapshot();

    CHECK_EQ(v1.find(o)->qty, 1);  // still alive, still old
    CHECK_EQ(v2.find(o)->qty, 999);
    CHECK(v1.version() < v2.version());
}

// Try to break invariant 3 (published state is immutable) by hammering
// the SAME id many times instead of once: if try_commit()'s apply step
// ever mutated an already-published object in place instead of cloning,
// a snapshot pinned before the churn would drift as the churn proceeds --
// this is the test that would catch a stray in-place write fastest, since
// old_snapshot_sees_old_values above only exercises a single update.
TEST(a_pinned_snapshot_never_changes_no_matter_how_much_the_same_object_is_updated_afterward) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a, {}, 0);

    const Snapshot pinned = m.snapshot();
    CHECK_EQ(pinned.find(o)->qty, std::int64_t{0});

    for (int i = 1; i <= 200; ++i) {
        update_field(m, o, [i](Order* p) { p->qty = i; });
        CHECK_EQ(pinned.find(o)->qty, std::int64_t{0});                    // never drifts
        CHECK_EQ(m.snapshot().find(o)->qty, static_cast<std::int64_t>(i));  // latest always current
    }
}

// An older Snapshot can still find and read a since-deleted object --
// the reclamation invariant (nothing freed while a reader can reach it)
// made directly observable; see the in-body comment for the ASan angle.
TEST(deleted_object_survives_in_older_snapshot) {
    // The reclamation invariant: nothing may be freed while a reader can still
    // reach it. Under ASan, getting this wrong is a use-after-free here.
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Snapshot old = m.snapshot();
    remove_and_commit(m, o);

    CHECK(m.snapshot().find(o) == nullptr);  // gone from the new version
    CHECK(old.find(o) != nullptr);           // still there in the old one
    CHECK_EQ(old.find(o)->code, std::string("O1"));
    CHECK(m.reap_backlog() > 0);  // held back from the free list
}

// wait_for_reclamation() reports objects as pinned while a Snapshot holds
// their version, and reports zero once that snapshot is dropped and the
// next commit moves the watermark past it.
TEST(reclamation_advances_when_snapshots_are_dropped) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    {
        Snapshot pin = m.snapshot();
        remove_and_commit(m, o);
        CHECK(m.reap_backlog() > 0);       // pinned by `pin`
        CHECK(m.wait_for_reclamation() > 0);  // still pinned: barrier confirms it can't free
    }
    // `pin` is gone; the next commit's watermark should free the backlog.
    make_order(m, "O2", a);
    CHECK_EQ(m.wait_for_reclamation(), std::size_t{0});
}

// Invariant 5: a stale handle from before a slot was recycled must NOT
// resolve to the new occupant, even though it shares the same slot index
// -- the generation mismatch is what stops the aliasing.
TEST(stale_id_does_not_alias_a_recycled_slot) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);

    remove_and_commit(m, o1);
    const Ref<Order> o2 = make_order(m, "O2", a);

    Snapshot s = m.snapshot();
    CHECK_EQ(o2.raw().index, o1.raw().index);  // slot was recycled
    CHECK(o2.raw().gen != o1.raw().gen);       // but the generation moved
    CHECK(s.find(o1) == nullptr);              // the stale handle must NOT resolve to O2
    CHECK(s.find(o2) != nullptr);
}

// Regression test: see the in-body comment for the exact bug shape --
// Transaction::local_updated_ is keyed by bare slot index, so peek_impl/
// update_impl must also check the FULL Id before trusting a hit.
TEST(transaction_local_update_does_not_alias_a_stale_generation_of_the_same_slot) {
    // Regression: Transaction::local_updated_ is keyed by bare slot index (a
    // transaction only ever clones ONE generation of a given slot -- the one
    // alive at base()). peek_impl/update_impl must verify the FULL Id
    // (index + generation), not just the index, before trusting that map --
    // otherwise a transaction that updates slot K, then separately asks
    // about a stale, dead generation of the SAME slot K (e.g. a handle from
    // an aging cache that was never pruned), gets a false "exists" answer
    // aliased to the WRONG object -- exactly the kind of dangling-looking
    // Ref this whole model exists to prevent.
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const std::uint32_t slot = o1.raw().index;

    remove_and_commit(m, o1);
    const Ref<Order> o2 =
        make_order(m, "O2", a);  // reuses `a` -- nothing else steals the freed slot
    CHECK_EQ(o2.raw().index, slot);
    CHECK(o2.raw().gen != o1.raw().gen);

    Transaction txn = m.begin();
    txn.update(o2)->qty = 42;  // populates local_updated_[slot], keyed by o2's generation

    // A stale handle for the SAME slot, a DIFFERENT (dead) generation, must
    // still be reported as dead -- not aliased to o2's live local clone.
    CHECK(txn.update(o1) == nullptr);
    CHECK(!txn.exists(o1));
    CHECK(txn.peek(o1) == nullptr);

    // The live one is unaffected by the check above.
    CHECK(txn.exists(o2));
    CHECK_EQ(txn.peek(o2)->qty, 42);
}


// A slot whose generation is forced to kGenMax is permanently withdrawn
// from free_slots_ rather than handed out again, and exhausted_slots()
// reports it.
TEST(an_exhausted_slot_is_retired_not_reused) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const std::uint32_t slot = o1.raw().index;

    remove_and_commit(m, o1);
    m.debug_set_generation(slot, kGenMax);

    const Ref<Order> o2 = make_order(m, "O2", a);

    CHECK_EQ(m.exhausted_slots(), std::size_t{1});
    CHECK(o2.raw().index != slot);
    CHECK(o2.raw().gen != 0);
    CHECK(m.snapshot().find(o2) != nullptr);
}

// A handle captured before its slot was exhaustion-retired never
// resolves to whatever object a later, freshly allocated DIFFERENT slot
// ends up holding.
TEST(a_stale_handle_never_aliases_across_exhaustion) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const Ref<Order> stale = o1;
    const std::uint32_t slot = o1.raw().index;

    remove_and_commit(m, o1);
    m.debug_set_generation(slot, kGenMax);
    make_order(m, "O2", a);

    Snapshot s = m.snapshot();
    CHECK(s.find(stale) == nullptr);
}


// debug_live_versions()'s doc comment: "(version, refcount) for every
// currently-live snapshot registration (readers' Snapshots AND every open
// Transaction::base())." Two Snapshots taken at the SAME version (no commit
// in between) must combine into ONE entry with refcount 2 -- not two
// separate entries -- and a Transaction's base() pinning a DIFFERENT version
// is its own, separate entry. Dropping one of the two same-version Snapshots
// decrements that entry's count without touching the other's.
TEST(debug_live_versions_reports_every_pinned_snapshot_and_transaction_base) {
    Model m;
    make_account(m, "A1");
    const std::uint64_t v1 = m.current_version();

    Snapshot s1 = m.snapshot();
    Snapshot s2 = m.snapshot();  // same version as s1 -- combines into one entry, refcount 2

    {
        const auto versions = m.debug_live_versions();
        int count_at_v1 = 0;
        for (const auto& [ver, cnt] : versions) {
            if (ver == v1) count_at_v1 = cnt;
        }
        CHECK_EQ(count_at_v1, 2);
    }

    make_account(m, "A2");
    const std::uint64_t v2 = m.current_version();
    CHECK(v2 > v1);

    Transaction txn = m.begin();  // pins a DIFFERENT version, v2, its own entry
    CHECK_EQ(txn.base_version(), v2);

    {
        const auto versions = m.debug_live_versions();
        CHECK_EQ(versions.size(), std::size_t{2});
        // The minimum key is the reclamation watermark -- v1 (still pinned
        // twice) sorts first, v2 (txn's base) second.
        CHECK_EQ(versions[0].first, v1);
        CHECK_EQ(versions[0].second, 2);
        CHECK_EQ(versions[1].first, v2);
        CHECK_EQ(versions[1].second, 1);
    }

    // Dropping ONE of the two same-version Snapshots decrements v1's count
    // to 1 -- v2's entry (txn's base) is untouched.
    { Snapshot drop = std::move(s2); }

    {
        const auto versions = m.debug_live_versions();
        CHECK_EQ(versions.size(), std::size_t{2});
        int count_at_v1 = 0, count_at_v2 = 0;
        for (const auto& [ver, cnt] : versions) {
            if (ver == v1) count_at_v1 = cnt;
            if (ver == v2) count_at_v2 = cnt;
        }
        CHECK_EQ(count_at_v1, 1);
        CHECK_EQ(count_at_v2, 1);
    }
}

// alloc_slot() only ever runs under commit_mu_ (invariant 7), so the real
// question under concurrency isn't "can two threads pop the same free slot"
// (the lock rules that out by construction) -- it's whether that lock
// discipline actually holds up: several writer threads hammering try_commit()
// concurrently, racing each other to pop a pool of slots this test has
// deliberately poisoned to kGenMax (same testing seam as
// an_exhausted_slot_is_retired_not_reused above), must retire EXACTLY the
// poisoned slots -- no fewer (a lost slot), no more (a double-issued Id),
// and the bookkeeping (slots_allocated/slots_free/slots_exhausted/
// live_object_count) must reconcile exactly afterward.
TEST(concurrent_alloc_slot_under_exhaustion_never_double_hands_out_a_withdrawn_slot) {
    Model m;

    // Build up a pool of freed slots, then poison every one of them.
    constexpr int kPoisoned = 40;
    std::vector<std::uint32_t> poisoned_slots;
    {
        std::vector<Ref<Account>> temp;
        for (int i = 0; i < kPoisoned; ++i) temp.push_back(make_account(m, "TEMP" + std::to_string(i)));
        for (const Ref<Account>& r : temp) poisoned_slots.push_back(r.raw().index);
        for (const Ref<Account>& r : temp) remove_and_commit(m, r);
    }
    CHECK_EQ(poisoned_slots.size(), static_cast<std::size_t>(kPoisoned));
    for (std::uint32_t slot : poisoned_slots) m.debug_set_generation(slot, kGenMax);
    CHECK_EQ(m.diagnostics().slots_free, static_cast<std::size_t>(kPoisoned));

    // Several writer threads doing plain, unrelated creates (no removes) --
    // conflict-free by construction (check_id_overlap() never looks at
    // creates), so every commit here is expected to succeed. With
    // free_slots_ a LIFO stack holding exactly the kPoisoned poisoned slots
    // and nothing else, and total creates well over kPoisoned, every single
    // poisoned slot is guaranteed to get popped, found exhausted, and
    // permanently retired -- deterministically, regardless of thread
    // interleaving, since alloc_slot() is fully serialized behind commit_mu_.
    constexpr int kThreads = 4;
    constexpr int kCreatesPerThread = 20;  // kThreads * kCreatesPerThread == 80 > kPoisoned
    std::atomic<int> committed{0};

    auto writer = [&](int thread_idx) {
        for (int i = 0; i < kCreatesPerThread; ++i) {
            Transaction txn = m.begin();
            auto a = std::make_unique<Account>();
            a->name = "W" + std::to_string(thread_idx) + "_" + std::to_string(i);
            txn.create(std::move(a));
            const CommitResult res = m.try_commit(txn);
            if (res.status == CommitStatus::Committed) committed.fetch_add(1);
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < kThreads; ++t) pool.emplace_back(writer, t);
    for (auto& th : pool) th.join();

    CHECK_EQ(committed.load(), kThreads * kCreatesPerThread);

    // Every poisoned slot was popped exactly once and retired -- none left
    // in circulation, none silently lost or double-counted.
    CHECK_EQ(m.exhausted_slots(), static_cast<std::size_t>(kPoisoned));

    const Model::Diagnostics d = m.diagnostics();
    CHECK_EQ(d.slots_exhausted, m.exhausted_slots());
    CHECK_EQ(d.slots_free, std::size_t{0});  // no removes happened during the concurrent phase
    CHECK_EQ(d.live_object_count, static_cast<std::size_t>(kThreads * kCreatesPerThread));
    // The core no-lost/no-double-issued-slot invariant: every ever-allocated
    // slot index is in exactly one of these three buckets.
    CHECK_EQ(d.slots_allocated, d.live_object_count + d.slots_free + d.slots_exhausted);

    // No two objects were ever handed the same Id: a final scan sees exactly
    // as many distinct Accounts as commits succeeded.
    Snapshot final_s = m.snapshot();
    std::size_t seen = 0;
    final_s.for_each<Account>([&](const Account&) { ++seen; });
    CHECK_EQ(seen, static_cast<std::size_t>(kThreads * kCreatesPerThread));
}

// wait_for_reclamation() as a barrier: it reports work still pinned
// while a Snapshot is held, and reports fully drained once that Snapshot
// is dropped.
TEST(reaper_frees_eventually_after_snapshots_drop) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Snapshot pin = m.snapshot();
    remove_and_commit(m, o);
    CHECK(m.wait_for_reclamation() > 0);  // pinned by `pin`, cannot free yet

    {
        Snapshot drop = std::move(pin);
    }
    CHECK_EQ(m.wait_for_reclamation(), std::size_t{0});
}

// Try to break invariant 4's OTHER clause: it names "a reader (or an open
// Transaction's base())" in the same breath, but every reclamation test
// above only pins with a Snapshot. base() is just an ordinary Snapshot
// under the hood (see Transaction::base()'s own doc comment), so this
// proves that pinning actually holds for an open, never-committed
// Transaction too -- through sustained, UNRELATED churn from other
// transactions, not just one op. Getting this wrong is a use-after-free
// only ASan catches, exactly like the Snapshot case above.
TEST(an_open_transactions_base_pins_reclamation_across_heavy_unrelated_churn) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Transaction txn = m.begin();  // pins base() at the current version; never committed

    // Heavy, UNRELATED churn from other transactions while txn stays open --
    // recycles slots and generations txn's base() has nothing to do with.
    for (int i = 0; i < 50; ++i) {
        const Ref<Account> throwaway = make_account(m, "T" + std::to_string(i));
        remove_and_commit(m, throwaway);
    }
    remove_and_commit(m, o);  // finally remove what txn's base() can actually see

    CHECK(m.wait_for_reclamation() > 0);   // pinned by txn.base(), cannot free yet
    CHECK(txn.base().find(o) != nullptr);  // still resolvable through the open transaction's base
    CHECK_EQ(txn.base().find(o)->code, std::string("O1"));

    {
        Transaction drop = std::move(txn);
    }
    CHECK_EQ(m.wait_for_reclamation(), std::size_t{0});
}

// A single remove() cascading through 500 objects completes and is
// fully reclaimable -- cascade fan-out isn't hard-capped by size, and
// the reaper can clear the whole backlog afterward.
TEST(a_large_cascade_does_not_block_forever_and_reclaims) {
    Model m;
    const Ref<Account> a = make_account(m, "hub");
    {
        // Scoped: a Transaction pins its base() Snapshot for its entire
        // lifetime, exactly like any reader's snapshot. Left unscoped, it
        // would still be pinning the pre-cascade version below, and
        // wait_for_reclamation() would (correctly) never reach zero.
        Transaction txn = m.begin();
        for (int i = 0; i < 500; ++i) {
            auto o = std::make_unique<Order>();
            o->code = "O" + std::to_string(i);
            o->account = a;
            txn.create(std::move(o));
        }
        commit_ok(m, txn);
    }
    CHECK_EQ(m.snapshot().size(), std::size_t{501});

    const std::size_t killed = remove_and_commit(m, a);  // takes the account + all 500 orders
    CHECK_EQ(killed, std::size_t{501});
    CHECK_EQ(m.snapshot().size(), std::size_t{0});

    CHECK_EQ(m.wait_for_reclamation(), std::size_t{0});
}

