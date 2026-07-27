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


// A subscription receives exactly one Update per commit, carrying that
// commit's own changeset paired with a matching post-commit Snapshot.
TEST(subscriber_gets_a_changeset_per_commit) {
    Model m;
    auto sub = m.subscribe(8);

    const Ref<Account> a = make_account(m, "A1");

    Update u;
    CHECK(sub->try_drain(u));
    CHECK_EQ(u.changes->size(), std::size_t{1});
    CHECK_EQ((*u.changes)[0].id, a.raw());
    CHECK((*u.changes)[0].kind == ChangeKind::Created);
    CHECK(!u.coalesced);
    CHECK(u.snapshot.find(a) != nullptr);
    CHECK(!sub->try_drain(u));
}

// Exceeding a subscriber's queue depth coalesces intermediate updates
// into one entry instead of growing the queue unboundedly -- and the
// coalesced delivery still carries the LATEST value, not a stale one.
TEST(overflow_coalesces_instead_of_growing) {
    Model m;
    auto sub = m.subscribe(/*depth=*/2);

    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a, {}, 1);

    for (int i = 0; i < 20; ++i) {  // never drained -- must not grow unbounded
        update_field(m, o, [&](Order* p) { p->qty = i; });
    }

    int batches = 0;
    bool saw_coalesced = false;
    std::int64_t final_qty = -1;
    Update u;
    while (sub->try_drain(u)) {
        ++batches;
        if (u.coalesced) saw_coalesced = true;
        if (const Order* p = u.snapshot.find(o)) final_qty = p->qty;
    }
    CHECK(batches <= 3);  // bounded, not 22
    CHECK(saw_coalesced);
    CHECK_EQ(final_qty, 19);  // the latest state still arrives
}

// Subscription::collapse's create+delete cancellation: an object created
// and deleted before the subscriber ever drains never appears in the
// delivered changeset at all, not even as a no-op pair.
TEST(create_then_delete_between_drains_cancels_out) {
    Model m;
    auto sub = m.subscribe(/*depth=*/1);

    const Ref<Account> a = make_account(m, "A1");

    Update drain;
    while (sub->try_drain(drain)) {
    }

    const Ref<Order> o = make_order(m, "O1", a);
    remove_and_commit(m, o);
    update_field(m, a, [](Account* p) { p->balance = 5; });

    int mentions_o = 0, mentions_a = 0;
    Update u;
    while (sub->try_drain(u)) {
        for (const Change& c : *u.changes) {
            if (c.id == o.raw()) ++mentions_o;
            if (c.id == a.raw()) ++mentions_a;
        }
    }
    CHECK_EQ(mentions_o, 0);
    CHECK(mentions_a >= 1);
}

// Real cross-thread producer/consumer: every other Subscription test above
// calls push()/try_drain()/collapse() synchronously, in one thread -- never
// the actual wait()/shutdown() wake-up path that's the entire point of the
// blocking API. Here a genuinely slow consumer thread (a deliberate sleep
// per item) forces the writer to overflow a small queue and coalesce under
// real timing pressure, not a manually-triggered one; and shutdown() must
// wake the blocked consumer so it exits instead of hanging forever.
TEST(a_slow_subscriber_thread_wakes_from_wait_after_shutdown_and_sees_coalescing) {
    Model m;
    auto sub = m.subscribe(/*queue_depth=*/2);

    std::atomic<int> updates_seen{0};
    std::atomic<bool> saw_coalesced{false};
    std::thread consumer([&] {
        Update u;
        while (sub->wait(u)) {
            updates_seen.fetch_add(1, std::memory_order_relaxed);
            if (u.coalesced) saw_coalesced = true;
            std::this_thread::sleep_for(
                std::chrono::milliseconds(2));  // deliberately the bottleneck
        }
    });

    constexpr int kCommits = 50;
    for (int i = 0; i < kCommits; ++i) make_account(m, "A" + std::to_string(i));

    m.shutdown();  // must wake the consumer -- wait() drains whatever's left, then returns false
    consumer.join();

    CHECK(updates_seen.load() > 0);
    CHECK(updates_seen.load() <= kCommits);  // never more entries than commits, coalesced or not
    CHECK(saw_coalesced.load());             // the actual proof that overflow/coalescing fired
}

// Two Subscriptions on the same Model are entirely independent queues: a
// slow (never-drained) subscriber overflowing and coalescing must not steal
// or fold away any of the individual deliveries a fast, promptly-draining
// subscriber sees. One subscriber's backpressure is purely local to it.
TEST(multiple_independent_subscribers_receive_independent_coalescing_streams) {
    Model m;
    auto fast = m.subscribe(/*depth=*/8);
    auto slow = m.subscribe(/*depth=*/2);

    constexpr int kCommits = 20;
    for (int i = 0; i < kCommits; ++i) {
        make_account(m, "A" + std::to_string(i));

        // Drained every commit -- never allowed to build up, so it must never
        // coalesce and must see exactly this commit's own one-object changeset.
        Update u;
        CHECK(fast->try_drain(u));
        CHECK(!u.coalesced);
        CHECK_EQ(u.changes->size(), std::size_t{1});
        CHECK(!fast->try_drain(u));  // nothing else queued behind it
    }

    // `slow` was never drained during the loop above (queue depth 2, 20
    // commits) -- it must have overflowed and coalesced, exactly like
    // overflow_coalesces_instead_of_growing, entirely independent of `fast`
    // having drained every single one of the same 20 commits cleanly.
    int batches = 0;
    bool saw_coalesced = false;
    Update u;
    while (slow->try_drain(u)) {
        ++batches;
        if (u.coalesced) saw_coalesced = true;
    }
    CHECK(batches <= 3);   // bounded, not 20
    CHECK(saw_coalesced);  // the slow subscriber's queue actually overflowed
}

// Model::shutdown() copies the current subscriber list and closes exactly
// those (see its implementation) -- it is not a persistent "the model is
// shut down" flag that subscribe() consults. So: (1) calling it with zero
// subscribers must not crash or assert, and (2) a Subscription created
// AFTER a shutdown() call starts open, not closed -- it only becomes closed
// once a LATER shutdown() call sees it in the list. That is exactly why
// Model::subscribe()'s doc comment says "Subscribe BEFORE shutdown()": this
// is a documented footgun, not something the API auto-detects for you.
TEST(subscribing_after_shutdown_returns_a_subscription_that_immediately_reports_closed) {
    Model m;
    m.shutdown();  // zero subscribers -- must not crash or assert

    auto late = m.subscribe(/*queue_depth=*/4);
    m.shutdown();  // shutdown() is repeatable: closes whatever is registered NOW, `late` included

    // closed_ is true and the queue is empty, so wait()'s predicate is
    // already satisfied -- this returns false immediately, no blocking.
    Update u;
    CHECK(!late->wait(u));
    CHECK(!late->try_drain(u));
}

/// Prints one Change, resolved against the Snapshot pair that brackets it --
/// `before` is the PREVIOUS Update's snapshot (null for the very first
/// Update), `after` is this Change's own Update::snapshot. Structural only
/// (CHECK that the object is where this ChangeKind says it should be, plus
/// to_string() of whatever's found) -- the caller still owns any assertions
/// about specific field values, since those are scenario-specific and this
/// is generic over T.
template <class T>
void print_change(const Change& c, const Snapshot& before, const Snapshot& after) {
    const Ref<T> r{c.id};
    switch (c.kind) {
        case ChangeKind::Created: {
            const T* v = after.find(r);
            CHECK(v != nullptr);
            if (v) std::printf("create: after=%s\n", v->to_string().c_str());
            break;
        }
        case ChangeKind::Updated: {
            CHECK(static_cast<bool>(before));  // an Updated always has a prior Update to diff against
            const T* b = before.find(r);
            const T* a = after.find(r);
            CHECK(b != nullptr);
            CHECK(a != nullptr);
            if (b && a) std::printf("update: before=%s after=%s\n", b->to_string().c_str(), a->to_string().c_str());
            break;
        }
        case ChangeKind::Deleted: {
            CHECK(static_cast<bool>(before));  // a Deleted always has a prior Update to diff against
            const T* b = before.find(r);
            CHECK(b != nullptr);
            CHECK(after.find(r) == nullptr);  // gone as of THIS Update's snapshot
            if (b) std::printf("delete: before=%s\n", b->to_string().c_str());
            break;
        }
    }
}

// Update::snapshot is the state AFTER its own commit -- which means the
// PREVIOUS Update's snapshot is exactly the state BEFORE the current one.
// A subscriber that just remembers the last Update it saw can therefore
// print before/after values for every ChangeKind without the model needing
// to carry an old-value field on Change itself: after-only for a Created
// (there is no "before"), before AND after for an Updated, before-only for
// a Deleted (the object is already gone from the current snapshot).
TEST(subscriber_prints_before_after_values_from_consecutive_snapshots) {
    Model m;
    auto sub = m.subscribe(/*depth=*/8);

    const Ref<Account> a = make_account(m, "A1", 100);         // commit 1: Created
    update_field(m, a, [](Account* p) { p->balance = 200; });  // commit 2: Updated
    remove_and_commit(m, a);                                   // commit 3: Deleted

    Snapshot prev;  // null until the first Update is drained
    int seen_created = 0, seen_updated = 0, seen_deleted = 0;

    Update u;
    while (sub->try_drain(u)) {
        for (const Change& c : *u.changes) {
            if (c.tag != type_tag<Account>()) continue;
            const Ref<Account> r{c.id};
            print_change<Account>(c, prev, u.snapshot);

            switch (c.kind) {
                case ChangeKind::Created:
                    CHECK_EQ(u.snapshot.find(r)->balance, std::int64_t{100});
                    ++seen_created;
                    break;
                case ChangeKind::Updated:
                    CHECK_EQ(prev.find(r)->balance, std::int64_t{100});
                    CHECK_EQ(u.snapshot.find(r)->balance, std::int64_t{200});
                    ++seen_updated;
                    break;
                case ChangeKind::Deleted:
                    CHECK_EQ(prev.find(r)->balance, std::int64_t{200});
                    ++seen_deleted;
                    break;
            }
        }
        prev = u.snapshot;
    }

    CHECK_EQ(seen_created, 1);
    CHECK_EQ(seen_updated, 1);
    CHECK_EQ(seen_deleted, 1);
}

