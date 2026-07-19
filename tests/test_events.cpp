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
    CHECK_EQ(u.changes.size(), std::size_t{1});
    CHECK_EQ(u.changes[0].id, a.raw());
    CHECK(u.changes[0].kind == ChangeKind::Created);
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
        for (const Change& c : u.changes) {
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

