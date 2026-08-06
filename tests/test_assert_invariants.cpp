// Every `assert` in include/model/model.h and src/model.cpp that guards a
// misuse an EXTERNAL caller can actually trigger (as opposed to an internal
// consistency check between two writer-private structures, which no amount
// of public-API misuse can reach) gets a test here that deliberately
// triggers it and checks the process dies of that assert -- via
// CHECK_ASSERT_FAILURE/dies_of_assert (test_harness.h), the same fork()-and-
// check-SIGABRT technique performance_tests.cpp uses for out-of-process
// measurement.
//
// The point of this file specifically (as opposed to folding each test into
// the file that already covers that feature) is to make every one of these
// misuse-triggers-an-assert regressions discoverable in ONE place, so a
// future change to model.h/model.cpp that accidentally weakens or deletes
// one of these asserts shows up as a test FAILURE here, not as a silently
// passing "well it didn't crash" run. Add a new TEST() here whenever a new
// assert is added anywhere in include/model/ or src/model.cpp -- see this
// file's own coverage note at the bottom for which asserts DON'T have a
// test here, and why.

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "model/model.h"
#include "test_harness.h"
#include "test_helpers.h"

using namespace model;

// ---------------------------------------------------------------------------
// Cross-Model misuse: a Snapshot, Transaction, or BulkTransaction built
// against one Model handed to a DIFFERENT Model's entry point. None of these
// types carry a compile-time link to their owning Model, so every one of
// these is an assert, not a compile error.
// ---------------------------------------------------------------------------

// Model::begin(Snapshot base, ...): unlike Snapshot::begin() (which derives
// the owning Model from the snapshot's own Lease and can't go wrong), this
// overload has no type-level link between `base` and `this` -- see
// Model::begin's own comment in src/model.cpp.
TEST(model_begin_with_snapshot_from_a_different_model_asserts) {
    Model a;
    make_account(a, "A1");
    Model b;
    Snapshot sb = b.snapshot();

    CHECK_ASSERT_FAILURE((void)a.begin(sb, "x", {}));
}

// Model::try_commit(Transaction&): a Transaction built via a.begin() handed
// directly to b.try_commit(). Distinct from the Model::begin case above --
// this one never goes through Model::begin at all, so it exercises
// try_commit_core's OWN txn.model_ == this check, not the Snapshot-pairing
// one.
TEST(try_commit_with_a_transaction_from_a_different_model_asserts) {
    Model a;
    Model b;
    Transaction txn = a.begin();
    auto acc = std::make_unique<Account>();
    acc->name = "X";
    txn.create(std::move(acc));

    CHECK_ASSERT_FAILURE((void)b.try_commit(txn));
}

// BulkTransaction has the identical hazard, on its own separate entry point
// (begin_bulk()/commit_bulk_without_undo()) -- not reachable through any of
// the Transaction-shaped checks above.
TEST(commit_bulk_without_undo_with_a_bulk_transaction_from_a_different_model_asserts) {
    Model a;
    Model b;
    BulkTransaction bt = a.begin_bulk();
    auto acc = std::make_unique<Account>();
    acc->name = "X";
    bt.create(std::move(acc));

    CHECK_ASSERT_FAILURE((void)b.commit_bulk_without_undo(bt));
}

// View(Snapshot,T)'s own doc comment (model.h) admits pairing an object from
// one snapshot with another compiles -- "exactly the version-mixing bug
// views exist to prevent" -- and is covered by an assert
// (find_raw(obj.id) == &obj) rather than left silent.
TEST(view_pairs_object_from_a_different_model_asserts) {
    Model a;
    const Ref<Account> acc = make_account(a, "A1");
    Snapshot sa = a.snapshot();
    const Account* obj = sa.find(acc);
    CHECK(obj != nullptr);

    Model b;  // empty: obj.id cannot resolve to anything in b's spine
    Snapshot sb = b.snapshot();

    CHECK_ASSERT_FAILURE((void)sb.view(*obj));
}

// ---------------------------------------------------------------------------
// PreTransactionsFn / run_pre_transaction() misuse: this API's whole
// contract is "call run_pre_transaction() only from inside a running
// PreTransactionsFn callback, on this same Model, at most once per failure."
// Every clause of that contract is its own assert.
// ---------------------------------------------------------------------------

// run_pre_transaction_without_undo() called with no PreTransactionsFn
// running at all: model.cpp's in_pre_transactions_phase_ check.
TEST(run_pre_transaction_called_outside_a_pre_transactions_hook_asserts) {
    Model m;
    Transaction pre = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "X";
    pre.create(std::move(a));

    CHECK_ASSERT_FAILURE((void)m.run_pre_transaction_without_undo(pre));
}

// run_pre_transaction_without_undo() called (from a running hook) with a
// Transaction built on a DIFFERENT Model -- distinct from try_commit's own
// cross-Model check above: this one guards commit_pretransaction_locked's
// entry, reached only through run_pre_transaction, never through try_commit.
TEST(run_pre_transaction_with_a_transaction_from_a_different_model_asserts) {
    Model a;
    Model b;
    a.set_pre_transactions([&](Model& model, const Transaction&) {
        Transaction foreign = b.begin();  // wrong Model
        auto acc = std::make_unique<Account>();
        acc->name = "X";
        foreign.create(std::move(acc));
        (void)model.run_pre_transaction_without_undo(foreign);
    });

    Transaction txn = a.begin();
    auto acc = std::make_unique<Account>();
    acc->name = "MAIN";
    txn.create(std::move(acc));

    CHECK(dies_of_assert([&] { (void)a.try_commit(txn); }));
    a.set_pre_transactions({});
}

// A second run_pre_transaction_without_undo() call in the SAME attempt after
// an earlier one already failed: model.cpp's precommit_failed_ check. The
// hook's own doc comment already says "check the return value and stop" --
// this is what happens if a hook doesn't.
TEST(run_pre_transaction_called_again_after_an_earlier_failure_in_the_same_attempt_asserts) {
    Model m;
    m.set_pre_transactions([](Model& model, const Transaction&) {
        Transaction bad = model.begin();
        auto o = std::make_unique<Order>();
        o->code = "BAD";  // account left default -> null non-nullable Ref -> Invalid
        bad.create(std::move(o));
        CommitResult r = model.run_pre_transaction_without_undo(bad);
        CHECK(r.status == CommitStatus::Invalid);

        // Misuse: calling again after the failure above, in the same attempt.
        Transaction again = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "SHOULD_NOT_RUN";
        again.create(std::move(a));
        (void)model.run_pre_transaction_without_undo(again);
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "MAIN";
    txn.create(std::move(a));

    CHECK(dies_of_assert([&] { (void)m.try_commit(txn); }));
    m.set_pre_transactions({});
}

// ---------------------------------------------------------------------------
// PreCommitFn / PreTransactionsFn reentering try_commit(): commit_mu_ is
// non-recursive, so a hook calling back into try_commit()/
// try_commit_without_undo() on the same Model, same thread, would deadlock
// (or hit UB) without commit_owner_'s check.
// ---------------------------------------------------------------------------

TEST(pre_commit_hook_calling_try_commit_reentrantly_asserts) {
    Model m;
    m.set_pre_commit([&](Model& model, const Transaction&, const std::vector<Change>&) {
        Transaction inner = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "INNER";
        inner.create(std::move(a));
        (void)model.try_commit(inner);  // reentrant: this thread already holds commit_mu_
        return true;
    });

    CHECK(dies_of_assert([&] { make_account(m, "OUTER"); }));
    m.set_pre_commit({});
}

// ---------------------------------------------------------------------------
// Model lifetime: ~Model() requires every Snapshot/Transaction::base() to
// have already been dropped, and (separately) every wait_for_reclamation()
// caller to have already returned. Neither is enforceable at compile time --
// both are asserted, loudly, rather than left to manifest as a
// Snapshot::Lease dereferencing a freed Model.
// ---------------------------------------------------------------------------

TEST(model_destructor_asserts_if_a_snapshot_is_still_live) {
    CHECK(dies_of_assert([] {
        auto m = std::make_unique<Model>();
        make_account(*m, "A1");
        Snapshot s = m->snapshot();
        m.reset();  // ~Model() while s is still live
        (void)s;
    }));
}

// commit_bulk_without_undo() shares the SAME exclusive-access precondition
// ~Model() has (see its own declaration's doc comment) -- checked twice
// inside (before validation, and again right before the wipe), but hitting
// the first is enough to prove the precondition is actually enforced.
TEST(commit_bulk_without_undo_asserts_if_a_snapshot_is_still_live) {
    Model m;
    make_account(m, "SEED");
    Snapshot s = m.snapshot();  // still alive when commit_bulk_without_undo runs

    BulkTransaction bt = m.begin_bulk();
    auto acc = std::make_unique<Account>();
    acc->name = "X";
    bt.create(std::move(acc));

    CHECK_ASSERT_FAILURE((void)m.commit_bulk_without_undo(bt));
}

// ~Model() waits for reap_waiters_ to reach zero (under reap_mu_) before
// flipping reaper_stop_, rather than asserting the count is already zero --
// see its comment in model.cpp for why an assert-and-proceed there is a
// hang waiting to happen: the reaper is still alive at that point, so every
// already-registered caller's target round genuinely will complete, and
// wait_for_reclamation() wakes this wait as each one returns. This test
// pins that down: kWaiters threads racing m.reset() must complete cleanly
// EVERY attempt, not just usually -- a flake here means the wait/notify
// pairing between ~Model() and wait_for_reclamation() has a gap again.
//
// wait_for_reclamation() keeps its own !reaper_stop_ assert for a narrower
// case this test can't reach: a caller that starts only after ~Model() has
// already finished tearing down. Provoking that deliberately means calling
// a method on an already-dead/freed Model -- not something a test can do
// safely, so it stays unverified by a test and relies on the reasoning in
// that assert's own comment instead.
//
// Unlike a Snapshot, a wait_for_reclamation() caller holds nothing
// registered in live_ until it's actually inside the call, so there's no
// external signal for "every thread has registered." The sleep below after
// every waiter thread signals readiness is a bias, not a guarantee: too
// short (or removed) and an unlucky thread can still land in the OTHER,
// genuinely-unfixable half of the race -- calling wait_for_reclamation() on
// a Model that's already fully torn down -- which crashes for a reason
// unrelated to whatever this test is meant to be checking, and would show
// up as a flaky, confusing failure here rather than a clean signal.
TEST(wait_for_reclamation_callers_in_flight_when_model_is_destroyed_complete_safely) {
    constexpr int kAttempts = 10;
    constexpr int kWaiters = 8;
    for (int attempt = 0; attempt < kAttempts; ++attempt) {
        const bool ok = completes_cleanly([] {
            auto m = std::make_unique<Model>();
            // Waiter threads call through this raw pointer, never through `m`
            // itself: std::unique_ptr's own control state (the pointer it
            // holds) is NOT thread-safe, so reading it via m->... on one
            // thread concurrently with m.reset() writing it on another is a
            // data race in its own right, independent of anything Model does
            // -- exactly what this test must NOT be exercising. raw is set
            // once, before any thread starts, so every thread sees the same
            // valid, unchanging value.
            Model* const raw = m.get();
            std::atomic<int> ready{0};
            std::vector<std::thread> waiters;
            waiters.reserve(kWaiters);
            for (int i = 0; i < kWaiters; ++i)
                waiters.emplace_back([&] {
                    ready.fetch_add(1, std::memory_order_relaxed);
                    (void)raw->wait_for_reclamation();
                });
            while (ready.load(std::memory_order_relaxed) < kWaiters) std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            m.reset();  // races the waiters above -- must now resolve safely, not hang or assert
            for (auto& t : waiters) t.join();
        });
        CHECK(ok);
    }
}

// ---------------------------------------------------------------------------
// Ref<T>(Id)/Opt<T>(Id): the documented "unchecked" escape hatch (see Ref's
// own comment) -- "the type claim is only verified when the ref is
// resolved... against a Snapshot, whose tag check catches a wrong T then."
// Snapshot::resolve(Ref<T>) is that check.
// ---------------------------------------------------------------------------

TEST(resolve_asserts_when_ref_type_does_not_match_the_resolved_object) {
    Model m;
    const Ref<Account> acc = make_account(m, "A1");
    Snapshot s = m.snapshot();
    const Ref<Order> wrong_type(acc.id());  // same id, wrong static type

    CHECK_ASSERT_FAILURE((void)s.resolve(wrong_type));
}

TEST(resolve_asserts_on_a_ref_that_does_not_resolve_at_all) {
    Model m;
    Snapshot s = m.snapshot();
    const Ref<Account> bogus(Id{999999, 1});  // never allocated

    CHECK_ASSERT_FAILURE((void)s.resolve(bogus));
}

// ---------------------------------------------------------------------------
// Object<Derived> per-type declaration hygiene: a field declared twice in
// define_fields()/define_references() (same or different LookupType) is an
// authoring bug this project wants impossible to compile clean AND silently
// tolerate at runtime -- see Object<Derived>::validate_field_declarations/
// validate_ref_declarations's own comment. Checked once per TYPE (a
// function-local static), the first time any instance's each_field()/
// each_ref() runs -- i.e. the first time one is ever created/committed --
// so these two throwaway types must never be touched outside a
// dies_of_assert child, or the static's one-time check would already have
// run (and passed or failed) in the parent process before the intended test.
// ---------------------------------------------------------------------------

namespace {
class DupCacheField final : public model::Object<DupCacheField> {
public:
    std::int64_t x = 0;
    template <class Self>
    static void define_fields(Self& s, const model::LookupFieldReader& v) {
        v.field<&DupCacheField::x>(s.x, model::LookupType::Cache, "x");
        v.field<&DupCacheField::x>(s.x, model::LookupType::Scan, "x_again");  // same field, twice
    }
};

class DupRefField final : public model::Object<DupRefField> {
public:
    model::Ref<Account> a;
    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&DupRefField::a>(), s.a, model::LookupType::Cache, "a");
        v(model::field_tag<&DupRefField::a>(), s.a, model::LookupType::Scan, "a_again");  // twice
    }
};
}  // namespace

TEST(creating_a_type_with_a_field_declared_twice_in_define_fields_asserts) {
    Model m;
    CHECK(dies_of_assert([&] {
        Transaction txn = m.begin();
        auto o = std::make_unique<DupCacheField>();
        o->x = 5;
        txn.create(std::move(o));
        (void)m.try_commit(txn);
    }));
}

TEST(creating_a_type_with_a_field_declared_twice_in_define_references_asserts) {
    Model m;
    const Ref<Account> acc = make_account(m, "A1");
    CHECK(dies_of_assert([&] {
        Transaction txn = m.begin();
        auto o = std::make_unique<DupRefField>();
        o->a = acc;
        txn.create(std::move(o));
        (void)m.try_commit(txn);
    }));
}

// ---------------------------------------------------------------------------
// Object<Derived>::assign_from: "other must be the same concrete type as
// this" (its own doc comment) -- unlike every OTHER checked downcast in
// model.h (Snapshot::cast<T>, peek_as<T>, ...), didn't used to be
// re-asserted here. A type mismatch through this path is a static_cast to
// the wrong dynamic type followed by a read/write through it: real UB, not
// just wrong data (Model::take_undo is the one real caller; only a bug in
// its id-remapping table could ever mismatch it in practice).
// ---------------------------------------------------------------------------

TEST(assign_from_rejects_a_type_mismatch) {
    Account a;
    Order o;
    CHECK_ASSERT_FAILURE(a.assign_from(o));
}

// ---------------------------------------------------------------------------
// Coverage note: asserts in include/model/model.h, src/model.cpp, and
// include/model/persistent_map.h that do NOT have a test here, and why:
//
// - Every `static_assert` (model.h: to_field_key's V-type check,
//   FieldKeyReader::key/LookupFieldReader::field's V-type check, Object<
//   Derived>'s std::is_final_v check, the two std::is_base_of_v checks) is a
//   COMPILE-time check. This project's test harness is runtime-only (no
//   negative-compilation infrastructure, deliberately -- see CLAUDE.md on
//   staying dependency-free); the whole test suite continuing to build IS
//   the regression check for these.
// - model.cpp:669 (slot allocation reaching the reserved local-id bit) --
//   unreachable below ~2^31 objects; not practical to construct in a test.
// - model.cpp:788, 1353, 1999, 2043, 2752 -- internal consistency checks
//   between two writer-private structures populated together earlier in the
//   SAME function (e.g. pending_undo_actions_ vs changes_, a remap table vs
//   its own pre-mint pass). No public-API misuse reaches these; they'd only
//   fire from an actual bug in the function that populates both sides.
// - model.cpp:2429 (commit_main_locked's txn.model_ == this) and 2410
//   (commit_pretransaction_locked's) are real checks, but every public path
//   that could reach them already passes through try_commit_core's (2500)
//   or run_pre_transaction_core's (also checked before calling in) identical
//   check first -- there is no distinct misuse that reaches the deeper copy
//   without also tripping the shallower one, so a separate test couldn't
//   prove anything the tests above don't already cover.
// - model.cpp:2432 ("pre_transactions_ invoked reentrantly") is, as of the
//   commit_owner_ fix above, DEAD CODE: commit_main_locked (which invokes
//   pre_transactions_) has exactly one call site, inside try_commit_core's
//   locked lambda -- and try_commit_core's own commit_owner_ check (2508)
//   now rejects any same-thread reentrant try_commit()/
//   try_commit_without_undo() call BEFORE it could ever reach commit_main_
//   locked a second time. Confirmed by tracing every call site; left as a
//   documented finding rather than removed, since removing pre-existing
//   code wasn't asked for.
// - persistent_map.h:357 (chain_copy's kPerfectHash branch) is also DEAD
//   CODE: every call site that could reach chain_copy while kPerfectHash is
//   true is itself gated behind `if constexpr (!kPerfectHash)` in its own
//   caller, except set_in's "ran out of hash bits" branch (persistent_map.h
//   ~576) -- which is separately unreachable for ANY two distinct 64-bit
//   hash values (13 five-bit slices fully cover 64 bits, so two distinct
//   hashes are always separated by some slice before the trie could run out
//   of bits).
// - persistent_map.h:418 (chain_erase_links) and 431/433 (chain_erase) guard
//   "the caller already verified this key is present" -- true by
//   construction, since erase_in only calls chain_erase after leaf_get()
//   confirms presence via a real per-entry key comparison (not a hash
//   shortcut), so 433's "declared perfect but collided" half can't actually
//   observe a mismatch: if leaf_get found the key, and kPerfectHash means
//   the chain can only ever hold one entry, that entry IS the key. (Contrast
//   persistent_map_tests.cpp's chain_set_asserts_when_a_hash_declared_
//   perfect_actually_collides: set_in has no equivalent leaf_get-style
//   double-check before calling chain_set, which is exactly why THAT assert
//   is reachable and this one isn't.)
// - model.cpp's wait_for_reclamation() !reaper_stop_ assert only has ONE of
//   its two triggers covered by a test: a caller already registered when
//   ~Model() runs -- see wait_for_reclamation_callers_in_flight_when_model_
//   is_destroyed_complete_safely above, which actually exercises ~Model()'s
//   fix for that case (it now waits for such a caller to finish rather than
//   racing a stale check) and expects it to resolve WITHOUT the assert
//   firing. The other trigger -- a call that starts only after ~Model() has
//   already finished tearing down -- has no test and can't safely get one:
//   provoking it means calling a method on an object concurrently with (or
//   after) its own destructor completing, which is calling into a
//   dying/freed Model, not a distinct bug this class could catch.
// - persistent_map.h:886 (iterator depth_ < kMaxDepth) is structurally
//   guaranteed by the write side: set_in's own "ran out of hash bits"
//   handling (see above) means a trie built entirely through set_in()/
//   erase_in() can never exceed kMaxDepth levels in the first place --
//   reaching this assert would require a set_in bug that ALSO breaks that
//   guarantee, not a misuse reachable through any public Hash/key choice.
// ---------------------------------------------------------------------------
