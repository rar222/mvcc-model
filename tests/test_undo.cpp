// Stage 1 of the undo-list design (see the plan): proves the RECONSTRUCTION
// primitives -- Transaction::create_raw/update_raw/remove_raw, ObjectBase::
// remap_undo_refs/assign_from, UndoRemapper -- actually work, especially for
// the hard case (a multi-object cascade, with victim-to-victim edges and a
// nulled survivor) BEFORE Stage 2 wires up automatic capture into Model
// itself. `TestUndoAction`/`apply_test_undo` below are a deliberately
// temporary, test-local preview of Stage 2's real Model::UndoAction/
// Model::apply_undo -- same shape, same two-pass algorithm -- so this file's
// tests double as a spec for what Stage 2 has to reproduce automatically.

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "model/model.h"
#include "test_harness.h"
#include "test_helpers.h"

using namespace model;

namespace {

struct TestUndoAction {
    enum class Kind { Recreate, Remove, RestoreUpdate };
    Kind kind;
    Id id;
    std::unique_ptr<ObjectBase> snapshot;  // pre-image clone (Recreate, RestoreUpdate); null for Remove
};

/// Preview of Stage 2's Model::apply_undo: mints every Recreate action's new
/// local id first (mirrors the pre-mint pass), so victim-to-victim edges
/// remap correctly regardless of visitation order, then applies every
/// action against the fresh Transaction. Clones each action's snapshot
/// again rather than consuming it, so the caller's `actions` survives a
/// failed/conflicting attempt intact, exactly like the real apply_undo will.
CommitResult apply_test_undo(Model& m, const std::vector<TestUndoAction>& actions) {
    Transaction inv = m.begin();
    std::unordered_map<Id, Id, IdHash> old_to_new;
    std::vector<Id> local(actions.size());

    for (std::size_t i = 0; i < actions.size(); ++i)
        if (actions[i].kind == TestUndoAction::Kind::Recreate) {
            local[i] = inv.create_raw(std::unique_ptr<ObjectBase>(actions[i].snapshot->clone()));
            old_to_new[actions[i].id] = local[i];
        }

    const UndoRemapper remapper{old_to_new};
    for (std::size_t i = 0; i < actions.size(); ++i) {
        const TestUndoAction& a = actions[i];
        switch (a.kind) {
            case TestUndoAction::Kind::Recreate:
                if (ObjectBase* p = inv.update_raw(local[i])) p->remap_undo_refs(remapper);
                break;
            case TestUndoAction::Kind::Remove:
                inv.remove_raw(a.id);
                break;
            case TestUndoAction::Kind::RestoreUpdate: {
                std::unique_ptr<ObjectBase> remapped(a.snapshot->clone());
                remapped->remap_undo_refs(remapper);
                if (ObjectBase* p = inv.update_raw(a.id)) p->assign_from(*remapped);
                break;
            }
        }
    }
    return m.try_commit(inv);
}

}  // namespace

// A Remove action (the inverse of a create) deletes the target -- no
// snapshot data needed at all.
TEST(undo_remove_action_deletes_the_target) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    std::vector<TestUndoAction> actions;
    actions.push_back({TestUndoAction::Kind::Remove, a.raw(), nullptr});
    const CommitResult res = apply_test_undo(m, actions);

    CHECK(res.status == CommitStatus::Committed);
    CHECK(m.snapshot().find(a) == nullptr);
}

// A RestoreUpdate action (the inverse of a plain update) restores the
// captured pre-image's field values onto the SAME still-real id, generically
// -- via assign_from(), never naming Account by type in apply_test_undo.
TEST(undo_restore_update_action_restores_old_field_values) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 100);
    std::unique_ptr<ObjectBase> pre_image(m.snapshot().find(a)->clone());

    update_field(m, a, [](Account* p) { p->balance = 999; });
    CHECK_EQ(m.snapshot().find(a)->balance, std::int64_t{999});

    std::vector<TestUndoAction> actions;
    actions.push_back({TestUndoAction::Kind::RestoreUpdate, a.raw(), std::move(pre_image)});
    const CommitResult res = apply_test_undo(m, actions);

    CHECK(res.status == CommitStatus::Committed);
    CHECK_EQ(m.snapshot().find(a)->balance, std::int64_t{100});
}

// A Recreate action (the inverse of a plain remove, no cascade) resurrects
// the object with the SAME field values but a genuinely NEW id -- generation
// never comes back (invariant 5), so this can never be the exact same Id.
TEST(undo_recreate_action_resurrects_with_a_new_id) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 42);
    std::unique_ptr<ObjectBase> pre_image(m.snapshot().find(a)->clone());

    remove_and_commit(m, a);
    CHECK(m.snapshot().find(a) == nullptr);

    std::vector<TestUndoAction> actions;
    actions.push_back({TestUndoAction::Kind::Recreate, a.raw(), std::move(pre_image)});
    const CommitResult res = apply_test_undo(m, actions);

    CHECK(res.status == CommitStatus::Committed);
    const Account* revived = m.snapshot().find_by_key<&Account::name>("A1");
    CHECK(revived != nullptr);
    CHECK_EQ(revived->balance, std::int64_t{42});
    CHECK(revived->id != a.raw());  // a new id, not the old (dead) one
}

// The hard case, and the one that actually answers "would cascaded items
// get correct new ids, reconnecting objects that way?": removing `hub`
// (an Account) cascades to kill mid1 and mid2 (both hold a non-nullable
// Ref<Account> to hub) and nulls survivor's Opt<Order> parent (which
// pointed at mid1, one of the victims -- a victim-to-victim AND a
// victim-to-survivor edge in the same cascade). Undoing it must resurrect
// hub/mid1/mid2 with new ids, correctly re-link mid2's parent to mid1's
// NEW id (not its old, dead one), and restore survivor's parent to that
// same new id -- never the stale one its captured pre-image actually holds.
TEST(undo_reconnects_a_multi_object_cascade_including_victim_to_victim_edges) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");     // survivor's account -- untouched throughout
    const Ref<Account> hub = make_account(m, "HUB");   // will be removed

    const Ref<Order> mid1 = make_order(m, "MID1", hub);
    const Ref<Order> mid2 = make_order(m, "MID2", hub, mid1);  // parent: victim-to-victim edge
    const Ref<Order> surv = make_order(m, "SURV", a, mid1);    // parent: victim-to-survivor edge

    // Capture pre-images BEFORE the cascade -- exactly what Stage 2's
    // automatic capture (at apply_update/remove_raw's existing baseline/
    // victim pointers) will do for us later.
    std::vector<TestUndoAction> actions;
    {
        Snapshot pre = m.snapshot();
        actions.push_back(
            {TestUndoAction::Kind::RestoreUpdate, surv.raw(),
             std::unique_ptr<ObjectBase>(pre.find(surv)->clone())});
        actions.push_back(
            {TestUndoAction::Kind::Recreate, mid2.raw(),
             std::unique_ptr<ObjectBase>(pre.find(mid2)->clone())});
        actions.push_back(
            {TestUndoAction::Kind::Recreate, mid1.raw(),
             std::unique_ptr<ObjectBase>(pre.find(mid1)->clone())});
        actions.push_back(
            {TestUndoAction::Kind::Recreate, hub.raw(),
             std::unique_ptr<ObjectBase>(pre.find(hub)->clone())});
    }

    const std::size_t killed = remove_and_commit(m, hub);
    CHECK_EQ(killed, std::size_t{3});  // hub, mid1, mid2

    Snapshot mid_state = m.snapshot();
    CHECK(mid_state.find(surv) != nullptr);
    CHECK(!mid_state.find(surv)->parent);  // nulled, not dangling

    const CommitResult undo_res = apply_test_undo(m, actions);
    CHECK(undo_res.status == CommitStatus::Committed);

    Snapshot post = m.snapshot();
    const Order* new_mid1 = post.find_by_key<&Order::computed_key>("ord:MID1");
    const Order* new_mid2 = post.find_by_key<&Order::computed_key>("ord:MID2");
    const Account* new_hub = post.find_by_key<&Account::name>("HUB");
    CHECK(new_mid1 != nullptr);
    CHECK(new_mid2 != nullptr);
    CHECK(new_hub != nullptr);
    CHECK(new_mid1->id != mid1.raw());  // genuinely new ids throughout
    CHECK(new_mid2->id != mid2.raw());
    CHECK(new_hub->id != hub.raw());

    CHECK(new_mid1->account == Ref<Account>(new_hub->id));  // remapped to the NEW hub
    CHECK(new_mid2->account == Ref<Account>(new_hub->id));
    CHECK(new_mid2->parent == Opt<Order>(Ref<Order>(new_mid1->id)));  // victim-to-victim, remapped

    // The survivor's restored parent must point at mid1's NEW id -- its
    // captured pre-image literally holds the OLD (now permanently dead) id;
    // reconnection only works if RestoreUpdate's snapshot is ALSO remapped,
    // not just Recreate's.
    CHECK(post.find(surv)->parent == Opt<Order>(Ref<Order>(new_mid1->id)));
}

// ---------------------------------------------------------------------------
// Stage 2: Model's OWN automatic capture (list_undo/take_undo/clear_undo_list/
// apply_undo), not the hand-built TestUndoAction preview above.
// ---------------------------------------------------------------------------

// A pure create's own undo entry is a single Remove action -- no snapshot
// data needed at all.
TEST(undo_list_captures_a_pure_create_and_apply_undo_removes_it) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{1});

    auto entry = m.take_undo(summaries.front().version);
    CHECK(entry.has_value());
    CHECK(m.list_undo().empty());  // taken -- no longer listed

    const CommitResult undo_res = m.apply_undo(*entry);
    CHECK(undo_res.status == CommitStatus::Committed);
    CHECK(m.snapshot().find(a) == nullptr);
}

// A pure update's own undo entry is a single RestoreUpdate action, captured
// at apply_update's existing `baseline` read.
TEST(undo_list_captures_a_pure_update_and_apply_undo_restores_the_old_value) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 100);
    m.clear_undo_list();  // only interested in the update's own entry below

    update_field(m, a, [](Account* p) { p->balance = 999; });
    CHECK_EQ(m.snapshot().find(a)->balance, std::int64_t{999});

    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{1});
    auto entry = m.take_undo(summaries.front().version);
    CHECK(entry.has_value());

    const CommitResult undo_res = m.apply_undo(*entry);
    CHECK(undo_res.status == CommitStatus::Committed);
    CHECK_EQ(m.snapshot().find(a)->balance, std::int64_t{100});
}

// A pure remove's own undo entry is a single Recreate action, captured at
// remove_raw's existing `victim` read -- resurrection gets a NEW id.
TEST(undo_list_captures_a_pure_remove_and_apply_undo_resurrects_with_a_new_id) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 42);
    m.clear_undo_list();

    remove_and_commit(m, a);
    CHECK(m.snapshot().find(a) == nullptr);

    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{1});
    auto entry = m.take_undo(summaries.front().version);
    CHECK(entry.has_value());

    const CommitResult undo_res = m.apply_undo(*entry);
    CHECK(undo_res.status == CommitStatus::Committed);
    const Account* revived = m.snapshot().find_by_key<&Account::name>("A1");
    CHECK(revived != nullptr);
    CHECK_EQ(revived->balance, std::int64_t{42});
    CHECK(revived->id != a.raw());
}

// The same multi-object cascade as undo_reconnects_a_multi_object_cascade_
// including_victim_to_victim_edges above, but captured AUTOMATICALLY by
// Model itself rather than hand-built -- proves the real capture sites
// (apply_update/clone_for_cascade_null/remove_raw) produce exactly the
// action set Stage 1 already verified reconnects correctly.
TEST(undo_list_captures_a_real_cascade_and_apply_undo_reconnects_it) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Account> hub = make_account(m, "HUB");
    const Ref<Order> mid1 = make_order(m, "MID1", hub);
    make_order(m, "MID2", hub, mid1);
    const Ref<Order> surv = make_order(m, "SURV", a, mid1);
    m.clear_undo_list();  // only interested in the removal's own entry below

    const std::size_t killed = remove_and_commit(m, hub);
    CHECK_EQ(killed, std::size_t{3});

    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{1});
    // hub, mid1, mid2 (Recreate) + surv (RestoreUpdate, its parent nulled).
    CHECK_EQ(summaries.front().action_count, std::size_t{4});

    auto entry = m.take_undo(summaries.front().version);
    CHECK(entry.has_value());
    const CommitResult undo_res = m.apply_undo(*entry);
    CHECK(undo_res.status == CommitStatus::Committed);

    Snapshot post = m.snapshot();
    const Order* new_mid1 = post.find_by_key<&Order::computed_key>("ord:MID1");
    const Order* new_mid2 = post.find_by_key<&Order::computed_key>("ord:MID2");
    const Account* new_hub = post.find_by_key<&Account::name>("HUB");
    CHECK(new_mid1 != nullptr);
    CHECK(new_mid2 != nullptr);
    CHECK(new_hub != nullptr);
    CHECK(new_mid2->account == Ref<Account>(new_hub->id));
    CHECK(new_mid2->parent == Opt<Order>(Ref<Order>(new_mid1->id)));
    CHECK(post.find(surv)->parent == Opt<Order>(Ref<Order>(new_mid1->id)));
}

// Try to break the conflict-invalidation rule: a later commit touching an
// id an EARLIER undo entry depends on must drop that entry from the list --
// otherwise take_undo()/apply_undo() could silently apply a stale
// RestoreUpdate over a legitimate, newer change.
TEST(a_later_conflicting_commit_invalidates_an_earlier_undo_entry) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 100);
    const auto after_create = m.list_undo();
    CHECK_EQ(after_create.size(), std::size_t{1});
    const std::uint64_t version_a = after_create.front().version;

    update_field(m, a, [](Account* p) { p->balance = 999; });  // touches `a` again

    const auto after_update = m.list_undo();
    for (const auto& s : after_update) CHECK(s.version != version_a);  // the create's entry is gone
    CHECK(!m.take_undo(version_a).has_value());
}

// clear_undo_list() drops every entry -- an explicit, caller-driven reclaim
// (undo_list_ is otherwise unbounded; see its own comment).
TEST(clear_undo_list_empties_the_list) {
    Model m;
    make_account(m, "A1");
    make_account(m, "A2");
    CHECK_EQ(m.list_undo().size(), std::size_t{2});

    m.clear_undo_list();
    CHECK(m.list_undo().empty());
}

// apply_undo is not a special-cased path: it just builds a Transaction and
// calls try_commit(), so a stale entry -- one whose reconstruction would now
// dangle -- reports Invalid exactly like any other build-time integrity
// violation, publishes nothing, and never consumes the caller's entry (it
// clones internally), so it's still usable afterward.
TEST(apply_undo_reports_invalid_instead_of_resurrecting_a_dangling_reference) {
    Model m;
    const Ref<Account> owner = make_account(m, "OWNER");
    const Ref<Order> o = make_order(m, "O1", owner);
    m.clear_undo_list();

    remove_and_commit(m, o);  // captures o's Recreate action, referencing `owner`
    auto entry = m.take_undo(m.list_undo().front().version);
    CHECK(entry.has_value());

    remove_and_commit(m, owner);  // owner is now ALSO gone

    const CommitResult undo_res = m.apply_undo(*entry);
    CHECK(undo_res.status == CommitStatus::Invalid);  // resurrecting o would dangle
    CHECK(m.snapshot().find_by_key<&Order::computed_key>("ord:O1") == nullptr);  // nothing published
    CHECK_EQ(entry->actions.size(), std::size_t{1});  // the caller's copy survives, untouched
}

// UndoEntry/UndoSummary copy the committing Transaction's name()/data() --
// both through list_undo() (the copy-only path) and take_undo() (the
// move-out path).
TEST(undo_entry_and_summary_copy_the_committing_transactions_name_and_data) {
    Model m;
    Transaction txn = m.begin("seed accounts", std::any(std::string("batch-7")));
    auto a = std::make_unique<Account>();
    a->name = "A1";
    txn.create(std::move(a));
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);

    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{1});
    CHECK_EQ(summaries.front().name, std::string("seed accounts"));
    CHECK(summaries.front().data.has_value());
    CHECK_EQ(std::any_cast<std::string>(summaries.front().data), std::string("batch-7"));

    auto entry = m.take_undo(summaries.front().version);
    CHECK(entry.has_value());
    CHECK_EQ(entry->name, std::string("seed accounts"));
    CHECK(entry->data.has_value());
    CHECK_EQ(std::any_cast<std::string>(entry->data), std::string("batch-7"));
}

// A plain, unlabeled begin() (the common case) produces an UndoEntry/
// UndoSummary with empty name and no data -- the feature is opt-in, not a
// tax paid by every ordinary commit.
TEST(undo_entry_and_summary_default_to_empty_name_and_no_data) {
    Model m;
    make_account(m, "A1");

    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{1});
    CHECK(summaries.front().name.empty());
    CHECK(!summaries.front().data.has_value());
}

// apply_undo() is not special-cased in the capture pipeline: it just calls
// try_commit() on an ordinary Transaction, so its OWN commit gets captured
// exactly like any other -- meaning undoing something produces a fresh undo
// entry of its own (an "undo of an undo" is a redo), and the chain can go
// on indefinitely (create -> undo removes it -> undo-the-undo recreates it
// with a NEW id -> ...).
TEST(applying_an_undo_produces_a_new_undo_entry_of_its_own) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 7);

    auto create_entry = m.take_undo(m.list_undo().front().version);
    CHECK(create_entry.has_value());
    CHECK_EQ(create_entry->actions.size(), std::size_t{1});
    CHECK(create_entry->actions.front().kind == Model::UndoAction::Kind::Remove);

    const CommitResult undo1 = m.apply_undo(*create_entry);
    CHECK(undo1.status == CommitStatus::Committed);
    CHECK(m.snapshot().find(a) == nullptr);  // A1 removed

    // Undoing the create produced its OWN entry -- a Recreate action, the
    // inverse of the Remove that just ran.
    const auto after_undo1 = m.list_undo();
    CHECK_EQ(after_undo1.size(), std::size_t{1});
    auto remove_entry = m.take_undo(after_undo1.front().version);
    CHECK(remove_entry.has_value());
    CHECK_EQ(remove_entry->actions.size(), std::size_t{1});
    CHECK(remove_entry->actions.front().kind == Model::UndoAction::Kind::Recreate);

    // Applying THAT entry (undo-of-the-undo, i.e. redo) resurrects A1 --
    // with a new id, per invariant 5 -- and produces yet another entry.
    const CommitResult undo2 = m.apply_undo(*remove_entry);
    CHECK(undo2.status == CommitStatus::Committed);
    const Account* revived = m.snapshot().find_by_key<&Account::name>("A1");
    CHECK(revived != nullptr);
    CHECK_EQ(revived->balance, std::int64_t{7});
    CHECK(revived->id != a.raw());

    const auto after_undo2 = m.list_undo();
    CHECK_EQ(after_undo2.size(), std::size_t{1});  // the chain keeps going
}

// ---------------------------------------------------------------------------
// try_commit_without_undo(): identical commit behavior, but never adds an
// entry to the undo list.
// ---------------------------------------------------------------------------

TEST(try_commit_without_undo_commits_normally_but_adds_no_undo_entry) {
    Model m;
    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "A1";
    txn.create(std::move(a));
    const CommitResult res = m.try_commit_without_undo(txn);

    CHECK(res.status == CommitStatus::Committed);
    CHECK(m.snapshot().find_by_key<&Account::name>("A1") != nullptr);  // committed normally
    CHECK(m.list_undo().empty());  // but no undo entry for it
}

// Existing undo_list_ entries are still pruned if a try_commit_without_undo()
// commit conflicts with them -- only the ADDITION of a new entry is skipped,
// not the conflict check against what's already there (see publish_now()).
TEST(try_commit_without_undo_still_prunes_conflicting_existing_undo_entries) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 100);
    CHECK_EQ(m.list_undo().size(), std::size_t{1});  // the create's own entry

    Transaction txn = m.begin();
    txn.update(a)->balance = 999;
    const CommitResult res = m.try_commit_without_undo(txn);

    CHECK(res.status == CommitStatus::Committed);
    CHECK_EQ(m.snapshot().find(a)->balance, std::int64_t{999});  // committed normally
    CHECK(m.list_undo().empty());  // the create's entry touched `a` too -- pruned
}

// Covers the Recreate and RestoreUpdate capture sites too (create/update are
// covered by the two tests above): a cascade removal produces neither for a
// try_commit_without_undo() commit, even though the cascade itself (the
// victim's removal and the survivor's field nulled) still happens correctly.
TEST(try_commit_without_undo_produces_no_undo_entry_even_for_a_cascade) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Account> hub = make_account(m, "HUB");
    const Ref<Order> mid1 = make_order(m, "MID1", hub);
    make_order(m, "MID2", hub, mid1);
    const Ref<Order> surv = make_order(m, "SURV", a, mid1);
    m.clear_undo_list();  // only interested in the removal below

    Transaction txn = m.begin();
    txn.remove(hub);
    const CommitResult res = m.try_commit_without_undo(txn);

    CHECK(res.status == CommitStatus::Committed);
    CHECK(m.snapshot().find(hub) == nullptr);
    CHECK(m.snapshot().find(surv) != nullptr);
    CHECK(!m.snapshot().find(surv)->parent);  // cascade-null still happened correctly
    CHECK(m.list_undo().empty());             // but none of it was captured for undo
}

// ---------------------------------------------------------------------------
// run_pre_transaction_without_undo(): same split, applied to a
// pre-transaction instead of the main one.
// ---------------------------------------------------------------------------

// The pre-transaction's own commit produces no undo entry, but the MAIN
// transaction (an ordinary try_commit()) still gets its own -- proving the
// suppression is per-call, not something that leaks across the two.
TEST(run_pre_transaction_without_undo_commits_normally_but_adds_no_undo_entry) {
    Model m;
    m.set_pre_transactions([](Model& model, const Transaction&) {
        Transaction pre = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "PRE";
        pre.create(std::move(a));
        CommitResult r = model.run_pre_transaction_without_undo(pre);
        CHECK(r.status == CommitStatus::Committed);
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "MAIN";
    txn.create(std::move(a));
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    m.set_pre_transactions({});

    CHECK(m.snapshot().find_by_key<&Account::name>("PRE") != nullptr);   // committed normally
    CHECK(m.snapshot().find_by_key<&Account::name>("MAIN") != nullptr);

    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{1});  // only MAIN's entry -- PRE's was suppressed
    auto entry = m.take_undo(summaries.front().version);
    CHECK(entry.has_value());
    CHECK_EQ(entry->actions.size(), std::size_t{1});
    const CommitResult undo_res = m.apply_undo(*entry);
    CHECK(undo_res.status == CommitStatus::Committed);
    CHECK(m.snapshot().find_by_key<&Account::name>("MAIN") == nullptr);  // MAIN undone
    CHECK(m.snapshot().find_by_key<&Account::name>("PRE") != nullptr);   // PRE unaffected -- never listed
}

// Existing undo_list_ entries are still pruned if a run_pre_transaction_
// without_undo() commit conflicts with them -- same "prune regardless,
// append conditionally" rule as try_commit_without_undo().
TEST(run_pre_transaction_without_undo_still_prunes_conflicting_existing_undo_entries) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 100);
    const std::uint64_t create_version = m.list_undo().front().version;  // the create's own entry

    m.set_pre_transactions([&](Model& model, const Transaction&) {
        Transaction pre = model.begin();
        pre.update(a)->balance = 999;
        CommitResult r = model.run_pre_transaction_without_undo(pre);
        CHECK(r.status == CommitStatus::Committed);
    });

    Transaction txn = m.begin();
    auto other = std::make_unique<Account>();
    other->name = "OTHER";
    txn.create(std::move(other));
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    m.set_pre_transactions({});

    CHECK_EQ(m.snapshot().find(a)->balance, std::int64_t{999});  // pre-transaction committed normally

    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{1});  // only the main txn's (OTHER's) entry
    CHECK_EQ(summaries.front().action_count, std::size_t{1});
    for (const auto& s : summaries) CHECK(s.version != create_version);  // a's create entry: pruned
}
