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
#include <atomic>
#include <memory>
#include <string>
#include <thread>
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
    // pre-image clone (Recreate, RestoreUpdate); null for Remove. Named to
    // match the real Model::UndoAction::previous_value -- not `snapshot`,
    // which would collide with the unrelated Snapshot class.
    std::unique_ptr<ObjectBase> previous_value;
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
            local[i] = inv.create_raw(std::unique_ptr<ObjectBase>(actions[i].previous_value->clone()));
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
                std::unique_ptr<ObjectBase> remapped(a.previous_value->clone());
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

// set_max_undo_list_size() bounds undo_list_ going forward: once the cap is
// in effect, a commit that would push the list over it instead drops the
// OLDEST entries first (list_undo()'s own documented ordering) to make
// room, so the list never exceeds the cap after an add.
TEST(set_max_undo_list_size_prunes_oldest_entries_to_make_room) {
    Model m;
    m.set_max_undo_list_size(2);

    const Ref<Account> a1 = make_account(m, "A1");  // each is its own commit, own UndoEntry,
    const Ref<Account> a2 = make_account(m, "A2");  // and none touch a shared id -- so only the
    const Ref<Account> a3 = make_account(m, "A3");  // size cap (not conflict-pruning) is at play
    (void)a1;
    (void)a2;
    (void)a3;

    const auto entries = m.list_undo();
    CHECK_EQ(entries.size(), std::size_t{2});
    // Oldest (A1's) is gone; the two most recent survive, oldest-of-the-
    // survivors first.
    CHECK(entries.front().version < entries.back().version);
}

// n == 0 means "keep no undo history at all" -- commits still succeed
// normally (this is a RETENTION cap, not a way to skip collecting undo data
// mid-apply), they just never gain an entry in undo_list_.
TEST(set_max_undo_list_size_of_zero_never_adds_to_the_list) {
    Model m;
    m.set_max_undo_list_size(0);

    CommitResult r1 = [&] {
        Transaction txn = m.begin();
        auto a = std::make_unique<Account>();
        a->name = "A1";
        txn.create(std::move(a));
        return m.try_commit(txn);
    }();
    CHECK(r1.status == CommitStatus::Committed);  // the commit itself is unaffected
    CHECK(m.list_undo().empty());

    make_account(m, "A2");
    CHECK(m.list_undo().empty());
}

// Lowering the cap does not retroactively prune what's already in the
// list -- set_max_undo_list_size() only takes effect the next time a
// commit would ADD an entry (see its own doc comment for why). At that
// point, the prune-to-make-room loop can drop more than one entry at once
// to get back under a cap that was lowered by more than one step.
TEST(lowering_max_undo_list_size_only_takes_effect_on_the_next_add) {
    Model m;
    make_account(m, "A1");
    make_account(m, "A2");
    make_account(m, "A3");
    CHECK_EQ(m.list_undo().size(), std::size_t{3});

    m.set_max_undo_list_size(1);
    CHECK_EQ(m.list_undo().size(), std::size_t{3});  // not retroactive

    make_account(m, "A4");  // the next add enforces the (now-lower) cap
    const auto entries = m.list_undo();
    CHECK_EQ(entries.size(), std::size_t{1});
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

// ---------------------------------------------------------------------------
// run_pre_transaction(): the WITH-undo counterpart of the _without_undo()
// tests above -- the pre-transaction's own commit gets its own undo entry
// too, independent of the main transaction's.
// ---------------------------------------------------------------------------

// The pre-transaction's own commit (run via run_pre_transaction(), not
// _without_undo()) produces its own undo entry, keyed by its OWN published
// version (not the main transaction's) -- and applying it reverses only the
// pre-transaction's effect, leaving the main transaction's own commit alone.
TEST(run_pre_transaction_captures_an_undo_entry_for_the_pre_transaction) {
    Model m;
    std::uint64_t pre_version = 0;
    m.set_pre_transactions([&](Model& model, const Transaction&) {
        Transaction pre = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "PRE";
        pre.create(std::move(a));
        CommitResult r = model.run_pre_transaction(pre);
        CHECK(r.status == CommitStatus::Committed);
        pre_version = r.snapshot.version();
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "MAIN";
    txn.create(std::move(a));
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    m.set_pre_transactions({});

    CHECK(m.snapshot().find_by_key<&Account::name>("PRE") != nullptr);
    CHECK(m.snapshot().find_by_key<&Account::name>("MAIN") != nullptr);

    // Two independent entries: the pre-transaction's own, and the main
    // transaction's own -- nothing about run_pre_transaction() folds its
    // capture into the main commit's entry.
    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{2});

    auto pre_entry = m.take_undo(pre_version);
    CHECK(pre_entry.has_value());
    CHECK_EQ(pre_entry->actions.size(), std::size_t{1});
    CHECK(pre_entry->actions.front().kind == Model::UndoAction::Kind::Remove);

    const CommitResult undo_res = m.apply_undo(*pre_entry);
    CHECK(undo_res.status == CommitStatus::Committed);
    // Only PRE's effect is reversed -- MAIN, committed separately by a
    // different call, is untouched by undoing PRE.
    CHECK(m.snapshot().find_by_key<&Account::name>("PRE") == nullptr);
    CHECK(m.snapshot().find_by_key<&Account::name>("MAIN") != nullptr);
}

// apply_undo() is not special-cased against concurrency: it is just
// begin()+try_commit() (see its own doc comment), so a commit that lands
// between its begin() (which fixes the reconstruction's base) and its own
// check_id_overlap is exactly as real a race as any two writer threads
// hammering try_commit() -- it is reported as an ordinary Conflict, not
// silently absorbed or misclassified as Invalid. Simulated deterministically
// via the pre-transactions hook: PreTransactionsFn runs at the very start of
// ANY try_commit() attempt (including the one apply_undo() makes internally)
// -- from apply_undo()'s reconstruction's perspective, a commit landing there
// is indistinguishable from a different writer thread racing it (see
// main_transaction_conflicts_with_a_pre_transaction_touching_the_same_object
// in test_hooks.cpp, which establishes the same equivalence for an ordinary
// transaction).
TEST(apply_undo_can_itself_report_conflict_against_a_concurrent_commit) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 100);
    m.clear_undo_list();  // only interested in the update's own entry below

    update_field(m, a, [](Account* p) { p->balance = 999; });
    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{1});
    auto entry = m.take_undo(summaries.front().version);
    CHECK(entry.has_value());

    // A "concurrent" commit that lands after apply_undo()'s begin() (fixing
    // its base) but before its own conflict check -- touching the SAME
    // object the taken entry's RestoreUpdate action targets.
    m.set_pre_transactions([a](Model& model, const Transaction&) {
        Transaction concurrent = model.begin();
        concurrent.update(a)->balance = 555;
        CommitResult r = model.run_pre_transaction_without_undo(concurrent);
        CHECK(r.status == CommitStatus::Committed);
    });

    const CommitResult undo_res = m.apply_undo(*entry);
    CHECK(undo_res.status == CommitStatus::Conflict);
    CHECK(undo_res.conflict.has_value());
    CHECK(undo_res.conflict->reason == ConflictReason::IdSetOverlap);

    m.set_pre_transactions({});
    // The "concurrent" commit's value stands -- the undo lost the race, and
    // (being a Conflict, not Committed) never touched published state.
    CHECK_EQ(m.snapshot().find(a)->balance, std::int64_t{555});
}

// take_undo() of a version that never had an undo entry at all -- distinct
// from the already-tested "pruned" case (a_later_conflicting_commit_
// invalidates_an_earlier_undo_entry above): this version is real (committed
// via try_commit_without_undo(), so it genuinely exists as a snapshot
// version) but simply never produced undo data to begin with, and an
// entirely made-up version number behaves the same way.
TEST(take_undo_of_an_unknown_version_returns_nullopt) {
    Model m;
    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "A1";
    txn.create(std::move(a));
    const CommitResult res = m.try_commit_without_undo(txn);
    CHECK(res.status == CommitStatus::Committed);

    CHECK(!m.take_undo(res.snapshot.version()).has_value());  // real version, no undo entry
    CHECK(!m.take_undo(std::uint64_t{999999}).has_value());   // never a real version at all
}

// take_undo() removes an entry from the live list (per its own doc comment),
// but the UndoEntry it hands back is the caller's alone from that point on --
// a later clear_undo_list() call (which only ever touches undo_list_ itself)
// must not reach into an entry that already left the list.
TEST(clear_undo_list_does_not_affect_an_already_taken_entrys_usability) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{1});
    auto entry = m.take_undo(summaries.front().version);
    CHECK(entry.has_value());
    CHECK(m.list_undo().empty());  // already gone from the live list

    m.clear_undo_list();  // a no-op on the (already empty) live list here

    const CommitResult undo_res = m.apply_undo(*entry);
    CHECK(undo_res.status == CommitStatus::Committed);
    CHECK(m.snapshot().find(a) == nullptr);
}

// Pruning is keyed on actual touched-id overlap (publish_now()'s own rule),
// not "any later commit invalidates everything": a commit that creates a
// brand-new, entirely unrelated object shares no id with an earlier entry
// and must leave it untouched.
TEST(undo_entry_survives_an_unrelated_commit_that_touches_no_shared_id) {
    Model m;
    const Ref<Account> x = make_account(m, "X", 10);
    const auto summaries_x = m.list_undo();
    CHECK_EQ(summaries_x.size(), std::size_t{1});
    const std::uint64_t version_x = summaries_x.front().version;

    make_account(m, "Y");  // unrelated -- touches only Y's (brand-new) id

    const auto after = m.list_undo();
    CHECK_EQ(after.size(), std::size_t{2});  // both X's and Y's entries survive
    bool found_x = false;
    for (const auto& s : after)
        if (s.version == version_x) found_x = true;
    CHECK(found_x);

    auto entry = m.take_undo(version_x);
    CHECK(entry.has_value());
    const CommitResult undo_res = m.apply_undo(*entry);
    CHECK(undo_res.status == CommitStatus::Committed);
    CHECK(m.snapshot().find(x) == nullptr);
    CHECK(m.snapshot().find_by_key<&Account::name>("Y") != nullptr);  // Y, untouched throughout
}

// ---------------------------------------------------------------------------
// Multi-threaded undo: the undo list is commit_mu_-protected exactly like
// changelog_/referrers_ (invariant 7), so every guarantee below already
// follows from that -- these tests make it directly observable instead of
// just trusting the reasoning. Following this suite's own established
// discipline (see test_concurrency_stress.cpp): no CHECK() from inside a
// thread body, ever -- results are collected into atomics or per-thread-
// owned slots, and every CHECK happens in the main thread after join().
// ---------------------------------------------------------------------------

// A commit's undo entry is just data once take_undo() hands it to a caller
// -- nothing ties it to the thread that produced it. One thread commits and
// takes the entry; a completely different thread applies it.
TEST(a_different_thread_can_apply_another_threads_undo_entry) {
    Model m;
    std::optional<Model::UndoEntry> entry;
    CommitStatus write_status{};

    std::thread writer([&] {
        Transaction txn = m.begin();
        auto a = std::make_unique<Account>();
        a->name = "FROM_WRITER";
        txn.create(std::move(a));
        const CommitResult res = m.try_commit(txn);
        write_status = res.status;
        if (res.status == CommitStatus::Committed) {
            auto taken = m.take_undo(res.snapshot.version());
            if (taken.has_value()) entry = std::move(taken);
        }
    });
    writer.join();
    CHECK(write_status == CommitStatus::Committed);
    CHECK(entry.has_value());

    CommitStatus undo_status{};
    std::thread undoer([&] {
        const CommitResult undo_res = m.apply_undo(*entry);
        undo_status = undo_res.status;
    });
    undoer.join();
    CHECK(undo_status == CommitStatus::Committed);
    CHECK(m.snapshot().find_by_key<&Account::name>("FROM_WRITER") == nullptr);
}

// Several threads hammer ONE shared Account (so their commits genuinely
// conflict and overlap) while each ALSO owns one private Account nobody
// else ever touches. publish_now()'s pruning must be exact: every earlier
// entry touching the shared Account gets invalidated by whichever commit
// touches it next, leaving exactly one survivor for it, while every
// private entry -- never overlapped by anything -- survives untouched.
TEST(concurrent_conflicting_commits_prune_exactly_the_undo_entries_they_invalidate) {
    Model m;
    const Ref<Account> shared = make_account(m, "SHARED", 0);
    m.clear_undo_list();  // only interested in what happens below

    constexpr int kThreads = 6;
    constexpr int kRoundsPerThread = 40;
    std::atomic<int> private_creates{0};
    std::atomic<int> shared_commits{0};

    std::vector<std::thread> pool;
    for (int t = 0; t < kThreads; ++t) {
        pool.emplace_back([&, t] {
            Transaction own_txn = m.begin();
            auto o = std::make_unique<Account>();
            o->name = "PRIVATE" + std::to_string(t);
            own_txn.create(std::move(o));
            if (m.try_commit(own_txn).status == CommitStatus::Committed)
                private_creates.fetch_add(1, std::memory_order_relaxed);

            for (int i = 0; i < kRoundsPerThread; ++i) {
                Transaction txn = m.begin();
                txn.update(shared)->balance = t * 1000 + i;
                if (m.try_commit(txn).status == CommitStatus::Committed)
                    shared_commits.fetch_add(1, std::memory_order_relaxed);
                // Conflicts are expected and fine here -- many threads racing
                // the same object via OCC -- just don't corrupt anything.
            }
        });
    }
    for (auto& th : pool) th.join();

    CHECK_EQ(private_creates.load(), kThreads);
    CHECK(shared_commits.load() > 0);

    // Exactly one entry survives for `shared`, plus one per thread for its
    // own private object.
    const auto summaries = m.list_undo();
    CHECK_EQ(summaries.size(), std::size_t{kThreads + 1});

    // Every survivor can be taken and later applied without conflict --
    // proof that the survivors are genuinely independent of each other and
    // of everything that got pruned along the way.
    std::vector<Model::UndoEntry> taken;
    for (const auto& s : summaries) {
        auto entry = m.take_undo(s.version);
        CHECK(entry.has_value());
        if (entry.has_value()) taken.push_back(std::move(*entry));
    }
    CHECK(m.list_undo().empty());  // every survivor was taken

    for (const auto& entry : taken) CHECK(m.apply_undo(entry).status == CommitStatus::Committed);

    for (int t = 0; t < kThreads; ++t)
        CHECK(m.snapshot().find_by_key<&Account::name>("PRIVATE" + std::to_string(t)) == nullptr);
    CHECK(m.snapshot().find(shared) != nullptr);  // restored, not removed -- RestoreUpdate, not Recreate
}

// Several threads each commit SEVERAL independent private transactions --
// one NEW Account per commit, never the same id touched twice -- so nothing
// ever overlaps, either across threads or within one thread's own history
// (sequentially updating the SAME id would self-prune its own earlier entry;
// that's genuinely a conflict, per publish_now()'s own rule, and is exactly
// what the test above already covers). Afterward, each thread undoes ALL of
// its own transactions, in commit order, one thread per history, all
// running concurrently again -- proving private-object undo entries are
// fully independent of both other threads' forward work AND other threads'
// concurrent undo work.
TEST(each_writer_thread_can_undo_all_of_its_own_private_transactions_in_order) {
    Model m;
    constexpr int kThreads = 4;
    constexpr int kCommitsPerThread = 7;

    std::vector<std::vector<std::uint64_t>> per_thread_versions(kThreads);
    std::atomic<int> writers_ok{0};

    std::vector<std::thread> writers;
    for (int t = 0; t < kThreads; ++t) {
        writers.emplace_back([&, t] {
            bool ok = true;
            for (int i = 0; i < kCommitsPerThread; ++i) {
                Transaction txn = m.begin();
                auto a = std::make_unique<Account>();
                a->name = "THREAD" + std::to_string(t) + "_" + std::to_string(i);
                txn.create(std::move(a));
                const CommitResult res = m.try_commit(txn);
                ok = ok && (res.status == CommitStatus::Committed);
                per_thread_versions[t].push_back(res.snapshot.version());
            }
            if (ok) writers_ok.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& th : writers) th.join();
    CHECK_EQ(writers_ok.load(), kThreads);

    // Every commit created a DIFFERENT, private object -- nothing pruned
    // anything: kThreads * kCommitsPerThread entries total, one per commit.
    CHECK_EQ(m.list_undo().size(), static_cast<std::size_t>(kThreads * kCommitsPerThread));

    std::atomic<int> undoers_ok{0};
    std::vector<std::thread> undoers;
    for (int t = 0; t < kThreads; ++t) {
        undoers.emplace_back([&, t] {
            bool ok = true;
            for (std::uint64_t version : per_thread_versions[t]) {  // in commit order
                auto entry = m.take_undo(version);
                ok = ok && entry.has_value();
                if (entry.has_value()) {
                    const CommitResult res = m.apply_undo(*entry);
                    ok = ok && (res.status == CommitStatus::Committed);
                }
            }
            if (ok) undoers_ok.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& th : undoers) th.join();
    CHECK_EQ(undoers_ok.load(), kThreads);

    // Every one of every thread's own transactions was undone.
    for (int t = 0; t < kThreads; ++t)
        for (int i = 0; i < kCommitsPerThread; ++i)
            CHECK(m.snapshot().find_by_key<&Account::name>(
                      "THREAD" + std::to_string(t) + "_" + std::to_string(i)) == nullptr);
}
