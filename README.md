# mvcc-model

A snapshot-isolated C++17 object model with **multi-writer optimistic concurrency**:
**many writers, many readers, objects referencing each other by key.**

Readers work on immutable snapshots; writers build private transactions and race to commit:

- Readers take an **O(1) immutable snapshot**. `Ref<T>` never dangles inside one.
- Any thread can write: `begin()` a `Transaction` from a `Snapshot`, mutate a private local
  overlay, `try_commit()` it.
- A commit **succeeds** only if (a) no other transaction has committed since that touched the
  same ids, and (b) every `Ref<T>` in the final state still resolves against the **latest**
  committed state, not just the transaction's base.
- **Cascade delete is resolved once, at commit time**, against the single authoritative
  reverse index — never eagerly, never per-transaction. That one design choice is what keeps
  multi-writer support tractable: the reverse index never needs to become a structure that
  many transactions merge into.
- No per-object refcounting anywhere — chunks hold raw pointers, lifetime is a version
  watermark. See [DESIGN.md](DESIGN.md) for why, and [SNAPSHOTS.md](SNAPSHOTS.md) for how the
  spine and the HAMT secondary indexes actually copy-on-write and get reclaimed.

## Build

Needs CMake ≥ 3.20 and a C++20 compiler (GCC 11+, Clang 14+, MSVC 19.30+).

```bash
cmake --preset default
cmake --build --preset default -j
ctest --preset default
./build/default/demo
```

In VS Code: install the recommended extensions, then **Ctrl+Shift+B** builds and the
Testing panel runs the suite. Tasks exist for the sanitizer builds too.

## Sanitizers

The design is only correct if reclamation never frees an object a reader can still see, if
`try_commit()`'s apply phase is genuinely exclusive, and if concurrent writers never corrupt
shared state. "It didn't crash" proves none of that, so:

```bash
cmake --preset asan && cmake --build --preset asan -j && ctest --preset asan   # use-after-free
cmake --preset tsan && cmake --build --preset tsan -j && ctest --preset tsan   # data races
```

Both must be clean. TSan is the one that actually exercises the multi-writer claims. Keep
them clean — see `CLAUDE.md`.

## Using it

```cpp
#include "model/model.h"

// 1. Define your types.
//    The field TYPE decides what a delete does to you:
//      Ref<T>  non-nullable -- deleting the target CASCADES and kills this object
//      Opt<T>  nullable     -- deleting the target NULLS this field; the object lives
class Account final : public model::Object<Account> {
public:
    std::string name;
    // Every declaration below is opt-in; a type that needs none declares none.
};

class Order final : public model::Object<Order> {
public:
    std::string code;
    model::Ref<Account> account;   // deleting the account kills this order
    model::Opt<Order>   parent;    // deleting the parent nulls this field
    std::int64_t qty = 0;

    std::string computed_key() const { return "ord:" + code; }

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Order::account>(), s.account, model::RefLookupType::Exact, "account");
        v(model::field_tag<&Order::parent>(), s.parent, model::RefLookupType::Scan, "parent");
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
    }

    // Multi-match lookups: each field's lookup type decides how find_by_field/
    // find_referrers resolves it -- Exact (or Coarse) via an O(log n + matches)
    // index, Scan via an O(#objects) fallback. A field is tagged exactly one way.
    template <class Self>
    static void define_fields(Self& s, const model::LookupFieldReader& v) {
        v.field<&Order::qty>(s.qty, model::LookupType::Scan, "qty");
    }
};

// 2. Any thread: begin a transaction, mutate it locally, try to commit.
model::Model m;

model::Transaction txn = m.begin();                    // pins the current committed state
auto a = std::make_unique<Account>(); a->name = "A1";
model::Ref<Account> acct = txn.create(std::move(a));    // local until commit

auto o = std::make_unique<Order>(); o->code = "O1";
o->account = acct;                                      // may reference another local create
model::Ref<Order> ord = txn.create(std::move(o));

model::CommitResult res = m.try_commit(txn);
if (res.status == model::CommitStatus::Conflict) {
    // Someone else committed a change to an id this transaction touched, or
    // one of this transaction's Ref<>s no longer resolves against the
    // latest state. Discard txn and retry with a fresh begin().
} else if (res.status == model::CommitStatus::Committed) {
    // res.snapshot is the newly published state; res.changes is the FULL
    // resolved changeset, including anything cascade-deleted. Refs returned by
    // txn.create() hold local ids: res.to_real(ref) yields the committed one.
}

// 3. Reader threads: take an O(1) snapshot and look things up through it.
model::Snapshot s = m.snapshot();                     // O(1)
const Order* x = s.find_by_key<&Order::computed_key>("ord:O1");

s.for_each<Order>([&](const Order& o) { /* type-filtered iteration */ });

// Multi-match: every order with qty == 5 -- find_by_field falls back to an
// unindexed scan here, since qty is tagged LookupType::Scan above...
std::vector<const Order*> q5 = s.find_by_field<&Order::qty>(5);
// ...vs. every order FOR this account -- find_referrers resolves via the
// index here, since account is tagged RefLookupType::Exact above.
std::vector<const Order*> mine = s.find_referrers<&Order::account>(res.to_real(acct));

// 3b. Views: a typed traversal handle scoped to a Snapshot, not a Transaction.
// Two ways in: look one up, or pair an object you already read with the
// snapshot it came from.
if (x) {
    model::View<Order> xv = s.view(*x);                     // the object is in hand, so this is a
    xv[&Order::account]->name;                              // View: nothing to check, here or on a
}                                                           // Ref<> hop off it
auto v = s.view_by_key<&Order::computed_key>("ord:O1");     // a lookup can miss: OptView<Order>

// An empty view propagates through every later hop, so ONE check at the end
// covers all three ways this chain can come up empty -- no such key, a null
// parent, or an account a cascade took.
if (auto d = v[&Order::parent][&Order::account]) d->name;   // d is an OptView<Account>

// Narrow to a View only to HOLD a non-nullable handle: view() asserts it is
// non-empty, and from there a Ref<> hop stays a View with no check to make.
if (v) {
    v->qty;                                                 // -> reaches the Order, not the view
    model::View<Account> owner2 = v.view()[&Order::account];
}

// 3c. The same hops without a view: resolve against a Snapshot by hand. You
// name the snapshot at every hop, which is also what makes resolving an object
// read from one snapshot against a different one compile. Views exist so that
// mistake is not expressible.
if (x) {
    const Account& owner = s.resolve(x->account);     // Ref<Account> -> const Account&
    const Order*   dad   = s.resolve(x->parent);      // Opt<Order>   -> const Order*
}

// 4. Subscribers: coalescing change events, bounded queue depth.
auto sub = m.subscribe(/*queue_depth=*/8);
while (auto u = sub->wait_for_update()) {
    for (const model::Change& c : *u->changes) { /* c.id, c.kind */ }
}
```

**`txn.remove(acct)` only records an intent.** The cascade fan-out — every order that
references `acct` via a non-nullable `Ref<>`, transitively — is computed once, inside
`try_commit()`'s serialized apply phase, against the single authoritative reverse index. The
resolved kill list shows up in `CommitResult::changes`, not as a return value from `remove()`.
Within the same transaction, an object that *would* cascade-die from a pending `remove()`
still looks alive via `txn.peek()`/`txn.exists()` until commit — cascade effects are
invisible locally, by construction.

**This is object-write-set OCC, not full serializable OCC.** Two transactions that only
*read* the same ids (never write or reference them) can both commit even if that produces a
write-skew anomaly. That's a faithful reading of "a commit succeeds if no one touched the
same ids and every Ref<> still resolves" — not a gap, a documented scope boundary.

**`try_commit()`'s apply step is fully serialized** behind one mutex (`commit_mu_`).
Transaction *building* — every `create`/`update`/`remove`/`peek` call — touches no shared
state and takes no lock, so any number of threads can build concurrently with zero
contention. *Applying* is not parallel: `examples/commit_bench.cpp`'s thread-count sweep
shows this plainly (throughput does not scale linearly with writer threads). That's the
tradeoff this design makes, not an oversight.

**Multi-match lookups are cost-transparent, not cost-fixed, and cost is why an index is
opt-in.** Beyond `define_keys()` (unique, indexed, `find_by_key`), `find_by_field`/
`find_referrers` are each ONE entry point that resolves via a persistent index —
O(log n + matches) — when the field is tagged `LookupType::Exact` (or `Coarse`) in
`define_fields()`, or `RefLookupType::Exact` in `define_references()`, and transparently falls
back to an O(#objects) scan — costs nothing until you call it — when it's tagged `Scan`
instead (every `Ref<>`/`Opt<>` field is scan-fallback-eligible the moment it's in
`define_references()`, regardless of its tag). The indexed path isn't free: each indexed field
costs roughly one extra index entry per object, upkept inside the serialized `try_commit()`
apply phase on every create, delete, and value change — the same write-side tax `by_type`
already pays, just per declared field instead of per object. That's why it's opt-in rather
than automatic: index only the fields actually queried often — `Model::lookup_stats()`
reports, per field, how many calls resolved via the index versus fell back to the scan, which
is the signal for that decision — and leave the rest on the scan (or on nothing at all —
`find_by_predicate()`'s predicate scan always works, undeclared or not, just at O(#objects)
per call instead of paid for once per commit).

## Layout

```
include/model/model.h   the model: identity, Ref<>/Opt<>, Object<T>, Snapshot, Transaction, Model
src/model.cpp           the implementation
include/example/types.h example user types (Account, Order) -- not part of the model
examples/demo.cpp       3 writer threads racing try_commit(), 2 readers, 1 slow subscriber
examples/bench.cpp      why View holds its Snapshot by pointer
examples/commit_bench.cpp   commit latency vs. model size, and throughput vs. writer-thread count
tests/                  test_*.cpp plus persistent_map_tests.cpp and performance_tests.cpp; no external dependencies
CLAUDE.md               design rules, for Claude Code
DESIGN.md               why it's built this way
SNAPSHOTS.md            how the spine and HAMT indexes COW and get reclaimed, with diagrams
```

## Status

Working, tested, clean under ASan/UBSan/TSan. On the read side: snapshot isolation, typed
`Ref<T>`/`Opt<T>`, cascade delete, `View<T>`/`OptView<T>`, generation-exhaustion handling, a
persistent HAMT secondary index, a background reaper thread, lock-free-ish snapshot
acquisition. On the write side: `Transaction`-scoped local writes, `try_commit()` with
conflict detection re-validated against latest state, and commit-time cascade resolution.

Reclamation is asynchronous — use `wait_for_reclamation()` as a barrier where you need
determinism (tests, shutdown).

Known scope boundaries, not bugs (see [CLAUDE.md](CLAUDE.md)): object-write-set OCC (not
full serializable), `try_commit()`'s apply step is fully serialized behind one mutex, and the
reverse index (`referrers_`) scans linearly per target.

## Glossary

Every abbreviation used in this codebase's comments, docs, and identifiers, what it stands
for, what it means specifically in this project, and where to find it.

**Core types** (spelled out once here, not re-abbreviated below):

- **`Id`** — short for Identifier: a `{index, generation}` pair naming a slot in the store.
  Never dangles silently — a stale generation is detected, not aliased. `include/model/model.h`
- **`Ref<T>`** — short for Reference: a non-nullable, 8-byte handle to a `T` that always
  resolves inside any snapshot containing its holder; deleting the target cascades. See
  "Using it" above. `include/model/model.h`
- **`Opt<T>`** — short for Optional: the nullable counterpart to `Ref<T>`; deleting the
  target nulls the field instead of cascading. See "Using it" above. `include/model/model.h`

**Concurrency & reclamation design:**

- **MVCC** — Multi-Version Concurrency Control. The project's whole paradigm and the source
  of its name: readers see an immutable, versioned snapshot instead of taking a lock; writers
  build a private overlay and publish a new version atomically. Project-wide; see `DESIGN.md`.
- **OCC** — Optimistic Concurrency Control. The write strategy: a `Transaction` builds
  entirely locally with no lock taken, then `try_commit()` validates it once against the
  latest committed state instead of locking upfront. Specifically **object-write-set OCC**
  (validates only touched ids and every `Ref<>`'s target) rather than full serializable OCC —
  a documented scope boundary, not a gap. `README.md`, `CLAUDE.md`, `DESIGN.md`.
- **COW** — Copy-On-Write. Published state (`Root`, its `Chunk`s) is cloned only when a
  commit actually dirties it; everything else stays shared, byte-for-byte, between versions —
  why publishing a new snapshot is cheap. `DESIGN.md`, `src/model.cpp`.
- **RCU** — Read-Copy-Update. Loosely describes the model's epoch/version-watermark
  reclamation: an object is freed only once no live snapshot's pinned version could still see
  it — reclamation by tracking who might still be reading, not by refcounting. `DESIGN.md`
  ("version-based (epoch/RCU) reclamation").
- **RAII** — Resource Acquisition Is Initialization. `Snapshot`'s version pin/release:
  pinning happens in the constructor, releasing (and waking the reaper) happens in the
  destructor, so a `Snapshot` going out of scope is what tells the reaper a version may now
  be reclaimable. `include/model/model.h`.
- **BFS** — Breadth-First Search. The traversal cascade delete uses to walk the reverse
  index (`referrers_`) outward from a removed id, resolved once inside `try_commit()`'s
  serialized apply phase — never eagerly. Its visited-set is what keeps it safe against
  reference cycles (see DAG, below). `Model::remove_raw` in `src/model.cpp`; `CLAUDE.md`
  invariant 8.
- **DAG** — Directed Acyclic Graph. Called out as something the non-nullable `Ref<>`
  reference graph is explicitly **not** guaranteed to be: a same-transaction pre-minted local
  id can create a genuine `Ref<>` cycle, which the cascade BFS's visited-set has to survive.
  `tests/test_helpers.h` (the `Link` type), `tests/test_cascade.cpp` (its cascade-cycle tests).

**Hot-path tricks** (see CLAUDE.md's "Things that look like improvements but are not"):

- **CRTP** — Curiously Recurring Template Pattern. `Object<Derived>` — the base every user
  type derives from (`class Order : public model::Object<Order>`) — uses it to give each
  type its `TypeTag` and default `clone()`/reference-visiting machinery without a
  vtable-heavy type hierarchy. `include/model/model.h`.
- **RTTI** — Run-Time Type Information. Deliberately avoided on the read path: the
  hand-rolled `TypeTag` (one virtual call plus a pointer compare) replaces what
  `dynamic_cast`/`typeid`-based downcasting would cost at millions of calls/sec.
  `include/model/model.h`; `CLAUDE.md` ("Why a hand-rolled TypeTag instead of dynamic_cast?").
- **ABI** — Application Binary Interface. Used only for diagnostic type-name demangling:
  Itanium-ABI (GCC/Clang) mangled names are demangled for readable diagnostics; MSVC's names
  are already readable and pass through unchanged. `include/model/model.h`
  (`demangle_type_name`).
- **FNV-1a** — Fowler–Noll–Vo hash, variant 1a. The string-hashing algorithm behind
  `StringHash`, used to key `PersistentMap`/`PersistentSet` instances on real `std::string`
  keys (e.g. `Root::by_key`). `include/model/persistent_map.h`.
- **HAMT** — Hash Array Mapped Trie. The from-scratch persistent (immutable,
  structure-sharing) map/set backing the model's secondary indexes — deriving a new version
  path-copies only O(log32 n) nodes instead of the whole index. `include/model/persistent_map.h`.

**Tooling & build:**

- **ASan** — AddressSanitizer. One of the two sanitizer presets that must stay clean before
  any change to the model counts as done (see `CLAUDE.md`). Catches use-after-free — the
  central risk of a design that frees objects by version watermark instead of refcounting.
  Also enables UBSan. `CMakeLists.txt`, `CMakePresets.json` (`asan` preset).
- **TSan** — ThreadSanitizer. The other mandatory sanitizer preset. Catches data races, and
  is the one that actually exercises concurrent `try_commit()` calls racing each other.
  `CMakeLists.txt`, `CMakePresets.json` (`tsan` preset).
- **UBSan** — UndefinedBehaviorSanitizer. Bundled into the `asan` preset (`MODEL_ASAN` turns
  on both). `CMakeLists.txt`.
- **NDEBUG** — the standard C++ macro that disables `assert()`. Deliberately **not** defined
  in the default preset: `Snapshot::resolve()`'s assert is the tripwire for the model's
  central invariant (invariant 1), so disabling it would hide the exact bug class the design
  most needs to catch. `CLAUDE.md`.
- **GCC / MSVC** — GNU Compiler Collection / Microsoft Visual C++. The compilers this
  project targets (GCC 11+, Clang 14+, MSVC 19.30+, for C++20 support). `README.md` (Build
  section).

**General:**

- **API** — Application Programming Interface. `Model` exposes no direct
  create/update/remove/commit methods: every write is made through a transaction and applied
  under `commit_mu_`, so a second write path can never touch the same shared state (see
  `CLAUDE.md` invariant 7).
