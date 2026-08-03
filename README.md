# mvcc-model

A snapshot-isolated C++20 object model with **multi-writer optimistic concurrency**:
**many writers, many readers, objects referencing each other by key.**

This is a sibling project to [`snapshot-model`](../snapshot-model), which is the
single-writer version. Everything on the read side is identical. The difference is entirely
on the write side:

- Any thread can write: `begin()` a `Transaction` from a `Snapshot`, mutate a private local
  overlay, `try_commit()` it.
- A commit **succeeds** only if (a) no other transaction has committed since that touched the
  same ids, and (b) every `Ref<T>` in the final state still resolves against the **latest**
  committed state, not just the transaction's base.
- **Cascade delete is resolved once, at commit time**, against the single authoritative
  reverse index — never eagerly, never per-transaction. That one design choice is what keeps
  multi-writer support tractable: the reverse index never needs to become a structure that
  many transactions merge into.
- Readers still take an **O(1) immutable snapshot**. `Ref<T>` still never dangles inside one.
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

Both must be clean. TSan matters more here than it did in the single-writer project — it's
the one that actually exercises the new multi-writer claims. Keep them clean — see
`CLAUDE.md`.

## Using it

```cpp
#include "model/model.h"

// 1. Define your types -- identical to the single-writer sibling project.
//    The field TYPE decides what a delete does to you:
//      Ref<T>  non-nullable -- deleting the target CASCADES and kills this object
//      Opt<T>  nullable     -- deleting the target NULLS this field; the object lives
class Order final : public model::Object<Order> {
public:
    std::string code;
    model::Ref<Account> account;   // deleting the account kills this order
    model::Opt<Order>   parent;    // deleting the parent nulls this field
    std::int64_t qty = 0;

    std::string computed_key() const { return "ord:" + code; }

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Order::account>(), "account", s.account);
        v(model::field_tag<&Order::parent>(), "parent", s.parent);
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::computed_key>(s.computed_key());
    }

    // Multi-match lookups are separate, opt-in declarations -- see below.
    template <class Self>
    static void define_scan_fields(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::qty>(s.qty);      // find_by_scan_field: every match, O(#objects) scan
    }
    template <class Self>
    static void define_cached_references(Self& s, const model::RefIndexReader& v) {
        v.index<&Order::account>();     // find_cached_referrers: every match, O(log n + matches)
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
    // resolved changeset, including anything cascade-deleted.
}

// 3. Reader threads: unchanged from the single-writer project.
model::Snapshot s = m.snapshot();                     // O(1)
const Order* x = s.find_by_key<&Order::computed_key>("ord:O1");
const Account& owner = s.resolve(x->account);         // Ref<Account> -> const Account&
const Order*   dad   = s.resolve(x->parent);          // Opt<Order>   -> const Order*

s.for_each<Order>([&](const Order& o) { /* type-filtered iteration */ });

// Multi-match: every order with qty == 5 (unindexed scan)...
std::vector<const Order*> q5 = s.find_by_scan_field<&Order::qty>(5);
// ...vs. every order FOR this account (indexed -- see define_cached_references above).
std::vector<const Order*> mine = s.find_cached_referrers<&Order::account>(acct);

// 3b. Views work exactly as before -- scoped to a Snapshot, not a Transaction.
auto v = *s.view_by_key<&Order::computed_key>("ord:O1");
v->qty;
View<Account> owner2 = v[&Order::account];
if (auto d = v[&Order::parent]) (*d)[&Order::account]->name;

// 4. Subscribers: unchanged.
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

**Multi-match lookups are opt-in, and cost is why.** Beyond `define_keys()` (unique,
indexed, `find_by_key`), a type can declare `define_scan_fields()` for "every match" via an
O(#objects) scan — costs nothing until you call it — or `define_cached_fields()` /
`define_cached_references()` for "every match" via a persistent index (`find_by_cached_field`
/ `find_cached_referrers`), O(log n + matches) per query. The indexed forms aren't free: each
declared field costs roughly one extra index entry per object, upkept inside the serialized
`try_commit()` apply phase on every create, delete, and value change — the same write-side
tax `by_type` already pays, just per declared field instead of per object. That's why it's
opt-in rather than automatic: index only the fields actually queried often, and leave the
rest on the scan (or on nothing at all — `find_all()`'s predicate scan and
`for_each_referrer()` always work, undeclared or not, just at O(#objects) per call instead of
paid for once per commit).

## Layout

```
include/model/model.h   the model: identity, Ref<>/Opt<>, Object<T>, Snapshot, Transaction, Model
src/model.cpp           the implementation
include/example/types.h example user types (Account, Order) -- same as the sibling project
examples/demo.cpp       3 writer threads racing try_commit(), 2 readers, 1 slow subscriber
examples/bench.cpp      why View holds its Snapshot by pointer (unchanged from the sibling)
examples/commit_bench.cpp   commit latency vs. model size, and throughput vs. writer-thread count
tests/tests.cpp         no external dependencies
CLAUDE.md               design rules, for Claude Code
DESIGN.md               why it's built this way, and how this differs from the single-writer sibling
SNAPSHOTS.md            how the spine and HAMT indexes COW and get reclaimed, with diagrams
```

## Status

Working, tested, clean under ASan/UBSan/TSan. Implements everything the single-writer sibling
project does on the read side (snapshot isolation, typed `Ref<T>`/`Opt<T>`, cascade delete,
`View<T>`, generation-exhaustion handling, a persistent HAMT secondary index, a background
reaper thread, lock-free-ish snapshot acquisition), plus, new here: `Transaction`-scoped local
writes, `try_commit()` with conflict detection re-validated against latest state, and
commit-time cascade resolution.

Reclamation is asynchronous — use `wait_for_reclamation()` as a barrier where you need
determinism (tests, shutdown).

Known scope boundaries, not bugs (see [CLAUDE.md](CLAUDE.md)): object-write-set OCC (not
full serializable), `try_commit()`'s apply step is fully serialized behind one mutex, and the
reverse index (`referrers_`) still scans linearly per target — same as the single-writer
sibling, unaffected by this redesign.

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
  `tests/tests.cpp` (the `Link` type and its cascade-cycle tests).

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
  keys (e.g. `Root::by_field`). `include/model/persistent_map.h`.
- **HAMT** — Hash Array Mapped Trie. The from-scratch persistent (immutable,
  structure-sharing) map/set backing the model's secondary indexes — deriving a new version
  path-copies only O(log32 n) nodes instead of the whole index. `include/model/persistent_map.h`.

**Tooling & build:**

- **ASan** — AddressSanitizer. One of the two sanitizer presets that must stay clean before
  any change to the model counts as done (see `CLAUDE.md`). Catches use-after-free — the
  central risk of a design that frees objects by version watermark instead of refcounting.
  Also enables UBSan. `CMakeLists.txt`, `CMakePresets.json` (`asan` preset).
- **TSan** — ThreadSanitizer. The other mandatory sanitizer preset. Catches data races, and
  matters more here than in the single-writer sibling project because it's the one that
  actually exercises concurrent `try_commit()` calls racing each other. `CMakeLists.txt`,
  `CMakePresets.json` (`tsan` preset).
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

- **API** — Application Programming Interface. Notably invoked to explain why `Model`'s old
  single-writer create/update/remove/commit methods and the new `Transaction`-based one are
  deliberately never allowed to coexist (see `CLAUDE.md` invariant 7).
- **GC** — Garbage Collection. Used loosely as a section label ("Changelog / GC") over
  changelog-pruning tests; the model's actual reclamation mechanism is the version-watermark
  reaper described under RCU, above — not a general-purpose GC. `tests/tests.cpp`.
