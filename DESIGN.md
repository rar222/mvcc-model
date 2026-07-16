# Design

Why the model is built the way it is. `CLAUDE.md` has the rules; this is the reasoning
behind them. This project is a sibling of `snapshot-model` (single-writer); read that
project's own DESIGN.md for the storage/reclamation/events narrative in full — it is
reproduced here because this project no longer depends on that one, but nothing about it
changed. The new material is the "Multi-writer optimistic concurrency" section.

## Requirements

- Object model where objects reference each other **by key**.
- **Many reader threads** obtain a consistent **snapshot**.
- **Many writer threads**, each batching mutations into a transaction and attempting to
  commit it optimistically.
- Fetch objects **by key** and **by reference**.
- **Callback/notification** on change.

Scale: 100k–1M objects, 10–100 commits/sec, constant creates and deletes, reads vastly
outnumbering writes. References must **always resolve within a snapshot** (never null),
except for explicitly nullable ones. Deleting an object **cascades** to its non-nullable
referrers.

## The decision everything else follows from

Snapshots must be cheap, so published state must be immutable and structurally shared. The
obvious shape is a `shared_ptr<const Root>` holding a map of `shared_ptr<const T>`.

That does not survive the numbers.

Rebuilding the whole map per commit is O(n) — at 1M objects and 100 Hz, the memcpy is
tolerable (~16MB, a few ms) but the **1M atomic refcount increments are not**. That is
5–20ms of cache-line traffic, 100 times a second: a core burned doing nothing.

The fix is to make commits proportional to *changes*, not to model size, and to get
refcounts out of the hot structure entirely:

- Objects live in a **chunked copy-on-write array**, indexed by a dense slot. A commit
  copies only the chunks it dirties.
- Chunks hold **raw `const ObjectBase*`**. Copying a chunk is a `memcpy` — no atomics,
  no allocator traffic.
- The **only refcount in the system** is `shared_ptr<const Root>`: one atomic bump when a
  reader acquires a snapshot, not one per object.
- Lifetime is therefore **version-based (epoch/RCU) reclamation**, not refcounting.

Everything unusual in the codebase, including the multi-writer redesign below, is downstream
of this decision. None of it changed by adding multiple writers — a `Transaction` still
publishes through the exact same immutable, COW `Root`.

## Structure

```
Root  { version, spine, by_type, by_field }    immutable, published atomically
  spine    : vector<shared_ptr<const Chunk>>      ~n/256 entries; copied whole per commit
  Chunk    : 256 x const ObjectBase*              COW; copied only when dirtied
             256 x uint32 generation
  by_type  : unordered_map<TypeTag, PersistentMap<Id>>   internal, Id-keyed: what for_each<T>() scans
  by_field : unordered_map<field_tag, PersistentMap<Id>> lookup index, see define_keys()
```

A read is `spine[k >> 8]->obj[k & 0xff]` — an array index. No hashing, no trie walk, one
cache miss.

### Identity

`Id { uint32 index; uint32 gen; }`. The generation is bumped on every reuse of a slot. With
constant deletes, slots recycle continuously, so without a generation a stale `Id` would
silently alias whatever object landed there next.

### Refs are typed, and thin

A reference is an `Id` wearing a type: `Ref<T>` for non-nullable, `Opt<T>` for nullable.
Nullability is a property of the field type — the visitor that nulls a field during a
cascade (`RefNuller`) has *no overload* that can clear a `Ref<T>`. A type's
`define_references()` enumerates its reference fields once, and that single list drives
`clone()`, the reverse index, cascade delete, nulling, **and** local-id remapping
(`RefRemapper` — new in this project, see below).

A reference is still just an `Id` underneath; it resolves through a `Snapshot`, never itself.
The alternative — a "fat" ref carrying a pointer to its snapshot root — means every stored
ref pins a snapshot alive, stalling reclamation.

## Multi-writer optimistic concurrency

The single-writer sibling project can treat its reverse index (`referrers_`), its secondary
indices, and its free list as **plain mutable writer-private state** — safe because there is
only ever one writer, and it is always looking at the latest version. That assumption is
gone here. The question this project had to answer: how do you let many threads mutate the
model concurrently without turning `referrers_` into a structure that many transactions have
to merge into?

**The answer is to not let transaction-building touch `referrers_` (or any other shared
state) at all.** A `Transaction` is a pile of purely local data: a pinned base `Snapshot`,
a list of not-yet-real objects (local ids), a map of copy-on-write clones for objects it
updates, and a set of ids it intends to remove. None of that is installed anywhere. Building
a transaction — any number of `create`/`update`/`remove`/`peek` calls, on any number of
threads, concurrently — touches nothing but that transaction's own memory. There is
genuinely nothing to synchronize during this phase.

The one place shared state gets touched is `Model::try_commit()`, and it is **fully
serialized** behind one mutex, `commit_mu_`. Inside that critical section:

1. **Conflict check**: has anything this transaction wrote (updates, remove-intents) been
   touched by a transaction that committed since this one's base version? A `changelog_` of
   resolved changesets, one entry per commit, answers this by scanning entries newer than
   `txn.base_version()`. (New local creates can never conflict — nobody else can know about
   an id that didn't exist until this transaction created it.)
2. **Apply creates, then updates, then remove-intents, in that order.** Creates get a real
   slot and generation; any local placeholder ids they reference (from other creates earlier
   in the same transaction) get rewritten to real ids via `RefRemapper`. Updates are
   installed and their referrer/field-key edges reconciled **immediately** — not deferred to
   a later pass, because the very next step needs to see them.
3. **Resolve remove-intents.** This is exactly the single-writer sibling's `remove_raw()`
   BFS, reused essentially unchanged, run once per intent — now against `referrers_` as it
   stands *after* this transaction's own updates have already been reconciled into it, so a
   same-transaction "repoint away from X, then delete X" behaves correctly.
4. Every `validate()` call in steps 2–3 checks against **the current state**, not the
   transaction's base — so a `Ref<>` this transaction just created or repointed is checked
   against whatever is *actually* alive right now, not a possibly-stale view. This is what
   makes "every Ref<T> still resolves" a property of the *latest* committed state, for free,
   with no separate re-validation pass.
5. On failure (an `IntegrityError`, or a `false` from the pre-commit hook), everything applied
   in this attempt unwinds via an undo log — the same rollback mechanism the single-writer
   project uses for a failed `commit()`, just scoped to one `try_commit()` attempt instead of
   an open-ended session.
6. On success: version bump, COW `Root` build, atomic publish, subscriber notify, append to
   `changelog_`, prune whatever the changelog no longer needs.

**Local placeholder ids** are what let a transaction build "create an Account, then create an
Order referencing it" before either has a real id: the top bit of `Id::index` marks a value
that is only meaningful inside the transaction that minted it, and `RefRemapper` rewrites
every such id to its real one during apply. This needed zero changes to `Ref<T>`/`Opt<T>`
themselves — remapping is just another visitor over `define_references()`, the same
mechanism that already drives cloning, cascading, and nulling.

**The changelog's retention piggybacks on the reclamation watermark that already exists.** A
`Transaction`'s base is a real, pinned `Snapshot` — registered in `live_` exactly like a
reader's. So the watermark can never advance past any open transaction's base version, which
means any changelog entry at or below that watermark can never be needed by a conflict check
again. No separate bookkeeping; the existing reclamation machinery already proves it safe.

### What this buys, and what it costs

- Transaction building is embarrassingly parallel: zero shared-state contention, at any
  thread count.
- `try_commit()`'s apply step is not parallel: `commit_mu_` serializes it model-wide. This is
  a deliberate scope boundary, not an oversight — see `examples/commit_bench.cpp`'s
  thread-count sweep, which is expected to show sub-linear throughput scaling.
- This is **object-write-set OCC**, not full serializable OCC: two transactions that only
  *read* overlapping state, without either writing or referencing it, can both commit even
  if the result is a write-skew anomaly. The user's own specification of this design ("a
  commit succeeds if no one touched the same ids and every Ref<T> is still valid") is
  exactly this level of isolation, not a stronger one — worth stating plainly so nobody
  mistakes a `Conflict` for a guarantee this design doesn't make.
- Cascade delete is **resolved once, at commit time**, not eagerly at `remove()`-call time.
  Within a transaction, an object that would cascade-die from a pending `remove()` still
  looks alive locally until commit. The alternative — a duplicate, transaction-local reverse
  index so cascade effects are visible immediately — was considered and rejected: real
  complexity for a narrow benefit, and it would reintroduce exactly the "many transactions
  touching a shared reverse-index-shaped structure" problem this whole design exists to
  avoid.

## Referential integrity

Unchanged in mechanism from the single-writer sibling, just re-homed: `Model::validate()`
rejects any object whose non-nullable refs are null or point at something dead, and it is
called from inside `try_commit()`'s apply phase against current state. The reverse index
(`Id -> [(referrer, field, nullable)]`) is still plain mutable state — it just now requires
`commit_mu_` instead of "only one thread exists" to be safe to touch.

**Cascade fan-out is still unbounded and invisible to the caller** — same open TODO as the
single-writer sibling (a two-phase plan/apply for `remove()` would help both projects
equally).

## Reclamation, Events

Identical to the single-writer sibling in every respect: version-watermark-based
reclamation, a background reaper, bounded coalescing subscriber queues. None of this needed
to change — a `Transaction`'s base `Snapshot` participates in the watermark exactly like a
reader's, and publication still happens at one point (the end of `try_commit()`), just
reached by many possible callers instead of one dedicated thread.

## What was learned building it

- **Reconciliation timing is the whole game.** The single-writer design deferred referrer/
  field-key reconciliation to the end of `commit()`, comparing a baseline captured at
  `update()`-call time against the final clone. This project's `Transaction::update()`
  clones once, eagerly, and the caller writes to it directly — so reconciliation has to run
  immediately, right after the object is fully written, not before. Getting this ordering
  wrong for the *nullable-referrer* case inside the cascade BFS (`remove_raw`'s
  now-immediate `clone_for_cascade_null` + `null_ref()` + reconcile sequence) would silently
  reconcile against an object that hadn't been nulled yet — a guaranteed no-op that leaves a
  phantom entry in the reverse index. Same failure shape as a bug the single-writer project
  hit and fixed once already; worth naming explicitly so it doesn't come back.
- **Deferring cascade to commit time is the one idea that makes any of this tractable.**
  Every other piece of shared state in `try_commit()` — the spine, the indices, the reverse
  index — can be touched by exactly one transaction at a time because `commit_mu_` says so.
  If `remove()` had to resolve cascades against a *live*, concurrently-mutating reverse
  index at call time instead, there would be no single moment where that index is safe to
  read.
