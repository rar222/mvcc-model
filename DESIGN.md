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
Root { version, spine, by_type, by_field, by_cached_field, by_cached_reference }
  spine             : vector<shared_ptr<const Chunk>>   ~n/256 entries; copied whole per commit
  Chunk             : 256 x const ObjectBase*            COW; copied only when dirtied
                       256 x uint32 generation
  by_type           : unordered_map<TypeTag, PersistentMap<Id>>
                       internal, Id-keyed: what for_each<T>() scans
  by_field          : unordered_map<field_tag, PersistentMap<Id>>
                       unique lookup index, see define_keys()
  by_cached_field   : unordered_map<field_tag, PersistentMap<PersistentMap<Id>>>
                       multi-match index, see define_fields()'s LookupType::Cache fields
  by_cached_reference : unordered_map<field_tag, PersistentMap<PersistentMap<Id>>>
                       reverse-lookup index, see define_references()'s LookupType::Cache fields
```

All published atomically as one immutable `Root`. A read is `spine[k >> 8]->obj[k & 0xff]`
— an array index. No hashing, no trie walk, one cache miss.

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

### Lookup families: unique, and cost-transparent multi-match

`define_keys()` is the baseline index: one `PersistentMap<Id>` per declared field, UNIQUE (a
duplicate value silently overwrites the earlier one), O(log n) via `find_by_key`. Everything
past that is "give me every match," and is exposed through a single entry point per shape —
`find_by_field` for a value field, `find_referrers` for "who points at this?" — that resolves
via an index when one exists and transparently falls back to a scan when it doesn't, rather
than requiring the caller to name which of two functions to call. Each field declared in
`define_fields()`/`define_references()` carries a `LookupType` tag, `Cache` or `Scan` — exactly
one, never both (enforced by an assert the first time each type's declarations run, see
`Object<Derived>::validate_field_declarations`/`validate_ref_declarations`):

- **`LookupType::Scan`.** No index at all — the declaration is purely a *visibility gate* (a
  field declared in neither `define_fields()` nor `define_keys()` returns empty from
  `find_by_field`, same rule `find_by_key` follows), and `find_by_field` falls back to an
  O(#objects) linear scan comparing the field's real typed value. Zero write-side cost:
  nothing is touched at commit time. Right choice for a field queried rarely enough that
  paying per-query beats paying per-commit.
- **`LookupType::Cache`.** A real index: `PersistentMap<PersistentMap<Id>>`, outer key the
  field's own value (or, for a reference field, the *target's* `Id`), inner map a persistent
  *set* of every matching object's `Id`. `find_by_field`/`find_referrers` resolve via this
  index whenever the field is tagged this way. The inner map is deliberately a `PersistentMap`
  bucket, never a flat `vector<Id>` — a flat bucket would make every mutation of a
  low-cardinality value (a status, a category, a popular hub object) O(#objects sharing that
  value), which is exactly the size-proportional cost this whole design exists to avoid. A
  `PersistentMap` bucket keeps every mutation O(log n) regardless of how many objects share
  the key. Maintained by the same three-function shape as every other index in this design
  (`add_*`/`drop_*`/`reconcile_*`, undo-logged like everything else `try_commit()` touches),
  wired into all four apply-phase sites that can change membership: create, update
  (reconcile), cascade-null (reconcile), and cascade-delete (drop).

For a reference field, `LookupType::Cache` is the read-side counterpart of `referrers_`
(below): same "who points at this?" question, but published and O(log n + matches) instead of
writer-private and O(1)-but-never-exposed. It supplies no value of its own (the target and
nullability are already known from `define_references()`'s own list), just which fields are
worth the index. `find_referrers` already works on any `Ref<>`/`Opt<>` field the moment it's
listed in `define_references()`, `LookupType::Cache` or `Scan` alike — the tag only decides
whether that lookup resolves via the index or the scan fallback.

The cost of a `LookupType::Cache` field or reference is real: roughly one index entry per
object per declared field, upkept inside `commit_mu_` on every commit that touches it — the
same tax `by_type` already pays per object, just per declared field on top. That is why `Cache`
is opt-in rather than automatic, and why the scan fallback exists at all: index only what gets
queried often at scale — `Model::lookup_stats()`/`lookup_diagnostics()` report, per field,
how many calls actually resolved via the index versus fell back to the scan, which is the
signal for that decision — and leave the rest on the always-correct, zero-upkeep scan.

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
5. On failure (an integrity violation — reported by value as `CommitStatus::Conflict` or
   `Invalid` with `CommitResult::error`; this project has no exceptions — or a `false` from
   the pre-commit hook), everything applied in this attempt unwinds via an undo log — the
   same rollback mechanism the single-writer project uses for a failed `commit()`, just
   scoped to one `try_commit()` attempt instead of an open-ended session.
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

### Scaling past target scale: what breaks first, and the fix

At target scale (100k–1M objects), `Root::spine`'s per-commit cost is negligible:
`publish_now()` republishes the *whole* spine — `r->spine = spine_`, one `shared_ptr<const
Chunk>` copy per chunk, `O(total_slots / kChunkSize)` — regardless of how many objects the
commit actually touched. At 1M objects and `kChunkBits = 8` (256 slots/chunk, the value this
project started with) that's under 4,000 chunks: cheap enough to not show up.

That cost does not stay flat as the model grows past target scale, and it becomes the
dominant cost well before 100M objects. Measured (default preset, `Order` — the heaviest
example type, touching all four index families — `try_commit`, `try_commit_without_undo`
against a pre-populated model; "clustered" = 20 updates to objects created together, ~1-2
chunks touched; "scattered" = 20 updates to objects sampled uniformly across the whole id
range, ~20 distinct chunks touched; both medians of 5 runs), comparing the original
`kChunkBits = 8` (256 slots/chunk) against `kChunkBits = 12` (4,096 slots/chunk, since adopted
as the default):

| Objects | Chunks 8-bit / 12-bit | RSS 8-bit / 12-bit | Clustered latency 8-bit / 12-bit | Scattered latency 8-bit / 12-bit | Throughput (1→8 threads) 8-bit | Throughput (1→8 threads) 12-bit |
|---|---|---|---|---|---|---|
| 500k | 1,954 / 123 | 498 / 498 MB | 234 / 75 μs | 156 / 93 μs | 7,285→13,309/s | 69,867→113,559/s |
| 1M | 3,907 / 245 | 967 / 969 MB | 6,737* / 2,819 μs | 307 / 332 μs | 5,593→5,447/s | 42,887→79,957/s |
| 2M | 7,813 / 489 | 1,979 / 1,979 MB | 3,337 / 2,884 μs | 2,457 / 507 μs | 2,511→3,056/s | 26,003→50,640/s |
| 4M | 15,625 / 977 | 3,816 / 3,798 MB | 3,677 / 3,240 μs | 30,341 / 28,211 μs | 1,221→1,390/s | 16,850→43,293/s |
| 8M | 31,250 / 1,954 | 6,859 / 6,853 MB | 8,956 / 15,004 μs | 34,175 / 36,039 μs | 494→659/s | 7,826→9,733/s |
| 12M | 46,875 / 2,930 | 10,123 / 10,107 MB | 8,718 / 5,298 μs | 39,905 / 33,398 μs | 445→460/s (flat) | 4,299→5,887/s (still scales) |

*(\*1M clustered, 8-bit is a noisy outlier — 5 samples is a small sample.)*

By 4M objects at `kChunkBits = 8`, more writer threads stop helping at all — the "What this
buys, and what it costs" sub-linear scaling above degrades to *flat* scaling: `commit_mu_`'s apply step is by
then dominated by the spine copy, not by per-object index maintenance. Extrapolating
(commits/sec × chunk-count is roughly constant across this range) to 50M objects: **~90
commits/sec, independent of thread count** — inside this project's 10–100 commits/sec target,
but with no headroom left.

**The fix, measured, is retuning `kChunkBits`, not restructuring `Root::spine`.** Raising it
from 8 to 12 (256 → 4,096 slots/chunk — `kChunkBits`'s current value) cuts total chunk count
16×; measured throughput improves 8–16× across the same range (12M objects: 445 → 4,299
commits/sec single-threaded), and thread scaling stops being flat (1→8 threads still buys
~37%, instead of nothing). Extrapolated to 50M objects, the ceiling moves to roughly
1,000–1,600 commits/sec — comfortably above target, with real headroom. Memory cost of the
bigger chunk was negligible in measurement (<0.3% difference at every checkpoint from 500k to
12M): the internal-fragmentation downside of a bigger `kChunkSize` (a mostly-empty chunk still
pays for its full `obj[]`/`gen[]` arrays) never materializes for a workload that creates in
large sequential batches, since sequential `alloc_slot()` packs each chunk full before moving
to the next.

**What retuning `kChunkBits` does *not* fix**: a single small transaction touching objects
scattered across the whole id space (not clustered from a recent create) costs 30–40ms once
the model is multi-GB, and that cost is roughly the *same* at `kChunkBits = 8` and `= 12` —
the 16×-fewer-chunks case barely moved it. It is therefore not the spine-copy cost; the
likely cause is cold-memory access (cache/TLB misses touching ~20 widely-separated
multi-kilobyte chunks scattered across a multi-gigabyte heap), a cost `Root::spine`'s
per-commit copy was never the source of. Not yet root-caused — would need a profiler (`perf
stat` / cachegrind), not a throughput benchmark, to pin down.

## Referential integrity

Unchanged in mechanism from the single-writer sibling, just re-homed: `Model::validate()`
rejects any object whose non-nullable refs are null or point at something dead, and it is
called from inside `try_commit()`'s apply phase against current state. The reverse index
(`Id -> [(referrer, field, nullable)]`) is still plain mutable state — it just now requires
`commit_mu_` instead of "only one thread exists" to be safe to touch, and it is still
writer-private: it can never be handed to a reader (that's the whole reason `by_cached_
reference` — see "Lookup families" above, populated from `define_references()`'s
`LookupType::Cache`-tagged fields — exists as a *separate*, published structure for the
fields worth exposing that way).

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
