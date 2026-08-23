# CLAUDE.md

Instructions for Claude Code working in this repo. Read this before changing anything
under `include/` or `src/`.

## What this is

A snapshot-isolated object model in C++20, with **multi-writer optimistic concurrency**:
**many writer threads, many reader threads**, objects referencing each other **by key**,
readers taking **O(1) immutable snapshots**, writers building a **local `Transaction`** and
racing `try_commit()`, and subscribers receiving **coalescing change events**.

This is a sibling project to `../snapshot-model`, which is the single-writer version. Read
sides are identical; the write side was redesigned from scratch. `DESIGN.md` has the
reasoning, especially the "Multi-writer optimistic concurrency" section. This file has the
rules.

Target scale, which is what justifies every unusual decision below:

| | |
|---|---|
| Objects | 100k – 1M |
| Commits | 10 – 100 / sec, from any number of writer threads |
| Churn | constant creates and deletes, not just field updates |
| Reads | many threads, far more frequent than writes |

## Build, run, test

```bash
cmake --preset default && cmake --build --preset default -j
ctest --preset default            # tests + the demo as an integration test
./build/default/demo              # multi-writer demo

cmake --preset asan && cmake --build --preset asan -j && ctest --preset asan
cmake --preset tsan && cmake --build --preset tsan -j && ctest --preset tsan
```

**A change to the model is not done until `ctest --preset asan` and `ctest --preset tsan`
both pass.** TSan matters more here than in the single-writer sibling: it's the preset that
actually exercises concurrent `try_commit()` calls racing each other. The default preset
deliberately keeps `assert` enabled (no `-DNDEBUG`), because `Snapshot::resolve()`'s assert
is the tripwire for the central invariant. Do not add `NDEBUG` to the default preset.

## Layout

```
include/model/model.h   Id, ObjectBase, Chunk, Root, Snapshot, Transaction, Model, CommitResult
src/model.cpp           all of the implementation
include/example/types.h example user types (Account, Order) -- NOT part of the model
examples/demo.cpp       concurrent demo: 3 writer threads racing try_commit(), 2 readers, 1 slow subscriber
examples/commit_bench.cpp   commit latency vs. size, and throughput vs. writer-thread count
tests/tests.cpp         dependency-free harness (no gtest/Catch2 -- keep it that way)
```

## The invariants. Do not break these.

1. **`Ref<T>` never dangles inside a snapshot.** A non-nullable reference stored in an
   object always resolves in any snapshot containing that object. Enforced by
   `Model::validate()` inside `try_commit()`'s apply phase, against **the latest committed
   state, not the transaction's base** — checked by the assert inside `Snapshot::resolve()`.
   The reader's guarantee is exactly as strong as that check and no stronger.
2. **`Opt<T>` may be null, and only becomes null via cascade or an explicit write.**
   Nullability is a property of the *field type*, not of a runtime flag: `RefNuller` has
   no overload that can clear a `Ref<T>`, so a non-nullable field is uncleanable by
   construction. Do not add one.
3. **Published state is immutable.** Once a `Root` is handed to a reader, nothing in it
   — spine, chunks, objects — is ever mutated. `try_commit()`'s apply step clones.
4. **Nothing is freed while a reader (or an open Transaction's base) can still see it.**
   Objects are retired with the version at which they became invisible, and freed only once
   no live snapshot — including any `Transaction::base()` — is older than that. Getting this
   wrong is a use-after-free that only ASan will catch.
5. **`Id` carries a generation.** Slots are recycled constantly. A stale `Id` must never
   silently resolve to whatever object landed in that slot next.
6. **A `View<T>` must not outlive its `Snapshot`, and must never be stored.** It holds the
   snapshot by pointer. Stack only: build it, traverse, drop it. A `View` in a member or a
   container is a fat ref and will pin a version forever.
7. **`commit_mu_`-protected state (`referrers_`, `by_type_`, `by_field_`, `spine_`,
   `free_slots_`, `next_slot_`, `changelog_`, ...) is touched ONLY from inside
   `try_commit()`, by whichever thread currently holds `commit_mu_`.** This is exactly the
   single-writer sibling's writer-private state, in the same shape — it's now protected by
   an actual mutex instead of being single-threaded by convention, which is strictly safer,
   not a rewrite. Do not read or write it from anywhere else, and do not add a second lock
   that could be held instead of `commit_mu_` while touching it.
8. **Cascade delete is resolved once, at commit time, never eagerly.**
   `Transaction::remove()` only records an intent (a real `Id`, added to
   `remove_intents_`) or, for a same-transaction local create, cancels it outright when
   nothing else pending references it — a still-referenced local create is instead
   deferred (`local_remove_intents_`): it installs at apply time and is then removed by
   the same commit-time BFS, so local and committed removes share one cascade semantics.
   The actual BFS against `referrers_` runs inside `try_commit()`'s serialized apply phase
   (`Model::remove_raw`), the one place `referrers_` is safe to read. Do not make
   `Transaction::remove()` resolve cascade fan-out itself — that would require a
   transaction-local reverse index, which is exactly the "shared mutable structure many
   transactions touch" problem this whole design exists to avoid. (`remove_impl`'s
   referenced-or-not scan is a one-shot linear walk of the transaction's own overlay, not
   an index, and it decides only WHERE the remove resolves, never what it fans out to.)
   See DESIGN.md.
9. **Reconciliation of `referrers_`/`by_field_` for an update must run AFTER the object is
   fully written, never before.** `Transaction::update()` hands back a mutable pointer to a
   clone the caller may write to any number of times, in any order, before commit;
   `Model::apply_update()` reconciles once, at apply time, against that final value. The
   cascade BFS's own nulling step (`Model::clone_for_cascade_null` + `null_ref()`) follows
   the same rule: reconcile is called by the caller, immediately after `null_ref()`, never
   inside the clone/install helper itself. Reconciling before the mutation lands is a
   guaranteed no-op that leaves a phantom (or missing) edge in the reverse index — this
   exact bug class has already been hit once in the single-writer sibling project; don't
   reintroduce it here.
10. **Lock order: `commit_mu_` → `ver_mu_` → `reap_mu_`, never reversed** — except the
    reaper's own `reap_mu_` → `ver_mu_`, and `release_version`, which takes them
    sequentially (not nested) to avoid the cycle. `snapshot()` and `Transaction`-building
    never take `commit_mu_` at all — that's what keeps transaction building fully
    contention-free with respect to committing.

## Things that look like improvements but are not

- **"Why not use `shared_ptr<const ObjectBase>` in `Chunk` instead of raw pointers?"**
  Because copying a chunk would then mean 256 atomic refcount increments. At 1M objects
  and 100 commits/sec with scattered writes, that is millions of atomic RMWs per second
  and it is what kills the design. Chunks hold raw pointers so a chunk copy is a memcpy.
  Lifetime is handled by the version watermark instead. **Do not "simplify" this.**
- **"Why not let every transaction apply independently, in parallel, and merge the
  results?"** That requires `referrers_` (and every other index) to become a mergeable,
  per-transaction structure — real complexity, and it reintroduces exactly the shared-
  mutable-state problem the single-writer sibling avoided by having only one writer. This
  design instead makes transaction *building* free of shared state entirely, and serializes
  only the (comparatively cheap, change-proportional) apply step behind `commit_mu_`. See
  DESIGN.md's "What this buys, and what it costs."
- **"Why not resolve cascade fan-out inside `Transaction::remove()`, so the caller finds out
  immediately?"** Because that requires reading `referrers_` outside of `commit_mu_`, which
  is either a race (reading concurrently-mutating shared state) or a second lock protecting
  it (contention on every `remove()` call, not just every commit). Deferring to apply time,
  behind the one lock that's already there, costs nothing extra and stays correct. See
  invariant 8.
- **"Why not full serializable OCC (validate every read, not just every write)?"** The
  user's own specification for this design is object-write-set OCC: "a commit succeeds if no
  one touched the same ids and every `Ref<T>` is still valid." That is a real, useful,
  cheaper isolation level — not a bug to be fixed toward serializability. Document the
  write-skew possibility; don't silently strengthen (or weaken) the guarantee.
- **"Why not let `Model::create/update/remove/commit` coexist with `Transaction`, for
  backward compatibility with code written against the single-writer API?"** Two APIs
  writing the same shared state is how invariant 7 gets violated by accident. There is no
  single-writer caller in this project to be backward-compatible with — it's a fresh
  project. Don't add it.
- **"Why not make `Ref<T>` self-dereferencing (`ref->field`)?"** A fat ref would have to
  carry a snapshot pointer, so every stored ref would silently pin a snapshot alive and
  stall the reaper. `Ref<T>` is 8 bytes; you resolve through a snapshot. Deliberate.
- **"Why a hand-rolled `TypeTag` instead of `dynamic_cast`?"** The tag check is one
  virtual call plus a pointer compare, and it sits on the read path that runs millions of
  times a second. `dynamic_cast` is not in that budget.
- **"Why not make `View<T>` hold its `Snapshot` by value, so it can't dangle?"** Because
  every traversal hop would then copy a `shared_ptr` -- two atomic RMWs on the hottest path
  in the system. A stored view is also a fat ref: it pins its version and stalls the
  reaper. `View` is scoped, 16 bytes, by pointer. **Do not "fix" this.**
- **"Why not let a slow subscriber's queue grow?"** Every queued event pins a snapshot,
  and every pinned snapshot pins the objects retired since. An unbounded queue is a
  memory leak with a slow fuse. Overflow coalesces; see `Subscription::collapse`.
- **"Why not a `VeryCoarse` tier below `Coarse`, for even more memory savings?"** It existed
  (`LookupType`/`KeyLookupType::VeryCoarse`, a 10-bit/1,024-bucket routing cap) and was
  removed after measurement found no scale at which it earns its keep. Its bucket ceiling
  is fixed regardless of object count, so once a field's cardinality is high enough for real
  merging, structural savings plateau (buckets max out at 1,024 either way) while the read
  cost keeps climbing with cardinality -- measured up to 40x `Coarse`'s own read time for a
  memory edge of only 7-21%. For `by_key_` (keys, never bucket-merged, cardinality forced to
  equal object count) it's worse: `Coarse` and `VeryCoarse` converge on nearly identical
  memory once both saturate their routing tree, while `VeryCoarse`'s collision-chain walk
  cost explodes -- measured 219x `Exact`'s read time at 500,000 objects, average chain
  length 488. `Coarse` alone remains a real, defensible tradeoff in both cases. See
  `examples/lookup_type_tuning_demo.cpp` and `examples/key_lookup_type_tuning_demo.cpp` to
  re-run the numbers if this is ever reconsidered.
- **Don't add gtest/Catch2.** The harness in `tests/tests.cpp` is deliberately
  dependency-free so the project builds anywhere with no network.

## Known scope boundaries (not TODOs to silently "fix")

- Object-write-set OCC, not full serializable OCC (see above).
- `try_commit()`'s apply step is fully serialized behind `commit_mu_`. Transaction building
  is fully parallel; applying is not. `examples/commit_bench.cpp` has a thread-count sweep
  that demonstrates this — expect sub-linear scaling, not linear.
- The reverse index (`referrers_`) still scans linearly per target (`vector<RefEdge>` with
  `find_if`), same as the single-writer sibling. A hub object with many referrers makes each
  mutate of one of them O(referrer count). Unaffected by this project's redesign; would
  benefit both projects equally if fixed.
- Cascade fan-out from a single `remove()` intent is still unbounded and invisible to the
  caller until `try_commit()` returns. A two-phase plan/inspect/apply API is the natural fix,
  same as the single-writer sibling's own TODO list.

## Working style in this repo

- **No C++ exceptions. None.** The whole project builds with `-fno-exceptions` (enforced in
  CMakeLists.txt), so `throw` and `try`/`catch` are compile errors — don't reintroduce them
  "just locally", in tests, or in example code. Failure is reported by value:
  `try_commit()` returns `CommitStatus::Conflict` / `Vetoed` / `Invalid` (with
  `CommitResult::error` carrying the `Model::IntegrityError` details), and internal apply
  helpers return `std::optional<Model::IntegrityError>`. A pre-commit hook signals "no" by
  returning `false`; under `-fno-exceptions` a throw from anywhere (including the standard
  library, e.g. `bad_alloc`) is `std::terminate` — loud, by design.
- **When adding a ref field to a type, update `define_references` (both directions: read and
  null) together**, keep field tags stable, and add a cascade test. A ref that
  `define_references` doesn't report is invisible to the reverse index AND to `RefRemapper`
  — integrity breaks silently, and a local-id create can also silently fail to remap.
- **Every new invariant gets a test in `tests/tests.cpp`.** The concurrency stress test
  (`concurrent_stress_many_writer_threads_hammering_try_commit_never_corrupts_referrers_or_leaks_a_dangling_ref`)
  is the one that matters most; extend it rather than writing a new one where you can.
- **A `CommitResult` with `status != Committed` means the `Transaction` you passed in is
  spent** — its local overlay was already moved from during apply. Don't try to reuse it;
  `begin()` a fresh one.
- **Prefer failing loudly.** This code would rather reject a commit as
  `CommitStatus::Invalid` (or fail an assert) than publish a broken graph to readers.
- **A new template entry point on `Snapshot`/`Transaction`/`Model`/`BulkTransaction`/
  `CommitResult`/`View` templated only on `<class T>` or `<auto Field>` (no `Pred`/`F`
  functor parameter) must be added to `scripts/gen_extern_templates.py`'s
  `render_entries()`, and exercised with a real call in `examples/extern_template_demo.cpp`**
  — a declaration alone doesn't prove the generated `extern template` matches actual usage.
  When adding one, also grep the rest of `model.h` for sibling entry points in the same
  family that might already be missing rather than fixing only the one that prompted the
  change — `CommitResult::to_real<T>` was found missing this way after the `range_*` family
  was added. Functions templated on an unbounded caller type (`for_each_*`/`all_of_*`/
  `find_by_predicate`/`view_by_predicate`, all taking `Pred`/`F`) are correctly excluded:
  there's no fixed `Pred` set to enumerate. After changing the generator or the demo, rebuild
  `extern_template_demo` and run `ctest --preset default` to confirm it still compiles,
  links, and passes.


## Code and comments

- Comments are a last resort: write one only when an experienced engineer would be
  surprised or misled by the code alone — never as narration, history, or a note to
  your future self (that belongs in the commit message, or nowhere).
- Any comment you do write is timeless: present tense, states what's currently true,
  no "now"/"previously". It reads like a spec, not a commit message.
- Within a function: one focused line. A second only if the why genuinely can't fit
  in one, never a third — needing more means the code needs restructuring.

## Prose

These rules govern every word a human will read: conversational turns, commit messages, PR descriptions, docs, reports, and any text you relay from a subagent. Relayed text is your prose; rewrite it to comply. They do not govern code, tool calls, or the instructions you send to a subagent.

- **Purpose.** The reader should spend their attention on the problem, never on decoding your prose.

- **Standard** Try to write prose close to the ISO 24495-1:2023 standard.

- **Order.** The first line must include the conclusion, the decision you need, or the answer to the question asked. Evidence, reasoning and caveats after - for the whole response, not just the sentence. If the reader must choose something, the choice comes before the findings that motivated it, however much those findings feel like setup. Stop when the evidence runs out.

- **Sentences.** One claim per sentence. At most one interpolation: a dash pair, a parenthesis, or a subordinate qualification, not two or three stacked. If a qualification matters, give it its own sentence. If it doesn't, drop it.

- **Words.** Use the most direct, familiar vocabulary available. Jargon only where it is the precise domain standard. Every term must already exist in the reader's world - in the codebase, the ticket, the domain, or this conversation - or be defined at first use. Never coin a term. Never surface a name that came out of your own reasoning, a plan you wrote, or a subagent's output without saying what it means.

- **References.** A pointer is not information. Naming a section, a decision label, a phase or "the fix" tells the reader nothing unless you restate the claim inline. Assume they have no document open beside your text.

- **Structure.** Structure follows the content's actual shape; it is not a template. Do not open by announcing a count. Emphasis is scarce or it means nothing: no bold lead-in on every paragraph, no headings unless the response has genuinely separable sections. A short answer needs no structure at all.

- **Tone.** Write as one who has understood the thing and is explaining it to an equal whose time matters. Understated, direct, active voice. When the subject is difficult, hold the tone steady: do not warm it to coddle, do not cool it to detach. Deliver truths completely and plainly.

- **Calibration.** Self-protective qualification is forbidden. Actual uncertainty is required: state your confidence and its basis once, plainly, and never convert a conclusion drawn from reading into one drawn from evidence.

- **Form before prose.** A finding is a line, not a paragraph. Per-item results and decisions go in a table or one bullet each. Paragraphs are for a single argument that cannot be tabulated.

- **Cut.** A response fits on one screen — past that, you're handing over your working, not summarizing. Send the conclusion and what it changes; hold evidence, alternatives, and verification detail until asked. Delete every sentence whose absence wouldn't change what the reader does or believes, hardest against: reasoning behind a recommendation already made, what you checked and found fine, verification inventories, non-material work, unasked questions.

- **Held, not sent.** Do the whole task and verify it fully, then say less about it — correct before complete, complete before brief. Cut detail isn't lost; name what you're holding in one line. Never buy brevity by omitting a risk, failure, caveat, or something you didn't do.

- Forbidden: hype, praise, performative empathy, apology, softeners, politeness rituals, hedging, qualifications that carry no weight. Intros, outros, transitions, recaps of the question. Antithesis for rhythm ("X, not Y"), and "the real" or "the actual" as intensifiers. Narrating what the reader can already see in a diff or on screen — but a verdict drawn from evidence they cannot see is required, and they have not read your tool output.

- Ask clarifying questions whenever the answers materially improve the work. We are colleagues; ask as many as are genuinely useful, batched when natural. Never ask to stall.
