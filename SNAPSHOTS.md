# Snapshots, COW, and the HAMT

How a commit derives a new, immutable `Root` from the old one in time proportional to what
it *changed*, never to how big the model is — and how the old `Root` a reader is still
holding stays valid, and eventually gets freed, while that happens.

Two independent copy-on-write structures live inside every `Root`. A third mechanism — a
version watermark, not a structure — is what makes freeing either of them safe.

| Structure | What it stores | COW granularity | File |
|---|---|---|---|
| **The spine** | every object, by `Id` | one `Chunk` (256 slots) | `include/model/model.h` (`Root::spine`, `Chunk`) |
| **The HAMT tries** (`by_type`, `by_field`, `by_cached_field`, `by_cached_reference`) | every index over those objects | one trie node per 5-bit hash slice on the path to the touched entry | `include/model/persistent_map.h` |

Both follow the same rule: a commit clones only what's on the path from root to the thing it
touched. Everything else is a `shared_ptr` copy pointing at the exact memory the previous
`Root` pointed at.

---

## 1. The spine: an array of chunks

An `Id` is `{index, generation}`. The spine turns `index` into an object with two shifts and
two indexed loads — no hashing, no probing:

```
Id{ index = 0x0000_0301, gen = 7 }
                    |
        index >> 8  |  index & 0xFF     (kChunkBits = 8, kChunkSize = 256)
       +------------+------------+
       v                         v
   chunk = 3                  slot = 1
       |
       v
 spine[3] --> Chunk
              +-----------------------------------------+
              | obj[0] obj[1] obj[2] obj[3] ... obj[255] |
              | gen[0] gen[1] gen[2] gen[3] ... gen[255] |
              +-----------------------------------------+
                        ^
                        `-- slot 1: obj[1] is this Id's object,
                            IF gen[1] == 7 (else: stale, return null)
```

`Root::spine` (model.h:749) is `std::vector<shared_ptr<const Chunk>>` — a flat array of
pointers to fixed-size, immutable chunks. `Chunk` itself holds RAW pointers (`const
ObjectBase* obj[256]`), not `shared_ptr<const ObjectBase>` — a deliberate choice (see
CLAUDE.md's "Why not shared_ptr in Chunk") so that cloning a chunk is one `memcpy`, not 256
atomic refcount bumps. Object lifetime is handled entirely by the watermark in §3 instead.

### COW on the spine

A commit that touches ONE object clones ONE chunk (`Model::cow`, `src/model.cpp:395`) —
every other chunk pointer is copied as-is into the new `Root`:

```
                chunk 0     chunk 1     chunk 2     chunk 3
Root v1.spine:  [ C0 ]--+   [ C1 ]--+   [ C2 ]--+   [ C3 ]--+
                        |           |           |           |
                        |   shared_ptr<const Chunk>, same address
                        |           |           |           |
Root v2.spine:  [ C0 ]--+   [ C1 ]--+   [C2']       [ C3 ]--+
                                          ^
                                    freshly cloned:
                                    the commit touched
                                    a slot in chunk 2
```

`C2` (the pre-commit chunk) is not freed the instant `C2'` replaces it in the writer's
working spine — a `Snapshot` still holding `Root v1` points straight at `C2`, and a published
`Root` is never mutated (invariant 3). `C2` only becomes reclaimable once no live `Root`
anywhere still needs to look through it, which is what §3 tracks.

A commit re-clones a chunk at most once per attempt: `dirty_` (a set of chunk indices,
cleared per attempt) makes the first touch in `cow()` clone, and every later write in that
same commit mutate the clone in place.

---

## 2. The HAMT tries: path-copying a 32-ary trie

Every secondary index (`by_type`, `by_field`, `by_cached_field`, `by_cached_reference`) is a
`PersistentMap`/`PersistentSet` — a HAMT (hash array mapped trie): a trie keyed on 5-bit
slices of the entry's 64-bit hash, 32-ary at every level (`TrieCore`,
`include/model/persistent_map.h:83`). A real node holds a 32-bit bitmap (which of the 32
possible children are present) plus two parallel, densely packed vectors (`children`,
`leaves`) — no wasted slots for absent children, and a leaf holds one entry inline plus a
collision-chain link (`Leaf::next`), almost always null.

Simplified to 4-way branching below for the diagram — the real trie is 32-way, 5 hash bits
consumed per level:

```
                     root (bitmap: children present at 0, 2, 3)
                    +-----+-----+-----+
                    |  0  |  2  |  3  |
                    +--+--+--+--+--+--+
                       |     |     |
                       v     v     v
                   [leafA][leafB][leafC]
```

### Setting one key

`set_in()` (`persistent_map.h:229`) walks the trie from the root, one 5-bit slice at a time.
At each level it clones ONLY the node on the path to the target slot (`clone_node`,
`persistent_map.h:146`) and re-links it to the untouched, shared siblings:

```
   old root (v1)                          new root (v2) -- CLONED
  +-----+-----+-----+                    +-----+-----+-----+
  |  0  |  2  |  3  |                    |  0  |  2  |  3  |
  +--+--+--+--+--+--+                    +--+--+--+--+--+--+
     |     |     |                          |     |     |
     v     v     v                          |     v     |
 [leafA][leafB][leafC]                      |   [leafB'] <- CLONED, new value
                                             |             written here
                                             +-------------+------------> leafA, leafC:
                                                                           SAME shared_ptr
                                                                           as v1, untouched
```

Only the nodes on ONE root-to-leaf path are ever cloned — `O(log32 n)` nodes for `n`
entries, the same bound `DESIGN.md` cites for why deriving a new `Root` costs proportional
to the CHANGE, never to the index's size. A true collision (two distinct keys sharing a full
64-bit hash) pushes down into a fresh subtree, or — once hash bits run out — chains inside a
`Leaf`; editing a chain deep-copies its surviving links (`chain_copy`/`chain_set`,
`persistent_map.h:165-201`) rather than sharing them, because a shared tail link could
otherwise be mutated out from under an old, published version.

---

## 3. Keeping snapshots alive: the version watermark

Cloning old chunks and trie nodes away from the writer's working copy doesn't free them — it
only stops the WRITER from pointing at them. Freeing is a separate, asynchronous decision,
driven by one small map:

```
Model::live_   (version -> refcount)          Model::root_  (atomic: current published Root)
+---------+----------+
| version | refcount |            watermark = live_.begin()->first
+---------+----------+                        (the OLDEST version anyone still needs)
|    5    |    2     |  <-- 2 Snapshots (or open Transaction bases) still pinned at v5
|    7    |    1     |  <-- 1 pinned at v7
+---------+----------+
     (v6, v8, ... : no live readers -- simply absent from the map)
```

Every `Snapshot` (and every `Transaction`'s `base()`, which is itself a `Snapshot`) holds a
`Lease` (`src/model.cpp:65`) — an RAII handle that increments `live_[version]` in its
constructor and decrements it in its destructor (`Model::release_version`,
`src/model.cpp:829`). `Model::snapshot()` registers a version under `ver_mu_` in the SAME
critical section that loads `root_`, and `publish_now()` does the same for the version it
just published (`src/model.cpp:330-347`, `1345-1349`) — so a version is never observably
"current" before it's already un-freeable.

When a commit overwrites or removes something, the OLD copy is handed to the reaper tagged
with the version it stops being visible from:

```
commit publishes v6 -- some object X's slot got overwritten
Model::retire(X)  --->  retired_.push_back({ version_ + 1  /* = 6 */, X })
                                    |
                                    v  (handed off to the reaper on publish)
             reap_queue_: [ (invisible_from = 6, X), ... ]
```

The background reaper thread (`Model::reaper_loop`, `src/model.cpp:245`) wakes on new work or
a dropped `Lease`, reads the current watermark, and frees exactly what no live version can
still see:

```
watermark (min_live) = 5   -->  5 < 6   -->  X might still be visible (a v5 Snapshot could
                                              still resolve it) -- stays in reap_queue_

  ... the last v5 Snapshot drops; live_ loses the "5" entry; watermark advances to 7 ...

watermark (min_live) = 7   -->  7 >= 6  -->  no live version is old enough to need X
                                              -> delete X, drop it from reap_queue_
```

This is the ONE mechanism underneath both structures in §1 and §2. A retired `Chunk`'s old
version and an orphaned trie node are freed by two different low-level means — plain
`shared_ptr` refcounting for trie nodes (the last `Root` pointing at one drops it
automatically when that `Root` itself is destroyed) and the explicit `reap_queue_`/watermark
walk above for the raw-pointer `ObjectBase`s a `Chunk` holds — but both are only safe because
of the same guarantee: nothing disappears while `live_`'s minimum key is still old enough to
need it.

---

## 4. One commit, end to end

Putting §1–§3 together: a `try_commit()` that changes one `Order`'s `qty`, a field also
covered by `define_cached_fields()`:

```
 1. Model::cow(chunk_of(order.id))                      -- clone ONE Chunk               (S1)
       spine_[c] = make_shared<Chunk>(*spine_[c]);
       write the updated Order into the clone's slot

 2. by_cached_field_[qty_tag].set(old_qty_key, ...)      -- path-copy the OLD-value
    by_cached_field_[qty_tag].set(new_qty_key, ...)      -- and NEW-value trie buckets  (S2)
       clone only the nodes on each root-to-leaf path

 3. publish_now():
       r->spine = spine_;                      -- O(#chunks) shared_ptr copies, one is new
       r->by_cached_field = by_cached_field_;   -- O(#cached fields) shared_ptr copies
       root_.store(r);  live_[r->version]++;    -- new Root visible, already un-freeable

 4. whatever this commit orphaned (the pre-commit Chunk, any displaced trie node):
       Model::retire(old_object) -> reap_queue_.push({version_ + 1, old_object})
       -- freed later, once live_'s watermark passes that version               (S3)
```

Every step is bounded by what the commit actually changed, never by how many objects or
index entries the model holds — the guarantee `README.md`'s "O(1) immutable snapshot" and
`DESIGN.md`'s "proportional to the CHANGE" both rest on. It only holds because both
persistent structures clone along a path instead of copying whole, and because the watermark
defers freeing until every live reader has moved past the version that orphaned them.
