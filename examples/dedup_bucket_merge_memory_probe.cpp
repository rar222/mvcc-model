// Ad hoc RSS measurement for the string-key-dedup (pmap::DedupMap) and
// cached-field bucket-merge (by_cached_field_merged_) work: same fork() +
// /proc/self/status VmHWM technique as tests/performance_tests.cpp's
// PERF_HAS_MEMORY_SECTION helpers (peak_rss_kb/measure_child_peak_kb),
// reused here rather than reimplemented. Not a permanent CI assertion --
// there's no "expected" number to assert against across two different
// checkouts of the tree -- just a standalone program meant to be built and
// run once against a baseline checkout (git stash) and once against a
// changed one (git stash pop), diffing VmHWM by hand.
//
// One type, four fields, each isolating exactly one LookupType/
// KeyLookupType path:
//   key_exact        -- by_key_,          KeyLookupType::Exact
//   field_exact      -- by_cached_field_,  LookupType::Exact   (DedupMap, unaffected by bucket-merge)
//   field_coarse     -- by_cached_field_merged_, LookupType::Coarse
//   field_verycoarse -- by_cached_field_merged_, LookupType::VeryCoarse
// Every field's value is unique per object, so leaf/bucket counts are
// directly comparable across fields at the same N.

#include "model/model.h"

#if defined(__linux__)
#include <malloc.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#define PROBE_HAS_MEMORY_SECTION 1
#else
#define PROBE_HAS_MEMORY_SECTION 0
#endif

#include <cstdio>
#include <memory>
#include <string>

using namespace model;

namespace {

class MeasureThing final : public Object<MeasureThing> {
public:
    std::string key_exact;
    std::string field_exact;
    std::string field_coarse;
    std::string field_verycoarse;

    template <class Self>
    static void define_keys(Self& s, const FieldKeyReader& v) {
        v.key<&MeasureThing::key_exact>(s.key_exact, "key_exact");
    }

    template <class Self>
    static void define_fields(Self& s, const LookupFieldReader& v) {
        v.field<&MeasureThing::field_exact>(s.field_exact, LookupType::Exact, "field_exact");
        v.field<&MeasureThing::field_coarse>(s.field_coarse, LookupType::Coarse, "field_coarse");
        v.field<&MeasureThing::field_verycoarse>(s.field_verycoarse, LookupType::VeryCoarse,
                                                 "field_verycoarse");
    }
};

void build(Model& m, int n) {
    m.set_max_undo_list_size(0);
    Transaction txn = m.begin();
    for (int i = 0; i < n; ++i) {
        auto o = std::make_unique<MeasureThing>();
        o->key_exact = "k" + std::to_string(i);
        o->field_exact = "f" + std::to_string(i);
        o->field_coarse = "c" + std::to_string(i);
        o->field_verycoarse = "v" + std::to_string(i);
        txn.create(std::move(o));
    }
    m.try_commit(txn);
}

void report_structural(const Model& m) {
    const auto d = m.slot_stats_diagnostics();
    auto line = [](const char* name, const pmap::SlotStats& s) {
        std::printf("  %-16s nodes=%9zu leaves=%9zu capacity(sum)=%10zu\n", name, s.node_count,
                    s.leaf_count, s.total_capacity);
    };
    std::printf("Structural (Model::slot_stats_diagnostics(), all 4 fields combined):\n");
    line("by_key", d.by_key);
    line("by_cached_field", d.by_cached_field);
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 500'000;
    std::printf("N = %d MeasureThing objects\n", n);

#if PROBE_HAS_MEMORY_SECTION
    // See performance_tests.cpp's measure_child_peak_kb for why: trims this
    // process's own already-freed glibc arena pages back to the OS before
    // forking, so the child's peak RSS reflects its own real demand instead
    // of quietly reusing a leftover free pool via COW.
    malloc_trim(0);
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        Model m;
        build(m, n);
        report_structural(m);
        FILE* f = std::fopen("/proc/self/status", "r");
        long kb = -1;
        if (f) {
            char line[256];
            while (std::fgets(line, sizeof line, f)) {
                if (std::sscanf(line, "VmHWM: %ld kB", &kb) == 1) break;
            }
            std::fclose(f);
        }
        std::printf("VmHWM (this process, peak RSS) = %ld kB (%.2f MB)\n", kb, kb / 1024.0);
        std::fflush(nullptr);  // _exit() skips the normal flush-on-exit; this child's own
                               // post-fork output would otherwise be silently discarded.
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
#else
    std::printf("PROBE_HAS_MEMORY_SECTION not available on this platform -- structural stats only.\n");
    Model m;
    build(m, n);
    report_structural(m);
#endif
    return 0;
}
