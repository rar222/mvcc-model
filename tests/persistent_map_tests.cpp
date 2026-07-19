#include "model/persistent_map.h"
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
using namespace model::pmap;

namespace {

// A key type distinct from std::string, to prove PersistentMap's key isn't
// hardcoded -- mirrors how model::IdHash hashes model::Id: a stateless
// functor, no scrambling, just enough to exercise the K/Hash template
// parameters end to end. Not model::IdHash itself: this file stays
// dependency-free of model.h, matching persistent_map.h's own independence
// from model::Id.
struct U64Hash {
    std::uint64_t operator()(std::uint64_t k) const noexcept { return k; }
};

}  // namespace

int main() {
    std::mt19937 rng(99);
    PersistentMap<std::string, int, StringHash> pm;
    std::unordered_map<std::string,int> ref;

    int fails = 0;
    auto check_equal = [&](const char* where){
        if (pm.size() != ref.size()){ std::printf("SIZE MISMATCH at %s: pm=%zu ref=%zu\n", where, pm.size(), ref.size()); ++fails; }
        for (auto& [k,v] : ref){ const int* p = pm.get(k); if(!p||*p!=v){ std::printf("GET MISMATCH at %s key=%s\n", where, k.c_str()); ++fails; return; } }
        size_t seen=0; pm.for_each([&](const std::string&k,int v){ auto it=ref.find(k); if(it==ref.end()||it->second!=v){ std::printf("ITER EXTRA at %s key=%s\n", where, k.c_str()); ++fails; } ++seen; });
        if (seen != ref.size()){ std::printf("ITER COUNT at %s: %zu vs %zu\n", where, seen, ref.size()); ++fails; }
    };

    // Heavy random churn, differential against unordered_map.
    for (int i=0;i<20000 && fails==0;i++){
        std::string k = "k" + std::to_string(rng()%2000);
        if (rng()%3){ int v=rng(); pm=pm.set(k,v); ref[k]=v; }
        else { pm=pm.erase(k); ref.erase(k); }
        if (i%500==0) check_equal("churn");
    }
    check_equal("final");

    // Structural sharing / persistence: an old version must be unaffected by
    // later edits to a derived version.
    PersistentMap<std::string, int, StringHash> a;
    for (int i=0;i<1000;i++) a=a.set("s"+std::to_string(i), i);
    PersistentMap<std::string, int, StringHash> b = a;
    for (int i=0;i<1000;i++) b=b.set("s"+std::to_string(i), i+100000);
    b = b.set("newkey", 7);
    b = b.erase("s500");
    // a must be pristine
    for (int i=0;i<1000;i++){ const int* p=a.get("s"+std::to_string(i)); if(!p||*p!=i){ std::printf("PERSIST FAIL: a[s%d] changed\n", i); ++fails; break; } }
    if (a.get("newkey")){ std::printf("PERSIST FAIL: newkey leaked into a\n"); ++fails; }
    if (!a.get("s500")){ std::printf("PERSIST FAIL: s500 erased from a\n"); ++fails; }
    if (a.size()!=1000){ std::printf("PERSIST FAIL: a.size=%zu\n", a.size()); ++fails; }
    if (b.size()!=1000){ std::printf("PERSIST FAIL: b.size=%zu\n", b.size()); ++fails; }

    std::printf("\npersistent map (string-keyed): %s (fails=%d)\n", fails? "FAIL":"OK", fails);

    // Same differential churn, but keyed on std::uint64_t via U64Hash -- the
    // shape by_type_/by_cached_reference_ need (see model.h's Root), with no
    // string round-trip anywhere.
    int fails_u64 = 0;
    PersistentMap<std::uint64_t, int, U64Hash> pm2;
    std::unordered_map<std::uint64_t,int> ref2;
    auto check_equal_u64 = [&](const char* where){
        if (pm2.size() != ref2.size()){ std::printf("U64 SIZE MISMATCH at %s: pm=%zu ref=%zu\n", where, pm2.size(), ref2.size()); ++fails_u64; }
        for (auto& [k,v] : ref2){ const int* p = pm2.get(k); if(!p||*p!=v){ std::printf("U64 GET MISMATCH at %s key=%llu\n", where, (unsigned long long)k); ++fails_u64; return; } }
        size_t seen=0; pm2.for_each([&](std::uint64_t k,int v){ auto it=ref2.find(k); if(it==ref2.end()||it->second!=v){ std::printf("U64 ITER EXTRA at %s key=%llu\n", where, (unsigned long long)k); ++fails_u64; } ++seen; });
        if (seen != ref2.size()){ std::printf("U64 ITER COUNT at %s: %zu vs %zu\n", where, seen, ref2.size()); ++fails_u64; }
    };
    for (int i=0;i<20000 && fails_u64==0;i++){
        std::uint64_t k = (std::uint64_t)(rng()%2000);
        if (rng()%3){ int v=rng(); pm2=pm2.set(k,v); ref2[k]=v; }
        else { pm2=pm2.erase(k); ref2.erase(k); }
        if (i%500==0) check_equal_u64("churn");
    }
    check_equal_u64("final");
    std::printf("persistent map (uint64-keyed): %s (fails=%d)\n", fails_u64? "FAIL":"OK", fails_u64);
    fails += fails_u64;

    // PersistentSet<uint64_t>: same differential-churn discipline, against
    // std::unordered_set instead of std::unordered_map -- no value half to
    // check, just membership.
    int fails_set = 0;
    PersistentSet<std::uint64_t, U64Hash> ps;
    std::unordered_set<std::uint64_t> refs;
    auto check_equal_set = [&](const char* where){
        if (ps.size() != refs.size()){ std::printf("SET SIZE MISMATCH at %s: ps=%zu ref=%zu\n", where, ps.size(), refs.size()); ++fails_set; }
        for (std::uint64_t k : refs){ if(!ps.contains(k)){ std::printf("SET CONTAINS MISMATCH at %s key=%llu\n", where, (unsigned long long)k); ++fails_set; return; } }
        size_t seen=0; ps.for_each([&](std::uint64_t k){ if(refs.find(k)==refs.end()){ std::printf("SET ITER EXTRA at %s key=%llu\n", where, (unsigned long long)k); ++fails_set; } ++seen; });
        if (seen != refs.size()){ std::printf("SET ITER COUNT at %s: %zu vs %zu\n", where, seen, refs.size()); ++fails_set; }
    };
    for (int i=0;i<20000 && fails_set==0;i++){
        std::uint64_t k = (std::uint64_t)(rng()%2000);
        if (rng()%3){ ps=ps.insert(k); refs.insert(k); }
        else { ps=ps.erase(k); refs.erase(k); }
        if (i%500==0) check_equal_set("churn");
    }
    check_equal_set("final");

    // Structural sharing / persistence, same discipline as the map case above.
    PersistentSet<std::uint64_t, U64Hash> sa;
    for (std::uint64_t i=0;i<1000;i++) sa=sa.insert(i);
    PersistentSet<std::uint64_t, U64Hash> sb = sa;
    for (std::uint64_t i=0;i<1000;i++) sb=sb.erase(i);  // empty sb entirely
    sb = sb.insert(9999);
    // sa must be pristine
    for (std::uint64_t i=0;i<1000;i++){ if(!sa.contains(i)){ std::printf("SET PERSIST FAIL: sa missing %llu\n", (unsigned long long)i); ++fails_set; break; } }
    if (sa.contains(9999)){ std::printf("SET PERSIST FAIL: 9999 leaked into sa\n"); ++fails_set; }
    if (sa.size()!=1000){ std::printf("SET PERSIST FAIL: sa.size=%zu\n", sa.size()); ++fails_set; }
    if (sb.size()!=1){ std::printf("SET PERSIST FAIL: sb.size=%zu\n", sb.size()); ++fails_set; }

    std::printf("persistent set (uint64-keyed): %s (fails=%d)\n", fails_set? "FAIL":"OK", fails_set);
    fails += fails_set;

    return fails?1:0;
}
