#include "model/persistent_map.h"
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
using namespace model::pmap;

int main() {
    std::mt19937 rng(99);
    PersistentMap<int> pm;
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
    PersistentMap<int> a;
    for (int i=0;i<1000;i++) a=a.set("s"+std::to_string(i), i);
    PersistentMap<int> b = a;
    for (int i=0;i<1000;i++) b=b.set("s"+std::to_string(i), i+100000);
    b = b.set("newkey", 7);
    b = b.erase("s500");
    // a must be pristine
    for (int i=0;i<1000;i++){ const int* p=a.get("s"+std::to_string(i)); if(!p||*p!=i){ std::printf("PERSIST FAIL: a[s%d] changed\n", i); ++fails; break; } }
    if (a.get("newkey")){ std::printf("PERSIST FAIL: newkey leaked into a\n"); ++fails; }
    if (!a.get("s500")){ std::printf("PERSIST FAIL: s500 erased from a\n"); ++fails; }
    if (a.size()!=1000){ std::printf("PERSIST FAIL: a.size=%zu\n", a.size()); ++fails; }
    if (b.size()!=1000){ std::printf("PERSIST FAIL: b.size=%zu\n", b.size()); ++fails; }

    std::printf("\npersistent map: %s (fails=%d)\n", fails? "FAIL":"OK", fails);
    return fails?1:0;
}
