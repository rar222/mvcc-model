#!/usr/bin/env python3
"""Fails if extern_template_demo.cpp.o -- the CALLER, not the generated
types_extern.cpp.o -- had to define any of its own copy of a
Snapshot/Transaction/Model/BulkTransaction/CommitResult/View entry point.
That's the compiled-output proof that scripts/gen_extern_templates.py is
missing coverage for something example/types.h now uses; see that script's
own "HOW TO CHECK IF THIS SCRIPT IS MISSING A SYMBOL" docstring section for
the manual version of this same check and the full reasoning behind it
(weak/vague vs. GNU_UNIQUE linkage, why a class needs its own `extern
template class ...;` line and not just its returning function, etc.) --
this script is that check, automated and given a fixed allowlist for the
noise that's expected to survive it regardless.

ONLY MEANINGFUL AT -O2 (the "default" CMake preset). At -O0 (debug/asan/tsan)
the compiler doesn't inline as aggressively, so things like type_tag<T>()'s
call chain (see ALLOWED_PATTERNS below) would show up as full weak function
symbols instead of collapsing to just their static-local's address -- a
different, unverified floor. CMakeLists.txt only registers this test when
CMAKE_BUILD_TYPE is RelWithDebInfo with neither sanitizer on.
"""

import argparse
import re
import subprocess
import sys

# Everything below is allowed to show up as a locally-defined weak (W),
# weak-object (V), or GNU_UNIQUE (u) symbol in the CALLER's object file --
# confirmed, one at a time, against this project's actual default-preset
# build. Not a template entry point this script's coverage could (or
# should) apply to; see gen_extern_templates.py's docstring for why each
# category is fine.
ALLOWED_PATTERNS = [
    # Destructors of the model's own movable-only types -- not template
    # entry points at all, see extern_template_demo.cpp's "Measured" comment.
    r'^model::CommitResult::~CommitResult\(\)$',
    r'^model::Snapshot::~Snapshot\(\)$',
    r'^model::Transaction::~Transaction\(\)$',
    # std:: library template instantiation, incidentally mentioning a
    # model:: type in its own template argument.
    r'^std::_Vector_base<model::View<example::\w+>, '
    r'std::allocator<model::View<example::\w+> > >::~_Vector_base\(\)$',
    # PersistentSet<Id, IdHash>'s implementation -- never parameterized on
    # the user type T, so it doesn't fit this generator's per-T loop at all
    # (and emitting it once per type would be a duplicate explicit
    # instantiation error). See gen_extern_templates.py's docstring.
    r'^model::pmap::detail::TrieCore<model::Id, model::Id, model::IdHash, '
    r'model::pmap::detail::IdentityKeyOf<model::Id> >::iterator::advance\(\)$',
    # type_tag<T>()/Snapshot::cast<T> are tiny one-liners the optimizer
    # inlines straight into hot range-for loops (e.g. advance_to_match() ->
    # cast<ClassT>() -> type_tag<ClassT>()) -- desirable, it's why range-for
    # beats a callback here. cast<T> leaves no trace once inlined;
    # type_tag<T>()'s `static const char anchor` needs real, ODR-unique
    # storage for its address even after the call around it is gone.
    # extern template coverage wouldn't change this either way (inlining
    # happens regardless of whether a declaration exists).
    r'^model::type_tag<example::\w+>\(\)::anchor$',
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--object', required=True, help='path to extern_template_demo.cpp.o')
    ap.add_argument('--nm', default='nm', help='nm binary to use (default: nm)')
    args = ap.parse_args()

    out = subprocess.run(
        [args.nm, '-C', '--defined-only', args.object],
        capture_output=True, text=True, check=True,
    ).stdout

    allowed = [re.compile(p) for p in ALLOWED_PATTERNS]
    offenders = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        _addr, sym_type, name = parts
        if sym_type not in ('W', 'V', 'u'):
            continue
        if 'model::' not in name:
            continue
        if any(p.match(name) for p in allowed):
            continue
        offenders.append(name)

    if not offenders:
        print(f"OK: no missing extern template coverage in {args.object}")
        return 0

    print(f"FAIL: {args.object} defines its own copy of the following "
          f"entry point(s), instead of resolving them from "
          f"generated/types_extern.cpp.o:\n", file=sys.stderr)
    for name in sorted(set(offenders)):
        print(f"  {name}", file=sys.stderr)
    print(
        "\nThis means scripts/gen_extern_templates.py is missing coverage "
        "for one of these -- most often because a new model.h entry point "
        "was added, or an existing one's returned class (a *Range<Field>-"
        "style nested class) never got its own `extern template class "
        "...;` line alongside the function that returns it. See CLAUDE.md's "
        "\"Working style in this repo\" and gen_extern_templates.py's own "
        "\"HOW TO CHECK IF THIS SCRIPT IS MISSING A SYMBOL\" docstring "
        "section for how to find and add the missing render_entries() "
        "line(s), then also add a real call to the new entry point in "
        "examples/extern_template_demo.cpp -- a declaration alone doesn't "
        "prove coverage, only a call that this test then verifies does.",
        file=sys.stderr,
    )
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
