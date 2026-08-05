#!/usr/bin/env python3
"""Generate a full-coverage pair of `extern template` header + explicit-
instantiation source for every model::Object<T> type declared in an input
header -- the same technique include/example/types_extern.h(/.cpp) used to
demonstrate by hand, before CMakeLists.txt switched extern_template_demo
over to generating them from example/types.h at build time (see the
"extern_template_demo" section of CMakeLists.txt). Never committed: it's
regenerated into the build directory on every configure, from whatever
example/types.h currently says, so the two files can't drift out of sync
with it the way a hand-maintained list eventually will.

WHAT "FULL COVERAGE" MEANS, AND ITS ACTUAL COST
--------------------------------------------------
This emits EVERY entry point that is structurally reachable for each type,
not just ones something currently calls -- the opposite of a hand-curated
list. That trade is fine for --mode header (an unused `extern template`
DECLARATION costs the frontend a name lookup, nothing more -- measured:
no difference outside noise). It is NOT free for --mode source: an unused
explicit instantiation DEFINITION is still unconditionally emitted code in
that one object file (measured on this project's own Account/Order: +22%
.text for +37% more entries going from a hand-curated 52-entry list to
this script's 71-entry full coverage of the same two types). That's a
trade worth making for a demo binary; for a real project's real types
consider using this script's output as a starting draft to prune by hand
instead of wiring up the CMake generation step verbatim.

WHAT IT DERIVES, AND FROM WHERE
--------------------------------
Two kinds of entry points exist in model.h, and they need different
information:

1. Type-uniform entry points (resolve/find/create/update/remove/exists/
   peek/peek_as/peek_before/view/range/range_view/to_real, both Ref<T> and
   Opt<T> overloads where both exist) -- same set for every T that derives
   model::Object<T>, independent of T's fields. These need nothing but the
   list of types.

2. Field-dependent entry points -- these need each type's define_keys() /
   define_fields() / define_references() bodies, because that's the ONLY
   place (other than the field's own declared type) that says which fields
   participate in which lookup family (and, for define_fields()/
   define_references(), which LookupType -- Cache or Scan -- each one is
   tagged). A Ref<T>/Opt<T> data member that's declared but
   left out of define_references() is invisible to the model (see
   CLAUDE.md invariant 1 / types.h's own header comment) -- this script
   warns about that case instead of silently guessing.

HOW IT PARSES -- READ THIS BEFORE TRUSTING THE OUTPUT
--------------------------------------------------------
This is regex + brace-counting, not a C++ parser. It works because this
project's user types follow one consistent, narrow shape (see types.h):
one CRTP base (`class T final : public model::Object<T>`), one member
per line, `v.key<&T::field>(...)` (define_keys()) / `v.field<&T::field>(...)`
(define_fields()) / `field_tag<&T::field>()` (define_references()) call
forms inside the define_* functions, and
plain (non-templated-on-the-user-side) field types. It will silently
produce wrong or incomplete output for anything outside that shape:
macros, multiple inheritance, nested namespaces, key methods that take
arguments, fields spread across multiple base classes, etc. If you're
adapting this for a project whose types don't fit that shape, verify the
output (e.g. the way CMakeLists.txt does here: compile the generated
--mode source file and check it builds clean) before trusting it.

HOW TO CHECK IF THIS SCRIPT IS MISSING A SYMBOL
--------------------------------------------------
This script only has full coverage of model.h's entry points if every
function templated on JUST `<class T>` or `<auto Field>` (no `Pred`/`F`
functor parameter -- those are excluded on purpose, see render_entries()'s
docstring) is present in render_entries() below, AND every class template
one of those functions hands back (FieldRange<Field>,
CachedBucketRange<ClassT>, ReferrerRange<Field>, and their View siblings)
has its own `extern template class ...;` line too -- covering the
FUNCTION that returns a range is not enough; the range's own
begin()/end()/iterator::operator++ etc. are a separate template entity
that gets implicitly re-instantiated in every calling TU if its class
was never explicitly instantiated anywhere. A source-level grep for
"does render_entries() mention this name" cannot see that second kind of
gap -- it only shows up in what the compiler actually had to emit. So
trust compiled output, not source text (see CLAUDE.md's "Working style
in this repo" for when this check is required, not just suggested):

    # Build extern_template_demo, then ask its own object file what it
    # actually had to define for itself. Filter on nm's type column, not on
    # NAME: a Range<Field> leak's demangled name always embeds the concrete
    # type as part of Field (e.g. "Snapshot::FieldRange<&example::
    # Account::name>::..."), so excluding lines that mention "example::"
    # would silently hide exactly the gap this check exists to catch --
    # tried that, don't. `u` alongside `W`/`V` catches GNU_UNIQUE too: the
    # linkage a C++17 `inline` variable or non-template inline function
    # gets, a different but equally-real "this TU had to define its own
    # copy" signal (a plain `$2=="W"` filter would miss it):
    cmake --build --preset default --target extern_template_demo
    nm -C --defined-only \\
        build/default/CMakeFiles/extern_template_demo.dir/examples/extern_template_demo.cpp.o \\
      | awk '$2=="W" || $2=="V" || $2=="u"' | grep 'model::' | sort -u

    # Expected output: only the inherent floor this technique can't remove --
    # destructors/vtable/typeinfo of the concrete user types and Model's own
    # movable-only types (Account/Order/CommitResult/Snapshot/Transaction --
    # not template entry points at all, see extern_template_demo.cpp's own
    # "Measured" comment on why), std:: library template instantiations
    # (std::_Vector_base<...>::~_Vector_base, std::make_unique<T>),
    # model::type_tag<T>()::anchor ('u' -- this is why 'u' is in the filter
    # at all: type_tag<T>() and Snapshot::cast<T> are tiny one-liners called
    # from deep inside the *Range<Field> iterator machinery, e.g.
    # advance_to_match() -> cast<ClassT>() -> type_tag<ClassT>(); the
    # optimizer fully inlines that chain into main()'s hot range-for loops
    # -- desirable, it's the whole reason range-for beats a callback here.
    # cast<T> is pure logic and leaves no trace once inlined; type_tag<T>()
    # leaves exactly one thing behind, its `static const char anchor`,
    # because that variable's ADDRESS is the return value and must stay
    # ODR-unique, so it needs real storage even after the call around it is
    # gone. Adding an extern template line for type_tag<T>/cast<T> would
    # NOT change this -- extern template only suppresses generating an
    # out-of-line copy of code that ISN'T inlined; it has no effect on code
    # the optimizer inlines away regardless. Not a gap, just what inlining
    # a hot path looks like in nm output), and
    # model::pmap::detail::TrieCore<Id, Id, IdHash, IdentityKeyOf<Id>>
    # (PersistentSet<Id, IdHash>'s implementation) -- NOT out of scope
    # because it's "a different header's templates" in some vague sense,
    # but for a sharper reason: unlike FieldRange<Field> and friends,
    # this one is never parameterized on the user type T at all -- every
    # Snapshot-side range walks Ids through the exact same
    # PersistentSet<Id, IdHash>, regardless of whether the objects being
    # iterated are Account or Order. It doesn't fit this script's per-T
    # loop -- and if render_entries() tried to emit it once per type
    # anyway, that would be a DUPLICATE EXPLICIT INSTANTIATION (a compile
    # error in --mode source): the exact same specialization instantiated
    # twice in one program. Leaving it as an ordinary implicit (weak)
    # instantiation is correct here, not a gap -- if its repeated
    # per-TU compile cost across the whole project (not just this demo)
    # ever mattered enough to fix, the right place is one hand-written
    # `extern template class ...;` for it in model.h + one explicit
    # instantiation in src/model.cpp (which already holds "all of the
    # implementation" per CLAUDE.md), not a line generated per-T here.
    #
    # NONE of the above should be a Snapshot::/Transaction::/Model::/
    # BulkTransaction::/CommitResult::/View:: entry point this script
    # targets. If one shows up here -- most often a `*Range<...>` class
    # member, since it's easy to add the function and forget the class --
    # that's proof it (or its underlying class template) needs a
    # render_entries() line: a class needs its own
    # `extern template class Snapshot::FieldRange<...>;`-style line,
    # alongside the function's own `extern template ...` line, not instead
    # of it.
    #
    # OPTIONAL: to browse a TU's full contents rather than hunt for a
    # specific leak, name-based filtering is fine -- unlike above, you're
    # not trying to catch a Field-parameterized symbol here, just skimming:
    #   nm -C --defined-only <object file> \\
    #     | grep 'model::' | grep -vE 'example:|std:' | sort -u
    # Run against generated/types_extern.cpp.o (not the caller) this turns
    # up model::detail::g_field_name_mu, model::detail::register_field_name,
    # and model::detail::field_names() -- an inline global mutex, a plain
    # (non-template) inline function, and an inline function wrapping a
    # function-local static, none of them templates at all, so none of
    # them are things this script could ever "cover" with an extern
    # template line in the first place. They land in types_extern.cpp.o
    # correctly, PULLED IN TRANSITIVELY: they're only ever called from
    # inside Object<T>'s own virtual-dispatch bodies (via define_keys()/
    # define_fields()/define_references()), so explicitly
    # instantiating `class Object<T>` (render_entries()'s very first line
    # for every type) forces the compiler to instantiate everything those
    # bodies call, too -- confirm they're absent from the CALLER
    # (extern_template_demo.cpp.o) with the main, type-column-filtered
    # command above, which is what actually proves there's no leak.
    # Why filter on W/V and not "any defined symbol": explicit instantiation
    # of an inline-defined member (the normal case here -- these are
    # one-liners in the class body) does NOT turn its linkage strong; it
    # stays vague/weak per the C++ ABI either way, so multiple identical weak
    # copies across TUs are legal and the linker folds them. The signal isn't
    # weak-vs-strong, it's DEFINED-in-the-caller-AT-ALL: a correctly
    # `extern template`-covered entry point has NO defined copy here -- it's
    # an undefined reference ('U'), resolved from generated/types_extern.cpp.o's
    # own (weak) definition at link time instead.
    #
    # But "the caller has zero defined copies" is ALSO exactly what you'd see
    # if extern_template_demo.cpp simply never called that particular
    # function at all -- an absence in the caller doesn't by itself
    # distinguish "correctly resolved via extern template" from "never
    # exercised, so who knows."
    #
    # Confirm the other side too: that types_extern.cpp.o -- the one dedicated 
    # explicit-instantiation TU -- actually DOES define these symbols (a 
    # positive-existence check, independent of whether the demo happens to exercise
    # every one of them).
    # If this were 0, the `extern template class ...;` lines aren't
    # producing real code at all (e.g. a typo'd or silently-dropped
    # instantiation) -- a bug the caller-side check alone can't reveal if
    # the caller also never references the symbol:
    nm -C --defined-only \\
        build/default/CMakeFiles/extern_template_demo.dir/generated/types_extern.cpp.o \\
      | grep 'model::Snapshot::.*Range<' | wc -l

    # Expected output: some large non-zero count -- across both Account and
    # Order's range/view-range families, now that the `extern template class
    # Snapshot::FieldRange<...>;`-style lines in render_entries() exist. The
    # exact count will keep drifting as types.h grows more scan/cached/ref
    # fields -- don't treat the number itself as the thing to check. Before
    # this fix (i.e. if render_entries() only covered the range_by_field-style
    # FUNCTIONS and not the range CLASSES they return), this same command
    # would have read 0: the class templates would never have been
    # explicitly instantiated anywhere, only implicitly (and invisibly to
    # this check) instantiated in extern_template_demo.cpp.o itself. Zero
    # here is the failure signal, not the count.

USAGE
-----
    python3 gen_extern_templates.py path/to/types.h --namespace ns \\
        --mode header --include-path types.h --output types_extern.h
    python3 gen_extern_templates.py path/to/types.h --namespace ns \\
        --mode source --include-path types_extern.h --output types_extern.cpp
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CLASS_RE = re.compile(
    r'\b(?:class|struct)\s+(\w+)\s+[^{;]*?\bObject<\s*\1\s*>[^{;]*\{'
)
METHOD_WITH_BODY_RE = re.compile(
    r'([\w:]+(?:<[^(){};]*>)?)\s+(\w+)\s*\(\s*\)\s*const\s*\{'
)
FIELD_RE = re.compile(
    r'^\s*(?!using\b|static\b|template\b|friend\b|typedef\b|return\b)'
    r'((?:const\s+)?(?:model::)?[\w:]+(?:<[^;{}]*>)?)\s+(\w+)\s*(?:=[^;]*)?;'
    r'\s*(?://.*)?$',
    re.MULTILINE,
)
REF_OPT_TYPE_RE = re.compile(r'^(?:model::)?(Ref|Opt)<\s*(?:\w+::)*(\w+)\s*>$')


@dataclass
class TypeInfo:
    name: str
    member_types: Dict[str, str] = field(default_factory=dict)  # field/method name -> declared type
    key_fields: List[str] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)  # define_fields()-declared (Cache or Scan tagged)
    ref_fields: List[Tuple[str, str, str]] = field(default_factory=list)  # (field, Ref|Opt, target_type)


def find_matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces starting at %d" % open_idx)


def strip_function_bodies(body: str) -> str:
    """Remove every `(...) ... { ... }` region, leaving field declarations
    and dangling (now bodyless) function signatures behind. Lets FIELD_RE
    run without tripping over local variables inside define_*() bodies."""
    out = []
    i, n = 0, len(body)
    while i < n:
        if body[i] == '{':
            j = i - 1
            while j >= 0 and body[j] in ' \t\n':
                j -= 1
            if j >= 0 and body[j] == ')':
                i = find_matching_brace(body, i) + 1
                continue
        out.append(body[i])
        i += 1
    return ''.join(out)


def extract_function_body(class_body: str, func_name: str) -> Optional[str]:
    m = re.search(r'\b' + re.escape(func_name) + r'\s*\([^)]*\)[^{;]*\{', class_body)
    if not m:
        return None
    open_idx = m.end() - 1
    close_idx = find_matching_brace(class_body, open_idx)
    return class_body[open_idx + 1:close_idx]


def parse_type(name: str, class_body: str, warn) -> TypeInfo:
    info = TypeInfo(name=name)

    # Plain field declarations (after stripping every function body).
    stripped = strip_function_bodies(class_body)
    for m in FIELD_RE.finditer(stripped):
        type_str, member_name = m.group(1).strip(), m.group(2)
        info.member_types[member_name] = type_str

    # Nullary const methods with an inline body (e.g. `computed_key()`),
    # read from the UNSTRIPPED body since that's exactly what got removed above.
    for m in METHOD_WITH_BODY_RE.finditer(class_body):
        type_str, member_name = m.group(1).strip(), m.group(2)
        info.member_types.setdefault(member_name, type_str)

    def field_names(func_name: str, call: str) -> List[str]:
        body = extract_function_body(class_body, func_name)
        if body is None:
            return []
        names = re.findall(call + r'<&(?:[\w:]+::)*(\w+)>', body)
        return list(dict.fromkeys(names))  # de-dup, keep first-seen order

    info.key_fields = field_names('define_keys', r'key')
    info.fields = field_names('define_fields', r'field')
    ref_field_names = field_names('define_references', r'field_tag')

    for fname in ref_field_names:
        decl_type = info.member_types.get(fname)
        if decl_type is None:
            warn(f"{name}::{fname}: listed in define_references() but its declared "
                 f"type wasn't found -- skipping (script limitation, not necessarily a bug)")
            continue
        rm = REF_OPT_TYPE_RE.match(decl_type)
        if not rm:
            warn(f"{name}::{fname}: declared type '{decl_type}' doesn't look like "
                 f"Ref<T>/Opt<T> -- skipping")
            continue
        info.ref_fields.append((fname, rm.group(1), rm.group(2)))

    # The invariant-1 tripwire this script can check for free: a Ref<>/Opt<>
    # member that define_references() never mentions is invisible to the
    # model (cascade/nulling silently won't happen for it) -- flag it.
    declared_ref_members = {
        mname for mname, mtype in info.member_types.items() if REF_OPT_TYPE_RE.match(mtype)
    }
    missing = declared_ref_members - set(ref_field_names)
    for fname in sorted(missing):
        warn(f"{name}::{fname} is a Ref<>/Opt<> field but is NOT in define_references() "
             f"-- no extern template emitted for it, and (per CLAUDE.md invariant 1/2) "
             f"cascade delete and nulling won't happen for it either. Probably a bug.")

    return info


def parse_header(text: str, warn) -> List[TypeInfo]:
    types = []
    for m in CLASS_RE.finditer(text):
        name = m.group(1)
        open_idx = m.end() - 1
        close_idx = find_matching_brace(text, open_idx)
        body = text[open_idx + 1:close_idx]
        types.append(parse_type(name, body, warn))
    return types


def detect_namespace(text: str) -> Optional[str]:
    m = re.search(r'\bnamespace\s+(\w+)\s*\{', text)
    return m.group(1) if m else None


def qualify(ns: str, name: str) -> str:
    return f"{ns}::{name}" if ns else name


def render_entries(types: List[TypeInfo], ns: str) -> str:
    """Body of the file: every `extern template ...;` line, keyed off
    'extern template ' so render() can strip that prefix verbatim for the
    definitions file -- same list, two spellings, one source of truth.

    Grouped type-first, category-second (everything for Account, then
    everything for Order), not category-first like the original hand-
    written types_extern.h -- so an entry for one type is never split
    across a scroll past unrelated types."""
    Q = lambda n: qualify(ns, n)  # noqa: E731
    out = []

    for t in types:
        T = Q(t.name)
        out.append(f'// ==== {t.name} ====')
        out.append('')

        out.append('// -- Object<T>: CRTP virtual overrides --')
        out.append(f'extern template class Object<{T}>;')
        out.append('')

        out.append('// -- Snapshot --')
        out.append(f'extern template const {T}& Snapshot::resolve<{T}>(Ref<{T}>) const noexcept;')
        out.append(f'extern template const {T}* Snapshot::resolve<{T}>(Opt<{T}>) const noexcept;')
        out.append(f'extern template const {T}* Snapshot::find<{T}>(Ref<{T}>) const noexcept;')
        out.append(f'extern template const {T}* Snapshot::find<{T}>(Opt<{T}>) const noexcept;')
        for fname in t.key_fields:
            arg = f'const {t.member_types[fname]}&'
            out.append(f'extern template const {T}* Snapshot::find_by_key<&{T}::{fname}>({arg}) const;')
            out.append(f'extern template std::optional<View<{T}>> Snapshot::view_by_key<&{T}::{fname}>({arg}) const;')
        # find_by_field/view_by_field/range_by_field are the single merged
        # entry point for a field declared via define_fields(), tagged
        # LookupType::Cache or LookupType::Scan (cache-first, scan fallback
        # -- see model.h's Object<Derived> class comment). One extern-
        # template row per field -- a field can only ever be tagged one way,
        # so unlike the old scan_fields/cached_fields split there's nothing
        # to union or dedup here.
        for fname in t.fields:
            arg = f'const {t.member_types[fname]}&'
            out.append(f'extern template std::vector<const {T}*> Snapshot::find_by_field<&{T}::{fname}>({arg}) const;')
            out.append(f'extern template std::vector<View<{T}>> Snapshot::view_by_field<&{T}::{fname}>({arg}) const;')
            out.append(f'extern template Snapshot::FieldRange<&{T}::{fname}> Snapshot::range_by_field<&{T}::{fname}>({arg}) const;')
            out.append(f'extern template Snapshot::FieldViewRange<&{T}::{fname}> Snapshot::range_view_by_field<&{T}::{fname}>({arg}) const;')
            # Class-template instantiations, not just the function that hands
            # one back: without these, begin()/end()/iterator::operator++ etc.
            # get implicitly (weakly) re-instantiated in every TU that
            # range-for's the result -- invisible to a source-level grep for
            # "does render_entries() mention range_by_field", only visible by
            # inspecting compiled objects (nm -C --defined-only, look for a
            # defined 'W'/'V' symbol under this class in a caller TU instead
            # of types_extern.cpp.o). See this file's docstring.
            out.append(f'extern template class Snapshot::FieldRange<&{T}::{fname}>;')
            out.append(f'extern template class Snapshot::FieldViewRange<&{T}::{fname}>;')
        for fname, kind, target in t.ref_fields:
            Y = Q(target)
            # find_referrers/view_referrers/range_referrers are the single
            # merged entry point for any Ref<>/Opt<> field (cache-first when
            # tagged LookupType::Cache in define_references(), scan fallback
            # when tagged LookupType::Scan) -- one row per field either way,
            # so unlike the by-field loop above there's no separate list to
            # fold in here.
            out.append(f'extern template std::vector<const {T}*> Snapshot::find_referrers<&{T}::{fname}>(Ref<{Y}>) const;')
            out.append(f'extern template std::vector<View<{T}>> Snapshot::view_referrers<&{T}::{fname}>(Ref<{Y}>) const;')
            out.append(f'extern template Snapshot::ReferrerRange<&{T}::{fname}> Snapshot::range_referrers<&{T}::{fname}>(Ref<{Y}>) const;')
            out.append(f'extern template Snapshot::ReferrerViewRange<&{T}::{fname}> Snapshot::range_view_referrers<&{T}::{fname}>(Ref<{Y}>) const;')
            out.append(f'extern template class Snapshot::ReferrerRange<&{T}::{fname}>;')
            out.append(f'extern template class Snapshot::ReferrerViewRange<&{T}::{fname}>;')
        out.append(f'extern template View<{T}> Snapshot::view<{T}>(const {T}&) const noexcept;')
        out.append(f'extern template std::optional<View<{T}>> Snapshot::view<{T}>(Ref<{T}>) const;')
        out.append(f'extern template Snapshot::CachedBucketRange<{T}> Snapshot::range<{T}>() const;')
        out.append(f'extern template Snapshot::CachedBucketViewRange<{T}> Snapshot::range_view<{T}>() const;')
        # One instantiation each covers every use of CachedBucketRange<T>/
        # CachedBucketViewRange<T> for this T: range()/range_view() above are
        # the only callers now that range_by_field/range_referrers return
        # their own FieldRange<Field>/ReferrerRange<Field> instead (emitted
        # per-field above, not here).
        out.append(f'extern template class Snapshot::CachedBucketRange<{T}>;')
        out.append(f'extern template class Snapshot::CachedBucketViewRange<{T}>;')
        out.append('')

        out.append('// -- Transaction --')
        out.append(f'extern template Ref<{T}> Transaction::create<{T}>(std::unique_ptr<{T}>);')
        out.append(f'extern template {T}* Transaction::update<{T}>(Ref<{T}>);')
        out.append(f'extern template {T}* Transaction::update<{T}>(Opt<{T}>);')
        out.append(f'extern template void Transaction::remove<{T}>(Ref<{T}>);')
        out.append(f'extern template void Transaction::remove<{T}>(Opt<{T}>);')
        out.append(f'extern template bool Transaction::exists<{T}>(Ref<{T}>) const;')
        out.append(f'extern template bool Transaction::exists<{T}>(Opt<{T}>) const;')
        out.append(f'extern template const {T}* Transaction::peek<{T}>(Ref<{T}>) const;')
        out.append(f'extern template const {T}* Transaction::peek<{T}>(Opt<{T}>) const;')
        out.append(f'extern template const {T}* Transaction::peek_as<{T}>(Id) const;')
        out.append(f'extern template const {T}* Transaction::peek_before<{T}>(Id) const;')
        out.append('')

        out.append('// -- Model --')
        out.append(f'extern template const {T}* Model::peek_as<{T}>(Id) const;')
        for fname in t.fields:
            out.append(f'extern template LookupCounts Model::lookup_stats<&{T}::{fname}>() const;')
        out.append('')

        out.append('// -- CommitResult --')
        out.append(f'extern template Ref<{T}> CommitResult::to_real<{T}>(Ref<{T}>) const;')
        out.append(f'extern template Opt<{T}> CommitResult::to_real<{T}>(Opt<{T}>) const;')
        out.append('')

        out.append('// -- BulkTransaction --')
        out.append(f'extern template Ref<{T}> BulkTransaction::create<{T}>(std::unique_ptr<{T}>);')
        out.append(f'extern template {T}* BulkTransaction::update<{T}>(Ref<{T}>) const;')
        out.append(f'extern template {T}* BulkTransaction::update<{T}>(Opt<{T}>) const;')
        out.append('')

        out.append('// -- View --')
        for fname, kind, target in t.ref_fields:
            Y = Q(target)
            if kind == 'Ref':
                out.append(f'extern template View<{Y}> View<{T}>::operator[]<{Y}>(Ref<{Y}> {T}::*) const noexcept;')
            else:
                out.append(f'extern template std::optional<View<{Y}>> View<{T}>::operator[]<{Y}>(Opt<{Y}> {T}::*) const noexcept;')
        for fname, kind, target in t.ref_fields:
            Y = Q(target)
            out.append(f'extern template std::vector<View<{T}>> View<{Y}>::find_referrers<&{T}::{fname}>() const;')
        out.append('')

    return '\n'.join(out).rstrip('\n')


def render(types: List[TypeInfo], ns: str, mode: str, include_line: str) -> str:
    """mode='header': `extern template` declarations, #includes the source
    header (types.h). mode='source': the matching `template` (explicit
    instantiation) DEFINITIONS, #includes the generated header instead --
    same entry list both times, satisfied from one call to render_entries()
    so the two files can never drift apart from each other (only from
    types.h itself, on the next regeneration)."""
    entries = render_entries(types, ns)
    if mode == 'source':
        entries = entries.replace('extern template ', 'template ')
        banner = (
            f'// AUTO-GENERATED by gen_extern_templates.py from "{include_line}"\'s own\n'
            '// header, in full-coverage mode. Do not edit -- edit example/types.h or\n'
            '// scripts/gen_extern_templates.py and let CMake regenerate this.'
        )
    else:
        banner = (
            f'// AUTO-GENERATED by gen_extern_templates.py from "{include_line}", in\n'
            '// full-coverage mode: every structurally reachable entry point, not just\n'
            '// ones genuinely called -- see the script\'s own docstring for the tradeoff\n'
            '// this makes against types_extern.h\'s original hand-curated version.\n'
            '// Do not edit -- edit example/types.h or scripts/gen_extern_templates.py\n'
            '// and let CMake regenerate this.'
        )
    return (
        f'{banner}\n'
        f'#include "{include_line}"\n'
        '\n'
        'namespace model {\n'
        '\n'
        f'{entries}\n'
        '\n'
        '}  // namespace model\n'
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('header', help='input header declaring one or more model::Object<T> types')
    ap.add_argument('--namespace', help='override auto-detected namespace')
    ap.add_argument('--include-path', required=True,
                     help='#include path to emit: the source header in --mode header, '
                          'the generated header itself in --mode source')
    ap.add_argument('--mode', choices=['header', 'source'], default='header',
                     help="'header': extern template declarations (default). "
                          "'source': matching explicit-instantiation definitions.")
    ap.add_argument('--output', help='write to this file instead of stdout')
    args = ap.parse_args()

    with open(args.header) as f:
        text = f.read()

    warnings = []

    def warn(msg: str) -> None:
        warnings.append(msg)

    types = parse_header(text, warn)
    if not types:
        print(f"warning: no `class/struct T : ... Object<T>` found in {args.header}", file=sys.stderr)

    ns = args.namespace or detect_namespace(text)

    rendered = render(types, ns, args.mode, args.include_path)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(rendered)
    else:
        sys.stdout.write(rendered)

    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
