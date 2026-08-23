// pack_narrow_key/is_narrow_key_eligible_v (model.h) -- step 1 of the native narrow-int
// key work: proves the packing primitive itself is correct in isolation, before anything
// in model.cpp reads or writes it. See CLAUDE.md's project memory
// (project_native_narrow_key_proposal) for the design rationale.

#include <cstdint>
#include <cstring>

#include "model/model.h"
#include "test_harness.h"

using namespace model;

namespace {

// A padding-free struct: two 1-byte members, no alignment gap between or
// after them (sizeof == 2, matches the sum of its members exactly).
struct Pair16 {
    std::int8_t a = 0;
    std::int8_t b = 0;
};

// A struct WITH padding: int32_t forces 3 bytes of alignment padding after
// the leading int8_t (sizeof == 8, not 5) -- must fail eligibility, since
// those 3 padding bytes are not guaranteed to hold any particular value.
struct PaddedStruct {
    std::int8_t a = 0;
    std::int32_t b = 0;
};

enum class Color : std::int32_t { Red = 0, Green = 1, Blue = 2 };

static_assert(is_narrow_key_eligible_v<std::int8_t>);
static_assert(is_narrow_key_eligible_v<std::uint8_t>);
static_assert(is_narrow_key_eligible_v<std::int16_t>);
static_assert(is_narrow_key_eligible_v<std::int32_t>);
static_assert(is_narrow_key_eligible_v<std::int64_t>);
static_assert(is_narrow_key_eligible_v<std::uint64_t>);
static_assert(is_narrow_key_eligible_v<bool>);
static_assert(is_narrow_key_eligible_v<Pair16>);
static_assert(is_narrow_key_eligible_v<Color>);

// float/double: has_unique_object_representations_v is false for both
// (IEEE-754's +0.0/-0.0 and multiple NaN encodings), deliberately excluded.
static_assert(!is_narrow_key_eligible_v<float>);
static_assert(!is_narrow_key_eligible_v<double>);
// A struct with padding: the 3 bytes after `a` are not guaranteed.
static_assert(!is_narrow_key_eligible_v<PaddedStruct>);
// A pointer has a unique representation but is excluded on its own terms --
// its bit pattern isn't a meaningful "value" for lookup purposes.
static_assert(!is_narrow_key_eligible_v<int*>);
// Too wide to fit in one uint64_t.
struct TooWide {
    std::int64_t a = 0;
    std::int64_t b = 0;
};
static_assert(!is_narrow_key_eligible_v<TooWide>);

}  // namespace

TEST(pack_narrow_key_round_trips_signed_and_unsigned_integral_values) {
    CHECK_EQ(pack_narrow_key<std::int64_t>(0), std::uint64_t{0});
    CHECK_EQ(pack_narrow_key<std::int64_t>(1), std::uint64_t{1});
    CHECK_EQ(pack_narrow_key<std::uint64_t>(0xFFFF'FFFF'FFFF'FFFFULL), 0xFFFF'FFFF'FFFF'FFFFULL);

    // -1 as int64_t is all-ones in two's complement -- packs to the same
    // bit pattern as UINT64_MAX, not to some sign-extended-into-a-smaller-
    // field artifact.
    CHECK_EQ(pack_narrow_key<std::int64_t>(-1), 0xFFFF'FFFF'FFFF'FFFFULL);

    // Two DIFFERENT int32_t values must never collide onto the same packed
    // key (would silently merge two distinct objects' keys).
    CHECK(pack_narrow_key<std::int32_t>(5) != pack_narrow_key<std::int32_t>(-5));
    CHECK(pack_narrow_key<std::int32_t>(0) != pack_narrow_key<std::int32_t>(1));
}

// The load-bearing property: a type narrower than 8 bytes must have its
// UPPER bytes deterministically zero, not whatever happened to be on the
// stack beforehand -- otherwise two objects with the identical logical
// value could pack to different keys depending on stack garbage.
TEST(pack_narrow_key_zero_extends_narrower_types_deterministically) {
    // Poison the stack region a narrow pack_narrow_key call would sit in,
    // by first running a call whose local uint64_t is filled with non-zero
    // bytes, then immediately calling pack_narrow_key on a narrower type in
    // a fresh, potentially-overlapping stack frame.
    volatile std::uint64_t poison = 0xAAAA'AAAA'AAAA'AAAAULL;
    (void)poison;

    const std::int8_t narrow_value = 5;
    const std::uint64_t packed = pack_narrow_key(narrow_value);
    CHECK_EQ(packed, std::uint64_t{5});  // upper 56 bits must be exactly zero, not 0xAAAA...

    const std::int16_t narrow16 = -1;  // 0xFFFF within its own 2 bytes
    const std::uint64_t packed16 = pack_narrow_key(narrow16);
    CHECK_EQ(packed16, std::uint64_t{0xFFFF});  // upper 48 bits zero, not sign-extended further
}

TEST(pack_narrow_key_round_trips_a_padding_free_struct) {
    Pair16 p1{5, 10};
    Pair16 p2{5, 10};
    Pair16 p3{5, 11};
    CHECK_EQ(pack_narrow_key(p1), pack_narrow_key(p2));  // identical logical value -> identical key
    CHECK(pack_narrow_key(p1) != pack_narrow_key(p3));   // different value -> different key
}

TEST(pack_narrow_key_round_trips_an_enum) {
    CHECK(pack_narrow_key(Color::Red) != pack_narrow_key(Color::Green));
    CHECK(pack_narrow_key(Color::Green) != pack_narrow_key(Color::Blue));
    CHECK_EQ(pack_narrow_key(Color::Red), pack_narrow_key(Color::Red));
}

// to_field_key's non-arithmetic fallback branch: only ever reached via
// force_string/Coarse (not exercised here yet, that's model.cpp's job) --
// this proves the STRING it would produce is well-formed and, critically,
// distinguishes different values the same way pack_narrow_key itself does
// (to_field_key literally calls std::to_string(pack_narrow_key(v))).
TEST(to_field_key_produces_distinct_strings_for_distinct_narrow_eligible_struct_values) {
    const std::string s1 = to_field_key(Pair16{5, 10});
    const std::string s2 = to_field_key(Pair16{5, 10});
    const std::string s3 = to_field_key(Pair16{5, 11});
    CHECK_EQ(s1, s2);
    CHECK(s1 != s3);
}

// The order-matters correctness point from to_field_key's own doc comment:
// an arithmetic type's string form is its natural signed decimal, never
// pack_narrow_key's reinterpreted-unsigned form.
TEST(to_field_key_keeps_natural_signed_decimal_form_for_arithmetic_types) {
    CHECK_EQ(to_field_key<std::int64_t>(-5), std::string("-5"));
    CHECK(to_field_key<std::int64_t>(-5) != std::to_string(pack_narrow_key<std::int64_t>(-5)));
}
