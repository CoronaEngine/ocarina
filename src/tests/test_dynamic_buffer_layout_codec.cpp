//
// Created by Z on 12/04/2026.
//

#include <iostream>

#include "ast/layout_resolver.h"
#include "core/dynamic_buffer/dynamic_buffer_layout_codec.h"
#include "core/type_desc.h"
#include "math/real.h"

using namespace ocarina;

struct CodecRealLeaf {
    real value;
    Vector<real, 2> index;
};

OC_MAKE_STRUCT_REFLECTION(CodecRealLeaf, value, index)
OC_MAKE_STORAGE_TYPE(CodecRealLeaf, value, index)
OC_MAKE_STRUCT_DESC(CodecRealLeaf, value, index)
OC_MAKE_STRUCT_IS_DYNAMIC(CodecRealLeaf, value, index)

struct CodecRealNested {
    CodecRealLeaf leaf;
    Vector<real, 3> samples;
    float4 extra;
};

OC_MAKE_STRUCT_REFLECTION(CodecRealNested, leaf, samples, extra)
OC_MAKE_STORAGE_TYPE(CodecRealNested, leaf, samples, extra)
OC_MAKE_STRUCT_DESC(CodecRealNested, leaf, samples, extra)
OC_MAKE_STRUCT_IS_DYNAMIC(CodecRealNested, leaf, samples, extra)

struct CodecRealArray {
    ocarina::array<real, 4> samples;
    float weight;
};

OC_MAKE_STRUCT_REFLECTION(CodecRealArray, samples, weight)
OC_MAKE_STORAGE_TYPE(CodecRealArray, samples, weight)
OC_MAKE_STRUCT_DESC(CodecRealArray, samples, weight)
OC_MAKE_STRUCT_IS_DYNAMIC(CodecRealArray, samples, weight)

namespace {

[[nodiscard]] bool check_impl(bool condition, const char *expr) {
    if (!condition) {
        std::cerr << "FAILED: " << expr << std::endl;
    }
    return condition;
}

#define CHECK(...)                            \
    do {                                      \
        if (!check_impl((__VA_ARGS__), #__VA_ARGS__)) { \
            return false;                     \
        }                                     \
    } while (false)

[[nodiscard]] bool close_float(float lhs, float rhs, float eps = 1e-3f) {
    return std::abs(lhs - rhs) <= eps;
}

[[nodiscard]] bool equal_real(real lhs, real rhs, float eps = 1e-3f) {
    return close_float(static_cast<float>(lhs), static_cast<float>(rhs), eps);
}

template<typename T, size_t N>
[[nodiscard]] bool equal_vec(const Vector<T, N> &lhs, const Vector<T, N> &rhs, float eps = 1e-3f) {
    for (size_t index = 0; index < N; ++index) {
        if (!close_float(static_cast<float>(lhs[index]), static_cast<float>(rhs[index]), eps)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool equal_leaf(const CodecRealLeaf &lhs, const CodecRealLeaf &rhs, float eps = 1e-3f) {
    return equal_real(lhs.value, rhs.value, eps) && equal_vec(lhs.index, rhs.index, eps);
}

[[nodiscard]] bool equal_nested(const CodecRealNested &lhs, const CodecRealNested &rhs, float eps = 1e-3f) {
    return equal_leaf(lhs.leaf, rhs.leaf, eps) &&
           equal_vec(lhs.samples, rhs.samples, eps) &&
           close_float(lhs.extra.x, rhs.extra.x, eps) &&
           close_float(lhs.extra.y, rhs.extra.y, eps) &&
           close_float(lhs.extra.z, rhs.extra.z, eps) &&
           close_float(lhs.extra.w, rhs.extra.w, eps);
}

[[nodiscard]] bool equal_real_array(const CodecRealArray &lhs, const CodecRealArray &rhs, float eps = 1e-6f) {
    for (size_t index = 0; index < lhs.samples.size(); ++index) {
        if (!equal_real(lhs.samples[index], rhs.samples[index], eps)) {
            return false;
        }
    }
    return close_float(lhs.weight, rhs.weight, eps);
}

[[nodiscard]] CodecRealLeaf make_leaf(uint index) {
    float value = static_cast<float>(index);
    return {
        .value = real{value + 0.125f},
        .index = Vector<real, 2>{real{value + 1.25f}, real{value + 2.5f}}};
}

[[nodiscard]] CodecRealNested make_nested(uint index) {
    float value = static_cast<float>(index);
    return {
        .leaf = make_leaf(index),
        .samples = Vector<real, 3>{real{value + 3.0f}, real{value + 4.5f}, real{value + 5.75f}},
        .extra = make_float4(value + 6.0f, value + 7.0f, value + 8.0f, value + 9.0f)};
}

[[nodiscard]] CodecRealArray make_real_array(uint index) {
    float value = static_cast<float>(index);
    return {
        .samples = {real{value + 0.25f}, real{value + 1.25f}, real{value + 2.25f}, real{value + 3.25f}},
        .weight = value + 4.5f};
}

[[nodiscard]] StoragePrecisionPolicy make_policy(PrecisionPolicy policy) {
    return StoragePrecisionPolicy{.policy = policy, .allow_real_in_storage = true};
}

template<typename T>
[[nodiscard]] bool test_aos_storage_matches_layout_resolver(StoragePrecisionPolicy policy) {
    LayoutResolver resolver(policy);
    const Type *resolved = resolver.resolve(Type::of<T>());
    CHECK(resolved != nullptr);
    CHECK(DynamicBufferLayoutCodec<T>::storage_bytes(3u, policy, DynamicBufferLayout::aos) == resolved->size() * 3u);
    return true;
}

[[nodiscard]] bool test_force_f32_array_uses_float_storage() {
    StoragePrecisionPolicy policy = make_policy(PrecisionPolicy::force_f32);
    HostByteBuffer bytes;
    CodecRealArray value = make_real_array(0u);
    DynamicBufferLayoutCodec<CodecRealArray>::encode(&value, 1u, bytes, policy, DynamicBufferLayout::aos);
    CHECK(bytes.size() == 20u);
    CHECK(close_float(bytes.load<float>(0u), 0.25f, 1e-6f));
    CHECK(close_float(bytes.load<float>(4u), 1.25f, 1e-6f));
    CHECK(close_float(bytes.load<float>(8u), 2.25f, 1e-6f));
    CHECK(close_float(bytes.load<float>(12u), 3.25f, 1e-6f));
    CHECK(close_float(bytes.load<float>(16u), 4.5f, 1e-6f));
    return true;
}

[[nodiscard]] bool test_storage_type_scheme() {
    using LeafImplHalf = resolved_storage_impl_t<CodecRealLeaf, half>;
    using NestedImplHalf = resolved_storage_impl_t<CodecRealNested, half>;
    using LeafF16 = storage_t<CodecRealLeaf, PrecisionPolicy::force_f16>;
    using LeafF32 = storage_t<CodecRealLeaf, PrecisionPolicy::force_f32>;
    using NestedF16 = storage_t<CodecRealNested, PrecisionPolicy::force_f16>;
    using ArrayF16 = storage_t<CodecRealArray, PrecisionPolicy::force_f16>;

    static_assert(std::is_same_v<LeafImplHalf, LeafF16>);
    static_assert(std::is_same_v<NestedImplHalf, NestedF16>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(std::declval<LeafF16>().value)>, half>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(std::declval<LeafF16>().index)>, half2>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(std::declval<LeafF32>().value)>, float>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(std::declval<NestedF16>().leaf)>, LeafF16>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(std::declval<NestedF16>().samples)>, half3>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(std::declval<ArrayF16>().samples)>, ocarina::array<half, 4>>);

    CodecRealNested nested = make_nested(2u);
    auto nested_f16 = to_storage_value<PrecisionPolicy::force_f16>(nested);
    auto nested_f32 = to_storage_value<PrecisionPolicy::force_f32>(nested);
    CHECK(close_float(static_cast<float>(nested_f16.leaf.value), 2.125f, 1e-2f));
    CHECK(close_float(nested_f32.leaf.value, 2.125f, 1e-6f));
    CHECK(equal_nested(nested, from_storage_value<CodecRealNested, PrecisionPolicy::force_f16>(nested_f16), 1e-2f));
    CHECK(equal_nested(nested, from_storage_value<CodecRealNested, PrecisionPolicy::force_f32>(nested_f32), 1e-6f));

    CodecRealArray array_value = make_real_array(1u);
    auto array_f16 = to_storage_value<PrecisionPolicy::force_f16>(array_value);
    CHECK(close_float(static_cast<float>(array_f16.samples[0]), 1.25f, 1e-2f));
    CHECK(equal_real_array(array_value, from_storage_value<CodecRealArray, PrecisionPolicy::force_f16>(array_f16), 1e-2f));
    return true;
}

[[nodiscard]] bool test_half_aos_round_trip() {
    StoragePrecisionPolicy policy = make_policy(PrecisionPolicy::force_f16);
    vector<CodecRealNested> src(3u);
    vector<CodecRealNested> dst(3u);
    for (uint index = 0; index < src.size(); ++index) {
        src[index] = make_nested(index);
    }
    HostByteBuffer bytes;
    DynamicBufferLayoutCodec<CodecRealNested>::encode(src.data(), src.size(), bytes, policy, DynamicBufferLayout::aos);
    DynamicBufferLayoutCodec<CodecRealNested>::decode(bytes, dst.size(), dst.data(), policy, DynamicBufferLayout::aos);
    for (size_t index = 0; index < src.size(); ++index) {
        CHECK(equal_nested(src[index], dst[index]));
    }
    return true;
}

[[nodiscard]] bool test_float_aos_round_trip() {
    StoragePrecisionPolicy policy = make_policy(PrecisionPolicy::force_f32);
    vector<CodecRealLeaf> src(4u);
    vector<CodecRealLeaf> dst(4u);
    for (uint index = 0; index < src.size(); ++index) {
        src[index] = make_leaf(index);
    }
    HostByteBuffer bytes;
    DynamicBufferLayoutCodec<CodecRealLeaf>::encode(src.data(), src.size(), bytes, policy, DynamicBufferLayout::aos);
    DynamicBufferLayoutCodec<CodecRealLeaf>::decode(bytes, dst.size(), dst.data(), policy, DynamicBufferLayout::aos);
    for (size_t index = 0; index < src.size(); ++index) {
        CHECK(equal_leaf(src[index], dst[index], 1e-6f));
    }
    return true;
}

[[nodiscard]] bool test_half_soa_round_trip() {
    StoragePrecisionPolicy policy = make_policy(PrecisionPolicy::force_f16);
    vector<CodecRealNested> src(2u);
    vector<CodecRealNested> dst(2u);
    for (uint index = 0; index < src.size(); ++index) {
        src[index] = make_nested(index);
    }
    HostByteBuffer bytes;
    DynamicBufferLayoutCodec<CodecRealNested>::encode(src.data(), src.size(), bytes, policy, DynamicBufferLayout::soa);
    CHECK(bytes.size() == 60u);
    DynamicBufferLayoutCodec<CodecRealNested>::decode(bytes, dst.size(), dst.data(), policy, DynamicBufferLayout::soa);
    for (size_t index = 0; index < src.size(); ++index) {
        CHECK(equal_nested(src[index], dst[index]));
    }
    return true;
}

[[nodiscard]] bool test_float_soa_array_round_trip() {
    StoragePrecisionPolicy policy = make_policy(PrecisionPolicy::force_f32);
    vector<CodecRealArray> src(3u);
    vector<CodecRealArray> dst(3u);
    for (uint index = 0; index < src.size(); ++index) {
        src[index] = make_real_array(index);
    }
    HostByteBuffer bytes;
    DynamicBufferLayoutCodec<CodecRealArray>::encode(src.data(), src.size(), bytes, policy, DynamicBufferLayout::soa);
    CHECK(bytes.size() == 60u);
    CHECK(close_float(bytes.load<float>(0u), 0.25f, 1e-6f));
    CHECK(close_float(bytes.load<float>(4u), 1.25f, 1e-6f));
    CHECK(close_float(bytes.load<float>(8u), 2.25f, 1e-6f));
    DynamicBufferLayoutCodec<CodecRealArray>::decode(bytes, dst.size(), dst.data(), policy, DynamicBufferLayout::soa);
    for (size_t index = 0; index < src.size(); ++index) {
        CHECK(equal_real_array(src[index], dst[index], 1e-6f));
    }
    return true;
}

}// namespace

int main() {
    bool passed = true;
    passed = test_aos_storage_matches_layout_resolver<CodecRealLeaf>(make_policy(PrecisionPolicy::force_f32)) && passed;
    passed = test_aos_storage_matches_layout_resolver<CodecRealLeaf>(make_policy(PrecisionPolicy::force_f16)) && passed;
    passed = test_aos_storage_matches_layout_resolver<CodecRealArray>(make_policy(PrecisionPolicy::force_f32)) && passed;
    passed = test_force_f32_array_uses_float_storage() && passed;
    passed = test_storage_type_scheme() && passed;
    passed = test_float_aos_round_trip() && passed;
    passed = test_half_aos_round_trip() && passed;
    passed = test_half_soa_round_trip() && passed;
    passed = test_float_soa_array_round_trip() && passed;

    if (!passed) {
        std::cerr << "dynamic buffer layout codec test failed" << std::endl;
        return 1;
    }
    std::cout << "dynamic buffer layout codec test passed" << std::endl;
    return 0;
}