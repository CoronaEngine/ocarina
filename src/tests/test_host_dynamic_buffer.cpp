//
// Created by GitHub Copilot on 2026/4/13.
//

#include <iostream>

#include "core/dynamic_buffer/host_dynamic_buffer.h"
#include "core/type_desc.h"
#include "math/real.h"

using namespace ocarina;

struct HostDynamicLeaf {
    real roughness;
    Vector<real, 3> direction;
};

OC_MAKE_STRUCT_REFLECTION(HostDynamicLeaf, roughness, direction)
OC_MAKE_STRUCT_DESC(HostDynamicLeaf, roughness, direction)
OC_MAKE_STRUCT_IS_DYNAMIC(HostDynamicLeaf, roughness, direction)

struct HostDynamicRecord {
    HostDynamicLeaf leaf;
    ocarina::array<real, 4> weights;
    Matrix<real, 2, 2> basis;
    float extra;
};

OC_MAKE_STRUCT_REFLECTION(HostDynamicRecord, leaf, weights, basis, extra)
OC_MAKE_STRUCT_DESC(HostDynamicRecord, leaf, weights, basis, extra)
OC_MAKE_STRUCT_IS_DYNAMIC(HostDynamicRecord, leaf, weights, basis, extra)

namespace {

[[nodiscard]] bool check_impl(bool condition, const char *expr) {
    if (!condition) {
        std::cerr << "FAILED: " << expr << std::endl;
    }
    return condition;
}

#define CHECK(...)                                  \
    do {                                            \
        if (!check_impl((__VA_ARGS__), #__VA_ARGS__)) { \
            return false;                           \
        }                                           \
    } while (false)

[[nodiscard]] bool close_float(float lhs, float rhs, float eps = 1e-3f) {
    return std::abs(lhs - rhs) <= eps;
}

[[nodiscard]] HostDynamicRecord make_record(float base) {
    return {
        .leaf = HostDynamicLeaf{
            .roughness = real{base + 0.1f},
            .direction = Vector<real, 3>{real{base + 1.0f}, real{base + 2.0f}, real{base + 3.0f}}},
        .weights = {real{base + 4.0f}, real{base + 5.0f}, real{base + 6.0f}, real{base + 7.0f}},
        .basis = Matrix<real, 2, 2>{Vector<real, 2>{real{base + 8.0f}, real{base + 9.0f}},
                                     Vector<real, 2>{real{base + 10.0f}, real{base + 11.0f}}},
        .extra = base + 12.0f};
}

[[nodiscard]] bool equal_record(const HostDynamicRecord &lhs,
                                const HostDynamicRecord &rhs,
                                float eps = 1e-3f) {
    CHECK(close_float(static_cast<float>(lhs.leaf.roughness), static_cast<float>(rhs.leaf.roughness), eps));
    for (size_t index = 0; index < 3u; ++index) {
        CHECK(close_float(static_cast<float>(lhs.leaf.direction[index]), static_cast<float>(rhs.leaf.direction[index]), eps));
    }
    for (size_t index = 0; index < 4u; ++index) {
        CHECK(close_float(static_cast<float>(lhs.weights[index]), static_cast<float>(rhs.weights[index]), eps));
    }
    for (size_t col = 0; col < 2u; ++col) {
        for (size_t row = 0; row < 2u; ++row) {
            CHECK(close_float(static_cast<float>(lhs.basis[col][row]), static_cast<float>(rhs.basis[col][row]), eps));
        }
    }
    CHECK(close_float(lhs.extra, rhs.extra, eps));
    return true;
}

[[nodiscard]] StoragePrecisionPolicy make_policy(PrecisionPolicy policy) {
    return StoragePrecisionPolicy{.policy = policy, .allow_real_in_storage = true};
}

[[nodiscard]] bool test_layout_plan_reports_expected_types() {
    auto plan = DynamicBufferLayoutPlan::create(Type::of<HostDynamicRecord>(),
                                                make_policy(PrecisionPolicy::force_f16),
                                                DynamicBufferLayout::aos);
    auto codec_bytes = DynamicBufferLayoutCodec<HostDynamicRecord>::storage_bytes(1u,
                                                                                  make_policy(PrecisionPolicy::force_f16),
                                                                                  DynamicBufferLayout::aos);
    CHECK(plan.logical_type() == Type::of<HostDynamicRecord>());
    CHECK(plan.resolved_type() != nullptr);
    CHECK(plan.contains_real());
    CHECK(plan.has_precision_lowering());
    CHECK(plan.element_size_bytes() == codec_bytes);
    return true;
}

[[nodiscard]] bool test_host_buffer_record_round_trip() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f16),
                                                         DynamicBufferLayout::aos,
                                                         2u);
    const auto lhs = make_record(0.0f);
    const auto rhs = make_record(16.0f);
    buffer.write(0u, lhs);
    buffer.write(1u, rhs);
    CHECK(equal_record(buffer.read<HostDynamicRecord>(0u), lhs, 1e-2f));
    CHECK(equal_record(buffer.read<HostDynamicRecord>(1u), rhs, 1e-2f));
    CHECK(buffer.dirty_range().dirty);
    CHECK(buffer.dirty_range().begin_byte == 0u);
    CHECK(buffer.dirty_range().end_byte == buffer.storage_size_bytes());
    return true;
}

[[nodiscard]] bool test_single_element_external_read_write_usage() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f32),
                                                         DynamicBufferLayout::aos,
                                                         1u);
    const auto input = make_record(3.0f);
    buffer.write<HostDynamicRecord>(0u, input);

    const auto output = buffer.read<HostDynamicRecord>(0u);
    CHECK(equal_record(output, input, 1e-6f));
    CHECK(buffer.element_count() == 1u);
    CHECK(buffer.dirty_range().dirty);
    auto record_region = buffer.layout_plan().record_region(0u);
    CHECK(buffer.dirty_range().begin_byte == record_region.begin_byte);
    CHECK(buffer.dirty_range().end_byte == record_region.end_byte);
    return true;
}

[[nodiscard]] bool test_typed_view_single_element_read_write_usage() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f32),
                                                         DynamicBufferLayout::aos,
                                                         0u);
    buffer.resize(1u);
    TypedHostDynamicBufferView<HostDynamicRecord> view{buffer};

    const auto input = make_record(5.0f);
    view.write(0u, input);

    const auto output = view.read(0u);
    CHECK(equal_record(output, input, 1e-6f));
    CHECK(view.element_count() == 1u);
    CHECK(!view.empty());
    return true;
}

[[nodiscard]] bool test_field_patch_updates_target_bytes() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f16),
                                                         DynamicBufferLayout::aos,
                                                         1u);
    buffer.write(0u, make_record(0.0f));
    buffer.clear_dirty();

    auto roughness_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<0u>>();
    auto roughness_region = buffer.layout_plan().field_region(0u, roughness_path);
    buffer.patch(0u, roughness_path, real{42.5f});
    auto record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.leaf.roughness), 42.5f));
    CHECK(buffer.dirty_range().begin_byte == roughness_region.begin_byte);
    CHECK(buffer.dirty_range().end_byte == roughness_region.end_byte);

    buffer.clear_dirty();
    auto weight_path = make_typed_field_path<FieldMemberStep<1u>, FieldIndexStep<2u>>();
    auto weight_region = buffer.layout_plan().field_region(0u, weight_path);
    buffer.patch(0u, weight_path, real{15.25f});
    record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.weights[2u]), 15.25f));
    CHECK(buffer.dirty_range().begin_byte == weight_region.begin_byte);
    CHECK(buffer.dirty_range().end_byte == weight_region.end_byte);

    buffer.clear_dirty();
    auto direction_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<1u>, FieldComponentStep<1u>>();
    auto direction_region = buffer.layout_plan().field_region(0u, direction_path);
    buffer.patch(0u, direction_path, real{7.75f});
    record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.leaf.direction[1u]), 7.75f));
    CHECK(buffer.dirty_range().begin_byte == direction_region.begin_byte);
    CHECK(buffer.dirty_range().end_byte == direction_region.end_byte);

    buffer.clear_dirty();
    auto matrix_path = make_typed_field_path<FieldMemberStep<2u>, FieldIndexStep<1u>, FieldComponentStep<0u>>();
    auto matrix_region = buffer.layout_plan().field_region(0u, matrix_path);
    buffer.patch(0u, matrix_path, real{9.5f});
    record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.basis[1u][0u]), 9.5f));
    CHECK(buffer.dirty_range().begin_byte == matrix_region.begin_byte);
    CHECK(buffer.dirty_range().end_byte == matrix_region.end_byte);
    return true;
}

[[nodiscard]] bool test_append_and_upload_view() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f32),
                                                         DynamicBufferLayout::aos,
                                                         0u);
    vector<HostDynamicRecord> values{make_record(1.0f), make_record(2.0f)};
    buffer.append<HostDynamicRecord>(values);
    CHECK(buffer.element_count() == 2u);
    CHECK(equal_record(buffer.read<HostDynamicRecord>(0u), values[0], 1e-6f));
    CHECK(equal_record(buffer.read<HostDynamicRecord>(1u), values[1], 1e-6f));
    auto upload = buffer.upload_view();
    CHECK(upload.element_count == 2u);
    CHECK(upload.layout == DynamicBufferLayout::aos);
    CHECK(upload.logical_type == Type::of<HostDynamicRecord>());
    CHECK(upload.resolved_type != nullptr);
    CHECK(upload.bytes.size() == buffer.storage_size_bytes());
    return true;
}

}// namespace

int main() {
    bool passed = true;
    passed = test_layout_plan_reports_expected_types() && passed;
    passed = test_host_buffer_record_round_trip() && passed;
    passed = test_single_element_external_read_write_usage() && passed;
    passed = test_typed_view_single_element_read_write_usage() && passed;
    passed = test_field_patch_updates_target_bytes() && passed;
    passed = test_append_and_upload_view() && passed;
    if (!passed) {
        std::cerr << "host dynamic buffer test failed" << std::endl;
        return 1;
    }
    std::cout << "host dynamic buffer test passed" << std::endl;
    return 0;
}