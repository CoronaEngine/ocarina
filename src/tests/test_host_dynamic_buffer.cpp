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

[[nodiscard]] size_t total_segment_bytes(span<const ByteSegment> segments) {
    size_t total = 0u;
    for (const auto &segment : segments) {
        total += segment.size_in_bytes;
    }
    return total;
}

[[nodiscard]] bool contains_region(span<const ByteRegion> segments,
                                   ByteRegion target) {
    for (const auto &segment : segments) {
        if (segment.begin_byte == target.begin_byte && segment.end_byte == target.end_byte) {
            return true;
        }
    }
    return false;
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
                                                DynamicBufferLayout::AOS);
    auto codec_bytes = DynamicBufferLayoutCodec<HostDynamicRecord>::storage_bytes(1u,
                                                                                  make_policy(PrecisionPolicy::force_f16),
                                                                                  DynamicBufferLayout::AOS);
    CHECK(plan.logical_type() == Type::of<HostDynamicRecord>());
    CHECK(plan.resolved_type() != nullptr);
    CHECK(plan.contains_real());
    CHECK(plan.has_precision_lowering());
    CHECK(plan.element_size_bytes() == codec_bytes);
    return true;
}

[[nodiscard]] bool test_aos_record_and_field_regions_report_expected_offsets() {
    auto plan = DynamicBufferLayoutPlan::create(Type::of<HostDynamicRecord>(),
                                                make_policy(PrecisionPolicy::force_f16),
                                                DynamicBufferLayout::AOS);

    auto record_region = plan.record_region(1u);
    CHECK(record_region.begin_byte == 40u);
    CHECK(record_region.end_byte == 80u);
    CHECK(record_region.size() == 40u);

    auto roughness_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<0u>>();
    auto roughness_region = plan.field_region(1u, roughness_path);
    CHECK(roughness_region.begin_byte == 40u);
    CHECK(roughness_region.end_byte == 42u);
    CHECK(roughness_region.size() == 2u);

    auto direction_z_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<1u>, FieldComponentStep<2u>>();
    auto direction_z_region = plan.field_region(1u, direction_z_path);
    CHECK(direction_z_region.begin_byte == 52u);
    CHECK(direction_z_region.end_byte == 54u);
    CHECK(direction_z_region.size() == 2u);

    auto weight_path = make_typed_field_path<FieldMemberStep<1u>, FieldIndexStep<2u>>();
    auto weight_region = plan.field_region(1u, weight_path);
    CHECK(weight_region.begin_byte == 60u);
    CHECK(weight_region.end_byte == 62u);
    CHECK(weight_region.size() == 2u);

    auto matrix_path = make_typed_field_path<FieldMemberStep<2u>, FieldIndexStep<1u>, FieldComponentStep<0u>>();
    auto matrix_region = plan.field_region(1u, matrix_path);
    CHECK(matrix_region.begin_byte == 68u);
    CHECK(matrix_region.end_byte == 70u);
    CHECK(matrix_region.size() == 2u);

    CHECK(plan.field_logical_type(direction_z_path) == Type::of<real>());
    CHECK(plan.field_resolved_type(direction_z_path) == Type::of<half>());
    return true;
}

[[nodiscard]] bool test_soa_dirty_segments_preserve_disjoint_field_segments() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f16),
                                                         DynamicBufferLayout::SOA,
                                                         2u);
    buffer.write(0u, make_record(0.0f));
    buffer.write(1u, make_record(10.0f));
    buffer.clear_dirty();

    auto first_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<0u>>();
    auto second_path = make_typed_field_path<FieldMemberStep<3u>>();
    auto first_segments = buffer.layout_plan().field_segments(buffer.element_count(), 0u, first_path);
    auto second_segments = buffer.layout_plan().field_segments(buffer.element_count(), 1u, second_path);
    CHECK(first_segments.size() == 1u);
    CHECK(second_segments.size() == 1u);

    buffer.patch(0u, first_path, real{1.5f});
    buffer.patch(1u, second_path, 99.0f);

    CHECK(buffer.dirty_segments().size() == 2u);
    CHECK(contains_region(buffer.dirty_segments(),
                          ByteRegion{first_segments[0].storage_begin_byte,
                                     first_segments[0].storage_begin_byte + first_segments[0].size_in_bytes}));
    CHECK(contains_region(buffer.dirty_segments(),
                          ByteRegion{second_segments[0].storage_begin_byte,
                                     second_segments[0].storage_begin_byte + second_segments[0].size_in_bytes}));
    CHECK(buffer.dirty_range().dirty);
    CHECK(buffer.dirty_range().begin_byte == std::min(first_segments[0].storage_begin_byte,
                                                      second_segments[0].storage_begin_byte));
    CHECK(buffer.dirty_range().end_byte == std::max(first_segments[0].storage_begin_byte + first_segments[0].size_in_bytes,
                                                    second_segments[0].storage_begin_byte + second_segments[0].size_in_bytes));
    return true;
}

[[nodiscard]] bool test_soa_record_segments_follow_expected_layout() {
    auto plan = DynamicBufferLayoutPlan::create(Type::of<HostDynamicRecord>(),
                                                make_policy(PrecisionPolicy::force_f16),
                                                DynamicBufferLayout::SOA);
    auto segments = plan.record_segments(2u, 1u);
    CHECK(segments.size() == 9u);
    CHECK(total_segment_bytes(segments) == DynamicBufferLayoutCodec<HostDynamicRecord>::storage_bytes(
                                             1u,
                                             make_policy(PrecisionPolicy::force_f16),
                                             DynamicBufferLayout::SOA));

    CHECK(segments[0].storage_begin_byte == 2u);
    CHECK(segments[0].staging_begin_byte == 0u);
    CHECK(segments[0].size_in_bytes == 2u);

    CHECK(segments[1].storage_begin_byte == 12u);
    CHECK(segments[1].staging_begin_byte == 2u);
    CHECK(segments[1].size_in_bytes == 8u);

    CHECK(segments[2].storage_begin_byte == 22u);
    CHECK(segments[2].staging_begin_byte == 10u);
    CHECK(segments[2].size_in_bytes == 2u);

    CHECK(segments[3].storage_begin_byte == 26u);
    CHECK(segments[3].staging_begin_byte == 12u);
    CHECK(segments[3].size_in_bytes == 2u);

    CHECK(segments[4].storage_begin_byte == 30u);
    CHECK(segments[4].staging_begin_byte == 14u);
    CHECK(segments[4].size_in_bytes == 2u);

    CHECK(segments[5].storage_begin_byte == 34u);
    CHECK(segments[5].staging_begin_byte == 16u);
    CHECK(segments[5].size_in_bytes == 2u);

    CHECK(segments[6].storage_begin_byte == 40u);
    CHECK(segments[6].staging_begin_byte == 18u);
    CHECK(segments[6].size_in_bytes == 4u);

    CHECK(segments[7].storage_begin_byte == 48u);
    CHECK(segments[7].staging_begin_byte == 22u);
    CHECK(segments[7].size_in_bytes == 4u);

    CHECK(segments[8].storage_begin_byte == 56u);
    CHECK(segments[8].staging_begin_byte == 26u);
    CHECK(segments[8].size_in_bytes == 4u);
    return true;
}

[[nodiscard]] bool test_soa_field_segments_report_expected_offsets() {
    auto plan = DynamicBufferLayoutPlan::create(Type::of<HostDynamicRecord>(),
                                                make_policy(PrecisionPolicy::force_f16),
                                                DynamicBufferLayout::SOA);

    auto roughness_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<0u>>();
    auto roughness_segments = plan.field_segments(2u, 1u, roughness_path);
    CHECK(roughness_segments.size() == 1u);
    CHECK(roughness_segments[0].storage_begin_byte == 2u);
    CHECK(roughness_segments[0].staging_begin_byte == 0u);
    CHECK(roughness_segments[0].size_in_bytes == 2u);

    auto direction_z_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<1u>, FieldComponentStep<2u>>();
    auto direction_z_segments = plan.field_segments(2u, 1u, direction_z_path);
    CHECK(direction_z_segments.size() == 1u);
    CHECK(direction_z_segments[0].storage_begin_byte == 16u);
    CHECK(direction_z_segments[0].staging_begin_byte == 0u);
    CHECK(direction_z_segments[0].size_in_bytes == 2u);

    auto weight_path = make_typed_field_path<FieldMemberStep<1u>, FieldIndexStep<2u>>();
    auto weight_segments = plan.field_segments(2u, 1u, weight_path);
    CHECK(weight_segments.size() == 1u);
    CHECK(weight_segments[0].storage_begin_byte == 30u);
    CHECK(weight_segments[0].staging_begin_byte == 0u);
    CHECK(weight_segments[0].size_in_bytes == 2u);

    auto matrix_path = make_typed_field_path<FieldMemberStep<2u>, FieldIndexStep<1u>, FieldComponentStep<0u>>();
    auto matrix_segments = plan.field_segments(2u, 1u, matrix_path);
    CHECK(matrix_segments.size() == 1u);
    CHECK(matrix_segments[0].storage_begin_byte == 48u);
    CHECK(matrix_segments[0].staging_begin_byte == 0u);
    CHECK(matrix_segments[0].size_in_bytes == 2u);

    CHECK(plan.field_logical_type(direction_z_path) == Type::of<real>());
    CHECK(plan.field_resolved_type(direction_z_path) == Type::of<half>());
    return true;
}

[[nodiscard]] bool test_host_buffer_record_round_trip() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f16),
                                                         DynamicBufferLayout::AOS,
                                                         2u);
    const auto lhs = make_record(0.0f);
    const auto rhs = make_record(16.0f);
    buffer.write(0u, lhs);
    buffer.write(1u, rhs);
    CHECK(equal_record(buffer.read<HostDynamicRecord>(0u), lhs, 1e-2f));
    CHECK(equal_record(buffer.read<HostDynamicRecord>(1u), rhs, 1e-2f));
    CHECK(buffer.dirty_range().dirty);
    CHECK(buffer.dirty_segments().size() == 1u);
    CHECK(buffer.dirty_range().begin_byte == 0u);
    CHECK(buffer.dirty_range().end_byte == buffer.storage_size_bytes());
    return true;
}

[[nodiscard]] bool test_single_element_external_read_write_usage() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f32),
                                                         DynamicBufferLayout::AOS,
                                                         1u);
    const auto input = make_record(3.0f);
    buffer.write<HostDynamicRecord>(0u, input);

    const auto output = buffer.read<HostDynamicRecord>(0u);
    CHECK(equal_record(output, input, 1e-6f));
    CHECK(buffer.element_count() == 1u);
    CHECK(buffer.dirty_range().dirty);
    CHECK(buffer.dirty_segments().size() == 1u);
    auto record_region = buffer.layout_plan().record_region(0u);
    CHECK(buffer.dirty_range().begin_byte == record_region.begin_byte);
    CHECK(buffer.dirty_range().end_byte == record_region.end_byte);
    return true;
}

[[nodiscard]] bool test_typed_view_single_element_read_write_usage() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f32),
                                                         DynamicBufferLayout::AOS,
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
                                                         DynamicBufferLayout::AOS,
                                                         1u);
    buffer.write(0u, make_record(0.0f));
    buffer.clear_dirty();

    auto roughness_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<0u>>();
    auto roughness_region = buffer.layout_plan().field_region(0u, roughness_path);
    buffer.patch(0u, roughness_path, real{42.5f});
    auto record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.leaf.roughness), 42.5f));
    CHECK(buffer.dirty_segments().size() == 1u);
    CHECK(buffer.dirty_range().begin_byte == roughness_region.begin_byte);
    CHECK(buffer.dirty_range().end_byte == roughness_region.end_byte);

    buffer.clear_dirty();
    auto weight_path = make_typed_field_path<FieldMemberStep<1u>, FieldIndexStep<2u>>();
    auto weight_region = buffer.layout_plan().field_region(0u, weight_path);
    buffer.patch(0u, weight_path, real{15.25f});
    record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.weights[2u]), 15.25f));
    CHECK(buffer.dirty_segments().size() == 1u);
    CHECK(buffer.dirty_range().begin_byte == weight_region.begin_byte);
    CHECK(buffer.dirty_range().end_byte == weight_region.end_byte);

    buffer.clear_dirty();
    auto direction_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<1u>, FieldComponentStep<1u>>();
    auto direction_region = buffer.layout_plan().field_region(0u, direction_path);
    buffer.patch(0u, direction_path, real{7.75f});
    record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.leaf.direction[1u]), 7.75f));
    CHECK(buffer.dirty_segments().size() == 1u);
    CHECK(buffer.dirty_range().begin_byte == direction_region.begin_byte);
    CHECK(buffer.dirty_range().end_byte == direction_region.end_byte);

    buffer.clear_dirty();
    auto matrix_path = make_typed_field_path<FieldMemberStep<2u>, FieldIndexStep<1u>, FieldComponentStep<0u>>();
    auto matrix_region = buffer.layout_plan().field_region(0u, matrix_path);
    buffer.patch(0u, matrix_path, real{9.5f});
    record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.basis[1u][0u]), 9.5f));
    CHECK(buffer.dirty_segments().size() == 1u);
    CHECK(buffer.dirty_range().begin_byte == matrix_region.begin_byte);
    CHECK(buffer.dirty_range().end_byte == matrix_region.end_byte);
    return true;
}

[[nodiscard]] bool test_append_and_upload_view() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f32),
                                                         DynamicBufferLayout::AOS,
                                                         0u);
    vector<HostDynamicRecord> values{make_record(1.0f), make_record(2.0f)};
    buffer.append<HostDynamicRecord>(values);
    CHECK(buffer.element_count() == 2u);
    CHECK(equal_record(buffer.read<HostDynamicRecord>(0u), values[0], 1e-6f));
    CHECK(equal_record(buffer.read<HostDynamicRecord>(1u), values[1], 1e-6f));
    auto upload = buffer.upload_view();
    CHECK(upload.element_count == 2u);
    CHECK(upload.layout == DynamicBufferLayout::AOS);
    CHECK(upload.logical_type == Type::of<HostDynamicRecord>());
    CHECK(upload.resolved_type != nullptr);
    CHECK(upload.bytes.size() == buffer.storage_size_bytes());
    CHECK(upload.dirty_segments.size() == 1u);
    return true;
}

[[nodiscard]] bool test_soa_write_all_and_upload_view() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f16),
                                                         DynamicBufferLayout::SOA,
                                                         0u);
    vector<HostDynamicRecord> values{make_record(4.0f), make_record(12.0f)};
    buffer.write_all<HostDynamicRecord>(values);

    CHECK(buffer.supports_record_access());
    CHECK(buffer.supports_field_patch());
    CHECK(buffer.element_count() == values.size());
    CHECK(buffer.dirty_range().dirty);
    CHECK(buffer.dirty_segments().size() == 1u);
    CHECK(buffer.dirty_range().begin_byte == 0u);
    CHECK(buffer.dirty_range().end_byte == buffer.storage_size_bytes());

    auto expected_bytes = DynamicBufferLayoutCodec<HostDynamicRecord>::storage_bytes(values.size(),
                                                                                     make_policy(PrecisionPolicy::force_f16),
                                                                                     DynamicBufferLayout::SOA);
    CHECK(buffer.storage_size_bytes() == expected_bytes);

    auto upload = buffer.upload_view();
    CHECK(upload.layout == DynamicBufferLayout::SOA);
    CHECK(upload.element_count == values.size());
    CHECK(upload.bytes.size() == expected_bytes);
    CHECK(upload.logical_type == Type::of<HostDynamicRecord>());
    CHECK(upload.resolved_type != nullptr);
    CHECK(upload.dirty_segments.size() == 1u);
    return true;
}

[[nodiscard]] bool test_soa_record_round_trip() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f16),
                                                         DynamicBufferLayout::SOA,
                                                         2u);
    const auto lhs = make_record(20.0f);
    const auto rhs = make_record(36.0f);
    buffer.write(0u, lhs);
    buffer.write(1u, rhs);
    CHECK(equal_record(buffer.read<HostDynamicRecord>(0u), lhs, 1e-2f));
    CHECK(equal_record(buffer.read<HostDynamicRecord>(1u), rhs, 1e-2f));
    CHECK(buffer.dirty_range().dirty);
    return true;
}

[[nodiscard]] bool test_soa_field_patch_updates_target_value() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f16),
                                                         DynamicBufferLayout::SOA,
                                                         1u);
    buffer.write(0u, make_record(0.0f));
    buffer.clear_dirty();

    auto roughness_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<0u>>();
    buffer.patch(0u, roughness_path, real{11.25f});
    auto record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.leaf.roughness), 11.25f));
    CHECK(buffer.dirty_segments().size() == 1u);

    buffer.clear_dirty();
    auto weight_path = make_typed_field_path<FieldMemberStep<1u>, FieldIndexStep<2u>>();
    buffer.patch(0u, weight_path, real{6.5f});
    record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.weights[2u]), 6.5f));
    CHECK(buffer.dirty_segments().size() == 1u);

    buffer.clear_dirty();
    auto matrix_path = make_typed_field_path<FieldMemberStep<2u>, FieldIndexStep<1u>, FieldComponentStep<0u>>();
    buffer.patch(0u, matrix_path, real{17.0f});
    record = buffer.read<HostDynamicRecord>(0u);
    CHECK(close_float(static_cast<float>(record.basis[1u][0u]), 17.0f));
    CHECK(buffer.dirty_range().dirty);
    CHECK(buffer.dirty_segments().size() == 1u);
    return true;
}

[[nodiscard]] bool test_soa_field_patch_is_isolated_per_record() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f16),
                                                         DynamicBufferLayout::SOA,
                                                         2u);
    const auto lhs = make_record(2.0f);
    const auto rhs = make_record(18.0f);
    buffer.write(0u, lhs);
    buffer.write(1u, rhs);
    buffer.clear_dirty();

    auto weight_path = make_typed_field_path<FieldMemberStep<1u>, FieldIndexStep<1u>>();
    buffer.patch(1u, weight_path, real{33.5f});

    const auto updated_lhs = buffer.read<HostDynamicRecord>(0u);
    const auto updated_rhs = buffer.read<HostDynamicRecord>(1u);
    CHECK(equal_record(updated_lhs, lhs, 1e-2f));
    CHECK(close_float(static_cast<float>(updated_rhs.weights[1u]), 33.5f));
    CHECK(close_float(static_cast<float>(updated_rhs.weights[0u]), static_cast<float>(rhs.weights[0u]), 1e-2f));
    CHECK(close_float(static_cast<float>(updated_rhs.weights[2u]), static_cast<float>(rhs.weights[2u]), 1e-2f));
    CHECK(close_float(static_cast<float>(updated_rhs.weights[3u]), static_cast<float>(rhs.weights[3u]), 1e-2f));
    CHECK(buffer.dirty_range().dirty);
    CHECK(buffer.dirty_segments().size() == 1u);
    return true;
}

[[nodiscard]] bool test_soa_append_round_trip() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f32),
                                                         DynamicBufferLayout::SOA,
                                                         0u);
    vector<HostDynamicRecord> first{make_record(1.0f), make_record(2.0f)};
    vector<HostDynamicRecord> second{make_record(50.0f)};
    buffer.append<HostDynamicRecord>(first);
    buffer.append<HostDynamicRecord>(second);
    CHECK(buffer.element_count() == 3u);
    CHECK(equal_record(buffer.read<HostDynamicRecord>(0u), first[0], 1e-6f));
    CHECK(equal_record(buffer.read<HostDynamicRecord>(1u), first[1], 1e-6f));
    CHECK(equal_record(buffer.read<HostDynamicRecord>(2u), second[0], 1e-6f));
    return true;
}

[[nodiscard]] bool test_typed_view_soa_read_write_patch_usage() {
    HostDynamicBuffer buffer = HostDynamicBuffer::create(Type::of<HostDynamicRecord>(),
                                                         make_policy(PrecisionPolicy::force_f32),
                                                         DynamicBufferLayout::SOA,
                                                         0u);
    buffer.resize(2u);
    TypedHostDynamicBufferView<HostDynamicRecord> view{buffer};

    const auto lhs = make_record(7.0f);
    const auto rhs = make_record(27.0f);
    view.write(0u, lhs);
    view.write(1u, rhs);
    auto direction_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<1u>, FieldComponentStep<2u>>();
    view.patch(1u, direction_path, real{88.0f});

    const auto read_lhs = view.read(0u);
    const auto read_rhs = view.read(1u);
    CHECK(equal_record(read_lhs, lhs, 1e-6f));
    CHECK(close_float(static_cast<float>(read_rhs.leaf.direction[2u]), 88.0f));
    CHECK(close_float(static_cast<float>(read_rhs.leaf.direction[0u]), static_cast<float>(rhs.leaf.direction[0u]), 1e-6f));
    CHECK(close_float(static_cast<float>(read_rhs.leaf.direction[1u]), static_cast<float>(rhs.leaf.direction[1u]), 1e-6f));
    return true;
}

}// namespace

int main() {
    bool passed = true;
    passed = test_layout_plan_reports_expected_types() && passed;
    passed = test_aos_record_and_field_regions_report_expected_offsets() && passed;
    passed = test_soa_record_segments_follow_expected_layout() && passed;
    passed = test_soa_field_segments_report_expected_offsets() && passed;
    passed = test_soa_dirty_segments_preserve_disjoint_field_segments() && passed;
    passed = test_host_buffer_record_round_trip() && passed;
    passed = test_single_element_external_read_write_usage() && passed;
    passed = test_typed_view_single_element_read_write_usage() && passed;
    passed = test_field_patch_updates_target_bytes() && passed;
    passed = test_append_and_upload_view() && passed;
    passed = test_soa_write_all_and_upload_view() && passed;
    passed = test_soa_record_round_trip() && passed;
    passed = test_soa_field_patch_updates_target_value() && passed;
    passed = test_soa_field_patch_is_isolated_per_record() && passed;
    passed = test_soa_append_round_trip() && passed;
    passed = test_typed_view_soa_read_write_patch_usage() && passed;
    if (!passed) {
        std::cerr << "host dynamic buffer test failed" << std::endl;
        return 1;
    }
    std::cout << "host dynamic buffer test passed" << std::endl;
    return 0;
}