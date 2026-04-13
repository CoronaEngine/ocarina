//
// Created by Z on 2026/4/13.
//

#include <iostream>

#include "core/dynamic_buffer/dynamic_buffer_gpu_bridge.h"
#include "core/type_desc.h"
#include "math/real.h"
#include "rhi/context.h"

using namespace ocarina;

struct GpuBridgeLeaf {
    real roughness;
    Vector<real, 3> direction;
};

OC_MAKE_STRUCT_REFLECTION(GpuBridgeLeaf, roughness, direction)
OC_MAKE_STRUCT_DESC(GpuBridgeLeaf, roughness, direction)
OC_MAKE_STRUCT_IS_DYNAMIC(GpuBridgeLeaf, roughness, direction)

struct GpuBridgeRecord {
    GpuBridgeLeaf leaf;
    ocarina::array<real, 4> weights;
    Matrix<real, 2, 2> basis;
    float extra;
};

OC_MAKE_STRUCT_REFLECTION(GpuBridgeRecord, leaf, weights, basis, extra)
OC_MAKE_STRUCT_DESC(GpuBridgeRecord, leaf, weights, basis, extra)
OC_MAKE_STRUCT_IS_DYNAMIC(GpuBridgeRecord, leaf, weights, basis, extra)

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

[[nodiscard]] StoragePrecisionPolicy make_policy(PrecisionPolicy policy) {
    return StoragePrecisionPolicy{.policy = policy, .allow_real_in_storage = true};
}

[[nodiscard]] GpuBridgeRecord make_record(float base) {
    return {
        .leaf = GpuBridgeLeaf{
            .roughness = real{base + 0.25f},
            .direction = Vector<real, 3>{real{base + 1.0f}, real{base + 2.0f}, real{base + 3.0f}}},
        .weights = {real{base + 4.0f}, real{base + 5.0f}, real{base + 6.0f}, real{base + 7.0f}},
        .basis = Matrix<real, 2, 2>{Vector<real, 2>{real{base + 8.0f}, real{base + 9.0f}},
                                     Vector<real, 2>{real{base + 10.0f}, real{base + 11.0f}}},
        .extra = base + 12.0f};
}

[[nodiscard]] bool equal_bytes(span<const std::byte> lhs,
                               span<const std::byte> rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t index = 0; index < lhs.size(); ++index) {
        if (lhs[index] != rhs[index]) {
            return false;
        }
    }
    return true;
}

template<typename T>
[[nodiscard]] bool device_matches_host(const DynamicBufferGpuBridge &bridge,
                                       const HostDynamicBuffer<T> &buffer) {
    vector<std::byte> downloaded(buffer.storage_size_bytes());
    bridge.device_buffer().download_immediately(downloaded.data());
    return equal_bytes(buffer.bytes(), downloaded);
}

[[nodiscard]] bool test_aos_upload_immediately_round_trips_full_buffer(Device &device) {
    auto buffer = HostDynamicBuffer<GpuBridgeRecord>::create(make_policy(PrecisionPolicy::force_f16));
    vector<GpuBridgeRecord> records = {make_record(0.f), make_record(20.f)};
    buffer.write_all(span<const GpuBridgeRecord>(records.data(), records.size()));

    DynamicBufferGpuBridge bridge{"test_dynamic_buffer_gpu_bridge_aos"};
    bridge.upload_immediately(device, buffer);

    CHECK(!bridge.needs_upload(buffer));
    CHECK(!buffer.dirty_range().dirty);
    CHECK(buffer.dirty_segments().empty());
    CHECK(bridge.state().byte_capacity == buffer.storage_size_bytes());
    CHECK(bridge.state().uploaded_generation == buffer.generation());
    CHECK(bridge.state().last_upload_was_full);
    CHECK(bridge.state().last_upload_segment_count == 1u);
    CHECK(bridge.state().last_uploaded_bytes == buffer.storage_size_bytes());
    CHECK(device_matches_host(bridge, buffer));
    return true;
}

[[nodiscard]] bool test_soa_dirty_segments_upload_only_changed_columns(Device &device) {
    auto buffer = HostDynamicBuffer<GpuBridgeRecord>::create(make_policy(PrecisionPolicy::force_f16),
                                                             3u);
    vector<GpuBridgeRecord> records = {make_record(3.f), make_record(30.f), make_record(60.f)};
    buffer.write_all(span<const GpuBridgeRecord>(records.data(), records.size()));

    DynamicBufferGpuBridge bridge{"test_dynamic_buffer_gpu_bridge_partial"};
    bridge.upload_immediately(device, buffer);

    auto roughness_path = make_typed_field_path<FieldMemberStep<0u>, FieldMemberStep<0u>>();
    auto extra_path = make_typed_field_path<FieldMemberStep<3u>>();
    buffer.patch(0u, roughness_path, real{91.5f});
    buffer.patch(2u, extra_path, 123.25f);

    vector<ByteRegion> dirty_segments(buffer.dirty_segments().begin(), buffer.dirty_segments().end());
    CHECK(dirty_segments.size() == 2u);
    size_t dirty_bytes = 0u;
    for (const auto &segment : dirty_segments) {
        dirty_bytes += segment.size();
    }

    bridge.upload_immediately(device, buffer);

    CHECK(!bridge.needs_upload(buffer));
    CHECK(!buffer.dirty_range().dirty);
    CHECK(buffer.dirty_segments().empty());
    CHECK(!bridge.state().last_upload_was_full);
    CHECK(bridge.state().last_upload_segment_count == dirty_segments.size());
    CHECK(bridge.state().last_uploaded_bytes == dirty_bytes);
    CHECK(device_matches_host(bridge, buffer));
    return true;
}

}// namespace

int main() {
    RHIContext &context = RHIContext::instance();
    Device device = context.create_device("cuda");

    bool passed = true;
    passed = test_aos_upload_immediately_round_trips_full_buffer(device) && passed;
    passed = test_soa_dirty_segments_upload_only_changed_columns(device) && passed;

    if (!passed) {
        std::cerr << "dynamic buffer gpu bridge test failed" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "dynamic buffer gpu bridge test passed" << std::endl;
    return EXIT_SUCCESS;
}