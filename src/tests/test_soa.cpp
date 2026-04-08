//
// Created by Zero on 2025/03/25.
//

#include <cmath>
#include <cstring>

#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "math/base.h"
#include "core/platform.h"
#include "rhi/context.h"

using namespace ocarina;

struct TestRecord {
    float4 lhs{};
    float4 rhs{};

    TestRecord() = default;
    TestRecord(float4 lhs, float4 rhs) : lhs(lhs), rhs(rhs) {}
};

OC_STRUCT(, TestRecord, lhs, rhs) {
};

static_assert(sizeof(TestRecord) == sizeof(float4) * 2,
              "TestRecord must stay tightly packed for SOA packing");

static bool close_float(float lhs, float rhs) {
    return std::abs(lhs - rhs) < 1e-5f;
}

static bool equal_float4(const float4 &lhs, const float4 &rhs) {
    return close_float(lhs.x, rhs.x) &&
           close_float(lhs.y, rhs.y) &&
           close_float(lhs.z, rhs.z) &&
           close_float(lhs.w, rhs.w);
}

static bool equal_record(const TestRecord &lhs, const TestRecord &rhs) {
    return equal_float4(lhs.lhs, rhs.lhs) && equal_float4(lhs.rhs, rhs.rhs);
}

static TestRecord make_test_record(uint index) {
    float value = static_cast<float>(index);
    return {
        make_float4(value + 0.25f, value + 1.25f, value + 2.25f, value + 3.25f),
        make_float4(-value - 0.5f, -value - 1.5f, -value - 2.5f, -value - 3.5f)};
}

static vector<std::byte> pack_records_soa(const vector<TestRecord> &records) {
    size_t count = records.size();
    vector<std::byte> bytes(count * sizeof(TestRecord));
    size_t lhs_offset = 0u;
    size_t rhs_offset = count * sizeof(float4);
    for (size_t index = 0; index < count; ++index) {
        std::memcpy(bytes.data() + lhs_offset + index * sizeof(float4), &records[index].lhs, sizeof(float4));
        std::memcpy(bytes.data() + rhs_offset + index * sizeof(float4), &records[index].rhs, sizeof(float4));
    }
    return bytes;
}

static vector<TestRecord> unpack_records_soa(const vector<std::byte> &bytes, size_t count) {
    vector<TestRecord> records(count);
    size_t lhs_offset = 0u;
    size_t rhs_offset = count * sizeof(float4);
    for (size_t index = 0; index < count; ++index) {
        std::memcpy(&records[index].lhs, bytes.data() + lhs_offset + index * sizeof(float4), sizeof(float4));
        std::memcpy(&records[index].rhs, bytes.data() + rhs_offset + index * sizeof(float4), sizeof(float4));
    }
    return records;
}

static int test_buffer_storage_basic() {
    std::cout << "=== test_buffer_storage_basic ===" << std::endl;
    static_assert(sizeof(BufferStorage<ByteBufferVar>) >= sizeof(ByteBufferVar),
                  "BufferStorage must be large enough to hold ByteBufferVar");
    static_assert(alignof(BufferStorage<ByteBufferVar>) >= alignof(ByteBufferVar),
                  "BufferStorage must be correctly aligned for ByteBufferVar");
    std::cout << "  PASSED" << std::endl;
    return 0;
}

static int test_buffer_read_write(Device &device, Stream &stream) {
    std::cout << "=== test_buffer_read_write ===" << std::endl;
    int failures = 0;

    constexpr uint count = 64u;
    auto src = device.create_buffer<float4>(count, "test_soa_buffer_src");
    auto dst = device.create_buffer<float4>(count, "test_soa_buffer_dst");

    vector<float4> host_src(count);
    vector<float4> host_dst(count, make_float4(0.f));
    for (uint index = 0; index < count; ++index) {
        float value = static_cast<float>(index);
        host_src[index] = make_float4(value, value + 1.f, value + 2.f, value + 3.f);
    }
    src.upload_immediately(host_src.data());

    Kernel kernel = [&](BufferVar<float4> input, BufferVar<float4> output, Uint n) {
        Uint index = dispatch_id();
        $if(index < n) {
            Float4 value = input.read(index);
            output.write(index, value + make_float4(10.f, 20.f, 30.f, 40.f));
        };
    };

    auto shader = device.compile(kernel, "test_buffer_read_write");
    stream << shader(src, dst, count).dispatch(count)
           << dst.download(host_dst.data())
           << synchronize()
           << commit();

    for (uint index = 0; index < count; ++index) {
        float value = static_cast<float>(index);
        float4 expected = make_float4(value + 10.f, value + 21.f, value + 32.f, value + 43.f);
        if (!equal_float4(host_dst[index], expected)) {
            std::cerr << "  FAIL at index " << index << std::endl;
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "  PASSED" << std::endl;
    }
    return failures;
}

static int test_byte_buffer_read_write(Device &device, Stream &stream) {
    std::cout << "=== test_byte_buffer_read_write ===" << std::endl;
    int failures = 0;

    constexpr uint count = 32u;
    auto src = device.create_byte_buffer(count * sizeof(float4), "test_soa_byte_buffer_src");
    auto dst = device.create_byte_buffer(count * sizeof(float4), "test_soa_byte_buffer_dst");

    vector<float4> host_src(count);
    vector<float4> host_dst(count, make_float4(0.f));
    for (uint index = 0; index < count; ++index) {
        float value = static_cast<float>(index * 2u);
        host_src[index] = make_float4(value + 0.5f, value + 1.5f, value + 2.5f, value + 3.5f);
    }
    src.upload_immediately(host_src.data());

    Kernel kernel = [&](ByteBufferVar input, ByteBufferVar output, Uint n) {
        Uint index = dispatch_id();
        $if(index < n) {
            Uint offset = index * static_cast<uint>(sizeof(float4));
            Float4 value = input.load_as<float4>(offset);
            output.store(offset, value * 2.f);
        };
    };

    auto shader = device.compile(kernel, "test_byte_buffer_read_write");
    stream << shader(src, dst, count).dispatch(count)
           << dst.download(host_dst.data())
           << synchronize()
           << commit();

    for (uint index = 0; index < count; ++index) {
        float4 expected = host_src[index] * 2.f;
        if (!equal_float4(host_dst[index], expected)) {
            std::cerr << "  FAIL at index " << index << std::endl;
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "  PASSED" << std::endl;
    }
    return failures;
}

static int test_aos_view_var_read_write(Device &device, Stream &stream) {
    std::cout << "=== test_aos_view_var_read_write ===" << std::endl;
    int failures = 0;

    constexpr uint count = 24u;
    auto src = device.create_byte_buffer(count * sizeof(TestRecord), "test_soa_aos_src");
    auto dst = device.create_byte_buffer(count * sizeof(TestRecord), "test_soa_aos_dst");

    vector<TestRecord> host_src(count);
    vector<TestRecord> host_dst(count);
    for (uint index = 0; index < count; ++index) {
        host_src[index] = make_test_record(index);
    }
    src.upload_immediately(host_src.data());

    Kernel kernel = [&](ByteBufferVar input, ByteBufferVar output, Uint n) {
        Uint index = dispatch_id();
        $if(index < n) {
            auto src_view = make_aos_view_var<TestRecord>(input);
            auto dst_view = make_aos_view_var<TestRecord>(output);
            Var<TestRecord> value = src_view.read(index);
            dst_view.write(index, value);
        };
    };

    auto shader = device.compile(kernel, "test_aos_view_var_read_write");
    stream << shader(src, dst, count).dispatch(count)
           << dst.download(host_dst.data())
           << synchronize()
           << commit();

    for (uint index = 0; index < count; ++index) {
        if (!equal_record(host_dst[index], host_src[index])) {
            std::cerr << "  FAIL at index " << index << std::endl;
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "  PASSED" << std::endl;
    }
    return failures;
}

static int test_soa_view_var_read_write(Device &device, Stream &stream) {
    std::cout << "=== test_soa_view_var_read_write ===" << std::endl;
    int failures = 0;

    constexpr uint count = 24u;
    auto src = device.create_byte_buffer(count * sizeof(TestRecord), "test_soa_soa_src");
    auto dst = device.create_byte_buffer(count * sizeof(TestRecord), "test_soa_soa_dst");

    vector<TestRecord> host_records(count);
    for (uint index = 0; index < count; ++index) {
        host_records[index] = make_test_record(index + 100u);
    }
    vector<std::byte> host_src = pack_records_soa(host_records);
    vector<std::byte> host_dst(count * sizeof(TestRecord));
    src.upload_immediately(host_src.data());

    Kernel kernel = [&](ByteBufferVar input, ByteBufferVar output, Uint n) {
        Uint index = dispatch_id();
        $if(index < n) {
            auto src_view = make_soa_view_var<TestRecord>(input);
            auto dst_view = make_soa_view_var<TestRecord>(output);
            Var<TestRecord> value = src_view.read(index);
            dst_view.write(index, value);
        };
    };

    auto shader = device.compile(kernel, "test_soa_view_var_read_write");
    stream << shader(src, dst, count).dispatch(count)
           << dst.download(host_dst.data())
           << synchronize()
           << commit();

    vector<TestRecord> unpacked = unpack_records_soa(host_dst, count);
    for (uint index = 0; index < count; ++index) {
        if (!equal_record(unpacked[index], host_records[index])) {
            std::cerr << "  FAIL at index " << index << std::endl;
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "  PASSED" << std::endl;
    }
    return failures;
}

int main(int argc, char *argv[]) {
    int total_failures = 0;

    total_failures += test_buffer_storage_basic();

    RHIContext &context = RHIContext::instance();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    Env::printer().init(device);
    Env::debugger().init(device);

    total_failures += test_buffer_read_write(device, stream);
    total_failures += test_byte_buffer_read_write(device, stream);
    total_failures += test_aos_view_var_read_write(device, stream);
    total_failures += test_soa_view_var_read_write(device, stream);

    std::cout << "\n==================" << std::endl;
    if (total_failures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << total_failures << " FAILURE(S)" << std::endl;
    }
    return total_failures;
}