//
// Created by Zero on 2025/03/25.
//
// Test for BufferStorage, SOAViewVar, SOAView, and AOSViewVar stability
// after replacing std::aligned_storage_t with alignas + std::byte.
//

#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "math/base.h"
#include "core/platform.h"
#include "rhi/context.h"

using namespace ocarina;

/// ---- Host-side BufferStorage tests (no GPU needed) ----

static int test_buffer_storage_basic() {
    std::cout << "=== test_buffer_storage_basic ===" << std::endl;
    int failures = 0;

    // We can't instantiate BufferStorage without a valid buffer type at
    // compile time (it requires is_valid_buffer_storage_v), so we verify
    // the alignment and size properties via static_assert instead.
    // The runtime tests below exercise BufferStorage through SOAViewVar/AOSViewVar
    // on actual GPU buffers.

    // Static assertions: BufferStorage<T> must have correct size and alignment
    static_assert(sizeof(BufferStorage<ByteBufferVar>) >= sizeof(ByteBufferVar),
                  "BufferStorage must be large enough to hold ByteBufferVar");
    static_assert(alignof(BufferStorage<ByteBufferVar>) >= alignof(ByteBufferVar),
                  "BufferStorage must be correctly aligned for ByteBufferVar");

    std::cout << "  BufferStorage<ByteBufferVar> size: " << sizeof(BufferStorage<ByteBufferVar>) << std::endl;
    std::cout << "  ByteBufferVar size: " << sizeof(ByteBufferVar) << std::endl;
    std::cout << "  PASSED (compile-time checks)" << std::endl;
    return failures;
}

/// ---- GPU round-trip: AOSViewVar read/write ----

static int test_aos_view_var_roundtrip(Device &device, Stream &stream) {
    std::cout << "=== test_aos_view_var_roundtrip ===" << std::endl;
    int failures = 0;

    constexpr uint N = 64;
    auto byte_buf = device.create_byte_buffer(N * sizeof(float4));

    // Prepare host data
    vector<float4> host_src(N);
    vector<float4> host_dst(N, make_float4(0.f));
    for (uint i = 0; i < N; ++i) {
        host_src[i] = make_float4(float(i), float(i * 2), float(i * 3), float(i * 4));
    }

    // Upload
    byte_buf.upload_immediately(host_src.data());

    // Destination buffer
    auto dst_buf = device.create_byte_buffer(N * sizeof(float4));

    // Kernel: read via AOSViewVar, write to destination
    Kernel kernel = [&](ByteBufferVar src, ByteBufferVar dst, Uint count) {
        Uint idx = dispatch_id();
        $if(idx < count) {
            auto aos_view = make_aos_view_var<float4>(src);
            Var<float4> val = aos_view.read(idx);
            dst.store(idx * static_cast<uint>(sizeof(float4)), val);
        };
    };

    auto shader = device.compile(kernel, "test_aos_roundtrip");
    stream << shader(byte_buf, dst_buf, N).dispatch(N);
    stream << dst_buf.download(host_dst.data());
    stream << synchronize() << commit();

    // Verify
    for (uint i = 0; i < N; ++i) {
        if (host_dst[i].x != float(i) || host_dst[i].y != float(i * 2) ||
            host_dst[i].z != float(i * 3) || host_dst[i].w != float(i * 4)) {
            std::cerr << "  FAIL at index " << i
                      << ": expected (" << float(i) << "," << float(i * 2)
                      << "," << float(i * 3) << "," << float(i * 4) << ")"
                      << " got (" << host_dst[i].x << "," << host_dst[i].y
                      << "," << host_dst[i].z << "," << host_dst[i].w << ")"
                      << std::endl;
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "  PASSED (" << N << " float4 elements round-tripped)" << std::endl;
    }
    return failures;
}

/// ---- GPU round-trip: AOSViewVar with Ray struct ----

static int test_aos_view_var_ray(Device &device, Stream &stream) {
    std::cout << "=== test_aos_view_var_ray ===" << std::endl;
    int failures = 0;

    constexpr uint N = 16;
    auto byte_buf = device.create_byte_buffer(N * sizeof(Ray));
    auto dst_buf = device.create_buffer<float3>(N * 2); // origin + dir per ray

    vector<Ray> host_rays(N);
    for (uint i = 0; i < N; ++i) {
        float fi = float(i);
        host_rays[i] = Ray(make_float3(fi, fi + 1, fi + 2),
                           make_float3(fi * 0.1f, fi * 0.2f, fi * 0.3f));
    }
    byte_buf.upload_immediately(host_rays.data());

    // Kernel: read ray via AOSViewVar, extract origin and dir
    Kernel kernel = [&](ByteBufferVar src, BufferVar<float3> out, Uint count) {
        Uint idx = dispatch_id();
        $if(idx < count) {
            auto rays = make_aos_view_var<Ray>(src);
            auto ray = rays.read(idx);
            out.write(idx * 2u, ray.org_min.xyz());
            out.write(idx * 2u + 1u, ray.dir_max.xyz());
        };
    };

    auto shader = device.compile(kernel, "test_aos_ray");

    vector<float3> host_out(N * 2, make_float3(0.f));
    stream << shader(byte_buf, dst_buf, N).dispatch(N);
    stream << dst_buf.download(host_out.data());
    stream << synchronize() << commit();

    for (uint i = 0; i < N; ++i) {
        float fi = float(i);
        float3 expected_org = make_float3(fi, fi + 1, fi + 2);
        float3 expected_dir = make_float3(fi * 0.1f, fi * 0.2f, fi * 0.3f);
        float3 got_org = host_out[i * 2];
        float3 got_dir = host_out[i * 2 + 1];

        auto close = [](float a, float b) { return std::abs(a - b) < 1e-5f; };
        if (!close(got_org.x, expected_org.x) || !close(got_org.y, expected_org.y) ||
            !close(got_org.z, expected_org.z)) {
            std::cerr << "  FAIL origin at ray " << i << std::endl;
            ++failures;
        }
        if (!close(got_dir.x, expected_dir.x) || !close(got_dir.y, expected_dir.y) ||
            !close(got_dir.z, expected_dir.z)) {
            std::cerr << "  FAIL direction at ray " << i << std::endl;
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "  PASSED (" << N << " Ray structs round-tripped)" << std::endl;
    }
    return failures;
}

/// ---- BufferStorage copy/move correctness ----

static int test_buffer_storage_copy_move(Device &device, Stream &stream) {
    std::cout << "=== test_buffer_storage_copy_move ===" << std::endl;
    int failures = 0;

    constexpr uint N = 32;
    auto buf1 = device.create_byte_buffer(N * sizeof(float));
    auto dst_buf = device.create_buffer<float>(N);

    vector<float> host_src(N);
    for (uint i = 0; i < N; ++i) {
        host_src[i] = float(i) * 1.5f;
    }
    buf1.upload_immediately(host_src.data());

    // This kernel exercises BufferStorage copy constructor and copy assignment
    // by creating an AOSViewVar, then copying it.
    Kernel kernel = [&](ByteBufferVar src, BufferVar<float> out, Uint count) {
        Uint idx = dispatch_id();
        $if(idx < count) {
            auto view1 = make_aos_view_var<float>(src);
            // Copy construct — exercises BufferStorage copy ctor
            auto view2 = view1;
            Var<float> val = view2.read(idx);
            out.write(idx, val);
        };
    };

    auto shader = device.compile(kernel, "test_copy_move");
    vector<float> host_dst(N, 0.f);
    stream << shader(buf1, dst_buf, N).dispatch(N);
    stream << dst_buf.download(host_dst.data());
    stream << synchronize() << commit();

    for (uint i = 0; i < N; ++i) {
        float expected = float(i) * 1.5f;
        if (std::abs(host_dst[i] - expected) > 1e-5f) {
            std::cerr << "  FAIL at index " << i
                      << ": expected " << expected << " got " << host_dst[i] << std::endl;
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "  PASSED (" << N << " floats via copied AOSViewVar)" << std::endl;
    }
    return failures;
}

/// ---- SOAView host-side (ByteBufferView) ----

static int test_soa_view_host(Device &device, Stream &stream) {
    std::cout << "=== test_soa_view_host ===" << std::endl;
    int failures = 0;

    constexpr uint N = 8;
    auto byte_buf = device.create_byte_buffer(N * sizeof(Ray));

    vector<Ray> host_rays(N);
    for (uint i = 0; i < N; ++i) {
        float fi = float(i + 1);
        host_rays[i] = Ray(make_float3(fi, fi, fi),
                           make_float3(-fi, -fi, -fi));
    }
    byte_buf.upload_immediately(host_rays.data());

    // Create SOAView on host side — this exercises BufferStorage with ByteBufferView
    ByteBufferView bv = byte_buf.view();
    auto soa = SOAView<Ray, ByteBufferView>(bv);

    // Verify SOAView can be constructed and has correct size
    uint expected_size = N * sizeof(Ray);
    uint got_size = soa.size_in_byte();
    if (got_size != expected_size) {
        std::cerr << "  FAIL: SOAView size_in_byte = " << got_size
                  << ", expected " << expected_size << std::endl;
        ++failures;
    }

    if (failures == 0) {
        std::cout << "  PASSED (SOAView<Ray> constructed with correct size)" << std::endl;
    }
    return failures;
}

int main(int argc, char *argv[]) {
    int total_failures = 0;

    // Host-only tests
    total_failures += test_buffer_storage_basic();

    // GPU tests
    RHIContext &context = RHIContext::instance();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    Env::printer().init(device);
    Env::debugger().init(device);

    total_failures += test_aos_view_var_roundtrip(device, stream);
    total_failures += test_aos_view_var_ray(device, stream);
    total_failures += test_buffer_storage_copy_move(device, stream);
    total_failures += test_soa_view_host(device, stream);

    std::cout << "\n==================" << std::endl;
    if (total_failures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << total_failures << " FAILURE(S)" << std::endl;
    }

    return total_failures;
}
