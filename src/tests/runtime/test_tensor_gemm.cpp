#include <cmath>
#include <cstdlib>
#include <iostream>

#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "rhi/context.h"

using namespace ocarina;

namespace {

[[nodiscard]] bool close_float(float lhs, float rhs, float eps = 1e-5f) {
    return std::abs(lhs - rhs) <= eps;
}

void log_mismatch(const char *label, uint row, uint col, float actual, float expected) {
    std::cerr << "  FAIL [" << label << "] at (" << row << ", " << col << ")"
              << " expected=" << expected
              << " actual=" << actual
              << std::endl;
}

[[nodiscard]] vector<float> host_gemm(const vector<float> &a,
                                      const vector<float> &b,
                                      const vector<float> &c,
                                      uint m,
                                      uint n,
                                      uint k,
                                      bool trans_a,
                                      bool trans_b,
                                      float alpha,
                                      float beta) {
    vector<float> out(m * n, 0.f);
    for (uint row = 0; row < m; ++row) {
        for (uint col = 0; col < n; ++col) {
            float accum = 0.f;
            for (uint inner = 0; inner < k; ++inner) {
                uint a_index = trans_a ? inner * m + row : row * k + inner;
                uint b_index = trans_b ? col * k + inner : inner * n + col;
                accum += a[a_index] * b[b_index];
            }
            out[row * n + col] = alpha * accum + beta * c[row * n + col];
        }
    }
    return out;
}

int verify_matrix(const char *label,
                  const vector<float> &actual,
                  const vector<float> &expected,
                  uint m,
                  uint n,
                  float eps = 1e-5f) {
    int failures = 0;
    for (uint row = 0; row < m; ++row) {
        for (uint col = 0; col < n; ++col) {
            uint index = row * n + col;
            if (!close_float(actual[index], expected[index], eps)) {
                log_mismatch(label, row, col, actual[index], expected[index]);
                ++failures;
            }
        }
    }
    return failures;
}

int test_basic_gemm(Device &device, Stream &stream) {
    std::cout << "=== test_basic_gemm ===" << std::endl;
    constexpr uint m = 2u;
    constexpr uint n = 4u;
    constexpr uint k = 3u;

    vector<float> host_a{
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f};
    vector<float> host_b{
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f};
    vector<float> host_c(m * n, 0.f);
    vector<float> host_out(m * n, -1.f);

    auto a = device.create_buffer<float>(host_a.size(), "test_tensor_gemm_basic_a");
    auto b = device.create_buffer<float>(host_b.size(), "test_tensor_gemm_basic_b");
    auto c = device.create_buffer<float>(host_c.size(), "test_tensor_gemm_basic_c");

    a.upload_immediately(host_a.data());
    b.upload_immediately(host_b.data());
    c.upload_immediately(host_c.data());

    GemmDesc desc{};
    desc.a = make_tensor_view(a, {m, k});
    desc.b = make_tensor_view(b, {k, n});
    desc.c = make_tensor_view(c, {m, n});
    desc.m = m;
    desc.n = n;
    desc.k = k;

    stream << gemm_commands(device, desc, "test_tensor_gemm_basic")
           << c.download(host_out.data())
           << synchronize()
           << commit();

    auto expected = host_gemm(host_a, host_b, host_c, m, n, k, false, false, 1.f, 0.f);
    int failures = verify_matrix("basic_gemm", host_out, expected, m, n);
    if (failures == 0) {
        std::cout << "  PASSED" << std::endl;
    }
    return failures;
}

int test_transposed_gemm(Device &device, Stream &stream) {
    std::cout << "=== test_transposed_gemm ===" << std::endl;
    constexpr uint m = 2u;
    constexpr uint n = 2u;
    constexpr uint k = 3u;

    vector<float> host_a{
        1.f, 4.f,
        2.f, 5.f,
        3.f, 6.f};
    vector<float> host_b{
        1.f, 3.f, 5.f,
        2.f, 4.f, 6.f};
    vector<float> host_c{
        0.5f, 1.0f,
        1.5f, 2.0f};
    vector<float> host_out(host_c.size(), -1.f);

    auto a = device.create_buffer<float>(host_a.size(), "test_tensor_gemm_trans_a");
    auto b = device.create_buffer<float>(host_b.size(), "test_tensor_gemm_trans_b");
    auto c = device.create_buffer<float>(host_c.size(), "test_tensor_gemm_trans_c");

    a.upload_immediately(host_a.data());
    b.upload_immediately(host_b.data());
    c.upload_immediately(host_c.data());

    GemmDesc desc{};
    desc.a = make_tensor_view(a, {k, m});
    desc.b = make_tensor_view(b, {n, k});
    desc.c = make_tensor_view(c, {m, n});
    desc.m = m;
    desc.n = n;
    desc.k = k;
    desc.trans_a = true;
    desc.trans_b = true;
    desc.alpha = 0.5f;
    desc.beta = 2.0f;

    stream << gemm_commands(device, desc, "test_tensor_gemm_transposed")
           << c.download(host_out.data())
           << synchronize()
           << commit();

    auto expected = host_gemm(host_a, host_b, host_c, m, n, k, true, true, desc.alpha, desc.beta);
    int failures = verify_matrix("transposed_gemm", host_out, expected, m, n);
    if (failures == 0) {
        std::cout << "  PASSED" << std::endl;
    }
    return failures;
}

}// namespace

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    RHIContext &context = RHIContext::instance();
    Device device = context.create_device("cuda");
    device.init_rtx();
    Stream stream = device.create_stream();

    int failures = 0;
    failures += test_basic_gemm(device, stream);
    failures += test_transposed_gemm(device, stream);

    stream << synchronize() << commit();
    if (failures != 0) {
        std::cerr << "test_tensor_gemm failed with " << failures << " mismatches." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "test_tensor_gemm PASSED" << std::endl;
    return EXIT_SUCCESS;
}