
#pragma once

namespace ocarina {

template<size_t N, size_t M>
struct Matrix {
public:
    static constexpr auto RowNum = N;
    static constexpr auto ColNum = M;
    static constexpr auto ElementNum = M * N;
    using scalar_type = float;
    using vector_type = Vector<scalar_type, N>;
    using array_t = oc_array<vector_type, M>;

private:
    array_t cols_{};

public:
    template<size_t... i>
    [[nodiscard]] static constexpr array_t diagonal_helper(index_sequence<i...>, scalar_type s) noexcept {
        array_t ret{};
        if constexpr (M == N) {
            ((ret[i][i] = s), ...);
        }
        return ret;
    }
    constexpr Matrix(scalar_type s = 1) noexcept
        : cols_(diagonal_helper(make_index_sequence<N>(), s)) {
    }

    template<typename... Args, enable_if_t<sizeof...(Args) == M, int> = 0>
    constexpr Matrix(Args... args) noexcept : cols_(array_t{args...}) {}

    template<size_t NN, size_t MM, size_t... i>
    [[nodiscard]] static constexpr auto construct_helper(Matrix<NN, MM> mat, index_sequence<i...>) {
        return oc_array<Vector<float, N>, M>{Vector<float, N>{mat[i]}...};
    }

    template<size_t NN, size_t MM, enable_if_t<(NN >= N && MM >= M), int> = 0>
    explicit constexpr Matrix(Matrix<NN, MM> mat) noexcept
        : cols_{construct_helper(mat, make_index_sequence<M>())} {}

    template<size_t... i>
    [[nodiscard]] static constexpr array_t construct_helper(ocarina::index_sequence<i...>,
                                                            const oc_array<scalar_type, ElementNum> &arr) {
        return array_t{vector_type{&(arr.data()[i * M])}...};
    }

    template<typename... Args, enable_if_t<sizeof...(Args) == ElementNum, int> = 0>
    explicit constexpr Matrix(Args &&...args) noexcept
        : cols_(construct_helper(ocarina::make_index_sequence<N>(),
                                 oc_array<scalar_type, ElementNum>{oc_static_cast<scalar_type>(args)...})) {}

    [[nodiscard]] constexpr vector_type &operator[](size_t i) noexcept { return cols_[i]; }
    [[nodiscard]] constexpr const vector_type &operator[](size_t i) const noexcept { return cols_[i]; }
};

template<size_t N, size_t M, typename... Args>
[[nodiscard]] constexpr Matrix<N, M> make_float(Args &&...args) noexcept {
    return Matrix<N, M>(args...);
}

#define OC_MAKE_MATRIX(N, M)                                                         \
    using float##N##x##M = Matrix<N, M>;                                             \
    template<typename... Args>                                                       \
    [[nodiscard]] constexpr float##N##x##M make_float##N##x##M(Args &&...args) {     \
        return make_float<N, M>(args...);                                            \
    }                                                                                \
    template<size_t NN, size_t MM>                                                   \
    [[nodiscard]] constexpr float##N##x##M make_float##N##x##M(Matrix<NN, MM> mat) { \
        return float##N##x##M(mat);                                                  \
    }

OC_MAKE_MATRIX(2, 2)
OC_MAKE_MATRIX(2, 3)
OC_MAKE_MATRIX(2, 4)
OC_MAKE_MATRIX(3, 2)
OC_MAKE_MATRIX(3, 3)
OC_MAKE_MATRIX(3, 4)
OC_MAKE_MATRIX(4, 2)
OC_MAKE_MATRIX(4, 3)
OC_MAKE_MATRIX(4, 4)

#undef OC_MAKE_MATRIX

[[nodiscard]] constexpr auto make_float3x3(float2x2 m) noexcept {
    return float3x3{float3(m[0].x, m[0].y, 0.0f),
                    float3(m[1].x, m[1].y, 0.0f),
                    float3(0.f, 0.f, 1.0f)};
}

[[nodiscard]] constexpr auto make_float4x4(float2x2 m) noexcept {
    return float4x4{float4(m[0].x, m[0].y, 0.0f, 0.0f),
                    float4(m[1].x, m[1].y, 0.0f, 0.0f),
                    float4{0.0f, 0.0f, 1.0f, 0.0f},
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

[[nodiscard]] constexpr auto make_float4x4(float4x3 m) noexcept {
    return float4x4{m[0], m[1], m[2],
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

[[nodiscard]] constexpr auto make_float4x4(float3x4 m) noexcept {
    return float4x4{float4(m[0].x, m[0].y, m[0].z, 0.0f),
                    float4(m[1].x, m[1].y, m[1].z, 0.0f),
                    float4(m[2].x, m[2].y, m[2].z, 0.0f),
                    float4{m[3].x, m[3].y, m[3].z, 1.0f}};
}

[[nodiscard]] constexpr auto make_float4x4(float3x3 m) noexcept {
    return float4x4{float4(m[0].x, m[0].y, m[0].z, 0.0f),
                    float4(m[1].x, m[1].y, m[1].z, 0.0f),
                    float4(m[2].x, m[2].y, m[2].z, 0.0f),
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

template<size_t N, size_t M, size_t... m>
constexpr Vector<float, N> transpose_helper_m(const Matrix<M, N> &mat, size_t i, index_sequence<m...>) {
    return Vector<float, N>((mat[m][i])...);
}

template<size_t N, size_t M, size_t... n>
constexpr Matrix<N, M> transpose_helper_n(const Matrix<M, N> &mat, index_sequence<n...>) {
    return Matrix<N, M>(transpose_helper_m(mat, n, make_index_sequence<N>())...);
}

template<size_t N, size_t M>
[[nodiscard]] constexpr Matrix<N, M> transpose(const Matrix<M, N> &mat) noexcept {
    return transpose_helper_n(mat, make_index_sequence<M>());
}

}// namespace ocarina

namespace ocarina {
template<size_t N, size_t M, size_t... i>
__device__ constexpr auto negate_matrix_impl(const ocarina::Matrix<N, M> &m, ocarina::index_sequence<i...>) {
    return ocarina::Matrix<N, M>{(-m[i])...};
}
}// namespace ocarina

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator-(ocarina::Matrix<N, M> m) {
    return ocarina::negate_matrix_impl(m, ocarina::make_index_sequence<N>());
}

namespace ocarina {
template<size_t N, size_t M, size_t... i>
__device__ constexpr auto multiply_impl(const ocarina::Matrix<N, M> &m, float s, ocarina::index_sequence<i...>) {
    return ocarina::Matrix<N, M>((m[i] * s)...);
}
}// namespace ocarina

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator*(ocarina::Matrix<N, M> m, float s) {
    return ocarina::multiply_impl(m, s, ocarina::make_index_sequence<N>());
}

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator*(float s, ocarina::Matrix<N, M> m) {
    return m * s;
}

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator/(ocarina::Matrix<N, M> m, float s) {
    return m * (1.0f / s);
}

namespace ocarina {
template<size_t N, size_t M, size_t... i>
__device__ constexpr auto multiply_impl(const ocarina::Matrix<N, M> &m, const ocarina::Vector<float, M> &v,
                                        ocarina::index_sequence<i...>) noexcept {
    return ((v[i] * m[i]) + ...);
}
}// namespace ocarina

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator*(ocarina::Matrix<N, M> m, ocarina::Vector<float, M> v) noexcept {
    return ocarina::multiply_impl(m, v, ocarina::make_index_sequence<M>());
}

namespace ocarina {
template<size_t N, size_t M, size_t Dim, size_t... i>
__device__ constexpr auto multiply_matrices_impl(const ocarina::Matrix<N, Dim> &lhs, const ocarina::Matrix<Dim, M> &rhs,
                                                 ocarina::index_sequence<i...>) noexcept {
    return ocarina::Matrix<N, M>{(lhs * rhs[i])...};
}
}// namespace ocarina

template<size_t N, size_t M, size_t Dim>
[[nodiscard]] __device__ constexpr auto operator*(ocarina::Matrix<N, Dim> lhs, ocarina::Matrix<Dim, M> rhs) noexcept {
    return ocarina::multiply_matrices_impl(lhs, rhs, ocarina::make_index_sequence<M>());
}

namespace ocarina {
template<size_t N, size_t M, size_t... i>
__device__ constexpr auto add_matrices_impl(const ocarina::Matrix<N, M> &lhs, const ocarina::Matrix<N, M> &rhs,
                                            ocarina::index_sequence<i...>) noexcept {
    return ocarina::Matrix<N, M>{(lhs[i] + rhs[i])...};
}
}// namespace ocarina

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator+(ocarina::Matrix<N, M> lhs, ocarina::Matrix<N, M> rhs) noexcept {
    return ocarina::add_matrices_impl(lhs, rhs, ocarina::make_index_sequence<N>());
}

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator-(ocarina::Matrix<N, M> lhs, ocarina::Matrix<N, M> rhs) noexcept {
    return lhs + (-rhs);
}

#define OC_MAKE_MATRIX(N, M)                                                         \
    using oc_float##N##x##M = ocarina::Matrix<N, M>;                                 \
    template<typename... Args>                                                       \
    [[nodiscard]] __device__ constexpr auto oc_make_float##N##x##M(Args &&...args) { \
        return ocarina::make_float##N##x##M(args...);                                \
    }

OC_MAKE_MATRIX(2, 2)
OC_MAKE_MATRIX(2, 3)
OC_MAKE_MATRIX(2, 4)
OC_MAKE_MATRIX(3, 2)
OC_MAKE_MATRIX(3, 3)
OC_MAKE_MATRIX(3, 4)
OC_MAKE_MATRIX(4, 2)
OC_MAKE_MATRIX(4, 3)
OC_MAKE_MATRIX(4, 4)

template<size_t N, size_t M>
[[nodiscard]] constexpr ocarina::Matrix<M, N> oc_transpose(const ocarina::Matrix<N, M> &mat) noexcept {
    return ocarina::transpose(mat);
}

#undef OC_MAKE_MATRIX
