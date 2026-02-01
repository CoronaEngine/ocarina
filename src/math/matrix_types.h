//
// Created by Zero on 2024/5/20.
//

#pragma once

#include "vector_types.h"

namespace ocarina {

template<typename T, size_t N, size_t M>
struct Matrix {
public:
    static constexpr auto row_num = N;
    static constexpr auto col_num = M;
    static constexpr auto element_num = M * N;
    using scalar_type = T;
    using vector_type = Vector<scalar_type, N>;
    using array_t = array<vector_type, M>;

private:
    array_t cols_{};

public:
    template<typename... Args>
    requires(sizeof...(Args) == M)
    explicit constexpr Matrix(Args &&...args) noexcept
        : cols_(array_t{static_cast<vector_type>(OC_FORWARD(args))...}) {}

    template<typename... Args>
    requires(sizeof...(Args) == element_num && is_all_scalar_v<Args...>)
    explicit constexpr Matrix(Args &&...args) noexcept
        : cols_([&]<size_t... i>(std::index_sequence<i...>,
                                 const array<scalar_type, element_num> &arr) {
              return array_t{vector_type{addressof(arr.data()[i * N])}...};
          }(std::make_index_sequence<M>(), array<scalar_type, element_num>{static_cast<scalar_type>(OC_FORWARD(args))...})) {
    }

    template<typename T, size_t NN, size_t MM>
    requires(NN >= N && MM >= M)
    explicit constexpr Matrix(Matrix<T, NN, MM> mat) noexcept
        : cols_{[&]<size_t... i>(std::index_sequence<i...>) {
              return std::array<Vector<float, N>, M>{Vector<float, N>{mat[i]}...};
          }(std::make_index_sequence<M>())} {}

    constexpr Matrix(scalar_type s = 1) noexcept
        : cols_([&]<size_t... i>(std::index_sequence<i...>) {
              array_t ret{};
              if constexpr (M == N) {
                  ((ret[i][i] = s), ...);
              }
              return ret;
          }(std::make_index_sequence<N>())) {
    }

    [[nodiscard]] constexpr vector_type &operator[](size_t i) noexcept { return cols_[i]; }
    [[nodiscard]] constexpr const vector_type &operator[](size_t i) const noexcept { return cols_[i]; }
};

#define OC_MATRIX_UNARY_FUNC(func)                                   \
    template<typename T, size_t N, size_t M>                         \
    [[nodiscard]] Matrix<T, N, M> func(Matrix<T, N, M> m) noexcept { \
        return [&]<size_t... i>(std::index_sequence<i...>) {         \
            return ocarina::Matrix<T, N, M>(func(m[i])...);          \
        }(std::make_index_sequence<M>());                            \
    }

OC_MATRIX_UNARY_FUNC(rcp)
OC_MATRIX_UNARY_FUNC(abs)
OC_MATRIX_UNARY_FUNC(sqrt)
OC_MATRIX_UNARY_FUNC(sqr)
OC_MATRIX_UNARY_FUNC(sign)
OC_MATRIX_UNARY_FUNC(cos)
OC_MATRIX_UNARY_FUNC(sin)
OC_MATRIX_UNARY_FUNC(tan)
OC_MATRIX_UNARY_FUNC(cosh)
OC_MATRIX_UNARY_FUNC(sinh)
OC_MATRIX_UNARY_FUNC(tanh)
OC_MATRIX_UNARY_FUNC(log)
OC_MATRIX_UNARY_FUNC(log2)
OC_MATRIX_UNARY_FUNC(log10)
OC_MATRIX_UNARY_FUNC(exp)
OC_MATRIX_UNARY_FUNC(exp2)
OC_MATRIX_UNARY_FUNC(asin)
OC_MATRIX_UNARY_FUNC(acos)
OC_MATRIX_UNARY_FUNC(atan)
OC_MATRIX_UNARY_FUNC(asinh)
OC_MATRIX_UNARY_FUNC(acosh)
OC_MATRIX_UNARY_FUNC(atanh)
OC_MATRIX_UNARY_FUNC(floor)
OC_MATRIX_UNARY_FUNC(ceil)
OC_MATRIX_UNARY_FUNC(degrees)
OC_MATRIX_UNARY_FUNC(radians)
OC_MATRIX_UNARY_FUNC(round)
OC_MATRIX_UNARY_FUNC(isnan)
OC_MATRIX_UNARY_FUNC(isinf)
OC_MATRIX_UNARY_FUNC(fract)
OC_MATRIX_UNARY_FUNC(copysign)

#undef OC_MATRIX_UNARY_FUNC

#define OC_MATRIX_BINARY_FUNC(func)                                    \
    template<typename T, size_t N, size_t M>                           \
    [[nodiscard]] Matrix<T, N, M> func(Matrix<T, N, M> lhs,            \
                                       Matrix<T, N, M> rhs) noexcept { \
        return [&]<size_t... i>(std::index_sequence<i...>) {           \
            return ocarina::Matrix<T, N, M>(func(lhs[i], rhs[i])...);  \
        }(std::make_index_sequence<N>());                              \
    }

OC_MATRIX_BINARY_FUNC(max)
OC_MATRIX_BINARY_FUNC(min)
OC_MATRIX_BINARY_FUNC(pow)
OC_MATRIX_BINARY_FUNC(atan2)

#undef OC_MATRIX_BINARY_FUNC

#define OC_MATRIX_TRIPLE_FUNC(func)                                          \
    template<typename T, size_t N, size_t M>                                 \
    [[nodiscard]] Matrix<T, N, M> func(Matrix<T, N, M> t, Matrix<T, N, M> u, \
                                       Matrix<T, N, M> v) noexcept {         \
        return [&]<size_t... i>(std::index_sequence<i...>) {                 \
            return ocarina::Matrix<T, N, M>(func(t[i], u[i], v[i])...);      \
        }(std::make_index_sequence<N>());                                    \
    }

OC_MATRIX_TRIPLE_FUNC(fma)
OC_MATRIX_TRIPLE_FUNC(clamp)
OC_MATRIX_TRIPLE_FUNC(lerp)

#undef OC_MATRIX_TRIPLE_FUNC

}// namespace ocarina

template<typename T, size_t N, size_t M>
[[nodiscard]] constexpr auto operator-(ocarina::Matrix<T, N, M> m) {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ocarina::Matrix<T, N, M>((-m[i])...);
    }(std::make_index_sequence<M>());
}

template<typename T, size_t N, size_t M>
[[nodiscard]] constexpr auto operator*(ocarina::Matrix<T, N, M> m, T s) {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ocarina::Matrix<T, N, M>((m[i] * s)...);
    }(std::make_index_sequence<M>());
}

template<typename T, size_t N, size_t M>
[[nodiscard]] constexpr auto operator*(T s, ocarina::Matrix<T, N, M> m) {
    return m * s;
}

template<typename T, size_t N, size_t M>
[[nodiscard]] constexpr auto operator/(ocarina::Matrix<T, N, M> m, float s) {
    return m * (1.0f / s);
}

template<typename T, size_t N, size_t M>
[[nodiscard]] constexpr auto operator*(ocarina::Matrix<T, N, M> m, ocarina::Vector<T, M> v) noexcept {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ((v[i] * m[i]) + ...);
    }(std::make_index_sequence<M>());
}

template<typename T, size_t N, size_t M, size_t Dim>
[[nodiscard]] constexpr auto operator*(ocarina::Matrix<T, N, Dim> lhs, ocarina::Matrix<T, Dim, M> rhs) noexcept {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ocarina::Matrix<T, N, M>((lhs * rhs[i])...);
    }(std::make_index_sequence<M>());
}

template<typename T, size_t N, size_t M>
[[nodiscard]] constexpr auto operator+(ocarina::Matrix<T, N, M> lhs,
                                       ocarina::Matrix<T, N, M> rhs) noexcept {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ocarina::Matrix<T, N, M>(lhs[i] + rhs[i]...);
    }(std::make_index_sequence<M>());
}

template<typename T, size_t N, size_t M>
[[nodiscard]] constexpr auto operator-(ocarina::Matrix<T, N, M> lhs, ocarina::Matrix<T, N, M> rhs) noexcept {
    return lhs + (-rhs);
}

namespace ocarina {

template<typename T, size_t N, size_t M, typename... Args>
requires is_all_basic_v<Args...>
[[nodiscard]] constexpr Matrix<T, N, M> make_float(Args &&...args) noexcept {
    return Matrix<T, N, M>(OC_FORWARD(args)...);
}

#define OC_MAKE_MATRIX_(N, M)                                                           \
    using float##N##x##M = Matrix<float, N, M>;                                         \
    template<typename... Args>                                                          \
    requires is_all_basic_v<Args...>                                                    \
    [[nodiscard]] constexpr float##N##x##M make_float##N##x##M(Args &&...args) {        \
        return make_float<float, N, M>(OC_FORWARD(args)...);                            \
    }                                                                                   \
    template<typename T, size_t NN, size_t MM>                                          \
    [[nodiscard]] constexpr float##N##x##M make_float##N##x##M(Matrix<T, NN, MM> mat) { \
        return float##N##x##M(mat);                                                     \
    }

OC_MAKE_MATRIX_(2, 2)
OC_MAKE_MATRIX_(2, 3)
OC_MAKE_MATRIX_(2, 4)
OC_MAKE_MATRIX_(3, 2)
OC_MAKE_MATRIX_(3, 3)
OC_MAKE_MATRIX_(3, 4)
OC_MAKE_MATRIX_(4, 2)
OC_MAKE_MATRIX_(4, 3)
OC_MAKE_MATRIX_(4, 4)

template<typename T, size_t M, size_t N>
[[nodiscard]] constexpr Matrix<T, N, M> transpose(const Matrix<T, M, N> &mat) noexcept {
    Matrix<T, N, M> ret = make_float<T, N, M>();
    auto func_m = [&]<size_t... m>(size_t i, std::index_sequence<m...>) {
        return Vector<float, N>((mat[m][i])...);
    };
    auto func_n = [&]<size_t... n>(std::index_sequence<n...>) {
        return Matrix<T, N, M>(func_m(n, std::make_index_sequence<N>())...);
    };
    return func_n(std::make_index_sequence<M>());
}

[[nodiscard]] constexpr auto make_float3x3(float2x2 m) noexcept {
    return float3x3{make_float3(m[0], 0.0f),
                    make_float3(m[1], 0.0f),
                    make_float3(0.f, 0.f, 1.0f)};
}

[[nodiscard]] constexpr auto make_float4x4(float2x2 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f, 0.0f),
                    make_float4(m[1], 0.0f, 0.0f),
                    float4{0.0f, 0.0f, 1.0f, 0.0f},
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

[[nodiscard]] constexpr auto make_float4x4(float4x3 m) noexcept {
    return float4x4{m[0], m[1], m[2],
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

[[nodiscard]] constexpr auto make_float4x4(float3x4 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f),
                    make_float4(m[1], 0.0f),
                    make_float4(m[2], 0.0f),
                    make_float4(m[3], 1.0f)};
}

[[nodiscard]] constexpr auto make_float4x4(float3x3 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f),
                    make_float4(m[1], 0.0f),
                    make_float4(m[2], 0.0f),
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

}// namespace ocarina