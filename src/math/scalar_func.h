//
// Created by Zero on 2024/5/16.
//

#pragma once

#include "half.h"
#include "math/constants.h"
#include "core/concepts.h"
#include <numeric>

namespace ocarina {

// math
using std::abs;
using std::acos;
using std::acosh;
using std::asin;
using std::asinh;
using std::atan;
using std::atan2;
using std::atanh;
using std::ceil;
using std::copysign;
using std::cos;
using std::cosh;
using std::exp;
using std::exp2;
using std::floor;
using std::fma;
using std::fmod;
using std::log;
using std::log10;
using std::log2;
using std::pow;
using std::round;
using std::roundf;
using std::sin;
using std::sinh;
using std::tan;
using std::tanh;

template<typename T>
requires ocarina::is_scalar_v<T>
[[nodiscard]] constexpr T max(T a, T b) noexcept {
    return std::max(a, b);
}

[[nodiscard]] constexpr half max(half a, half b) noexcept {
    return half2float(a) > half2float(b) ? a : b;
}

template<typename T>
requires ocarina::is_scalar_v<T>
[[nodiscard]] constexpr auto min(T a, T b) noexcept {
    return std::min(a, b);
}

[[nodiscard]] constexpr half min(half a, half b) noexcept {
    return half2float(a) < half2float(b) ? a : b;
}

template<typename T, typename F>
requires concepts::selectable<T, F>
[[nodiscard]] constexpr auto select(bool pred, T &&t, F &&f) noexcept {
    using ret_type = decltype(std::declval<T>() + std::declval<F>());
    return pred ? ret_type(t) : ret_type(f);
}

template<typename T>
requires std::is_unsigned_v<T> && (sizeof(T) == 4u || sizeof(T) == 8u)
[[nodiscard]] constexpr auto next_pow2(T v) noexcept {
    v--;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    if constexpr (sizeof(T) == 8u) { v |= v >> 32u; }
    return v + 1u;
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr T sign(T val) {
    return select(val >= 0, T(1), T(-1));
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr auto
radians(const T &deg) noexcept {
    return deg * (constants::Pi / 180.0f);
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr auto
degrees(T rad) noexcept {
    return rad * (constants::InvPi * 180.0f);
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr auto
rcp(const T &v) {
    return T(1.f) / v;
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr auto
sqrt(const T &v) {
    if constexpr (is_half_v<T>) {
        return std::sqrt(static_cast<float>(v));
    } else {
        return std::sqrt(v);
    }
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr auto
rsqrt(const T &v) {
    return rcp(sqrt(v));
}

template<typename T>
requires is_floating_point_v<T>
[[nodiscard]] constexpr auto
fract(const T &v) {
    return v - floorf(v);
}

[[nodiscard]] constexpr auto fract(half v) {
    return float2half(fract(half2float(v)));
}

[[nodiscard]] inline float mod(float x, float y) {
    return x - y * floor(x / y);
}

[[nodiscard]] inline half mod(half x, half y) {
    return float2half(x - y * floor(half2float(x / y)));
}

template<typename T>
//requires is_scalar_v<T>
[[nodiscard]] auto saturate(const T &f) { return min(1.f, max(0.f, f)); }

template<typename T>
requires is_scalar_v<T>
OC_NODISCARD constexpr auto sqr(const T &v) {
    return v * v;
}

[[nodiscard]] inline bool isnan(float x) noexcept {
    auto u = 0u;
    ::memcpy(&u, &x, sizeof(float));
    return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u;
}

[[nodiscard]] inline bool isnan(half x) noexcept {
    return x.is_nan();
}

[[nodiscard]] inline bool isinf(float x) noexcept {
    auto u = 0u;
    ::memcpy(&u, &x, sizeof(float));
    return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) == 0u;
}

[[nodiscard]] inline bool isinf(half x) noexcept {
    return x.is_inf();
}

template<typename X, typename A, typename B>
requires is_all_basic_v<X, A, B>
[[nodiscard]] constexpr auto clamp(X x, A a, B b) noexcept {
    return min(max(x, a), b);
}

template<typename X, typename A, typename B>
requires is_all_basic_v<X, A, B>
[[nodiscard]] constexpr auto inverse_lerp(X x, A a, B b) noexcept {
    return (x - a) / (b - a);
}

}// namespace ocarina
