//
// Created by GitHub Copilot on 2026/4/10.
//

#pragma once

#include "core/stl.h"
#include "core/stl.h"
#include "basic_traits.h"

namespace ocarina {

class real;

namespace detail {
template<typename T>
struct is_real_op_enable : std::bool_constant<std::is_arithmetic_v<std::remove_cvref_t<T>>> {};

template<>
struct is_real_op_enable<real> : std::true_type {};

template<typename T>
requires is_real_op_enable<std::remove_cvref_t<T>>::value
struct binary_op_real_target {
    using raw_t = std::remove_cvref_t<T>;
    using type = std::conditional_t<std::is_floating_point_v<raw_t>, raw_t, real>;
};
}// namespace detail

template<typename T>
static constexpr auto is_real_op_enable_v = detail::is_real_op_enable<std::remove_cvref_t<T>>::value;

template<typename T>
using binary_op_real_target_t = typename detail::binary_op_real_target<std::remove_cvref_t<T>>::type;

struct real {
    float value{};

    constexpr real() noexcept = default;

    template<typename T>
    requires is_real_op_enable_v<T>
    constexpr real(T v) noexcept : value(static_cast<float>(v)) {}

    constexpr real(const real &) noexcept = default;
    constexpr real(real &&) noexcept = default;
    constexpr real &operator=(const real &) noexcept = default;
    constexpr real &operator=(real &&) noexcept = default;

#define OC_REAL_CAST_OP(type)                                \
    [[nodiscard]] constexpr operator type() const noexcept { \
        return static_cast<type>(value);                     \
    }
    OC_REAL_CAST_OP(float)
    OC_REAL_CAST_OP(double)
    OC_REAL_CAST_OP(short)
    OC_REAL_CAST_OP(size_t)
    OC_REAL_CAST_OP(int)
    OC_REAL_CAST_OP(bool)
    OC_REAL_CAST_OP(uint)
#undef OC_REAL_CAST_OP

    template<typename T>
    requires is_real_op_enable_v<T>
    constexpr real &operator=(T v) noexcept {
        value = static_cast<float>(v);
        return *this;
    }

    constexpr real &operator+=(const real &rhs) noexcept {
        value += rhs.value;
        return *this;
    }

    constexpr real &operator-=(const real &rhs) noexcept {
        value -= rhs.value;
        return *this;
    }

    constexpr real &operator*=(const real &rhs) noexcept {
        value *= rhs.value;
        return *this;
    }

    constexpr real &operator/=(const real &rhs) noexcept {
        value /= rhs.value;
        return *this;
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    [[nodiscard]] constexpr binary_op_real_target_t<T> operator+(T rhs) const noexcept {
        using ret_t = binary_op_real_target_t<T>;
        return ret_t{value + static_cast<float>(rhs)};
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    [[nodiscard]] constexpr binary_op_real_target_t<T> operator-(T rhs) const noexcept {
        using ret_t = binary_op_real_target_t<T>;
        return ret_t{value - static_cast<float>(rhs)};
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    [[nodiscard]] constexpr binary_op_real_target_t<T> operator*(T rhs) const noexcept {
        using ret_t = binary_op_real_target_t<T>;
        return ret_t{value * static_cast<float>(rhs)};
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    [[nodiscard]] constexpr binary_op_real_target_t<T> operator/(T rhs) const noexcept {
        using ret_t = binary_op_real_target_t<T>;
        return ret_t{value / static_cast<float>(rhs)};
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    friend constexpr binary_op_real_target_t<T> operator+(T lhs, real rhs) noexcept {
        using ret_t = binary_op_real_target_t<T>;
        return ret_t{static_cast<float>(lhs) + rhs.value};
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    friend constexpr binary_op_real_target_t<T> operator-(T lhs, real rhs) noexcept {
        using ret_t = binary_op_real_target_t<T>;
        return ret_t{static_cast<float>(lhs) - rhs.value};
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    friend constexpr binary_op_real_target_t<T> operator*(T lhs, real rhs) noexcept {
        using ret_t = binary_op_real_target_t<T>;
        return ret_t{static_cast<float>(lhs) * rhs.value};
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    friend constexpr binary_op_real_target_t<T> operator/(T lhs, real rhs) noexcept {
        using ret_t = binary_op_real_target_t<T>;
        return ret_t{static_cast<float>(lhs) / rhs.value};
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    constexpr real &operator+=(T rhs) noexcept {
        value += static_cast<float>(rhs);
        return *this;
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    constexpr real &operator-=(T rhs) noexcept {
        value -= static_cast<float>(rhs);
        return *this;
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    constexpr real &operator*=(T rhs) noexcept {
        value *= static_cast<float>(rhs);
        return *this;
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    constexpr real &operator/=(T rhs) noexcept {
        value /= static_cast<float>(rhs);
        return *this;
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    [[nodiscard]] constexpr bool operator==(T rhs) const noexcept {
        return value == static_cast<float>(rhs);
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    [[nodiscard]] constexpr bool operator!=(T rhs) const noexcept {
        return value != static_cast<float>(rhs);
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    [[nodiscard]] constexpr bool operator<(T rhs) const noexcept {
        return value < static_cast<float>(rhs);
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    [[nodiscard]] constexpr bool operator<=(T rhs) const noexcept {
        return value <= static_cast<float>(rhs);
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    [[nodiscard]] constexpr bool operator>(T rhs) const noexcept {
        return value > static_cast<float>(rhs);
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    [[nodiscard]] constexpr bool operator>=(T rhs) const noexcept {
        return value >= static_cast<float>(rhs);
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    friend constexpr bool operator==(T lhs, real rhs) noexcept {
        return static_cast<float>(lhs) == rhs.value;
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    friend constexpr bool operator!=(T lhs, real rhs) noexcept {
        return static_cast<float>(lhs) != rhs.value;
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    friend constexpr bool operator<(T lhs, real rhs) noexcept {
        return static_cast<float>(lhs) < rhs.value;
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    friend constexpr bool operator<=(T lhs, real rhs) noexcept {
        return static_cast<float>(lhs) <= rhs.value;
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    friend constexpr bool operator>(T lhs, real rhs) noexcept {
        return static_cast<float>(lhs) > rhs.value;
    }

    template<typename T>
    requires is_real_op_enable_v<T>
    friend constexpr bool operator>=(T lhs, real rhs) noexcept {
        return static_cast<float>(lhs) >= rhs.value;
    }

    constexpr real &operator++() noexcept {
        *this += 1.f;
        return *this;
    }

    constexpr real operator++(int) noexcept {
        real temp = *this;
        ++(*this);
        return temp;
    }

    constexpr real &operator--() noexcept {
        *this -= 1.f;
        return *this;
    }

    constexpr real operator--(int) noexcept {
        real temp = *this;
        --(*this);
        return temp;
    }

    [[nodiscard]] constexpr real operator+() const noexcept {
        return *this;
    }

    [[nodiscard]] constexpr real operator-() const noexcept {
        return real{-value};
    }

    friend std::ostream &operator<<(std::ostream &os, const real &r) {
        os << r.value;
        return os;
    }

    friend std::istream &operator>>(std::istream &is, real &r) {
        is >> r.value;
        return is;
    }

    [[nodiscard]] constexpr uint32_t bits() const noexcept {
        return bit_cast<uint32_t>(value);
    }

    [[nodiscard]] constexpr bool is_nan() const noexcept {
        auto raw = bits();
        return ((raw & 0x7F800000u) == 0x7F800000u) && (raw & 0x007FFFFFu);
    }

    [[nodiscard]] constexpr bool is_inf() const noexcept {
        return (bits() & 0x7FFFFFFFu) == 0x7F800000u;
    }

    [[nodiscard]] constexpr bool is_neg() const noexcept {
        return (bits() & 0x80000000u) != 0u;
    }
};



}// namespace ocarina