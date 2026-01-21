//
// Created by z on 2026/1/20.
//

#pragma once

#include "core/stl.h"
#include "basic_traits.h"

namespace ocarina {

namespace detail {
template<typename T>
struct is_half_op_enable : std::false_type {};

template<>
struct is_half_op_enable<float> : std::true_type {};

template<>
struct is_half_op_enable<int> : std::true_type {};

template<>
struct is_half_op_enable<uint> : std::true_type {};

template<>
struct is_half_op_enable<double> : std::true_type {};
}// namespace detail

template<typename T>
static constexpr auto is_half_op_enable_v = detail::is_half_op_enable<std::remove_cvref_t<T>>::value;

class half {
private:
    uint16_t bits_;

    static constexpr uint16_t float_to_half(float f) {
        uint32_t bits_ = bit_cast<uint32_t>(f);
        uint16_t sign = (bits_ >> 31) & 0x1;
        uint16_t exp = (bits_ >> 23) & 0xFF;
        uint32_t mantissa = bits_ & 0x7FFFFF;

        if (exp == 0xFF) {
            // inf or nan
            return (sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0);
        }

        if (exp == 0) {
            if (mantissa == 0) {
                return sign << 15;
            }
            int shift = 0;
            while ((mantissa & 0x400000) == 0) {
                mantissa <<= 1;
                shift++;
            }
            mantissa &= 0x7FFFFF;
            exp = 1 - shift;
        }

        exp = exp - 127 + 15;

        if (exp >= 31) {
            return (sign << 15) | 0x7C00;
        } else if (exp <= 0) {
            if (exp < -10) {
                return sign << 15;
            }
            uint32_t shifted = (mantissa | 0x800000) >> (1 - exp);
            return (sign << 15) | (shifted >> 13);
        }

        uint16_t half_exp = exp & 0x1F;
        uint16_t half_mantissa = (mantissa >> 13) & 0x3FF;
        return (sign << 15) | (half_exp << 10) | half_mantissa;
    }
    static constexpr float half_to_float(uint16_t h) {
        uint16_t sign = (h >> 15) & 0x1;
        uint16_t exp = (h >> 10) & 0x1F;
        uint16_t mantissa = h & 0x3FF;

        if (exp == 0x1F) {
            // inf or nan
            uint32_t result = (sign << 31) | 0x7F800000 | (mantissa ? 0x00400000 : 0);
            return *reinterpret_cast<float *>(&result);
        }

        if (exp == 0) {
            if (mantissa == 0) {
                uint32_t result = sign << 31;
                return *reinterpret_cast<float *>(&result);
            }
            uint32_t result = (sign << 31) | 0x00800000;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                result -= 0x00800000;
            }
            result |= (mantissa & 0x3FF) << 13;
            return *reinterpret_cast<float *>(&result);
        }

        uint32_t result = (sign << 31) | ((exp - 15 + 127) << 23) | (mantissa << 13);
        return *reinterpret_cast<float *>(&result);
    }

    template<typename T>
    requires is_floating_point_v<T>
    static constexpr T cast(half src) {
        if constexpr (is_float_v<T>) {
            return half_to_float(src.bits());
        } else {
            return src;
        }
    }

public:
    constexpr half() : bits_(0) {}
    template<typename T>
    requires is_half_op_enable_v<T>
    constexpr half(T &&val) : bits_(float_to_half(static_cast<float>(std::forward<T>(val)))) {}

#define OC_HALF_CAST_OP(type)\
    constexpr operator type() const {\
        return static_cast<type>(half_to_float(bits_));\
    }
    OC_HALF_CAST_OP(float)
    OC_HALF_CAST_OP(int)
    OC_HALF_CAST_OP(uint)
    OC_HALF_CAST_OP(double)
#undef OC_HALF_CAST_OP

    constexpr half &operator=(const half &other) {
        if (this != &other) {
            bits_ = other.bits_;
        }
        return *this;
    }

    constexpr half operator+(const half &other) const {
        return half(half_to_float(bits_) + half_to_float(other.bits_));
    }
    constexpr float operator+(float other) const {
        return half_to_float(bits_) + other;
    }
    friend constexpr float operator+(float lhs, half rhs) {
        return lhs + half_to_float(rhs.bits_);
    }
    constexpr half operator+(int other) const {
        return half(half_to_float(bits_) + static_cast<float>(other));
    }
    friend constexpr half operator+(int lhs, half rhs) {
        return half(static_cast<float>(lhs) + half_to_float(rhs.bits()));
    }
    template<typename T>
    constexpr half &operator+=(const T &other) {
        *this = *this + other;
        return *this;
    }

    constexpr half operator-(const half &other) const {
        return half(half_to_float(bits_) - half_to_float(other.bits_));
    }
    constexpr half operator*(const half &other) const {
        return half(half_to_float(bits_) * half_to_float(other.bits_));
    }
    constexpr half operator/(const half &other) const {
        return half(half_to_float(bits_) / half_to_float(other.bits_));
    }

    constexpr half &operator-=(const half &other) {
        *this = *this - other;
        return *this;
    }
    constexpr half &operator*=(const half &other) {
        *this = *this * other;
        return *this;
    }
    constexpr half &operator/=(const half &other) {
        *this = *this / other;
        return *this;
    }

    constexpr half &operator++() {
        *this = *this + half(1.0f);
        return *this;
    }
    constexpr half operator++(int) {
        half temp = *this;
        ++(*this);
        return temp;
    }
    constexpr half &operator--() {
        *this = *this - half(1.0f);
        return *this;
    }
    constexpr half operator--(int) {
        half temp = *this;
        --(*this);
        return temp;
    }

    constexpr bool operator==(const half &other) const {
        if (is_nan() || other.is_nan()) {
            return false;
        }
        return bits_ == other.bits_;
    }

    constexpr bool operator==(float other) const {
        return (*this) == half(other);
    }

    constexpr bool operator==(int other) const {
        return (*this) == half(other);
    }

    constexpr bool operator!=(const half &other) const {
        return !(*this == other);
    }
    constexpr bool operator<(const half &other) const {
        if (is_nan() || other.is_nan()) {
            return false;
        }
        return half_to_float(bits_) < half_to_float(other.bits_);
    }
    constexpr bool operator<=(const half &other) const {
        if (is_nan() || other.is_nan()) {
            return false;
        }
        return half_to_float(bits_) <= half_to_float(other.bits_);
    }
    constexpr bool operator>(const half &other) const {
        if (is_nan() || other.is_nan()) {
            return false;
        }
        return half_to_float(bits_) > half_to_float(other.bits_);
    }
    constexpr bool operator>=(const half &other) const {
        if (is_nan() || other.is_nan()) {
            return false;
        }
        return half_to_float(bits_) >= half_to_float(other.bits_);
    }

    constexpr half operator+() const {
        return *this;
    }
    constexpr half operator-() const {
        half result;
        result.bits_ = bits_ ^ 0x8000;
        return result;
    }

    friend std::ostream &operator<<(std::ostream &os, const half &h) {
        os << static_cast<float>(h);
        return os;
    }
    friend std::istream &operator>>(std::istream &is, half &h) {
        float f;
        is >> f;
        h = f;
        return is;
    }

    [[nodiscard]] constexpr uint16_t bits() const { return bits_; }

    [[nodiscard]] constexpr bool is_nan() const {
        return ((bits_ & 0x7C00) == 0x7C00) && (bits_ & 0x03FF);
    }
    [[nodiscard]] constexpr bool is_inf() const {
        return ((bits_ & 0x7FFF) == 0x7C00);
    }
    [[nodiscard]] constexpr bool is_neg() const {
        return (bits_ & 0x8000) != 0;
    }
};
}// namespace ocarina