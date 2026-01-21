//
// Created by z on 2026/1/20.
//

#pragma once

#include "core/stl.h"
#include "basic_traits.h"

namespace ocarina {
class half {
private:
    uint16_t bits;

    static constexpr uint16_t floatToHalf(float f) {
        uint32_t bits = bit_cast<uint32_t>(f);
        uint16_t sign = (bits >> 31) & 0x1;
        uint16_t exp = (bits >> 23) & 0xFF;
        uint32_t mantissa = bits & 0x7FFFFF;

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
    static constexpr float halfToFloat(uint16_t h) {
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

public:
    constexpr half() : bits(0) {}
    constexpr half(float f) : bits(floatToHalf(f)) {}
    constexpr half(double d) : bits(floatToHalf(static_cast<float>(d))) {}
    constexpr half(int i) : bits(floatToHalf(static_cast<float>(i))) {}

    constexpr operator float() const {
        return halfToFloat(bits);
    }
    constexpr operator double() const {
        return static_cast<double>(halfToFloat(bits));
    }
    constexpr operator int() const {
        return static_cast<int>(halfToFloat(bits));
    }

    constexpr half &operator=(float f) {
        bits = floatToHalf(f);
        return *this;
    }
    constexpr half &operator=(double d) {
        bits = floatToHalf(static_cast<float>(d));
        return *this;
    }
    constexpr half &operator=(int i) {
        bits = floatToHalf(static_cast<float>(i));
        return *this;
    }
    constexpr half &operator=(const half &other) {
        if (this != &other) {
            bits = other.bits;
        }
        return *this;
    }

    constexpr half operator+(const half &other) const {
        return half(halfToFloat(bits) + halfToFloat(other.bits));
    }
    constexpr half operator-(const half &other) const {
        return half(halfToFloat(bits) - halfToFloat(other.bits));
    }
    constexpr half operator*(const half &other) const {
        return half(halfToFloat(bits) * halfToFloat(other.bits));
    }
    constexpr half operator/(const half &other) const {
        return half(halfToFloat(bits) / halfToFloat(other.bits));
    }

    constexpr half &operator+=(const half &other) {
        *this = *this + other;
        return *this;
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
        if (isNaN() || other.isNaN()) {
            return false;
        }
        return bits == other.bits;
    }
    constexpr bool operator!=(const half &other) const {
        return !(*this == other);
    }
    constexpr bool operator<(const half &other) const {
        if (isNaN() || other.isNaN()) {
            return false;
        }
        return halfToFloat(bits) < halfToFloat(other.bits);
    }
    constexpr bool operator<=(const half &other) const {
        if (isNaN() || other.isNaN()) {
            return false;
        }
        return halfToFloat(bits) <= halfToFloat(other.bits);
    }
    constexpr bool operator>(const half &other) const {
        if (isNaN() || other.isNaN()) {
            return false;
        }
        return halfToFloat(bits) > halfToFloat(other.bits);
    }
    constexpr bool operator>=(const half &other) const {
        if (isNaN() || other.isNaN()) {
            return false;
        }
        return halfToFloat(bits) >= halfToFloat(other.bits);
    }

    constexpr half operator+() const {
        return *this;
    }
    constexpr half operator-() const {
        half result;
        result.bits = bits ^ 0x8000;
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

    constexpr uint16_t getBits() const { return bits; }

    [[nodiscard]] constexpr bool isNaN() const {
        return ((bits & 0x7C00) == 0x7C00) && (bits & 0x03FF);
    }
    [[nodiscard]] constexpr bool isInfinity() const {
        return ((bits & 0x7FFF) == 0x7C00);
    }
    constexpr bool isNegative() const {
        return (bits & 0x8000) != 0;
    }
};
}// namespace ocarina