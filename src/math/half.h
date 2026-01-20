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

    // 内部转换函数
    static uint16_t floatToHalf(float f);
    static float halfToFloat(uint16_t h);

public:
    half();
    half(float f);
    half(double d);
    half(int i);

    operator float() const;
    operator double() const;
    operator int() const;

    half &operator=(float f);
    half &operator=(double d);
    half &operator=(int i);
    half &operator=(const half &other);

    half operator+(const half &other) const;
    half operator-(const half &other) const;
    half operator*(const half &other) const;
    half operator/(const half &other) const;

    half &operator+=(const half &other);
    half &operator-=(const half &other);
    half &operator*=(const half &other);
    half &operator/=(const half &other);

    half &operator++();
    half operator++(int);
    half &operator--();
    half operator--(int);

    bool operator==(const half &other) const;
    bool operator!=(const half &other) const;
    bool operator<(const half &other) const;
    bool operator<=(const half &other) const;
    bool operator>(const half &other) const;
    bool operator>=(const half &other) const;

    half operator+() const;
    half operator-() const;

    friend std::ostream &operator<<(std::ostream &os, const half &h);
    friend std::istream &operator>>(std::istream &is, half &h);

    uint16_t getBits() const { return bits; }

    bool isNaN() const;
    bool isInfinity() const;
    bool isNegative() const;
};
}// namespace ocarina