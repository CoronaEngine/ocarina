//
// Created by ling.zhu on 2024/10/28.
//

#pragma once

#include "rhi/resources/buffer.h"
#include "pyexporter/ocapi.h"
#include "rhi/common.h"
#include "base/using.h"

namespace py = pybind11;
template<typename T, size_t N, typename M>
void export_vector_op(M &m) {
    if constexpr (ocarina::is_number_v<T>) {
        m = m.def("__add__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a + b; }, py::is_operator());
        m = m.def("__neg__", [](const Vector<T, N> &a) { return -a; }, py::is_operator());
        m = m.def("__add__", [](const Vector<T, N> &a, const T &b) { return a + b; }, py::is_operator());
        m = m.def("__radd__", [](const Vector<T, N> &a, const T &b) { return a + b; }, py::is_operator());
        m = m.def("__sub__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a - b; }, py::is_operator());
        m = m.def("__sub__", [](const Vector<T, N> &a, const T &b) { return a - b; }, py::is_operator());
        m = m.def("__rsub__", [](const Vector<T, N> &a, const T &b) { return b - a; }, py::is_operator());
        m = m.def("__mul__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a * b; }, py::is_operator());
        m = m.def("__mul__", [](const Vector<T, N> &a, const T &b) { return a * b; }, py::is_operator());
        m = m.def("__rmul__", [](const Vector<T, N> &a, const T &b) { return a * b; }, py::is_operator());
        m = m.def("__truediv__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a / b; }, py::is_operator());
        m = m.def("__truediv__", [](const Vector<T, N> &a, const T &b) { return a / b; }, py::is_operator());
        m = m.def("__rtruediv__", [](const Vector<T, N> &a, const T &b) { return b / a; }, py::is_operator());
        m = m.def("__gt__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a > b; }, py::is_operator());
        m = m.def("__ge__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a >= b; }, py::is_operator());
        m = m.def("__lt__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a < b; }, py::is_operator());
        m = m.def("__le__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a <= b; }, py::is_operator());
    }
    if constexpr (ocarina::is_integral_v<T>) {
        m = m.def("__mod__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a % b; }, py::is_operator());
        m = m.def("__mod__", [](const Vector<T, N> &a, const T &b) { return a % b; }, py::is_operator());
        m = m.def("__rmod__", [](const Vector<T, N> &a, const T &b) { return b % a; }, py::is_operator());
        m = m.def("__shl__", [](const Vector<T, N> &a, const T &b) { return a << b; }, py::is_operator());
        m = m.def("__shr__", [](const Vector<T, N> &a, const T &b) { return a >> b; }, py::is_operator());

        m = m.def("__mod__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a % b; }, py::is_operator());
        m = m.def("__rmod__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return b % a; }, py::is_operator());
        m = m.def("__shl__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a << b; }, py::is_operator());
        m = m.def("__shr__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a >> b; }, py::is_operator());

        m = m.def("__xor__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a ^ b; }, py::is_operator());
        m = m.def("__and__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a & b; }, py::is_operator());
        m = m.def("__or__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a | b; }, py::is_operator());
    }
    m = m.def("__eq__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a == b; }, py::is_operator());
    m = m.def("__ne__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a != b; }, py::is_operator());
}

template<typename T, size_t N, typename M>
void export_vector_unary_func(M &m) {
#define OC_EXPORT_UNARY_FUNC(func) \
    m.def(#func, [](const Vector<T, N> &v) { return ocarina::func(v); });
    if constexpr (is_signed_v<T>) {
        OC_EXPORT_UNARY_FUNC(abs)
        OC_EXPORT_UNARY_FUNC(sign)
    }
    if constexpr (is_floating_point_v<T>) {
        OC_EXPORT_UNARY_FUNC(rcp)
        OC_EXPORT_UNARY_FUNC(sqrt)
        OC_EXPORT_UNARY_FUNC(cos)
        OC_EXPORT_UNARY_FUNC(sin)
        OC_EXPORT_UNARY_FUNC(tan)
        OC_EXPORT_UNARY_FUNC(cosh)
        OC_EXPORT_UNARY_FUNC(sinh)
        OC_EXPORT_UNARY_FUNC(tanh)
        OC_EXPORT_UNARY_FUNC(log)
        OC_EXPORT_UNARY_FUNC(log2)
        OC_EXPORT_UNARY_FUNC(log10)
        OC_EXPORT_UNARY_FUNC(exp)
        OC_EXPORT_UNARY_FUNC(exp2)
        OC_EXPORT_UNARY_FUNC(asin)
        OC_EXPORT_UNARY_FUNC(acos)
        OC_EXPORT_UNARY_FUNC(atan)
        OC_EXPORT_UNARY_FUNC(asinh)
        OC_EXPORT_UNARY_FUNC(acosh)
        OC_EXPORT_UNARY_FUNC(atanh)
        OC_EXPORT_UNARY_FUNC(floor)
        OC_EXPORT_UNARY_FUNC(ceil)
        OC_EXPORT_UNARY_FUNC(degrees)
        OC_EXPORT_UNARY_FUNC(radians)
        OC_EXPORT_UNARY_FUNC(round)
        OC_EXPORT_UNARY_FUNC(isnan)
        OC_EXPORT_UNARY_FUNC(isinf)
        OC_EXPORT_UNARY_FUNC(fract)
        OC_EXPORT_UNARY_FUNC(length)
        OC_EXPORT_UNARY_FUNC(normalize)
        OC_EXPORT_UNARY_FUNC(length_squared)
        OC_EXPORT_UNARY_FUNC(sqr)
    }
    if constexpr (std::is_same_v<bool, T>) {
        OC_EXPORT_UNARY_FUNC(all)
        OC_EXPORT_UNARY_FUNC(any)
        OC_EXPORT_UNARY_FUNC(none)
    }
#undef OC_EXPORT_UNARY_FUNC
}

template<typename T, size_t N, typename M>
void export_vector_binary_func(M &m) {
#define OC_EXPORT_BINARY_FUNC(func) \
    m.def(#func, [](const Vector<T, N> &lhs, const Vector<T, N> &rhs) { return ocarina::func(lhs, rhs); });
    if constexpr (is_floating_point_v<T>) {
        OC_EXPORT_BINARY_FUNC(pow)
        OC_EXPORT_BINARY_FUNC(atan2)
    }

    if constexpr (is_number_v<T>) {
        OC_EXPORT_BINARY_FUNC(min)
        OC_EXPORT_BINARY_FUNC(max)
        OC_EXPORT_BINARY_FUNC(distance_squared)
        OC_EXPORT_BINARY_FUNC(distance)
        OC_EXPORT_BINARY_FUNC(dot)
        if constexpr (N == 3) {
            OC_EXPORT_BINARY_FUNC(cross)
        }
    }

#undef OC_EXPORT_BINARY_FUNC
}

template<typename T, size_t N, typename M>
void export_vector_triple_func(M &m) {
#define OC_EXPORT_TRIPLE_FUNC(func) \
    m.def(#func, [](const Vector<T, N> &a, const Vector<T, N> &b, const Vector<T, N> &c) { return ocarina::func(a, b, c); });
    if constexpr (is_number_v<T>) {
        OC_EXPORT_TRIPLE_FUNC(clamp)
    }
    if constexpr (is_floating_point_v<T>) {
        OC_EXPORT_TRIPLE_FUNC(fma)
        OC_EXPORT_TRIPLE_FUNC(lerp)
    }
#undef OC_EXPORT_TRIPLE_FUNC
}

template<typename T, size_t N, typename M>
void export_vector_cast(M &m) {
    traverse_tuple(std::tuple<uint, int, float, bool>{}, [&]<typename Src>(const Src &_, uint index) {
        if constexpr (std::is_same_v<T, Src>) {
            return;
        }
        string cast_func_name = ocarina::format("make_{}{}", TypeDesc<T>::name(), N);
        m.def(cast_func_name.c_str(), [](const Vector<Src, N> &v) { return Vector<T, N>(v); });

        if constexpr (!std::is_same_v<Src, bool> && !std::is_same_v<T, bool>) {
            string func_name = ocarina::format("as_{}{}", TypeDesc<T>::name(), N);
            m.def(func_name.c_str(), [](const Vector<Src, N> &v) { return ocarina::bit_cast<Vector<T, N>>(v); });
        }
    });
}

template<typename T, size_t N, typename M>
void export_vector_member_func(M &mt, PythonExporter &exporter) {
    mt.def(py::init([&](std::array<T, N> a) {
        return Vector<T, N>(a.data());
    }));
}

template<typename T, size_t N, typename M>
void export_vector_func(M &mt, PythonExporter &exporter) {
    auto &m = exporter.module;
    export_vector_op<T, N>(mt);
    export_vector_unary_func<T, N>(exporter.module);
    export_vector_binary_func<T, N>(exporter.module);
    export_vector_triple_func<T, N>(exporter.module);
    export_vector_cast<T, N>(exporter.module);
    export_vector_member_func<T, N>(mt, exporter);
}
