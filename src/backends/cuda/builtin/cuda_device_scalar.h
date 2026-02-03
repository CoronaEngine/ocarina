
#pragma once

#include <cuda_fp16.h>

using oc_int = int;
using oc_uint = unsigned int;
using oc_half = half;
using oc_float = float;
using oc_bool = bool;
using oc_uchar = unsigned char;
using oc_ushort = unsigned short;
using oc_ulong = unsigned long long;

oc_half oc_float2half(oc_float f) {
    return __float2half(f);
}

oc_float oc_half2float(oc_half h) {
    return __half2float(h);
}

template<typename Dst, typename Src>
Dst oc_static_cast(Src src) {
    return static_cast<Dst>(src);
}

template<typename Dst>
Dst oc_static_cast(oc_half src) {
    return static_cast<Dst>(oc_half2float(src));
}

namespace ocarina {
namespace detail {
template<typename T>
struct is_half_op_enable : ocarina::false_type {};

template<>
struct is_half_op_enable<float> : ocarina::true_type {};

template<>
struct is_half_op_enable<int> : ocarina::true_type {};

template<>
struct is_half_op_enable<oc_uint> : ocarina::true_type {};

template<>
struct is_half_op_enable<double> : ocarina::true_type {};
}// namespace detail

template<typename T>
static constexpr auto is_half_op_enable_v = detail::is_half_op_enable<remove_cvref_t<T>>::value;

// namespace detail {

// template<typename T, ocarina::enable_if_t<is_half_op_enable_v<T>, int> = 0>
// struct binary_op_half_target {
//     using type = ocarina::conditional_t<is_integral_v<T>, half, T>;
// };

// }// namespace detail

// template<typename T>
// using binary_op_half_target_t = typename detail::binary_op_half_target<ocarina::remove_cvref_t<T>>::type;

}// namespace ocarina
