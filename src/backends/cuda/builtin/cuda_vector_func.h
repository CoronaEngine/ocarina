
#pragma once

#include "cuda_device_vector.h"
#include "cuda_device_builtin.h"

namespace ocarina {

namespace detail {

template<size_t N, typename T, typename U, size_t... index>
static constexpr auto dot_helper(Vector<T, N> a, Vector<U, N> b, ocarina::index_sequence<index...>) {
    return ((a[index] * b[index]) + ...);
}
}// namespace detail

template<size_t N, typename T, typename U>
static constexpr auto dot(Vector<T, N> lhs, Vector<U, N> rhs) {
    return detail::dot_helper(lhs, rhs, ocarina::make_index_sequence<N>());
}

}// namespace ocarina

namespace ocarina {

template<typename T, typename U, typename V,
         ocarina::enable_if_t<is_all_scalar_v<T, U, V>, int> = 0>
static constexpr auto lerp(T t, U a, V b) {
    return a + (b - a) * t;
}

namespace detail {
template<template<typename, size_t> typename Container, typename T,
         typename U, typename V, size_t N, size_t... i>
static constexpr auto lerp_helper(Container<T, N> t, Container<U, N> a,
                                  Container<V, N> b, ocarina::index_sequence<i...>) {
    using scalar_type = decltype(lerp(T{}, U{}, V{}));
    return Container<scalar_type, N>{lerp(t[i], a[i], b[i])...};
}
}// namespace detail

template<template<typename, size_t> typename Container, typename T, typename U, typename V,
         size_t N, ocarina::void_t<decltype(lerp(T{}, U{}, V{}))> * = nullptr>
static constexpr auto lerp(Container<T, N> t, Container<U, N> a, Container<V, N> b) {
    return detail::lerp_helper(t, a, b, ocarina::make_index_sequence<N>{});
}
}// namespace ocarina

template<typename... Args>
static constexpr auto oc_lerp(Args... args) noexcept { return ocarina::lerp(args...); }

namespace ocarina {

template<typename T, typename U, typename V,
         ocarina::enable_if_t<is_all_scalar_v<T, U, V>, int> = 0>
static constexpr auto clamp(T v0, U v1, V v2) {
    return oc_min(v2, oc_max(v1, v0));
}

namespace detail {
template<template<typename, size_t> typename Container, typename T,
         typename U, typename V, size_t N, size_t... i>
static constexpr auto clamp_helper(Container<T, N> v0, Container<U, N> v1,
                                   Container<V, N> v2, ocarina::index_sequence<i...>) {
    using scalar_type = decltype(clamp(T{}, U{}, V{}));
    return Container<scalar_type, N>{clamp(v0[i], v1[i], v2[i])...};
}
}// namespace detail

template<template<typename, size_t> typename Container, typename T, typename U, typename V,
         size_t N, ocarina::void_t<decltype(clamp(T{}, U{}, V{}))> * = nullptr>
static constexpr auto clamp(Container<T, N> v0, Container<U, N> v1, Container<V, N> v2) {
    return detail::clamp_helper(v0, v1, v2, ocarina::make_index_sequence<N>{});
}

}// namespace ocarina

template<typename... Args>
static constexpr auto oc_clamp(Args... args) noexcept { return ocarina::clamp(args...); }