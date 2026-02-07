
#pragma once

#include "cuda_device_vector.h"

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



}