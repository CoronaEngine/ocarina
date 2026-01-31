
#pragma once

namespace ocarina {
template<typename... Ts>
struct always_false {
    static constexpr bool value = false;
};

template<typename... Ts>
static constexpr bool always_false_v = always_false<Ts...>::value;

template<size_t... Ints>
struct index_sequence {};

template<size_t N, size_t... Ints>
struct make_index_sequence_helper : make_index_sequence_helper<N - 1, N - 1, Ints...> {
};

template<size_t... Ints>
struct make_index_sequence_helper<0, Ints...> {
    using type = index_sequence<Ints...>;
};

template<size_t N>
using make_index_sequence = typename make_index_sequence_helper<N>::type;

template<bool B, typename T = void>
struct enable_if {};

template<typename T>
struct enable_if<true, T> {
    using type = T;
};

template<bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;

}

