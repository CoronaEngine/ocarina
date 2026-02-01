
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

template<typename T, unsigned int N>
class array {
private:
    T _data[N];

public:
    __device__ constexpr array() noexcept : _data{} {}
    template<typename... Elem>
    __device__ constexpr array(Elem... elem) noexcept : _data{elem...} {}
    __device__ constexpr array(array &&) noexcept = default;
    __device__ constexpr array(const array &) noexcept = default;
    __device__ constexpr array &operator=(array &&) noexcept = default;
    __device__ constexpr array &operator=(const array &) noexcept = default;
    __device__ constexpr T *data() noexcept { return &_data[0]; }
    __device__ constexpr const T *data() const noexcept { return &_data[0]; }
    [[nodiscard]] __device__ T &operator[](size_t i) noexcept { return _data[i]; }
    [[nodiscard]] __device__ T operator[](size_t i) const noexcept { return _data[i]; }
};

}// namespace ocarina

template<typename T, unsigned int N>
using oc_array = ocarina::array<T, N>;