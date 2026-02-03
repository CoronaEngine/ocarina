
#pragma once

namespace ocarina {
template<typename... Ts>
struct always_false {
    static constexpr bool value = false;
};

template<typename... Ts>
static constexpr bool always_false_v = always_false<Ts...>::value;

struct true_type {
    static constexpr bool value = true;
};

struct false_type {
    static constexpr bool value = false;
};

template<typename T>
struct remove_reference {
    using type = T;
};

template<typename T>
struct remove_reference<T &> {
    using type = T;
};

template<typename T>
struct remove_reference<T &&> {
    using type = T;
};

template<typename T>
struct remove_cv {
    using type = T;
};

template<typename T>
struct remove_cv<const T> {
    using type = T;
};

template<typename T>
struct remove_cv<volatile T> {
    using type = T;
};

template<typename T>
struct remove_cv<const volatile T> {
    using type = T;
};

template<typename T>
struct remove_cvref {
    using type = typename remove_cv<typename remove_reference<T>::type>::type;
};

template<typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

namespace detail {
template<typename T>
struct is_integral : public false_type {};

template<>
struct is_integral<int> : public true_type {};

template<>
struct is_integral<unsigned int> : public true_type {};

template<>
struct is_integral<unsigned long long> : public true_type {};
}// namespace detail

template<typename T>
static constexpr bool is_integral_v = detail::is_integral<remove_cvref_t<T>>::value;

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

template<bool B, typename T, typename F>
struct conditional {
    using type = F;
};

template<typename T, typename F>
struct conditional<true, T, F> {
    using type = T;
};

template<bool B, typename T, typename F>
using conditional_t = typename conditional<B, T, F>::type;

}// namespace ocarina

template<typename T, unsigned int N>
using oc_array = ocarina::array<T, N>;