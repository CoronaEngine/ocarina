
#pragma once

#include <cuda_fp16.h>

#define OC_DEFINE_TEMPLATE_VALUE(template_name) \
    template<typename T>                        \
    static constexpr auto template_name##_v = template_name<T>::value;

#define OC_DEFINE_TEMPLATE_VALUE_MULTI(template_name) \
    template<typename... Ts>                          \
    static constexpr auto template_name##_v = template_name<Ts...>::value;

#define OC_DEFINE_TEMPLATE_TYPE(template_name) \
    template<typename T>                       \
    using template_name##_t = typename template_name<T>::type;

#define OC_DEFINE_TEMPLATE_TYPE_MULTI(template_name) \
    template<typename... Ts>                         \
    using template_name##_t = typename template_name<Ts...>::type;

namespace ocarina {
template<typename... Ts>
struct always_false {
    static constexpr bool value = false;
};

template<typename... Ts>
static constexpr bool always_false_v = always_false<Ts...>::value;

template<typename T, T v>
struct integral_constant {
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

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

template<typename... Ts>
struct disjunction {};

template<>
struct disjunction<> : false_type {};

template<typename First, typename... Rest>
struct disjunction<First, Rest...> : conditional_t<bool(First::value), First, disjunction<Rest...>> {};

OC_DEFINE_TEMPLATE_VALUE(disjunction)

template<typename... B>
struct conjunction;

template<>
struct conjunction<> : true_type {};

template<typename First, typename... Rest>
struct conjunction<First, Rest...>
    : conditional_t<(sizeof...(Rest) == 0 || !bool(First::value)), First, conjunction<Rest...>> {};

template<typename B>
struct negation : integral_constant<bool, !bool(B::value)> {};

namespace detail {
template<typename T>
struct is_integral_impl : public false_type {};

template<>
struct is_integral_impl<int> : public true_type {};

template<>
struct is_integral_impl<unsigned int> : public true_type {};

template<>
struct is_integral_impl<unsigned long long> : public true_type {};
}// namespace detail

template<typename T>
using is_integral = detail::is_integral_impl<remove_cvref_t<T>>;
OC_DEFINE_TEMPLATE_VALUE(is_integral)

namespace detail {
template<typename T>
struct is_floating_point_impl : public false_type {};

template<>
struct is_floating_point_impl<float> : public true_type {};

template<>
struct is_floating_point_impl<half> : public true_type {};
}// namespace detail
template<typename T>
using is_floating_point = detail::is_floating_point_impl<remove_cvref_t<T>>;
OC_DEFINE_TEMPLATE_VALUE(is_floating_point)

namespace detail {
template<typename T>
struct is_boolean_impl : public false_type {};

template<>
struct is_boolean_impl<bool> : public true_type {};
}// namespace detail
template<typename T>
using is_boolean = detail::is_boolean_impl<remove_cvref_t<T>>;
OC_DEFINE_TEMPLATE_VALUE(is_boolean)

template<typename T>
struct is_scalar : disjunction<is_integral<T>, is_floating_point<T>, is_boolean<T>> {};
OC_DEFINE_TEMPLATE_VALUE(is_scalar)

#define MAKE_TYPE_TRAITS(type)                             \
    template<typename... Ts>                               \
    using is_all_##type = conjunction<is_##type<Ts>...>;   \
    OC_DEFINE_TEMPLATE_VALUE_MULTI(is_all_##type)          \
    template<typename... Ts>                               \
    using is_any_##type = disjunction<is_##type<Ts>...>;   \
    OC_DEFINE_TEMPLATE_VALUE_MULTI(is_any_##type)          \
    template<typename... Ts>                               \
    using is_none_##type = negation<is_any_##type<Ts...>>; \
    OC_DEFINE_TEMPLATE_VALUE_MULTI(is_none_##type)

MAKE_TYPE_TRAITS(scalar)
MAKE_TYPE_TRAITS(integral)
MAKE_TYPE_TRAITS(floating_point)
MAKE_TYPE_TRAITS(boolean)

#undef MAKE_TYPE_TRAITS

namespace detail {
template<typename... Ts>
struct make_void {
    using type = void;
};
}// namespace detail

template<typename... Ts>
using void_t = typename detail::make_void<Ts...>::type;

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

template<typename, typename, typename = void>
struct is_addable : false_type {};

template<typename T, typename F>
struct is_addable<T, F, void_t<decltype(T{} + T{}), decltype(static_cast<decltype(T{} + T{})>(T{})), decltype(static_cast<decltype(T{} + T{})>(F{}))>> : true_type {};

template<typename T, typename F>
struct is_selectable : conjunction<is_all_scalar<T, F>, is_addable<T, F>> {};
OC_DEFINE_TEMPLATE_VALUE_MULTI(is_selectable)


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

#undef OC_DEFINE_TEMPLATE_VALUE_MULTI
#undef OC_DEFINE_TEMPLATE_VALUE
#undef OC_DEFINE_TEMPLATE_TYPE
#undef OC_DEFINE_TEMPLATE_TYPE_MULTI