
#pragma once

namespace ocarina {

namespace detail {
template<typename T, size_t N>
struct VectorStorage {
    static_assert(always_false_v<T>, "Invalid vector storage");
};

template<typename T>
struct alignas(sizeof(T) * 2) VectorStorage<T, 2> {
    T x{}, y{};
    __device__ constexpr VectorStorage(T s = T{}) noexcept : x{s}, y{s} {}
    __device__ constexpr VectorStorage(T x, T y) noexcept : x{x}, y{y} {}
    __device__ constexpr VectorStorage(const T *ptr) noexcept : x{ptr[0]}, y{ptr[1]} {}
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 3> {
    T x{}, y{}, z{};
    __device__ constexpr VectorStorage(T s = T{}) noexcept : x{s}, y{s}, z{s} {}
    __device__ constexpr VectorStorage(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
    __device__ constexpr VectorStorage(const T *ptr) noexcept : x{ptr[0]}, y{ptr[1]}, z{ptr[2]} {}
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 4> {
    T x{}, y{}, z{}, w{};
    __device__ constexpr VectorStorage(T s = T{}) noexcept : x{s}, y{s}, z{s}, w{s} {}
    __device__ constexpr VectorStorage(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}
    __device__ constexpr VectorStorage(const T *ptr) noexcept : x{ptr[0]}, y{ptr[1]}, z{ptr[2]}, w{ptr[3]} {}
};
}// namespace detail

namespace detail {

}// namespace detail

template<typename T, size_t N>
struct Vector : public detail::VectorStorage<T, N> {
    using detail::VectorStorage<T, N>::VectorStorage;

private:
    template<typename U, size_t NN, size_t... i>
    static Vector<T, N> construct_helper(Vector<U, NN> v,
                                         ocarina::index_sequence<i...>) {
        return Vector<T, N>(oc_static_cast<T>(v[i])...);
    }

    template<typename U, size_t ...i>
    void assign_helper(const Vector<U, N> &other, ocarina::index_sequence<i...>) noexcept {
        ((this->operator[](i) = static_cast<T>(other[i])), ...);
    }


public:
    template<typename U>
    explicit constexpr Vector(U s) noexcept : Vector(oc_static_cast<T>(s)) {}

    template<typename U, size_t NN, ocarina::enable_if_t<NN >= N, int> = 0>
    explicit constexpr Vector(Vector<U, NN> v)
        : Vector{construct_helper(v, ocarina::make_index_sequence<N>())} {}

    template<typename U>
    constexpr Vector &operator=(Vector<U, N> other) noexcept {
        assign_helper<U>(other, ocarina::make_index_sequence<N>());
        return *this;
    }

    __device__ constexpr T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    __device__ constexpr const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
};

using uint = unsigned int;
using ulong = unsigned long long;
using uchar = unsigned char;
using ushort = unsigned short;

#define OC_MAKE_VECTOR_TYPES(T) \
    using T##2 = Vector<T, 2>;  \
    using T##3 = Vector<T, 3>;  \
    using T##4 = Vector<T, 4>;

OC_MAKE_VECTOR_TYPES(bool)
OC_MAKE_VECTOR_TYPES(float)
OC_MAKE_VECTOR_TYPES(half)
OC_MAKE_VECTOR_TYPES(int)
OC_MAKE_VECTOR_TYPES(char)
OC_MAKE_VECTOR_TYPES(short)
OC_MAKE_VECTOR_TYPES(ushort)
OC_MAKE_VECTOR_TYPES(uchar)
OC_MAKE_VECTOR_TYPES(uint)
OC_MAKE_VECTOR_TYPES(ulong)

#undef OC_MAKE_VECTOR_TYPES

}// namespace ocarina



#define OC_MAKE_VECTOR_N(type, dim) using type##dim = ocarina::Vector<type, dim>;

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto
operator+(const ocarina::Vector<T, N> v) noexcept {
    return v;
}

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto
operator-(const ocarina::Vector<T, N> v) noexcept {
    using R = ocarina::Vector<T, N>;
    if constexpr (N == 2) {
        return R{-v.x, -v.y};
    } else if constexpr (N == 3) {
        return R{-v.x, -v.y, -v.z};
    } else {
        return R{-v.x, -v.y, -v.z, -v.w};
    }
}

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto operator!(const ocarina::Vector<T, N> v) noexcept {
    if constexpr (N == 2u) {
        return ocarina::Vector<bool, 2>{!v.x, !v.y};
    } else if constexpr (N == 3u) {
        return ocarina::Vector<bool, 3>{!v.x, !v.y, !v.z};
    } else {
        return ocarina::Vector<bool, 3>{!v.x, !v.y, !v.z, !v.w};
    }
}

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto
operator~(const ocarina::Vector<T, N> v) noexcept {
    using R = ocarina::Vector<T, N>;
    if constexpr (N == 2) {
        return R{~v.x, ~v.y};
    } else if constexpr (N == 3) {
        return R{~v.x, ~v.y, ~v.z};
    } else {
        return R{~v.x, ~v.y, ~v.z, ~v.w};
    }
}

#define OC_MAKE_VECTOR(type)  \
    OC_MAKE_VECTOR_N(type, 2) \
    OC_MAKE_VECTOR_N(type, 3) \
    OC_MAKE_VECTOR_N(type, 4)

OC_MAKE_VECTOR(oc_int)
OC_MAKE_VECTOR(oc_uint)
OC_MAKE_VECTOR(oc_float)
OC_MAKE_VECTOR(oc_half)
OC_MAKE_VECTOR(oc_bool)
OC_MAKE_VECTOR(oc_uchar)
OC_MAKE_VECTOR(oc_ushort)
OC_MAKE_VECTOR(oc_ulong)

#define OC_MAKE_VECTOR_BINARY_OPERATOR(op, ...)                          \
    template<typename T, typename U, size_t N>                           \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(                                                         \
        ocarina::Vector<T, N> lhs, ocarina::Vector<U, N> rhs) noexcept { \
        using ret_type = decltype(T{} + U{});                            \
        if constexpr (N == 2) {                                          \
            return ocarina::Vector<ret_type, 2>{                         \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y};                                         \
        } else if constexpr (N == 3) {                                   \
            return ocarina::Vector<ret_type, 3>{                         \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z};                                         \
        } else {                                                         \
            return ocarina::Vector<ret_type, 4>{                         \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z,                                          \
                lhs.w op rhs.w};                                         \
        }                                                                \
    }                                                                    \
    template<typename T, typename U, size_t N>                           \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(ocarina::Vector<T, N> lhs, U rhs) noexcept {             \
        return lhs op ocarina::Vector<U, N>{rhs};                        \
    }                                                                    \
    template<typename T, typename U, size_t N>                           \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(T lhs, ocarina::Vector<U, N> rhs) noexcept {             \
        return ocarina::Vector<T, N>{lhs} op rhs;                        \
    }

OC_MAKE_VECTOR_BINARY_OPERATOR(+, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(-, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(*, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(/, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(%, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(>>, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(<<, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(|, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(&, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(^, ocarina::is_all_integral_v<T, U>)

#define OC_MAKE_VECTOR_ASSIGN_OPERATOR(op, ...)                           \
    template<typename T, typename U, size_t N>                            \
    __device__ constexpr decltype(auto) operator op(                      \
        ocarina::Vector<T, N> &lhs, ocarina::Vector<U, N> rhs) noexcept { \
        lhs.x op rhs.x;                                                   \
        lhs.y op rhs.y;                                                   \
        if constexpr (N >= 3) { lhs.z op rhs.z; }                         \
        if constexpr (N == 4) { lhs.w op rhs.w; }                         \
        return (lhs);                                                     \
    }                                                                     \
    template<typename T, typename U, size_t N>                            \
    __device__ constexpr decltype(auto) operator op(                      \
        ocarina::Vector<T, N> &lhs, U rhs) noexcept {                     \
        return (lhs op ocarina::Vector<U, N>{rhs});                       \
    }

OC_MAKE_VECTOR_ASSIGN_OPERATOR(+=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(-=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(*=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(/=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(%=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(<<=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(>>=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(|=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(&=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(^=, ocarina::is_all_integral_v<T, U>)

#define OC_MAKE_VECTOR_LOGIC_OPERATOR(op, ...)                           \
    template<typename T, size_t N>                                       \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(                                                         \
        ocarina::Vector<T, N> lhs, ocarina::Vector<T, N> rhs) noexcept { \
        if constexpr (N == 2) {                                          \
            return ocarina::Vector<bool, 2>{                             \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y};                                         \
        } else if constexpr (N == 3) {                                   \
            return ocarina::Vector<bool, 3>{                             \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z};                                         \
        } else {                                                         \
            return ocarina::Vector<bool, 4>{                             \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z,                                          \
                lhs.w op rhs.w};                                         \
        }                                                                \
    }                                                                    \
    template<typename T, size_t N>                                       \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(ocarina::Vector<T, N> lhs, T rhs) noexcept {             \
        return lhs op ocarina::Vector<T, N>{rhs};                        \
    }                                                                    \
    template<typename T, size_t N>                                       \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(T lhs, ocarina::Vector<T, N> rhs) noexcept {             \
        return ocarina::Vector<T, N>{lhs} op rhs;                        \
    }
OC_MAKE_VECTOR_LOGIC_OPERATOR(||, ocarina::is_all_boolean_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(&&, ocarina::is_all_boolean_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(==, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(!=, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(<, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(>, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(<=, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(>=, ocarina::is_all_number_v<T>)
