//
// Created by Zero on 2024/11/3.
//

#pragma once

#include "core/string_util.h"

namespace ocarina {
template<typename T, int... dims>
class Buffer;

template<typename T>
class BufferDesc;

template<typename T, int... dims>
class BufferView;

class Texture3D;
class Texture2D;
class ByteBuffer;

class Accel;

class BindlessArray;

template<typename T>
struct TypeDesc {
    static_assert(always_false_v<T>, "Invalid type.");
};

#define OC_MAKE_VECTOR_DESC_NAME(S, N)                                 \
    template<>                                                         \
    struct TypeDesc<Vector<S, N>> {                                    \
        static constexpr ocarina::string_view description() noexcept { \
            return ocarina::string_view("vector<" #S "," #N ">");      \
        }                                                              \
        static constexpr ocarina::string_view name() noexcept {        \
            return ocarina::string_view(#S #N);                        \
        }                                                              \
    };

#define OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(S)          \
    template<>                                                         \
    struct TypeDesc<S> {                                               \
        static constexpr ocarina::string_view description() noexcept { \
            using namespace std::string_view_literals;                 \
            return #S##sv;                                             \
        }                                                              \
        static constexpr ocarina::string_view name() noexcept {        \
            return description();                                      \
        }                                                              \
    };                                                                 \
    OC_MAKE_VECTOR_DESC_NAME(S, 2)                                     \
    OC_MAKE_VECTOR_DESC_NAME(S, 3)                                     \
    OC_MAKE_VECTOR_DESC_NAME(S, 4)

OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(bool)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(float)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(half)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(int)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uint)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uchar)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(char)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(ulong)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(ushort)

#undef OC_MAKE_VECTOR_DESC_NAME
#undef OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION

template<>
struct TypeDesc<void> {
    static constexpr ocarina::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "void"sv;
    }

    static constexpr ocarina::string_view name() noexcept {
        return description();
    }
};

/// matrices
template<size_t N, size_t M>
struct TypeDesc<ocarina::Matrix<N, M>> {
    static ocarina::string &description() noexcept {
        static thread_local auto s = ocarina::format(
            "matrix<{},{}>",
            N, M);
        return s;
    }
    static ocarina::string &name() noexcept {
        static thread_local auto s = ocarina::format(
            "float{}x{}",
            N, M);
        return s;
    }
};

template<typename T, size_t N>
struct TypeDesc<ocarina::array<T, N>> {
    static_assert(alignof(T) >= 4u);
    static ocarina::string &description() noexcept {
        static thread_local auto s = ocarina::format(
            "array<{},{}>",
            TypeDesc<T>::description(), N);
        return s;
    }

    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<typename T, int... Dims>
struct TypeDesc<Buffer<T, Dims...>> {
    static_assert(alignof(T) >= 4u);
    static ocarina::string &description() noexcept {
        static thread_local string str = []() -> string {
            auto ret = ocarina::format("buffer<{}", TypeDesc<T>::description());
            (ret.append(",").append(to_string(Dims)), ...);
            ret.append(">");
            return ret;
        }();
        return str;
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<typename T>
struct TypeDesc<BufferDesc<T>> : public TypeDesc<Buffer<T>> {};

template<>
struct TypeDesc<ByteBuffer> {
    static ocarina::string_view description() noexcept {
        return "bytebuffer";
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<>
struct TypeDesc<BufferDesc<>> : public TypeDesc<ByteBuffer> {};

template<>
struct TypeDesc<Texture3D> {
    static ocarina::string_view description() noexcept {
        return "texture3d";
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<>
struct TypeDesc<Texture2D> {
    static ocarina::string_view description() noexcept {
        return "texture2d";
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<>
struct TypeDesc<Accel> {
    static ocarina::string_view description() noexcept {
        return "accel";
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<typename T, size_t N>
struct TypeDesc<T[N]> : public TypeDesc<ocarina::array<T, N>> {};

template<typename... T>
struct TypeDesc<ocarina::tuple<T...>> {
    static ocarina::string &description() noexcept {
        static thread_local ocarina::string str = []() -> ocarina::string {
            auto ret = ocarina::format("struct<_Tuple,{},false,false", alignof(ocarina::tuple<T...>));
            (ret.append(",").append(TypeDesc<T>::description()), ...);
            ret.append(">");
            return ret;
        }();
        return str;
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<>
struct TypeDesc<BindlessArray> {
    static ocarina::string_view description() noexcept {
        return "bindlessArray";
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

/// make struct type description
#define OC_MAKE_STRUCT_MEMBER_FMT(member) ",{}"

#define OC_MAKE_STRUCT_MEMBER_DESC(member) \
    ocarina::TypeDesc<std::remove_cvref_t<decltype(this_type::member)>>::description()

#define OC_MAKE_STRUCT_DESC(S, ...)                                                         \
    template<>                                                                              \
    struct ocarina::TypeDesc<S> {                                                           \
        using this_type = S;                                                                \
        static ocarina::string description() noexcept {                                     \
            static thread_local ocarina::string s = ocarina::format(                        \
                "struct<" #S ",{},{},{}" MAP(OC_MAKE_STRUCT_MEMBER_FMT, ##__VA_ARGS__) ">", \
                alignof(this_type), ocarina::is_builtin_struct_v<this_type>,                \
                ocarina::is_param_struct_v<this_type>,                                      \
                MAP_LIST(OC_MAKE_STRUCT_MEMBER_DESC, ##__VA_ARGS__));                       \
            return s;                                                                       \
        }                                                                                   \
        static constexpr string_view name() noexcept {                                      \
            return #S;                                                                      \
        }                                                                                   \
    };

}// namespace ocarina