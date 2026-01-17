//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/header.h"
#include "math/basic_types.h"
#include "core/stl.h"
#include "core/macro_map.h"
#include "core/hash.h"
#include "core/image_base.h"

namespace ocarina {

template<typename T>
struct array_dimension {
    static constexpr size_t value = 0u;
};

template<typename T, size_t N>
struct array_dimension<T[N]> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct array_dimension<ocarina::array<T, N>> {
    static constexpr auto value = N;
};

OC_DEFINE_TEMPLATE_VALUE(array_dimension)

template<typename T>
struct array_element {
    using type = T;
};

template<typename T, size_t N>
struct array_element<T[N]> {
    using type = T;
};

template<typename T, size_t N>
struct array_element<ocarina::array<T, N>> {
    using type = T;
};

template<typename T>
using array_element_t = typename array_element<T>::type;

template<typename T>
class is_array : public std::false_type {};

template<typename T, size_t N>
class is_array<T[N]> : public std::true_type {};

template<typename T, size_t N>
class is_array<ocarina::array<T, N>> : public std::true_type {};

template<typename T>
constexpr auto is_array_v = is_array<T>::value;

template<typename T>
struct is_tuple : std::false_type {};

template<typename... T>
struct is_tuple<ocarina::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_tuple_v = is_tuple<T>::value;

template<typename T>
struct is_struct : std::false_type {};

template<typename... T>
struct is_struct<ocarina::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_struct_v = is_struct<T>::value;

namespace detail {

template<typename T, size_t>
using array_to_tuple_element_t = T;

template<typename T, size_t N, size_t... i>
[[nodiscard]] constexpr auto array_to_tuple_impl(ocarina::array<T, N> array, std::index_sequence<i...>) noexcept {
    return ocarina::tuple<array_to_tuple_element_t<T, i>...>(array[i]...);
}

template<typename T, size_t N>
[[nodiscard]] constexpr auto array_to_tuple_impl(ocarina::array<T, N> array = {}) noexcept {
    return array_to_tuple_impl(array, std::make_index_sequence<N>());
}

};// namespace detail

namespace detail {
template<typename T>
struct array_to_tuple {
    using type = T;
};

template<typename T, size_t N>
struct array_to_tuple<ocarina::array<T, N>> {
    using type = decltype(detail::array_to_tuple_impl<typename array_to_tuple<T>::type, N>());
};

template<typename... T>
struct array_to_tuple<ocarina::tuple<T...>> {
    using type = ocarina::tuple<T...>;
};
}// namespace detail

template<typename T>
using array_to_tuple_t = typename detail::array_to_tuple<T>::type;

template<typename T>
struct struct_member_tuple {
    using type = T;
};

template<typename... T>
struct struct_member_tuple<ocarina::tuple<T...>> {
    using type = ocarina::tuple<T...>;
};

template<typename T, size_t N>
struct struct_member_tuple<ocarina::array<T, N>> {
    using type = array_to_tuple_t<ocarina::array<T, N>>;
};

template<typename T, size_t N>
struct struct_member_tuple<T[N]> {
    using type = typename struct_member_tuple<ocarina::array<T, N>>::type;
};

template<typename T, size_t N>
struct struct_member_tuple<Vector<T, N>> {
    using type = typename struct_member_tuple<ocarina::array<T, N>>::type;
};

template<size_t N, size_t M>
struct struct_member_tuple<Matrix<N, M>> {
    using type = typename struct_member_tuple<ocarina::array<Vector<float, M>, N>>::type;
};

/// make struct reflection
#define OC_MEMBER_TYPE_MAP(member) std::remove_cvref_t<decltype(this_type::member)>
#define OC_TYPE_OFFSET_OF(member) OC_OFFSET_OF(this_type, member)
#define OC_TYPE_SIZE(member) sizeof(this_type::member)

#define OC_MAKE_STRUCT_REFLECTION(S, ...)                                                         \
    template<>                                                                                    \
    struct ocarina::is_struct<S> : std::true_type {};                                             \
    template<>                                                                                    \
    struct ocarina::struct_member_tuple<S> {                                                      \
        using this_type = S;                                                                      \
        static constexpr string_view members[] = {MAP_LIST(OC_STRINGIFY, __VA_ARGS__)};           \
        using type = ocarina::tuple<MAP_LIST(OC_MEMBER_TYPE_MAP, ##__VA_ARGS__)>;                 \
        using offset = std::index_sequence<MAP_LIST(OC_TYPE_OFFSET_OF, ##__VA_ARGS__)>;           \
        static constexpr array offset_array = {MAP_LIST(OC_TYPE_OFFSET_OF, ##__VA_ARGS__)};       \
        static constexpr auto min_size = std::min({MAP_LIST(OC_TYPE_SIZE, ##__VA_ARGS__)});       \
        static_assert(min_size >= 4 || ocarina::is_builtin_struct_v<S>,                           \
                      "Due to the memory alignment, min member size must >= 4");                  \
        static_assert(is_valid_reflection_v<this_type, type, offset>,                             \
                      "may be order of members is wrong!");                                       \
        static_assert(sizeof(this_type) >= 4);                                                    \
        static constexpr auto member_index(ocarina::string_view name) {                           \
            return std::find(std::begin(members), std::end(members), name) - std::begin(members); \
        }                                                                                         \
    };

template<typename T>
using struct_member_tuple_t = typename struct_member_tuple<T>::type;

template<typename T>
struct canonical_layout {
    using type = struct_member_tuple_t<T>;
};

template<typename... T>
struct canonical_layout<ocarina::tuple<T...>> {
    using type = ocarina::tuple<typename canonical_layout<T>::type...>;
};

template<typename T>
using canonical_layout_t = typename canonical_layout<T>::type;

/// tuple join
template<typename... T>
struct tuple_join {
    static_assert(always_false_v<T...>);
};

template<typename... T, typename... U>
struct tuple_join<ocarina::tuple<T...>, U...> {
    using type = ocarina::tuple<T..., U...>;
};

template<typename... A, typename... B, typename... C>
struct tuple_join<ocarina::tuple<A...>, ocarina::tuple<B...>, C...> {
    using type = typename tuple_join<ocarina::tuple<A..., B...>, C...>::type;
};

template<typename... T>
using tuple_join_t = typename tuple_join<T...>::type;

namespace detail {
template<typename A, typename B>
struct linear_layout_impl {
    using type = ocarina::tuple<B>;
};

template<typename... A, typename... B>
struct linear_layout_impl<ocarina::tuple<A...>, ocarina::tuple<B...>> {
    using type = tuple_join_t<ocarina::tuple<A...>,
                              typename linear_layout_impl<ocarina::tuple<>, B>::type...>;
};

};// namespace detail

template<typename T>
using linear_layout = detail::linear_layout_impl<ocarina::tuple<>, canonical_layout_t<T>>;

template<typename T>
using linear_layout_t = typename linear_layout<T>::type;

namespace detail {

template<typename T>
struct dimension_impl {
    static constexpr auto value = dimension_impl<canonical_layout_t<T>>::value;
};

template<typename T, size_t N>
struct dimension_impl<T[N]> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct dimension_impl<ocarina::array<T, N>> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct dimension_impl<Vector<T, N>> {
    static constexpr auto value = N;
};

template<size_t N, size_t M>
struct dimension_impl<Matrix<N, M>> {
    static constexpr auto value = N;
};

template<typename... T>
struct dimension_impl<ocarina::tuple<T...>> {
    static constexpr auto value = sizeof...(T);
};

}// namespace detail

template<typename T>
using dimension = detail::dimension_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto dimension_v = dimension<T>::value;

namespace detail {
template<typename T>
struct is_builtin_struct_impl {
    static constexpr bool value = false;
};
}// namespace detail

template<typename T>
struct is_builtin_struct : public detail::is_builtin_struct_impl<std::remove_cvref_t<T>> {};
OC_DEFINE_TEMPLATE_VALUE(is_builtin_struct)

#define OC_MAKE_BUILTIN_STRUCT(S)                       \
    template<>                                          \
    struct ocarina::detail::is_builtin_struct_impl<S> { \
        static constexpr bool value = true;             \
    };

namespace detail {
template<typename T>
struct is_param_struct_impl {
    static constexpr bool value = false;
};
}// namespace detail
template<typename T>
struct is_param_struct : public detail::is_param_struct_impl<std::remove_cvref_t<T>> {};
OC_DEFINE_TEMPLATE_VALUE(is_param_struct)

#define OC_MAKE_PARAM_STRUCT(S)                       \
    template<>                                        \
    struct ocarina::detail::is_param_struct_impl<S> { \
        static constexpr bool value = true;           \
    };

class Type;

class TypeRegistry;

struct BindlessArrayDesc {
    handle_ty buffer_slot;
    handle_ty tex3d_slot;
    handle_ty tex2d_slot;
};

struct TextureDesc {
    handle_ty texture{};
    handle_ty surface{};
    handle_ty array;
    PixelStorage pixel_storage{};
};

template<typename T = std::byte>
struct BufferDesc {
    T *handle{};
    uint offset{};
    uint64_t size{};

    [[nodiscard]] size_t data_alignment() const noexcept {
        return alignof(decltype(*this));
    }

    [[nodiscard]] size_t data_size() const noexcept {
        return sizeof(*this);
    }

    [[nodiscard]] MemoryBlock memory_block() const noexcept {
        return {this, data_size(), data_alignment(), sizeof(handle_ty)};
    }

    [[nodiscard]] handle_ty head() const noexcept {
        return reinterpret_cast<handle_ty>(handle);
    }

    [[nodiscard]] uint64_t size_in_byte() const noexcept {
        return size * sizeof(T);
    }

    [[nodiscard]] uint offset_in_byte() const noexcept {
        return offset * sizeof(T);
    }
};

namespace detail {
template<typename T>
struct is_buffer_proxy_impl {
    static constexpr bool value = false;
};

template<typename T>
struct is_buffer_proxy_impl<BufferDesc<T>> {
    static constexpr bool value = true;
};
}// namespace detail

template<typename T>
struct is_buffer_proxy : public detail::is_buffer_proxy_impl<std::remove_cvref_t<T>> {};
OC_DEFINE_TEMPLATE_VALUE(is_buffer_proxy);

using ByteBufferDesc = BufferDesc<>;

struct TypeVisitor {
    virtual void visit(const Type *) noexcept = 0;
};

class OC_AST_API Type : public concepts::Noncopyable, public Hashable {
public:
    enum struct Tag : uint32_t {
        BOOL,
        FLOAT,
        INT,
        UINT,
        UCHAR,
        CHAR,
        SHORT,
        USHORT,
        UINT64T,

        VECTOR,
        MATRIX,

        ARRAY,
        STRUCTURE,

        BUFFER,
        BYTE_BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        ACCEL,

        NONE
    };
    friend class TypeRegistry;

private:
    size_t size_{0};
    size_t index_{0};
    size_t alignment_{0};
    uint32_t dimension_{0};
    Tag tag_{Tag::NONE};
    ocarina::string description_;
    ocarina::string name_;
    mutable ocarina::string cname_;
    mutable ocarina::vector<string_view> member_name_;
    ocarina::vector<const Type *> members_;
    [[nodiscard]] uint64_t compute_hash() const noexcept override;
    vector<int> dims_;
    bool builtin_struct_{false};
    bool param_struct_{false};

private:
    void update_name(ocarina::string_view desc) noexcept;
    void set_description(ocarina::string_view desc) noexcept {
        description_ = desc;
        update_name(desc);
    }
    void update_member_name(const string_view *names, int num) noexcept {
        member_name_.clear();
        for (int i = 0; i < num; ++i) {
            member_name_.push_back(names[i]);
        }
    }

public:
    Type() = default;
    static void for_each(TypeVisitor *visitor);
    template<typename T>
    [[nodiscard]] static const Type *of() noexcept;

    template<typename T>
    [[nodiscard]] static auto of(T &&) noexcept { return of<std::remove_cvref_t<T>>(); }
    [[nodiscard]] static const Type *from(ocarina::string_view description) noexcept;
    [[nodiscard]] static const Type *at(uint32_t uid) noexcept;
    [[nodiscard]] static size_t count() noexcept;
    [[nodiscard]] const Type *get_member(ocarina::string_view name) const noexcept;
    [[nodiscard]] ocarina::span<const string_view> member_name() const noexcept { return member_name_; }
    [[nodiscard]] bool operator==(const Type &rhs) const noexcept { return hash() == rhs.hash(); }
    [[nodiscard]] bool operator!=(const Type &rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] bool operator<(const Type &rhs) const noexcept { return index_ < rhs.index_; }
    [[nodiscard]] constexpr size_t index() const noexcept { return index_; }
    [[nodiscard]] constexpr size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr size_t alignment() const noexcept { return alignment_; }
    [[nodiscard]] const vector<int> &dims() const noexcept { return dims_; }
    [[nodiscard]] bool has_multi_dims() const noexcept { return !dims_.empty(); }
    [[nodiscard]] constexpr Tag tag() const noexcept { return tag_; }
    [[nodiscard]] auto description() const noexcept { return ocarina::string_view{description_}; }
    [[nodiscard]] ocarina::string name() const noexcept { return name_; }
    [[nodiscard]] ocarina::string cname() const noexcept { return cname_; }
    void set_cname(string s) const noexcept;
    [[nodiscard]] ocarina::string simple_cname() const noexcept;
    [[nodiscard]] constexpr int dimension() const noexcept { return dimension_; }
    [[nodiscard]] ocarina::span<const Type *const> members() const noexcept;
    [[nodiscard]] const Type *element() const noexcept;
    [[nodiscard]] bool is_valid() const noexcept;
    [[nodiscard]] constexpr bool is_scalar() const noexcept {
        return tag_ == Tag::BOOL || tag_ == Tag::FLOAT || tag_ == Tag::INT ||
               tag_ == Tag::UINT || tag_ == Tag::UCHAR || tag_ == Tag::CHAR ||
               tag_ == Tag::USHORT || tag_ == Tag::SHORT;
    }
    [[nodiscard]] size_t max_member_size() const noexcept;
    [[nodiscard]] constexpr bool is_builtin_struct() const noexcept { return builtin_struct_; }
    [[nodiscard]] constexpr bool is_param_struct() const noexcept { return param_struct_; }
    [[nodiscard]] constexpr bool is_basic() const noexcept { return is_scalar() || is_vector() || is_matrix(); }
    [[nodiscard]] constexpr bool is_array() const noexcept { return tag_ == Tag::ARRAY; }
    [[nodiscard]] constexpr bool is_vector() const noexcept { return tag_ == Tag::VECTOR; }
    [[nodiscard]] constexpr bool is_matrix() const noexcept { return tag_ == Tag::MATRIX; }
    [[nodiscard]] constexpr bool is_structure() const noexcept { return tag_ == Tag::STRUCTURE; }
    [[nodiscard]] constexpr bool is_buffer() const noexcept { return tag_ == Tag::BUFFER; }
    [[nodiscard]] constexpr bool is_byte_buffer() const noexcept { return tag_ == Tag::BYTE_BUFFER; }
    [[nodiscard]] constexpr bool is_texture() const noexcept { return tag_ == Tag::TEXTURE; }
    [[nodiscard]] constexpr bool is_bindless_array() const noexcept { return tag_ == Tag::BINDLESS_ARRAY; }
    [[nodiscard]] constexpr bool is_accel() const noexcept { return tag_ == Tag::ACCEL; }
    [[nodiscard]] constexpr bool is_resource() const noexcept {
        return is_buffer() || is_byte_buffer() || is_texture() || is_accel() || is_bindless_array();
    }
};

}// namespace ocarina