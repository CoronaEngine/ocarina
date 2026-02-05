//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/stl.h"
#include "type.h"
#include <mutex>
#include "core/util.h"
#include "type_desc.h"

namespace ocarina {

template<typename T>
const Type *Type::of() noexcept {
    using raw_type = std::remove_cvref_t<T>;
    constexpr bool is_builtin = is_builtin_struct_v<raw_type>;
    const Type *ret = Type::from(TypeDesc<raw_type>::description());
    if constexpr (ocarina::is_struct_v<T>) {
        if constexpr (requires {
                          ocarina::struct_member_tuple<raw_type>::members;
                      }) {
            constexpr auto arr = ocarina::struct_member_tuple<raw_type>::members;
            constexpr int num = sizeof(ocarina::struct_member_tuple<raw_type>::members) / sizeof(arr[0]);
            const_cast<Type *>(ret)->update_member_name(arr, num);
        }
        using member_tuple = typename ocarina::struct_member_tuple<raw_type>::type;
        traverse_tuple(member_tuple{}, [&](auto elm) {
            using elm_t = decltype(elm);
            auto t = Type::of<elm_t>();
        });
    }
    return ret;
}

template<typename T>
[[nodiscard]] string to_str(const T &val) noexcept {
    static string type_string = string(TypeDesc<T>::name());
    if constexpr (is_vector2_v<T>) {
        return ocarina::format(type_string + "({}, {})", to_str(val.x), to_str(val.y));
    } else if constexpr (is_vector3_v<T>) {
        return ocarina::format(type_string + "({}, {}, {})", to_str(val.x), to_str(val.y), to_str(val.z));
    } else if constexpr (is_vector4_v<T>) {
        return ocarina::format(type_string + "({}, {}, {}, {})", to_str(val.x), to_str(val.y), to_str(val.z), to_str(val.w));
    } else if constexpr (is_matrix2_v<T>) {
        return ocarina::format("[{},\n {}]", to_str(val[0]), to_str(val[1]));
    } else if constexpr (is_matrix3_v<T>) {
        return ocarina::format("[{},\n {},\n {}]", to_str(val[0]), to_str(val[1]), to_str(val[2]));
    } else if constexpr (is_matrix4_v<T>) {
        return ocarina::format("[{},\n {},\n {},\n {}]", to_str(val[0]), to_str(val[1]), to_str(val[2]), to_str(val[3]));
    } else if constexpr (is_scalar_v<T>) {
        if constexpr (is_half_v<T>) {
            return std::to_string(half2float(val));
        } else {
            return std::to_string(val);
        }
    } else if constexpr (is_struct_v<T>) {
        string ret = ocarina::format("{}[", struct_member_tuple<T>::struct_name);
        traverse_tuple(struct_member_tuple_t<T>{}, [&]<typename Elm>(const Elm &_, uint index) {
            constexpr auto offset_array = struct_member_tuple<T>::offset_array;
            auto head = reinterpret_cast<const std::byte *>(addressof(val));
            auto addr = head + offset_array[index];
            const Elm &elm = reinterpret_cast<const Elm &>(*addr);
            if (index == offset_array.size() - 1) {
                ret += to_str(elm);
            } else {
                ret += to_str(elm) + ",";
            }
        });
        return ret + "]";
    } else {
        static_assert(always_false_v<T>);
        return "";
    }
}

namespace detail {
template<typename S, typename Members, typename offsets>
struct is_valid_reflection : std::false_type {};

template<typename S, typename... M, typename I, I... os>
struct is_valid_reflection<S, ocarina::tuple<M...>, std::integer_sequence<I, os...>> {
    //    static_assert(((!is_struct_v<M>)&&...));
    static_assert((!is_bool_vector_v<M> && ...),
                  "Boolean vectors are not allowed in DSL "
                  "structures since their may have different "
                  "layouts on different platforms.");

private:
    [[nodiscard]] static constexpr bool _check() noexcept {
        constexpr auto count = sizeof...(M);
        static_assert(sizeof...(os) == count);
        constexpr ocarina::array<size_t, count> sizes{sizeof(M)...};
        constexpr ocarina::array<size_t, count> alignments{alignof(M)...};
        constexpr ocarina::array<size_t, count> offsets{os...};
        size_t cur_offset = 0u;
        for (auto i = 0u; i < count; ++i) {
            auto offset = offsets[i];
            auto size = sizes[i];
            auto alignment = alignments[i];
            cur_offset = mem_offset(cur_offset, alignment);
            if (cur_offset != offset) {
                return false;
            }
            cur_offset += size;
        }
        constexpr auto struct_size = sizeof(S);
        constexpr auto struct_alignment = alignof(S);
        cur_offset = mem_offset(cur_offset, struct_alignment);
        return cur_offset == struct_size;
    };

public:
    static constexpr bool value = _check();
};
}// namespace detail

template<typename S, typename M, typename I>
static constexpr bool is_valid_reflection_v = detail::is_valid_reflection<S, M, I>::value;

class OC_AST_API TypeRegistry {
public:
    struct TypePtrHash {
        using is_transparent = void;
        [[nodiscard]] uint64_t operator()(const Type *type) const noexcept { return type->hash(); }
        [[nodiscard]] uint64_t operator()(uint64_t hash) const noexcept { return hash; }
    };

    struct TypePtrEqual {
        using is_transparent = void;
        template<typename Lhs, typename Rhs>
        [[nodiscard]] bool operator()(Lhs &&lhs, Rhs &&rhs) const noexcept {
            constexpr TypePtrHash hash;
            return hash(std::forward<Lhs>(lhs)) == hash(std::forward<Rhs>(rhs));
        }
    };

    ocarina::vector<ocarina::unique_ptr<Type>> types_;
    ocarina::unordered_set<Type *, TypePtrHash, TypePtrEqual> type_set_;
    mutable std::mutex mutex_;
    TypeRegistry() = default;

private:
    void parse_vector(Type *type, ocarina::string_view desc) noexcept;
    void parse_matrix(Type *type, ocarina::string_view desc) noexcept;
    void parse_array(Type *type, ocarina::string_view desc) noexcept;
    void parse_buffer(Type *type, ocarina::string_view desc) noexcept;
    void parse_texture3d(Type *type, ocarina::string_view desc) noexcept;
    void parse_texture2d(Type *type, ocarina::string_view desc) noexcept;
    void parse_accel(Type *type, ocarina::string_view desc) noexcept;
    void parse_byte_buffer(Type *type, ocarina::string_view desc) noexcept;
    void parse_struct(Type *type, ocarina::string_view desc) noexcept;
    void parse_bindless_array(Type *type, ocarina::string_view desc) noexcept;

public:
    [[nodiscard]] static uint64_t compute_hash(ocarina::string_view desc) noexcept;
    TypeRegistry &operator=(const TypeRegistry &) = delete;
    TypeRegistry &operator=(TypeRegistry &&) = delete;
    [[nodiscard]] static TypeRegistry &instance() noexcept;
    [[nodiscard]] const Type *parse_type(ocarina::string_view desc) noexcept;
    [[nodiscard]] bool is_exist(ocarina::string_view desc) const noexcept;
    [[nodiscard]] bool is_exist(uint64_t hash) const noexcept;
    [[nodiscard]] const Type *type_from(ocarina::string_view desc) noexcept;
    [[nodiscard]] const Type *type_at(uint i) const noexcept;
    [[nodiscard]] size_t type_count() const noexcept;
    void add_type(ocarina::unique_ptr<Type> type);
    static void try_add_to_current_function(const Type *type) noexcept;
    void for_each(TypeVisitor *visitor) const noexcept;
};

};// namespace ocarina