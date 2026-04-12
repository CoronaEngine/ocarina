//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/stl.h"
#include "core/type.h"
#include <mutex>
#include "core/util.h"
#include "type_desc.h"

namespace ocarina {

namespace detail {
OC_AST_API void ensure_type_registry_callbacks_registered() noexcept;
}

template<typename T>
const Type *Type::of() noexcept {
    using raw_type = std::remove_cvref_t<T>;
    constexpr bool is_builtin = is_builtin_struct_v<raw_type>;
    detail::ensure_type_registry_callbacks_registered();
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