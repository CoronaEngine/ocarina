//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/stl.h"
#include "core/type.h"
#include "core/type_desc.h"
#include "core/util.h"

namespace ocarina {

class OC_CORE_API TypeRegistry {
public:
    TypeRegistry() = default;
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
    void for_each(TypeVisitor *visitor) const noexcept;
};

}// namespace ocarina