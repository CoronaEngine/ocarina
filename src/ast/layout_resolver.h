//
// Created by Z on 11/04/2026.
//

#pragma once

#include <algorithm>

#include "core/stl.h"
#include "type_registry.h"

namespace ocarina {

enum class PrecisionPolicy : uint8_t {
    force_f32,
    force_f16,
    auto_select,
};

struct DevicePrecisionCaps {
    uint32_t compute_capability{0};
    size_t total_vram_bytes{0};
    bool has_native_fp16{false};
    bool has_fast_fp16{false};
    bool has_tensor_fp16{false};

    [[nodiscard]] PrecisionPolicy recommend_policy(
        size_t estimated_scene_vram_bytes = 0) const noexcept;
};

struct StoragePrecisionPolicy {
    PrecisionPolicy policy = PrecisionPolicy::force_f32;
    bool allow_real_in_storage = false;
};

class LayoutResolver {
private:
    StoragePrecisionPolicy policy_{};

public:
    explicit LayoutResolver(StoragePrecisionPolicy policy = {}) noexcept;
    OC_MAKE_MEMBER_GETTER_SETTER(policy, &)
    [[nodiscard]] const Type *resolve(const Type *type) const noexcept;
    template<typename T>
    [[nodiscard]] const Type *resolve() const noexcept {
        return resolve(Type::of<T>());
    }
    [[nodiscard]] string resolve_description(const Type *type) const noexcept;

private:
    [[nodiscard]] string resolve_real_description() const noexcept;
    [[nodiscard]] const Type *resolve_real_container_element(const Type *type) const noexcept;
    [[nodiscard]] string resolve_vector_description(const Type *type) const noexcept;
    [[nodiscard]] string resolve_matrix_description(const Type *type) const noexcept;
    [[nodiscard]] string resolve_array_description(const Type *type) const noexcept;
    [[nodiscard]] string resolve_buffer_description(const Type *type) const noexcept;
    [[nodiscard]] size_t resolved_alignment(const Type *type) const noexcept;
    [[nodiscard]] string resolve_structure_description(const Type *type) const noexcept;
};

}// namespace ocarina