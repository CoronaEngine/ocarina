//
// Created by GitHub Copilot on 2026/4/13.
//

#pragma once

#include <cstdint>

#include "core/dynamic_buffer/dynamic_buffer_layout_codec.h"
#include "core/type_system/precision_policy.h"
#include "core/type.h"

namespace ocarina {

struct ByteRegion {
    size_t begin_byte{0u};
    size_t end_byte{0u};

    [[nodiscard]] size_t size() const noexcept {
        return end_byte - begin_byte;
    }

    [[nodiscard]] bool empty() const noexcept {
        return begin_byte >= end_byte;
    }
};

struct ByteSegment {
    size_t storage_begin_byte{0u};
    size_t staging_begin_byte{0u};
    size_t size_in_bytes{0u};

    [[nodiscard]] bool empty() const noexcept {
        return size_in_bytes == 0u;
    }
};

class TypedFieldPath {
public:
    enum class StepKind : uint8_t {
        member,
        index,
        component
    };

    struct Step {
        StepKind kind{};
        uint32_t value{0u};
    };

private:
    vector<Step> steps_{};

public:
    TypedFieldPath() = default;

    [[nodiscard]] span<const Step> steps() const noexcept { return steps_; }
    [[nodiscard]] bool empty() const noexcept { return steps_.empty(); }

    [[nodiscard]] TypedFieldPath append_member(uint32_t member_index) const;
    [[nodiscard]] TypedFieldPath append_index(uint32_t element_index) const;
    [[nodiscard]] TypedFieldPath append_component(uint32_t component_index) const;

    [[nodiscard]] static TypedFieldPath member(uint32_t member_index) noexcept;
};

template<uint32_t Index>
struct FieldMemberStep {};

template<uint32_t Index>
struct FieldIndexStep {};

template<uint32_t Index>
struct FieldComponentStep {};

namespace detail {

template<typename Step>
struct TypedFieldPathStepAppender;

template<uint32_t Index>
struct TypedFieldPathStepAppender<FieldMemberStep<Index>> {
    [[nodiscard]] static TypedFieldPath append(const TypedFieldPath &path) {
        return path.append_member(Index);
    }
};

template<uint32_t Index>
struct TypedFieldPathStepAppender<FieldIndexStep<Index>> {
    [[nodiscard]] static TypedFieldPath append(const TypedFieldPath &path) {
        return path.append_index(Index);
    }
};

template<uint32_t Index>
struct TypedFieldPathStepAppender<FieldComponentStep<Index>> {
    [[nodiscard]] static TypedFieldPath append(const TypedFieldPath &path) {
        return path.append_component(Index);
    }
};

}// namespace detail

template<typename... Steps>
[[nodiscard]] TypedFieldPath make_typed_field_path() {
    TypedFieldPath path;
    ((path = detail::TypedFieldPathStepAppender<Steps>::append(path)), ...);
    return path;
}

class DynamicBufferLayoutPlan {
private:
    const Type *logical_type_{nullptr};
    StoragePrecisionPolicy policy_{};
    size_t element_size_bytes_{0u};
    size_t element_alignment_{0u};
    bool has_precision_lowering_{false};

public:
    DynamicBufferLayoutPlan() = default;
    DynamicBufferLayoutPlan(const Type *logical_type,
                            StoragePrecisionPolicy policy,
                            size_t element_size_bytes,
                            size_t element_alignment,
                            bool has_precision_lowering) noexcept
        : logical_type_(logical_type),
          policy_(policy),
          element_size_bytes_(element_size_bytes),
          element_alignment_(element_alignment),
          has_precision_lowering_(has_precision_lowering) {}

    [[nodiscard]] static DynamicBufferLayoutPlan create(const Type *logical_type,
                                                        StoragePrecisionPolicy policy);

    [[nodiscard]] const Type *logical_type() const noexcept { return logical_type_; }
    [[nodiscard]] const Type *resolved_type() const noexcept { return Type::resolve(logical_type_, policy_); }
    [[nodiscard]] StoragePrecisionPolicy policy() const noexcept { return policy_; }
    [[nodiscard]] size_t element_size_bytes() const noexcept { return element_size_bytes_; }
    [[nodiscard]] size_t element_alignment() const noexcept { return element_alignment_; }
    [[nodiscard]] bool has_precision_lowering() const noexcept { return has_precision_lowering_; }

    [[nodiscard]] size_t storage_bytes(size_t element_count) const noexcept;

    [[nodiscard]] bool is_compatible_with(const Type *type) const noexcept {
        return logical_type_ == type;
    }

    [[nodiscard]] ByteRegion record_region(size_t index) const noexcept;
    [[nodiscard]] ByteRegion field_region(size_t index,
                                          const TypedFieldPath &path) const noexcept;
    [[nodiscard]] vector<ByteSegment> record_segments(size_t element_count,
                                                      size_t index) const noexcept;
    [[nodiscard]] vector<ByteSegment> field_segments(size_t element_count,
                                                     size_t index,
                                                     const TypedFieldPath &path) const noexcept;
    [[nodiscard]] const Type *field_logical_type(const TypedFieldPath &path) const noexcept;
    [[nodiscard]] const Type *field_resolved_type(const TypedFieldPath &path) const noexcept;
};

}// namespace ocarina