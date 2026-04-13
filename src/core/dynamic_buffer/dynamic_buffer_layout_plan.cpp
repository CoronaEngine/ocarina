//
// Created by GitHub Copilot on 2026/4/13.
//

#include "dynamic_buffer_layout_plan.h"

#include "core/type_desc.h"

namespace ocarina {

namespace {

[[nodiscard]] size_t align_up_size(size_t value, size_t alignment) noexcept {
    OC_ASSERT(alignment != 0u);
    return (value + alignment - 1u) / alignment * alignment;
}

[[nodiscard]] size_t resolved_runtime_alignment(const Type *type) noexcept;

[[nodiscard]] size_t resolved_runtime_size(const Type *type) noexcept {
    OC_ASSERT(type != nullptr);
    if (type->is_scalar()) {
        return type->size();
    }
    if (type->is_vector()) {
        return resolved_runtime_size(type->element()) *
               (type->dimension() == 3 ? 4u : static_cast<size_t>(type->dimension()));
    }
    if (type->is_matrix() || type->is_array()) {
        return resolved_runtime_size(type->element()) * static_cast<size_t>(type->dimension());
    }
    if (type->is_structure()) {
        size_t size = 0u;
        for (const auto *member : type->members()) {
            size = align_up_size(size, resolved_runtime_alignment(member));
            size += resolved_runtime_size(member);
        }
        return align_up_size(size, resolved_runtime_alignment(type));
    }
    return type->size();
}

[[nodiscard]] size_t resolved_runtime_alignment(const Type *type) noexcept {
    OC_ASSERT(type != nullptr);
    if (type->is_scalar()) {
        return type->alignment();
    }
    if (type->is_vector()) {
        return resolved_runtime_size(type);
    }
    if (type->is_matrix() || type->is_array()) {
        return resolved_runtime_alignment(type->element());
    }
    if (type->is_structure()) {
        size_t align = 0u;
        for (const auto *member : type->members()) {
            align = std::max(align, resolved_runtime_alignment(member));
        }
        return align;
    }
    return type->alignment();
}

[[nodiscard]] string resolve_real_description(StoragePrecisionPolicy policy) noexcept {
    if (!policy.allow_real_in_storage) {
        return {};
    }
    switch (policy.policy) {
        case PrecisionPolicy::force_f16:
            return string(TypeDesc<half>::description());
        case PrecisionPolicy::force_f32:
        default:
            return string(TypeDesc<float>::description());
    }
}

[[nodiscard]] const Type *resolve_type_description(const Type *type,
                                                   StoragePrecisionPolicy policy) noexcept;

[[nodiscard]] const Type *resolve_real_container_element(const Type *type,
                                                         StoragePrecisionPolicy policy) noexcept {
    const auto *resolved = resolve_type_description(type, policy);
    if (resolved == nullptr) {
        return nullptr;
    }
    return resolved;
}

[[nodiscard]] size_t resolved_alignment(const Type *type,
                                        StoragePrecisionPolicy policy) noexcept {
    if (type == nullptr) {
        return 0u;
    }
    if (type->tag() == Type::Tag::REAL) {
        const auto *resolved = resolve_type_description(type, policy);
        return resolved == nullptr ? 0u : resolved->alignment();
    }
    if (type->is_scalar() || type->is_byte_buffer() || type->is_texture() ||
        type->is_bindless_array() || type->is_accel()) {
        return type->alignment();
    }
    if (type->is_vector() || type->is_matrix() || type->is_array() || type->is_buffer()) {
        const auto *elem = resolve_real_container_element(type->element(), policy);
        return elem == nullptr ? 0u : elem->alignment();
    }
    if (type->is_structure()) {
        size_t align = 0u;
        for (const auto *member : type->members()) {
            align = std::max(align, resolved_alignment(member, policy));
        }
        return align == 0u ? type->alignment() : align;
    }
    return type->alignment();
}

[[nodiscard]] string resolve_description(const Type *type,
                                         StoragePrecisionPolicy policy) noexcept {
    if (type == nullptr) {
        return {};
    }
    if (type->tag() == Type::Tag::REAL) {
        return resolve_real_description(policy);
    }
    if (type->is_scalar()) {
        return string(type->description());
    }
    if (type->is_vector()) {
        const auto *elem = resolve_type_description(type->element(), policy);
        return elem == nullptr ? string{} : ocarina::format("vector<{},{}>", elem->description(), type->dimension());
    }
    if (type->is_matrix()) {
        const auto *col = resolve_type_description(type->element(), policy);
        if (col == nullptr || !col->is_vector()) {
            return {};
        }
        const auto *scalar = col->element();
        return scalar == nullptr ? string{} : ocarina::format("matrix<{},{},{}>",
                                                               scalar->description(),
                                                               type->dimension(),
                                                               col->dimension());
    }
    if (type->is_array()) {
        const auto *elem = resolve_real_container_element(type->element(), policy);
        return elem == nullptr ? string{} : ocarina::format("array<{},{}>", elem->description(), type->dimension());
    }
    if (type->is_buffer()) {
        const auto *elem = resolve_real_container_element(type->element(), policy);
        return elem == nullptr ? string{} : ocarina::format("buffer<{}>", elem->description());
    }
    if (type->is_structure()) {
        string ret = ocarina::format("struct<{},{},{},{}",
                                     type->cname(),
                                     resolved_alignment(type, policy),
                                     type->is_builtin_struct(),
                                     type->is_param_struct());
        for (const auto *member : type->members()) {
            const auto desc = resolve_description(member, policy);
            if (desc.empty()) {
                return {};
            }
            ret.append(",").append(desc);
        }
        ret.push_back('>');
        return ret;
    }
    if (type->is_byte_buffer() || type->is_texture() || type->is_bindless_array() || type->is_accel()) {
        return string(type->description());
    }
    return {};
}

[[nodiscard]] const Type *resolve_type_description(const Type *type,
                                                   StoragePrecisionPolicy policy) noexcept {
    const auto desc = resolve_description(type, policy);
    return desc.empty() ? nullptr : Type::from(desc);
}

[[nodiscard]] bool contains_real_recursive(const Type *type) noexcept {
    if (type == nullptr) {
        return false;
    }
    if (type->tag() == Type::Tag::REAL) {
        return true;
    }
    if (type->is_vector() || type->is_matrix() || type->is_array() || type->is_buffer()) {
        return contains_real_recursive(type->element());
    }
    if (type->is_structure()) {
        for (const auto *member : type->members()) {
            if (contains_real_recursive(member)) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] bool has_precision_lowering_recursive(const Type *logical, const Type *resolved) noexcept {
    if (logical == nullptr || resolved == nullptr) {
        return false;
    }
    if (logical->tag() == Type::Tag::REAL) {
        return resolved->tag() == Type::Tag::HALF;
    }
    if ((logical->is_vector() || logical->is_matrix() || logical->is_array()) &&
        (resolved->is_vector() || resolved->is_matrix() || resolved->is_array())) {
        return has_precision_lowering_recursive(logical->element(), resolved->element());
    }
    if (logical->is_structure() && resolved->is_structure()) {
        const auto logical_members = logical->members();
        const auto resolved_members = resolved->members();
        if (logical_members.size() != resolved_members.size()) {
            return false;
        }
        for (size_t index = 0; index < logical_members.size(); ++index) {
            if (has_precision_lowering_recursive(logical_members[index], resolved_members[index])) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] size_t soa_storage_bytes(const Type *type, size_t count) noexcept {
    if (type == nullptr || count == 0u) {
        return 0u;
    }
    if (type->is_scalar() || type->is_vector()) {
        return resolved_runtime_size(type) * count;
    }
    if (type->is_matrix() || type->is_array()) {
        return type->dimension() * soa_storage_bytes(type->element(), count);
    }
    if (type->is_structure()) {
        size_t total = 0u;
        for (const auto *member : type->members()) {
            total += soa_storage_bytes(member, count);
        }
        return total;
    }
    return type->size() * count;
}

struct FieldRegionInfo {
    const Type *logical_type{nullptr};
    const Type *resolved_type{nullptr};
    size_t offset_in_record{0u};
    size_t size_in_bytes{0u};

    [[nodiscard]] bool valid() const noexcept {
        return logical_type != nullptr && resolved_type != nullptr;
    }
};

struct FieldStorageInfo {
    const Type *logical_type{nullptr};
    const Type *resolved_type{nullptr};
    size_t storage_offset{0u};
    size_t record_stride{0u};

    [[nodiscard]] bool valid() const noexcept {
        return logical_type != nullptr && resolved_type != nullptr;
    }
};

enum class FieldOffsetMode : uint8_t {
    aos,
    soa,
};

struct FieldAccessInfo {
    const Type *logical_type{nullptr};
    const Type *resolved_type{nullptr};
    size_t offset{0u};
    size_t size_in_bytes{0u};
    size_t record_stride{0u};

    [[nodiscard]] bool valid() const noexcept {
        return logical_type != nullptr && resolved_type != nullptr;
    }
};

[[nodiscard]] size_t field_member_offset(const Type *resolved_type,
                                         uint32_t member_index,
                                         FieldOffsetMode mode,
                                         size_t element_count) noexcept {
    size_t offset = 0u;
    for (uint32_t index = 0u; index < member_index; ++index) {
        const auto *member = resolved_type->members()[index];
        if (mode == FieldOffsetMode::aos) {
            offset = align_up_size(offset, resolved_runtime_alignment(member));
            offset += resolved_runtime_size(member);
        } else {
            offset += soa_storage_bytes(member, element_count);
        }
    }
    return offset;
}

[[nodiscard]] size_t field_element_offset(const Type *resolved_elem,
                                          uint32_t element_index,
                                          FieldOffsetMode mode,
                                          size_t element_count) noexcept {
    if (mode == FieldOffsetMode::aos) {
        return element_index * resolved_runtime_size(resolved_elem);
    }
    return element_index * soa_storage_bytes(resolved_elem, element_count);
}

[[nodiscard]] FieldAccessInfo resolve_field_access_info(const Type *logical_type,
                                                        const Type *resolved_type,
                                                        span<const TypedFieldPath::Step> steps,
                                                        FieldOffsetMode mode,
                                                        size_t element_count,
                                                        size_t current_offset = 0u) noexcept {
    if (logical_type == nullptr || resolved_type == nullptr) {
        return {};
    }
    size_t record_stride = 0u;
    while (!steps.empty()) {
        const auto step = steps.front();
        steps = steps.subspan(1u);
        if (logical_type->is_structure()) {
            if (step.kind != TypedFieldPath::StepKind::member) {
                return {};
            }
            const auto logical_members = logical_type->members();
            const auto resolved_members = resolved_type->members();
            if (logical_members.size() != resolved_members.size() || step.value >= logical_members.size()) {
                return {};
            }
            current_offset += field_member_offset(resolved_type, step.value, mode, element_count);
            const auto *resolved_member = resolved_members[step.value];
            if (mode == FieldOffsetMode::aos) {
                current_offset = align_up_size(current_offset, resolved_runtime_alignment(resolved_member));
            }
            logical_type = logical_members[step.value];
            resolved_type = resolved_member;
            continue;
        }
        if (logical_type->is_array()) {
            if (step.kind != TypedFieldPath::StepKind::index || step.value >= logical_type->dimension()) {
                return {};
            }
            const auto *resolved_elem = resolved_type->element();
            current_offset += field_element_offset(resolved_elem, step.value, mode, element_count);
            logical_type = logical_type->element();
            resolved_type = resolved_elem;
            continue;
        }
        if (logical_type->is_vector()) {
            if (step.kind != TypedFieldPath::StepKind::component || step.value >= logical_type->dimension()) {
                return {};
            }
            const auto *logical_elem = logical_type->element();
            const auto *resolved_elem = resolved_type->element();
            current_offset += step.value * resolved_runtime_size(resolved_elem);
            if (mode == FieldOffsetMode::soa && steps.empty()) {
                record_stride = resolved_runtime_size(resolved_type);
            }
            logical_type = logical_elem;
            resolved_type = resolved_elem;
            continue;
        }
        if (logical_type->is_matrix()) {
            if ((step.kind != TypedFieldPath::StepKind::index && step.kind != TypedFieldPath::StepKind::component) ||
                step.value >= logical_type->dimension()) {
                return {};
            }
            const auto *resolved_elem = resolved_type->element();
            current_offset += field_element_offset(resolved_elem, step.value, mode, element_count);
            logical_type = logical_type->element();
            resolved_type = resolved_elem;
            continue;
        }
        return {};
    }
    return FieldAccessInfo{logical_type,
                           resolved_type,
                           current_offset,
                           resolved_runtime_size(resolved_type),
                           record_stride == 0u ? resolved_runtime_size(resolved_type) : record_stride};
}

[[nodiscard]] FieldRegionInfo resolve_field_region_info(const Type *logical_type,
                                                        const Type *resolved_type,
                                                        span<const TypedFieldPath::Step> steps,
                                                        size_t current_offset) noexcept {
    auto info = resolve_field_access_info(logical_type,
                                          resolved_type,
                                          steps,
                                          FieldOffsetMode::aos,
                                          1u,
                                          current_offset);
    if (!info.valid()) {
        return {};
    }
    return FieldRegionInfo{info.logical_type, info.resolved_type, info.offset, info.size_in_bytes};
}

[[nodiscard]] FieldStorageInfo resolve_field_storage_info(const Type *logical_type,
                                                          const Type *resolved_type,
                                                          span<const TypedFieldPath::Step> steps,
                                                          size_t current_offset,
                                                          size_t element_count) noexcept {
    auto info = resolve_field_access_info(logical_type,
                                          resolved_type,
                                          steps,
                                          FieldOffsetMode::soa,
                                          element_count,
                                          current_offset);
    if (!info.valid()) {
        return {};
    }
    return FieldStorageInfo{info.logical_type, info.resolved_type, info.offset, info.record_stride};
}

void collect_soa_segments(const Type *resolved_type,
                          size_t element_count,
                          size_t index,
                          size_t storage_offset,
                          size_t staging_offset,
                          vector<ByteSegment> &segments) noexcept {
    if (resolved_type == nullptr) {
        return;
    }
    if (resolved_type->is_scalar() || resolved_type->is_vector()) {
        const auto stride = resolved_runtime_size(resolved_type);
        segments.emplace_back(ByteSegment{
            .storage_begin_byte = storage_offset + index * stride,
            .staging_begin_byte = staging_offset,
            .size_in_bytes = stride});
        return;
    }
    if (resolved_type->is_matrix() || resolved_type->is_array()) {
        const auto *resolved_elem = resolved_type->element();
        size_t current_storage = storage_offset;
        size_t current_staging = staging_offset;
        for (uint32_t element_index = 0u; element_index < resolved_type->dimension(); ++element_index) {
            collect_soa_segments(resolved_elem,
                                 element_count,
                                 index,
                                 current_storage,
                                 current_staging,
                                 segments);
            current_storage += soa_storage_bytes(resolved_elem, element_count);
            current_staging += soa_storage_bytes(resolved_elem, 1u);
        }
        return;
    }
    if (resolved_type->is_structure()) {
        size_t current_storage = storage_offset;
        size_t current_staging = staging_offset;
        for (const auto *member : resolved_type->members()) {
            collect_soa_segments(member,
                                 element_count,
                                 index,
                                 current_storage,
                                 current_staging,
                                 segments);
            current_storage += soa_storage_bytes(member, element_count);
            current_staging += soa_storage_bytes(member, 1u);
        }
        return;
    }
    const auto size = resolved_runtime_size(resolved_type);
    segments.emplace_back(ByteSegment{
        .storage_begin_byte = storage_offset + index * size,
        .staging_begin_byte = staging_offset,
        .size_in_bytes = size});
}

void collect_soa_field_segments(const FieldStorageInfo &info,
                                size_t element_count,
                                size_t index,
                                vector<ByteSegment> &segments) noexcept {
    if (!info.valid()) {
        return;
    }
    if (info.resolved_type->is_scalar()) {
        const auto size = resolved_runtime_size(info.resolved_type);
        segments.emplace_back(ByteSegment{
            .storage_begin_byte = info.storage_offset + index * info.record_stride,
            .staging_begin_byte = 0u,
            .size_in_bytes = size});
        return;
    }
    collect_soa_segments(info.resolved_type, element_count, index, info.storage_offset, 0u, segments);
}

}// namespace

TypedFieldPath TypedFieldPath::append_member(uint32_t member_index) const {
    auto copy = *this;
    copy.steps_.push_back(Step{.kind = StepKind::member, .value = member_index});
    return copy;
}

TypedFieldPath TypedFieldPath::append_index(uint32_t element_index) const {
    auto copy = *this;
    copy.steps_.push_back(Step{.kind = StepKind::index, .value = element_index});
    return copy;
}

TypedFieldPath TypedFieldPath::append_component(uint32_t component_index) const {
    auto copy = *this;
    copy.steps_.push_back(Step{.kind = StepKind::component, .value = component_index});
    return copy;
}

TypedFieldPath TypedFieldPath::member(uint32_t member_index) noexcept {
    TypedFieldPath path;
    path.steps_.push_back(Step{.kind = StepKind::member, .value = member_index});
    return path;
}

DynamicBufferLayoutPlan DynamicBufferLayoutPlan::create(const Type *logical_type,
                                                        StoragePrecisionPolicy policy,
                                                        DynamicBufferLayout layout) {
    OC_ASSERT(logical_type != nullptr);
    const auto *resolved_type = resolve_type_description(logical_type, policy);
    OC_ASSERT(resolved_type != nullptr);
    return DynamicBufferLayoutPlan{logical_type,
                                   resolved_type,
                                   policy,
                                   layout,
                                   resolved_runtime_size(resolved_type),
                                   resolved_runtime_alignment(resolved_type),
                                   contains_real_recursive(logical_type),
                                   has_precision_lowering_recursive(logical_type, resolved_type)};
}

size_t DynamicBufferLayoutPlan::storage_bytes(size_t element_count) const noexcept {
    switch (layout_) {
        case DynamicBufferLayout::aos:
            return element_size_bytes_ * element_count;
        case DynamicBufferLayout::soa:
            return soa_storage_bytes(resolved_type_, element_count);
        default:
            return 0u;
    }
}

ByteRegion DynamicBufferLayoutPlan::record_region(size_t index) const noexcept {
    OC_ASSERT(layout_ == DynamicBufferLayout::aos);
    const auto begin = index * element_size_bytes_;
    return ByteRegion{begin, begin + element_size_bytes_};
}

ByteRegion DynamicBufferLayoutPlan::field_region(size_t index,
                                                 const TypedFieldPath &path) const noexcept {
    OC_ASSERT(layout_ == DynamicBufferLayout::aos);
    auto info = resolve_field_region_info(logical_type_, resolved_type_, path.steps(), 0u);
    OC_ASSERT(info.valid());
    const auto record = record_region(index);
    return ByteRegion{record.begin_byte + info.offset_in_record,
                      record.begin_byte + info.offset_in_record + info.size_in_bytes};
}

vector<ByteSegment> DynamicBufferLayoutPlan::record_segments(size_t element_count,
                                                             size_t index) const noexcept {
    vector<ByteSegment> segments;
    switch (layout_) {
        case DynamicBufferLayout::aos: {
            const auto region = record_region(index);
            segments.emplace_back(ByteSegment{
                .storage_begin_byte = region.begin_byte,
                .staging_begin_byte = 0u,
                .size_in_bytes = region.size()});
            break;
        }
        case DynamicBufferLayout::soa:
            collect_soa_segments(resolved_type_, element_count, index, 0u, 0u, segments);
            break;
        default:
            break;
    }
    return segments;
}

vector<ByteSegment> DynamicBufferLayoutPlan::field_segments(size_t element_count,
                                                            size_t index,
                                                            const TypedFieldPath &path) const noexcept {
    vector<ByteSegment> segments;
    switch (layout_) {
        case DynamicBufferLayout::aos: {
            const auto region = field_region(index, path);
            segments.emplace_back(ByteSegment{
                .storage_begin_byte = region.begin_byte,
                .staging_begin_byte = 0u,
                .size_in_bytes = region.size()});
            break;
        }
        case DynamicBufferLayout::soa: {
            auto info = resolve_field_storage_info(logical_type_, resolved_type_, path.steps(), 0u, element_count);
            OC_ASSERT(info.valid());
            collect_soa_field_segments(info, element_count, index, segments);
            break;
        }
        default:
            break;
    }
    return segments;
}

const Type *DynamicBufferLayoutPlan::field_logical_type(const TypedFieldPath &path) const noexcept {
    auto info = resolve_field_region_info(logical_type_, resolved_type_, path.steps(), 0u);
    OC_ASSERT(info.valid());
    return info.logical_type;
}

const Type *DynamicBufferLayoutPlan::field_resolved_type(const TypedFieldPath &path) const noexcept {
    auto info = resolve_field_region_info(logical_type_, resolved_type_, path.steps(), 0u);
    OC_ASSERT(info.valid());
    return info.resolved_type;
}

}// namespace ocarina