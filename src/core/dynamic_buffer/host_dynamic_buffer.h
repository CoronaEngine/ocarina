//
// Created by GitHub Copilot on 2026/4/13.
//

#pragma once

#include <algorithm>

#include "core/dynamic_buffer/dynamic_buffer_layout_plan.h"
#include "core/dynamic_buffer/host_byte_buffer.h"

namespace ocarina {

struct DirtyByteRange {
    bool dirty{false};
    size_t begin_byte{0u};
    size_t end_byte{0u};

    void clear() noexcept {
        dirty = false;
        begin_byte = 0u;
        end_byte = 0u;
    }

    void merge(ByteRegion region) noexcept {
        if (region.empty()) {
            return;
        }
        if (!dirty) {
            dirty = true;
            begin_byte = region.begin_byte;
            end_byte = region.end_byte;
            return;
        }
        begin_byte = std::min(begin_byte, region.begin_byte);
        end_byte = std::max(end_byte, region.end_byte);
    }

    [[nodiscard]] ByteRegion region() const noexcept {
        if (!dirty) {
            return {};
        }
        return ByteRegion{begin_byte, end_byte};
    }

    [[nodiscard]] bool empty() const noexcept {
        return !dirty || begin_byte >= end_byte;
    }
};

struct HostDynamicBufferUploadView {
    span<const std::byte> bytes{};
    size_t element_count{0u};
    DynamicBufferLayout layout{DynamicBufferLayout::aos};
    const Type *logical_type{nullptr};
    const Type *resolved_type{nullptr};
    DirtyByteRange dirty{};
};

class HostDynamicBuffer {
private:
    DynamicBufferLayoutPlan layout_plan_{};
    HostByteBuffer storage_{};
    size_t element_count_{0u};
    size_t element_capacity_{0u};
    DirtyByteRange dirty_range_{};
    uint64_t generation_{0u};

private:
    void ensure_capacity(size_t required_capacity);
    void validate_index(size_t index) const noexcept;
    void mark_dirty(ByteRegion region) noexcept;

public:
    HostDynamicBuffer() = default;
    explicit HostDynamicBuffer(DynamicBufferLayoutPlan layout_plan,
                               size_t initial_count = 0u);

    [[nodiscard]] static HostDynamicBuffer create(const Type *logical_type,
                                                  StoragePrecisionPolicy policy,
                                                  DynamicBufferLayout layout,
                                                  size_t initial_count = 0u);

    [[nodiscard]] const DynamicBufferLayoutPlan &layout_plan() const noexcept { return layout_plan_; }
    [[nodiscard]] size_t element_count() const noexcept { return element_count_; }
    [[nodiscard]] size_t element_capacity() const noexcept { return element_capacity_; }
    [[nodiscard]] size_t storage_size_bytes() const noexcept { return storage_.size(); }
    [[nodiscard]] bool empty() const noexcept { return element_count_ == 0u; }

    void reserve(size_t element_capacity);
    void resize(size_t element_count);
    void clear() noexcept;

    [[nodiscard]] span<const std::byte> bytes() const noexcept { return storage_.bytes_span(); }
    [[nodiscard]] span<std::byte> bytes() noexcept { return storage_.bytes_span(); }

    [[nodiscard]] const DirtyByteRange &dirty_range() const noexcept { return dirty_range_; }
    void clear_dirty() noexcept { dirty_range_.clear(); }

    [[nodiscard]] uint64_t generation() const noexcept { return generation_; }

    [[nodiscard]] HostDynamicBufferUploadView upload_view() const noexcept {
        return HostDynamicBufferUploadView{
            .bytes = storage_.bytes_span(),
            .element_count = element_count_,
            .layout = layout_plan_.layout(),
            .logical_type = layout_plan_.logical_type(),
            .resolved_type = layout_plan_.resolved_type(),
            .dirty = dirty_range_};
    }

    template<typename T>
    [[nodiscard]] T read(size_t index) const;

    template<typename T>
    void write(size_t index, const T &value);

    template<typename T>
    void append(span<const T> values);

    template<typename T>
    void write_all(span<const T> values);

    template<typename TField>
    void patch(size_t index, const TypedFieldPath &path, const TField &value);
};

template<typename T>
class TypedHostDynamicBufferView {
private:
    HostDynamicBuffer *buffer_{nullptr};

public:
    explicit TypedHostDynamicBufferView(HostDynamicBuffer &buffer) noexcept
        : buffer_(addressof(buffer)) {
        OC_ASSERT(buffer_->layout_plan().is_compatible_with(Type::of<T>()));
    }

    [[nodiscard]] const DynamicBufferLayoutPlan &layout_plan() const noexcept {
        return buffer_->layout_plan();
    }

    [[nodiscard]] size_t element_count() const noexcept { return buffer_->element_count(); }
    [[nodiscard]] bool empty() const noexcept { return buffer_->empty(); }

    [[nodiscard]] T read(size_t index) const {
        return buffer_->template read<T>(index);
    }

    void write(size_t index, const T &value) {
        buffer_->template write<T>(index, value);
    }

    void append(span<const T> values) {
        buffer_->template append<T>(values);
    }

    void write_all(span<const T> values) {
        buffer_->template write_all<T>(values);
    }

    template<typename TField>
    void patch(size_t index, const TypedFieldPath &path, const TField &value) {
        buffer_->template patch<TField>(index, path, value);
    }
};

template<typename T>
T HostDynamicBuffer::read(size_t index) const {
    OC_ASSERT(layout_plan_.layout() == DynamicBufferLayout::aos);
    OC_ASSERT(layout_plan_.is_compatible_with(Type::of<T>()));
    validate_index(index);
    T value{};
    HostByteBuffer encoded_record;
    auto region = layout_plan_.record_region(index);
    encoded_record.resize(region.size());
    encoded_record.copy_from(storage_.data() + region.begin_byte, region.size());
    DynamicBufferLayoutCodec<T>::decode(encoded_record,
                                        1u,
                                        addressof(value),
                                        layout_plan_.policy(),
                                        DynamicBufferLayout::aos);
    return value;
}

template<typename T>
void HostDynamicBuffer::write(size_t index, const T &value) {
    OC_ASSERT(layout_plan_.layout() == DynamicBufferLayout::aos);
    OC_ASSERT(layout_plan_.is_compatible_with(Type::of<T>()));
    validate_index(index);
    HostByteBuffer encoded_record;
    DynamicBufferLayoutCodec<T>::encode(addressof(value),
                                        1u,
                                        encoded_record,
                                        layout_plan_.policy(),
                                        DynamicBufferLayout::aos);
    auto region = layout_plan_.record_region(index);
    OC_ASSERT(encoded_record.size() == region.size());
    storage_.copy_from(encoded_record.data(), encoded_record.size(), region.begin_byte);
    mark_dirty(region);
    generation_++;
}

template<typename T>
void HostDynamicBuffer::append(span<const T> values) {
    OC_ASSERT(layout_plan_.layout() == DynamicBufferLayout::aos);
    OC_ASSERT(layout_plan_.is_compatible_with(Type::of<T>()));
    if (values.empty()) {
        return;
    }
    const auto old_count = element_count_;
    const auto new_count = old_count + values.size();
    ensure_capacity(new_count);
    element_count_ = new_count;
    storage_.resize(layout_plan_.storage_bytes(element_count_));
    HostByteBuffer encoded;
    DynamicBufferLayoutCodec<T>::encode(values.data(),
                                        values.size(),
                                        encoded,
                                        layout_plan_.policy(),
                                        DynamicBufferLayout::aos);
    auto begin = layout_plan_.record_region(old_count).begin_byte;
    storage_.copy_from(encoded.data(), encoded.size(), begin);
    mark_dirty(ByteRegion{begin, begin + encoded.size()});
    generation_++;
}

template<typename T>
void HostDynamicBuffer::write_all(span<const T> values) {
    OC_ASSERT(layout_plan_.is_compatible_with(Type::of<T>()));
    element_count_ = values.size();
    element_capacity_ = std::max(element_capacity_, element_count_);
    DynamicBufferLayoutCodec<T>::encode(values.data(),
                                        values.size(),
                                        storage_,
                                        layout_plan_.policy(),
                                        layout_plan_.layout());
    if (!values.empty()) {
        mark_dirty(ByteRegion{0u, storage_.size()});
        generation_++;
    }
}

template<typename TField>
void HostDynamicBuffer::patch(size_t index, const TypedFieldPath &path, const TField &value) {
    OC_ASSERT(layout_plan_.layout() == DynamicBufferLayout::aos);
    validate_index(index);
    OC_ASSERT(layout_plan_.field_logical_type(path) == Type::of<TField>());
    HostByteBuffer encoded_field;
    DynamicBufferLayoutCodec<TField>::encode(addressof(value),
                                             1u,
                                             encoded_field,
                                             layout_plan_.policy(),
                                             DynamicBufferLayout::aos);
    auto region = layout_plan_.field_region(index, path);
    OC_ASSERT(encoded_field.size() == region.size());
    storage_.copy_from(encoded_field.data(), encoded_field.size(), region.begin_byte);
    mark_dirty(region);
    generation_++;
}

}// namespace ocarina