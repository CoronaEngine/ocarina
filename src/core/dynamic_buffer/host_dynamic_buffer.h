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

    /// The current model exports one conservative continuous byte interval.
    /// AoS updates usually map naturally to one region.
    /// SoA updates may touch multiple disjoint columns, and those writes are currently merged.

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

class DirtyByteSegments {
private:
    vector<ByteRegion> segments_{};

public:
    void clear() noexcept {
        segments_.clear();
    }

    void merge(ByteRegion region) noexcept {
        if (region.empty()) {
            return;
        }
        vector<ByteRegion> merged;
        merged.reserve(segments_.size() + 1u);
        ByteRegion pending = region;
        bool inserted = false;
        for (const auto &segment : segments_) {
            if (segment.end_byte < pending.begin_byte) {
                merged.emplace_back(segment);
                continue;
            }
            if (pending.end_byte < segment.begin_byte) {
                if (!inserted) {
                    merged.emplace_back(pending);
                    inserted = true;
                }
                merged.emplace_back(segment);
                continue;
            }
            pending.begin_byte = std::min(pending.begin_byte, segment.begin_byte);
            pending.end_byte = std::max(pending.end_byte, segment.end_byte);
        }
        if (!inserted) {
            merged.emplace_back(pending);
        }
        segments_ = std::move(merged);
    }

    [[nodiscard]] span<const ByteRegion> segments() const noexcept {
        return segments_;
    }

    [[nodiscard]] DirtyByteRange merged_range() const noexcept {
        DirtyByteRange dirty_range;
        for (const auto &segment : segments_) {
            dirty_range.merge(segment);
        }
        return dirty_range;
    }

    [[nodiscard]] bool empty() const noexcept {
        return segments_.empty();
    }
};

struct HostDynamicBufferUploadView {
    span<const std::byte> bytes{};
    size_t element_count{0u};
    DynamicBufferLayout layout{DynamicBufferLayout::aos};
    const Type *logical_type{nullptr};
    const Type *resolved_type{nullptr};
    /// Dirty export exposes precise byte segments and also keeps one merged range for compatibility.
    span<const ByteRegion> dirty_segments{};
    DirtyByteRange dirty{};
};

class HostDynamicBuffer {
private:
    DynamicBufferLayoutPlan layout_plan_{};
    HostByteBuffer storage_{};
    size_t element_count_{0u};
    size_t element_capacity_{0u};
    DirtyByteSegments dirty_segments_{};
    DirtyByteRange dirty_range_{};
    uint64_t generation_{0u};

private:
    void ensure_capacity(size_t required_capacity);
    void validate_index(size_t index) const noexcept;
    void mark_dirty(ByteRegion region) noexcept;
    void gather_segments(span<const ByteSegment> segments, HostByteBuffer &dst) const noexcept;
    void scatter_segments(const HostByteBuffer &src,
                          span<const ByteSegment> segments) noexcept;

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

    /// Record-level random access is available for both AoS and SoA.
    [[nodiscard]] bool supports_record_access() const noexcept {
        return true;
    }

    /// Field-level patching is available for both AoS and SoA.
    [[nodiscard]] bool supports_field_patch() const noexcept {
        return true;
    }

    void reserve(size_t element_capacity);
    void resize(size_t element_count);
    void clear() noexcept;

    [[nodiscard]] span<const std::byte> bytes() const noexcept { return storage_.bytes_span(); }
    [[nodiscard]] span<std::byte> bytes() noexcept { return storage_.bytes_span(); }

    [[nodiscard]] const DirtyByteRange &dirty_range() const noexcept { return dirty_range_; }
    [[nodiscard]] span<const ByteRegion> dirty_segments() const noexcept { return dirty_segments_.segments(); }
    void clear_dirty() noexcept {
        dirty_segments_.clear();
        dirty_range_.clear();
    }

    [[nodiscard]] uint64_t generation() const noexcept { return generation_; }

    /// Upload view is available for both AoS and SoA.
    [[nodiscard]] HostDynamicBufferUploadView upload_view() const noexcept {
        return HostDynamicBufferUploadView{
            .bytes = storage_.bytes_span(),
            .element_count = element_count_,
            .layout = layout_plan_.layout(),
            .logical_type = layout_plan_.logical_type(),
            .resolved_type = layout_plan_.resolved_type(),
            .dirty_segments = dirty_segments_.segments(),
            .dirty = dirty_range_};
    }

    /// Read one logical record.
    template<typename T>
    [[nodiscard]] T read(size_t index) const;

    /// Write one logical record.
    template<typename T>
    void write(size_t index, const T &value);

    /// Append records at the end of the buffer.
    template<typename T>
    void append(span<const T> values);

    /// Replace the full buffer contents in the active layout.
    /// This is the currently supported write path for SoA layout.
    template<typename T>
    void write_all(span<const T> values);

    /// Patch one logical subfield inside one record.
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

    /// This path is available for both AoS and SoA.
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
    OC_ASSERT(layout_plan_.is_compatible_with(Type::of<T>()));
    validate_index(index);
    T value{};
    HostByteBuffer encoded_record;
    const auto layout = layout_plan_.layout();
    encoded_record.resize(DynamicBufferLayoutCodec<T>::storage_bytes(1u,
                                                                     layout_plan_.policy(),
                                                                     layout));
    const auto segments = layout_plan_.record_segments(element_count_, index);
    gather_segments(segments, encoded_record);
    DynamicBufferLayoutCodec<T>::decode(encoded_record,
                                        1u,
                                        addressof(value),
                                        layout_plan_.policy(),
                                        layout);
    return value;
}

template<typename T>
void HostDynamicBuffer::write(size_t index, const T &value) {
    OC_ASSERT(layout_plan_.is_compatible_with(Type::of<T>()));
    validate_index(index);
    HostByteBuffer encoded_record;
    const auto layout = layout_plan_.layout();
    DynamicBufferLayoutCodec<T>::encode(addressof(value),
                                        1u,
                                        encoded_record,
                                        layout_plan_.policy(),
                                        layout);
    const auto segments = layout_plan_.record_segments(element_count_, index);
    scatter_segments(encoded_record, segments);
    generation_++;
}

template<typename T>
void HostDynamicBuffer::append(span<const T> values) {
    OC_ASSERT(layout_plan_.is_compatible_with(Type::of<T>()));
    if (values.empty()) {
        return;
    }
    if (layout_plan_.layout() == DynamicBufferLayout::soa) {
        const auto old_count = element_count_;
        const auto new_count = old_count + values.size();
        ensure_capacity(new_count);
        vector<T> merged(new_count);
        if (old_count != 0u) {
            DynamicBufferLayoutCodec<T>::decode(storage_,
                                                old_count,
                                                merged.data(),
                                                layout_plan_.policy(),
                                                DynamicBufferLayout::soa);
        }
        std::copy(values.begin(), values.end(), merged.begin() + static_cast<ptrdiff_t>(old_count));
        element_count_ = new_count;
        DynamicBufferLayoutCodec<T>::encode(merged.data(),
                                            merged.size(),
                                            storage_,
                                            layout_plan_.policy(),
                                            DynamicBufferLayout::soa);
        if (!storage_.empty()) {
            mark_dirty(ByteRegion{0u, storage_.size()});
        }
        generation_++;
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
    ensure_capacity(element_count_);
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
    validate_index(index);
    OC_ASSERT(layout_plan_.field_logical_type(path) == Type::of<TField>());
    HostByteBuffer encoded_field;
    const auto layout = layout_plan_.layout();
    DynamicBufferLayoutCodec<TField>::encode(addressof(value),
                                             1u,
                                             encoded_field,
                                             layout_plan_.policy(),
                                             layout);
    const auto segments = layout_plan_.field_segments(element_count_, index, path);
    scatter_segments(encoded_field, segments);
    generation_++;
}

}// namespace ocarina