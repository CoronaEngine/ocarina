//
// Created by GitHub Copilot on 2026/4/13.
//

#include "host_dynamic_buffer.h"

namespace ocarina {

namespace {

template<typename Func>
void for_each_non_empty_segment(span<const ByteSegment> segments, Func &&func) noexcept {
    for (const auto &segment : segments) {
        if (segment.empty()) {
            continue;
        }
        func(segment);
    }
}

}// namespace

RawHostDynamicBuffer::RawHostDynamicBuffer(DynamicBufferLayoutPlan layout_plan,
                                                                                     size_t initial_count)
    : layout_plan_(std::move(layout_plan)),
      element_count_(initial_count),
      element_capacity_(initial_count) {
    storage_.resize(layout_plan_.storage_bytes(initial_count));
}

RawHostDynamicBuffer RawHostDynamicBuffer::create(const Type *logical_type,
                                                                                                    StoragePrecisionPolicy policy,
                                                                                                    size_t initial_count) {
        return RawHostDynamicBuffer{DynamicBufferLayoutPlan::create(logical_type, policy), initial_count};
}

void RawHostDynamicBuffer::ensure_capacity(size_t required_capacity) {
    if (required_capacity <= element_capacity_) {
        return;
    }
    storage_.reserve(layout_plan_.storage_bytes(required_capacity));
    element_capacity_ = required_capacity;
}

void RawHostDynamicBuffer::validate_index(size_t index) const noexcept {
    OC_ASSERT(index < element_count_);
}

void RawHostDynamicBuffer::mark_dirty(ByteRegion region) noexcept {
    dirty_segments_.merge(region);
    dirty_range_.merge(region);
}

void RawHostDynamicBuffer::gather_segments(span<const ByteSegment> segments,
                                           HostByteBuffer &dst) const noexcept {
    for_each_non_empty_segment(segments, [&](const ByteSegment &segment) {
        dst.copy_from(storage_.data() + segment.storage_begin_byte,
                      segment.size_in_bytes,
                      segment.staging_begin_byte);
    });
}

void RawHostDynamicBuffer::scatter_segments(const HostByteBuffer &src,
                                            span<const ByteSegment> segments) noexcept {
    for_each_non_empty_segment(segments, [&](const ByteSegment &segment) {
        storage_.copy_from(src.data() + segment.staging_begin_byte,
                           segment.size_in_bytes,
                           segment.storage_begin_byte);
        mark_dirty(ByteRegion{segment.storage_begin_byte,
                              segment.storage_begin_byte + segment.size_in_bytes});
    });
}

void RawHostDynamicBuffer::reserve(size_t element_capacity) {
    ensure_capacity(element_capacity);
}

void RawHostDynamicBuffer::resize(size_t element_count) {
    ensure_capacity(element_count);
    storage_.resize(layout_plan_.storage_bytes(element_count));
    element_count_ = element_count;
    if (!storage_.empty()) {
        mark_dirty(ByteRegion{0u, storage_.size()});
    }
    generation_++;
}

void RawHostDynamicBuffer::clear() noexcept {
    storage_.clear();
    element_count_ = 0u;
    dirty_segments_.clear();
    dirty_range_.clear();
    generation_++;
}

}// namespace ocarina