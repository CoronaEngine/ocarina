//
// Created by Z on 2026/4/13.
//

#pragma once

#include "core/dynamic_buffer/host_dynamic_buffer.h"
#include "rhi/resources/dynamic_buffer.h"

namespace ocarina {

struct DynamicBufferGpuState {
    DynamicBuffer device_buffer{};
    size_t byte_capacity{0u};
    uint64_t uploaded_generation{0u};
    const Type *logical_type{nullptr};
    const Type *resolved_type{nullptr};
    size_t last_upload_segment_count{0u};
    size_t last_uploaded_bytes{0u};
    bool last_upload_was_full{false};
};

class DynamicBufferGpuBridge {
private:
    DynamicBufferGpuState state_{};
    string device_buffer_name_{};

private:
    void release_device_buffer() noexcept {
        if (state_.device_buffer.valid()) {
            state_.device_buffer.destroy();
        }
        state_.device_buffer = {};
        state_.byte_capacity = 0u;
    }

    void reset_upload_stats() noexcept {
        state_.last_upload_segment_count = 0u;
        state_.last_uploaded_bytes = 0u;
        state_.last_upload_was_full = false;
    }

    void record_upload_region(ByteRegion region) noexcept {
        if (region.empty()) {
            return;
        }
        state_.last_upload_segment_count++;
        state_.last_uploaded_bytes += region.size();
    }

    [[nodiscard]] bool needs_recreate(const HostDynamicBufferUploadView &view) const noexcept {
        return state_.device_buffer.needs_recreate(view);
    }

public:
    explicit DynamicBufferGpuBridge(string device_buffer_name = "dynamic-buffer-gpu-bridge") noexcept
        : device_buffer_name_(std::move(device_buffer_name)) {}

    [[nodiscard]] const DynamicBufferGpuState &state() const noexcept { return state_; }
    [[nodiscard]] const DynamicBuffer &device_buffer() const noexcept { return state_.device_buffer; }

    [[nodiscard]] bool needs_upload(const RawHostDynamicBuffer &buffer) const noexcept {
        const auto view = buffer.upload_view();
        return buffer.generation() != state_.uploaded_generation || needs_recreate(view);
    }

    template<typename T>
    [[nodiscard]] bool needs_upload(const HostDynamicBuffer<T> &buffer) const noexcept {
        return needs_upload(buffer.raw());
    }

    void prepare_device_buffer(const Device &device,
                               const HostDynamicBufferUploadView &view) noexcept {
        if (view.bytes.empty()) {
            release_device_buffer();
            if (view.logical_type != nullptr) {
                state_.device_buffer = device.create_dynamic_buffer(view.logical_type,
                                                                    view.policy,
                                                                    0u,
                                                                    device_buffer_name_);
            }
            state_.logical_type = view.logical_type;
            state_.resolved_type = view.resolved_type;
            return;
        }
        if (needs_recreate(view)) {
            release_device_buffer();
            state_.device_buffer = device.create_dynamic_buffer(view.logical_type,
                                                                view.policy,
                                                                0u,
                                                                device_buffer_name_);
            state_.device_buffer.reserve(view.element_count);
            state_.byte_capacity = state_.device_buffer.capacity_in_byte();
        }
        state_.logical_type = view.logical_type;
        state_.resolved_type = view.resolved_type;
    }

    void upload_immediately(const Device &device,
                            RawHostDynamicBuffer &buffer) noexcept {
        const auto view = buffer.upload_view();
        const bool full_upload_required = needs_recreate(view) || view.dirty_segments.empty();
        prepare_device_buffer(device, view);
        reset_upload_stats();
        if (!view.bytes.empty()) {
            auto upload_stats = state_.device_buffer.sync_immediately(view, full_upload_required);
            state_.last_upload_segment_count = upload_stats.uploaded_segment_count;
            state_.last_uploaded_bytes = upload_stats.uploaded_bytes;
            state_.last_upload_was_full = upload_stats.full_upload;
            state_.byte_capacity = state_.device_buffer.capacity_in_byte();
        }
        state_.uploaded_generation = buffer.generation();
        buffer.clear_dirty();
    }

    template<typename T>
    void upload_immediately(const Device &device,
                            HostDynamicBuffer<T> &buffer) noexcept {
        upload_immediately(device, buffer.raw());
    }
};

}// namespace ocarina