//
// Created by Z on 2026/4/13.
//

#pragma once

#include "core/dynamic_buffer/host_dynamic_buffer.h"
#include "rhi/resources/byte_buffer.h"

namespace ocarina {

struct DynamicBufferGpuState {
    ByteBuffer device_buffer{};
    size_t byte_capacity{0u};
    uint64_t uploaded_generation{0u};
    DynamicBufferLayout layout{DynamicBufferLayout::AOS};
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
        return !state_.device_buffer.valid() ||
               view.bytes.size() > state_.byte_capacity ||
               state_.layout != view.layout ||
               state_.logical_type != view.logical_type ||
               state_.resolved_type != view.resolved_type;
    }

    void upload_full_immediately(const HostDynamicBufferUploadView &view) noexcept {
        if (view.bytes.empty()) {
            return;
        }
        state_.device_buffer.upload_immediately(view.bytes.data());
        state_.last_upload_was_full = true;
        record_upload_region(ByteRegion{0u, view.bytes.size()});
    }

    void upload_dirty_segments_immediately(const HostDynamicBufferUploadView &view) noexcept {
        for (const auto &segment : view.dirty_segments) {
            if (segment.empty()) {
                continue;
            }
            auto upload = state_.device_buffer.view(segment.begin_byte, segment.size())
                              .upload_sync(view.bytes.data() + segment.begin_byte);
            upload->accept(*state_.device_buffer.device()->command_visitor());
            record_upload_region(segment);
        }
    }

public:
    explicit DynamicBufferGpuBridge(string device_buffer_name = "dynamic-buffer-gpu-bridge") noexcept
        : device_buffer_name_(std::move(device_buffer_name)) {}

    [[nodiscard]] const DynamicBufferGpuState &state() const noexcept { return state_; }
    [[nodiscard]] const ByteBuffer &device_buffer() const noexcept { return state_.device_buffer; }

    [[nodiscard]] bool needs_upload(const HostDynamicBuffer &buffer) const noexcept {
        const auto view = buffer.upload_view();
        return buffer.generation() != state_.uploaded_generation || needs_recreate(view);
    }

    void prepare_device_buffer(const Device &device,
                               const HostDynamicBufferUploadView &view) noexcept {
        if (view.bytes.empty()) {
            release_device_buffer();
            state_.layout = view.layout;
            state_.logical_type = view.logical_type;
            state_.resolved_type = view.resolved_type;
            return;
        }
        if (needs_recreate(view)) {
            release_device_buffer();
            state_.device_buffer = device.create_byte_buffer(view.bytes.size(), device_buffer_name_);
            state_.byte_capacity = view.bytes.size();
        }
        state_.layout = view.layout;
        state_.logical_type = view.logical_type;
        state_.resolved_type = view.resolved_type;
    }

    void upload_immediately(const Device &device,
                            HostDynamicBuffer &buffer) noexcept {
        const auto view = buffer.upload_view();
        const bool full_upload_required = needs_recreate(view) || view.dirty_segments.empty();
        prepare_device_buffer(device, view);
        reset_upload_stats();
        if (!view.bytes.empty()) {
            if (full_upload_required) {
                upload_full_immediately(view);
            } else {
                upload_dirty_segments_immediately(view);
            }
        }
        state_.uploaded_generation = buffer.generation();
        buffer.clear_dirty();
    }
};

}// namespace ocarina