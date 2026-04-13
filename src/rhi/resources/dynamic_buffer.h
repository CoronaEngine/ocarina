//
// Created by Z on 2026/4/13.
//

#pragma once

#include "core/dynamic_buffer/host_dynamic_buffer.h"
#include "rhi/command.h"
#include "rhi/resources/byte_buffer.h"

namespace ocarina {

struct DynamicBufferUploadStats {
    size_t uploaded_segment_count{0u};
    size_t uploaded_bytes{0u};
    bool full_upload{false};
};

class DynamicBuffer;

class DynamicBufferView {
private:
    ByteBufferView byte_view_{};
    size_t element_offset_{0u};
    size_t element_count_{0u};
    size_t total_element_count_{0u};
    size_t element_stride_{0u};
    size_t element_alignment_{0u};
    StoragePrecisionPolicy policy_{};
    const Type *logical_type_{nullptr};
    const Type *resolved_type_{nullptr};

public:
    DynamicBufferView() = default;
    DynamicBufferView(ByteBufferView byte_view,
                      size_t element_offset,
                      size_t element_count,
                      size_t total_element_count,
                      size_t element_stride,
                      size_t element_alignment,
                      const Type *logical_type,
                      const Type *resolved_type,
                      StoragePrecisionPolicy policy) noexcept
        : byte_view_(byte_view),
          element_offset_(element_offset),
          element_count_(element_count),
          total_element_count_(total_element_count),
          element_stride_(element_stride),
          element_alignment_(element_alignment),
          policy_(policy),
          logical_type_(logical_type),
          resolved_type_(resolved_type) {}

    explicit DynamicBufferView(const DynamicBuffer &buffer) noexcept;

    [[nodiscard]] handle_ty handle() const noexcept { return byte_view_.handle(); }
    [[nodiscard]] size_t size() const noexcept { return element_count_; }
    [[nodiscard]] size_t size_in_byte() const noexcept { return byte_view_.size(); }
    [[nodiscard]] size_t offset() const noexcept { return element_offset_; }
    [[nodiscard]] size_t offset_in_byte() const noexcept { return byte_view_.offset(); }
    [[nodiscard]] size_t total_size() const noexcept { return total_element_count_; }
    [[nodiscard]] size_t total_size_in_byte() const noexcept { return byte_view_.total_size(); }
    [[nodiscard]] size_t element_stride() const noexcept { return element_stride_; }
    [[nodiscard]] size_t element_alignment() const noexcept { return element_alignment_; }
    [[nodiscard]] StoragePrecisionPolicy policy() const noexcept { return policy_; }
    [[nodiscard]] const Type *logical_type() const noexcept { return logical_type_; }
    [[nodiscard]] const Type *resolved_type() const noexcept { return resolved_type_; }
    [[nodiscard]] bool empty() const noexcept { return element_count_ == 0u; }

    [[nodiscard]] ByteBufferView byte_view() const noexcept {
        return ByteBufferView(handle(), offset_in_byte(), size_in_byte(), total_size_in_byte());
    }

    template<typename T>
    [[nodiscard]] BufferView<T> aos_view() const noexcept {
        OC_ASSERT(resolved_type_ == Type::of<T>());
        OC_ASSERT(element_stride_ == sizeof(T));
        return byte_view().template view_as<T>();
    }

    [[nodiscard]] DynamicBufferView subview(size_t offset, size_t size = 0u) const noexcept {
        size = size == 0u ? element_count_ - offset : size;
        return DynamicBufferView{ByteBufferView(handle(),
                                                offset_in_byte() + offset * element_stride_,
                                                size * element_stride_,
                                                total_size_in_byte()),
                                 element_offset_ + offset,
                                 size,
                                 total_element_count_,
                                 element_stride_,
                                 element_alignment_,
                                 logical_type_,
                                 resolved_type_,
                                 policy_};
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg> && std::is_same_v<buffer_element_t<Arg>, std::byte>
    [[nodiscard]] BufferCopyCommand *copy_from(const Arg &src, uint dst_offset = 0u) const noexcept {
        return BufferCopyCommand::create(src.handle(), handle(),
                                         src.offset_in_byte(),
                                         offset_in_byte() + dst_offset,
                                         src.size_in_byte(), true);
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg> && std::is_same_v<buffer_element_t<Arg>, std::byte>
    [[nodiscard]] BufferCopyCommand *copy_to(Arg &dst, uint src_offset = 0u) const noexcept {
        return BufferCopyCommand::create(handle(), dst.handle(),
                                         offset_in_byte() + src_offset,
                                         dst.offset_in_byte(),
                                         dst.size_in_byte(), true);
    }

    [[nodiscard]] BufferUploadCommand *upload(const void *data, bool async = true) const noexcept {
        return BufferUploadCommand::create(data, handle(), offset_in_byte(), size_in_byte(), async);
    }

    [[nodiscard]] BufferUploadCommand *upload_sync(const void *data) const noexcept {
        return upload(data, false);
    }

    [[nodiscard]] BufferDownloadCommand *download(void *data, bool async = true) const noexcept {
        return BufferDownloadCommand::create(data, handle(), offset_in_byte(), size_in_byte(), async);
    }

    [[nodiscard]] BufferDownloadCommand *download_sync(void *data) const noexcept {
        return download(data, false);
    }

    [[nodiscard]] BufferByteSetCommand *byte_set(uchar value, bool async = true) const noexcept {
        return BufferByteSetCommand::create(handle(), size_in_byte(), value, async);
    }

    [[nodiscard]] BufferByteSetCommand *reset(bool async = true) const noexcept {
        return byte_set(0u, async);
    }
};

class DynamicBuffer {
private:
    Device::Impl *device_{nullptr};
    ByteBuffer byte_buffer_{};
    const Type *logical_type_{nullptr};
    const Type *resolved_type_{nullptr};
    StoragePrecisionPolicy policy_{};
    size_t element_stride_{0u};
    size_t element_alignment_{0u};
    size_t element_count_{0u};
    size_t element_capacity_{0u};
    string name_{};

private:
    void assign_layout_plan(const DynamicBufferLayoutPlan &layout_plan) noexcept {
        logical_type_ = layout_plan.logical_type();
        resolved_type_ = layout_plan.resolved_type();
        policy_ = layout_plan.policy();
        element_stride_ = layout_plan.element_size_bytes();
        element_alignment_ = layout_plan.element_alignment();
    }

    void assign_resolved_layout(const Type *resolved_type) noexcept {
        logical_type_ = nullptr;
        resolved_type_ = resolved_type;
        policy_ = StoragePrecisionPolicy{.policy = PrecisionPolicy::force_f32, .allow_real_in_storage = true};
        element_stride_ = resolved_type->size();
        element_alignment_ = resolved_type->alignment();
    }

    void validate_byte_size(size_t byte_size) const noexcept {
        OC_ASSERT(element_stride_ > 0u || byte_size == 0u);
        OC_ASSERT(element_stride_ == 0u || byte_size % element_stride_ == 0u);
    }

    void reserve_for_count(size_t required_count) noexcept {
        if (required_count <= element_capacity_) {
            return;
        }
        auto *device = this->device();
        OC_ASSERT(device != nullptr);
        ByteBuffer new_buffer{device, required_count * element_stride_, name_};
        if (valid() && size_in_byte() != 0u) {
            auto *copy = BufferCopyCommand::create(handle(), new_buffer.handle(), 0u, 0u, size_in_byte(), false);
            copy->accept(*device->command_visitor());
        }
        if (byte_buffer_.valid()) {
            byte_buffer_.destroy();
        }
        byte_buffer_ = std::move(new_buffer);
        element_capacity_ = required_count;
    }

public:
    DynamicBuffer() = default;

    DynamicBuffer(Device::Impl *device,
                  const Type *logical_type,
                  StoragePrecisionPolicy policy,
                  size_t element_count,
                  const string &name = "")
        : DynamicBuffer(device, DynamicBufferLayoutPlan::create(logical_type, policy), element_count, name) {}

    DynamicBuffer(Device::Impl *device,
                  const Type *resolved_type,
                  size_t element_count,
                  const string &name = "")
                : device_(device),
                    byte_buffer_(element_count == 0u ? ByteBuffer{} : ByteBuffer(device, resolved_type->size() * element_count, name)),
          element_count_(element_count),
          element_capacity_(element_count),
                    name_(name) {
                assign_resolved_layout(resolved_type);
        }

    DynamicBuffer(Device::Impl *device,
                  const DynamicBufferLayoutPlan &layout_plan,
                  size_t element_count,
                  const string &name = "")
                : device_(device),
                    byte_buffer_(element_count == 0u ? ByteBuffer{} : ByteBuffer(device, layout_plan.storage_bytes(element_count), name)),
          element_count_(element_count),
          element_capacity_(element_count),
                    name_(name) {
                assign_layout_plan(layout_plan);
        }

    [[nodiscard]] bool valid() const noexcept { return byte_buffer_.valid(); }
    [[nodiscard]] handle_ty handle() const noexcept { return byte_buffer_.handle(); }
    [[nodiscard]] Device::Impl *device() const noexcept {
        return byte_buffer_.valid() ? byte_buffer_.device() : device_;
    }
    [[nodiscard]] const string &name() const noexcept { return name_; }
    void set_name(string name) noexcept { name_ = std::move(name); }
    [[nodiscard]] const Type *logical_type() const noexcept { return logical_type_; }
    [[nodiscard]] const Type *resolved_type() const noexcept { return resolved_type_; }
    [[nodiscard]] StoragePrecisionPolicy policy() const noexcept { return policy_; }
    [[nodiscard]] size_t element_stride() const noexcept { return element_stride_; }
    [[nodiscard]] size_t element_alignment() const noexcept { return element_alignment_; }
    [[nodiscard]] size_t size() const noexcept { return element_count_; }
    [[nodiscard]] size_t capacity() const noexcept { return element_capacity_; }
    [[nodiscard]] bool empty() const noexcept { return element_count_ == 0u; }
    [[nodiscard]] size_t size_in_byte() const noexcept { return element_count_ * element_stride_; }
    [[nodiscard]] size_t capacity_in_byte() const noexcept { return element_capacity_ * element_stride_; }
    [[nodiscard]] uint offset_in_byte() const noexcept { return 0u; }

    void destroy() noexcept {
        byte_buffer_.destroy();
        element_count_ = 0u;
        element_capacity_ = 0u;
    }

    void reset_layout(const Type *logical_type,
                      StoragePrecisionPolicy policy) noexcept {
        const auto layout_plan = DynamicBufferLayoutPlan::create(logical_type, policy);
        destroy();
        assign_layout_plan(layout_plan);
    }

    void reserve(size_t element_capacity) noexcept {
        reserve_for_count(element_capacity);
    }

    [[nodiscard]] ByteBufferView byte_view(size_t offset_in_byte = 0u, size_t size_in_byte = 0u) const noexcept {
        size_in_byte = size_in_byte == 0u ? this->size_in_byte() - offset_in_byte : size_in_byte;
        return ByteBufferView(handle(), offset_in_byte, size_in_byte, this->size_in_byte());
    }

    [[nodiscard]] DynamicBufferView view(size_t offset = 0u, size_t size = 0u) const noexcept {
        size = size == 0u ? element_count_ - offset : size;
        return DynamicBufferView{byte_view(offset * element_stride_, size * element_stride_),
                                 offset,
                                 size,
                                 element_count_,
                                 element_stride_,
                                 element_alignment_,
                                 logical_type_,
                                 resolved_type_,
                                 policy_};
    }

    [[nodiscard]] DynamicBufferView subview(size_t offset, size_t size = 0u) const noexcept {
        return view().subview(offset, size);
    }

    template<typename T>
    [[nodiscard]] BufferView<T> aos_view() const noexcept {
        return view().template aos_view<T>();
    }

    [[nodiscard]] const Expression *expression() const noexcept {
        return byte_buffer_.expression();
    }

    [[nodiscard]] Expr<ByteBuffer> expr() const noexcept {
        return byte_buffer_.expr();
    }

    template<typename Target, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] Var<Target> load_as(Offset &&offset) const noexcept {
        return byte_buffer_.template load_as<Target>(OC_FORWARD(offset));
    }

    template<typename Elm, typename Offset>
    requires is_integral_expr_v<Offset>
    void store(Offset &&offset, const Elm &val) noexcept {
        byte_buffer_.store(OC_FORWARD(offset), val);
    }

    template<typename Elm, typename int_type = uint>
    [[nodiscard]] auto aos_view_var() const noexcept {
        return byte_buffer_.template aos_view_var<Elm>(static_cast<int_type>(size()));
    }

    template<typename Elm, typename int_type = uint>
    [[nodiscard]] auto aos_view_var(const Var<int_type> &view_size) const noexcept {
        return byte_buffer_.template aos_view_var<Elm>(view_size);
    }

    [[nodiscard]] CommandBatch upload(const void *data, size_t byte_size, bool async = true) noexcept {
        validate_byte_size(byte_size);
        const auto required_count = element_stride_ == 0u ? 0u : byte_size / element_stride_;
        reserve_for_count(required_count);
        element_count_ = required_count;
        if (byte_size == 0u) {
            return {};
        }
        return {BufferUploadCommand::create(data, handle(), 0u, byte_size, async)};
    }

    [[nodiscard]] CommandBatch upload(span<const std::byte> bytes, bool async = true) noexcept {
        return upload(bytes.data(), bytes.size(), async);
    }

    void upload_immediately(const void *data, size_t byte_size) noexcept {
        auto commands = upload(data, byte_size, false);
        commands.accept(*device()->command_visitor());
    }

    void upload_immediately(span<const std::byte> bytes) noexcept {
        upload_immediately(bytes.data(), bytes.size());
    }

    [[nodiscard]] bool needs_recreate(const HostDynamicBufferUploadView &view) const noexcept {
        return !valid() ||
               view.bytes.size() > capacity_in_byte() ||
               logical_type() != view.logical_type ||
               resolved_type() != view.resolved_type ||
               policy().policy != view.policy.policy ||
               policy().allow_real_in_storage != view.policy.allow_real_in_storage;
    }

    [[nodiscard]] DynamicBufferUploadStats sync_immediately(const HostDynamicBufferUploadView &view,
                                                            bool force_full_upload = false) noexcept {
        DynamicBufferUploadStats stats;
        if (view.logical_type != nullptr && (logical_type() != view.logical_type ||
                                             resolved_type() != view.resolved_type ||
                                             policy().policy != view.policy.policy ||
                                             policy().allow_real_in_storage != view.policy.allow_real_in_storage)) {
            reset_layout(view.logical_type, view.policy);
        }
        if (view.bytes.empty()) {
            destroy();
            return stats;
        }
        const bool full_upload_required = force_full_upload || needs_recreate(view) || view.dirty_segments.empty();
        reserve(view.element_count);
        if (full_upload_required) {
            upload_immediately(view.bytes);
            stats.full_upload = true;
            stats.uploaded_segment_count = view.bytes.empty() ? 0u : 1u;
            stats.uploaded_bytes = view.bytes.size();
            return stats;
        }
        element_count_ = view.element_count;
        for (const auto &segment : view.dirty_segments) {
            if (segment.empty()) {
                continue;
            }
            auto upload_cmd = byte_view(segment.begin_byte, segment.size())
                                  .upload_sync(view.bytes.data() + segment.begin_byte);
            upload_cmd->accept(*device()->command_visitor());
            stats.uploaded_segment_count++;
            stats.uploaded_bytes += segment.size();
        }
        return stats;
    }

    [[nodiscard]] DynamicBufferUploadStats sync_immediately(RawHostDynamicBuffer &buffer,
                                                            bool force_full_upload = false) noexcept {
        const auto view = buffer.upload_view();
        auto stats = sync_immediately(view, force_full_upload);
        buffer.clear_dirty();
        return stats;
    }

    template<typename T>
    [[nodiscard]] DynamicBufferUploadStats sync_immediately(HostDynamicBuffer<T> &buffer,
                                                            bool force_full_upload = false) noexcept {
        return sync_immediately(buffer.raw(), force_full_upload);
    }

    [[nodiscard]] BufferDownloadCommand *download(void *data, bool async = true) const noexcept {
        return BufferDownloadCommand::create(data, handle(), 0u, size_in_byte(), async);
    }

    void download_immediately(void *data) const noexcept {
        download(data, false)->accept(*device()->command_visitor());
    }

    [[nodiscard]] BufferByteSetCommand *byte_set(uchar value, bool async = true) const noexcept {
        return BufferByteSetCommand::create(handle(), size_in_byte(), value, async);
    }

    [[nodiscard]] BufferByteSetCommand *reset(bool async = true) const noexcept {
        return byte_set(0u, async);
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg> && std::is_same_v<buffer_element_t<Arg>, std::byte>
    [[nodiscard]] BufferCopyCommand *copy_from(const Arg &src, uint dst_offset = 0u) const noexcept {
        return BufferCopyCommand::create(src.handle(), handle(),
                                         src.offset_in_byte(),
                                         dst_offset,
                                         src.size_in_byte(), true);
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg> && std::is_same_v<buffer_element_t<Arg>, std::byte>
    [[nodiscard]] BufferCopyCommand *copy_to(Arg &dst, uint src_offset = 0u) const noexcept {
        return BufferCopyCommand::create(handle(), dst.handle(),
                                         src_offset,
                                         dst.offset_in_byte(),
                                         size_in_byte(), true);
    }
};

inline DynamicBufferView::DynamicBufferView(const DynamicBuffer &buffer) noexcept
    : DynamicBufferView(buffer.byte_view(),
                        0u,
                        buffer.size(),
                        buffer.size(),
                        buffer.element_stride(),
                        buffer.element_alignment(),
                        buffer.logical_type(),
                        buffer.resolved_type(),
                        buffer.policy()) {}

}// namespace ocarina