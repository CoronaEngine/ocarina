//
// Created by Zero on 12/01/2023.
//

#include "cuda_bindless_array.h"
#include "cuda_device.h"

namespace ocarina {

CUDABindlessArray::CUDABindlessArray(CUDADevice *device)
    : device_(device), buffers_(device, c_max_slot_num, "CUDABindlessArray::buffers_"),
      texture3ds_(device, c_max_slot_num, "CUDABindlessArray::texture3ds_"),
      texture2ds_(device, c_max_slot_num, "CUDABindlessArray::texture2ds_") {
    slot_soa_.buffer_slot = buffers_.handle();
    slot_soa_.tex3d_slot = texture3ds_.handle();
    slot_soa_.tex2d_slot = texture2ds_.handle();
}

void CUDABindlessArray::prepare_slotSOA(Device &device) noexcept {
    buffers_.reset_device_buffer_immediately(device);
    texture3ds_.reset_device_buffer_immediately(device);
    texture2ds_.reset_device_buffer_immediately(device);
    slot_soa_.buffer_slot = buffers_.handle();
    slot_soa_.tex3d_slot = texture3ds_.handle();
    slot_soa_.tex2d_slot = texture2ds_.handle();
}

CommandList CUDABindlessArray::update_slotSOA(bool async) noexcept {
    CommandList ret;
    append(ret, buffers_.device_buffer().reallocate(buffers_.host_buffer().size(), async));
    append(ret, texture3ds_.device_buffer().reallocate(texture3ds_.host_buffer().size(), async));
    append(ret, texture2ds_.device_buffer().reallocate(texture2ds_.host_buffer().size(), async));
    ret.push_back(HostFunctionCommand::create([&]() {
        slot_soa_.buffer_slot = buffers_.handle();
        slot_soa_.tex3d_slot = texture3ds_.handle();
        slot_soa_.tex2d_slot = texture2ds_.handle();
    },
                                              async));
    return ret;
}

namespace detail {
template<typename T>
void remove_by_index(vector<T> &v, handle_ty index) noexcept {
    auto iter = v.begin();
    for (int i = 0; i < v.size(); ++i, ++iter) {
        if (i == index) {
            v.erase(iter);
            return;
        }
    }
}
}// namespace detail

size_t CUDABindlessArray::emplace_buffer(handle_ty handle, uint offset_in_byte, size_t size_in_byte) noexcept {
    auto ret = buffers_.host_buffer().size();
    buffers_.emplace_back(reinterpret_cast<std::byte *>(handle), offset_in_byte, size_in_byte);
    OC_ERROR_IF(ret >= c_max_slot_num, ocarina::format("slot_size is {}, buffer_num is {}", c_max_slot_num, ret));
    return ret;
}

void CUDABindlessArray::remove_buffer(handle_ty index) noexcept {
    detail::remove_by_index(buffers_.host_buffer(), index);
}

void CUDABindlessArray::set_buffer(ocarina::handle_ty index, ocarina::handle_ty handle,
                                   uint offset_in_byte, size_t size_in_byte) noexcept {
    OC_ASSERT(index < buffers_.host_buffer().size());
    buffers_.host_buffer().at(index) = {reinterpret_cast<std::byte *>(handle),
                                        offset_in_byte, size_in_byte};
}

size_t CUDABindlessArray::buffer_num() const noexcept {
    return buffers_.host_buffer().size();
}

size_t CUDABindlessArray::buffer_slot_size() const noexcept {
    return sizeof(ByteBufferDesc) * c_max_slot_num;
}

BufferUploadCommand *CUDABindlessArray::upload_buffer_handles(bool async) const noexcept {
    return buffers_.upload(async);
}

size_t CUDABindlessArray::emplace_texture3d(handle_ty handle) noexcept {
    auto ret = texture3ds_.host_buffer().size();
    texture3ds_.push_back(handle);
    OC_ERROR_IF(ret >= c_max_slot_num, ocarina::format("slot_size is {}, tex_num is {}", c_max_slot_num, ret));
    return ret;
}

void CUDABindlessArray::remove_texture3d(handle_ty index) noexcept {
    detail::remove_by_index(texture3ds_.host_buffer(), index);
}

void CUDABindlessArray::set_texture3d(ocarina::handle_ty index, ocarina::handle_ty handle) noexcept {
    OC_ASSERT(index < texture3ds_.host_buffer().size());
    texture3ds_.host_buffer().at(index) = handle;
}

size_t CUDABindlessArray::texture3d_num() const noexcept {
    return texture3ds_.host_buffer().size();
}

size_t CUDABindlessArray::tex3d_slot_size() const noexcept {
    return sizeof(CUtexObject) * c_max_slot_num;
}

BufferUploadCommand *CUDABindlessArray::upload_texture3d_handles(bool async) const noexcept {
    return texture3ds_.upload(async);
}

size_t CUDABindlessArray::emplace_texture2d(handle_ty handle) noexcept {
    auto ret = texture2ds_.host_buffer().size();
    texture2ds_.push_back(handle);
    OC_ERROR_IF(ret >= c_max_slot_num, ocarina::format("slot_size is {}, tex_num is {}", c_max_slot_num, ret));
    return ret;
}

void CUDABindlessArray::remove_texture2d(handle_ty index) noexcept {
    detail::remove_by_index(texture2ds_.host_buffer(), index);
}

void CUDABindlessArray::set_texture2d(ocarina::handle_ty index, ocarina::handle_ty handle) noexcept {
    OC_ASSERT(index < texture2ds_.host_buffer().size());
    texture2ds_.host_buffer().at(index) = handle;
}
size_t CUDABindlessArray::tex2d_slot_size() const noexcept {
    return sizeof(CUtexObject) * c_max_slot_num;
}

size_t CUDABindlessArray::texture2d_num() const noexcept {
    return texture2ds_.host_buffer().size();
}

BufferUploadCommand *CUDABindlessArray::upload_texture2d_handles(bool async) const noexcept {
    return texture2ds_.upload(async);
}

ByteBufferDesc CUDABindlessArray::buffer_view(ocarina::uint index) const noexcept {
    return buffers_.host_buffer().at(index);
}

}// namespace ocarina