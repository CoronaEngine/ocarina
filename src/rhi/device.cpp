//
// Created by Zero on 06/06/2022.
//

#include "device.h"
#include "resources/texture.h"
#include "resources/stream.h"
#include "rtx/accel.h"
#include "resources/bindless_array.h"
#include "resources/byte_buffer.h"
#include "context.h"
#include "core/dynamic_module.h"

namespace ocarina {

ByteBuffer Device::create_byte_buffer(size_t size, const std::string &name) const noexcept {
    return ByteBuffer(impl_.get(), size, name);
}

Stream Device::create_stream() noexcept {
    return create<Stream>();
}

Accel Device::create_accel() const noexcept {
    return create<Accel>();
}

BindlessArray Device::create_bindless_array() const noexcept {
    return create<BindlessArray>();
}

Texture Device::create_texture(uint3 res, PixelStorage storage, const string &desc) const noexcept {
    return create<Texture>(res, storage, 1, desc);
}

Texture Device::create_texture(uint2 res, PixelStorage storage, const string &desc) const noexcept {
    return create_texture(make_uint3(res, 1u), storage, desc);
}

Texture Device::create_texture(Image *image_resource, const TextureViewCreation &texture_view, const TextureSampler& sampler) const noexcept {
    Texture tex(impl_.get(), image_resource, texture_view, sampler);
    return tex;
}

Device Device::create_device(const string &backend_name, const ocarina::InstanceCreation &instance_creation) {
    RHIContext &rhi_context = RHIContext::instance();
    return rhi_context.create_device(backend_name, instance_creation);
    //std::string full_backend_name = detail::backend_full_name(backend_name);
    //auto d = context.obtain_module(dynamic_module_name(full_backend_name));
    //using Constructor = Device::Impl *(RHIContext *, const InstanceCreation &instance_creation);
    //auto create_device = reinterpret_cast<Constructor *>(d->function_ptr("create_device"));
    //auto destroy_func = reinterpret_cast<Device::Deleter *>(d->function_ptr("destroy"));
    //return Device{Device::Handle{create_device(&context, instance_creation), destroy_func}};
}

Device Device::create_device(const string &backend_name) {
    RHIContext &rhi_context = RHIContext::instance();
    return rhi_context.create_device(backend_name);
    //std::string full_backend_name = detail::backend_full_name(backend_name);
    //auto d = context.obtain_module(dynamic_module_name(full_backend_name));
    //using Constructor = Device::Impl *(RHIContext *, const InstanceCreation &instance_creation);
    //auto create_device = reinterpret_cast<Constructor *>(d->function_ptr("create_device"));
    //auto destroy_func = reinterpret_cast<Device::Deleter *>(d->function_ptr("destroy"));
    //return Device{Device::Handle{create_device(&context, instance_creation), destroy_func}};
}

}// namespace ocarina