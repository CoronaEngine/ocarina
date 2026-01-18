//
// Created by Z on 2026/1/18.
//

#include "texture.h"

namespace ocarina {

Texture::Texture(Device::Impl *device, ocarina::PixelStorage pixel_storage,
                 RHIResource::Tag tag, ocarina::handle_ty handle)
    : RHIResource(device, tag, handle) {
}


Texture3D::Texture3D(Device::Impl *device, uint3 res,
                     PixelStorage pixel_storage, uint level_num,
                     const string &desc)
    : Texture(device, Tag::TEXTURE3D,
              device->create_texture3d(res, pixel_storage,
                                       detail::compute_mip_level_num(res, level_num), desc)) {}

Texture3D::Texture3D(Device::Impl *device, Image *image_resource, const TextureViewCreation &texture_view)
    : Texture(device, Tag::TEXTURE3D,
              device->create_texture3d(image_resource, texture_view)) {}

Texture3DUploadCommand *Texture3D::upload(const void *data, bool async) const noexcept {
    return Texture3DUploadCommand::create(data, array_handle(), impl()->resolution(),
                                          impl()->pixel_storage(), async);
}

Texture3DUploadCommand *Texture3D::upload_sync(const void *data) const noexcept {
    return upload(data, false);
}

Texture3DDownloadCommand *Texture3D::download(void *data, bool async) const noexcept {
    return Texture3DDownloadCommand::create(data, array_handle(), impl()->resolution(),
                                            impl()->pixel_storage(), async);
}

Texture3DDownloadCommand *Texture3D::download_sync(void *data) const noexcept {
    return download(data, false);
}

void Texture3D::upload_immediately(const void *data) const noexcept {
    upload_sync(data)->accept(*device_->command_visitor());
}

void Texture3D::download_immediately(void *data) const noexcept {
    download_sync(data)->accept(*device_->command_visitor());
}

const Expression *Texture3D::expression() const noexcept {
    const CapturedResource &captured_resource = Function::current()->get_captured_resource(Type::of<decltype(*this)>(),
                                                                                           Variable::Tag::TEXTURE3D,
                                                                                           memory_block());
    return captured_resource.expression();
}


}// namespace ocarina