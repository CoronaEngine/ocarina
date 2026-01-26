//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "resource.h"
#include "rhi/command.h"
#include "texture_sampler.h"
#include "../graphics_descriptions.h"

namespace ocarina {

class Image;
namespace detail {
[[nodiscard]] constexpr uint compute_mip_level_num(uint3 res, uint request_level_num) noexcept {
    uint max_size = std::max({res.x, res.y, res.z});
    auto max_levels = 0u;
    while (max_size != 0u) {
        max_size >>= 1u;
        max_levels++;
    }
    return request_level_num == 0 ? max_levels : std::min(request_level_num, max_levels);
}
}// namespace detail

class Texture : public RHIResource {
protected:
    uint channel_num_{};
    mutable uint gl_handle_{0};
    mutable void *gl_shared_handle_{0};

public:
    class Impl {
    public:
        virtual ~Impl() = default;
        [[nodiscard]] virtual uint3 resolution() const noexcept = 0;
        [[nodiscard]] virtual PixelStorage pixel_storage() const noexcept = 0;
        [[nodiscard]] virtual handle_ty array_handle() const noexcept = 0;
        [[nodiscard]] virtual const handle_ty *array_handle_ptr() const noexcept = 0;
        [[nodiscard]] virtual handle_ty tex_handle() const noexcept = 0;
        [[nodiscard]] virtual const TextureSampler* get_sampler_pointer() const noexcept = 0;

        /// for device side structure
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;
        [[nodiscard]] virtual size_t data_size() const noexcept = 0;
        [[nodiscard]] virtual size_t data_alignment() const noexcept = 0;
        [[nodiscard]] virtual size_t max_member_size() const noexcept = 0;
    };

public:
    Texture() = default;
    explicit Texture(Device::Impl *device, uint3 res,
                     PixelStorage pixel_storage, uint level_num = 1u,
                     const string &desc = "")
        : RHIResource(device, Tag::TEXTURE,
                      device->create_texture(res, pixel_storage,
                                             detail::compute_mip_level_num(res, level_num), desc)),
          channel_num_(ocarina::channel_num(pixel_storage)) {}

    explicit Texture(Device::Impl *device, Image* image_resource, const TextureViewCreation& texture_view, const TextureSampler& sampler)
        : RHIResource(device, Tag::TEXTURE,
                      device->create_texture(image_resource, texture_view, sampler)),
          channel_num_(ocarina::channel_num(texture_view.format)) {}

    OC_MAKE_MEMBER_GETTER(channel_num, )

    [[nodiscard]] uint pixel_num() const noexcept {
        uint3 res = impl()->resolution();
        return res.x * res.y * res.z;
    }

    [[nodiscard]] uint size_in_byte() const noexcept {
        return pixel_num() * pixel_size();
    }

    [[nodiscard]] uint &gl_handle() const noexcept { return gl_handle_; }
    [[nodiscard]] void *&gl_shared_handle() const noexcept { return gl_shared_handle_; }

    void register_shared() const noexcept {
        device()->register_shared_tex(gl_shared_handle_, gl_handle_);
    }

    void mapping() const noexcept {
        device()->mapping_shared_tex(gl_shared_handle_,
                                     *const_cast<handle_ty *>(impl()->array_handle_ptr()));
    }

    void unmapping() const noexcept {
        device()->unmapping_shared(gl_shared_handle_);
    }

    void unregister_shared() const noexcept {
        device()->unregister_shared(gl_shared_handle_);
    }

    [[nodiscard]] uint pixel_size() const noexcept {
        return ::ocarina::pixel_size(impl()->pixel_storage());
    }

    [[nodiscard]] uint3 resolution() const noexcept {
        return impl()->resolution();
    }

    /// for dsl
    [[nodiscard]] const Expression *expression() const noexcept override {
        const CapturedResource &captured_resource = Function::current()->get_captured_resource(Type::of<decltype(*this)>(),
                                                                                               Variable::Tag::TEXTURE,
                                                                                               memory_block());
        return captured_resource.expression();
    }

    template<typename U, typename V>
    requires(is_all_floating_point_expr_v<U, V>)
    [[nodiscard]] auto sample(uint channel_num, const U &u, const V &v) const noexcept {
        return make_expr<Texture>(expression()).sample(channel_num, u, v);
    }

    template<typename UV>
    requires(is_float_vector2_v<expr_value_t<UV>>)
    OC_NODISCARD auto sample(uint channel_num, const UV &uv) const noexcept {
        return sample(channel_num, uv.x, uv.y);
    }

    template<typename U, typename V, typename W>
    requires(is_all_floating_point_expr_v<U, V>)
    [[nodiscard]] auto sample(uint channel_num, const U &u, const V &v, const W &w) const noexcept {
        return make_expr<Texture>(expression()).sample(channel_num, u, v, w);
    }

    template<typename UVW>
    requires(is_float_vector3_v<expr_value_t<UVW>>)
    OC_NODISCARD auto sample(uint channel_num, const UVW &uvw) const noexcept {
        return sample(channel_num, uvw.x, uvw.y, uvw.z);
    }

    template<typename Target, typename X, typename Y>
    requires(is_all_integral_expr_v<X, Y>)
    OC_NODISCARD auto read(const X &x, const Y &y) const noexcept {
        return make_expr<Texture>(expression()).read<Target>(x, y);
    }

    template<typename Target, typename XY>
    requires(is_int_vector2_v<expr_value_t<XY>> ||
             is_uint_vector2_v<expr_value_t<XY>> &&
                 (is_uchar_element_expr_v<Target> || is_float_element_expr_v<Target>))
    OC_NODISCARD auto read(const XY &xy) const noexcept {
        return read<Target>(xy.x, xy.y);
    }

    template<typename X, typename Y, typename Val>
    requires(is_all_integral_expr_v<X, Y> &&
             (is_uchar_element_expr_v<Val> || is_float_element_expr_v<Val>))
    void write(const Val &elm, const X &x, const Y &y) noexcept {
        make_expr<Texture>(expression()).write(elm, x, y);
    }

    template< typename Val>
    void write(const Val &elm, const Uint2 &xy) noexcept {
        write(elm, xy.x, xy.y);
    }

    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(handle_); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(handle_); }
    [[nodiscard]] Impl *operator->() noexcept { return impl(); }
    [[nodiscard]] const Impl *operator->() const noexcept { return impl(); }
    [[nodiscard]] handle_ty array_handle() const noexcept { return impl()->array_handle(); }
    [[nodiscard]] handle_ty tex_handle() const noexcept { return impl()->tex_handle(); }
    [[nodiscard]] const void *handle_ptr() const noexcept override { return impl()->handle_ptr(); }
    [[nodiscard]] size_t data_size() const noexcept override { return impl()->data_size(); }
    [[nodiscard]] size_t data_alignment() const noexcept override { return impl()->data_alignment(); }
    [[nodiscard]] size_t max_member_size() const noexcept override { return impl()->max_member_size(); }
    [[nodiscard]] const TextureSampler* get_sampler_pointer() const noexcept {
        return impl()->get_sampler_pointer();
    }
    [[nodiscard]] TextureUploadCommand *upload(const void *data, bool async = true) const noexcept {
        return TextureUploadCommand::create(data, array_handle(), impl()->resolution(),
                                            impl()->pixel_storage(), async);
    }
    [[nodiscard]] TextureUploadCommand *upload_sync(const void *data) const noexcept {
        return upload(data, false);
    }
    [[nodiscard]] TextureDownloadCommand *download(void *data, bool async = true) const noexcept {
        return TextureDownloadCommand::create(data, array_handle(), impl()->resolution(),
                                              impl()->pixel_storage(), async);
    }
    [[nodiscard]] TextureDownloadCommand *download_sync(void *data) const noexcept {
        return download(data, false);
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg>
    [[nodiscard]] BufferToTextureCommand *copy_from(const Arg &buffer, size_t buffer_offset, bool async = true) const noexcept {
        return BufferToTextureCommand::create(buffer.handle(), buffer_offset * buffer.element_size(), array_handle(),
                                              impl()->pixel_storage(),
                                              impl()->resolution(), 0, async);
    }

    void upload_immediately(const void *data) const noexcept {
        upload_sync(data)->accept(*device_->command_visitor());
    }

    void download_immediately(void *data) const noexcept {
        download_sync(data)->accept(*device_->command_visitor());
    }
};

template<typename T>
class Texture2D : public Texture {
public:
    using Super = Texture;
    static constexpr auto Dim = vector_dimension_v<T>;

public:
    Texture2D() = default;
    explicit Texture2D(Device::Impl *device, uint2 res,
                       PixelStorage pixel_storage, uint level_num = 1u,
                       const string &desc = "")
        : Texture(device, make_uint3(res, 1u), pixel_storage, level_num, desc) {}

    template<typename... Args>
    [[nodiscard]] auto sample(Args &&...args) const noexcept {
        return make_expr<Texture2D<T>>(expression()).sample(channel_num_, OC_FORWARD(args)...).template as_vec<Dim>();
    }

    template<typename... Args>
    [[nodiscard]] auto read(Args &&...args) const noexcept {
        return make_expr<Texture2D<T>>(expression()).template read<T>(OC_FORWARD(args)...);
    }

    template<typename... Args>
    void write(T &&elm, Args &&...args) noexcept {
        make_expr<Texture2D<T>>(expression()).write(OC_FORWARD(args)...);
    }
};

}// namespace ocarina