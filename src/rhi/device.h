//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"
#include "dsl/rtx_type.h"
#include "core/image_base.h"
#include "core/concepts.h"
#include "core/thread_pool.h"
#include "params.h"
#include "graphics_descriptions.h"
#include "pipeline_state.h"

namespace ocarina {

class RHIContext;

template<typename T, int... Dims>
class Buffer;

class ByteBuffer;

template<typename T, AccessMode mode = AOS, typename TBuffer = ByteBuffer>
class List;

template<typename T, AccessMode mode = AOS>
class ManagedList;

template<typename T>
class Managed;

class BindlessArray;

template<typename T>
class Shader;

class Stream;

class Texture;

class RHIMesh;

class Accel;

class CommandVisitor;

class VertexBuffer;
class IndexBuffer;
class RHIRenderPass;
class DescriptorSet;
class DescriptorSetLayout;
struct RHIPipeline;
class Image;

class OC_RHI_API Device : public concepts::Noncopyable {
public:
    class Impl : public concepts::Noncopyable {
    protected:
        RHIContext *file_manager_{};
        friend class Device;

    public:
        explicit Impl(RHIContext *ctx) : file_manager_(ctx) {}
        explicit Impl(RHIContext *ctx, const InstanceCreation &instance_creation) : file_manager_(ctx) {}
        [[nodiscard]] virtual handle_ty create_buffer(size_t size, const string &desc, bool exported = true) noexcept = 0;
        virtual void destroy_buffer(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_texture(uint3 res, PixelStorage pixel_storage,
                                                       uint level_num, const string &desc) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_texture(Image *image, const TextureViewCreation &texture_view) noexcept = 0;
        virtual void destroy_texture(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_shader(const Function &function) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_shader_from_file(const std::string &file_name, ShaderType shader_type, const std::set<string> &options) noexcept = 0;
        virtual void destroy_shader(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_accel() noexcept = 0;
        virtual void destroy_accel(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_stream() noexcept = 0;
        virtual void destroy_stream(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_mesh(const MeshParams &params) noexcept = 0;
        virtual void destroy_mesh(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_bindless_array() noexcept = 0;
        virtual void destroy_bindless_array(handle_ty handle) noexcept = 0;
        virtual void register_shared_buffer(void *&shared_handle, uint &gl_handle) noexcept = 0;
        virtual void register_shared_tex(void *&shared_handle, uint &gl_handle) noexcept = 0;
        virtual void mapping_shared_buffer(void *&shared_handle, handle_ty &handle) noexcept = 0;
        virtual void mapping_shared_tex(void *&shared_handle, handle_ty &handle) noexcept = 0;
        virtual void unmapping_shared(void *&shared_handle) noexcept = 0;
        virtual void unregister_shared(void *&shared_handle) noexcept = 0;
        [[nodiscard]] RHIContext *file_manager() noexcept { return file_manager_; }
        virtual void init_rtx() noexcept = 0;
        [[nodiscard]] virtual CommandVisitor *command_visitor() noexcept = 0;
        virtual void submit_frame() noexcept = 0;
        virtual VertexBuffer *create_vertex_buffer() noexcept = 0;
        virtual IndexBuffer *create_index_buffer(const void *initial_data, uint32_t indices_count, bool bit16) noexcept = 0;
        virtual void begin_frame() noexcept = 0;
        virtual void end_frame() noexcept = 0;
        virtual RHIRenderPass *create_render_pass(const RenderPassCreation &render_pass_creation) noexcept = 0;
        virtual void destroy_render_pass(RHIRenderPass *render_pass) noexcept = 0;
        virtual std::array<DescriptorSetLayout *, MAX_DESCRIPTOR_SETS_PER_SHADER> create_descriptor_set_layout(void **shaders, uint32_t shaders_count) noexcept = 0;
        virtual void bind_pipeline(const handle_ty pipeline) noexcept = 0;
        virtual RHIPipeline *get_pipeline(const PipelineState &pipeline_state, RHIRenderPass *render_pass) noexcept = 0;
        virtual DescriptorSet *get_global_descriptor_set(const string &name) noexcept = 0;
        virtual void bind_descriptor_sets(DescriptorSet **descriptor_set, uint32_t descriptor_sets_num, RHIPipeline *pipeline) noexcept = 0;

        virtual void memory_allocate(handle_ty *handle, size_t size, bool exported = true) {}
        virtual void memory_free(handle_ty *handle) {}

#if _WIN32 || _WIN64
        virtual handle_ty import_handle(handle_ty handle, size_t size) { return 0; }
        virtual uint64_t export_handle(handle_ty handle_) { return 0; }
#endif
    };

    using Creator = Device::Impl *(RHIContext *);
    using Deleter = void(Device::Impl *);
    using Handle = ocarina::unique_ptr<Device::Impl, Device::Deleter *>;

private:
    Handle impl_;

public:
    explicit Device(Handle impl) : impl_(std::move(impl)) {}
    [[nodiscard]] RHIContext *file_manager() const noexcept { return impl_->file_manager_; }
    template<typename T, typename... Args>
    [[nodiscard]] auto create(Args &&...args) const noexcept {
        return T(this->impl_.get(), std::forward<Args>(args)...);
    }
    template<typename T = std::byte, int... Dims>
    [[nodiscard]] Buffer<T, Dims...> create_buffer(size_t size, const string &name = "") const noexcept {
        return Buffer<T, Dims...>(impl_.get(), size, name);
    }

    template<typename T = std::byte, int... Dims>
    [[nodiscard]] Buffer<T, Dims...> create_buffer(size_t size, handle_ty handle) const noexcept {
        return Buffer<T, Dims...>(impl_.get(), size, handle);
    }

    [[nodiscard]] ByteBuffer create_byte_buffer(size_t size, const string &name = "") const noexcept;

    template<typename T, AccessMode mode = AOS>
    [[nodiscard]] List<T, mode> create_list(size_t size, const string &name = "") const noexcept;// implement in byte_buffer.h

    template<typename T, AccessMode mode = AOS>
    [[nodiscard]] ManagedList<T, mode> create_managed_list(size_t size, const string &name = "") const noexcept {
        return ManagedList<T, mode>(create_list<T, mode>(size, name));
    }

    template<typename T = std::byte, int... Dims>
    [[nodiscard]] Buffer<T, Dims...> create_buffer(size_t size, handle_ty stream) noexcept {
        return Buffer<T, Dims...>(impl_.get(), size, stream);
    }

    void destroy_buffer(handle_ty handle) noexcept {
        impl_->destroy_buffer(handle);
    }

    handle_ty import_handle(handle_ty handle, size_t size) noexcept {
        return impl_->import_handle(handle, size);
    }

    uint64_t export_handle(handle_ty handle) noexcept {
        return impl_->export_handle(handle);
    }

    template<typename T = std::byte>
    [[nodiscard]] Managed<T> create_managed(size_t size) noexcept {
        return Managed<T>(impl_.get(), size);
    }

    template<typename T = std::byte>
    [[nodiscard]] Managed<T> create_managed(size_t size, handle_ty stream) noexcept {
        return Managed<T>(impl_.get(), size, stream);
    }

    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] RHIMesh create_mesh(const VBuffer &v_buffer,
                                      const TBuffer &t_buffer,
                                      AccelUsageTag usage_tag = AccelUsageTag::FAST_TRACE,
                                      AccelGeomTag geom_tag = AccelGeomTag::DISABLE_ANYHIT) const noexcept;// implement in mesh.h
    [[nodiscard]] Stream create_stream() noexcept;
    [[nodiscard]] Accel create_accel() const noexcept;
    [[nodiscard]] BindlessArray create_bindless_array() const noexcept;
    void init_rtx() noexcept { impl_->init_rtx(); }
    [[nodiscard]] Texture create_texture(uint3 res, PixelStorage storage, const string &desc = "") const noexcept;
    [[nodiscard]] Texture create_texture(uint2 res, PixelStorage storage, const string &desc = "") const noexcept;
    [[nodiscard]] Texture create_texture(Image *image_resource, const TextureViewCreation &texture_view) const noexcept;
    template<typename T>
    [[nodiscard]] auto compile(const Kernel<T> &kernel, const string &shader_desc = "", ShaderTag tag = CS) const noexcept {
        OC_INFO_FORMAT("compile shader : {}", shader_desc.c_str());
        kernel.function()->set_description(shader_desc);
        return create<Shader<T>>(kernel.function(), tag);
    }
    template<typename T>
    [[nodiscard]] auto async_compile(Kernel<T> &&kernel, const string &shader_desc = "", ShaderTag tag = CS) const noexcept {
        return async([=, this, kernel = ocarina::move(kernel)] {
            return compile(kernel, shader_desc, tag);
        });
    }

    [[nodiscard]] handle_ty create_shader_from_file(const std::string &file_name, ShaderType shader_type, std::set<std::string> &options) {
        return impl_->create_shader_from_file(file_name, shader_type, options);
    }

    [[nodiscard]] VertexBuffer *create_vertex_buffer() {
        return impl_->create_vertex_buffer();
    }

    [[nodiscard]] IndexBuffer *create_index_buffer(const void *initial_data, uint32_t indices_count, bool bit16 = true) {
        return impl_->create_index_buffer(initial_data, indices_count, bit16);
    }

    void begin_frame() {
        impl_->begin_frame();
    }

    void end_frame() {
        impl_->end_frame();
    }

    void submit_frame() {
        impl_->submit_frame();
    }

    [[nodiscard]] RHIRenderPass *create_render_pass(const RenderPassCreation &render_pass_creation) {
        return impl_->create_render_pass(render_pass_creation);
    }

    void destroy_render_pass(RHIRenderPass *render_pass) {
        impl_->destroy_render_pass(render_pass);
    }

    [[nodiscard]] std::array<DescriptorSetLayout *, MAX_DESCRIPTOR_SETS_PER_SHADER> create_descriptor_set_layout(void **shaders, uint32_t shaders_count) {
        return impl_->create_descriptor_set_layout(shaders, shaders_count);
    }

    void bind_pipeline(const handle_ty pipeline) noexcept {
        impl_->bind_pipeline(pipeline);
    }

    RHIPipeline *get_pipeline(const PipelineState &pipeline_state, RHIRenderPass *render_pass) noexcept {
        return impl_->get_pipeline(pipeline_state, render_pass);
    }

    DescriptorSet *get_global_descriptor_set(const string &name) noexcept {
        return impl_->get_global_descriptor_set(name);
    }

    void bind_descriptor_sets(DescriptorSet **descriptor_sets, uint32_t descriptor_sets_num, RHIPipeline *pipeline) noexcept {
        impl_->bind_descriptor_sets(descriptor_sets, descriptor_sets_num, pipeline);
    }

    [[nodiscard]] static Device create_device(const string &backend_name, const ocarina::InstanceCreation &instance_creation);
    [[nodiscard]] static Device create_device(const string &backend_name);
};

namespace rhi_global {
//OC_EXPORT_API Device rhi_create_device(const string &backend_name, const ocarina::InstanceCreation &instance_creation);
}
}// namespace ocarina