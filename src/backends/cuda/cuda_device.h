//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "rhi/resources/resource.h"
#include "core/thread_safety.h"
#include <cuda.h>
#include "util.h"
#include "cuda_compiler.h"

namespace ocarina {
class CUDADevice : public Device::Impl {
public:
    static constexpr size_t size(Type::Tag tag) {
        using Tag = Type::Tag;
        switch (tag) {
            case Tag::BUFFER: return sizeof(BufferDesc<>);
            case Tag::BYTE_BUFFER: return sizeof(BufferDesc<>);
            case Tag::ACCEL: return sizeof(handle_ty);
            case Tag::TEXTURE: return sizeof(TextureDesc);
            case Tag::BINDLESS_ARRAY: return sizeof(BindlessArrayDesc);
            default:
                return 0;
        }
    }
    // return size of type on device memory
    static constexpr size_t size(const Type *type) {
        auto ret = size(type->tag());
        return ret == 0 ? type->size() : ret;
    }
    static constexpr size_t alignment(Type::Tag tag) {
        using Tag = Type::Tag;
        switch (tag) {
            case Tag::BUFFER: return alignof(BufferDesc<>);
            case Tag::BYTE_BUFFER: return alignof(BufferDesc<>);
            case Tag::ACCEL: return alignof(handle_ty);
            case Tag::TEXTURE: return alignof(TextureDesc);
            case Tag::BINDLESS_ARRAY: return alignof(BindlessArrayDesc);
            default:
                return 0;
        }
    }
    // return alignment of type on device memory
    static constexpr size_t alignment(const Type *type) {
        auto ret = alignment(type->tag());
        return ret == 0 ? type->alignment() : ret;
    }
    // return the size of max member recursive
    static size_t max_member_size(const Type *type) {
        auto ret = type->max_member_size();
        return ret == 0 ? sizeof(handle_ty) : ret;
    }

private:
    CUdevice cu_device_{};
    CUcontext cu_ctx_{};
    OptixDeviceContext optix_device_context_{};
    std::unique_ptr<CommandVisitor> cmd_visitor_;
    uint32_t compute_capability_{};

    thread_safety<std::mutex> memory_guard_;
    std::unordered_map<handle_ty, ExportableResource::Data> exported_resources;

    class ContextGuard {
    private:
        CUcontext _ctx{};

    public:
        explicit ContextGuard(CUcontext handle)
            : _ctx(handle) {
            OC_CU_CHECK(cuCtxPushCurrent(_ctx));
        }
        ~ContextGuard() {
            CUcontext ctx = nullptr;
            OC_CU_CHECK(cuCtxPopCurrent(&ctx));
            if (ctx != _ctx) [[unlikely]] {
                OC_ERROR_FORMAT(
                    "Invalid CUDA context {} (expected {}).",
                    fmt::ptr(ctx), fmt::ptr(_ctx));
            }
        }
    };

public:
    explicit CUDADevice(RHIContext *context);
    void init_hardware_info();
    template<typename Func>
    decltype(auto) use_context(Func &&func) noexcept {
        ContextGuard cg(cu_ctx_);
        return func();
    }

    template<typename T>
    void download(T *host_ptr, CUdeviceptr device_ptr, size_t num = 1, size_t offset = 0) {
        use_context([&] {
            OC_CU_CHECK(cuMemcpyDtoH(host_ptr, device_ptr + offset * sizeof(T), num * sizeof(T)));
        });
    }
    template<typename T>
    [[nodiscard]] T download(CUdeviceptr device_ptr, size_t offset = 0) {
        T ret;
        download<T>(&ret, device_ptr, 1, offset);
        return ret;
    }
    void init_optix_context() noexcept;
    [[nodiscard]] OptixDeviceContext optix_device_context() const noexcept { return optix_device_context_; }
    [[nodiscard]] handle_ty create_buffer(size_t size, const string &desc, bool exported) noexcept override;
    void destroy_buffer(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_texture(uint3 res, PixelStorage pixel_storage,
                                           uint level_num,
                                           const string &desc) noexcept override;
    [[nodiscard]] handle_ty create_texture(Image *image, const TextureViewCreation &texture_view) noexcept override { return 0; }
    void destroy_texture(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_shader(const Function &function) noexcept override;
    [[nodiscard]] handle_ty create_shader_from_file(const std::string &file_name, ShaderType shader_type,
                                                    const std::set<string> &options) noexcept override { return InvalidUI64; }
    void destroy_shader(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_accel() noexcept override;
    void destroy_accel(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_stream() noexcept override;
    void destroy_stream(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_mesh(const MeshParams &params) noexcept override;
    void destroy_mesh(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_bindless_array() noexcept override;
    void destroy_bindless_array(handle_ty handle) noexcept override;
    void register_shared_buffer(void *&shared_handle, ocarina::uint &gl_handle) noexcept override;
    void register_shared_tex(void *&shared_handle, ocarina::uint &gl_handle) noexcept override;
    void mapping_shared_buffer(void *&shared_handle, handle_ty &handle) noexcept override;
    void mapping_shared_tex(void *&shared_handle, handle_ty &handle) noexcept override;
    void unmapping_shared(void *&shared_handle) noexcept override;
    void unregister_shared(void *&shared_handle) noexcept override;
    void init_rtx() noexcept override { init_optix_context(); }
    [[nodiscard]] CommandVisitor *command_visitor() noexcept override;
    void submit_frame() noexcept override {}
    [[nodiscard]] VertexBuffer *create_vertex_buffer() noexcept override { return nullptr; }
    [[nodiscard]] IndexBuffer *create_index_buffer(const void *initial_data, uint32_t indices_count, bool bit16) noexcept override { return nullptr; }
    [[nodiscard]] RHIRenderPass *create_render_pass(const RenderPassCreation &render_pass_creation) noexcept override { return nullptr; }
    void destroy_render_pass(RHIRenderPass *render_pass) noexcept override {}
    std::array<DescriptorSetLayout *, MAX_DESCRIPTOR_SETS_PER_SHADER> create_descriptor_set_layout(void **shaders, uint32_t shaders_count) noexcept override { return {}; }
    void bind_pipeline(const handle_ty pipeline) noexcept override {}
    [[nodiscard]] RHIPipeline *get_pipeline(const PipelineState &pipeline_state, RHIRenderPass *render_pass) noexcept override { return nullptr; }
    [[nodiscard]] DescriptorSet *get_global_descriptor_set(const string &name) noexcept override { return nullptr; }
    void bind_descriptor_sets(DescriptorSet **descriptor_set, uint32_t descriptor_sets_num, RHIPipeline *pipeline) noexcept override {}
    void begin_frame() noexcept override {}
    void end_frame() noexcept override {}

    void memory_allocate(handle_ty *handle, size_t size, bool exported) override;
    void memory_free(handle_ty *handle) override;

    [[nodiscard]] uint64_t get_aligned_memory_size(handle_ty handle) const override;

#if _WIN32 || _WIN64
    [[nodiscard]] handle_ty import_handle(handle_ty handle, size_t size) override;
    [[nodiscard]] uint64_t export_handle(handle_ty handle_) override;
#endif
};
}// namespace ocarina
