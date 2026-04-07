//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "rhi/resources/resource.h"
#include "rhi/command.h"
#include "mesh.h"
#include "dsl/var.h"

namespace ocarina {
class OC_RHI_API Accel : public RHIResource {
private:
    uint triangle_num_{};
    uint vertex_num_{};

public:
    class Impl {
    protected:
        AccelUsageTag usage_tag_{FAST_TRACE};
        vector<RHIMesh> meshes_;
        vector<float4x4> transforms_;

    public:
        explicit Impl(AccelUsageTag usage_tag = FAST_TRACE) noexcept : usage_tag_(usage_tag) {}
        virtual void add_instance(RHIMesh mesh, float4x4 transform) noexcept {
            meshes_.push_back(ocarina::move(mesh));
            transforms_.push_back(transform);
        }
        virtual void set_transform(size_t index, float4x4 transform) noexcept {
            transforms_[index] = transform;
        }
        [[nodiscard]] AccelUsageTag usage_tag() const noexcept { return usage_tag_; }
        [[nodiscard]] virtual handle_ty handle() const noexcept = 0;
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;
        [[nodiscard]] size_t mesh_num() const noexcept { return meshes_.size(); }
        [[nodiscard]] virtual size_t data_size() const noexcept = 0;
        virtual void clear() noexcept {
            meshes_.clear();
            transforms_.clear();
        }
        [[nodiscard]] virtual size_t data_alignment() const noexcept = 0;
    };

public:
    Accel() = default;
    explicit Accel(Device::Impl *device, AccelUsageTag usage_tag = FAST_TRACE)
        : RHIResource(device, Tag::ACCEL, device->create_accel(usage_tag)) {}
    [[nodiscard]] uint triangle_num() const noexcept { return triangle_num_; }
    [[nodiscard]] uint vertex_num() const noexcept { return vertex_num_; }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(handle_); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(handle_); }

    void add_instance(RHIMesh mesh, float4x4 transform) noexcept {
        triangle_num_ += mesh.triangle_num();
        vertex_num_ += mesh.vertex_num();
        impl()->add_instance(ocarina::move(mesh), transform);
    }

    void set_transform(size_t index, float4x4 transform) noexcept {
        impl()->set_transform(index, transform);
    }

    [[nodiscard]] size_t mesh_num() const noexcept { return impl()->mesh_num(); }
    [[nodiscard]] AccelUsageTag usage_tag() const noexcept { return impl()->usage_tag(); }

    void clear() noexcept {
        if (impl()) {
            impl()->clear();
        }
        triangle_num_ = 0;
        vertex_num_ = 0;
    }

    [[nodiscard]] const Expression *expression() const noexcept override {
        const CapturedResource &captured_resource = Function::current()->get_captured_resource(Type::of<decltype(*this)>(),
                                                                               Variable::Tag::ACCEL,
                                                                               memory_block());
        return captured_resource.expression();
    }

    template<typename TRay>
    [[nodiscard]] Var<bool> trace_occlusion(const TRay &ray) const noexcept {
        return make_expr<Accel>(expression()).trace_occlusion(ray);
    }

    template<typename TRay>
    [[nodiscard]] Var<TriangleHit> trace_closest(const TRay &ray) const noexcept {
        return make_expr<Accel>(expression()).trace_closest(ray);
    }

    [[nodiscard]] handle_ty handle() const noexcept override { return impl()->handle(); }
    [[nodiscard]] const void *handle_ptr() const noexcept override { return impl()->handle_ptr(); }
    [[nodiscard]] TLASBuildCommand *build_bvh() noexcept {
        return TLASBuildCommand::create(handle_);
    }
    [[nodiscard]] TLASUpdateCommand *update_bvh() noexcept {
        return TLASUpdateCommand::create(handle_);
    }
};
}// namespace ocarina
