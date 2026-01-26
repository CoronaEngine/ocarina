//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/resources/texture.h"
#include "driver_types.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDATexture : public Texture::Impl {
private:
    TextureDesc proxy_;
    CUDADevice *device_{};
    uint3 res_{};
    CUarray array_handle_{};
    uint level_num_{1u};
    mutable cudaGraphicsResource *gfx_resource_{};

public:
    CUDATexture(CUDADevice *device, uint3 res, PixelStorage pixel_storage, uint level_num);
    ~CUDATexture() override;
    void init();
    [[nodiscard]] uint3 resolution() const noexcept override { return res_; }
    [[nodiscard]] handle_ty array_handle() const noexcept override {
        return reinterpret_cast<handle_ty>(array_handle_);
    }
    [[nodiscard]] const handle_ty *array_handle_ptr() const noexcept override {
        return reinterpret_cast<handle_ty *>(array_handle_);
    }
    [[nodiscard]] handle_ty tex_handle() const noexcept override {
        return proxy_.texture;
    }
    [[nodiscard]] const void *handle_ptr() const noexcept override {
        return &proxy_;
    }
    [[nodiscard]] size_t data_size() const noexcept override;
    [[nodiscard]] size_t data_alignment() const noexcept override;
    [[nodiscard]] size_t max_member_size() const noexcept override;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept override { return proxy_.pixel_storage; }
    [[nodiscard]] const TextureSampler* get_sampler_pointer() const noexcept { return nullptr; }
};
}// namespace ocarina