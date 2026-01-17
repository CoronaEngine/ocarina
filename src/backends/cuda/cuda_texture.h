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
class CUDATexture3D : public Texture3D::Impl {
private:
    mutable TextureDesc descriptor_;
    CUDADevice *device_{};
    uint3 res_{};
    uint level_num_{1u};

public:
    CUDATexture3D(CUDADevice *device, uint3 res, PixelStorage pixel_storage, uint level_num);
    ~CUDATexture3D() override;
    void init();
    [[nodiscard]] uint3 resolution() const noexcept override { return res_; }
    [[nodiscard]] handle_ty array_handle() const noexcept override {
        return descriptor_.array;
    }
    [[nodiscard]] const handle_ty *array_handle_ptr() const noexcept override {
        return &descriptor_.array;
    }
    [[nodiscard]] CUarray *cu_array_ptr() { return reinterpret_cast<CUarray*>(&descriptor_.array);}
    [[nodiscard]] CUarray cu_array() const { return reinterpret_cast<CUarray>(descriptor_.array);}
    [[nodiscard]] CUarray cu_array() { return reinterpret_cast<CUarray>(descriptor_.array);}
    [[nodiscard]] const TextureDesc & descriptor() const noexcept override;
    [[nodiscard]] handle_ty tex_handle() const noexcept override {
        return descriptor_.texture;
    }
    [[nodiscard]] const void *handle_ptr() const noexcept override {
        return &descriptor_;
    }
    [[nodiscard]] size_t data_size() const noexcept override;
    [[nodiscard]] size_t data_alignment() const noexcept override;
    [[nodiscard]] size_t max_member_size() const noexcept override;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept override { return descriptor_.pixel_storage; }
};
}// namespace ocarina