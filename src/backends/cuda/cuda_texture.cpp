//
// Created by Zero on 06/08/2022.
//

#include "cuda_texture.h"
#include "util.h"
#include "cuda_device.h"
#include "cuda_runtime_api.h"
//#include "texture_fetch_functions.h"
#include <cuda_gl_interop.h>

namespace ocarina {

CUDATexture3D::CUDATexture3D(CUDADevice *device, uint3 res, PixelStorage pixel_storage, uint level_num)
    : device_(device), res_(res), level_num_(level_num) {
    descriptor_.pixel_storage = pixel_storage;
    init();
}

size_t CUDATexture3D::data_size() const noexcept { return CUDADevice::size(Type::Tag::TEXTURE); }
size_t CUDATexture3D::data_alignment() const noexcept { return CUDADevice::alignment(Type::Tag::TEXTURE); }
size_t CUDATexture3D::max_member_size() const noexcept { return sizeof(handle_ty); }

const TextureDesc &CUDATexture3D::descriptor() const noexcept {
    return descriptor_;
}

void CUDATexture3D::init() {
    CUDA_ARRAY3D_DESCRIPTOR array_desc{};
    array_desc.Width = res_.x;
    array_desc.Height = res_.y;
    array_desc.Depth = res_.z;
    switch (descriptor_.pixel_storage) {
        case PixelStorage::BYTE1:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 1;
            break;
        case PixelStorage::BYTE2:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 2;
            break;
        case PixelStorage::BYTE4:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 4;
            break;
        case PixelStorage::FLOAT1:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 1;
            break;
        case PixelStorage::FLOAT2:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 2;
            break;
        case PixelStorage::FLOAT4:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 4;
            break;
        default: OC_ASSERT(0); break;
    }

    OC_CU_CHECK(cuArray3DCreate(cu_array_ptr(), &array_desc));

    CUDA_RESOURCE_DESC res_desc{};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = cu_array();
    res_desc.flags = 0;
    CUDA_TEXTURE_DESC tex_desc{};
    tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_MIRROR;
    tex_desc.addressMode[1] = CU_TR_ADDRESS_MODE_MIRROR;
    tex_desc.addressMode[2] = CU_TR_ADDRESS_MODE_MIRROR;
    tex_desc.maxAnisotropy = 2;
    tex_desc.maxMipmapLevelClamp = 9;
    tex_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
    tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    OC_CU_CHECK(cuSurfObjectCreate(&descriptor_.surface, &res_desc));
    OC_CU_CHECK(cuTexObjectCreate(&descriptor_.texture, &res_desc, &tex_desc, nullptr));
}

CUDATexture3D::~CUDATexture3D() {
    OC_CU_CHECK(cuArrayDestroy(cu_array()));
    OC_CU_CHECK(cuTexObjectDestroy(descriptor_.texture));
    OC_CU_CHECK(cuSurfObjectDestroy(descriptor_.surface));
}


}// namespace ocarina