//
// Created by Zero on 06/06/2022.
//

#include "cuda_device.h"
#include "cuda_stream.h"
#include "cuda_texture.h"
#include "cuda_shader.h"
#include "cuda_mesh.h"
#include "cuda_bindless_array.h"
#include "rhi/context.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <nvrtc.h>
#include "cuda_gl_interop.h"
#include "driver_types.h"
#include "cuda_compiler.h"
#include "optix_accel.h"
#include "cuda_command_visitor.h"

namespace ocarina {

CUDADevice::CUDADevice(RHIContext *file_manager)
    : Device::Impl(file_manager) {
    OC_CU_CHECK(cuInit(0));
    OC_CU_CHECK(cuDeviceGet(&cu_device_, 0));
    OC_CU_CHECK(cuDevicePrimaryCtxRetain(&cu_ctx_, cu_device_));
    cmd_visitor_ = std::make_unique<CUDACommandVisitor>(this);
    init_hardware_info();
}

void CUDADevice::init_hardware_info() {
    auto compute_cap_major = 0;
    auto compute_cap_minor = 0;
    OC_CU_CHECK(cuDeviceGetAttribute(&compute_cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device_));
    OC_CU_CHECK(cuDeviceGetAttribute(&compute_cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device_));
    OC_INFO_FORMAT(
        "Created CUDA device : (capability = {}.{}).",
        compute_cap_major, compute_cap_minor);
    compute_capability_ = 10u * compute_cap_major + compute_cap_minor;
}

handle_ty CUDADevice::create_buffer(size_t size, const string &desc) noexcept {
    OC_ASSERT(size > 0);
    return use_context([&] {
        handle_ty handle{};
        OC_CU_CHECK(cuMemAlloc(&handle, size));
        MemoryStats::instance().on_buffer_allocate(handle, size, desc);
        return handle;
    });
}

namespace detail {
void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}
}// namespace detail

void CUDADevice::init_optix_context() noexcept {
    if (optix_device_context_) {
        return;
    }
    use_context([&] {
        OC_CU_CHECK(cuMemFree(0));
        OC_OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions ctx_options = {};
#ifndef NDEBUG
        ctx_options.logCallbackLevel = 4;// status/progress
#else
        ctx_options.logCallbackLevel = 2;// error
#endif
        ctx_options.logCallbackFunction = detail::context_log_cb;
#if (OPTIX_VERSION >= 70200)
        ctx_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif
        CUcontext cu_context = nullptr;
        OC_OPTIX_CHECK(optixDeviceContextCreate(cu_context, &ctx_options, &optix_device_context_));
    });
}

handle_ty CUDADevice::create_stream() noexcept {
    return use_context([&] {
        CUDAStream *stream = ocarina::new_with_allocator<CUDAStream>(this);
        return reinterpret_cast<handle_ty>(stream);
    });
}

handle_ty CUDADevice::create_texture(uint3 res, PixelStorage pixel_storage,
                                     uint level_num,
                                     const string &desc) noexcept {
    return use_context([&] {
        auto texture = ocarina::new_with_allocator<CUDATexture>(this, res, pixel_storage, level_num);
        MemoryStats::instance().on_tex_allocate(reinterpret_cast<handle_ty>(texture),
                                                res, pixel_storage, desc);
        return reinterpret_cast<handle_ty>(texture);
    });
}

handle_ty CUDADevice::create_shader(const Function &function) noexcept {
    CUDACompiler compiler(this);
    ocarina::string ptx = compiler.compile(function, compute_capability_);

    auto ptr = use_context([&] {
        auto shader = CUDAShader::create(this, ptx, function);
        return reinterpret_cast<handle_ty>(shader);
    });

    return ptr;
}

handle_ty CUDADevice::create_mesh(const MeshParams &params) noexcept {
    auto ret = new_with_allocator<CUDAMesh>(this, params);
    return reinterpret_cast<handle_ty>(ret);
}

void CUDADevice::destroy_mesh(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<CUDAMesh *>(handle));
}

handle_ty CUDADevice::create_bindless_array() noexcept {
    auto ret = new_with_allocator<CUDABindlessArray>(this);
    return reinterpret_cast<handle_ty>(ret);
}

void CUDADevice::destroy_bindless_array(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<CUDABindlessArray *>(handle));
}

void CUDADevice::register_shared_buffer(void *&shared_handle, ocarina::uint &gl_handle) noexcept {
    if (shared_handle != nullptr) {
        return;
    }
    OC_CUDA_CHECK(cudaGraphicsGLRegisterBuffer(reinterpret_cast<cudaGraphicsResource_t *>(&shared_handle),
                                               gl_handle, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
}

void CUDADevice::register_shared_tex(void *&shared_handle, ocarina::uint &gl_handle) noexcept {
    if (shared_handle != nullptr) {
        return;
    }
    OC_CUDA_CHECK(cudaGraphicsGLRegisterImage(reinterpret_cast<cudaGraphicsResource_t *>(&shared_handle),
                                              gl_handle, GL_TEXTURE_2D,
                                              CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
}

void CUDADevice::mapping_shared_buffer(void *&shared_handle, handle_ty &handle) noexcept {
    OC_CUDA_CHECK(cudaGraphicsMapResources(1, reinterpret_cast<cudaGraphicsResource_t *>(&shared_handle)));
    size_t buffer_size = 0u;
    OC_CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void **>(&handle),
        &buffer_size,
        reinterpret_cast<cudaGraphicsResource_t>(shared_handle)));
}

void CUDADevice::mapping_shared_tex(void *&shared_handle, handle_ty &handle) noexcept {
    OC_CUDA_CHECK(cudaGraphicsMapResources(1, reinterpret_cast<cudaGraphicsResource_t *>(&shared_handle)));
    OC_CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(reinterpret_cast<cudaArray_t *>(handle),
                                                        reinterpret_cast<cudaGraphicsResource_t>(shared_handle), 0, 0));
}

void CUDADevice::unmapping_shared(void *&shared_handle) noexcept {
    OC_CUDA_CHECK(cudaGraphicsUnmapResources(1,
                                             reinterpret_cast<cudaGraphicsResource_t *>(&shared_handle)));
}

void CUDADevice::unregister_shared(void *&shared_handle) noexcept {
    if (shared_handle == nullptr) {
        return;
    }
    OC_CUDA_CHECK(cudaGraphicsUnregisterResource(reinterpret_cast<cudaGraphicsResource_t>(shared_handle)));
}

void CUDADevice::destroy_buffer(handle_ty handle) noexcept {
    if (handle != 0) {
        MemoryStats::instance().on_buffer_free(handle);
        OC_CU_CHECK(cuMemFree(handle));
    }
}

void CUDADevice::destroy_shader(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<CUDAShader *>(handle));
}

void CUDADevice::destroy_texture(handle_ty handle) noexcept {
    use_context([&] {
        MemoryStats::instance().on_tex_free(handle);
        ocarina::delete_with_allocator(reinterpret_cast<CUDATexture *>(handle));
    });
}

void CUDADevice::destroy_stream(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<CUDAStream *>(handle));
}
handle_ty CUDADevice::create_accel() noexcept {
    return use_context([&] {
        auto accel = new_with_allocator<OptixAccel>(this);
        return reinterpret_cast<handle_ty>(accel);
    });
}
void CUDADevice::destroy_accel(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<OptixAccel *>(handle));
}
CommandVisitor *CUDADevice::command_visitor() noexcept {
    return cmd_visitor_.get();
}

#if _WIN32 || _WIN64
handle_ty CUDADevice::import_handle(handle_ty handle, size_t size) {
    //CUDA_EXTERNAL_MEMORY_HANDLE_DESC externalMemoryHandleDesc = {};
    //externalMemoryHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
    //externalMemoryHandleDesc.handle.win32.handle = reinterpret_cast<void *>(handle);
    //externalMemoryHandleDesc.size = size;
    //externalMemoryHandleDesc.flags = 0;

    //CUexternalMemory externalMemory;
    //CUresult res = cuImportExternalMemory(&externalMemory, &externalMemoryHandleDesc);
    //if (res != CUDA_SUCCESS) {
    //    const char *error_str;
    //    cuGetErrorString(res, &error_str);
    //    throw std::runtime_error(std::string("Failed to import external memory: ") + error_str);
    //}

    //CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {};
    //bufferDesc.offset = 0;
    //bufferDesc.size = size;
    //bufferDesc.flags = 0;

    CUdeviceptr devicePtr;
    //res = cuExternalMemoryGetMappedBuffer(&devicePtr, externalMemory, &bufferDesc);
    //if (res != CUDA_SUCCESS) {
    //    const char *error_str;
    //    cuGetErrorString(res, &error_str);
    //    throw std::runtime_error(std::string("Failed to get mapped buffer: ") + error_str);
    //}

    return static_cast<handle_ty>(devicePtr);
}

uint64_t CUDADevice::export_handle(handle_ty handle_) {
    void *exported_win32_handle = nullptr;
    //CUresult res = cuMemExportToShareableHandle(
    //    &exported_win32_handle,
    //    reinterpret_cast<CUmemGenericAllocationHandle>(handle_),
    //    CU_MEM_HANDLE_TYPE_WIN32,
    //    0);

    //if (res != CUDA_SUCCESS) {
    //    const char *error_str;
    //    cuGetErrorString(res, &error_str);
    //    throw std::runtime_error(std::string("Failed to export shareable handle: ") + error_str);
    //}

    return reinterpret_cast<uint64_t>(exported_win32_handle);
}
#endif

}// namespace ocarina

OC_EXPORT_API ocarina::CUDADevice *create(ocarina::RHIContext *file_manager) {
    return ocarina::new_with_allocator<ocarina::CUDADevice>(file_manager);
}

OC_EXPORT_API ocarina::CUDADevice *create_device(ocarina::RHIContext *file_manager) {
    return ocarina::new_with_allocator<ocarina::CUDADevice>(file_manager);
}

OC_EXPORT_API void destroy(ocarina::CUDADevice *device) {
    ocarina::delete_with_allocator(device);
    OC_INFO("cuda device is destroy!");
}