//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/resources/texture.h"
#include "rhi/resources/texture_sampler.h"
#include <vulkan/vulkan.h>

namespace ocarina {

class VulkanDevice;
class Image;

class VulkanTexture : public Texture::Impl {
private:
    VkImageType imageType_;
    VkImage image_ = VK_NULL_HANDLE;
    VkImageView image_view_ = VK_NULL_HANDLE;
    VkImageLayout image_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkSampler sampler_ = VK_NULL_HANDLE;
    PixelStorage pixel_storage_;
    uint3 res_{};
    VulkanDevice* device_ = nullptr;
    //Image *image_resource_ = nullptr;
    VkFormat image_format_;
    uint32_t mip_levels_ = 1;
    VkDeviceMemory image_memory_ = VK_NULL_HANDLE;
    TextureSampler texture_sampler_;
public:
    VulkanTexture(VulkanDevice *device, Image *image, const TextureViewCreation& texture_view, const TextureSampler& sampler);
    VulkanTexture(VulkanDevice *device, uint32_t width, uint32_t height, uint32_t depth, PixelStorage format, const TextureViewCreation &texture_view, 
        const TextureSampler& sampler, uint4 default_color);
    ~VulkanTexture() override;
    void init(Image *image, const TextureViewCreation &texture_view);
    void load_cpu_data(Image *image);
    void generate_mipmaps();
    void create_image_view(const TextureViewCreation &texture_view);
    void create_sampler(const TextureSampler& sampler_creation);
    void transition_image_layout(VkImageLayout old_layout, VkImageLayout new_layout);
    [[nodiscard]] uint3 resolution() const noexcept override { return res_; }
    [[nodiscard]] handle_ty array_handle() const noexcept override {
        return reinterpret_cast<handle_ty>(image_);
    }
    [[nodiscard]] const handle_ty *array_handle_ptr() const noexcept override {
        return reinterpret_cast<handle_ty *>(image_);
    }
    [[nodiscard]] handle_ty tex_handle() const noexcept override {
        return reinterpret_cast<handle_ty>(image_);
    }
    [[nodiscard]] const void *handle_ptr() const noexcept override {
        return &image_;
    }
    [[nodiscard]] size_t data_size() const noexcept override;
    [[nodiscard]] size_t data_alignment() const noexcept override;
    [[nodiscard]] size_t max_member_size() const noexcept override;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept override { return pixel_storage_; }
    [[nodiscard]] const TextureSampler* get_sampler_pointer() const noexcept { return &texture_sampler_; }
    uint32_t mip_levels() const noexcept { return mip_levels_; }
    uint32_t width() const noexcept { return res_.x; }
    uint32_t height() const noexcept { return res_.y; }
    uint32_t depth() const noexcept { return res_.z; }

    void set_image_layout(VkImageLayout new_image_layout)
    {
        image_layout_ = new_image_layout;
    }

    VkDescriptorImageInfo get_descriptor_info() const
    {
        VkDescriptorImageInfo image_info{};
        image_info.imageLayout = image_layout_;
        image_info.imageView = image_view_;
        image_info.sampler = sampler_;
        return image_info;
    }
};
}// namespace ocarina