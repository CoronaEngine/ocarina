
#include "vulkan_device.h"
#include "vulkan_descriptorset.h"
#include "vulkan_shader.h"
#include "util.h"
#include "vulkan_driver.h"
#include "vulkan_descriptorset_writer.h"

namespace ocarina {

VulkanDescriptorSet::VulkanDescriptorSet(VulkanDevice *device, VulkanDescriptorSetLayout *layout, VkDescriptorSet descriptor_set) : layout_(layout), 
device_(device), descriptor_set_(descriptor_set) {
    
    //set_is_global(layout->is_global_ubo());
    // Might want to create a "DescriptorPoolManager" class that handles this case, and builds
    // a new pool whenever an old pool fills up. But this is beyond our current scope

    writer_ = ocarina::new_with_allocator<VulkanDescriptorSetWriter>(device, this);
}

VulkanDescriptorSet::~VulkanDescriptorSet() {
    if (writer_) {
        ocarina::delete_with_allocator(writer_);
        writer_ = nullptr;
    }

    if (descriptor_set_ != VK_NULL_HANDLE) {
        layout_->free_descriptor_set(descriptor_set_);
    }
}

void VulkanDescriptorSet::update_buffer(uint64_t name_id, void *data, uint32_t size) {
    if (writer_) {
        writer_->update_buffer(name_id, data, size);
    }
}

void VulkanDescriptorSet::update_texture(uint64_t name_id, Texture *texture) {
    //if bindless texture, we need to update the descriptor set
    VulkanShaderVariableBinding* binding = layout_->get_binding_by_nameid(name_id);
    if (binding)
    {
        if (writer_) {
            writer_->update_texture(name_id, texture);
        }
        
    }
}

void VulkanDescriptorSet::update_sampler(uint64_t name_id, const TextureSampler& sampler) {
    VulkanShaderVariableBinding* binding = layout_->get_binding_by_nameid(name_id);
    if (binding)
    {
        if (writer_) {
            VkSampler vk_sampler = VulkanDriver::instance().get_vulkan_sampler(sampler);
            writer_->update_sampler(name_id, vk_sampler);
        }
    }
}

uint32_t VulkanDescriptorSet::update_bindless_texture(uint64_t name_id, Texture *texture) {
    size_t binding_count = layout_->get_bindings_count();
    for (size_t i = 0; i < binding_count; ++i)
    {
        VulkanShaderVariableBinding *binding = layout_->get_binding(i);
        if (binding && binding->is_bindless && binding->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            if (writer_) {
                return writer_->update_bindless_texture(name_id, texture);
            }
        }
    }
    return InvalidUI32;
}

VulkanDescriptorSetLayout::VulkanDescriptorSetLayout(VulkanDevice *device, uint8_t descriptor_set_index) : device_(device), descriptor_set_index_(descriptor_set_index) {

}

VulkanDescriptorSetLayout::~VulkanDescriptorSetLayout()
{
    for (auto& descriptor_set : allocated_descriptor_sets_)
    {
        if (descriptor_set) {
            ocarina::delete_with_allocator<VulkanDescriptorSet>(descriptor_set);
        }
    }
    allocated_descriptor_sets_.clear();
    if (descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_->logicalDevice(), descriptor_pool_, nullptr);
    }

    if (layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_->logicalDevice(), layout_, nullptr);
    }
}

void VulkanDescriptorSetLayout::add_binding(const char* name,
    uint32_t binding,
    VkDescriptorType descriptor_type,
    VkShaderStageFlags stage_flags,
    bool is_bindless,
    uint32_t size,
    uint32_t count)
{
    char *end;
    uint64_t nameid = hash64(name);

    if (name_to_bindings_.find(nameid) == name_to_bindings_.end())
    {
        name_to_bindings_.insert(std::make_pair(nameid, bindings_.size()));
        VulkanShaderVariableBinding descriptor_binding;
        strcpy(descriptor_binding.name, name);
        descriptor_binding.binding = binding;
        descriptor_binding.type = descriptor_type;
        descriptor_binding.shader_stage = stage_flags;
        descriptor_binding.count = count;
        descriptor_binding.size = size;
        descriptor_binding.is_bindless = is_bindless;

        if (is_bindless)
        {
            has_bindless_ = true;
        }

        //bindings_.insert(std::make_pair(binding, descriptor_binding));
        bindings_.push_back(descriptor_binding);

        switch (descriptor_type)
        {
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            descriptor_count_.srv += count;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            descriptor_count_.ubo += count;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            descriptor_count_.uav += count;
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            descriptor_count_.samplers += count;
            break;
        }

        layout_built_ = false;
        hashkey_ = InvalidUI64;
    }
}

bool VulkanDescriptorSetLayout::build_layout()
{
    if (layout_built_)
        return false;
    if (layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_->logicalDevice(), layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }

    std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
    std::vector<VkDescriptorBindingFlags> binding_flags;
    std::vector < VkDescriptorPoolSize> pool_sizes;
    uint32_t binding_flag = 0;
    layout_bindings.reserve(bindings_.size());
    pool_sizes.reserve(bindings_.size());
    for (auto& it : bindings_)
    {
        VkDescriptorSetLayoutBinding descriptor_binding{};
        descriptor_binding.binding = it.binding;
        descriptor_binding.descriptorType = it.type;
        descriptor_binding.stageFlags = it.shader_stage;
        descriptor_binding.descriptorCount = it.count; 
        layout_bindings.push_back(descriptor_binding);
        pool_sizes.push_back({
            .type = it.type,
            .descriptorCount = it.count
            });

        if (it.is_bindless) {
            binding_flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
                | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
            //| VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT);

            binding_flag |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT;
        } 
        else if (has_bindless_)
        {
            binding_flags.push_back(0);
        }
    }

    
    VkDescriptorSetLayoutBindingFlagsCreateInfoEXT binding_flags_info{};
    if (has_bindless_ && !binding_flags.empty())
    {
        binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
        binding_flags_info.bindingCount = bindings_.size();
        binding_flags_info.pBindingFlags = binding_flags.data();
    }

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.pNext = nullptr;
    info.bindingCount = (uint32_t)layout_bindings.size();
    info.pBindings = layout_bindings.data();
    info.pNext = !binding_flags.empty() ? &binding_flags_info : nullptr;
    info.flags = binding_flag;

    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_->logicalDevice(), &info, nullptr, &layout_));

    VkDescriptorPoolCreateInfo pool_create_info{};
    pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_create_info.poolSizeCount = pool_sizes.size();
    pool_create_info.pPoolSizes = pool_sizes.size() == 0 ? nullptr : pool_sizes.data();
    pool_create_info.maxSets = std::max((uint32_t)pool_sizes.size(), (uint32_t)1);
    pool_create_info.flags = free_descriptor_set_ ? VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT : 0;

    if (info.flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT) {
        pool_create_info.flags |= VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT;
    }

    VK_CHECK_RESULT(vkCreateDescriptorPool(device_->logicalDevice(), &pool_create_info, nullptr, &descriptor_pool_));
    pool_max_sets_count_ = pool_create_info.maxSets;
    layout_built_ = true;
    return true;
}

DescriptorSet *VulkanDescriptorSetLayout::allocate_descriptor_set() {
    allocated_sets_count_++;
    if (allocated_sets_count_ > pool_max_sets_count_ && descriptor_pool_ != VK_NULL_HANDLE)
    {
        assert(false);
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptor_pool_;
    VkDescriptorSetLayout layouts[1] = {layout_};
    allocInfo.pSetLayouts = layouts;
    allocInfo.descriptorSetCount = 1;

    if (has_bindless_)
    {
        uint32_t variableCounts[] = { 1024 };

        VkDescriptorSetVariableDescriptorCountAllocateInfo countInfo{};
        countInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
        countInfo.descriptorSetCount = 1;
        countInfo.pDescriptorCounts = variableCounts;
    }

    VkDescriptorSet descriptor_set;
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device_->logicalDevice(), &allocInfo, &descriptor_set));
    VulkanDescriptorSet* vulkan_descriptor_set = ocarina::new_with_allocator<VulkanDescriptorSet>(device_, this, descriptor_set);
    allocated_descriptor_sets_.push_back(vulkan_descriptor_set);
    return vulkan_descriptor_set;
}

void VulkanDescriptorSetLayout::free_descriptor_set(VkDescriptorSet descriptor_set)
{
    if (free_descriptor_set_ && descriptor_pool_ != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(device_->logicalDevice(), descriptor_pool_, 1, &descriptor_set);
        allocated_sets_count_--;
    }
}

VulkanShaderVariableBinding* VulkanDescriptorSetLayout::get_binding(uint64_t index) {
    //auto it = bindings_.find(binding);
    //if (it != bindings_.end()) {
    //    return &it->second;
    //}
    if (index >= bindings_.size())
        return nullptr;
    return &bindings_[index];
}

VulkanShaderVariableBinding *VulkanDescriptorSetLayout::get_binding_by_nameid(uint64_t name_id) {
    auto it = name_to_bindings_.find(name_id);
    if (it != name_to_bindings_.end()) {
        uint32_t binding_index = it->second;
        return get_binding(binding_index);
    }
    return nullptr;
}

VulkanDescriptorPool::VulkanDescriptorPool(const DescriptorPoolCreation &creation, VulkanDevice *device) : device_(device), descriptor_pool_creation_(creation) {
    VkDescriptorPoolSize sizes[4];
    uint8_t npools = 0;

    if (creation.ubo > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = creation.ubo};
    }

    if (creation.srv > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = creation.srv};
    }

    if (creation.uav > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = creation.uav};
    }

    if (creation.samplers > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = creation.samplers};
    }

    VkDescriptorPoolCreateInfo info;
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = npools;
    info.pPoolSizes = sizes;
    info.maxSets = 1;

    info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    vkCreateDescriptorPool(device_->logicalDevice(), &info, nullptr, &descriptor_pool_);
}

VulkanDescriptorPool::~VulkanDescriptorPool() {
    descriptor_sets.clear();
    vkDestroyDescriptorPool(device_->logicalDevice(), descriptor_pool_, nullptr);
}

VkDescriptorSet VulkanDescriptorPool::get_descriptor_set(VkDescriptorSetLayout layout) {
    auto it = descriptor_sets.find(layout);
    if (it != descriptor_sets.end())
    {
        return it->second;
    }
    
    // Creating a new set
    VkDescriptorSetLayout layouts[1] = {layout};
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = descriptor_pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = layouts,
    };
    VkDescriptorSet vkSet;
    VkResult result = vkAllocateDescriptorSets(device_->logicalDevice(), &allocInfo, &vkSet);
    descriptor_sets.insert(std::make_pair(layout, vkSet));
    return vkSet;
}


std::array<DescriptorSetLayout*, MAX_DESCRIPTOR_SETS_PER_SHADER> VulkanDescriptorManager::create_descriptor_set_layout(VulkanShader **shaders, uint32_t shaders_count) {

    std::array<DescriptorSetLayout*, MAX_DESCRIPTOR_SETS_PER_SHADER> descriptor_set_layouts = {};

    for (auto& layout_key : cached_descriptor_set_layout_keys_) {
        layout_key.bindings.clear();
    }

    for (uint32_t i = 0; i < shaders_count; ++i)
    {
        VulkanShader* shader = shaders[i];

        uint32_t variables_count = shader->get_shader_variables_count();
        for (uint32_t j = 0; j < variables_count; ++j)
        {
            const VulkanShaderVariableBinding& binding = shader->get_shader_variable(j);
            cached_descriptor_set_layout_keys_[binding.descriptor_set].add_binding(binding);
        }
    }

    for (auto& layout_key : cached_descriptor_set_layout_keys_) {
        layout_key.normalize();
        uint64_t hashkey = layout_key.generate_hash();
        auto it = descriptor_set_layouts_.find(hashkey);
        if (it != descriptor_set_layouts_.end()) {
            descriptor_set_layouts[it->second->get_descriptor_set_index()] = it->second;
        }
        else if (layout_key.bindings.size() > 0)
        {
            VulkanDescriptorSetLayout* layout = ocarina::new_with_allocator<VulkanDescriptorSetLayout>(device_, layout_key.bindings[0].descriptor_set);
            for (const auto& binding : layout_key.bindings) {
                layout->add_binding(binding.name, binding.binding, binding.type, binding.shader_stage, binding.is_bindless, binding.size, binding.count);
                layout->set_name(binding.name);
            }
            descriptor_set_layouts[layout->get_descriptor_set_index()] = layout;
            descriptor_set_layouts_.insert(std::make_pair(hashkey, layout));
        }
    }

    uint32_t layouts_count = 0;
    for (size_t i = 0; i < descriptor_set_layouts.size(); ++i)
    {
        VulkanDescriptorSetLayout* layout = static_cast<VulkanDescriptorSetLayout*>(descriptor_set_layouts[i]);
        if (layout != nullptr)
        {
            layouts_count++;
        }
        if (layout != nullptr && layout->build_layout())
        {
            if (i == (size_t)DescriptorSetIndex::GLOBAL_SET)
            {
                VulkanDescriptorSet *global_descriptor_set = static_cast<VulkanDescriptorSet *>(layout->allocate_descriptor_set());
                size_t bindings_count = layout->get_bindings_count(); 

                for (size_t j = 0; j < bindings_count; ++j)
                {
                    VulkanShaderVariableBinding* binding = layout->get_binding(j);
                    VulkanDriver::instance().add_global_descriptor_set(hash64(binding->name), global_descriptor_set);
                }
            }
        }
    }

    /*
    //supplyment the empty layout
    uint32_t descriptor_set_layouts_num = 0;
    uint32_t empty_descriptor_sets_num = 0;
    for (uint32_t i = 0; i < MAX_DESCRIPTOR_SETS_PER_SHADER; ++i) {
        if (descriptor_set_layouts[i] == nullptr && descriptor_set_layouts_num < layouts_count) {
            descriptor_set_layouts[i] = get_empty_descriptor_set_layout();
            empty_descriptor_sets_num++;
        }
        else
        {
            descriptor_set_layouts_num++;
        }

        if (descriptor_set_layouts_num >= layouts_count) {
            break;
        }
    }
    */

    return descriptor_set_layouts;
}

void VulkanDescriptorManager::clear()
{
    //pools_.clear();
    for (auto layout : descriptor_set_layouts_)
    {
        VulkanDescriptorSetLayout* descriptor_set_layout = layout.second;
        ocarina::delete_with_allocator<VulkanDescriptorSetLayout>(descriptor_set_layout);
    }

    global_descriptor_set_layouts_ = nullptr;
    bindless_descriptor_set_layouts_ = nullptr;

    //if (empty_descriptor_set_layout_)
    //{
    //    ocarina::delete_with_allocator<VulkanDescriptorSetLayout>(empty_descriptor_set_layout_);
    //    empty_descriptor_set_layout_ = nullptr;
    //}
    //descriptor_set_layouts_.clear();
}

//VulkanDescriptorSetLayout* VulkanDescriptorManager::get_empty_descriptor_set_layout()
//{
//    if (empty_descriptor_set_layout_ == nullptr)
//    {
//        empty_descriptor_set_layout_ = ocarina::new_with_allocator<VulkanDescriptorSetLayout>(device_, 0);
//        empty_descriptor_set_layout_->build_layout();
//        descriptor_set_layouts_.insert(std::make_pair(0, empty_descriptor_set_layout_));
//    }
//    return empty_descriptor_set_layout_;
//}

}// namespace ocarina


