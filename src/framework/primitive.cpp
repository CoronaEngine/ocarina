//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "primitive.h"
#include "rhi/vertex_buffer.h"
#include "rhi/index_buffer.h"
#include "rhi/device.h"
#include "rhi/descriptor_set.h"
#include "rhi/resources/shader.h"
#include "rhi/resources/texture_sampler.h"


namespace ocarina {


Primitive::~Primitive() {
    if (vertex_buffer_) {
        ocarina::delete_with_allocator<VertexBuffer>(vertex_buffer_);
    }

    if (index_buffer_) {
        ocarina::delete_with_allocator<IndexBuffer>(index_buffer_);
    }

    if (item_.push_constant_data)
    {
        ocarina::deallocate(item_.push_constant_data);
        item_.push_constant_data = nullptr;
    }
}

void Primitive::set_geometry_data_setup(Device *device, GeometryDataSetup setup) {
    geometry_data_setup_ = setup;
    if (geometry_data_setup_) {
        geometry_data_setup_(*this);
    }

    update_descriptor_sets(device);
    pipeline_state_dirty = true;
}

void Primitive::set_vertex_buffer(VertexBuffer *vertex_buffer) {
    vertex_buffer_ = vertex_buffer;
    pipeline_state_.vertex_buffer = vertex_buffer;
}

void Primitive::set_index_buffer(IndexBuffer *index_buffer) {
    index_buffer_ = index_buffer;
}

void Primitive::set_vertex_shader(handle_ty vertex_shader) {
    if (vertex_shader_ != vertex_shader) {
        shader_dirty = true;
    }
    vertex_shader_ = vertex_shader;
    pipeline_state_.shaders[0] = vertex_shader;
}

void Primitive::set_pixel_shader(handle_ty pixel_shader) {
    if (pixel_shader_ != pixel_shader) {
        shader_dirty = true;
    }
    pixel_shader_ = pixel_shader;
    pipeline_state_.shaders[1] = pixel_shader;
}

void Primitive::add_descriptor_set(DescriptorSet *descriptor_set) {
    auto it = std::find(descriptor_sets_.begin(), descriptor_sets_.end(), descriptor_set);
    if (it == descriptor_sets_.end()) {
        descriptor_sets_.push_back(descriptor_set);
    }
}

DrawCallItem Primitive::get_draw_call_item(Device *device, RHIRenderPass *render_pass) {
    if (update_push_constant_function_ != nullptr) {
        update_push_constant_function_(*this);
    }
    update_descriptor_sets(device);
    if (pipeline_state_dirty)
    {
        pipeline_ = device->get_pipeline(pipeline_state_, render_pass);
        pipeline_state_dirty = false;
    }
    item_.pipeline_state = &pipeline_state_;
    item_.index_buffer = index_buffer_;

    uint32_t push_constant_size = pipeline_->push_constant_size;
    if (push_constant_data_ == nullptr && push_constant_size > 0) {
        push_constant_data_ = ocarina::allocate(push_constant_size);
        memset(push_constant_data_, 0, push_constant_size);
    }

    item_.push_constant_size = pipeline_->push_constant_size;
    if (item_.push_constant_size > 0) {
        if (item_.push_constant_data == nullptr) {

            item_.push_constant_data = ocarina::allocate(item_.push_constant_size);
        }
    }
    if (item_.push_constant_data != nullptr) {
        //memcpy(item_.push_constant_data, &world_matrix_, sizeof(float4x4));
        memcpy(item_.push_constant_data, push_constant_data_, item_.push_constant_size);
    }

    item_.pre_render_function = drawcall_pre_draw_function_;
    //item_.descriptor_set_count = descriptor_sets_.size();
    //for (size_t i = 0; i < MAX_DESCRIPTOR_SETS_PER_SHADER; ++i) {
    //    if (i < descriptor_sets_.size()) {
    //        item_.descriptor_sets[i] = descriptor_sets_[i];
    //    } else {
    //        item_.descriptor_sets[i] = nullptr;
    //    }
    //}

    item_.pipeline = pipeline_;

    return item_;
}

void Primitive::add_texture(uint64_t name_id, Texture *texture) {
    
    uint32_t bindless_index = InvalidUI32;
    for (auto &descriptor_set : descriptor_sets_) {
        descriptor_set->update_texture(name_id, texture);
    }
    TextureHandle texture_handle{bindless_index, texture};
    textures_.insert(std::make_pair(name_id, texture_handle));
}

void Primitive::add_bindless_texture(uint64_t name_id, Texture *texture) {
    uint32_t bindless_index = InvalidUI32;
    for (auto &descriptor_set : descriptor_sets_) {
        bindless_index = descriptor_set->update_bindless_texture(name_id, texture);
    }
    TextureHandle texture_handle{bindless_index, texture};
    bindless_textures_.insert(std::make_pair(name_id, texture_handle));
}

void Primitive::add_sampler(uint64_t name_id, const TextureSampler& sampler)
{
    samplers_.insert(std::make_pair(name_id, sampler));
}

void Primitive::update_descriptor_sets(Device *device) {
    if (shader_dirty) {
        for (auto &descriptor_set : descriptor_sets_) {
            ocarina::delete_with_allocator<DescriptorSet>(descriptor_set);
        }
        descriptor_sets_.clear();
        item_.descriptor_sets.clear();
        void *shaders[2] = {reinterpret_cast<void *>(vertex_shader_), reinterpret_cast<void *>(pixel_shader_)};
        std::array<DescriptorSetLayout *, MAX_DESCRIPTOR_SETS_PER_SHADER> descriptor_set_layouts = device->create_descriptor_set_layout(shaders, 2);
        uint32_t group_index = 0;
        for (size_t i = 1; i < MAX_DESCRIPTOR_SETS_PER_SHADER; ++i) {
            if (descriptor_set_layouts[i]/* && !descriptor_set_layouts[i]->is_global_ubo()*/) {
                DescriptorSet* descriptor_set = descriptor_set_layouts[i]->allocate_descriptor_set();
                add_descriptor_set(descriptor_set);
                if (item_.descriptor_sets.empty())
                {
                    item_.first_set = i;
                }
                item_.descriptor_sets.push_back(descriptor_set);
                /*
                if (item_.descriptor_sets_binding_group[group_index].descriptor_sets.empty())
                {
                    
                    item_.descriptor_sets_binding_group[group_index].first_set = i;
                    item_.descriptor_sets_binding_group[group_index].descriptor_set_count = 1;
                    item_.descriptor_set_count++;
                }
                item_.descriptor_sets_binding_group[group_index].descriptor_sets.push_back(descriptor_set);
                
                //check the next
                if (i + 1 < MAX_DESCRIPTOR_SETS_PER_SHADER && descriptor_set_layouts[i + 1])
                {
                    item_.descriptor_sets_binding_group[group_index].descriptor_set_count++;
                }
                */
            }
            else
            {
                group_index++;
            }
        }

        for (auto &descriptor_set : descriptor_sets_) {
            for (auto &bindless_tex_pair : bindless_textures_) {
                descriptor_set->update_bindless_texture(bindless_tex_pair.first, bindless_tex_pair.second.texture_);
            }
            for (auto &tex_pair : textures_) {
                descriptor_set->update_texture(tex_pair.first, tex_pair.second.texture_);
            }
            for (auto& sampler_pair : samplers_) {
                descriptor_set->update_sampler(sampler_pair.first, sampler_pair.second);
            }
        }
        shader_dirty = false;
    }
}

void Primitive::set_push_constant_variable(uint64_t name_id, const std::byte *data, size_t size) {
    if (push_constant_data_ == nullptr) {
        return;
    }
    auto it = pipeline_->push_constant_variables_.find(name_id);
    if (it != pipeline_->push_constant_variables_.end()) {
        memcpy(push_constant_data_ + it->second.offset, data, size);
    } 
}
}// namespace ocarina