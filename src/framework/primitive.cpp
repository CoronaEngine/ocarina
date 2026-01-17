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


namespace ocarina {


Primitive::~Primitive() {
    if (vertex_buffer_) {
        ocarina::delete_with_allocator<VertexBuffer>(vertex_buffer_);
    }

    if (index_buffer_) {
        ocarina::delete_with_allocator<IndexBuffer>(index_buffer_);
    }
}

void Primitive::set_geometry_data_setup(Device *device, GeometryDataSetup setup) {
    geometry_data_setup_ = setup;
    if (geometry_data_setup_) {
        geometry_data_setup_(*this);
    }

    //descriptor_sets_.clear();
    //void *shaders[2] = {reinterpret_cast<void *>(vertex_shader_), reinterpret_cast<void *>(pixel_shader_)};
    //std::array<DescriptorSetLayout *, max_descriptor_sets_per_shader> descriptor_set_layouts = device->create_descriptor_set_layout(shaders, 2);
    //for (size_t i = 0; i < max_descriptor_sets_per_shader; ++i) {
    //    if (descriptor_set_layouts[i] && !descriptor_set_layouts[i]->is_global_ubo()) {
    //        add_descriptor_set(descriptor_set_layouts[i]->allocate_descriptor_set());
    //    }
    //}
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
    update_descriptor_sets(device);
    if (pipeline_state_dirty)
    {
        pipeline_ = device->get_pipeline(pipeline_state_, render_pass);
        pipeline_state_dirty = false;
    }
    item_.pipeline_state = &pipeline_state_;
    item_.index_buffer = index_buffer_;
    //item.descriptor_set_writer = descriptor_set_writer_;
    //item.world_matrix = world_matrix_;
    if (item_.push_constant_data == nullptr) {
        Shader<>::Impl *vertex_shader_impl = reinterpret_cast<Shader<>::Impl *>(vertex_shader_);
        item_.push_constant_data = ocarina::allocate(vertex_shader_impl->get_push_constant_size());
        item_.push_constant_size = vertex_shader_impl->get_push_constant_size();
    }
    if (item_.push_constant_data != nullptr) {
        memcpy(item_.push_constant_data, &world_matrix_, sizeof(float4x4));
    }
    item_.pre_render_function = drawcall_pre_draw_function_;
    item_.descriptor_set_count = descriptor_sets_.size();
    for (size_t i = 0; i < max_descriptor_sets_per_shader; ++i) {
        if (i < descriptor_sets_.size() && !descriptor_sets_[i]->is_global()) {
            item_.descriptor_sets[i] = descriptor_sets_[i];
        } else {
            item_.descriptor_sets[i] = nullptr;
        }
    }

    item_.pipeline = pipeline_;

    return item_;
}

void Primitive::add_texture(uint64_t name_id, Texture *texture) {
    textures_.insert(std::make_pair(name_id, texture));

    for (auto &descriptor_set : descriptor_sets_) {
        descriptor_set->update_texture(name_id, texture);
    }
}

void Primitive::update_descriptor_sets(Device *device) {
    if (shader_dirty) {
        for (auto &descriptor_set : descriptor_sets_) {
            ocarina::delete_with_allocator<DescriptorSet>(descriptor_set);
        }
        descriptor_sets_.clear();

        void *shaders[2] = {reinterpret_cast<void *>(vertex_shader_), reinterpret_cast<void *>(pixel_shader_)};
        std::array<DescriptorSetLayout *, max_descriptor_sets_per_shader> descriptor_set_layouts = device->create_descriptor_set_layout(shaders, 2);
        for (size_t i = 1; i < max_descriptor_sets_per_shader; ++i) {
            if (descriptor_set_layouts[i]/* && !descriptor_set_layouts[i]->is_global_ubo()*/) {
                add_descriptor_set(descriptor_set_layouts[i]->allocate_descriptor_set());
            }
        }

        for (auto &descriptor_set : descriptor_sets_) {
            for (auto &tex_pair : textures_) {
                descriptor_set->update_texture(tex_pair.first, tex_pair.second);
            }
        }
        shader_dirty = false;
    }
}
}// namespace ocarina