//
// Created by Zero on 2022/8/16.
//


#include "core/stl.h"
#include "dsl/dsl.h"
//#include "util/file_manager.h"
#include "rhi/common.h"
#include "rhi/context.h"
#include <windows.h>
#include "math/base.h"
//#include "util/image.h"
#include "rhi/vertex_buffer.h"
#include "rhi/index_buffer.h"
#include "rhi/resources/buffer.h"
#include "rhi/resources/texture.h"
#include "GUI/window.h"
#include "framework/renderer.h"
#include "framework/primitive.h"
#include "rhi/descriptor_set.h"
#include "rhi/renderpass.h"
#include "framework/camera.h"

using namespace ocarina;


struct GlobalUniformBuffer {
    math3d::Matrix4 projection_matrix;
    math3d::Matrix4 view_matrix;
};

struct PushConstants {
    float4x4 world_matrix;
    uint32_t albedo_index;
};


int main(int argc, char *argv[]) {

    fs::path path(argv[0]);
    //FileManager &file_manager = FileManager::instance();
    RHIContext &file_manager = RHIContext::instance();

    auto window = file_manager.create_window("display", make_uint2(800, 600), WindowLibrary::SDL3);

    InstanceCreation instanceCreation = {};
    //instanceCreation.instanceExtentions =
    instanceCreation.windowHandle = window->get_window_handle();
    Device device = file_manager.create_device("vulkan", instanceCreation);

    //Shader
    std::set<string> options;
    handle_ty vertex_shader = device.create_shader_from_file("D:\\github\\Vision\\src\\ocarina\\src\\backends\\vulkan\\builtin\\texture.vert", ShaderType::VertexShader, options);
    handle_ty pixel_shader = device.create_shader_from_file("D:\\github\\Vision\\src\\ocarina\\src\\backends\\vulkan\\builtin\\bindless_texture.frag", ShaderType::PixelShader, options);

    Primitive quad;
    PipelineState pipeline_state;
    pipeline_state.shaders[0] = vertex_shader;
    pipeline_state.shaders[1] = pixel_shader;
    pipeline_state.blend_state = BlendState::Opaque();
    pipeline_state.raster_state = RasterState::Default();
    pipeline_state.depth_stencil_state = DepthStencilState::Default();
    
    Image image = Image::load("D:\\Github\\Vision\\src\\ocarina\\res\\textures\\granite.png", ColorSpace::SRGB);
    TextureViewCreation texture_view = {};
    texture_view.mip_level_count = 0;
    texture_view.usage = TextureUsageFlags::ShaderReadOnly;
    TextureSampler sampler{ TextureSampler::Filter::LINEAR_LINEAR, TextureSampler::Address::REPEAT };
    Texture texture = device.create_texture(&image, texture_view, sampler);

    uint64_t albedo_name_id = hash64("albedo");

    auto setup_quad = [&](Primitive& quad) {
        quad.set_vertex_shader(vertex_shader);
        quad.set_pixel_shader(pixel_shader);
        
        VertexBuffer* vertex_buffer = device.create_vertex_buffer();
        Vector3 positions[4] = {{1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}};
        vertex_buffer->add_vertex_stream(VertexAttributeType::Enum::Position, 4, sizeof(Vector3), (const void *)&positions[0]);
        Vector2 uvs[4] = {{1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}};
        vertex_buffer->add_vertex_stream(VertexAttributeType::Enum::TexCoord0, 4, sizeof(Vector2), (const void *)&uvs[0]);
        Vector4 colors[4] = {{1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};
        vertex_buffer->add_vertex_stream(VertexAttributeType::Enum::Color0, 4, sizeof(Vector4), (const void *)&colors[0]);
        vertex_buffer->upload_data();
        quad.set_vertex_buffer(vertex_buffer);
        pipeline_state.vertex_buffer = vertex_buffer;

        // Setup indices
        std::vector<uint16_t> indices{ 0, 1, 2, 2, 3, 0 };
        uint32_t indices_count = static_cast<uint32_t>(indices.size());
        uint32_t indices_bytes = indices_count * sizeof(uint16_t);
        IndexBuffer *index_buffer = device.create_index_buffer(indices.data(), indices_count);
        quad.set_index_buffer(index_buffer);
        quad.set_pipeline_state(pipeline_state);

        
        quad.add_bindless_texture(albedo_name_id, &texture);
        uint64_t name_id = hash64("sampler_albedo");
        quad.add_sampler(name_id, *texture.get_sampler_pointer());
    };

    Camera camera;
    camera.set_aspect_ratio(800.0f / 600.0f);
    camera.set_position({0.0f, 0.0f, -2.5f});
    camera.set_target({0.0f, 0.0f, 0.0f});

    auto setup_renderer = [&]() {
        // Setup renderer if needed
        };

    auto release_renderer = [&]() {
    };

    uint64_t push_constant_name_id = hash64("PushConstants");

    auto pre_render_draw_item = [&](const DrawCallItem &item) {
        //update push constants before draw
        //item.descriptor_set_writer->update_push_constants(push_constant_name_id, (void *)&item.world_matrix, sizeof(item.world_matrix), item.pipeline_line);
    };

    PushConstants push_constants_data = {};

    uint64_t name_id_world_matrix = hash64("modelMatrix");
    uint64_t name_id_albedo_index = hash64("albedo_index");

    auto update_push_constant = [&](Primitive &primitive) {
        // Setup push constant data if needed
        push_constants_data.world_matrix = primitive.get_world_matrix();
        primitive.set_push_constant_variable(
            name_id_world_matrix,
            reinterpret_cast<std::byte *>(const_cast<void *>(static_cast<const void *>(&primitive.get_world_matrix()))),
            sizeof(primitive.get_world_matrix()));
        Primitive::TextureHandle albedo_texture_handle = primitive.get_texture_handle(albedo_name_id);
        push_constants_data.albedo_index = albedo_texture_handle.bindless_index_;
        //primitive.set_push_constant_data(reinterpret_cast<const std::byte *>(&push_constants_data));
        primitive.set_push_constant_variable(
            name_id_albedo_index,
            reinterpret_cast<std::byte *>(&albedo_texture_handle.bindless_index_),
            sizeof(push_constants_data.albedo_index));
    };

    quad.set_geometry_data_setup(&device, setup_quad);
    quad.set_draw_call_pre_render_function(pre_render_draw_item);
    quad.set_update_push_constant_function(update_push_constant);

    Renderer renderer(&device);

    RenderPassCreation render_pass_creation;
    render_pass_creation.swapchain_clear_color = make_float4(0.1f, 0.1f, 0.1f, 1.0f);
    render_pass_creation.swapchain_clear_depth = 1.0f;
    render_pass_creation.swapchain_clear_stencil = 0;
    RHIRenderPass *render_pass = device.create_render_pass(render_pass_creation);
    void *shaders[2] = {reinterpret_cast<void *>(vertex_shader), reinterpret_cast<void *>(pixel_shader)};
    //DescriptorSetWriter *global_descriptor_set_writer = device.create_descriptor_set_writer(device.get_global_descriptor_set("global_ubo"), shaders, 2);
    DescriptorSet *global_descriptor_set = device.get_global_descriptor_set("global_ubo");
    render_pass->add_global_descriptor_set("global_ubo", global_descriptor_set);
    render_pass->set_begin_renderpass_callback([&](RHIRenderPass *rp) {
        GlobalUniformBuffer global_ubo_data = {camera.get_projection_matrix().transpose(), camera.get_view_matrix().transpose()};
        global_descriptor_set->update_buffer(hash64("global_ubo"), &global_ubo_data, sizeof(GlobalUniformBuffer));
        auto draw_item = quad.get_draw_call_item(&device, rp);
        rp->add_draw_call(draw_item);
    });


    
    renderer.add_render_pass(render_pass);

    auto image_io = Image::pure_color(make_float4(1, 0, 0, 1), ColorSpace::LINEAR, make_uint2(500));
    window->set_background(image_io.pixel_ptr<float4>(), make_uint2(800, 600));
    window->run([&](double d) {
        while (!window->should_close())
        {
            Window::WindowLoop win_loop(window.get());
            renderer.render_frame();
        }
    });

    texture.destroy();
}