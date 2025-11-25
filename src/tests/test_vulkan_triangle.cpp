//
// Created by Zero on 2022/8/16.
//


#include "core/stl.h"
#include "dsl/dsl.h"
//#include "util/context.h"
#include "rhi/common.h"
#include "rhi/context.h"
#include <windows.h>
#include "math/base.h"
//#include "util/image.h"
#include "rhi/vertex_buffer.h"
#include "rhi/index_buffer.h"
#include "rhi/resources/buffer.h"
#include "GUI/window.h"
#include "framework/renderer.h"
#include "framework/primitive.h"
#include "rhi/descriptor_set.h"
#include "rhi/renderpass.h"
#include "framework/camera.h"

using namespace ocarina;

class TestA
{
public:
    TestA() = default;
    explicit TestA(const char *str) {
        int length = strlen(str);
        _a = new char[length + 1];
        _a[length] = 0;
        memcpy(_a, str, length);
    }
    ~TestA()
    {
        if (_a)
        {
            delete[] _a;
            _a = nullptr;
        }
    }

    TestA(const TestA& other) {
        if (_a != nullptr)
        {
            delete[] _a;
        }

        if (other._a != nullptr)
        {
            int length = strlen(other._a);
            _a = new char[length + 1];
            _a[length] = 0;
            memcpy(_a, other._a, length);
        }
        std::cout << "Copy constuctor." << std::endl;
    }
    TestA(TestA&& rvalue) noexcept
    {
        std::cout << "Move constuctor." << std::endl;
        _a = std::move(rvalue._a);
        rvalue._a = nullptr;
    }
    TestA& operator=(const TestA& other)
    {
        if (_a != nullptr) {
            delete[] _a;
        }

        if (other._a != nullptr) {
            int length = strlen(other._a);
            _a = new char[length + 1];
            _a[length] = 0;
            memcpy(_a, other._a, length);
        }
        std::cout << "Copy assignment constuctor." << std::endl;
        return *this;
    }

    TestA &operator=(TestA &&rvalue) noexcept
    {
        _a = std::move(rvalue._a);
        rvalue._a = nullptr;
        std::cout << "Move assignment constuctor." << std::endl;
        return *this;
    }

public: 
    char* _a = nullptr;
};

struct ExampleStruct {
    char a;
    int b;
    short c;
    float d;
};

class TestRef
{
public:
    TestRef() { ranges.resize(1); }
    std::vector<int> ranges;
};

struct TestContainer
{
    TestContainer()
    {
        m_ref = new TestRef();
    }

    ~TestContainer()
    {
        if (m_ref)
        {
            delete m_ref;
        }
    }

    TestRef *m_ref = nullptr;
};

class TestB
{
public:
    TestB(const TestRef &ref) : m_Ref(ref) {}

    const TestRef& GetTestRef()
    {
        return m_Ref;
    }
private:
    const TestRef &m_Ref;
};

struct PerframeUniformBuffer
{
    float4x4 view_matrix;
    float4x4 projection_matrix;
};

struct PerframeUBOWriter
{
    PerframeUniformBuffer per_frame_uniform_buffer = {};
    std::unique_ptr<Buffer<std::byte>> per_frame_buffer_ = nullptr;
    std::unique_ptr<DescriptorSet> per_frame_descriptor_set_ = nullptr;
};

struct GlobalUniformBuffer {
    math3d::Matrix4 projection_matrix;
    math3d::Matrix4 view_matrix;
};

//struct Vector3
//{
//    float x, y, z;
//};
//
//struct Vector4
//{
//    float x, y, z, w;
//};

int main(int argc, char *argv[]) {
    TestContainer *container = new TestContainer();

    TestB testB(*(container->m_ref));

    const TestRef &ref = testB.GetTestRef();

    for (size_t i = 0; i < ref.ranges.size(); ++i)
    {
        std::cout << ref.ranges[i] << std::endl;
    }

    delete container;
    container = nullptr;

    for (size_t i = 0; i < ref.ranges.size(); ++i) {
        std::cout << ref.ranges[i] << std::endl;
    }

    fs::path path(argv[0]);
    //FileManager &context = FileManager::instance();
    RHIContext &context = RHIContext::instance();

    auto window = context.create_window("display", make_uint2(800, 600), WindowLibrary::SDL3);

    InstanceCreation instanceCreation = {};
    //instanceCreation.instanceExtentions =
    instanceCreation.windowHandle = window->get_window_handle();
    Device device = context.create_device("vulkan", instanceCreation);//Device::create_device("vulkan", instanceCreation);

    //Shader
    std::set<string> options;
    handle_ty vertex_shader = device.create_shader_from_file("D:\\github\\Vision\\src\\ocarina\\src\\backends\\vulkan\\builtin\\triangle.vert", ShaderType::VertexShader, options);
    handle_ty pixel_shader = device.create_shader_from_file("D:\\github\\Vision\\src\\ocarina\\src\\backends\\vulkan\\builtin\\triangle.frag", ShaderType::PixelShader, options);
    //void **shaders = reinterpret_cast<void **>(vertex_shader, pixel_shader);
    //device.create_descriptor_set_layout(shaders, 2);

    Primitive triangle;
    //std::vector<Primitive> opaques;
    PipelineState pipeline_state;
    pipeline_state.shaders[0] = vertex_shader;
    pipeline_state.shaders[1] = pixel_shader;
    pipeline_state.blend_state = BlendState::Opaque();
    pipeline_state.raster_state = RasterState::Default();
    pipeline_state.depth_stencil_state = DepthStencilState::Default();

    auto setup_triangle = [&](Primitive& triangle) {
        triangle.set_vertex_shader(vertex_shader);
        triangle.set_pixel_shader(pixel_shader);
        
        VertexBuffer* vertex_buffer = device.create_vertex_buffer();
        Vector3 positions[3] = {{1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}};
        vertex_buffer->add_vertex_stream(VertexAttributeType::Enum::Position, 3, sizeof(Vector3), (const void *)&positions[0]);
        Vector4 colors[3] = {{1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}};
        vertex_buffer->add_vertex_stream(VertexAttributeType::Enum::Color0, 3, sizeof(Vector4), (const void *)&colors[0]);
        vertex_buffer->upload_data();
        triangle.set_vertex_buffer(vertex_buffer);
        pipeline_state.vertex_buffer = vertex_buffer;

        // Setup indices
        std::vector<uint16_t> indices{ 0, 1, 2 };
        uint32_t indices_count = static_cast<uint32_t>(indices.size());
        uint32_t indices_bytes = indices_count * sizeof(uint16_t);
        IndexBuffer *index_buffer = device.create_index_buffer(indices.data(), indices_count);
        triangle.set_index_buffer(index_buffer);
        triangle.set_pipeline_state(pipeline_state);

        
        //opaques.push_back(triangle);
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

    triangle.set_geometry_data_setup(&device, setup_triangle);
    triangle.set_draw_call_pre_render_function(pre_render_draw_item);

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
        //rp->set_clear_color(make_float4(0.1f, 0.1f, 0.1f, 1.0f));
        GlobalUniformBuffer global_ubo_data = {camera.get_projection_matrix().transpose(), camera.get_view_matrix().transpose()};
        global_descriptor_set->update_buffer(hash64("global_ubo"), &global_ubo_data, sizeof(GlobalUniformBuffer));
        //global_descriptor_set_writer->update_buffer(hash64("global_ubo"), &global_ubo_data, sizeof(GlobalUniformBuffer));
    });


    auto draw_item = triangle.get_draw_call_item(&device, render_pass);
    render_pass->add_draw_call(draw_item);
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

    
}