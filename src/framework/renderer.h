//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "core/concepts.h"
#include "core/concurrency/thread_pool.h"
#include "rhi/params.h"
#include "rhi/graphics_descriptions.h"

namespace ocarina {
class Primitive;
class RHIRenderPass;
class Device;

class Renderer : public concepts::Noncopyable {
public:
    Renderer(Device *device) : device_(device) {}
    ~Renderer();

    using UpdateFrameCallBack = ocarina::function<void(double)>;
    using RenderCallback = ocarina::function<void(double)>;
    using SetupCallback = ocarina::function<void()>;
    using ReleaseCallback = ocarina::function<void()>;
    using UpdateDescriptorPerObjectCallback = ocarina::function<void(Primitive&)>;

    void set_update_frame_callback(UpdateFrameCallBack cb)
    {
        update_frame = cb;
    }
    void set_render_callback(RenderCallback cb);
    void set_setup_callback(SetupCallback cb);
    void set_release_callback(ReleaseCallback cb);
    void set_clear_color(const float4& color)
    {
        clear_color = color;
    }

    void render_frame();
    void add_render_pass(RHIRenderPass *render_pass)
    {
        render_passes_.emplace_back(render_pass);
    }

    void remove_render_pass(RHIRenderPass *render_pass)
    {
        render_passes_.remove(render_pass);
    }

private:
    UpdateFrameCallBack update_frame = nullptr;
    SetupCallback setup = nullptr;
    RenderCallback render = nullptr;
    ReleaseCallback release = nullptr;
    float4 clear_color = {0, 0, 0, 1};
    
protected:
    std::list<RHIRenderPass *> render_passes_;
    Device* device_ = nullptr;

    
};

}// namespace ocarina