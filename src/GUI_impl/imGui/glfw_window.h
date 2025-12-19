//
// Created by Zero on 2022/8/16.
//

#pragma once

#include "GUI/window.h"
#include "widgets.h"


namespace ocarina {

class GLTexture;
class GLFWContext;

class GLWindow : public Window {
private:
    ocarina::shared_ptr<GLFWContext> context_;
    GLFWwindow *handle_{nullptr};
    GLFWmonitor *monitor_{nullptr};
    int monitor_index = 0;
    bool isFullscreen = false;
    int windowedX, windowedY, windowedWidth, windowedHeight;
    std::chrono::steady_clock::time_point lastF11Toggle;

    mutable ocarina::unique_ptr<GLTexture> texture_;

private:
    void _begin_frame() noexcept override;
    void _end_frame() noexcept override;

public:
    GLWindow(const char *name, uint2 initial_size, bool resizable = false) noexcept;
    void init(const char *name, uint2 initial_size, bool resizable) noexcept override;
    void init_widgets() noexcept override;
    GLWindow(GLWindow &&) noexcept = delete;
    GLWindow(const GLWindow &) noexcept = delete;
    GLWindow &operator=(GLWindow &&) noexcept = delete;
    GLWindow &operator=(const GLWindow &) noexcept = delete;
    ~GLWindow() noexcept override;
    [[nodiscard]] uint2 size() const noexcept override;
    [[nodiscard]] bool should_close() const noexcept override;
    [[nodiscard]] auto handle() const noexcept { return handle_; }
    void full_screen() override;
    void swap_monitor() override;
    void set_background(const uchar4 *pixels, uint2 size) noexcept override;
    void set_background(const float4 *pixels, uint2 size) noexcept override;
    void gen_buffer(ocarina::uint &handle, ocarina::uint size_in_byte) const noexcept override;
    void bind_buffer(ocarina::uint &handle, ocarina::uint size_in_byte) const noexcept override;
    void unbind_buffer(ocarina::uint &handle) const noexcept override;
    void set_background(const Buffer<ocarina::float4> &buffer, ocarina::uint2 size) noexcept override;
    void set_should_close() noexcept override;
    void set_size(uint2 size) noexcept override;
    void show_window() noexcept override;
    void hide_window() noexcept override;
};
}// namespace ocarina