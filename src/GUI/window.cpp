//
// Created by Zero on 19/10/2022.
//

#include "window.h"

namespace ocarina {

void dependency_window() {}

Window::Window(bool resizable) noexcept
    : resizable_{resizable} {}

Window &Window::set_mouse_callback(Window::MouseButtonCallback cb) noexcept {
    mouse_button_callback_ = std::move(cb);
    return *this;
}

Window &Window::set_cursor_position_callback(Window::CursorPositionCallback cb) noexcept {
    cursor_position_callback_ = std::move(cb);
    return *this;
}

Window &Window::set_window_size_callback(Window::WindowSizeCallback cb) noexcept {
    window_size_callback_ = std::move(cb);
    return *this;
}

Window &Window::set_key_callback(Window::KeyCallback cb) noexcept {
    key_callback_ = std::move(cb);
    return *this;
}

Window &Window::set_scroll_callback(Window::ScrollCallback cb) noexcept {
    scroll_callback_ = std::move(cb);
    return *this;
}

void Window::run_one_frame(Window::UpdateCallback &&draw, double dt) noexcept {
    _begin_frame();
    draw(dt);
    _end_frame();
}

void Window::run(Window::UpdateCallback &&draw) noexcept {
    while (!should_close()) {
        clock_.begin();
        run_one_frame(OC_FORWARD(draw), dt_);
        dt_ = clock_.elapse_s();
    }
}

Window &Window::set_begin_frame_callback(Window::BeginFrame cb) noexcept {
    begin_frame_callback_ = std::move(cb);
    return *this;
}

Window &Window::set_end_frame_callback(Window::EndFrame cb) noexcept {
    end_frame_callback_ = std::move(cb);
    return *this;
}

Window &Window::set_render_callback(Window::RenderCallback cb) noexcept {
    render_callback_ = std::move(cb);
    return *this;
}

void Window::_begin_frame() noexcept {
    if (auto &&cb = begin_frame_callback_) { cb(); }
}

void Window::_end_frame() noexcept {
    if (auto &&cb = end_frame_callback_) { cb(); }
}

}// namespace ocarina