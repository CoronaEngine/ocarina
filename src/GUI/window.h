//
// Created by Zero on 19/10/2022.
//

#pragma once

#include "decl.h"
#include "widgets.h"

namespace ocarina {
class OC_GUI_API Window {
public:
    using MouseButtonCallback = ocarina::function<void(int /* button */, int /* action */, float2 /* (x, y) */)>;
    using CursorPositionCallback = ocarina::function<void(float2 /* (x, y) */)>;
    using WindowSizeCallback = ocarina::function<void(uint2 /* (width, height) */)>;
    using KeyCallback = ocarina::function<void(int /* key */, int /* action */)>;
    using ScrollCallback = ocarina::function<void(float2 /* (dx, dy) */)>;
    using UpdateCallback = ocarina::function<void(double)>;
    using BeginFrame = ocarina::function<void()>;
    using EndFrame = ocarina::function<void()>;

protected:
    MouseButtonCallback mouse_button_callback_;
    CursorPositionCallback cursor_position_callback_;
    WindowSizeCallback window_size_callback_;
    KeyCallback key_callback_;
    ScrollCallback scroll_callback_;
    BeginFrame begin_frame_callback_;
    EndFrame end_frame_callback_;
    float4 clear_color_{make_float4(0, 0, 0, 0)};
    bool resizable_{false};
    Clock clock_;
    double dt_{};
    unique_ptr<Widgets> widgets_{};
    uint64_t window_handle_ = InvalidUI64;

protected:
    virtual void _begin_frame() noexcept;
    virtual void _end_frame() noexcept;

public:
    explicit Window(bool resizable = false) noexcept;
    virtual void init(const char *name, uint2 initial_size, bool resizable) noexcept = 0;
    virtual void init_widgets() noexcept = 0;
    Window(Window &&) noexcept = delete;
    virtual void full_screen() = 0;
    virtual void swap_monitor() = 0;
    Window(const Window &) noexcept = delete;
    Window &operator=(Window &&) noexcept = delete;
    Window &operator=(const Window &) noexcept = delete;
    virtual ~Window() noexcept = default;
    [[nodiscard]] Widgets *widgets() noexcept { return widgets_.get(); }
    [[nodiscard]] const Widgets *widgets() const noexcept { return widgets_.get(); }
    [[nodiscard]] double dt() const noexcept { return dt_; }
    [[nodiscard]] virtual uint2 size() const noexcept = 0;
    [[nodiscard]] virtual bool should_close() const noexcept = 0;
    [[nodiscard]] virtual uint shared_texture_handle() const noexcept { OC_ASSERT(0); return 0; };
    [[nodiscard]] explicit operator bool() const noexcept { return !should_close(); }
    [[nodiscard]] uint64_t get_window_handle() const { return window_handle_; }
    virtual Window &set_mouse_callback(MouseButtonCallback cb) noexcept;
    virtual Window &set_cursor_position_callback(CursorPositionCallback cb) noexcept;
    virtual Window &set_window_size_callback(WindowSizeCallback cb) noexcept;
    virtual Window &set_key_callback(KeyCallback cb) noexcept;
    virtual Window &set_scroll_callback(ScrollCallback cb) noexcept;
    virtual Window &set_begin_frame_callback(BeginFrame cb) noexcept;
    virtual Window &set_end_frame_callback(EndFrame cb) noexcept;
    virtual void gen_buffer(uint &handle, uint size_in_byte) const noexcept = 0;
    virtual void bind_buffer(uint &handle, uint size_in_byte) const noexcept = 0;
    virtual void unbind_buffer(uint &handle) const noexcept = 0;
    virtual void set_background(const uchar4 *pixels, uint2 size) noexcept = 0;
    void set_background(const uchar4 *pixels) noexcept {
        set_background(pixels, size());
    }
    virtual void set_background(const Buffer<float4> &buffer, uint2 size) noexcept = 0;
    void set_background(const Buffer<float4> &buffer) noexcept {
        set_background(buffer, size());
    }
    void set_clear_color(float4 color) noexcept { clear_color_ = color; }
    virtual void set_background(const float4 *pixels, uint2 size) noexcept = 0;
    void set_background(const float4 *pixels) noexcept {
        set_background(pixels, size());
    }
    virtual void set_should_close() noexcept = 0;
    virtual void set_size(uint2 size) noexcept = 0;
    virtual void run(UpdateCallback &&draw) noexcept;
    virtual void run_one_frame(UpdateCallback &&draw, double dt) noexcept;
    virtual void run_one_frame(UpdateCallback &&draw) noexcept {
        run_one_frame(OC_FORWARD(draw), 0);
    }
    virtual void show_window() noexcept = 0;
    virtual void hide_window() noexcept = 0;

    class WindowLoop {
    public:
        WindowLoop(Window *window) : window_(window) {
            if (window_) {
                window_->_begin_frame();
            }
        }

        ~WindowLoop() {
            if (window_) {
                window_->_end_frame();
            }
        }

    private:
        Window *window_{nullptr};
    };
};
void dependency_window();

}// namespace ocarina