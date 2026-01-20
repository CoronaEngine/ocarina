//
// Created by Zero on 2022/8/16.
//

#include "sdl_window.h"
#include "widgets.h"
#include "ext/imgui/gizmo/ImGuizmo.h"
#include "core/logging.h"
#include "rhi/common.h"
//#include <ext/SDL/include/SDL3/SDL_syswm.h>

namespace ocarina {


void SDLWindow::init_widgets() noexcept {
    widgets_ = make_unique<ImGuiWidgets>(this);
}

void SDLWindow::init(const char *name, uint2 initial_size, bool resizable) noexcept {
    SDL_WindowFlags flags = SDL_WINDOW_VULKAN;
    flags |= resizable ? SDL_WINDOW_RESIZABLE : 0;
    handle_ = SDL_CreateWindow(name, initial_size.x, initial_size.y, flags);
    HWND hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(handle_), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    window_handle_ = (uint64_t)hwnd;
    IMGUI_CHECKVERSION();
}

SDLWindow::SDLWindow(const char *name, uint2 initial_size, bool resizable) noexcept
    : Window(resizable) {
    init(name, initial_size, resizable);
}

SDLWindow::~SDLWindow() noexcept {
    //glfwMakeContextCurrent(handle_);
    //texture_.reset();
    //widgets_.reset();
    //ImGui_ImplOpenGL3_Shutdown();
    //ImGui_ImplGlfw_Shutdown();
    
    SDL_DestroyWindow(handle_);
}

uint2 SDLWindow::size() const noexcept {
    auto width = 0;
    auto height = 0;
    //glfwGetWindowSize(handle_, &width, &height);
    SDL_GetWindowSize(handle_, &width, &height);
    return make_uint2(
        static_cast<uint>(width),
        static_cast<uint>(height));
}

bool SDLWindow::should_close() const noexcept {
    return should_close_;
}

void SDLWindow::set_background(const uchar4 *pixels, uint2 size) noexcept {
    //if (texture_ == nullptr) { texture_ = ocarina::make_unique<GLTexture>(); }
    //texture_->load(pixels, size);
}

void SDLWindow::set_background(const float4 *pixels, uint2 size) noexcept {
    //if (texture_ == nullptr) {
    //    texture_ = ocarina::make_unique<GLTexture>();
    //}
    //texture_->load(pixels, size);
}


void SDLWindow::set_background(const Buffer<ocarina::float4> &buffer, ocarina::uint2 size) noexcept {
    //if (texture_ == nullptr) {
    //    texture_ = ocarina::make_unique<GLTexture>();
    //}
    //texture_->bind();
    //bind_buffer(buffer.gl_handle(), buffer.size_in_byte());
}


void SDLWindow::set_should_close() noexcept {
    should_close_ = true;
}

void SDLWindow::_begin_frame() noexcept {
    //if (!should_close()) {
        SDL_Event windowEvent;
        while (SDL_PollEvent(&windowEvent))
            if (windowEvent.type == SDL_EVENT_QUIT) {
                should_close_ = true;
                break;
            } else if (windowEvent.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
                break;
            }
        Window::_begin_frame();
    //}
}

void SDLWindow::_end_frame() noexcept {
    if (!should_close()) {
        Window::_end_frame();
    }
}

void SDLWindow::set_size(uint2 size) noexcept {
    if (resizable_) {
        SDL_SetWindowSize(handle_, static_cast<int>(size.x), static_cast<int>(size.y));
    } else {
        OC_WARNING("Ignoring resize on non-resizable window.");
    }
}

void SDLWindow::show_window() noexcept {
    SDL_ShowWindow(handle_);
}

void SDLWindow::hide_window() noexcept
{
    SDL_HideWindow(handle_);
}

}// namespace ocarina

OC_EXPORT_API ocarina::SDLWindow *create_sdlwindow(const char *name, ocarina::uint2 initial_size, bool resizable) {
    return ocarina::new_with_allocator<ocarina::SDLWindow>(name, initial_size, resizable);
}

OC_EXPORT_API void destroy_sdlwindow(ocarina::SDLWindow *ptr) {
    ocarina::delete_with_allocator(ptr);
}