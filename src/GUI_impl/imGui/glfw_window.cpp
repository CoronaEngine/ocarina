//
// Created by Zero on 2022/8/16.
//

#include "glfw_window.h"
#include "widgets.h"
#include "ext/imgui/gizmo/ImGuizmo.h"
#include "core/logging.h"
#include "rhi/common.h"
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include "GLFW/glfw3native.h"// for glfwGetWin32Window
#endif

namespace ocarina {

class GLFWContext {

public:
    GLFWContext() noexcept {
        glfwSetErrorCallback([](int error, const char *message) noexcept {
            OC_WARNING_FORMAT("GLFW error (code = {}): {}", error, message);
        });
        if (!glfwInit()) { OC_ERROR("Failed to initialize GLFW."); }
    }
    GLFWContext(GLFWContext &&) noexcept = delete;
    GLFWContext(const GLFWContext &) noexcept = delete;
    GLFWContext &operator=(GLFWContext &&) noexcept = delete;
    GLFWContext &operator=(const GLFWContext &) noexcept = delete;
    ~GLFWContext() noexcept { glfwTerminate(); }
    [[nodiscard]] static auto retain() noexcept {
        static std::weak_ptr<GLFWContext> instance;
        if (auto p = instance.lock()) { return p; }
        auto p = ocarina::make_shared<GLFWContext>();
        instance = p;
        return p;
    }
};

void GLWindow::init_widgets() noexcept {
    widgets_ = make_unique<ImGuiWidgets>(this);
}

void GLWindow::init(const char *name, uint2 initial_size, bool resizable) noexcept {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, resizable);

    monitor_ =glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor_);
    handle_ = glfwCreateWindow(
        static_cast<int>(mode->width),
        static_cast<int>(mode->height),
        name, nullptr, nullptr);
    if (handle_ == nullptr) {
        const char *error = nullptr;
        glfwGetError(&error);
        OC_ERROR_FORMAT("Failed to create GLFW window: {}.", error);
    }
    #ifdef _WIN32
    window_handle_ = (uint64_t)glfwGetWin32Window(handle_);
    #endif
    glfwMakeContextCurrent(handle_);
    glfwSwapInterval(0);// disable vsync

    if (!gladLoadGL()) { OC_ERROR("Failed to initialize GLAD."); }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    init_widgets();
    ImGuizmo::SetImGuiContext(ImGui::CreateContext());
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(handle_, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    glfwSetWindowUserPointer(handle_, this);
    glfwSetMouseButtonCallback(handle_, [](GLFWwindow *window, int button, int action, int mods) noexcept {
        if (ImGui::GetIO().WantCaptureMouse) {// ImGui is handling the mouse
            ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
        } else {
            auto self = static_cast<GLWindow *>(glfwGetWindowUserPointer(window));
            auto x = 0.0;
            auto y = 0.0;
            glfwGetCursorPos(self->handle(), &x, &y);
            if (auto &&cb = self->mouse_button_callback_) {
                cb(button, action, make_float2(static_cast<float>(x), static_cast<float>(y)));
            }
        }
    });
    glfwSetCursorPosCallback(handle_, [](GLFWwindow *window, double x, double y) noexcept {
        auto self = static_cast<GLWindow *>(glfwGetWindowUserPointer(window));
        if (auto &&cb = self->cursor_position_callback_) { cb(make_float2(static_cast<float>(x), static_cast<float>(y))); }
    });
    glfwSetWindowSizeCallback(handle_, [](GLFWwindow *window, int width, int height) noexcept {
        auto self = static_cast<GLWindow *>(glfwGetWindowUserPointer(window));
        if (auto &&cb = self->window_size_callback_) { cb(make_uint2(width, height)); }
    });
    glfwSetKeyCallback(handle_, [](GLFWwindow *window, int key, int scancode, int action, int mods) noexcept {
        // ImGui is handling the keyboard
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
        if (!ImGui::GetIO().WantCaptureKeyboard) {
            auto self = static_cast<GLWindow *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->key_callback_) { cb(key, action); }
        }
    });
    glfwSetScrollCallback(handle_, [](GLFWwindow *window, double dx, double dy) noexcept {
        if (ImGui::GetIO().WantCaptureMouse) {// ImGui is handling the mouse
            ImGui_ImplGlfw_ScrollCallback(window, dx, dy);
        } else {
            auto self = static_cast<GLWindow *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->scroll_callback_) {
                cb(make_float2(static_cast<float>(dx), static_cast<float>(dy)));
            }
        }
    });
    glfwSetCharCallback(handle_, ImGui_ImplGlfw_CharCallback);
}

GLWindow::GLWindow(const char *name, uint2 initial_size, bool resizable) noexcept
    : Window(resizable), context_{GLFWContext::retain()} {
    init(name, initial_size, resizable);

    glfwGetWindowPos(handle_, &windowedX, &windowedY);
    glfwGetWindowSize(handle_, &windowedWidth, &windowedHeight);

    lastF11Toggle = std::chrono::steady_clock::now();
    constexpr auto f11Cooldown = std::chrono::milliseconds(100);
}

GLWindow::~GLWindow() noexcept {
    glfwMakeContextCurrent(handle_);
    texture_.reset();
    widgets_.reset();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(handle_);
}

uint2 GLWindow::size() const noexcept {
    auto width = 0;
    auto height = 0;
//    const GLFWvidmode* mode = glfwGetVideoMode(monitor_);
//    return make_uint2(mode->width, mode->height);
    glfwGetWindowSize(handle_, &width, &height);
    return make_uint2(
        static_cast<uint>(width),
        static_cast<uint>(height));
}

bool GLWindow::should_close() const noexcept {
    return glfwWindowShouldClose(handle_);
}

void GLWindow::full_screen() {
    auto now = std::chrono::steady_clock::now();
    if (now - lastF11Toggle > std::chrono::milliseconds(100)) {
        lastF11Toggle = now;
        //static bool isFullscreen = false;
        if (isFullscreen) {
            glfwSetWindowMonitor(handle_, NULL, windowedX, windowedY, windowedWidth, windowedHeight, 0);
        } else {
            const GLFWvidmode *mode = glfwGetVideoMode(monitor_);
            glfwGetWindowPos(handle_, &windowedX, &windowedY);
            glfwGetWindowSize(handle_, &windowedWidth, &windowedHeight);
            glfwSetWindowMonitor(handle_, monitor_, 0, 0, mode->width, mode->height, mode->refreshRate);
        }
        isFullscreen = !isFullscreen;
    }
}

void GLWindow::swap_monitor() {
    if (isFullscreen) {
        int count;
        GLFWmonitor **monitors = glfwGetMonitors(&count);
        monitor_index = (monitor_index + 1) % count;
        monitor_ = monitors[monitor_index];
        const GLFWvidmode *mode = glfwGetVideoMode(monitor_);
        glfwSetWindowMonitor(handle_, monitor_, 0, 0, mode->width, mode->height, mode->refreshRate);
    }
}

void GLWindow::set_background(const uchar4 *pixels, uint2 size) noexcept {
    if (texture_ == nullptr) { texture_ = ocarina::make_unique<GLTexture>(); }
    texture_->load(pixels, size);
}

void GLWindow::set_background(const float4 *pixels, uint2 size) noexcept {
    if (texture_ == nullptr) {
        texture_ = ocarina::make_unique<GLTexture>();
    }
    texture_->load(pixels, size);
}

void GLWindow::gen_buffer(ocarina::uint &handle, ocarina::uint size_in_byte) const noexcept {
    CHECK_GL(glGenBuffers(1, addressof(handle)));
    CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, handle));
    CHECK_GL(glBufferData(GL_ARRAY_BUFFER, size_in_byte,
                          nullptr, GL_STREAM_DRAW));
    CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0u));
}

void GLWindow::bind_buffer(ocarina::uint &handle, ocarina::uint size_in_byte) const noexcept {
    if (handle == 0) {
        gen_buffer(handle, size_in_byte);
    }
    CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, handle));
}

void GLWindow::set_background(const Buffer<ocarina::float4> &buffer, ocarina::uint2 size) noexcept {
    if (texture_ == nullptr) {
        texture_ = ocarina::make_unique<GLTexture>();
    }
    texture_->bind();
}

void GLWindow::unbind_buffer(ocarina::uint &handle) const noexcept {
    CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void GLWindow::set_should_close() noexcept {
    glfwSetWindowShouldClose(handle_, true);
}

void GLWindow::_begin_frame() noexcept {
    if (!should_close()) {
        glfwMakeContextCurrent(handle_);
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();
        Window::_begin_frame();
    }
}

void GLWindow::_end_frame() noexcept {
    if (!should_close()) {
        Window::_end_frame();
        // background
        if (texture_ != nullptr) {
            ImVec2 background_size{
                static_cast<float>(texture_->size().x),
                static_cast<float>(texture_->size().y)};
            ImGui::GetBackgroundDrawList()->AddImage(
                reinterpret_cast<ImTextureID>(static_cast<uint64_t>(texture_->handle())), {}, background_size);
        }
        // rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(handle_, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color_.x, clear_color_.y, clear_color_.z, clear_color_.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(handle_);
    }
}

void GLWindow::set_size(uint2 size) noexcept {
    if (resizable_) {
        glfwSetWindowSize(handle_, static_cast<int>(size.x), static_cast<int>(size.y));
    } else {
        OC_WARNING("Ignoring resize on non-resizable window.");
    }
}

void GLWindow::show_window() noexcept
{

}

void GLWindow::hide_window() noexcept
{

}

}// namespace ocarina

//OC_EXPORT_API ocarina::GLWindow *create(const char *name, ocarina::uint2 initial_size, bool resizable) {
//    return ocarina::new_with_allocator<ocarina::GLWindow>(name, initial_size, resizable);
//}
//
//OC_EXPORT_API void destroy(ocarina::GLWindow *ptr) {
//    ocarina::delete_with_allocator(ptr);
//}