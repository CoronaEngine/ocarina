//
// Created by Zero on 2024/3/16.
//

#include "widgets.h"
#include "imgui_internal.h"

namespace ocarina {
template<typename T>
ImVec2 to_ImVec2(const T &t) noexcept {
    return ImVec2(t.x, t.y);
}

float2 from_ImVec2(ImVec2 v) noexcept {
    return make_float2(v.x, v.y);
}

uint64_t ImGuiWidgets::calculate_key(const ocarina::ImageView &image) noexcept {
    return hash64(image.resolution(), image.pixel_storage());
}

GLTexture *ImGuiWidgets::obtain_texture(const ocarina::ImageView &image) noexcept {
    uint64_t key = calculate_key(image);
    if (!texture_map_.contains(key)) {
        texture_map_.insert(make_pair(key, TextureVec{}));
    }
    TextureVec &texture_vec = texture_map_.at(key);
    uint unbinding_index = texture_vec.size();
    for (int i = 0; i < texture_vec.size(); ++i) {
        GLTexture *texture = texture_vec[i].get();
        if (!texture->binding()) {
            unbinding_index = i;
        }
    }
    if (unbinding_index == texture_vec.size()) {
        texture_vec.emplace_back(make_unique<GLTexture>());
    }
    return texture_vec[unbinding_index].get();
}

ImGuiWidgets::ImGuiWidgets(Window *window)
    : Widgets(window) {
}

void ImGuiWidgets::push_item_width(int width) noexcept {
    ImGui::PushItemWidth(width);
}

void ImGuiWidgets::pop_item_width() noexcept {
    ImGui::PopItemWidth();
}

void ImGuiWidgets::begin_tool_tip() noexcept {
    ImGui::BeginTooltip();
}

void ImGuiWidgets::end_tool_tip() noexcept {
    ImGui::EndTooltip();
}

void ImGuiWidgets::begin_disabled() noexcept {
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
}

void ImGuiWidgets::end_disabled() noexcept {
    ImGui::PopStyleVar();
    ImGui::PopItemFlag();
}

bool ImGuiWidgets::radio_button(const std::string &label, bool active) noexcept {
    return ImGui::RadioButton(label.c_str(), active);
}

void ImGuiWidgets::image(ocarina::uint tex_handle, ocarina::uint2 size,
                         ocarina::float2 uv0, ocarina::float2 uv1) noexcept {
    auto tex_id = reinterpret_cast<ImTextureID>(static_cast<handle_ty>(tex_handle));
    ImGui::Image(tex_id, to_ImVec2(size),
                 to_ImVec2(uv0), to_ImVec2(uv1));
}

uint2 ImGuiWidgets::node_size() noexcept {
    ImVec2 size = ImGui::GetContentRegionAvail();
    return make_uint2(size.x, size.y);
}

void ImGuiWidgets::image(const Image &obj) noexcept {
    image(obj.view());
}

void ImGuiWidgets::image(const ocarina::ImageView &image_view) noexcept {
    GLTexture *gl_texture = obtain_texture(image_view);
    if (image_view.pixel_storage() == PixelStorage::BYTE4) {
        gl_texture->load(image_view.pixel_ptr<uchar4>(), image_view.resolution());
    } else if (image_view.pixel_storage() == PixelStorage::FLOAT4) {
        gl_texture->load(image_view.pixel_ptr<float4>(), image_view.resolution());
    }
    uint2 res = image_view.resolution();
    adaptive_image(gl_texture->tex_handle(), res);
}

bool ImGuiWidgets::push_window(const string &label) noexcept {
    return ImGui::Begin(label.c_str());
}

bool ImGuiWidgets::push_window(const string &label, ocarina::WindowFlag flag) noexcept {
    return ImGui::Begin(label.c_str(), nullptr, flag);
}

void ImGuiWidgets::pop_window() noexcept {
    ImGui::End();
}

bool ImGuiWidgets::tree_node(const string &label) noexcept {
    return ImGui::TreeNode(label.c_str());
}

void ImGuiWidgets::tree_pop() noexcept {
    ImGui::TreePop();
}

void ImGuiWidgets::push_id(char *str) noexcept {
    ImGui::PushID(str);
}

void ImGuiWidgets::pop_id() noexcept {
    ImGui::PopID();
}

bool ImGuiWidgets::folding_header(const string &label) noexcept {
    return ImGui::CollapsingHeader(label.c_str());
}

bool ImGuiWidgets::begin_main_menu_bar() noexcept {
    return ImGui::BeginMainMenuBar();
}

void ImGuiWidgets::end_main_menu_bar() noexcept {
    ImGui::EndMainMenuBar();
}

bool ImGuiWidgets::begin_menu_bar() noexcept {
    return ImGui::BeginMenuBar();
}

bool ImGuiWidgets::begin_menu(const string &label) noexcept {
    return ImGui::BeginMenu(label.c_str());
}

bool ImGuiWidgets::menu_item(const string &label) noexcept {
    return ImGui::MenuItem(label.c_str());
}

void ImGuiWidgets::end_menu() noexcept {
    ImGui::EndMenu();
}

void ImGuiWidgets::end_menu_bar() noexcept {
    ImGui::EndMenuBar();
}

void ImGuiWidgets::text(const char *format, ...) noexcept {
    va_list args;
    va_start(args, format);
    ImGui::TextV(format, args);
    va_end(args);
}

bool ImGuiWidgets::input_text(const std::string &label, char *buf, size_t buf_size) noexcept {
    return ImGui::InputText(label.c_str(), buf, buf_size);
}

void ImGuiWidgets::text_wrapped(const char *format, ...) noexcept {
    va_list args;
    va_start(args, format);
    ImGui::TextWrappedV(format, args);
    va_end(args);
}

bool ImGuiWidgets::check_box(const string &label, bool *val) noexcept {
    return ImGui::Checkbox(label.c_str(), val);
}

bool ImGuiWidgets::slider_float(const string &label, float *val, float min, float max) noexcept {
    return ImGui::SliderFloat(label.c_str(), val, min, max);
}

bool ImGuiWidgets::slider_float2(const string &label, float2 *val, float min, float max) noexcept {
    return ImGui::SliderFloat2(label.c_str(), reinterpret_cast<float *>(val), min, max);
}

bool ImGuiWidgets::slider_float3(const string &label, float3 *val, float min, float max) noexcept {
    return ImGui::SliderFloat3(label.c_str(), reinterpret_cast<float *>(val), min, max);
}

bool ImGuiWidgets::slider_float4(const string &label, float4 *val, float min, float max) noexcept {
    return ImGui::SliderFloat4(label.c_str(), reinterpret_cast<float *>(val), min, max);
}

bool ImGuiWidgets::slider_int(const string &label, int *val, int min, int max) noexcept {
    return ImGui::SliderInt(label.c_str(), val, min, max);
}

bool ImGuiWidgets::slider_int2(const string &label, int2 *val, int min, int max) noexcept {
    return ImGui::SliderInt2(label.c_str(), reinterpret_cast<int *>(val), min, max);
}

bool ImGuiWidgets::slider_int3(const string &label, int3 *val, int min, int max) noexcept {
    return ImGui::SliderInt3(label.c_str(), reinterpret_cast<int *>(val), min, max);
}

bool ImGuiWidgets::slider_int4(const string &label, int4 *val, int min, int max) noexcept {
    return ImGui::SliderInt4(label.c_str(), reinterpret_cast<int *>(val), min, max);
}

bool ImGuiWidgets::slider_uint(const string &label, uint *val, uint min, uint max) noexcept {
    return ImGui::SliderScalar(label.c_str(), ImGuiDataType_U32, val, &min, &max, "%u", 0);
}

bool ImGuiWidgets::slider_uint2(const string &label, uint2 *val, uint min, uint max) noexcept {
    return ImGui::SliderScalarN(label.c_str(), ImGuiDataType_U32, val, 2, &min, &max, "%u", 0);
}

bool ImGuiWidgets::slider_uint3(const string &label, uint3 *val, uint min, uint max) noexcept {
    return ImGui::SliderScalarN(label.c_str(), ImGuiDataType_U32, val, 3, &min, &max, "%u", 0);
}

bool ImGuiWidgets::slider_uint4(const string &label, uint4 *val, uint min, uint max) noexcept {
    return ImGui::SliderScalarN(label.c_str(), ImGuiDataType_U32, val, 4, &min, &max, "%u", 0);
}

bool ImGuiWidgets::color_edit(const string &label, float3 *val) noexcept {
    return ImGui::ColorEdit3(label.c_str(), reinterpret_cast<float *>(val));
}

bool ImGuiWidgets::color_edit(const string &label, float4 *val) noexcept {
    return ImGui::ColorEdit4(label.c_str(), reinterpret_cast<float *>(val));
}

bool ImGuiWidgets::button(const string &label, uint2 size) noexcept {
    return ImGui::Button(label.c_str(), ImVec2(size.x, size.y));
}

bool ImGuiWidgets::button(const string &label) noexcept {
    return ImGui::Button(label.c_str());
}

void ImGuiWidgets::same_line() noexcept {
    ImGui::SameLine();
}

void ImGuiWidgets::new_line() noexcept {
    ImGui::NewLine();
}

bool ImGuiWidgets::input_int(const string &label, int *val) noexcept {
    return ImGui::InputInt(label.c_str(), val);
}

bool ImGuiWidgets::input_int(const string &label, int *val, int step, int step_fast) noexcept {
    return ImGui::InputInt(label.c_str(), val, step, step_fast);
}

bool ImGuiWidgets::input_int2(const string &label, ocarina::int2 *val) noexcept {
    return ImGui::InputInt2(label.c_str(), reinterpret_cast<int *>(val));
}

bool ImGuiWidgets::input_int3(const string &label, ocarina::int3 *val) noexcept {
    return ImGui::InputInt3(label.c_str(), reinterpret_cast<int *>(val));
}

bool ImGuiWidgets::input_int4(const string &label, ocarina::int4 *val) noexcept {
    return ImGui::InputInt4(label.c_str(), reinterpret_cast<int *>(val));
}

bool ImGuiWidgets::input_uint(const string &label, uint *val) noexcept {
    return ImGui::InputUint(label.c_str(), val);
}

bool ImGuiWidgets::input_uint(const string &label, uint *val, uint step, uint step_fast) noexcept {
    return ImGui::InputUint(label.c_str(), val, step, step_fast);
}

bool ImGuiWidgets::input_uint2(const string &label, ocarina::uint2 *val) noexcept {
    return ImGui::InputUint2(label.c_str(), reinterpret_cast<uint *>(val));
}

bool ImGuiWidgets::input_uint3(const string &label, ocarina::uint3 *val) noexcept {
    return ImGui::InputUint3(label.c_str(), reinterpret_cast<uint *>(val));
}

bool ImGuiWidgets::input_uint4(const string &label, ocarina::uint4 *val) noexcept {
    return ImGui::InputUint4(label.c_str(), reinterpret_cast<uint *>(val));
}

bool ImGuiWidgets::input_float(const string &label, float *val) noexcept {
    return ImGui::InputFloat(label.c_str(), val);
}

bool ImGuiWidgets::input_float(const string &label, float *val, float step, float step_fast) noexcept {
    return ImGui::InputFloat(label.c_str(), val, step, step_fast);
}

bool ImGuiWidgets::input_float2(const string &label, ocarina::float2 *val) noexcept {
    return ImGui::InputFloat2(label.c_str(), reinterpret_cast<float *>(val));
}

bool ImGuiWidgets::input_float3(const string &label, ocarina::float3 *val) noexcept {
    return ImGui::InputFloat3(label.c_str(), reinterpret_cast<float *>(val));
}

bool ImGuiWidgets::input_float4(const string &label, ocarina::float4 *val) noexcept {
    return ImGui::InputFloat4(label.c_str(), reinterpret_cast<float *>(val));
}

bool ImGuiWidgets::drag_int(const string &label, int *val, float speed, int min, int max) noexcept {
    return ImGui::DragInt(label.c_str(), val, speed, min, max);
}

bool ImGuiWidgets::drag_int2(const string &label, ocarina::int2 *val, float speed, int min, int max) noexcept {
    return ImGui::DragInt2(label.c_str(), reinterpret_cast<int *>(val), speed, min, max);
}

bool ImGuiWidgets::drag_int3(const string &label, ocarina::int3 *val, float speed, int min, int max) noexcept {
    return ImGui::DragInt3(label.c_str(), reinterpret_cast<int *>(val), speed, min, max);
}

bool ImGuiWidgets::drag_int4(const string &label, ocarina::int4 *val, float speed, int min, int max) noexcept {
    return ImGui::DragInt4(label.c_str(), reinterpret_cast<int *>(val), speed, min, max);
}

bool ImGuiWidgets::drag_uint(const string &label, ocarina::uint *val, float speed, ocarina::uint min, ocarina::uint max) noexcept {
    return ImGui::DragUint(label.c_str(), val, speed, min, max);
}

bool ImGuiWidgets::drag_uint2(const string &label, ocarina::uint2 *val, float speed, ocarina::uint min, ocarina::uint max) noexcept {
    return ImGui::DragUint2(label.c_str(), reinterpret_cast<uint *>(val), speed, min, max);
}

bool ImGuiWidgets::drag_uint3(const string &label, ocarina::uint3 *val, float speed, ocarina::uint min, ocarina::uint max) noexcept {
    return ImGui::DragUint3(label.c_str(), reinterpret_cast<uint *>(val), speed, min, max);
}

bool ImGuiWidgets::drag_uint4(const string &label, ocarina::uint4 *val, float speed, ocarina::uint min, ocarina::uint max) noexcept {
    return ImGui::DragUint4(label.c_str(), reinterpret_cast<uint *>(val), speed, min, max);
}

bool ImGuiWidgets::drag_float(const string &label, float *val, float speed, float min, float max, const char *fmt) noexcept {
    return ImGui::DragFloat(label.c_str(), val, speed, min, max, fmt);
}

bool ImGuiWidgets::drag_float2(const string &label, ocarina::float2 *val, float speed, float min, float max, const char *fmt) noexcept {
    return ImGui::DragFloat2(label.c_str(), reinterpret_cast<float *>(val), speed, min, max, fmt);
}

bool ImGuiWidgets::drag_float3(const string &label, ocarina::float3 *val, float speed, float min, float max, const char *fmt) noexcept {
    return ImGui::DragFloat3(label.c_str(), reinterpret_cast<float *>(val), speed, min, max, fmt);
}

bool ImGuiWidgets::drag_float4(const string &label, ocarina::float4 *val, float speed, float min, float max, const char *fmt) noexcept {
    return ImGui::DragFloat4(label.c_str(), reinterpret_cast<float *>(val), speed, min, max, fmt);
}

bool ImGuiWidgets::combo(const std::string &label, int *current_item, const char *const *items, int item_num) noexcept {
    return ImGui::Combo(label.c_str(), current_item, items, item_num);
}

bool ImGuiWidgets::is_item_hovered() noexcept {
    return ImGui::IsItemHovered();
}

float2 ImGuiWidgets::mouse_pos() noexcept {
    return from_ImVec2(ImGui::GetMousePos());
}

}// namespace ocarina