//
// Created by Zero on 2022/8/18.
//

#include "core/image.h"
#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "math/base.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "cornell_box.h"
#include "rhi/context.h"

using namespace ocarina;

struct Material {
    float3 albedo;
    float3 emission;
};

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

struct Triangle {
public:
    uint i, j, k;
    Triangle(uint i, uint j, uint k) : i(i), j(j), k(k) {}
    Triangle() = default;
};
OC_STRUCT(,Triangle, i, j, k){};

struct MeshHandle {
    uint tri_offset;
    uint tri_num;

public:
    MeshHandle(uint o, uint n) : tri_offset(o), tri_num(n) {}
};

OC_STRUCT(,MeshHandle, tri_offset, tri_num){};

// clang-format off
OC_STRUCT(,Material, albedo, emission) {};
OC_STRUCT(,Onb, tangent, binormal, normal) {
    [[nodiscard]] auto to_world(Var<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};
// clang-format on

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    RHIContext &context = RHIContext::instance();
    context.clear_cache();
    Device device = context.create_device("cuda");
    device.init_rtx();
    Stream stream = device.create_stream();
    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    uint2 res = make_uint2(768, 768);

    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromString(obj_string, "", obj_reader_config)) {
        std::string_view error_message = "unknown error.";
        if (auto &&e = obj_reader.Error(); !e.empty()) { error_message = e; }
        OC_ERROR_FORMAT("Failed to load OBJ file: {}", error_message);
    }
    if (auto &&e = obj_reader.Warning(); !e.empty()) {
        OC_ERROR_FORMAT("{}", e);
    }

    auto &&p = obj_reader.GetAttrib().vertices;
    std::vector<float3> vertices;
    vertices.reserve(p.size() / 3u);
    for (auto i = 0u; i < p.size(); i += 3u) {
        vertices.emplace_back(float3{
            p[i + 0u],
            p[i + 1u],
            p[i + 2u]});
    }
    OC_INFO_FORMAT(
        "Loaded mesh with {} shape(s) and {} vertices.",
        obj_reader.GetShapes().size(), vertices.size());
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.upload_sync(vertices.data())
           << commit();
    size_t num = 0;
    std::vector<Triangle> triangles;
    std::vector<MeshHandle> meshHandles;

    uint offset = 0;
    for (auto &&shape : obj_reader.GetShapes()) {
        auto &&t = shape.mesh.indices;
        uint triangle_count = t.size() / 3u;
        OC_INFO_FORMAT(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, num++, triangle_count);
        for (int i = 0; i < t.size(); i += 3) {
            triangles.emplace_back(t[i].vertex_index, t[i + 1].vertex_index, t[i + 2].vertex_index);
        }
        meshHandles.emplace_back(offset, triangle_count);
        offset += triangle_count;
    }
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    stream << triangle_buffer.upload_sync(triangles.data());
    stream << synchronize() << commit();
    std::vector<RHIMesh> meshes;
    for (auto handle : meshHandles) {
        auto mesh = device.create_mesh(vertex_buffer, triangle_buffer.view(handle.tri_offset, handle.tri_num));
        meshes.push_back(std::move(mesh));
    }
    for (RHIMesh &mesh : meshes) {
        stream << mesh.build_bvh();
    }
    stream << synchronize() << commit();

    auto accel = device.create_accel();
    for (const RHIMesh &mesh : meshes) {
//        accel.add_mesh((mesh), make_float4x4(1.f));
    }

    Callable lcg = [](Uint &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
    };

    stream << accel.build_bvh() << synchronize() << commit();

    std::vector<Material> materials;
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// floor
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// ceiling
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// back wall
    materials.emplace_back(Material{make_float3(0.14f, 0.45f, 0.091f), make_float3(0.0f)});// right wall
    materials.emplace_back(Material{make_float3(0.63f, 0.065f, 0.05f), make_float3(0.0f)});// left wall
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// short box
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// tall box
    materials.emplace_back(Material{make_float3(0.0f), make_float3(17.0f, 12.0f, 4.0f)});  // light

    auto material_buffer = device.create_buffer<Material>(materials.size());
    stream << material_buffer.upload_sync(materials.data());
    static constexpr auto fov = radians(27.8f);
    static constexpr auto origin = make_float3(-0.01f, 0.995f, 5.0f);

    auto seed_buffer = device.create_buffer<uint>(res.x * res.y);



//    Callable generate_ray = [](Float2 p) noexcept {
//
//        Var pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
//        Var direction = normalize(pixel - origin);
//        return make_ray(origin, direction);
//    };

    auto image = Image::pure_color(make_float4(0, 0, 0, 1), ColorSpace::LINEAR, res);
    auto frame = device.create_texture(res, PixelStorage::FLOAT4);

    Kernel raytracing = [&](Var<Texture> output) {
        //        Var ray = make_ray(make_float3(0), make_float3(0));
//        Var coord = dispatch_idx().xy();
        Var state = seed_buffer.read(dispatch_id());
        Var rx = lcg(state);
        Var ry = lcg(state);
        seed_buffer.write(dispatch_id(), state);
//        Var pixel = (make_float2(coord) + make_float2(rx, ry)) / (res.x * 2.0f) - 1.0f;
//        Var ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
////        Var ray = make_ray(origin, make_float3(0,0,-1));
//        output.write(make_uint2(dispatch_idx()), make_float4(1, 1, 0, 1));
//        Var hit = accel.trace_closest(ray);
//        print("{},{},{}", hit.bary.x, hit.bary.y, hit.inst_id);
        //        Var mat = material_buffer.read(0).emission;
    };
    auto shader = device.compile(raytracing);

    auto window = context.create_window("display", res);
//    window->run([&](double t) {


    return 0;
}