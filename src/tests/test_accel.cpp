//
// Created by Zero on 09/08/2022.
//

#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/context.h"
#include "rhi/common.h"
#include <windows.h>
#include "math/base.h"
#include "core/image.h"
#include "dsl/dsl.h"

using namespace ocarina;

struct Triangle {
public:
    uint i, j, k;
    Triangle(uint i, uint j, uint k) : i(i), j(j), k(k) {}
    Triangle() = default;
};
OC_STRUCT(,Triangle, i, j, k){};

auto get_cube(float x = 1, float y = 1, float z = 1) {
    x = x / 2.f;
    y = y / 2.f;
    z = z / 2.f;
    auto vertices = vector<float3>{
        float3(-x, -y, z), float3(x, -y, z), float3(-x, y, z), float3(x, y, z),    // +z
        float3(-x, y, -z), float3(x, y, -z), float3(-x, -y, -z), float3(x, -y, -z),// -z
        float3(-x, y, z), float3(x, y, z), float3(-x, y, -z), float3(x, y, -z),    // +y
        float3(-x, -y, z), float3(x, -y, z), float3(-x, -y, -z), float3(x, -y, -z),// -y
        float3(x, -y, z), float3(x, y, z), float3(x, y, -z), float3(x, -y, -z),    // +x
        float3(-x, -y, z), float3(-x, y, z), float3(-x, y, -z), float3(-x, -y, -z),// -x
    };
    auto triangles = vector<Triangle>{
        Triangle(0, 1, 3),
        Triangle(0, 3, 2),
        Triangle(6, 5, 7),
        Triangle(4, 5, 6),
        Triangle(10, 9, 11),
        Triangle(8, 9, 10),
        Triangle(13, 14, 15),
        Triangle(13, 12, 14),
        Triangle(18, 17, 19),
        Triangle(17, 16, 19),
        Triangle(21, 22, 23),
        Triangle(20, 21, 23),
    };

    return ocarina::make_pair(vertices, triangles);
}

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    RHIContext &context = RHIContext::instance();
    context.clear_cache();
    Device device = context.create_device("cuda");
    device.init_rtx();
    Stream stream = device.create_stream();
    auto [vertices, triangle] = get_cube();

    Buffer v_buffer = device.create_buffer<float3>(vertices.size());
    Buffer t_buffer = device.create_buffer<Triangle>(triangle.size());

    RHIMesh cube = device.create_mesh(v_buffer.view(), t_buffer.view());

    stream << v_buffer.upload_sync(vertices.data());
    stream << t_buffer.upload_sync(triangle.data());

    stream << cube.build_bvh();

    Accel accel = device.create_accel();
    accel.add_instance(ocarina::move(cube), make_float4x4(1.f));
    stream << accel.build_bvh();

    Callable cb = [&](Var<Triangle> t) {
        print("{},{},{}--", t.i,t.j,t.k);
    };

    Kernel kernel = [&](BufferVar<float3> v, Var<Triangle> triangle) {
        Var<float3> pos = v_buffer.read(dispatch_idx().x);
        Var<float3> pos2 = v[dispatch_id()];
        Var t = t_buffer.read(dispatch_id());
        cb(t);
        print("{},{},{}, {}", pos.x, pos2.y, pos.z, dispatch_dim().x);
    };

    auto shader = device.compile(kernel);
    stream << shader(v_buffer, triangle[2]).dispatch(t_buffer.size());
    stream << synchronize() << commit();

    return 0;
}