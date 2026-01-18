//
// Created by Z 10/10/2022.
//

#include "math/box.h"
#include "core/stl.h"
#include "rhi/common.h"
#include "core/string_util.h"
#include "math/transform.h"
#include "math/geometry.h"
#include "base/scattering/interaction.h"
#include "base/scattering/bxdf.h"

using namespace vision;
using namespace ocarina;

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
    Device device = context.create_device("cuda");
    device.init_rtx();
    Stream stream = device.create_stream();
    Env::printer().init(device);
    uint count = 1;

    auto path1 = R"(D:\work\corona\CoronaTestScenes\test_vision\render_scene\cbox\checker.jpg)";
    auto dst_path = "D:\\work\\corona\\CoronaTestScenes\\test_vision\\render_scene\\cbox\\checker2.jpg";

    auto image = Image::load(path1, ColorSpace::SRGB);

    auto bindless = device.create_bindless_array();
    auto tex = device.create_texture2d(image.resolution(), image.pixel_storage(), "test");
    auto tex2 = device.create_texture2d(image.resolution(), image.pixel_storage(), "test");
    auto buffer = device.create_buffer<float>(image.pixel_num());
    stream << buffer.upload(image.pixel_ptr());
    stream << tex.upload(image.pixel_ptr());
    bindless->emplace_texture2d(tex.tex_handle());
    bindless->emplace_texture2d(tex2.tex_handle());


    stream << bindless.upload_handles() ;
//    stream << tex.copy_from(tex2);

    Ray ray;
    cout << to_str(ray) << endl;


    // 我擦
    Kernel kernel = [&](Texture2DVar texture_var) {
        Float2 uv = make_float2(dispatch_idx()) / make_float2(dispatch_dim());
        //        static_assert(is_general_vector2_v<decltype(float2{}.xy())>);
        Float4 val = bindless.tex2d_var(0).sample(4, make_float3(uv, 0).xy()).as_vec4();
        Float4 v = tex.template read<float4>(dispatch_idx().xy());
//        val = tex.sample(4, uv).as_vec4();
//        tex.write(make_float4(1,0.3,0,1),dispatch_idx().xy());
        tex.write(v * 0.5f,dispatch_idx().xy());
        Uint2 xy = dispatch_idx().xy();
        //        static_assert(is_all_integral_expr_v<Uint>);
                auto va2l = texture_var.read<float4>(dispatch_idx().xy());
//        $info("{} {}, {} {} {} {}", uv, val);
    };
    auto shader = device.compile(kernel);
    stream << shader(tex).dispatch(image.resolution()) << Env::printer().retrieve()<< synchronize() << commit();
//    stream << buffer.
//    stream << buffer.upload(image.pixel_ptr()) << synchronize() << commit();
    stream << tex.copy_from_buffer(buffer,0, true);
    stream << tex.download(image.pixel_ptr()) << synchronize() << commit();
    image.save(dst_path);


    return 0;
}