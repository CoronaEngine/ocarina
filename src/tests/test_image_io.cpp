//
// Created by Zero on 05/08/2022.
//

#include "core/image.h"
#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/context.h"
#include "rhi/common.h"
#include <windows.h>
#include "math/base.h"
#include "core/image.h"
#include "dsl/dsl.h"

using namespace ocarina;

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    RHIContext &context = RHIContext::instance();
//    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    auto path1 = R"(D:\work\engine\Vision\gallery\cbox-sss.png)";
    auto path2 = R"(D:\work\engine\Vision\gallery\cbox-sss.jpg)";
    auto image_io = Image::load(path1, LINEAR);

    auto image = device.create_texture3d(image_io.resolution(), image_io.pixel_storage());
    auto image_out = device.create_texture3d(image_io.resolution(), image_io.pixel_storage());
    stream << image.upload_sync(image_io.pixel_ptr());

    Kernel kernel = [&](Texture3DVar img) {
        uint2 res = image_io.resolution();
        int r = 5;
        Var p = img.sample(4, 0.6f,0.6f).as_vec4();
//        print("{}  {}", p.x, p.y);
        Var<uint> min_x = max(0u, dispatch_idx().x - r);
        Var<uint> max_x = min(res.x - 1, dispatch_idx().x + r);
        Var<uint> min_y = max(0u, dispatch_idx().y - r);
        Var<uint> max_y = min(res.y - 1, dispatch_idx().y + r);


        Var<float4> var = make_float4(dispatch_idx().x / cast<float>(res.x));
        Var<uint> count = 0u;
        $for(x, min_x, max_x) {
            $for(y, min_y, max_y) {
                Var v = img.read<float4>(x, y);
                var += v;
                count += 1;
            };
        };
        var /= count.cast<float>();
        image_out.write(var,dispatch_idx().xy());
    };
    kernel.function()->set_raytracing(true);
//    kernel.function()->set_raytracing(false);
    auto shader = device.compile(kernel);
    Clock clk;
    stream << shader(image).dispatch(image_io.resolution());
    stream << synchronize()
           << image_out.download_sync(image_io.pixel_ptr())
           << commit();
    image_io.save(path2);

    cout << "elapse ms " << clk.elapse_ms() << endl;

    return 0;
}