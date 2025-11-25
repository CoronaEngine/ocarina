//
// Created by Zero on 26/04/2022.
//

#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/context.h"
#include "generator/cpp_codegen.h"
#include "rhi/common.h"
#include <windows.h>
#include "math/base.h"
#include "core/image.h"
#include "dsl/dsl.h"

using namespace ocarina;

int main(int argc, char *argv[]) {

    float3 v1 = make_float3(2), v2, v3;
    coordinate_system(v1,v2, v3);

    ocarina::vector<float> v;
    const int count = 10;
    for (int i = 0; i < count; ++i) {
        v.push_back(i);
    }

    Callable add = [&](Var<float> a, Var<float> b) {
        return a + b;
    };

    Callable comp = [&](Var<float> a, Var<float> b) {
        Var<tuple<float, float>> ret {};
        ret.set<0>(a);
        ret.set<1>(b);
        return ret;
    };

    fs::path path(argv[0]);
    RHIContext &context = RHIContext::instance();
//    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();

    auto path2 = R"(E:/work/compile/ocarina/res/test.png)";
    auto image = Image::load(path2, LINEAR);

    auto texture = device.create_texture(image.resolution(), image.pixel_storage());
    stream << texture.upload_sync(image.pixel_ptr());

    Buffer<float> f_buffer = device.create_buffer<float>(count);
    Kernel kn = [&](Var<float> a, Var<float> b, BufferVar<float> c, TextureVar tex) {

        Float ll = log(a);

        Var<float3> v1 = make_float3(a), v2,v3;
        auto ret = comp(1,5);
        Float3 ff3{float3()};

        auto pp = Pow<5>(1 - ff3.z);

        v1 = normalize(v1);
        coordinate_system(v1,v2,v3);
        prints("{},{},{}",v2);
        return;
        //                configure_block(1,2,1);
        DynamicArray<uint> ua(10);
        ua[5] = 1u;
        Var uuu = ua[5];
        Var<int3> vec = make_int3(1,2, a.cast<int>());
        Var<int> ii = 2;
        Var<int2> v22 = make_int2(ii, ii);
        Var<bool2> bv;

        Var tex_v = tex.sample(4, a, b).as_vec4();
        Var tv2 = texture.sample(4, a, b).as_vec4();

        Var vv = all(bv);
        Var f = 0.5f;
        f = fma(f, a, b);
        Var mat = make_float4x4(1.f);
        Var<int2> vec2 = make_int2(1);

        vec2 = -vec2;
        //        Var<bool3> pred = vec > make_int3(5);
        vec = select(vec > make_int3(5), vec, -vec);
        Var tr = tex.read<float4>(20u,20u);
        print("{}, {}---------{}--{}>>>>>{}", sqr(a), tv2.x, f_buffer.read(5), tex_v.x, tr.x);
        f_buffer.write(thread_id(), f_buffer.read(thread_id()) * 2);
//        c.write(thread_id(), c.read(thread_id()) * 2);
        c[thread_id()] *= 2;
        a = add(a, b);
        Bool bbbb = ocarina::isinf(f);
    };

    auto shader = device.compile(kn);
    //    shader.compute_fit_size();
//    return 0;
    stream << f_buffer.upload_sync(v.data());
    stream << shader(0.1f, 0.9f, f_buffer, texture).dispatch(1);
    stream << synchronize();
    stream << f_buffer.download_sync(v.data());
    stream << commit();
    exit(0);
//    for (int i = 0; i < count; ++i) {
//        cout << v[i] << endl;
//    }

    stream << shader(0.95f, 0.55f, f_buffer, texture).dispatch(10);
    stream << synchronize();
    stream << f_buffer.download_sync(v.data());
    stream << commit();
    //
//        for (int i = 0; i < count; ++i) {
//            cout << v[i] << endl;
//        }

    return 0;
}
