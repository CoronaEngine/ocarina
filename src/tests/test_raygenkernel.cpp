//
// Created by Zero on 2022/8/18.
//

#include "core/image.h"
#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "rhi/context.h"
#include "math/base.h"
#include "tiny_obj_loader.h"
#include "cornell_box.h"

using namespace ocarina;

struct Triangle {
public:
    uint i, j, k;
//    array<float, 3> f;
    Triangle(uint i, uint j, uint k) : i(i), j(j), k(k) {}
    Triangle() = default;
};
OC_STRUCT(,Triangle, i, j, k){};

struct Triangle2 {
public:
    Triangle t;
};
OC_STRUCT(,Triangle2,t){};

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

auto operator-(DynamicArray<float> arr) {
    const Expression *expr = Function::current()->unary(arr.type(), UnaryOp::NEGATIVE, arr.expression());
    return DynamicArray<float>(arr.size(), expr);
}

//template<typename... Args>
//void func(Args... args) {
//    auto tp = args_to_tuple(args...);
//
//    int i = 0;
//}

template<typename T, typename F2>
[[nodiscard]] T triangle_lerp2(const F2 &barycentric, const T &v0, const T &v1, const T &v2) noexcept {
    auto u = barycentric.x;
    auto v = barycentric.y;
    auto w = 1 - barycentric.x - barycentric.y;
    return u * v0 + v * v1 + w * v2;
}

float3 barycentric2(float2 p, float2 p0, float2 p1, float2 p2) {
    float3 U = cross(make_float3(p1.x - p0.x, p2.x - p0.x, p0.x - p.x),
                     make_float3(p1.y - p0.y, p2.y - p0.y, p0.y - p.y));
    return make_float3(1 - (U.x + U.y) / U.z, U.y / U.z, U.x / U.z);
}


void test() {
    float2 a = make_float2(0, 0);
    float2 b = make_float2(1, 0);
    float2 c = make_float2(0, 1);

    float2 p = make_float2(0.1,0.3);

    float2 bary = barycentric<H>(p, a, b, c);

    float2 p2 = triangle_lerp(bary, a, b, c);

    auto tb = traceback_string();

    cout << tb << endl;
    exit(0);

    return;
}

int main(int argc, char *argv[]) {
    Type::of<Triangle2>();
    //    func(1, 2, 3, 4, 5, 6, 7, 8, 18);
    //
    //    return 0;
    log_level_debug();
    auto type = Type::of<Triangle>();

//    test();

    int pp = alignof(void *);

    fs::path path(argv[0]);
    RHIContext &context = RHIContext::instance();
//    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    Env::printer().init(device);
//    auto path1 = R"(E:/work/test_dir/D5.png)";
//    auto path2 = R"(E:/work/test_dir/D5.jpg)";
//    auto image_io = Image::load(path1, LINEAR);
//    auto image = device.create_texture(image_io.resolution(), image_io.pixel_storage());
//    stream << image.upload_sync(image_io.pixel_ptr());
    device.init_rtx();

    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    auto [vertices, triangle] = get_cube();

    auto vert = device.create_managed<float3>(vertices.size());

    vert.host_buffer() = std::move(vertices);

    Buffer v_buffer = device.create_buffer<float3>(vert.host_buffer().size());
    Buffer t_buffer = device.create_buffer<Triangle>(triangle.size());

    Managed managed = device.create_managed<float>(100);
    for (int i = 0; i < 100; ++i) {
        managed.push_back(i);
    }
    managed.upload_immediately();
    RHIMesh cube = device.create_mesh(vert.device_buffer(), t_buffer);

    BindlessArray bindless_array = device.create_bindless_array();

    auto r1 = bindless_array.emplace(v_buffer.view(1));
    auto r2 = bindless_array.emplace(v_buffer);

    uint index = bindless_array.emplace(managed.device_buffer());

    stream << vert.upload_sync();
    stream << v_buffer.upload_sync(vert.host_buffer().data());
    stream << t_buffer.upload_sync(triangle.data());

    bindless_array.prepare_slotSOA(device);
    stream << bindless_array->upload_buffer_handles(true) << synchronize();
    stream << bindless_array->upload_texture_handles(true) << synchronize();

    stream << cube.build_bvh();

    Accel accel = device.create_accel();
    accel.add_instance(ocarina::move(cube), make_float4x4(1.f));
    stream << accel.build_bvh();
    stream << synchronize() << commit();

    Callable cb = [&]() {
        return Var<ocarina::array<float, 10>>();
    };

    vector<float> aaa = {1, 2, 3};
    vector<float> bb = {6};

    float2 bar = make_float2(0, 1.0001);
    float2 a = make_float2(1, 1);
    float2 b = make_float2(-1, 1);
    float2 c = make_float2(0, -1);
    bool in = in_triangle<H>(bar, a, b, c);

    vector<float> t = {0.5f};

    auto ll = lerp(t, aaa, bb);

    Kernel kernel = [&](
                        BindlessArrayVar ba) {
        //        t_buffer.atomic()
        //        managed.device().atomic(1).fetch_sub(2);
        //        Var<Ray> r = make_ray(Var(float3(0, 0.1, -5)), float3(1.6f, 0, 1));
        //        Var hit = accel.trace_closest(r);
//        Var t = t_buffer.read(0);
//        Int3 f = make_int3(ba.byte_buffer_var(index).read<float>(19 * 4).cast<int>(), 6, 9);
//        auto arr = bindless_array.byte_buffer_var(index).load_dynamic_array<float>(3, 19 * 4);
//        Env::printer().warn_with_location("{} {} {}, {} {} ,{} {} {}", f, arr.sub(1, 3).as_vec2(), t.i.cast<float>() + 2.4f, t.j, t.k);

        DynamicArray<float> p{3};
        DynamicArray<float> t{3};
        DynamicArray<float> f{3};
        for (int i = 0; i < 3; ++i) {
            p[i] = 0;
            t[i] = -i;
            f[i] = -2 * i;
        }

        p[1] = 1;

//        DynamicArray<float> r = select(p, t, f);
//        r = t +f;
        DynamicArray<float3> rr{3};
        rr[0] = make_float3(-9);
        rr[1] = make_float3(-8);
        rr = abs(rr);
        Env::printer().info("{} {} {} {} {} {}", rr[0], rr[1]);
//                Container<int> container{4};
//        Var<Triangle> tri;
//        tri.f[0] = 1.f;
//        container.push_back(5);
//        container.push_back(8);
//
//        container.for_each([&](Int i) {
//            Env::printer().info("{}", tri.f[0]);
//        });
//
//        container.pop();

//        Var ff = select(Var(false), tri, tri);
        //        prints("{}---", is_null(img));
        //      Int a = 1, b = 2, c = 3;
        //      printer.log_debug("--{} {} {}", a, b, c);
        //        prints("++{} {} {}", f);
        //        printer.log_debug("--------{} {} {}", 1.5f,f.x,1.11f);
        //        print("sdfasdasdfasdsdafsdafasdfsda{} {} {}", 1.5f,f.x,1.9f);
        //        print("------adfasdfsdafasdasasfasdfasdfasdf--{} {} {}", 1.5f,f.x,1.11f);
        //        DynamicArray<float> arr = DynamicArray<float>::create(1.f, 2.f, 3.f, 4.f);
        //        arr *= arr;
        //        prints("{} {} {} {}", arr.wzyx().as_vec4());
        //        Float3 pos = r->direction();
        //        Float4 pix = img.read<float4>(200, 150);
        //        Float2 uv = make_float2(0.7f);
        //        Float4 pix2 = img.sample(4, uv).as_vec4();
        //        Float3 p = vert.read(1);
        //        Var f2 = make_float2(Var(7.f));
        //        auto t = bindless_array.buffer<array<float3, 1>>(0).read(0);
        //        print("{},{}----------{} {}", hit.prim_id, hit.inst_id, hit->bary.x, hit.bary.y);
        //        print("{}  {}  {}  {} {}", tri.i, f2.x, f2.y, p.x, p.y);
        //        //        prints("{} {} {}", t);
        //        prints("{} {} {} {}", pix2);
        //        prints("{} {} {} {}", ba.tex(0).sample(4, uv).as_vec4());
        //        prints("{} {} {} {}", bindless_array.tex(0).sample(4, uv).as_vec4());
    };
    auto shader = device.compile(kernel);
    stream << shader(bindless_array).dispatch(3);
    stream << Env::printer().retrieve() << synchronize() <<commit();

//    float tf = bit_cast<float>(19);
//    OC_WARNING_FORMAT("{}", tf);
//    Env::printer().retrieve_immediately();
//    //    cout << "sdafasdf" << endl;
//    Env::printer().retrieve_immediately();
//    Printer::destroy_instance();
    return 0;
}