//
// Created by Zero on 2023/11/23.
//

#include "core/image.h"
#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "math/base.h"
#include "base/scattering/interaction.h"
#include "core/platform.h"
#include "rhi/context.h"
//#include "numpy.h"

using namespace ocarina;

struct Triple {
    uint i{50}, j{}, k{};
    TriangleHit h{};
    Triple(uint i, uint j, uint k) : i(i), j(j), k(k) {}
    Triple() = default;
};

/// register a DSL struct, if you need upload a struct to device, be sure to register
OC_STRUCT(, Triple, i, j, k, h){
    [[nodiscard]] Uint sum() const noexcept {
        return i + j + k;
}
}
;

struct Pair {
    uint i{50};
    Triple triple;
    //    BufferDesc<float3> b;
    //    BufferDesc<Triple> t;
    Pair() = default;
};

/// register a DSL struct, if you need upload a struct to device, be sure to register
OC_STRUCT(, Pair, i, triple){

};

struct Param {
    uint i{50};
    BufferDesc<float3> b;
    BufferDesc<Triple> t;
    Pair pa;
    BufferDesc<float3> b2;
    Param() = default;
};

/// register a DSL struct, if you need upload a struct to device, be sure to register
OC_PARAM_STRUCT(, Param, i, b, t, pa, b2){

};

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
    auto triangles = vector<Triple>{
        Triple(0, 1, 3),
        Triple(0, 3, 2),
        Triple(6, 5, 7),
        Triple(4, 5, 6),
        Triple(10, 9, 11),
        Triple(8, 9, 10),
        Triple(13, 14, 15),
        Triple(13, 12, 14),
        Triple(18, 17, 19),
        Triple(17, 16, 19),
        Triple(21, 22, 23),
        Triple(20, 21, 23),
    };

    return ocarina::make_pair(vertices, triangles);
}

void test_compute_shader(Device &device, Stream &stream) {
    auto [vertices, triangles] = get_cube();

    Buffer<float3> vert = device.create_buffer<float3>(vertices.size());
    Buffer tri = device.create_buffer<Triple>(triangles.size());

    /// used for store the handle of texture or buffer
    BindlessArray bindless_array = device.create_bindless_array();
    //    uint v_idx = bindless_array.emplace(vert);
    //    uint t_idx = bindless_array.emplace(tri);

    using Elm = float4x4;
    uint len = 11;
    auto byte_buffer = device.create_byte_buffer((len) * sizeof(Elm) + 4, "");
    auto byte_buffer2 = device.create_byte_buffer((len + 1) * sizeof(Elm), "");
    vector<Elm> byte_vec;

    auto fbuffer = device.create_buffer<Elm>(len + 1);

    byte_vec.resize(len + 1);

    stream << byte_buffer.upload(byte_vec.data(), false);

    uint byte_handle = bindless_array.emplace(byte_buffer);

    /// upload buffer and texture handle to device memory
    stream << bindless_array->upload_buffer_handles(true) << synchronize();
    stream << bindless_array->upload_texture3d_handles(true) << synchronize();

    stream << vert.upload(vertices.data())
           << tri.upload(triangles.data());

    //    BufferView<float4> f4v = byte_buffer.view_as<float4>();

    Callable add = [&](Float a, Float b) {
        return a + b;
    };
    Pair pa;

    std::tuple<int, float> tp;

    List<float4x4,  SOA, ByteBuffer> lst= create_list<Elm, SOA>(std::move(byte_buffer));

    ManagedList<Elm> ml = device.create_managed_list<Elm>(10, "");

    auto mcd = ml.upload();
    auto cmd = ml.download();
    ml.upload_immediately();
    ml.download_immediately();



    traverse_tuple(tp, [&](auto elm) {
        int i = 0;
    });
    //    byte_buffer.handle_ = fbuffer.handle_;
    //    fbuffer.handle_ = byte_buffer.handle_;
    Kernel kernel = [&](Var<Pair> p, BufferVar<Triple> triangle,BindlessArrayVar ba,
                        ByteBufferVar byte_buffer_var, BufferVar<float3> vert_buffer) {
//List< float4x4 ,SOA, BindlessArrayByteBuffer> list = create_list<float4x4, SOA>(ba.byte_buffer_var(byte_handle));
//List< float4x4 ,SOA, BindlessArrayByteBuffer> list = create_list<float4x4, SOA>(bindless_array.byte_buffer_var(byte_handle));
        List<float4x4 ,  SOA,ByteBufferVar> list = create_list<float4x4, SOA>(byte_buffer_var);
        //        return ;
        //        auto soa = ba.byte_buffer_var(byte_handle).soa_view_var<Elm>();
        //                auto soa = ba.byte_buffer_var(byte_handle).aos_view_var<Elm>();
        //        auto soa = lst.buffer().soa_view_var<Elm>();
        //        auto soa = list.buffer().soa_view_var<Elm>();
//        list.count() = 2;
        list.push_back( make_float4x4(dispatch_id() * 1.f));
        ml.write(dispatch_id(), make_float4x4(dispatch_id() * 2.f));
        //        list.write(dispatch_id(), make_float4x4(dispatch_id() * 1.f));
        //      fbuffer.write(11, float4x4{6});
        //      $info("{} ", list.advance_index());
        //      auto soa1 = soa;a
//        auto soa = byte_buffer_var.soa_view_var<Elm>();
//        soa.write(0, make_float4x4(1.f * dispatch_id() + 1));
        //        list.at(dispatch_id()) = make_float4x4(1.f * dispatch_id() + 1);
                        Var a = list.read(dispatch_id());
        //
        //                Uint2 aa = make_uint2(1);
        //                Float2 bb = make_float2(1.5f);
        //
        //                bb += bb + aa;
        //                lst.count() = 20;
        //                list.advance_index();
                        $info("\n {} {} {} {}  \n"
                              "{} {} {} {}  \n"
                              "{} {} {} {}  \n"
                              "{} {} {} {}  {}\n",
                              a[0], a[1], a[2], a[3], list.storage_size_in_byte());

        //                $info("{} {}   ", bb);
    };
    Triple triple1{1, 2, 3};

    /// set debug range
    Env::debugger().set_lower(make_uint2(0));
    Env::debugger().set_upper(make_uint2(1));
    auto shader = device.compile(kernel, "test desc");
    lst.clear_immediately();
    stream << Env::debugger().upload() ;
    stream << lst.clear() ;
    stream << shader(pa, tri, bindless_array,byte_buffer.view(), vert).dispatch(5)
           /// explict retrieve log
           << byte_buffer.download(byte_vec.data(), 0)
           << ml.download()
           << Env::printer().retrieve()
           << synchronize() << commit();

    auto ijk = lst.host_count();

    RegistrableList<Elm, SOA> rl{move(lst)};

    int iii = 0;
}

struct Test {
    Uint a;
};

template<EPort p = EPort::D>
[[nodiscard]] oc_float<p> PDF_wi_transmission(const oc_float<p> &PDF_wh, const oc_float3<p> &wo, const oc_float3<p> &wh,
                                              const oc_float3<p> &wi, const oc_float<p> &eta) {
    return $outline {
//        oc_float<p> denom = sqr(dot(wi, wh) * eta + dot(wo, wh));
//        oc_float<p> dwh_dwi = abs_dot(wi, wh) / denom;
//        oc_float<p> ret = PDF_wh * dwh_dwi;
        return PDF_wh + eta;
    };
}

void test_lambda(Device &device, Stream &stream) {
    auto [vertices, triangles] = get_cube();

    {
        float3 a, b;
        static_assert(match_basic_func_v<float, float>);
    }

    Buffer<float3> vert = device.create_buffer<float3>(vertices.size());
    Buffer tri = device.create_buffer<Triple>(triangles.size());

    BindlessArray bindless_array = device.create_bindless_array();
    uint v_idx = bindless_array.emplace(vert);
    stream << bindless_array->upload_buffer_handles(true) << synchronize();
    stream << vert.upload(vertices.data())
           << tri.upload(triangles.data());

    float3 f3 = make_float3(1, 2, 3);
    auto f2 = make_int2(5, 6);

    f3.xy() += f2;
    static_assert(is_device_swizzle<Swizzle<Var<float>, 3, 0,1>,2>::value);

    auto f34 = make_float2(f3.xy());

    _bstr_t _bstr;
    ////    float3 aa = f3.xyy() + f3.xyy();
    //    auto fcc = ~make_uint3(f3);
    //    1 == aa.xy();
    //    float3 bb = f3 + f3.xyz();
    //    float3 cc = 5 + f3.xyz();

    //    std::remove_cvref_t<decltype(f3.xyy())>::vec_type __a;

    //    aa.xy() == 10;
    //    f3 =  2.f + f3.xyz;

    float4 f4 = make_float4(1, 2, 666, 4);
    float4 f = (make_float4(-1, -2, -3, -4).xyzw());

    float fe = dot(f.xxx(), f.zww());

    auto bnan = ocarina::cross(f.zyx(), f.zxx());

    //    max(fe, fe);

    //    auto fm = max(f4.xyxz(), f4.xyxz());

    auto bs = ocarina::detail::is_swizzle_impl<std::remove_cvref_t<decltype(f4.xyz())>>::value;

    auto fn = select(make_bool4(1, 0, 1, 0), f4, f);

    bool aaa = match_dsl_unary_func_v<decltype(f.xyz())>;

    auto rb = device.create_byte_buffer(2 * sizeof(Ray));

    auto rb2 = device.create_buffer<Ray>(2);

    vector<Ray> host_ray;
    host_ray.emplace_back(make_float3(1,2,3), float3(4,5,6));
    host_ray.emplace_back(make_float3(7,9,9), float3(7,6,3));

    rb.upload_immediately(host_ray.data());
    rb2.upload_immediately(host_ray.data());
    ByteBufferView rbv = rb.view();
    auto sv = SOAView<Ray, ByteBufferView>(rbv);

    //    auto inv = int4::rcp_impl(f4);
    //    auto ab = float4::abs_impl(make_float4(-1).xxxx());
    //    auto ab2 = absf(make_int4(-1));
    //    AVector<float, 4> af;
    //    Vector<float, 4> af1;
    //    bool abaa = ocarina::is_vector2_v<ocarina::detail::VectorStorage<int, 2>>;
    Kernel kernel = [&](Var<ulong> i, ByteBufferVar rbn) {
//        ocarina::SOAViewVar<Ray, ByteBufferVar> soa;
//            auto ray = rb2.read(0);
        auto rays = make_aos_view_var<Ray>(rbn);
            auto ray2 = rb.load_as<Ray>(0);
            auto ray3 = rays.read(0);
        $info("{} {} {} {}    {}  {}", ray2.dir_max, i, i);
        $info("{} {} {} {}", ray3.dir_max);
//
//        Float3x4 m3 = float3x4{};
//        Float4x3 m4 = float4x3{};
//        $info("{} {} {} {}", m4[0]);

//        auto arr = DynamicArray<float>{1};
//        outline("principled transmission", [&] {
//            rcp(arr);
//        });
        return ;
//        float_array *arr;
//        Float f1 = 6;
//        Float3 f3;


//        $outline{
//            arr = new float_array({1, 5});
//            int a  = 0;
////            Float a = (*arr)[0];
////            f3.xyz() = make_float3(2,5,6);
////            return a;
//        };
////        f1 = PDF_wi_transmission((*arr)[1], f3,f3,f3,f1);
//        Float a = (*arr)[0];
////        int j = 0;
//        $info("{} {} {}", (*arr)[0], (*arr)[1], f1);
//        $info("{} {} {}", f3.xyz().decay());

        //        Float *p;
        //        TriangleHitVar *hit;
        //        Float b;
        //
        //        Float3 f3;
        //
        //        f3.x = 1;
        //        f3.y = 2;
        //        stk.push_back(102u);
        //        stk.push_back(101u);
        //        stk.at(0) = 9;
        //        atomic_add(stk.count(), 10u);
        //        stk.count() = 0;
        //        $info("{} {} {}", stk.at(0), stk.at(1), stk.count());
        //        $info("{} {} {}", stk.at(2), stk.at(3), stk.count());
        //        Float2x3 tran0 = float2x3{};
        //        float3x2 mat(1,2,3,4,5,6);
        //        Float3x2 mat2 = mat;
        //        Float2x3 tran = transpose(mat);
        //        $info("{} {} {}", tran[0]);
        //        $info("{} {} {}", tran[1]);
        //        vert.at(0).x = 10.f;
        //        auto attt = vert.at(2) + 10;
        return ;

////        float3 f3 = make_float3(1, 2, 3);
////
////        auto fm = max(1.f, f3);
//
//        Float3 aa;
//
//        auto func = []<typename Arg>(Arg &arg) {
//            arg.xyz() = make_float3(5,6,7);
//            arg.xy() += arg.z;
//            arg.xy() += arg.yz();
//            arg.xy() += arg.yz();
//            arg.xy() = arg.xy() + arg.x;
//            arg.xy() = arg.x + arg.xy();
//            arg.xy() = arg.xy() + arg.xy();
//            arg.xy() = arg.xy() + arg.xy();
//
//            int i = 0;
//        };
//        func(f3);
//        func(aa);
//        //        aa.xy() == aa.xy();
//
//        //      aa.xy += 1;
//        //      Float3 bbb = + aa.xyy();
//        //        bool bbb = ocarina::is_scalar_v<Float3>;
//        $info("{} {} {} {} func ", aa, aa.zyx()[0]);
//        aa = aa.zxx();
//        $info("{} {} {}  ", aa);
//        auto at = aa >= aa.yyy();
//        $info("{} {} {}  {}  ", make_uint3(at), none(at.xyz()).cast<int>());
//        //        Float3 aac = 19.f;
//
//        auto ma = max(f3.xyz(), 2.f);
//
//        int fdgsi = 0;
//        auto ax = rcp(aa.x);
//        //        $info("{} {} {}  ", aac.call_min(aac, -19.f));
//        $info("{} max_  ", max(aa.x, aa.y));
//        $info("{} {}  {}  ", aa);
//        {
//            Float3 t = make_float3(7,8,9);
//            Float3 a = make_float3(2, 4, 6);
//            float3 b = make_float3(1, 2, 3);
//            float3 rgb = clamp(b, 0.f, 1.f);
//            Uint3 ui = make_uint3(7,8,9);
//
//            a = -a.xyz() ;
//            $info("{} {}  {}  aa ", a);
//
//            Float3 t2 = make_float3(t.zyx());
//            //
//            //            DynamicArray<float> fa{123.f};
//            auto axyz = a.xyz();
//            //            auto axy = select(make_bool2(true), make_float2(1),make_float2(2).xy());
//            Float3 sel = select(make_bool3(1,0,0).xyz(), a.xyz(), b.xyz());
//
//            sel = face_forward(a, b.xyz(),a.xyz());
//            auto mf = make_float4(a.xyz(), t.x);
//
//            //            using tp = decltype(make_float4(remove_device_t<std::remove_cvref_t<decltype(a.xyz())>>{}, remove_device_t<Float>{}));
//
//            //            a.xyz() * t.x;
//
//            max(a.xyz(), b);
//            $info("{} {}  {}  call_select ", sel);
//            $info("{} {}  {}  call_lerp ", lerp(t, b.xyz(),a));
            //            $info("{} {}  {}  {} ", t2, fa[0]);
//        }
        //        f3 = xyz;

        //        $outline {
        //
        //            Var<Triple> ttt;
        //            Var<TriangleHit> hit;
        //
        //            //            auto sn = hit.expression()->type()->simple_cname();
        //
        //            $outline {
        //                b = 123;
        //
        //                $outline {
        //                    p = new Float(15);
        //                    //                    hit = new TriangleHitVar{};
        //                    //                    *p = b;
        //                    //                    b = 19;
        //                };
        //            };
        //            //            $info("{}   i  ---   ", *p);
        //        };
        //        $outline {
        //            //            Float a = *p;
        //            //            //        Float bb = $outline {
        //            //            //            return (*hit).inst_id;
        //            //            //        };
        //            ////            b = 10;
        //
        //            $info("{}    {}  ---   ", min(b, *p), call<float>("oc_max", b, *p));
        //        };
    };
    Shader shader = device.compile(kernel);
    vector<uint> vvv;
    vvv.resize(101, 0);
    uint ui{0u};



    stream << shader(ulong(1-2), rb).dispatch(1)
           //           << stk.download(vvv.data())
           //           << stk.view(400, 4).upload(&ui)
//           << shader(1).dispatch(1)
           << Env::printer().retrieve()
           << synchronize() << commit();

    //    stk.host_count();

    int i = 0;
}

struct Base {
    int a = 1;
    int b = 3;
    virtual ~Base() = default;
    virtual Base &operator=(const Base &base) noexcept = default;
};

struct Derive1 : public Base {
    int d = 9;
    SP<int> sp;
    Derive1(int arg) : d(arg) {}
    virtual ~Derive1() = default;
};

struct Derive : public Base {
    int c = 9;
    SP<int> sp;
    Derive(int arg) : c(arg) {}

    Derive &operator=(const Base &other) noexcept override {
        //        Derive *ptr = dynamic_cast<decltype(this) >(const_cast<Base *>(&other));
        *this = *dynamic_cast<decltype(this)>(const_cast<Base *>(&other));

        return *this;
    }

    Derive &operator=(const Derive &other) noexcept {

        Base::operator=(other);
        *sp = *other.sp;
        c = other.c;
        //        *this = dynamic_cast<decltype(*this) &>(const_cast<Base &>(other));

        return *this;
    }
};

void func(deep_copy_shared_ptr<Derive> d) {
}

void func(deep_copy_unique_ptr<Derive> d) {
}

void test_poly() {
    deep_copy_shared_ptr<Base> p1 = make_deep_copy_shared<Derive>(1);
    deep_copy_shared_ptr<Derive> p2 = make_deep_copy_shared<Derive>(2);

    deep_copy_unique_ptr<Base> p3 = make_deep_copy_unique<Derive>(1);
    deep_copy_unique_ptr<Derive> p4 = make_deep_copy_unique<Derive>(2);
    //    p0 = p1;
    //    p0 = p1;
    func(p2);
    p1->a = 10;
    //    deep_copy_shared_ptr<Base> p2 = make_deep_copy_shared<Derive>(2);
    p2->a = 8;

    dynamic_cast<Derive *>(p1.get())->sp = std::make_shared<int>(123);
    dynamic_cast<Derive *>(p2.get())->sp = std::make_shared<int>(456);
    //
    cout << "before p1->a = " << p1->a << ", p1->c = " << dynamic_cast<Derive *>(p1.get())->c << endl;
    cout << "before p2->a = " << p2->a << ", p2->c = " << dynamic_cast<Derive *>(p2.get())->c << endl;
    p1 = p2;
    cout << "after p1->a = " << p1->a << ", p1->c = " << dynamic_cast<Derive *>(p1.get())->c << endl;
    cout << "after p2->a = " << p2->a << ", p2->c = " << dynamic_cast<Derive *>(p2.get())->c << endl;
}

void test_parameter_struct(Device &device, Stream &stream) {
    auto [vertices, triangles] = get_cube();

    Buffer<float3> vert = device.create_buffer<float3>(vertices.size());
    Buffer tri = device.create_buffer<Triple>(triangles.size());

    stream << vert.upload(vertices.data());
    stream << tri.upload(triangles.data());

    Param p;
    p.b = vert.descriptor();
    p.t = tri.descriptor();
    auto sss = alignof( Triple);
    auto ofs0 = offsetof(Param, i);
    auto ofs1 = offsetof(Param, b);
    auto ofs2 = offsetof(Param, t);
    auto ofs3 = offsetof(Param, pa);
    auto ofs4 = offsetof(Param, b2);
    //    p.pa.b = vert.proxy();

    Kernel kernel = [&](Var<Param> pp) {
        //        $info("{} ", pp.pa.b.at(dispatch_id()).x);
        //        vert.at(dispatch_id()).x += 90;
//        pa.triple.h.bary = make_float2(1.f);
//        $outline {
//            auto v = pa.triple.h.bary.xy();
//            int i = 0;
//            //            auto v = pp.pa.b.read(dispatch_id());
//            //            $info("{} {} {}  -- ", v);
//        };
                atomic_add(pp.t.at(0).i, 5.6f);
                atomic_sub(pp.t.at(0).j, 1);
                atomic_exch(pp.t.at(dispatch_id()).k, dispatch_id() * 25 + 2);
                auto j = atomic_CAS(pp.t.at(0).j, 4294967295u, 100u);
                auto v =  pp.t.at(dispatch_id()) ;
                $info("{} {} {}  {}  {}", v.i, v.j, v.k, j, pp.t.at(0).j);
    };
    auto shader = device.compile(kernel, "param struct");

    stream << shader(p).dispatch(2);
    stream << Env::printer().retrieve();
    stream << synchronize() << commit();
}

template<size_t N, size_t M>
[[nodiscard]] constexpr auto mul(ocarina::Matrix<N, M> m, ocarina::Vector<float, M> v) noexcept {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ((v[i] * m[i]) + ...);
    }(std::make_index_sequence<M>());
}

template<size_t N, size_t M, size_t Dim>
[[nodiscard]] constexpr auto mul(ocarina::Matrix<N, Dim> lhs, ocarina::Matrix<Dim, M> rhs) noexcept {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ocarina::Matrix<N, M>(mul(lhs , rhs[i])...);
    }(std::make_index_sequence<M>());
}

int main(int argc, char *argv[]) {

    float3x2 m2x = float3x2(1.f, 1.f, 1.f, 1.f, 1.f ,1.f);

    auto m = (float3x2() * float2x3());
    cout << to_str(m) << endl;

    auto m2 = (float2x3() * float3x2());
    cout << to_str(m2) << endl;
//
    auto m3 = (float4x2()* float2x3());
    cout << to_str(m3) << endl;
    fs::path path(argv[0]);
    RHIContext &context = RHIContext::instance();

    /**
     * Conventional scheme
     * create device and stream
     * stream used for process some command,e.g buffer upload and download, dispatch shader
     * default is asynchronous operation
     */
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    Env::printer().init(device);
    Env::debugger().init(device);

    //    Env::set_code_obfuscation(true);
    Env::set_valid_check(false);

    /// create rtx context if need
    device.init_rtx();

    //    ocarina::detail::Vector<float, 3> aaaaa;

    //    AVector<float, 3> aaaa;

    float3 a = make_float3(1, 2, 3);
    auto la = a.xz() < 1.5f;
    int3 b = make_int3(4, 5, 6);

    a += a;
    a += 1;
    a *= a;
    a = 1 + a;
    bool4 bool_4 = make_bool4(1, 0, 1, 1);
    auto bbb = bool_4 || bool_4.xxxx();
    auto b4 = all(bool_4.ww());

//    test_compute_shader(device, stream);
        test_parameter_struct(device, stream);
//        test_lambda(device, stream);

    //    test_poly();
    return 0;
}