//
// Created by Zero on 21/09/2022.
//

#include <utility>
#include "core/stl.h"
#include "dsl/dsl.h"
#include "dsl/polymorphic.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "rhi/context.h"
#include <type_traits>

using namespace ocarina;

struct Data : public Encodable{
    EncodedData<float> f;
    EncodedData<float4> f4;

    OC_ENCODABLE_FUNC(Encodable,f, f4)
};

struct Data2 : public Data {
    EncodedData<float3> f3;
    OC_ENCODABLE_FUNC(Data, f3)
};

struct Test : public Encodable{
    EncodedData<float2> a;
    EncodedData<int3> b;
    EncodedData<float> c;
    EncodedData<int> d;
    EncodedData<vector<float>> e;
    EncodedData<float3x3> f;
    RegistrableManaged<float> mw;
    Data2 data;
    OC_ENCODABLE_FUNC(Encodable,a, b, c, d, e, f,mw, data)
};



union oc_scalar{
    int i;
    uint u;
    float f;
};

void test(Device &device, Stream &stream) {
    Test t;
    t.a = make_float2(1,2);
    t.a = [&]() {
        return make_float2(1.1,2.2);
    };
    t.b = make_int3(3,4,5);
    t.c = 6.8f;
    t.d = 100;
    for (int i = 0; i < 3; ++i) {
        t.e.hv().push_back(i);
    }
    t.f = make_float3x3(56.1f);
    t.data.f = 106;
    t.data.f4 = make_float4(199.f);
    BindlessArray ra = device.create_bindless_array();
    t.mw.set_bindless_array(ra);
    t.mw.register_self();
    t.mw.push_back(9.98);
    t.mw.push_back(9.98);
    RegistrableManaged<buffer_ty> vv(ra);

    oc_scalar os{.f = 2.3f};
    os.f = 2.f;

    vv.resize(t.aligned_size());
    t.encode(vv);
    vv.reset_device_buffer_immediately(device);
    vv.upload_immediately();
    vv.register_self();
    stream << ra->upload_buffer_handles(true) << synchronize();



    Kernel kernel = [&](Float a) {
        DataAccessor da{0u, vv};
        t.decode(&da);
        Env::printer().info("a = {} {}", t.a.dv());
        Env::printer().info("b = {} {} {}", t.b.dv());
        Env::printer().info("c = {}", t.c.dv());
        Env::printer().info("d = {}", t.d.dv());
        Env::printer().info("e = {} {} {}", t.e.dv().as_vec3());
        Env::printer().info("f0 = {} {} {}", t.f.dv()[0]);
        Env::printer().info("f1 = {} {} {}", t.f.dv()[1]);
        Env::printer().info("f2 = {} {} {}", t.f.dv()[2]);
        Env::printer().info("data.f = {}", t.data.f.dv());
        Env::printer().info("data.f4 = {} {} {} {}", *t.data.f4);
    };
    auto shader = device.compile(kernel);
    stream << shader(1.5f).dispatch(1);
    stream << synchronize() << commit();
    Env::printer().retrieve_immediately();
}

struct Mat : public Encodable {
    EncodedData<float> a;
    EncodedData<float> b;
//    EncodedData<float> c;
    EncodedData<float> d;
    EncodedData<vector<float>> e;
    OC_ENCODABLE_FUNC(Encodable, a, b,  d, e)
};

void test2(Device &device, Stream &stream) {
    BindlessArray ba = device.create_bindless_array();

    Mat m;
    m.a = 0.25f;
    m.b = 0.5f;
//    m.c = 0.75;
    m.d = 1;
    m.e.hv().push_back(0.5f);
    m.e.hv().push_back(0.25f);
    m.a.set_encode_type(Uint8);
    m.b.set_encode_type(Uint8);
//    m.c.set_encode_type(Uint8);
    m.d.set_encode_type(Uint8);
    m.e.set_encode_type(Uint8);

    auto as = m.aligned_size();

    RegistrableManaged<buffer_ty> vv(ba);
    vv.resize(m.aligned_size());
    m.encode(vv);
    vv.reset_device_buffer_immediately(device);
    vv.upload_immediately();
    vv.register_self();
//    ba.prepare_slotSOA(device);
    stream << ba->upload_buffer_handles(true) << synchronize();

    Kernel kernel = [&](Float a) {
        DataAccessor da{0u, vv};
//        m.decode(&da);

        auto array = da.load_dynamic_array<buffer_ty>(m.aligned_size() / 4);
        m.decode(array);
        Env::printer().info("a = {}", m.a.dv());
        Env::printer().info("b = {}", m.b.dv());
//        Env::printer().info("c = {}", m.c.dv());
        Env::printer().info("d = {}", m.d.dv());
        Env::printer().info("e = {} {}", m.e.dv().as_vec2());

    };
    auto shader = device.compile(kernel);
    stream << shader(1.5f).dispatch(1);
    stream << synchronize() << commit();
    Env::printer().retrieve_immediately();

    return;
}

int main(int argc, char *argv[]) {
    log_level_debug();

    fs::path path(argv[0]);
    RHIContext &context = RHIContext::instance();
    //    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    Env::printer().init(device);

//    test(device, stream);
    test2(device, stream);


    return 0;

    //    cout << typeid(scalar_t<int3>).name() << endl;
    //    cout << typeid(scalar_t<float>).name() << endl;
    //    cout << typeid(scalar_t<float4x4>).name() << endl;
    //    cout << typeid(scalar_t<Float3>).name() << endl;
    //    cout << typeid(scalar_t<Float>).name() << endl;
    //    cout << typeid(scalar_t<Float4x4>).name() << endl;
    //
    //    cout << endl;
    //
    //    cout << typeid(vec_t<float3, 2>).name() << endl;
    //    cout << typeid(vec_t<float, 2>).name() << endl;
    //    cout << typeid(vec_t<float4x4, 2>).name() << endl;
    //    cout << typeid(vec_t<Float3, 2>).name() << endl;
    //    cout << typeid(vec_t<Float, 2>).name() << endl;
    //    cout << typeid(vec_t<Float4x4, 2>).name() << endl;
    //
    //    cout << endl;
    //
    //    cout << typeid(matrix_t<float3, 2>).name() << endl;
    //    cout << typeid(matrix_t<float, 2>).name() << endl;
    //    cout << typeid(matrix_t<float4x4, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float3, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float4x4, 2>).name() << endl;
    //
    //    cout << typeid(boolean_t<Float>).name() << endl;
    //    cout << typeid(boolean_t<float>).name() << endl;

    return 0;
}