//
// Created by z on 21/01/2026.
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

//template<typename A, typename B, typename C>
//requires any_device_type_v<A, B, C> && requires { lerp(remove_device_t<A>{}, remove_device_t<B>{}, remove_device_t<B>{}); }
//[[nodiscard]] auto lerp(const A &a, const B &b, const C &c) noexcept {
//    static constexpr auto dimension = std::max({type_dimension_v<remove_device_t<A>>, type_dimension_v<remove_device_t<B>>, type_dimension_v<remove_device_t<C>>});
//    using scalar_type = type_element_t<remove_device_t<A>>;
//    using var_type = Var<general_vector_t<scalar_type, dimension>>;
//    static_assert(dimension == 3);
//    return MemberAccessor::lerp<var_type>(a, b, a);
//}
//
//template<typename A, typename B, typename C>
//requires any_device_type_v<A, B, C> && requires { lerp(remove_device_t<A>{}, remove_device_t<B>{}, remove_device_t<B>{}); }
// auto lerp(const A &a, const B &b, const C &c) noexcept {
//    static constexpr auto dimension = std::max({type_dimension_v<remove_device_t<A>>, type_dimension_v<remove_device_t<B>>, type_dimension_v<remove_device_t<C>>});
//    using scalar_type = type_element_t<remove_device_t<A>>;
//    using var_type = Var<general_vector_t<scalar_type, dimension>>;
//    return MemberAccessor::lerp<var_type>(
//
//
//        to_general_vector<dimension>(a),
//        to_general_vector<dimension>(b),
////        to_general_vector<dimension>(c),
//            to_general_vector<dimension>(c));
//}

int main() {
    RHIContext &context = RHIContext::instance();
    Device device = context.create_device("cuda");
    device.init_rtx();
    Stream stream = device.create_stream();
    Env::printer().init(device);

    auto ts = TypeDesc<half3x3>::name();

    float3 h3;
    half3x3 h33;

    auto rrr = int{-1} + uint{0};

    auto re = uint3{}.x + int3{1,2,3}.xyz();

    auto lll = lerp(float{} ,float3 {}, float3{});

    auto res = h33 * float{};

    Buffer<float3> buffer = device.create_buffer<float3>(1);




    Kernel kernel = [&](Float f) {
//        $info("{} ", f);
        sqrt(Half{});
//        Uint3 a = make_uint3(3);
//        Int3 b ;
//        b = a;

        Float3 f3 = make_float3(1.f,5.f, 9.f);
        Half3 h3 = make_float3(1.f, 2.f, 3.f);
//        buffer.write(0, h3);
//        Float4 f4 = make_float4(h3,half( 1.f));
//        h3 = f3;

        h3 = f3.xyz();

        auto bbb = ocarina::select(false , half{} , float{});

//        auto a3 = ocarina::select(make_bool3(1,0,1).xyz(), uint3{5,6,7}, float3{1,2,3}.xyz());

//        auto re = f3.x * h3.xyz() ;

//        re = h3.xxx();

//      $info("------  {} {} {} ", Uint3{});
      $info("---  {} {} {} ", sqrt(h3) );
      $info("------==  {} {} {} ", Uint3{});
    };
    auto shader = device.compile(kernel);
    stream << shader(6.f).dispatch(1) << Env::printer().retrieve();
    stream << synchronize() << commit();

    half h = 0.66f;





//    float3 f3 = make_float3(1.f, 2.f, 3.f);
//    half3 h2(f3);
//    auto m = ocarina::max(h2, h2);
//
//    float3x3 f3x3 = make_float3x3(3.f);
//    half3x3 h3x3(f3x3);
//    cout << to_str(m) << endl;
//    cout << to_str(h3x3 * h3x3) << endl;
//
//    h = 100.66;
//    auto a = 1.f+h ;
//
//    cout << 1 + h << endl;
//    cout << h + h << endl;
//    cout << (h += h) << endl;
////    cout <<  (h+=2) ;

    return 0;
}