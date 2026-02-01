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

int main() {
    RHIContext &context = RHIContext::instance();
    Device device = context.create_device("cuda");
    device.init_rtx();
    Stream stream = device.create_stream();
    Env::printer().init(device);

    Kernel kernel = [&](Float f) {
        $info("{} ", f);
    };
    auto shader = device.compile(kernel);
    stream << shader(6.f).dispatch(1) << Env::printer().retrieve();
    stream << synchronize() << commit();

    half h = 0.66f;



    float3 f3 = make_float3(1.f, 2.f, 3.f);
    half3 h2(f3);


    float3x3 f3x3 = make_float3x3(3.f);
    half3x3 h3x3(f3x3);

    cout << to_str(h3x3 * h3x3) << endl;

    h = 100.66;
    auto a = 1.f+h ;

    cout << 1 + h << endl;
    cout << h + h << endl;
    cout << (h += h) << endl;
//    cout <<  (h+=2) ;

    return 0;
}