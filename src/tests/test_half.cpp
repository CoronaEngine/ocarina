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

    return 0;
}