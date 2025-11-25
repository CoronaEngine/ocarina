//
// Created by Zero on 19/03/2025.
//

#include "core/image.h"
#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "math/base.h"
#include "base/scattering/interaction.h"
#include "core/platform.h"
#include "rhi/context.h"

using namespace ocarina;

int main(int argc, char *argv[]) {

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

    uint num = 200000000;
    uint float_num = 4 * num;
    uint size = num * sizeof(float4);
    ByteBuffer b0 = device.create_byte_buffer(size);
    ByteBuffer b1 = device.create_byte_buffer(size);

    vector<float> cb0, cb1;

    cb0.resize(float_num);
    cb1.resize(float_num);

    for (int i = 0; i < float_num; ++i) {
        cb0[i] = i;
    }

//    b0.upload_immediately(cb0.data());

    Kernel kernel = [&](Uint num) {
//        Uint index = dispatch_id() * sizeof(float4);
//        Float4 f4 = b0.load_as<float4>(index);
//        b1.store(index, f4);
        Uint index = dispatch_id();
        for (int i = 0; i < 4; ++i) {
            Uint idx = index * sizeof(float4) + i * sizeof(float);
            Float f = b0.load_as<float>(idx);
//            $info("{}" ,f);
            b1.store(idx, f);
        }
    };


    auto shader = device.compile(kernel);

    Clock clk;
    clk.start();
    stream << shader(6).dispatch(num) << synchronize() << commit();
    auto ms = clk.elapse_ms();
    Env::printer().retrieve_immediately();
//    b1.download_immediately(cb1.data());
    std::cout << "it take " << ms << std::endl;

    return 0;
}