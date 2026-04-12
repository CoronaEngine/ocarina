//
// Created by Zero on 2024/11/3.
//

#include "pyexporter/ocapi.h"

namespace py = pybind11;
using namespace ocarina;

void export_vector(PythonExporter &exporter);
void export_matrix(PythonExporter &exporter);
void export_scalar_op(PythonExporter &exporter) {
    auto &m = exporter.module;
    using Tuple = std::tuple<uint, int, float>;
    traverse_tuple(Tuple{}, [&]<typename Src>(const Src &_, uint index) {
        traverse_tuple(Tuple{}, [&]<typename Dst>(const Dst &_, uint index) {
            if constexpr (std::is_same_v<Src, Dst>) {
                return;
            }
            string func_name = ocarina::format("as_{}", TypeDesc<Dst>::name());
            m.def(func_name.c_str(), [&](const Src &src) { return ocarina::bit_cast<Dst>(src); });
        });
    });
}

void export_math(PythonExporter &exporter) {
    export_vector(exporter);
    export_scalar_op(exporter);
    export_matrix(exporter);
}