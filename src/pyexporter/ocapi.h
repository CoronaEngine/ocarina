//
// Created by Zero on 2024/11/3.
//

#pragma once

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "ext/pybind11/include/pybind11/operators.h"
#include "core/stl.h"
#include "core/type_desc.h"
#include "rhi/common.h"
#include "numpy.h"
#include "base/using.h"

namespace py = pybind11;
#define OC_EXPORT_ENUM_VALUE(v, enum_type) \
    .value(#v, enum_type::v)

#define OC_EXPORT_ENUM(m, Enum, ...) \
    py::enum_<Enum>(m, #Enum)        \
        MAP_UD(OC_EXPORT_ENUM_VALUE, Enum, ##__VA_ARGS__);

#define OC_EXPORT_STRUCT_MEMBER(member, S) \
    .def_readwrite(#member, &S::member)

#define OC_EXPORT_STRUCT(m, S, ...)      \
    py::class_<S>(m, #S).def(py::init()) \
        MAP_UD(OC_EXPORT_STRUCT_MEMBER, S, ##__VA_ARGS__);

struct PythonExporter {
    py::module module;
    UP<py::class_<Device, concepts::Noncopyable>> m_device;
};

using ret_policy = py::return_value_policy;

struct Context {
    static Context *s_context;
    UP<Device> device;
    UP<Stream> stream;
    BindlessArray bindless_array;
    [[nodiscard]] static Context &instance() noexcept;
    static void destroy_instance() noexcept;
    Context() {
        OC_INFO("ocapi load!");
    }
    ~Context() {
        OC_INFO("ocapi unload!");
    }
};

template<typename T, typename... Base>
requires(is_basic_v<T> || is_struct_v<T>)
auto export_pod_type(PythonExporter &exporter, const char *name = nullptr) {
    auto &m = exporter.module;
    name = name ? name : TypeDesc<T>::name().data();
    auto mt = py::class_<T, Base...>(m, name);
    mt.def_static("desc", []() {
        return TypeDesc<T>::description();
    });
    mt.def("clone", [](const T &self) {
        return T{self};
    });
    mt.def_static("sizeof", []() {
        return sizeof(T);
    });
    exporter.module.def("hash64", [](const T &arg) {
        return hash64(arg);
    });
    mt.def_static("alignof", []() {
        return alignof(T);
    });
    mt.def_static("max_member_size", []() {
        return Type::of<T>()->max_member_size();
    });
    mt.def("to_bytes", [](T &self) {
        return py::bytes(reinterpret_cast<const char *>(addressof(self)), sizeof(T));
    });
    mt.def_static("from_bytes", [](const py::bytes &byte) {
        T ret;
        oc_memcpy(addressof(ret), py::buffer(byte).request().ptr, sizeof(T));
        return ret;
    });
    mt.def("to_floats", [](T &self) {
        py::array_t<float, alignof(T)> ret{sizeof(T) / sizeof(float)};
        oc_memcpy(ret.request().ptr, addressof(self), sizeof(T));
        return ret;
    });
    mt.def_static("from_floats", [](const py::array_t<float> &arr) {
        T ret;
        oc_memcpy(addressof(ret), arr.request().ptr, sizeof(T));
        return ret;
    });
    mt.def("__repr__", [](const T &self) {
        return to_str(self);
    });
    mt.def(py::init<>());
    return mt;
}

template<typename Elm, typename T>
decltype(auto) get_member_by_offset(T &t, uint offset) {
    return *reinterpret_cast<Elm *>(reinterpret_cast<std::byte *>(&t) + offset);
}

template<typename Elm, typename T>
decltype(auto) get_member_by_offset(const T &t, uint offset) {
    return *reinterpret_cast<const Elm *>(reinterpret_cast<const std::byte *>(&t) + offset);
}

template<typename T, typename... Base>
requires(is_struct_v<T>)
auto export_struct(PythonExporter &exporter) {
    string_view cname = TypeDesc<T>::name();
    static string_view simple_name = cname.substr(cname.find_last_of("::") + 1);
    auto mt = export_pod_type<T, Base...>(exporter, simple_name.data());
    traverse_tuple(struct_member_tuple_t<T>{}, [&]<typename Elm>(const Elm &_, uint index) {
        auto getter = [index = index](const T &self) {
            auto ofs = struct_member_tuple<T>::offset_array[index];
            return get_member_by_offset<Elm>(self, ofs);
        };
        auto setter = [index = index](T &self, const Elm &val) {
            auto ofs = struct_member_tuple<T>::offset_array[index];
            get_member_by_offset<Elm>(self, ofs) = val;
        };
        mt.def_property(struct_member_tuple<T>::members[index].data(), getter, setter);
    });
    return mt;
}