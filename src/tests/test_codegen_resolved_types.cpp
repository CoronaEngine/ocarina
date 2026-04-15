#include <iostream>
#include <string_view>

#include "backends/cuda/cuda_codegen.h"
#include "core/type_system/precision_policy.h"
#include "dsl/dsl.h"
#include "generator/cpp_codegen.h"
#include "math/real.h"

using namespace ocarina;

struct CodegenPolicyRecord {
    real weight;
    Vector<real, 2> uv;
};

OC_STRUCT(, CodegenPolicyRecord, weight, uv) {
};

struct CodegenNestedInner {
    real weight;
    Vector<real, 2> uv;
};

OC_STRUCT(, CodegenNestedInner, weight, uv) {
};

struct CodegenNestedRecord {
    CodegenNestedInner inner;
    real gain;
};

OC_STRUCT(, CodegenNestedRecord, inner, gain) {
};

namespace {

[[nodiscard]] bool check_impl(bool condition, const char *expr) {
    if (!condition) {
        std::cerr << "FAILED: " << expr << std::endl;
    }
    return condition;
}

#define CHECK(...)                            \
    do {                                      \
        if (!check_impl((__VA_ARGS__), #__VA_ARGS__)) { \
            return false;                     \
        }                                     \
    } while (false)

[[nodiscard]] StoragePrecisionPolicy make_policy(PrecisionPolicy policy) {
    return StoragePrecisionPolicy{.policy = policy, .allow_real_in_storage = false};
}

[[nodiscard]] bool contains(std::string_view source, std::string_view needle) {
    return source.find(needle) != std::string_view::npos;
}

[[nodiscard]] const Type *resolved_type(const Type *type, PrecisionPolicy precision) {
    return Type::resolve(type, make_policy(precision));
}

[[nodiscard]] std::string emit_cpp_source(const Function &function) {
    CppCodegen codegen(false);
    codegen.emit(function);
    return std::string(codegen.scratch().view());
}

[[nodiscard]] std::string emit_cuda_source(const Function &function) {
    CUDACodegen codegen(false);
    codegen.emit(function);
    return std::string(codegen.scratch().view());
}

template<PrecisionPolicy precision>
[[nodiscard]] shared_ptr<Function> make_callable_function() {
    StoragePrecisionPolicy previous_policy = global_storage_policy();
    set_global_storage_policy(make_policy(precision));
    Callable callable = [](Var<CodegenPolicyRecord> record, Var<real> bias) {
        return record.weight + bias + record.uv.x;
    };
    callable.function()->set_description(precision == PrecisionPolicy::force_f16 ? "callable_f16" : "callable_f32");
    auto function = callable.function();
    set_global_storage_policy(previous_policy);
    return function;
}

template<PrecisionPolicy precision>
[[nodiscard]] shared_ptr<Function> make_kernel_function() {
    StoragePrecisionPolicy previous_policy = global_storage_policy();
    set_global_storage_policy(make_policy(precision));
    Kernel kernel = [](Var<real> bias, Var<CodegenPolicyRecord> record) {
        Var<real> sum = record.weight + bias + record.uv.y;
        comment("resolved type kernel body");
        sum = sum + cast<real>(1.0f);
    };
    kernel.function()->set_description(precision == PrecisionPolicy::force_f16 ? "kernel_f16" : "kernel_f32");
    auto function = kernel.function();
    set_global_storage_policy(previous_policy);
    return function;
}

template<PrecisionPolicy precision>
[[nodiscard]] shared_ptr<Function> make_nested_kernel_function() {
    StoragePrecisionPolicy previous_policy = global_storage_policy();
    set_global_storage_policy(make_policy(precision));
    Kernel kernel = [](Var<real> bias, Var<CodegenNestedRecord> record) {
        Var<CodegenNestedRecord> local_record = record;
        Var<CodegenNestedInner> local_inner = local_record.inner;
        Var<real> sum = local_inner.weight + local_record.gain + bias + local_inner.uv.x;
        comment("resolved nested kernel body");
        sum = sum + cast<real>(1.0f);
    };
    kernel.function()->set_description(precision == PrecisionPolicy::force_f16 ? "nested_kernel_f16" : "nested_kernel_f32");
    auto function = kernel.function();
    set_global_storage_policy(previous_policy);
    return function;
}

[[nodiscard]] bool expect_cpp_markers(std::string_view source, PrecisionPolicy precision) {
    if (precision == PrecisionPolicy::force_f16) {
        CHECK(contains(source, "half "));
        CHECK(contains(source, "half2"));
        CHECK(!contains(source, "real "));
    } else {
        CHECK(contains(source, "float "));
        CHECK(contains(source, "float2"));
        CHECK(!contains(source, "real "));
    }
    return true;
}

[[nodiscard]] bool expect_cuda_markers(std::string_view source, PrecisionPolicy precision) {
    if (precision == PrecisionPolicy::force_f16) {
        CHECK(contains(source, "oc_half "));
        CHECK(contains(source, "oc_half2"));
        CHECK(!contains(source, "oc_real"));
    } else {
        CHECK(contains(source, "oc_float "));
        CHECK(contains(source, "oc_float2"));
        CHECK(!contains(source, "oc_real"));
    }
    return true;
}

template<PrecisionPolicy precision>
[[nodiscard]] bool test_callable_codegen() {
    auto function = make_callable_function<precision>();
    std::string cpp_source = emit_cpp_source(*function);
    std::string cuda_source = emit_cuda_source(*function);
    CHECK(contains(cpp_source, precision == PrecisionPolicy::force_f16 ? "callable_f16" : "callable_f32"));
    CHECK(contains(cuda_source, "__device__"));
    CHECK(expect_cpp_markers(cpp_source, precision));
    CHECK(expect_cuda_markers(cuda_source, precision));
    return true;
}

template<PrecisionPolicy precision>
[[nodiscard]] bool test_callable_ast_types() {
    auto function = make_callable_function<precision>();
    CHECK(function->arguments().size() == 2u);
    CHECK(function->arguments()[0].type() == resolved_type(Type::of<CodegenPolicyRecord>(), precision));
    CHECK(function->arguments()[1].type() == resolved_type(Type::of<real>(), precision));
    CHECK(function->return_type() == resolved_type(Type::of<real>(), precision));
    return true;
}

template<PrecisionPolicy precision>
[[nodiscard]] bool test_kernel_codegen() {
    auto function = make_kernel_function<precision>();
    std::string cpp_source = emit_cpp_source(*function);
    std::string cuda_source = emit_cuda_source(*function);
    CHECK(contains(cpp_source, precision == PrecisionPolicy::force_f16 ? "kernel_f16" : "kernel_f32"));
    CHECK(contains(cuda_source, "extern \"C\" __global__"));
    CHECK(expect_cpp_markers(cpp_source, precision));
    CHECK(expect_cuda_markers(cuda_source, precision));
    return true;
}

template<PrecisionPolicy precision>
[[nodiscard]] bool test_nested_kernel_ast_and_codegen() {
    auto function = make_nested_kernel_function<precision>();
    const Type *resolved_record = resolved_type(Type::of<CodegenNestedRecord>(), precision);
    const Type *resolved_inner = resolved_type(Type::of<CodegenNestedInner>(), precision);
    const Type *resolved_real = resolved_type(Type::of<real>(), precision);
    CHECK(function->arguments().size() == 2u);
    CHECK(function->arguments()[0].type() == resolved_real);
    CHECK(function->arguments()[1].type() == resolved_record);
    CHECK(function->body()->local_vars().size() >= 3u);
    CHECK(function->body()->local_vars()[0].type() == resolved_record);
    CHECK(function->body()->local_vars()[1].type() == resolved_inner);
    CHECK(function->body()->local_vars()[2].type() == resolved_real);
    bool saw_record = false;
    bool saw_inner = false;
    function->for_each_structure([&](const Type *type) {
        if (type == resolved_record) {
            saw_record = true;
        }
        if (type == resolved_inner) {
            saw_inner = true;
        }
    });
    CHECK(saw_record);
    CHECK(saw_inner);

    std::string cpp_source = emit_cpp_source(*function);
    std::string cuda_source = emit_cuda_source(*function);
    CHECK(contains(cpp_source, precision == PrecisionPolicy::force_f16 ? "nested_kernel_f16" : "nested_kernel_f32"));
    CHECK(contains(cuda_source, "extern \"C\" __global__"));
    CHECK(expect_cpp_markers(cpp_source, precision));
    CHECK(expect_cuda_markers(cuda_source, precision));
    return true;
}

}// namespace

int main() {
    bool passed = true;
    passed = test_callable_ast_types<PrecisionPolicy::force_f16>() && passed;
    passed = test_callable_ast_types<PrecisionPolicy::force_f32>() && passed;
    passed = test_callable_codegen<PrecisionPolicy::force_f16>() && passed;
    passed = test_callable_codegen<PrecisionPolicy::force_f32>() && passed;
    passed = test_kernel_codegen<PrecisionPolicy::force_f16>() && passed;
    passed = test_kernel_codegen<PrecisionPolicy::force_f32>() && passed;
    passed = test_nested_kernel_ast_and_codegen<PrecisionPolicy::force_f16>() && passed;
    passed = test_nested_kernel_ast_and_codegen<PrecisionPolicy::force_f32>() && passed;
    if (!passed) {
        return 1;
    }
    std::cout << "codegen resolved type test passed" << std::endl;
    return 0;
}
