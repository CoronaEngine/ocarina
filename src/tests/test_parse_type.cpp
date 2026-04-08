//
// Created by Zero on 30/05/2022.
//

#include "ast/type_registry.h"
#include "core/stl.h"

using namespace ocarina;

struct ParseLeaf {
    float value;
    Vector<uint, 2> index;
};

OC_MAKE_STRUCT_REFLECTION(ParseLeaf, value, index)
OC_MAKE_STRUCT_DESC(ParseLeaf, value, index)

struct ParseBuiltin {
    bool enabled;
    float weight;
};

OC_MAKE_BUILTIN_STRUCT(ParseBuiltin)
OC_MAKE_STRUCT_REFLECTION(ParseBuiltin, enabled, weight)
OC_MAKE_STRUCT_DESC(ParseBuiltin, enabled, weight)

struct ParseParam {
    Vector<float, 4> color;
    uint count;
};

OC_MAKE_PARAM_STRUCT(ParseParam)
OC_MAKE_STRUCT_REFLECTION(ParseParam, color, count)
OC_MAKE_STRUCT_DESC(ParseParam, color, count)

struct ParseNested {
    ParseLeaf leaf;
    ocarina::array<float, 4> samples;
};

OC_MAKE_STRUCT_REFLECTION(ParseNested, leaf, samples)
OC_MAKE_STRUCT_DESC(ParseNested, leaf, samples)

static bool check_impl(bool condition, string_view message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << std::endl;
        return false;
    }
    return true;
}

#define CHECK(...)                            \
    do {                                      \
        if (!check_impl((__VA_ARGS__), #__VA_ARGS__)) { \
            return false;                     \
        }                                     \
    } while (false)

static string_view invalid_case_desc(string_view case_name) {
    if (case_name == "invalid-data-type") {
        return "not_a_type";
    }
    if (case_name == "invalid-vector-element") {
        return "vector<texture2d,4>";
    }
    if (case_name == "invalid-matrix-dimension") {
        return "matrix<float,5,5>";
    }
    return {};
}

static int run_invalid_case(string_view case_name) {
    string_view desc = invalid_case_desc(case_name);
    if (desc.empty()) {
        std::cerr << "unknown invalid case: " << case_name << std::endl;
        return 2;
    }
    const Type *type = TypeRegistry::instance().parse_type(desc);
    (void)type;
    std::cerr << "invalid case unexpectedly succeeded: " << case_name << std::endl;
    return 3;
}

static bool expect_invalid_parse_failure(const char *exe_path, string_view case_name) {
    string command = ocarina::format("\"{}\" --invalid-case {} >nul 2>&1", exe_path, case_name);
    int exit_code = std::system(command.c_str());
    CHECK(exit_code != 0);
    return true;
}

static bool test_void_and_scalar_types(TypeRegistry &registry) {
    CHECK(registry.parse_type(TypeDesc<void>::description()) == nullptr);

    const Type *float_type = registry.parse_type(TypeDesc<float>::description());
    CHECK(float_type != nullptr);
    CHECK(float_type->is_scalar());
    CHECK(float_type->tag() == Type::Tag::FLOAT);
    CHECK(float_type->size() == sizeof(float));
    CHECK(float_type->alignment() == alignof(float));
    CHECK(float_type->dimension() == 1);
    CHECK(float_type->description() == TypeDesc<float>::description());
    CHECK(float_type->name() == "float");
    CHECK(registry.is_exist(TypeDesc<float>::description()));
    CHECK(registry.parse_type(TypeDesc<float>::description()) == float_type);

    const Type *bool_type = registry.parse_type(TypeDesc<bool>::description());
    CHECK(bool_type != nullptr);
    CHECK(bool_type->tag() == Type::Tag::BOOL);
    CHECK(bool_type->size() == sizeof(bool));
    CHECK(bool_type->alignment() == alignof(bool));
    CHECK(bool_type->dimension() == 1);

    const Type *half_type = registry.parse_type(TypeDesc<half>::description());
    CHECK(half_type != nullptr);
    CHECK(half_type->tag() == Type::Tag::HALF);
    CHECK(half_type->size() == sizeof(half));
    CHECK(half_type->alignment() == alignof(half));
    return true;
}

static bool test_vector_and_matrix_types(TypeRegistry &registry) {
    using Float3 = Vector<float, 3>;
    using Float4 = Vector<float, 4>;
    using Float3x4 = Matrix<float, 3, 4>;

    const Type *float_type = Type::of<float>();
    const Type *float3_type = registry.parse_type(TypeDesc<Float3>::description());
    CHECK(float3_type != nullptr);
    CHECK(float3_type->is_vector());
    CHECK(float3_type->tag() == Type::Tag::VECTOR);
    CHECK(float3_type->dimension() == 3);
    CHECK(float3_type->element() == float_type);
    CHECK(float3_type->members().size() == 1);
    CHECK(float3_type->size() == sizeof(Float3));
    CHECK(float3_type->alignment() == alignof(Float3));
    CHECK(float3_type->name() == "float3");
    CHECK(registry.parse_type(TypeDesc<Float3>::description()) == float3_type);

    const Type *float4_type = registry.parse_type(TypeDesc<Float4>::description());
    CHECK(float4_type != nullptr);
    CHECK(float4_type->is_vector());
    CHECK(float4_type->dimension() == 4);
    CHECK(float4_type->element() == float_type);

    const Type *matrix_type = registry.parse_type(TypeDesc<Float3x4>::description());
    CHECK(matrix_type != nullptr);
    CHECK(matrix_type->is_matrix());
    CHECK(matrix_type->tag() == Type::Tag::MATRIX);
    CHECK(matrix_type->dimension() == 3);
    CHECK(matrix_type->members().size() == 1);
    CHECK(matrix_type->element() == float4_type);
    CHECK(matrix_type->size() == sizeof(Float3x4));
    CHECK(matrix_type->alignment() == alignof(Float3x4));
    CHECK(matrix_type->name() == "float3x4");
    CHECK(registry.parse_type(TypeDesc<Float3x4>::description()) == matrix_type);
    return true;
}

static bool test_array_and_resource_types(TypeRegistry &registry) {
    using Float4 = Vector<float, 4>;
    using Float4Array = ocarina::array<Float4, 5>;

    const Type *float4_type = Type::of<Float4>();
    const Type *array_type = registry.parse_type(TypeDesc<Float4Array>::description());
    CHECK(array_type != nullptr);
    CHECK(array_type->is_array());
    CHECK(array_type->tag() == Type::Tag::ARRAY);
    CHECK(array_type->dimension() == 5);
    CHECK(array_type->element() == float4_type);
    CHECK(array_type->size() == float4_type->size() * 5u);
    CHECK(array_type->alignment() == float4_type->alignment());
    CHECK(array_type->is_valid());

    const Type *buffer_type = registry.parse_type(TypeDesc<Buffer<float, 2, 3>>::description());
    CHECK(buffer_type != nullptr);
    CHECK(buffer_type->is_buffer());
    CHECK(buffer_type->tag() == Type::Tag::BUFFER);
    CHECK(buffer_type->element() == Type::of<float>());
    CHECK(buffer_type->dims().size() == 2);
    CHECK(buffer_type->dims()[0] == 2);
    CHECK(buffer_type->dims()[1] == 3);
    CHECK(buffer_type->size() == sizeof(BufferDesc<>));
    CHECK(buffer_type->alignment() == alignof(BufferDesc<>));

    const Type *byte_buffer_type = registry.parse_type(TypeDesc<ByteBuffer>::description());
    CHECK(byte_buffer_type != nullptr);
    CHECK(byte_buffer_type->is_byte_buffer());
    CHECK(byte_buffer_type->tag() == Type::Tag::BYTE_BUFFER);
    CHECK(byte_buffer_type->alignment() == alignof(BufferDesc<>));

    const Type *texture2d_type = registry.parse_type(TypeDesc<Texture2D>::description());
    CHECK(texture2d_type != nullptr);
    CHECK(texture2d_type->tag() == Type::Tag::TEXTURE2D);
    CHECK(texture2d_type->size() == sizeof(TextureDesc));
    CHECK(texture2d_type->alignment() == alignof(TextureDesc));

    const Type *texture3d_type = registry.parse_type(TypeDesc<Texture3D>::description());
    CHECK(texture3d_type != nullptr);
    CHECK(texture3d_type->tag() == Type::Tag::TEXTURE3D);
    CHECK(texture3d_type->size() == sizeof(TextureDesc));
    CHECK(texture3d_type->alignment() == alignof(TextureDesc));

    const Type *bindless_array_type = registry.parse_type(TypeDesc<BindlessArray>::description());
    CHECK(bindless_array_type != nullptr);
    CHECK(bindless_array_type->tag() == Type::Tag::BINDLESS_ARRAY);
    CHECK(bindless_array_type->alignment() == alignof(BindlessArrayDesc));

    const Type *accel_type = registry.parse_type(TypeDesc<Accel>::description());
    CHECK(accel_type != nullptr);
    CHECK(accel_type->tag() == Type::Tag::ACCEL);
    return true;
}

static bool test_tuple_and_struct_types(TypeRegistry &registry) {
    using TupleType = ocarina::tuple<uint, uint, Vector<float, 2>>;
    using SampleArray = ocarina::array<float, 4>;

    const Type *tuple_type = registry.parse_type(TypeDesc<TupleType>::description());
    CHECK(tuple_type != nullptr);
    CHECK(tuple_type->is_structure());
    CHECK(tuple_type->tag() == Type::Tag::STRUCTURE);
    CHECK(tuple_type->cname() == "_Tuple");
    CHECK(tuple_type->members().size() == 3);
    CHECK(tuple_type->members()[0] == Type::of<uint>());
    CHECK(tuple_type->members()[1] == Type::of<uint>());
    CHECK(tuple_type->members()[2] == Type::of<Vector<float, 2>>());
    CHECK(tuple_type->size() == sizeof(TupleType));
    CHECK(tuple_type->alignment() == alignof(TupleType));
    CHECK(!tuple_type->is_builtin_struct());
    CHECK(!tuple_type->is_param_struct());

    const Type *leaf_type = Type::of<ParseLeaf>();
    CHECK(leaf_type != nullptr);
    CHECK(leaf_type->is_structure());
    CHECK(leaf_type->description() == TypeDesc<ParseLeaf>::description());
    CHECK(leaf_type->cname() == "ParseLeaf");
    CHECK(leaf_type->simple_cname() == "ParseLeaf");
    CHECK(leaf_type->members().size() == 2);
    CHECK(leaf_type->member_name().size() == 2);
    CHECK(leaf_type->member_name()[0] == "value");
    CHECK(leaf_type->member_name()[1] == "index");
    CHECK(leaf_type->get_member("value") == Type::of<float>());
    CHECK(leaf_type->get_member("index") == Type::of<Vector<uint, 2>>());
    CHECK(leaf_type->size() == sizeof(ParseLeaf));
    CHECK(leaf_type->alignment() == alignof(ParseLeaf));
    CHECK(!leaf_type->is_builtin_struct());
    CHECK(!leaf_type->is_param_struct());

    const Type *builtin_type = Type::of<ParseBuiltin>();
    CHECK(builtin_type != nullptr);
    CHECK(builtin_type->is_structure());
    CHECK(builtin_type->is_builtin_struct());
    CHECK(!builtin_type->is_param_struct());
    CHECK(builtin_type->members().size() == 2);
    CHECK(builtin_type->get_member("enabled") == Type::of<bool>());
    CHECK(builtin_type->get_member("weight") == Type::of<float>());
    CHECK(builtin_type->size() == sizeof(ParseBuiltin));
    CHECK(builtin_type->alignment() == alignof(ParseBuiltin));

    const Type *param_type = Type::of<ParseParam>();
    CHECK(param_type != nullptr);
    CHECK(param_type->is_structure());
    CHECK(!param_type->is_builtin_struct());
    CHECK(param_type->is_param_struct());
    CHECK(param_type->members().size() == 2);
    CHECK(param_type->get_member("color") == Type::of<Vector<float, 4>>());
    CHECK(param_type->get_member("count") == Type::of<uint>());
    CHECK(param_type->size() == sizeof(ParseParam));
    CHECK(param_type->alignment() == alignof(ParseParam));

    const Type *nested_type = Type::of<ParseNested>();
    CHECK(nested_type != nullptr);
    CHECK(nested_type->is_structure());
    CHECK(nested_type->members().size() == 2);
    CHECK(nested_type->get_member("leaf") == leaf_type);
    CHECK(nested_type->get_member("samples") == Type::of<SampleArray>());
    CHECK(nested_type->size() == sizeof(ParseNested));
    CHECK(nested_type->alignment() == alignof(ParseNested));
    CHECK(registry.parse_type(TypeDesc<ParseNested>::description()) == nested_type);
    return true;
}

static bool test_invalid_descriptions(const char *exe_path) {
    CHECK(expect_invalid_parse_failure(exe_path, "invalid-data-type"));
    CHECK(expect_invalid_parse_failure(exe_path, "invalid-vector-element"));
    CHECK(expect_invalid_parse_failure(exe_path, "invalid-matrix-dimension"));
    return true;
}

int main(int argc, char **argv) {
    if (argc == 3 && string_view(argv[1]) == "--invalid-case") {
        return run_invalid_case(argv[2]);
    }

    TypeRegistry &registry = TypeRegistry::instance();

    bool passed = true;
    passed = test_void_and_scalar_types(registry) && passed;
    passed = test_vector_and_matrix_types(registry) && passed;
    passed = test_array_and_resource_types(registry) && passed;
    passed = test_tuple_and_struct_types(registry) && passed;
    passed = test_invalid_descriptions(argv[0]) && passed;

    if (!passed) {
        std::cerr << "type parse test failed" << std::endl;
        return 1;
    }
    std::cout << "type parse test passed" << std::endl;
    return 0;
}