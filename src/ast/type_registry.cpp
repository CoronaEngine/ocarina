//
// Created by Zero on 30/04/2022.
//

#include "type_registry.h"
#include "core/hash.h"
#include "core/util.h"
#include "core/logging.h"
#include "dsl/rtx_type.h"

namespace ocarina {

TypeRegistry &TypeRegistry::instance() noexcept {
    static TypeRegistry type_registry;
    return type_registry;
}

namespace detail {
[[nodiscard]] bool is_letter(char ch) noexcept {
    return std::isalpha(ch) || ch == '_';
}

[[nodiscard]] bool is_letter_or_num(char ch) noexcept {
    return std::isalnum(ch) || ch == '_' || ch == ':';
}

[[nodiscard]] bool is_num(char ch) noexcept {
    return ch >= '0' && ch <= '9';
}

[[nodiscard]] std::pair<int, int> bracket_matching_far(ocarina::string_view str, char l = '<', char r = '>') {
    int start = 0;
    int end = 0;
    int pair_count = 0;
    for (int i = 0; i < str.size(); ++i) {
        char ch = str[i];
        if (ch == l) {
            if (pair_count == 0) {
                start = i;
            }
            pair_count += 1;
        } else if (ch == r) {
            pair_count -= 1;
            if (pair_count == 0) {
                end = i;
            }
        }
    }
    return std::make_pair(start, end);
}

[[nodiscard]] std::pair<int, int> bracket_matching_near(ocarina::string_view str, char l = '<', char r = '>') {
    int start = -1;
    int end = -1;
    int pair_count = 0;
    for (int i = 0; i < str.size(); ++i) {
        char ch = str[i];
        if (ch == l) {
            if (pair_count == 0) {
                start = i;
            }
            pair_count += 1;
        } else if (ch == r) {
            pair_count -= 1;
            if (pair_count == 0) {
                end = i;
            }
        }
        if (pair_count == 0 && start != -1 && end != -1) {
            break;
        }
    }
    return std::make_pair(start, end);
}

[[nodiscard]] ocarina::string_view find_identifier(ocarina::string_view &str,
                                                   bool check_start_with_num = false) {
    OC_USING_SV
    uint i = 0u;
    for (; i < str.size() && is_letter_or_num(str[i]); ++i)
        ;
    auto ret = str.substr(0, i);
    if (ret == "vector"sv ||
        ret == "matrix"sv ||
        ret == "struct"sv ||
        ret == "buffer"sv ||
        ret == "texture3d"sv ||
        ret == "array"sv) {
        auto [start, end] = bracket_matching_near(str);
        ret = str.substr(0, end + 1);
        str = str.substr(end + 1);
    } else {
        str = str.substr(i);
    }
    if (is_num(ret[0]) && check_start_with_num) [[unlikely]] {
        OC_ERROR_FORMAT("invalid identifier {} !", ret)
    }
    return ret;
}

[[nodiscard]] auto find_content(ocarina::string_view &str, char l = '<', char r = '>') {
    ocarina::vector<ocarina::string_view> ret;
    OC_USING_SV
    auto prev_token = str.find(l);
    constexpr auto token = ',';
    str = str.substr(prev_token + 1);
    uint count = 0;
    constexpr uint limit = 10000;
    while (true) {
        auto content = find_identifier(str);
        auto new_cursor = str.find(token) + 1;
        str = str.substr(new_cursor);
        ++count;
        if (count > limit) {
            OC_ERROR("The number of loops has exceeded the upper limit. Please check the code");
        }
        ret.push_back(content);
        if (str[0] == r) {
            break;
        }
    }
    return ret;
}

}// namespace detail

/*
 * TYPE: BASIC | ARRAY | VECTOR | MATRIX | STRUCT | BUFFER
 * BASIC: int | uint | bool | float
 * ARRAY: array<BASIC | STRUCTURE, N>
 * BUFFER : buffer<BASIC | STRUCTURE>
 * VECTOR: vector<BASIC,2> | vector<BASIC,3> | vector<BASIC,4>
 * MATRIX: matrix<2> | matrix<3> | matrix<4>
 * STRUCT: struct<4,TYPE...> | struct<8,TYPE...> | struct<16,TYPE...>
 */
const Type *TypeRegistry::parse_type(ocarina::string_view desc) noexcept {
    if (desc == "void") {
        return nullptr;
    }
    uint64_t hash = compute_hash(desc);
    if (auto iter = type_set_.find(hash); iter != type_set_.cend()) {
        try_add_to_current_function(*iter);
        return *iter;
    }

    OC_USING_SV

    auto type = ocarina::make_unique<Type>();

#define OC_PARSE_BASIC_TYPE(T, TAG)    \
    if (desc == #T##sv) {              \
        type->size_ = sizeof(T);       \
        type->tag_ = Type::Tag::TAG;   \
        type->alignment_ = alignof(T); \
        type->set_description(#T);     \
        type->dimension_ = 1;          \
    } else

    OC_PARSE_BASIC_TYPE(int, INT)
    OC_PARSE_BASIC_TYPE(uint, UINT)
    OC_PARSE_BASIC_TYPE(bool, BOOL)
    OC_PARSE_BASIC_TYPE(float, FLOAT)
    OC_PARSE_BASIC_TYPE(half, HALF)
    OC_PARSE_BASIC_TYPE(uchar, UCHAR)
    OC_PARSE_BASIC_TYPE(char, CHAR)
    OC_PARSE_BASIC_TYPE(ushort, USHORT)
    OC_PARSE_BASIC_TYPE(ulong, ULONG)
    OC_PARSE_BASIC_TYPE(short, SHORT)

#undef OC_PARSE_BASIC_TYPE

    if (desc.starts_with("vector")) {
        parse_vector(type.get(), desc);
    } else if (desc.starts_with("matrix")) {
        parse_matrix(type.get(), desc);
    } else if (desc.starts_with("array")) {
        parse_array(type.get(), desc);
    } else if (desc.starts_with("struct")) {
        parse_struct(type.get(), desc);
    } else if (desc.starts_with("bytebuffer")) {
        parse_byte_buffer(type.get(), desc);
    } else if (desc.starts_with("buffer")) {
        parse_buffer(type.get(), desc);
    } else if (desc.starts_with("texture3d")) {
        parse_texture3d(type.get(), desc);
    } else if (desc.starts_with("texture2d")) {
        parse_texture2d(type.get(), desc);
    } else if (desc.starts_with("accel")) {
        parse_accel(type.get(), desc);
    } else if (desc.starts_with("bindlessArray")) {
        parse_bindless_array(type.get(), desc);
    } else {
        OC_ERROR("invalid data type ", desc);
    }
    type->set_description(desc);
    const Type *ret = type.get();
    add_type(std::move(type));
    return ret;
}

void TypeRegistry::parse_vector(Type *type, ocarina::string_view desc) noexcept {
    type->tag_ = Type::Tag::VECTOR;
    auto [start, end] = detail::bracket_matching_far(desc, '<', '>');
    auto content = desc.substr(start + 1, end - start - 1);
    auto lst = string_split(content, ',');
    OC_ASSERT(lst.size() == 2);
    auto type_str = lst[0];
    auto dimension_str = lst[1];
    auto dimension = std::stoi(string(dimension_str));
    type->dimension_ = dimension;
    type->members_.push_back(parse_type(type_str));
    auto member = type->members_.front();
    if (!member->is_scalar()) [[unlikely]] {
        OC_ERROR("invalid vector element: {}!", member->description());
    }
    type->size_ = member->size() * (dimension == 3 ? 4 : dimension);
    type->alignment_ = type->size();
}

void TypeRegistry::parse_matrix(Type *type, ocarina::string_view desc) noexcept {
    type->tag_ = Type::Tag::MATRIX;
    auto [start, end] = detail::bracket_matching_far(desc, '<', '>');
    auto dimension_str = desc.substr(start + 1, end - start - 1);
    auto dims = string_split(dimension_str, ',');
    int N = std::stoi(string(dims[0]));
    int M = std::stoi(string(dims[1]));
    type->dimension_ = N;
    auto tmp_desc = ocarina::format("vector<float,{}>", M);
    type->members_.push_back(parse_type((tmp_desc)));

#define OC_SIZE_ALIGN(NN, MM)                       \
    if (N == (NN) && M == (MM)) {                   \
        type->size_ = sizeof(Matrix<NN, MM>);       \
        type->alignment_ = alignof(Matrix<NN, MM>); \
    } else
    OC_SIZE_ALIGN(2, 2)
    OC_SIZE_ALIGN(2, 3)
    OC_SIZE_ALIGN(2, 4)
    OC_SIZE_ALIGN(3, 2)
    OC_SIZE_ALIGN(3, 3)
    OC_SIZE_ALIGN(3, 4)
    OC_SIZE_ALIGN(4, 2)
    OC_SIZE_ALIGN(4, 3)
    OC_SIZE_ALIGN(4, 4) {
        OC_ERROR("invalid matrix dimension <{}, {}>!", N, M);
    }
#undef OC_SIZE_ALIGN
}

void TypeRegistry::parse_struct(Type *type, string_view desc) noexcept {
    type->tag_ = Type::Tag::STRUCTURE;
    auto lst = detail::find_content(desc);
    type->cname_ = lst[0];
    auto alignment_str = lst[1];
    bool is_builtin_struct = lst[2] == "true";
    type->builtin_struct_ = is_builtin_struct;
    bool is_param_struct = lst[3] == "true";
    type->param_struct_ = is_param_struct;
    auto alignment = std::stoi(string(alignment_str));
    type->alignment_ = alignment;
    auto size = 0u;
    static constexpr uint member_offset = 4;
    for (int i = member_offset; i < lst.size(); ++i) {
        auto type_str = lst[i];
        type->members_.push_back(parse_type(type_str));
        auto member = type->members_[i - member_offset];
        size = mem_offset(size, member->alignment());
        size += member->size();
    }
    type->size_ = mem_offset(size, type->alignment());
}

void TypeRegistry::parse_bindless_array(Type *type, ocarina::string_view desc) noexcept {
    type->tag_ = Type::Tag::BINDLESS_ARRAY;
    type->alignment_ = alignof(BindlessArrayDesc);
}

void TypeRegistry::parse_buffer(Type *type, ocarina::string_view desc) noexcept {
    type->tag_ = Type::Tag::BUFFER;
    auto lst = detail::find_content(desc);
    auto type_str = lst[0];
    const Type *element_type = parse_type(type_str);
    type->members_.push_back(element_type);
    for (int i = 1; i < lst.size(); ++i) {
        type->dims_.push_back(std::stoi(lst[i].data()));
    }
    type->alignment_ = alignof(BufferDesc<>);
    type->size_ = sizeof(BufferDesc<>);
}

void TypeRegistry::parse_texture3d(Type *type, ocarina::string_view desc) noexcept {
    type->tag_ = Type::Tag::TEXTURE3D;
    type->alignment_ = alignof(TextureDesc);
    type->size_ = sizeof(TextureDesc);
}

void TypeRegistry::parse_texture2d(Type *type, ocarina::string_view desc) noexcept {
    type->tag_ = Type::Tag::TEXTURE2D;
    type->alignment_ = alignof(TextureDesc);
    type->size_ = sizeof(TextureDesc);
}


void TypeRegistry::parse_accel(Type *type, ocarina::string_view desc) noexcept {
    type->tag_ = Type::Tag::ACCEL;
}

void TypeRegistry::parse_byte_buffer(ocarina::Type *type, ocarina::string_view desc) noexcept {
    type->tag_ = Type::Tag::BYTE_BUFFER;
    type->alignment_ = alignof(BufferDesc<>);
}

void TypeRegistry::parse_array(Type *type, ocarina::string_view desc) noexcept {
    type->tag_ = Type::Tag::ARRAY;
    auto lst = detail::find_content(desc);
    auto type_str = lst[0];
    auto len = std::stoi(string(lst[1]));
    const Type *element_type = parse_type(type_str);
    type->members_.push_back(element_type);
    auto alignment = element_type->alignment();
    auto size = element_type->size() * len;
    type->alignment_ = alignment;
    type->dimension_ = len;
    type->size_ = size;
}

void TypeRegistry::add_type(ocarina::unique_ptr<Type> type) {
    type_set_.insert(type.get());
    type->index_ = types_.size();
    try_add_to_current_function(type.get());
    types_.push_back(std::move(type));
}

void TypeRegistry::try_add_to_current_function(const ocarina::Type *type) noexcept {
    if (auto f = Function::current(); f != nullptr && type->is_structure()) {
        f->add_used_structure(type);
    }
}

const Type *TypeRegistry::type_from(ocarina::string_view desc) noexcept {
    return parse_type(desc);
}

size_t TypeRegistry::type_count() const noexcept {
    std::unique_lock lock{mutex_};
    return types_.size();
}

const Type *TypeRegistry::type_at(uint i) const noexcept {
    std::unique_lock lock{mutex_};
    return types_[i].get();
}

uint64_t TypeRegistry::compute_hash(ocarina::string_view desc) noexcept {
    return Hashable::compute_hash<Type>(hash64(desc));
}
bool TypeRegistry::is_exist(ocarina::string_view desc) const noexcept {
    return is_exist(compute_hash(desc));
}

bool TypeRegistry::is_exist(uint64_t hash) const noexcept {
    return type_set_.find(hash) != type_set_.cend();
}

void TypeRegistry::for_each(TypeVisitor *visitor) const noexcept {
    for (const auto &t : types_) {
        visitor->visit(t.get());
    }
}

}// namespace ocarina