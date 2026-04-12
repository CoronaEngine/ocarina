//
// Created by Zero on 30/04/2022.
//

#include "type_registry.h"

namespace ocarina {

TypeRegistry &TypeRegistry::instance() noexcept {
    static TypeRegistry type_registry;
    return type_registry;
}

const Type *TypeRegistry::parse_type(ocarina::string_view desc) noexcept {
    return Type::from(desc);
}

const Type *TypeRegistry::type_from(ocarina::string_view desc) noexcept {
    return Type::from(desc);
}

size_t TypeRegistry::type_count() const noexcept {
    return Type::count();
}

const Type *TypeRegistry::type_at(uint i) const noexcept {
    return Type::at(i);
}

uint64_t TypeRegistry::compute_hash(ocarina::string_view desc) noexcept {
    return Hashable::compute_hash<Type>(hash64(desc));
}

bool TypeRegistry::is_exist(ocarina::string_view desc) const noexcept {
    return Type::exists(desc);
}

bool TypeRegistry::is_exist(uint64_t hash) const noexcept {
    return Type::exists(hash);
}

void TypeRegistry::for_each(TypeVisitor *visitor) const noexcept {
    Type::for_each(visitor);
}

}// namespace ocarina