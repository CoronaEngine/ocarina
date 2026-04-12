//
// Created by Zero on 30/04/2022.
//

#include "type_registry.h"
#include "function.h"

namespace ocarina {

namespace {

void on_type_access_bridge(const Type *type) noexcept {
    TypeRegistry::try_add_to_current_function(type);
}

void register_type_system_callbacks_once() noexcept {
    static bool registered = [] {
        detail::register_type_system_callbacks(detail::TypeSystemCallbacks{
            .on_type_access = &on_type_access_bridge,
        });
        return true;
    }();
    (void)registered;
}

[[maybe_unused]] const bool type_system_callbacks_registered = [] {
    register_type_system_callbacks_once();
    return true;
}();

}// namespace

TypeRegistry &TypeRegistry::instance() noexcept {
    register_type_system_callbacks_once();
    static TypeRegistry type_registry;
    return type_registry;
}
const Type *TypeRegistry::parse_type(ocarina::string_view desc) noexcept {
    return Type::from(desc);
}

void TypeRegistry::try_add_to_current_function(const ocarina::Type *type) noexcept {
    if (auto f = Function::current(); f != nullptr && type->is_structure()) {
        f->add_used_structure(type);
    }
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