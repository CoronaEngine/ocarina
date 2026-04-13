//
// Created by Z on 12/04/2026.
//

#pragma once

#include "core/dynamic_buffer/host_byte_buffer.h"
#include "core/precision_policy.h"
#include "core/type.h"
#include "math/half.h"
#include "math/real.h"

namespace ocarina {

enum class DynamicBufferLayout : uint8_t {
    aos,
    soa,
};

namespace detail {

[[nodiscard]] inline size_t align_up_size(size_t value, size_t alignment) noexcept {
    OC_ASSERT(alignment != 0u);
    return (value + alignment - 1u) / alignment * alignment;
}

[[nodiscard]] inline bool store_real_as_f32(StoragePrecisionPolicy policy,
                                            bool direct_array_element = false) noexcept {
    (void)direct_array_element;
    return policy.policy == PrecisionPolicy::force_f32;
}

template<typename T>
[[nodiscard]] size_t resolved_size(StoragePrecisionPolicy policy) noexcept;

template<typename T>
[[nodiscard]] size_t resolved_alignment(StoragePrecisionPolicy policy) noexcept;

template<typename T>
[[nodiscard]] size_t resolved_size_in_context(StoragePrecisionPolicy policy,
                                              bool direct_array_element) noexcept;

template<typename T>
[[nodiscard]] size_t resolved_alignment_in_context(StoragePrecisionPolicy policy,
                                                   bool direct_array_element) noexcept;

template<typename T>
[[nodiscard]] size_t resolved_scalar_size(StoragePrecisionPolicy policy,
                                          bool direct_array_element = false) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_real_v<raw_t>) {
        return store_real_as_f32(policy, direct_array_element) ? sizeof(float) : sizeof(uint16_t);
    } else if constexpr (is_half_v<raw_t>) {
        return sizeof(uint16_t);
    } else {
        return sizeof(raw_t);
    }
}

template<typename T>
[[nodiscard]] size_t resolved_scalar_alignment(StoragePrecisionPolicy policy,
                                               bool direct_array_element = false) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_real_v<raw_t>) {
        return store_real_as_f32(policy, direct_array_element) ? alignof(float) : alignof(uint16_t);
    } else if constexpr (is_half_v<raw_t>) {
        return alignof(uint16_t);
    } else {
        return alignof(raw_t);
    }
}

template<typename T>
[[nodiscard]] size_t resolved_matrix_size(StoragePrecisionPolicy policy) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    constexpr size_t rows = matrix_dimension_v<raw_t>;
    using column_t = tuple_element_t<0, struct_member_tuple_t<raw_t>>;
    constexpr size_t cols = vector_dimension_v<column_t>;
    using scalar_t = type_element_t<column_t>;
    if constexpr (is_real_v<scalar_t>) {
        if (policy.policy == PrecisionPolicy::force_f16) {
            return sizeof(Matrix<half, rows, cols>);
        }
        return sizeof(Matrix<float, rows, cols>);
    } else {
        return sizeof(Matrix<scalar_t, rows, cols>);
    }
}

template<typename T>
[[nodiscard]] size_t resolved_matrix_alignment(StoragePrecisionPolicy policy) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    constexpr size_t rows = matrix_dimension_v<raw_t>;
    using column_t = tuple_element_t<0, struct_member_tuple_t<raw_t>>;
    constexpr size_t cols = vector_dimension_v<column_t>;
    using scalar_t = type_element_t<column_t>;
    if constexpr (is_real_v<scalar_t>) {
        if (policy.policy == PrecisionPolicy::force_f16) {
            return alignof(Matrix<half, rows, cols>);
        }
        return alignof(Matrix<float, rows, cols>);
    } else {
        return alignof(Matrix<scalar_t, rows, cols>);
    }
}

template<typename T>
[[nodiscard]] size_t resolved_size(StoragePrecisionPolicy policy) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_scalar_v<raw_t>) {
        return resolved_scalar_size<raw_t>(policy, false);
    } else if constexpr (is_vector_v<raw_t>) {
        using element_t = type_element_t<raw_t>;
        constexpr size_t dim = vector_dimension_v<raw_t>;
        return resolved_scalar_size<element_t>(policy, false) * (dim == 3 ? 4 : dim);
    } else if constexpr (is_matrix_v<raw_t>) {
        return resolved_matrix_size<raw_t>(policy);
    } else if constexpr (is_array_v<raw_t>) {
        using element_t = array_element_t<raw_t>;
        constexpr size_t dim = array_dimension_v<raw_t>;
        return resolved_size_in_context<element_t>(policy, true) * dim;
    } else if constexpr (is_struct_v<raw_t>) {
        size_t size = 0u;
        [&]<size_t... I>(std::index_sequence<I...>) {
            (([&] {
                using member_t = tuple_element_t<I, struct_member_tuple_t<raw_t>>;
                size = align_up_size(size, resolved_alignment<member_t>(policy));
                size += resolved_size<member_t>(policy);
            }()), ...);
        }(std::make_index_sequence<tuple_size_v<struct_member_tuple_t<raw_t>>>{});
        return align_up_size(size, resolved_alignment<raw_t>(policy));
    } else {
        static_assert(always_false_v<raw_t>, "Unsupported type for DynamicBufferLayoutCodec::resolved_size");
    }
}

template<typename T>
[[nodiscard]] size_t resolved_alignment(StoragePrecisionPolicy policy) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_scalar_v<raw_t>) {
        return resolved_scalar_alignment<raw_t>(policy, false);
    } else if constexpr (is_vector_v<raw_t>) {
        return resolved_size<raw_t>(policy);
    } else if constexpr (is_matrix_v<raw_t>) {
        return resolved_matrix_alignment<raw_t>(policy);
    } else if constexpr (is_array_v<raw_t>) {
        using element_t = array_element_t<raw_t>;
        return resolved_alignment_in_context<element_t>(policy, true);
    } else if constexpr (is_struct_v<raw_t>) {
        size_t align = 0u;
        [&]<size_t... I>(std::index_sequence<I...>) {
            (([&] {
                using member_t = tuple_element_t<I, struct_member_tuple_t<raw_t>>;
                align = std::max(align, resolved_alignment<member_t>(policy));
            }()), ...);
        }(std::make_index_sequence<tuple_size_v<struct_member_tuple_t<raw_t>>>{});
        return align;
    } else {
        static_assert(always_false_v<raw_t>, "Unsupported type for DynamicBufferLayoutCodec::resolved_alignment");
    }
}

template<typename T>
[[nodiscard]] size_t resolved_size_in_context(StoragePrecisionPolicy policy,
                                              bool direct_array_element) noexcept {
    (void)direct_array_element;
    if constexpr (is_real_v<T>) {
        return resolved_scalar_size<T>(policy, false);
    }
    return resolved_size<T>(policy);
}

template<typename T>
[[nodiscard]] size_t resolved_alignment_in_context(StoragePrecisionPolicy policy,
                                                   bool direct_array_element) noexcept {
    (void)direct_array_element;
    if constexpr (is_real_v<T>) {
        return resolved_scalar_alignment<T>(policy, false);
    }
    return resolved_alignment<T>(policy);
}

template<size_t I, typename T>
[[nodiscard]] decltype(auto) member_ref(T &value) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_array_v<raw_t> || is_vector_v<raw_t> || is_matrix_v<raw_t>) {
        return (value[I]);
    } else {
        using member_t = tuple_element_t<I, struct_member_tuple_t<raw_t>>;
        constexpr size_t offset = struct_member_tuple<raw_t>::offset_array[I];
        auto *ptr = reinterpret_cast<std::byte *>(addressof(value)) + offset;
        return *std::launder(reinterpret_cast<member_t *>(ptr));
    }
}

template<size_t I, typename T>
[[nodiscard]] decltype(auto) member_ref(const T &value) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_array_v<raw_t> || is_vector_v<raw_t> || is_matrix_v<raw_t>) {
        return (value[I]);
    } else {
        using member_t = tuple_element_t<I, struct_member_tuple_t<raw_t>>;
        constexpr size_t offset = struct_member_tuple<raw_t>::offset_array[I];
        auto *ptr = reinterpret_cast<const std::byte *>(addressof(value)) + offset;
        return *std::launder(reinterpret_cast<const member_t *>(ptr));
    }
}

template<typename T>
[[nodiscard]] constexpr bool is_soa_atomic_v = is_scalar_v<T> || is_vector_v<T>;

template<typename T>
void encode_aos_value(HostByteBuffer &dst,
                      size_t offset,
                      const T &value,
                      StoragePrecisionPolicy policy,
                      bool direct_array_element = false) noexcept;

template<typename T>
[[nodiscard]] T decode_aos_value(const HostByteBuffer &src,
                                 size_t offset,
                                 StoragePrecisionPolicy policy,
                                 bool direct_array_element = false) noexcept;

template<typename T>
void zero_bytes(HostByteBuffer &dst, size_t offset, size_t size) noexcept {
    auto out = dst.subspan(offset, size);
    std::fill(out.begin(), out.end(), std::byte{0});
}

template<typename T>
void encode_scalar(HostByteBuffer &dst,
                   size_t offset,
                   const T &value,
                   StoragePrecisionPolicy policy,
                   bool direct_array_element) noexcept {
    (void)direct_array_element;
    if constexpr (is_real_v<T>) {
        if (policy.policy == PrecisionPolicy::force_f32) {
            dst.store<float>(offset, static_cast<float>(value));
        } else {
            dst.store<uint16_t>(offset, float_to_half(static_cast<float>(value)));
        }
    } else if constexpr (is_half_v<T>) {
        dst.store<uint16_t>(offset, value.bits());
    } else {
        dst.store<T>(offset, value);
    }
}

template<typename T>
[[nodiscard]] T decode_scalar(const HostByteBuffer &src,
                              size_t offset,
                              StoragePrecisionPolicy policy,
                              bool direct_array_element) noexcept {
    (void)direct_array_element;
    if constexpr (is_real_v<T>) {
        if (policy.policy == PrecisionPolicy::force_f32) {
            return real{src.load<float>(offset)};
        }
        return real{half_to_float(src.load<uint16_t>(offset))};
    } else if constexpr (is_half_v<T>) {
        return half{half_to_float(src.load<uint16_t>(offset))};
    } else {
        return src.load<T>(offset);
    }
}

template<typename T>
void encode_aos_value(HostByteBuffer &dst,
                      size_t offset,
                      const T &value,
                      StoragePrecisionPolicy policy,
                      bool direct_array_element) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_scalar_v<raw_t>) {
        encode_scalar(dst, offset, value, policy, direct_array_element);
    } else if constexpr (is_vector_v<raw_t>) {
        using element_t = type_element_t<raw_t>;
        constexpr size_t dim = vector_dimension_v<raw_t>;
        size_t total_size = resolved_size<raw_t>(policy);
        size_t element_size = resolved_size<element_t>(policy);
        zero_bytes<raw_t>(dst, offset, total_size);
        for (size_t index = 0; index < dim; ++index) {
            encode_scalar<element_t>(dst, offset + index * element_size, value[index], policy, false);
        }
    } else if constexpr (is_matrix_v<raw_t>) {
        constexpr size_t dim = matrix_dimension_v<raw_t>;
        using column_t = tuple_element_t<0, struct_member_tuple_t<raw_t>>;
        size_t column_size = resolved_size<column_t>(policy);
        for (size_t index = 0; index < dim; ++index) {
            encode_aos_value<column_t>(dst, offset + index * column_size, value[index], policy, false);
        }
    } else if constexpr (is_array_v<raw_t>) {
        using element_t = array_element_t<raw_t>;
        constexpr size_t dim = array_dimension_v<raw_t>;
        size_t element_size = resolved_size_in_context<element_t>(policy, true);
        for (size_t index = 0; index < dim; ++index) {
            encode_aos_value<element_t>(dst, offset + index * element_size, value[index], policy, true);
        }
    } else if constexpr (is_struct_v<raw_t>) {
        zero_bytes<raw_t>(dst, offset, resolved_size<raw_t>(policy));
        size_t current = offset;
        [&]<size_t... I>(std::index_sequence<I...>) {
            (([&] {
                using member_t = tuple_element_t<I, struct_member_tuple_t<raw_t>>;
                current = align_up_size(current, resolved_alignment<member_t>(policy));
                encode_aos_value<member_t>(dst, current, member_ref<I>(value), policy, false);
                current += resolved_size<member_t>(policy);
            }()), ...);
        }(std::make_index_sequence<tuple_size_v<struct_member_tuple_t<raw_t>>>{});
    } else {
        static_assert(always_false_v<raw_t>, "Unsupported type for DynamicBufferLayoutCodec::encode_aos_value");
    }
}

template<typename T>
[[nodiscard]] T decode_aos_value(const HostByteBuffer &src,
                                 size_t offset,
                                 StoragePrecisionPolicy policy,
                                 bool direct_array_element) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_scalar_v<raw_t>) {
        return decode_scalar<raw_t>(src, offset, policy, direct_array_element);
    } else if constexpr (is_vector_v<raw_t>) {
        using element_t = type_element_t<raw_t>;
        constexpr size_t dim = vector_dimension_v<raw_t>;
        size_t element_size = resolved_size<element_t>(policy);
        raw_t value{};
        for (size_t index = 0; index < dim; ++index) {
            value[index] = decode_scalar<element_t>(src, offset + index * element_size, policy, false);
        }
        return value;
    } else if constexpr (is_matrix_v<raw_t>) {
        constexpr size_t dim = matrix_dimension_v<raw_t>;
        using column_t = tuple_element_t<0, struct_member_tuple_t<raw_t>>;
        size_t column_size = resolved_size<column_t>(policy);
        raw_t value{};
        for (size_t index = 0; index < dim; ++index) {
            value[index] = decode_aos_value<column_t>(src, offset + index * column_size, policy, false);
        }
        return value;
    } else if constexpr (is_array_v<raw_t>) {
        using element_t = array_element_t<raw_t>;
        constexpr size_t dim = array_dimension_v<raw_t>;
        size_t element_size = resolved_size_in_context<element_t>(policy, true);
        raw_t value{};
        for (size_t index = 0; index < dim; ++index) {
            value[index] = decode_aos_value<element_t>(src, offset + index * element_size, policy, true);
        }
        return value;
    } else if constexpr (is_struct_v<raw_t>) {
        raw_t value{};
        size_t current = offset;
        [&]<size_t... I>(std::index_sequence<I...>) {
            (([&] {
                using member_t = tuple_element_t<I, struct_member_tuple_t<raw_t>>;
                current = align_up_size(current, resolved_alignment<member_t>(policy));
                member_ref<I>(value) = decode_aos_value<member_t>(src, current, policy, false);
                current += resolved_size<member_t>(policy);
            }()), ...);
        }(std::make_index_sequence<tuple_size_v<struct_member_tuple_t<raw_t>>>{});
        return value;
    } else {
        static_assert(always_false_v<raw_t>, "Unsupported type for DynamicBufferLayoutCodec::decode_aos_value");
    }
}

template<typename T, typename Getter>
[[nodiscard]] size_t encode_soa_from_getter(size_t count,
                                            HostByteBuffer &dst,
                                            size_t offset,
                                            StoragePrecisionPolicy policy,
                                            Getter &&getter,
                                            bool direct_array_element = false) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_soa_atomic_v<raw_t>) {
        size_t stride = resolved_size_in_context<raw_t>(policy, direct_array_element);
        for (size_t index = 0; index < count; ++index) {
            encode_aos_value<raw_t>(dst, offset + index * stride, getter(index), policy, direct_array_element);
        }
        return count * stride;
    } else if constexpr (is_matrix_v<raw_t>) {
        size_t current = offset;
        [&]<size_t... I>(std::index_sequence<I...>) {
            ((current += encode_soa_from_getter<tuple_element_t<I, struct_member_tuple_t<raw_t>>>(
                  count, dst, current, policy,
                  [&getter](size_t record_index) -> const auto & {
                      return member_ref<I>(getter(record_index));
                  })), ...);
        }(std::make_index_sequence<tuple_size_v<struct_member_tuple_t<raw_t>>>{});
        return current - offset;
    } else if constexpr (is_array_v<raw_t>) {
        size_t current = offset;
        [&]<size_t... I>(std::index_sequence<I...>) {
            ((current += encode_soa_from_getter<array_element_t<raw_t>>(
                  count, dst, current, policy,
                  [&getter](size_t record_index) -> const auto & {
                      return member_ref<I>(getter(record_index));
                  }, true)), ...);
        }(std::make_index_sequence<array_dimension_v<raw_t>>{});
        return current - offset;
    } else if constexpr (is_struct_v<raw_t>) {
        size_t current = offset;
        [&]<size_t... I>(std::index_sequence<I...>) {
            ((current += encode_soa_from_getter<tuple_element_t<I, struct_member_tuple_t<raw_t>>>(
                  count, dst, current, policy,
                  [&getter](size_t record_index) -> const auto & {
                      return member_ref<I>(getter(record_index));
                  })), ...);
        }(std::make_index_sequence<tuple_size_v<struct_member_tuple_t<raw_t>>>{});
        return current - offset;
    } else {
        static_assert(always_false_v<raw_t>, "Unsupported type for DynamicBufferLayoutCodec::encode_soa_from_getter");
    }
}

template<typename T, typename Getter>
[[nodiscard]] size_t decode_soa_to_getter(size_t count,
                                          const HostByteBuffer &src,
                                          size_t offset,
                                          StoragePrecisionPolicy policy,
                                          Getter &&getter,
                                          bool direct_array_element = false) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_soa_atomic_v<raw_t>) {
        size_t stride = resolved_size_in_context<raw_t>(policy, direct_array_element);
        for (size_t index = 0; index < count; ++index) {
            getter(index) = decode_aos_value<raw_t>(src, offset + index * stride, policy, direct_array_element);
        }
        return count * stride;
    } else if constexpr (is_matrix_v<raw_t>) {
        size_t current = offset;
        [&]<size_t... I>(std::index_sequence<I...>) {
            ((current += decode_soa_to_getter<tuple_element_t<I, struct_member_tuple_t<raw_t>>>(
                  count, src, current, policy,
                  [&getter](size_t record_index) -> auto & {
                      return member_ref<I>(getter(record_index));
                  })), ...);
        }(std::make_index_sequence<tuple_size_v<struct_member_tuple_t<raw_t>>>{});
        return current - offset;
    } else if constexpr (is_array_v<raw_t>) {
        size_t current = offset;
        [&]<size_t... I>(std::index_sequence<I...>) {
            ((current += decode_soa_to_getter<array_element_t<raw_t>>(
                  count, src, current, policy,
                  [&getter](size_t record_index) -> auto & {
                      return member_ref<I>(getter(record_index));
                  }, true)), ...);
        }(std::make_index_sequence<array_dimension_v<raw_t>>{});
        return current - offset;
    } else if constexpr (is_struct_v<raw_t>) {
        size_t current = offset;
        [&]<size_t... I>(std::index_sequence<I...>) {
            ((current += decode_soa_to_getter<tuple_element_t<I, struct_member_tuple_t<raw_t>>>(
                  count, src, current, policy,
                  [&getter](size_t record_index) -> auto & {
                      return member_ref<I>(getter(record_index));
                  })), ...);
        }(std::make_index_sequence<tuple_size_v<struct_member_tuple_t<raw_t>>>{});
        return current - offset;
    } else {
        static_assert(always_false_v<raw_t>, "Unsupported type for DynamicBufferLayoutCodec::decode_soa_to_getter");
    }
}

template<typename T>
[[nodiscard]] size_t soa_storage_bytes(size_t count,
                                       StoragePrecisionPolicy policy,
                                       bool direct_array_element = false) noexcept {
    using raw_t = std::remove_cvref_t<T>;
    if constexpr (is_soa_atomic_v<raw_t>) {
        return count * resolved_size_in_context<raw_t>(policy, direct_array_element);
    } else if constexpr (is_matrix_v<raw_t>) {
        size_t total = 0;
        [&]<size_t... I>(std::index_sequence<I...>) {
            ((total += soa_storage_bytes<tuple_element_t<I, struct_member_tuple_t<raw_t>>>(count, policy)), ...);
        }(std::make_index_sequence<tuple_size_v<struct_member_tuple_t<raw_t>>>{});
        return total;
    } else if constexpr (is_array_v<raw_t>) {
        return array_dimension_v<raw_t> * soa_storage_bytes<array_element_t<raw_t>>(count, policy, true);
    } else if constexpr (is_struct_v<raw_t>) {
        size_t total = 0;
        [&]<size_t... I>(std::index_sequence<I...>) {
            ((total += soa_storage_bytes<tuple_element_t<I, struct_member_tuple_t<raw_t>>>(count, policy)), ...);
        }(std::make_index_sequence<tuple_size_v<struct_member_tuple_t<raw_t>>>{});
        return total;
    } else {
        static_assert(always_false_v<raw_t>, "Unsupported type for DynamicBufferLayoutCodec::soa_storage_bytes");
    }
}

}// namespace detail

template<typename T>
struct DynamicBufferLayoutCodec {
    [[nodiscard]] static size_t storage_bytes(size_t count,
                                              StoragePrecisionPolicy policy,
                                              DynamicBufferLayout layout) noexcept {
        switch (layout) {
            case DynamicBufferLayout::aos:
                return count * detail::resolved_size<T>(policy);
            case DynamicBufferLayout::soa:
                return detail::soa_storage_bytes<T>(count, policy);
            default:
                return 0u;
        }
    }

    static void encode(const T *src,
                       size_t count,
                       HostByteBuffer &dst,
                       StoragePrecisionPolicy policy,
                       DynamicBufferLayout layout) noexcept {
        OC_ASSERT(src != nullptr || count == 0u);
        dst.resize(storage_bytes(count, policy, layout));
        if (count == 0u) {
            return;
        }
        switch (layout) {
            case DynamicBufferLayout::aos: {
                size_t stride = detail::resolved_size<T>(policy);
                for (size_t index = 0; index < count; ++index) {
                    detail::encode_aos_value<T>(dst, index * stride, src[index], policy, false);
                }
                break;
            }
            case DynamicBufferLayout::soa:
                (void)detail::encode_soa_from_getter<T>(count, dst, 0u, policy,
                                                        [src](size_t index) -> const T & { return src[index]; });
                break;
            default:
                break;
        }
    }

    static void decode(const HostByteBuffer &src,
                       size_t count,
                       T *dst,
                       StoragePrecisionPolicy policy,
                       DynamicBufferLayout layout) noexcept {
        OC_ASSERT(dst != nullptr || count == 0u);
        OC_ASSERT(src.size() >= storage_bytes(count, policy, layout));
        if (count == 0u) {
            return;
        }
        switch (layout) {
            case DynamicBufferLayout::aos: {
                size_t stride = detail::resolved_size<T>(policy);
                for (size_t index = 0; index < count; ++index) {
                    dst[index] = detail::decode_aos_value<T>(src, index * stride, policy, false);
                }
                break;
            }
            case DynamicBufferLayout::soa:
                (void)detail::decode_soa_to_getter<T>(count, src, 0u, policy,
                                                      [dst](size_t index) -> T & { return dst[index]; });
                break;
            default:
                break;
        }
    }
};

}// namespace ocarina