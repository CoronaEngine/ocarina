//
// Created by Zero on 2024/2/25.
//

#pragma once

#include <utility>

#include "math/basic_types.h"
#include "builtin.h"
#include "var.h"
#include "type_trait.h"
#include "func.h"

namespace ocarina {

namespace detail {

template<typename T>
struct is_valid_buffer_storage : public std::false_type {};

template<>
struct is_valid_buffer_storage<BindlessArrayByteBuffer> : public std::true_type {};

template<>
struct is_valid_buffer_storage<ByteBufferVar> : public std::true_type {};

template<>
struct is_valid_buffer_storage<Ref<ByteBuffer>> : public std::true_type {};

}// namespace detail

template<typename T>
static constexpr bool is_valid_buffer_storage_v = detail::is_valid_buffer_storage<std::remove_cvref_t<T>>::value;

template<typename T>
requires is_valid_buffer_storage_v<T>
struct BufferStorage {
public:
    using Storage = std::aligned_storage_t<sizeof(T), alignof(T)>;

private:
    Storage storage_{};

public:
    BufferStorage() = default;
    explicit BufferStorage(const T &buffer) {
        oc_memcpy(addressof(storage_), addressof(buffer), sizeof(T));
    }

    BufferStorage(const BufferStorage<T> &other) {
        oc_memcpy(addressof(storage_), addressof(other.storage_), sizeof(T));
    }

    BufferStorage &operator=(const BufferStorage<T> &other) {
        oc_memcpy(addressof(storage_), addressof(other.storage_), sizeof(T));
        return *this;
    }

    [[nodiscard]] T *get() noexcept {
        return reinterpret_cast<T *>(addressof(storage_));
    }

    [[nodiscard]] const T *get() const noexcept {
        return reinterpret_cast<const T *>(addressof(storage_));
    }

    [[nodiscard]] T *operator->() noexcept { return get(); }
    [[nodiscard]] const T *operator->() const noexcept { return get(); }
};

enum AccessMode {
    AOS,
    SOA
};

template<typename T, typename TBuffer>
struct SOAViewVar {
    static_assert(always_false_v<T, TBuffer>, "The SOAViewVar template must be specialized");
};

template<typename T, typename TBufferView>
struct SOAView {
    static_assert(always_false_v<T, TBufferView>, "The SOAView template must be specialized");
};

}// namespace ocarina

#define OC_MAKE_ATOMIC_SOA_VAR(TemplateArgs, TypeName)                                        \
    TemplateArgs struct ocarina::SOAViewVar<TypeName, TBuffer> {                              \
    public:                                                                                   \
        static_assert(is_valid_buffer_element_v<TypeName>);                                   \
        using atomic_type = TypeName;                                                         \
        static constexpr ocarina::uint type_size = sizeof(atomic_type);                       \
        static constexpr AccessMode access_mode = SOA;                                        \
                                                                                              \
    private:                                                                                  \
        ocarina::BufferStorage<TBuffer> buffer_var_{};                                        \
        ocarina::Uint view_size_{};                                                           \
        ocarina::Uint offset_{};                                                              \
        ocarina::uint stride_{};                                                              \
                                                                                              \
    public:                                                                                   \
        SOAViewVar() = default;                                                               \
        explicit SOAViewVar(const TBuffer &buffer,                                            \
                            const ocarina::Uint &view_size = ocarina::InvalidUI32,            \
                            const ocarina::Uint &ofs = 0u, ocarina::uint stride = type_size)  \
            : buffer_var_(buffer),                                                            \
              view_size_(ocarina::min(view_size,                                              \
                                      buffer.template size_in_byte<ocarina::uint>())),        \
              offset_(ofs), stride_(stride) {}                                                \
                                                                                              \
        template<typename Index>                                                              \
        requires ocarina::is_integral_expr_v<Index>                                           \
        [[nodiscard]] ocarina::Var<atomic_type> read(Index &&index) const noexcept {          \
            return buffer_var_->template load_as<atomic_type>(offset_ +                       \
                                                              OC_FORWARD(index) * type_size); \
        }                                                                                     \
                                                                                              \
        template<typename Index>                                                              \
        requires ocarina::is_integral_expr_v<Index>                                           \
        [[nodiscard]] ocarina::Var<atomic_type> at(Index &&index) const noexcept {            \
            return buffer_var_->template load_as<atomic_type>(offset_ +                       \
                                                              OC_FORWARD(index) * type_size); \
        }                                                                                     \
                                                                                              \
        template<typename Index>                                                              \
        requires ocarina::is_integral_expr_v<Index>                                           \
        [[nodiscard]] ocarina::Var<atomic_type> &at(Index &&index) noexcept {                 \
            return buffer_var_->template load_as<atomic_type>(offset_ +                       \
                                                              OC_FORWARD(index) * type_size); \
        }                                                                                     \
                                                                                              \
        template<typename Index>                                                              \
        requires ocarina::is_integral_expr_v<Index>                                           \
        void write(Index &&index, const ocarina::Var<atomic_type> &val) noexcept {            \
            buffer_var_->store(offset_ + OC_FORWARD(index) * type_size, val);                 \
        }                                                                                     \
                                                                                              \
        template<typename int_type = ocarina::uint>                                           \
        [[nodiscard]] ocarina::Var<int_type> size_in_byte() const noexcept {                  \
            return view_size_ / stride_ * type_size;                                          \
        }                                                                                     \
    };

#define OC_MAKE_ATOMIC_SOA_VIEW(TemplateArgs, TypeName)                                                 \
    TemplateArgs                                                                                        \
    requires ocarina::is_byte_buffer_view_v<TBuffer>                                                    \
    struct ocarina::SOAView<TypeName, TBuffer> {                                                        \
    public:                                                                                             \
        static_assert(is_valid_buffer_element_v<TypeName>);                                             \
        using atomic_type = TypeName;                                                                   \
        static constexpr ocarina::uint type_size = sizeof(atomic_type);                                 \
        static constexpr AccessMode access_mode = SOA;                                                  \
        static constexpr bool is_atomic = true;                                                         \
                                                                                                        \
    private:                                                                                            \
        TBuffer buffer_view_{};                                                                         \
        ocarina::uint view_size_{};                                                                     \
        ocarina::uint offset_{};                                                                        \
        ocarina::uint stride_{};                                                                        \
                                                                                                        \
    public:                                                                                             \
        SOAView() = default;                                                                            \
        SOAView(TBuffer bv, uint view_size = ocarina::InvalidUI32,                                      \
                uint ofs = 0u, uint stride = type_size)                                                 \
            : buffer_view_(bv), view_size_(ocarina::min(view_size, uint(buffer_view_.size_in_byte()))), \
              offset_(ofs),                                                                             \
              stride_(stride) {}                                                                        \
                                                                                                        \
        OC_MAKE_MEMBER_GETTER(offset, )                                                                 \
                                                                                                        \
        [[nodiscard]] TBuffer view() const noexcept {                                                   \
            return TBuffer{buffer_view_.handle(),                                                       \
                           offset_,                                                                     \
                           view_size_,                                                                  \
                           buffer_view_.total_size()};                                                  \
        }                                                                                               \
                                                                                                        \
        [[nodiscard]] ocarina::uint size_in_byte() const noexcept {                                     \
            return view_size_ / stride_ * type_size;                                                    \
        }                                                                                               \
    };

#define OC_MAKE_ATOMIC_SOA_VAR_VIEW(TemplateArgs, TypeName) \
    OC_MAKE_ATOMIC_SOA_VAR(TemplateArgs, TypeName)          \
    OC_MAKE_ATOMIC_SOA_VIEW(TemplateArgs, TypeName)

OC_MAKE_ATOMIC_SOA_VAR_VIEW(template<typename TBuffer>, ocarina::uint)
OC_MAKE_ATOMIC_SOA_VAR_VIEW(template<typename TBuffer>, ocarina::ulong)
OC_MAKE_ATOMIC_SOA_VAR_VIEW(template<typename TBuffer>, float)
OC_MAKE_ATOMIC_SOA_VAR_VIEW(template<typename TBuffer>, int)
OC_MAKE_ATOMIC_SOA_VAR(template<typename T OC_COMMA ocarina::uint N OC_COMMA typename TBuffer>,
                       ocarina::Vector<T OC_COMMA N>)
OC_MAKE_ATOMIC_SOA_VIEW(template<typename T OC_COMMA ocarina::uint N OC_COMMA typename TBuffer>,
                        ocarina::Vector<T OC_COMMA N>)

///#region make structure soa

#define OC_MAKE_SOA_VAR_MEMBER(field_name) ocarina::SOAViewVar<decltype(struct_type::field_name), TBuffer> field_name;

#define OC_MAKE_SOA_VAR_MEMBER_CONSTRUCT(field_name)                                                                     \
    field_name = ocarina::SOAViewVar<decltype(struct_type::field_name), TBuffer>(buffer_var, view_size, offset, stride); \
    offset += field_name.size_in_byte();

#define OC_MAKE_SOA_VAR_MEMBER_READ(field_name) ret.field_name = field_name.read(OC_FORWARD(index));

#define OC_MAKE_SOA_VAR_MEMBER_WRITE(field_name) field_name.write(OC_FORWARD(index), val.field_name);

#define OC_MAKE_SOA_VAR_MEMBER_SIZE(field_name) ret += field_name.size_in_byte();

#define OC_MAKE_STRUCT_SOA_VAR(TemplateArgs, S, ...)                                    \
    TemplateArgs struct ocarina::SOAViewVar<S, TBuffer> {                               \
    public:                                                                             \
        using struct_type = S;                                                          \
        static_assert(ocarina::is_valid_buffer_element_v<struct_type>);                 \
        static constexpr ocarina::uint type_size = sizeof(struct_type);                 \
        static constexpr AccessMode access_mode = SOA;                                  \
                                                                                        \
    public:                                                                             \
        MAP(OC_MAKE_SOA_VAR_MEMBER, ##__VA_ARGS__)                                      \
    public:                                                                             \
        SOAViewVar() = default;                                                         \
        explicit SOAViewVar(const TBuffer &buffer_var,                                  \
                            ocarina::Uint view_size = InvalidUI32,                      \
                            ocarina::Uint offset = 0u,                                  \
                            ocarina::uint stride = type_size) {                         \
            view_size = ocarina::min(buffer_var.template size_in_byte<ocarina::uint>(), \
                                     view_size);                                        \
            MAP(OC_MAKE_SOA_VAR_MEMBER_CONSTRUCT, ##__VA_ARGS__)                        \
        }                                                                               \
                                                                                        \
        template<typename Index>                                                        \
        requires ocarina::is_integral_expr_v<Index>                                     \
        [[nodiscard]] ocarina::Var<struct_type> read(Index &&index) const noexcept {    \
            return ocarina::outline(                                                    \
                [&] {                                                                   \
                    ocarina::Var<struct_type> ret;                                      \
                    MAP(OC_MAKE_SOA_VAR_MEMBER_READ, ##__VA_ARGS__)                     \
                    return ret;                                                         \
                },                                                                      \
                "SOAViewVar<" #S ">::read");                                            \
        }                                                                               \
                                                                                        \
        template<typename Index>                                                        \
        requires ocarina::is_integral_expr_v<Index>                                     \
        void write(Index &&index, const ocarina::Var<struct_type> &val) noexcept {      \
            ocarina::outline(                                                           \
                [&] {                                                                   \
                    MAP(OC_MAKE_SOA_VAR_MEMBER_WRITE, ##__VA_ARGS__)                    \
                },                                                                      \
                "SOAViewVar<" #S ">::write");                                           \
        }                                                                               \
        template<typename int_type = ocarina::uint>                                     \
        [[nodiscard]] ocarina::Var<int_type> size_in_byte() const noexcept {            \
            return ocarina::outline(                                                    \
                [&] {                                                                   \
                    ocarina::Var<int_type> ret = 0;                                     \
                    MAP(OC_MAKE_SOA_VAR_MEMBER_SIZE, ##__VA_ARGS__)                     \
                    return ret;                                                         \
                },                                                                      \
                "SOAViewVar<" #S ">::size_in_byte");                                    \
        }                                                                               \
    };

///#endregion

#define OC_MAKE_STRUCT_SOA_VIEW_MEMBER(member_name) \
    ocarina::SOAView<decltype(struct_type::member_name), TBuffer> member_name;

#define OC_MAKE_STRUCT_SOA_VIEW_CONSTRUCT(member_name)                                              \
    member_name = SOAView<decltype(struct_type::member_name), TBuffer>(bv, view_size, ofs, stride); \
    ofs += member_name.size_in_byte();

#define OC_MAKE_STRUCT_SOA_VIEW_SIZE(member_name) +member_name.size_in_byte()

#define OC_MAKE_STRUCT_SOA_VIEW(TemplateArgs, S, first_member, ...)                    \
    TemplateArgs                                                                       \
    requires ocarina::is_byte_buffer_view_v<TBuffer>                                   \
    struct ocarina::SOAView<S, TBuffer> {                                              \
    public:                                                                            \
        using struct_type = S;                                                         \
        static constexpr ocarina::uint type_size = sizeof(struct_type);                \
        static constexpr AccessMode access_mode = SOA;                                 \
        static constexpr bool is_atomic = false;                                       \
                                                                                       \
    public:                                                                            \
        MAP(OC_MAKE_STRUCT_SOA_VIEW_MEMBER, first_member, ##__VA_ARGS__)               \
                                                                                       \
    public:                                                                            \
        SOAView() = default;                                                           \
        explicit SOAView(TBuffer bv, ocarina::uint view_size = ocarina::InvalidUI32,   \
                         ocarina::uint ofs = 0u, ocarina::uint stride = type_size) {   \
            view_size = ocarina::min(uint(bv.size_in_byte()), view_size);              \
            MAP(OC_MAKE_STRUCT_SOA_VIEW_CONSTRUCT, first_member, ##__VA_ARGS__)        \
        }                                                                              \
                                                                                       \
        [[nodiscard]] TBuffer view() const noexcept {                                  \
            return TBuffer{first_member.view().handle(), first_member.view().offset(), \
                           size_in_byte(), first_member.view().total_size()};          \
        }                                                                              \
                                                                                       \
        [[nodiscard]] ocarina::uint size_in_byte() const noexcept {                    \
            return 0u MAP(OC_MAKE_STRUCT_SOA_VIEW_SIZE, first_member, ##__VA_ARGS__);  \
        }                                                                              \
    };

#define OC_MAKE_ARRAY_SOA_VAR(TemplateArgs, TypeName, ElementType)                      \
    TemplateArgs struct ocarina::SOAViewVar<TypeName, TBuffer> {                        \
    public:                                                                             \
        using struct_type = TypeName;                                                   \
        static_assert(is_valid_buffer_element_v<struct_type>);                          \
        static constexpr AccessMode access_mode = SOA;                                  \
        using element_type = ElementType;                                               \
        static constexpr ocarina::uint type_size = sizeof(struct_type);                 \
                                                                                        \
    private:                                                                            \
        ocarina::array<ocarina::SOAViewVar<element_type, TBuffer>, N> array_{};         \
                                                                                        \
    public:                                                                             \
        SOAViewVar() = default;                                                         \
        explicit SOAViewVar(const TBuffer &buffer_var,                                  \
                            ocarina::Uint view_size = InvalidUI32,                      \
                            ocarina::Uint offset = 0u,                                  \
                            ocarina::uint stride = type_size) {                         \
            view_size = ocarina::min(buffer_var.template size_in_byte<ocarina::uint>(), \
                                     view_size);                                        \
            for (int i = 0; i < N; ++i) {                                               \
                array_[i] = SOAViewVar<element_type, TBuffer>(buffer_var, view_size,    \
                                                              offset, stride);          \
                offset += array_[i].template size_in_byte<ocarina::uint>();             \
            }                                                                           \
        }                                                                               \
                                                                                        \
        [[nodiscard]] auto operator[](size_t index) const noexcept {                    \
            return array_[index];                                                       \
        }                                                                               \
        [[nodiscard]] auto &operator[](size_t index) noexcept {                         \
            return array_[index];                                                       \
        }                                                                               \
                                                                                        \
        template<typename Index>                                                        \
        requires ocarina::is_integral_expr_v<Index>                                     \
        [[nodiscard]] ocarina::Var<struct_type> read(Index &&index) const noexcept {    \
            return ocarina::outline(                                                    \
                [&] {                                                                   \
                    ocarina::Var<struct_type> ret;                                      \
                    for (int i = 0; i < N; ++i) {                                       \
                        ret[i] = array_[i].read(OC_FORWARD(index));                     \
                    }                                                                   \
                    return ret;                                                         \
                },                                                                      \
                "SOAViewVar<" #TypeName ">::read");                                     \
        }                                                                               \
                                                                                        \
        template<typename Index>                                                        \
        requires ocarina::is_integral_expr_v<Index>                                     \
        void write(Index &&index, const ocarina::Var<struct_type> &val) noexcept {      \
            ocarina::outline(                                                           \
                [&] {                                                                   \
                    for (int i = 0; i < N; ++i) {                                       \
                        array_[i].write(OC_FORWARD(index), val[i]);                     \
                    }                                                                   \
                },                                                                      \
                "SOAViewVar<" #TypeName ">::write");                                    \
        }                                                                               \
                                                                                        \
        template<typename int_type = ocarina::uint>                                     \
        [[nodiscard]] ocarina::Var<int_type> size_in_byte() const noexcept {            \
            return ouline(                                                              \
                [&] {                                                                   \
                    ocarina::Var<int_type> ret = 0;                                     \
                    for (int i = 0; i < N; ++i) {                                       \
                        ret += array_[i].size_in_byte();                                \
                    }                                                                   \
                    return ret;                                                         \
                },                                                                      \
                "SOAViewVar<" #TypeName ">::size_in_byte");                             \
        }                                                                               \
    };

OC_MAKE_ARRAY_SOA_VAR(template<typename T OC_COMMA ocarina::uint N OC_COMMA ocarina::uint M OC_COMMA typename TBuffer>,
                      ocarina::Matrix<T OC_COMMA N OC_COMMA M>, Vector<float OC_COMMA M>)
OC_MAKE_ARRAY_SOA_VAR(template<ocarina::uint N OC_COMMA typename T OC_COMMA typename TBuffer>,
                      ocarina::array<T OC_COMMA N>, T)

namespace ocarina {
template<typename Elm, typename TBuffer>
[[nodiscard]] SOAViewVar<Elm, TBuffer> make_soa_view_var(TBuffer &buffer) noexcept {
    return SOAViewVar<Elm, TBuffer>(buffer);
}

template<typename Elm, typename TBuffer>
requires ocarina::is_byte_buffer_view_v<TBuffer>
[[nodiscard]] SOAView<Elm, TBuffer> make_soa_view(TBuffer &buffer) noexcept {
    return SOAView<Elm, TBuffer>(buffer);
}
}// namespace ocarina

#define OC_MAKE_ARRAY_SOA_VIEW(TemplateArgs, TypeName, ElementType)          \
    TemplateArgs struct ocarina::SOAView<TypeName, TBuffer> {                \
    public:                                                                  \
        using struct_type = TypeName;                                        \
        static_assert(is_valid_buffer_element_v<struct_type>);               \
        static constexpr AccessMode access_mode = SOA;                       \
        using element_type = ElementType;                                    \
        static constexpr ocarina::uint type_size = sizeof(struct_type);      \
        static constexpr bool is_atomic = false;                             \
                                                                             \
    private:                                                                 \
        ocarina::array<ocarina::SOAView<element_type, TBuffer>, N> array_{}; \
                                                                             \
    public:                                                                  \
        SOAView() = default;                                                 \
        explicit SOAView(TBuffer bv, uint view_size = ocarina::InvalidUI32,  \
                         uint offset = 0u, uint stride = type_size) {        \
            view_size = ocarina::min(bv.size_in_byte(), view_size);          \
            for (uint i = 0; i < N; ++i) {                                   \
                array_[i] = ocarina::SOAView{bv, view_size, offset, stride}; \
                offset += array_[i].size_in_byte();                          \
            }                                                                \
        }                                                                    \
        [[nodiscard]] ocarina::uint size_in_byte() const noexcept {          \
            uint ret = 0u;                                                   \
            for (int i = 0; i < N; ++i) {                                    \
                ret += array_[i].size_in_byte();                             \
            }                                                                \
            return ret;                                                      \
        }                                                                    \
        [[nodiscard]] TBuffer view() const noexcept {                        \
            auto first_view = array_[0].view();                              \
            return TBuffer{first_view.handle(), first_view.offset(),         \
                           size_in_byte(), first_view.total_size()};         \
        }                                                                    \
        [[nodiscard]] auto operator[](size_t index) const noexcept {         \
            return array_[index];                                            \
        }                                                                    \
        [[nodiscard]] auto &operator[](size_t index) noexcept {              \
            return array_[index];                                            \
        }                                                                    \
    };

OC_MAKE_ARRAY_SOA_VIEW(template<typename T OC_COMMA ocarina::uint N OC_COMMA ocarina::uint M OC_COMMA typename TBuffer>,
                       ocarina::Matrix<T OC_COMMA N OC_COMMA M>, Vector<float OC_COMMA M>)
OC_MAKE_ARRAY_SOA_VIEW(template<ocarina::uint N OC_COMMA typename T OC_COMMA typename TBuffer>,
                       ocarina::array<T OC_COMMA N>, T)

namespace ocarina {

template<typename T, typename TBuffer>
struct AOSViewVar {
public:
    using buffer_type = TBuffer;
    using element_type = T;
    static constexpr AccessMode access_mode = AOS;
    static constexpr auto stride = sizeof(T);

private:
    ocarina::BufferStorage<TBuffer> buffer_var_;
    ocarina::Uint offset_;

public:
    AOSViewVar() = default;
    explicit AOSViewVar(const TBuffer &buffer, const Uint &view_size = InvalidUI32,
                        Uint ofs = 0u)
        : buffer_var_(buffer), offset_(std::move(ofs)) {}

    template<typename Index, typename Size = uint>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<T> at(const Index &index) const noexcept {
        Var<Size> offset = index * sizeof(T);
        return buffer_var_->template load_as<T>(offset);
    }

    template<typename Index, typename Size = uint>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<T> &at(const Index &index) noexcept {
        Var<Size> offset = index * sizeof(T);
        return buffer_var_->template load_as<T>(offset);
    }

    template<typename Index, typename Arg, typename Size = uint>
    requires std::is_same_v<T, remove_device_t<Arg>> && is_integral_expr_v<Index>
    void write(const Index &index, const Arg &arg) noexcept {
        buffer_var_->store(index * stride, arg);
    }

    template<typename Index, typename Size = uint>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<T> read(const Index &index) const noexcept {
        Var<Size> offset = index * sizeof(T);
        return buffer_var_->template load_as<T>(offset);
    }
};

template<typename Elm, typename TBuffer>
[[nodiscard]] AOSViewVar<Elm, TBuffer> make_aos_view_var(const TBuffer &buffer) noexcept {
    return AOSViewVar<Elm, TBuffer>(buffer);
}

}// namespace ocarina
