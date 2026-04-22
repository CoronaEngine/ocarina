//
// Created by Z on 22/04/2026.
//

#pragma once

#include "common.h"

namespace ocarina {

struct GemmOp {
    GemmDesc desc;

    [[nodiscard]] bool valid() const noexcept {
        return desc.valid();
    }
};

[[nodiscard]] inline GemmOp gemm(GemmDesc desc) noexcept {
    OC_ASSERT(desc.valid());
    return GemmOp{std::move(desc)};
}

}// namespace ocarina