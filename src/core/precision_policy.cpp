//
// Created by Z on 12/04/2026.
//

#include "precision_policy.h"

namespace ocarina {

/// Decision matrix:
///  SM      | fp16        | VRAM tight | Result
///  < 53    | none        | -          | force_f32
///  53-59   | native,slow | -          | force_f32
///  60-69   | fast (2x)   | no         | force_f32
///  60-69   | fast (2x)   | yes        | force_f16
///  >= 70   | tensor      | no         | auto_select
///  >= 70   | tensor      | yes        | force_f16
PrecisionPolicy DevicePrecisionCaps::recommend_policy(
    size_t estimated_scene_vram_bytes) const noexcept {
    if (!has_native_fp16) {
        return PrecisionPolicy::force_f32;
    }
    constexpr size_t vram_tight_threshold = size_t(4) << 30;
    bool vram_pressure = (total_vram_bytes < vram_tight_threshold) ||
        (estimated_scene_vram_bytes > 0 &&
         estimated_scene_vram_bytes > total_vram_bytes * 6 / 10);
    if (has_fast_fp16 && vram_pressure) {
        return PrecisionPolicy::force_f16;
    }
    if (has_tensor_fp16) {
        return PrecisionPolicy::auto_select;
    }
    return PrecisionPolicy::force_f32;
}

}// namespace ocarina