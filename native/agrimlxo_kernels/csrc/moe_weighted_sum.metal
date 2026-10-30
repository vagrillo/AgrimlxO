// SPDX-License-Identifier: Apache-2.0
// Vendored from omlx (github.com/pmarreck/omlx, omlx/custom_kernels/
// quant_matmul/csrc/moe_weighted_sum.metal, Apache-2.0) and MODIFIED for
// AgrimlxO: added tiled instantiations for the MoE expert expansion budgets
// (12/16/20/24/32, on top of the upstream 6/8/10) and a runtime-topk
// dynamic kernel that covers any other budget in [2, 64].
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "affine_quantized.h"

#define instantiate_moe_weighted_sum_tiled(type, score_type, topk, threads)    \
  instantiate_kernel(                                                          \
      "moe_weighted_sum_tiled_" #type "_score_" #score_type "_topk_" #topk     \
      "_t_" #threads,                                                          \
      moe_weighted_sum_tiled,                                                  \
      type,                                                                    \
      score_type,                                                              \
      topk,                                                                    \
      threads)

instantiate_moe_weighted_sum_tiled(float16_t, float, 8, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 8, 256);
instantiate_moe_weighted_sum_tiled(float16_t, float, 6, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 6, 256);
instantiate_moe_weighted_sum_tiled(float16_t, float, 10, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 10, 256);
// AgrimlxO: MoE expert expansion budgets (native top-K 8 raised to N).
instantiate_moe_weighted_sum_tiled(float16_t, float, 12, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 12, 256);
instantiate_moe_weighted_sum_tiled(float16_t, float, 16, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 16, 256);
instantiate_moe_weighted_sum_tiled(float16_t, float, 20, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 20, 256);
instantiate_moe_weighted_sum_tiled(float16_t, float, 24, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 24, 256);
instantiate_moe_weighted_sum_tiled(float16_t, float, 32, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 32, 256);

// AgrimlxO addition: runtime-topk dynamic kernel. Element-indexed (no
// threadgroup arrays, so topk is a plain loop bound), scores always float —
// covers any expansion budget the tiled kernels do not instantiate.
template <typename T>
[[kernel]] void moe_weighted_sum_dynamic(
    const device T* x_sorted [[buffer(0)]],
    const device uint32_t* inv_order [[buffer(1)]],
    const device float* scores [[buffer(2)]],
    device T* y [[buffer(3)]],
    const constant int& tokens [[buffer(4)]],
    const constant int& D [[buffer(5)]],
    const constant int& topk [[buffer(6)]],
    uint elem [[thread_position_in_grid]]) {
  const uint total = uint(tokens * D);
  if (elem >= total) {
    return;
  }
  const int token = elem / D;
  const int d = elem - token * D;
  const int route_base = token * topk;
  float acc = 0.0f;
  for (int k = 0; k < topk; ++k) {
    const uint32_t row = inv_order[route_base + k];
    acc += float(x_sorted[row * D + d]) * float(scores[route_base + k]);
  }
  y[elem] = T(acc);
}

#define instantiate_moe_weighted_sum_dynamic(type)                             \
  instantiate_kernel(                                                          \
      "moe_weighted_sum_dynamic_" #type "_score_float",                        \
      moe_weighted_sum_dynamic,                                                \
      type)

instantiate_moe_weighted_sum_dynamic(float16_t);
instantiate_moe_weighted_sum_dynamic(bfloat16_t);
