// SPDX-License-Identifier: Apache-2.0
// Vendored from omlx (github.com/pmarreck/omlx, omlx/custom_kernels/
// quant_matmul/csrc/moe_weighted_sum.h, Apache-2.0) — namespace renamed to
// agrimlxo::kernels.
#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace agrimlxo::kernels {

mx::array moe_weighted_sum(
    const mx::array& x_sorted,
    const mx::array& inv_order,
    const mx::array& scores,
    mx::StreamOrDevice s = {});

} // namespace agrimlxo::kernels
