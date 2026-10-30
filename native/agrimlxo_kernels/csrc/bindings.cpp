// SPDX-License-Identifier: Apache-2.0
// Python bindings for the AgrimlxO native MoE kernels.
//
// Trimmed from omlx (github.com/pmarreck/omlx, omlx/custom_kernels/
// quant_matmul/csrc/bindings.cpp, Apache-2.0) to the fused MoE weighted
// sum, generalized for the AgrimlxO MoE expert expansion (any routed
// budget in [2, 64]).
//
// The abi_probe canary follows omlx: an extension built with a nanobind
// whose ABI tag differs from the mlx wheel's imports cleanly but rejects
// every mx.array argument ("incompatible function arguments"); kernels.py
// probes once at import and disables the native symbols on failure.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "moe_weighted_sum.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native MoE kernels for AgrimlxO (fused routed weighted sum)";

  m.def(
      "abi_probe",
      [](const mlx::core::array& a) {
        return static_cast<int64_t>(a.size());
      },
      "a"_a);

  m.def(
      "moe_weighted_sum",
      &agrimlxo::kernels::moe_weighted_sum,
      "x_sorted"_a,
      "inv_order"_a,
      "scores"_a,
      "stream"_a = nb::none());
}
