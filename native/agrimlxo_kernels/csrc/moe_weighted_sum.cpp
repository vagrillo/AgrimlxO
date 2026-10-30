// SPDX-License-Identifier: Apache-2.0
// Vendored from omlx (github.com/pmarreck/omlx, omlx/custom_kernels/
// quant_matmul/csrc/moe_weighted_sum.cpp, Apache-2.0) and MODIFIED for
// AgrimlxO: the routed-expert count is no longer restricted to 6/8/10 so
// the fused weighted sum also serves the MoE expert expansion path
// (budget N up to 64 per token — tiled kernels for the common budgets,
// a runtime-topk dynamic kernel for everything else).
//
// Fused weighted sum of the sorted routed-expert outputs: instead of
// scattering expert outputs back to [tokens, topk, D] and reducing with
// router scores, this reads the sorted rows through inv_order and folds
// the scores in fp32 — one kernel, no [tokens, topk, D] intermediate.
//
//   x_sorted  [tokens*topk, 1, D] row-contiguous fp16/bf16 expert outputs,
//             sorted by expert id (SwitchGLU's sorted_indices=true order)
//   inv_order [tokens*topk] uint32: sorted slot -> slot in router order
//             (the inverse of the _gather_sort order)
//   scores    [tokens, topk] float32 router weights (already cut/decayed/
//             renormalized when the expansion is active)
//   returns   [tokens, D] fp16/bf16

#include "moe_weighted_sum.h"
#include "agrimlxo/binary_dir.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace agrimlxo::kernels {

namespace {

using namespace mlx::core;

// topk values with dedicated tiled instantiations in the metallib; anything
// else in [2, MAX_TOPK] takes the dynamic runtime-topk kernel.
constexpr int MAX_TOPK = 64;
bool has_tiled_kernel(int topk) {
  switch (topk) {
    case 6:
    case 8:
    case 10:
    case 12:
    case 16:
    case 20:
    case 24:
    case 32:
      return true;
    default:
      return false;
  }
}

bool row_contiguous(const array& arr) {
  return arr.flags().row_contiguous && arr.strides(-1) == 1;
}

std::string metal_type_name(Dtype dtype) {
  if (dtype == float16) {
    return "float16_t";
  }
  if (dtype == bfloat16) {
    return "bfloat16_t";
  }
  std::ostringstream msg;
  msg << "Unsupported MoE weighted sum dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

class MoeWeightedSumPrimitive : public Primitive {
 public:
  explicit MoeWeightedSumPrimitive(Stream stream) : Primitive(stream) {}

  static bool unsupported(
      const array& x_sorted,
      const array& inv_order,
      const array& scores,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (x_sorted.dtype() != float16 && x_sorted.dtype() != bfloat16) {
      return true;
    }
    if (scores.dtype() != float32 || inv_order.dtype() != uint32) {
      return true;
    }
    if (x_sorted.ndim() != 3 || x_sorted.shape(-2) != 1 ||
        scores.ndim() < 2 || inv_order.ndim() != 1) {
      return true;
    }
    if (!row_contiguous(x_sorted) || !row_contiguous(inv_order) ||
        !row_contiguous(scores)) {
      return true;
    }
    const int topk = scores.shape(-1);
    // AgrimlxO: any expansion budget in [2, MAX_TOPK] is supported (the
    // upstream kernel only admitted 6/8/10).
    if (topk < 2 || topk > MAX_TOPK || x_sorted.shape(0) != scores.size() ||
        inv_order.size() != scores.size()) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("MoeWeightedSumPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x_sorted = inputs[0];
    const auto& inv_order = inputs[1];
    const auto& scores = inputs[2];

    out.set_data(allocator::malloc(out.nbytes()));

    const int topk = scores.shape(-1);
    const int tokens = scores.size() / topk;
    const int D = x_sorted.shape(-1);

    constexpr int tiled_threads = 256;
    const bool tiled = has_tiled_kernel(topk);

    std::string kname;
    if (tiled) {
      concatenate(
          kname,
          "moe_weighted_sum_tiled_",
          metal_type_name(x_sorted.dtype()),
          "_score_float_topk_",
          topk,
          "_t_",
          tiled_threads);
    } else {
      concatenate(
          kname,
          "moe_weighted_sum_dynamic_",
          metal_type_name(x_sorted.dtype()),
          "_score_float");
    }

    auto lib = d.get_library("agrimlxo_kernels_lib", binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x_sorted, 0);
    compute_encoder.set_input_array(inv_order, 1);
    compute_encoder.set_input_array(scores, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(tokens, 4);
    compute_encoder.set_bytes(D, 5);
    if (!tiled) {
      // dynamic kernel: runtime topk as an extra constant buffer
      compute_encoder.set_bytes(topk, 6);
    }

    const int threads = tiled ? tiled_threads : 256;
    const int total = tokens * D;
    MTL::Size group_dims(threads, 1, 1);
    MTL::Size grid_dims(
        tiled ? tokens : (total + threads - 1) / threads, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(MoeWeightedSumPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }
};

} // namespace

array moe_weighted_sum(
    const array& x_sorted,
    const array& inv_order,
    const array& scores,
    StreamOrDevice s) {
  if (x_sorted.ndim() != 3 || x_sorted.shape(-2) != 1) {
    std::ostringstream msg;
    msg << "[agrimlxo.moe_weighted_sum] expected "
        << "x_sorted shape [N, 1, D], got " << x_sorted.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (scores.ndim() < 2) {
    std::ostringstream msg;
    msg << "[agrimlxo.moe_weighted_sum] expected scores "
        << "rank >= 2, got " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (inv_order.ndim() != 1 || inv_order.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[agrimlxo.moe_weighted_sum] expected uint32 "
        << "inv_order rank 1, got " << inv_order.shape() << " dtype "
        << inv_order.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  const int topk = scores.shape(-1);
  const int64_t routed_rows = scores.size();
  const int D = x_sorted.shape(-1);
  if (x_sorted.shape(0) != routed_rows || inv_order.size() != routed_rows) {
    std::ostringstream msg;
    msg << "[agrimlxo.moe_weighted_sum] incompatible "
        << "shapes: " << x_sorted.shape() << ", " << inv_order.shape()
        << ", " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk <= 0 || D <= 0) {
    std::ostringstream msg;
    msg << "[agrimlxo.moe_weighted_sum] invalid topk or "
        << "hidden dim: topk=" << topk << ", D=" << D << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(x_sorted.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[agrimlxo.moe_weighted_sum] expected floating "
        << "x_sorted, got " << x_sorted.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x_sorted, inv_order, scores};
  Shape out_shape = scores.shape();
  out_shape.pop_back();
  out_shape.push_back(D);
  if (MoeWeightedSumPrimitive::unsupported(
          x_sorted, inv_order, scores, stream)) {
    throw std::invalid_argument(
        "[agrimlxo.moe_weighted_sum] unsupported shape.");
  }
  return array(
      std::move(out_shape),
      x_sorted.dtype(),
      std::make_shared<MoeWeightedSumPrimitive>(stream),
      std::move(inputs));
}

} // namespace agrimlxo::kernels
