# Native kernels (optional, Metal)

AgrimlxO is pure Python by default. The optional package `agrimlxo-kernels`
(`native/`) ships the fused MoE routed **weighted-sum** Metal kernel vendored
from omlx (`omlx/custom_kernels/quant_matmul/csrc/`, Apache-2.0) and
**rewritten for the MoE expert expansion**: the upstream kernel only admitted
routed budgets of 6/8/10 experts per token (it was built for the native
routing), while the expansion needs any budget N up to 64.

## What changed vs the omlx kernel

* `moe_weighted_sum.cpp` — the `unsupported()` gate now accepts any routed
  budget in `[2, 64]` (dtype/contiguity checks unchanged).
* `moe_weighted_sum.metal` — new tiled instantiations for the expansion
  budgets **12 / 16 / 20 / 24 / 32** (on top of the upstream 6/8/10) plus a
  **runtime-topk dynamic kernel** (`moe_weighted_sum_dynamic_*`, element
  indexed, no threadgroup arrays) that covers any other budget. The C++
  dispatch selects a tiled kernel when one exists, the dynamic kernel
  otherwise.
* Namespace and library name moved to `agrimlxo::kernels` /
  `agrimlxo_kernels_lib`; the bindings are trimmed to `abi_probe` +
  `moe_weighted_sum`.
* `affine_quantized.h` (the shared Metal kernel templates) is vendored
  unmodified; `include/agrimlxo/binary_dir.h` is the omlx helper renamed.

## What it does

Stock ``SwitchGLU`` sorts routed tokens, computes expert outputs, scatters
back to ``[B, T, N, D]`` and applies the router scores. On large prefill
batches the native path consumes the sorted expert outputs directly:

```
y[t] = Σ_k x_sorted[inv_order[t·N + k]] · scores[t, k]
```

in one kernel, in fp32 accumulation, with no `[B, T, N, D]` intermediate.
The scores are the **post-pass expansion weights** (already cut by the
threshold, decayed, renormalized), so the kernel serves the expansion path
unchanged — that is exactly why the arbitrary-budget support matters.

## Build (macOS, Apple Silicon)

```bash
AGRIMLXO_WITH_KERNELS=1 pip install ./native
```

Requires the Metal toolchain (`xcrun -f metal`) and the pinned
`mlx==0.32.4` / `nanobind==3.1.0` build pair (same pins as omlx, so the
extension shares MLX's NB_DOMAIN). The build uses MLX's own extension
helper (`mlx.extension.MetalExtension`) to compile the Metal shaders into a
`.metallib` shipped next to the extension.

## Use

Off by default (the fused sum reorders the floating-point accumulation vs
the stock ops; the flags-off bit-identical guarantee stays airtight):

```bash
agrimlxo serve -m <model> --fast-moe-prefill     # or AGRIMLXO_FAST_MOE=1
```

With the kernels missing (or an ABI mismatch — probed once via `abi_probe`,
the omlx issue-#2139 canary), the server logs one line and keeps the stock
MLX ops; nothing fails. Gating: Metal available, fp16/bf16 tensors, prefill
batches ≥ 1024 tokens, budget in [2, 64] — the same shape gates as omlx.
Decode and small prefills always run the stock ops.

Python surface: `agrimlxo.kernels` (`native_available()`,
`should_use_fast_weighted_sum`, `fast_switch_weighted_sum`,
`sort_routes`, `fallback_weighted_sum`); integration point:
`models/moe_block.routed_forward` (stock path, expansion path, reduce and
mixed-quant switch_mlps all flow through it).

## Not vendored (yet)

omlx's `gather_qmm` block-sorting variants and `moe_staging` (bounded
expert-read transport) are tuned to omlx's batched engine plumbing; the
mixed-quant path here uses MLX's built-in `mx.gather_qmm`. Wiring the
native block-sorting kernels into `MixedSwitchGLU` is the natural next
step on the Mac (see docs/mixed-experts.md).
