# SPDX-License-Identifier: Apache-2.0
"""Optional native Metal kernels for AgrimlxO.

Currently provides the fused routed-expert weighted sum
(``moe_weighted_sum``), generalized for the MoE expert expansion: any
routed budget in [2, 64] per token (tiled kernels for the common budgets
6/8/10/12/16/20/24/32, a runtime-topk dynamic kernel for the rest).

Import ``_ext`` through :mod:`agrimlxo.kernels` instead: it applies the
ABI probe and degrades to the pure-MLX fallback when the extension is
missing or was built against a mismatched mlx/nanobind.
"""

import importlib
import logging
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)


def load_ext() -> tuple[ModuleType | None, Exception | None]:
    """Import ``agrimlxo_kernels._ext`` and return ``(ext, import_error)``.

    Never raises: the caller decides how to degrade.
    """
    try:
        ext = importlib.import_module(f"{__name__}._ext")
    except Exception as exc:  # pragma: no cover - depends on native build
        logger.debug("agrimlxo_kernels._ext import failed: %s", exc)
        return None, exc
    # sanity: the metallib ships next to the extension
    if not any(Path(ext.__file__).parent.glob("*.metallib")):
        logger.warning(
            "agrimlxo_kernels: extension found but no .metallib next to it;"
            " native kernels disabled"
        )
        return None, None
    return ext, None
