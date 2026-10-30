# SPDX-License-Identifier: Apache-2.0
"""Build the optional AgrimlxO native Metal kernels.

Builds nothing unless explicitly requested (the base AgrimlxO install is
pure Python). On an Apple Silicon Mac with the Metal toolchain:

    AGRIMLXO_WITH_KERNELS=1 pip install ./native

or, from this directory:

    pip install --with-custom-kernel .

The extension is built with MLX's own extension helper (``mlx.extension``),
which pins the nanobind ABI against the installed mlx wheel — the same
mechanism omlx uses, so a mismatched rebuild produces a clean
"native kernels disabled" degradation instead of import errors.
"""

import os
import sys

from setuptools import setup

CUSTOM_KERNEL_FLAG = "--with-custom-kernel"
TRUTHY = {"1", "true", "yes", "on"}


def _with_kernels() -> bool:
    if CUSTOM_KERNEL_FLAG in sys.argv:
        sys.argv.remove(CUSTOM_KERNEL_FLAG)
        return True
    return os.environ.get("AGRIMLXO_WITH_KERNELS", "").strip().lower() in TRUTHY


def _kernels_build_kwargs() -> dict:
    if not _with_kernels():
        return {}

    if sys.platform != "darwin":
        raise RuntimeError(
            "AGRIMLXO_WITH_KERNELS=1 requires macOS (the kernels are Metal)."
        )

    from mlx import extension

    root = os.path.join(os.path.dirname(__file__), "agrimlxo_kernels", "csrc")
    shared = os.path.join(root, "include")

    return {
        "ext_modules": [
            extension.MetalExtension(
                "agrimlxo_kernels._ext",
                sources=[
                    os.path.join(root, "bindings.cpp"),
                    os.path.join(root, "moe_weighted_sum.cpp"),
                ],
                include_dirs=[root, shared],
                metal_libraries=[
                    extension.MetalLibrary(
                        "agrimlxo_kernels_lib",
                        sources=[
                            os.path.join(root, "moe_weighted_sum.metal"),
                        ],
                    ),
                ],
                extra_link_args=[
                    "SHELL:-framework Accelerate",
                    "SHELL:-framework Foundation",
                    "SHELL:-framework Metal",
                ],
            )
        ],
        "cmdclass": {
            "build_ext": extension.BuildExtension.with_options(
                generate_stubs=False
            ),
        },
    }


setup(**_kernels_build_kwargs())
