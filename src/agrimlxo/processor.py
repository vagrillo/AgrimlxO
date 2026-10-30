# SPDX-License-Identifier: Apache-2.0
"""Vision preprocessing for AgrimlxO: data-URL images -> processor outputs.

Uses the checkpoint's own processor (transformers AutoProcessor with the
Qwen3.5 preprocessor config) to turn PIL images + the rendered chat text
(into <|vision_start|><|image_pad|><|vision_end|> blocks) into
input_ids / pixel_values / image_grid_thw ready for the engine.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx


class VisionProcessor:
    def __init__(self, model_path: Path):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(model_path))

    def prepare(
        self,
        rendered_text: str,
        image_dataurls: List[str],
    ) -> Dict[str, Any]:
        """Rendered chat text (with vision placeholders) + data URLs ->
        engine-ready inputs."""
        images = [self._decode(url) for url in image_dataurls]
        out = self.processor(
            text=[rendered_text],
            images=images if images else None,
            padding=False,
            return_tensors=None,
        )
        input_ids = [int(t) for t in out["input_ids"][0]]
        prepared: Dict[str, Any] = {"input_ids": input_ids}
        pv = out.get("pixel_values")
        if pv is not None:
            prepared["pixel_values"] = mx.array(np_stack(pv))
        grid = out.get("image_grid_thw")
        if grid is not None:
            prepared["image_grid_thw"] = mx.array(
                [[int(x) for x in row] for row in grid]
            )
        return prepared

    @staticmethod
    def _decode(dataurl: str):
        from PIL import Image

        payload = dataurl.split(",", 1)[1] if "," in dataurl else dataurl
        img = Image.open(io.BytesIO(base64.b64decode(payload)))
        return img.convert("RGB")


def np_stack(pv):
    """Processor pixel values: nested lists (or numpy) -> float32 array."""
    import numpy as np

    return np.asarray(pv, dtype=np.float32)
