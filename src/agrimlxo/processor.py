# SPDX-License-Identifier: Apache-2.0
"""Vision preprocessing for AgrimlxO: data-URL images -> engine inputs.

Two modes:
  * ``processor`` — the checkpoint's full AutoProcessor (needs torchvision
    on some transformers builds, for the video sub-processor);
  * ``manual`` — fallback without torchvision: AutoImageProcessor (images
    only) + the tokenizer, with the <|image_pad|> token expansion done here
    (each image's pad run grows to t*(h/merge)*(w/merge) tokens).

Both return {input_ids, pixel_values, image_grid_thw} ready for the engine.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class VisionProcessor:
    def __init__(
        self,
        model_path: Path,
        image_token_id: Optional[int] = None,
        spatial_merge_size: int = 2,
    ):
        self.image_token_id = image_token_id
        self.merge = int(spatial_merge_size)
        self.mode = "processor"
        try:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(str(model_path))
            self.tokenizer = None
            self.image_processor = None
            return
        except ImportError as e:
            if "torchvision" not in str(e).lower():
                raise
            logger.info(
                "AutoProcessor richiede torchvision (solo per il video): uso"
                " il percorso manuale image-processor + tokenizer"
            )
        from transformers import AutoImageProcessor, AutoTokenizer

        self.processor = None
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.image_processor = AutoImageProcessor.from_pretrained(str(model_path))
        self.mode = "manual"

    def prepare(
        self,
        rendered_text: str,
        image_dataurls: List[str],
    ) -> Dict[str, Any]:
        """Rendered chat text (with vision placeholders) + data URLs ->
        engine-ready inputs."""
        images = [self._decode(url) for url in image_dataurls]
        prepared: Dict[str, Any] = {}

        if self.mode == "manual":
            # grid first: the pad expansion needs the merged-token counts
            iv = self.image_processor(images=images, return_tensors=None)
            grid = [[int(x) for x in row] for row in np.asarray(iv["image_grid_thw"])]
            counts = [int(t * (h // self.merge) * (w // self.merge))
                      for t, h, w in grid]
            prepared["input_ids"] = self._manual_ids(rendered_text, counts)
            prepared["pixel_values"] = mx.array(
                np.asarray(iv["pixel_values"], dtype=np.float32)
            )
            prepared["image_grid_thw"] = mx.array(grid)
            return prepared

        out = self.processor(
            text=[rendered_text],
            images=images if images else None,
            padding=False,
            return_tensors=None,
        )
        prepared["input_ids"] = [int(t) for t in out["input_ids"][0]]
        pv = out.get("pixel_values")
        grid = out.get("image_grid_thw")
        if pv is not None:
            prepared["pixel_values"] = mx.array(np.asarray(pv, dtype=np.float32))
        if grid is not None:
            prepared["image_grid_thw"] = mx.array(
                [[int(x) for x in row] for row in np.asarray(grid)]
            )
        return prepared

    def _manual_ids(self, rendered_text: str, counts: List[int]) -> List[int]:
        """Encode the rendered text; each image-pad run expands to the
        merged-token count of its image (counts in image order)."""
        assert self.image_token_id is not None
        ids = self.tokenizer.encode(rendered_text, add_special_tokens=False)
        out: List[int] = []
        run = 0
        img_idx = 0
        for t in ids:
            if t == self.image_token_id:
                run += 1
                continue
            if run:
                n = counts[img_idx] if img_idx < len(counts) else run
                out.extend([self.image_token_id] * n)
                img_idx += 1
                run = 0
            out.append(t)
        if run:
            n = counts[img_idx] if img_idx < len(counts) else run
            out.extend([self.image_token_id] * n)
        if img_idx < len(counts) - 1 and counts:
            logger.warning(
                "image pad runs (%d) fewer than grids (%d)", img_idx + 1, len(counts)
            )
        return out

    @staticmethod
    def _decode(dataurl: str):
        from PIL import Image

        payload = dataurl.split(",", 1)[1] if "," in dataurl else dataurl
        img = Image.open(io.BytesIO(base64.b64decode(payload)))
        return img.convert("RGB")
