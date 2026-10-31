# SPDX-License-Identifier: Apache-2.0
"""./kache — persistent conversation KV snapshots, resistant to restarts.

Each entry: a conversation fingerprint -> {tokens, cache states} captured at
the prefill boundary (see engine.snapshots). States are serialized with
safetensors (mx-native roundtrip): the state tree (arrays, lists, scalars) is
described by a JSON sidecar and the arrays stored in a .safetensors file. An
LRU index (last_used, then uses) evicts the oldest entries when the directory
exceeds the byte budget.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def state_nbytes(state: Any) -> int:
    if isinstance(state, (mx.array, np.ndarray)):
        return int(state.nbytes)
    if isinstance(state, (list, tuple)):
        return sum(state_nbytes(x) for x in state)
    return 0


class Kache:
    def __init__(self, kache_dir: str = "./kache", budget_bytes: int = 8 << 30):
        self.dir = Path(kache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget_bytes)
        self.index_path = self.dir / "index.json"
        self.index: Dict[str, dict] = {}
        self._load_index()

    def _load_index(self):
        try:
            if self.index_path.exists():
                self.index = json.loads(self.index_path.read_text())
        except Exception:
            self.index = {}

    def _save_index(self):
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index))
        tmp.replace(self.index_path)

    def _total_bytes(self) -> int:
        return int(sum(e.get("nbytes", 0) for e in self.index.values()))

    def _evict(self):
        while self._total_bytes() > self.budget and self.index:
            oldest = min(
                self.index,
                key=lambda k: (self.index[k].get("last_used", 0),
                               self.index[k].get("uses", 0)),
            )
            entry = self.index.pop(oldest)
            for suffix in (".safetensors", ".json"):
                f = self.dir / f"{oldest}{suffix}"
                if f.exists():
                    f.unlink()

    def save(self, key: str, tokens: List[int], states: List[Tuple[str, Any]]):
        try:
            arrays: Dict[str, mx.array] = {}
            tree: List[Any] = [["n", len(states)]]

            def flat(prefix: str, x: Any):
                if isinstance(x, np.ndarray):
                    x = mx.array(x)
                if isinstance(x, mx.array):
                    arrays[prefix] = x
                    tree.append(["arr", prefix])
                elif isinstance(x, (list, tuple)):
                    tree.append(["seq" if isinstance(x, list) else "tup", len(x)])
                    for i, item in enumerate(x):
                        flat(f"{prefix}.{i}", item)
                else:
                    tree.append(["scalar", x])

            for i, (cls_name, st) in enumerate(states):
                tree.append(["cls", cls_name])
                flat(f"s{i}", st)

            f = self.dir / f"{key}.safetensors"
            tmp = self.dir / f".{key}.tmp.safetensors"
            mx.save_safetensors(str(tmp), arrays)
            tmp.replace(f)
            meta = {"tokens": [int(t) for t in tokens], "tree": tree}
            (self.dir / f"{key}.json").write_text(
                json.dumps(meta, default=str)
            )

            nbytes = sum(state_nbytes(s) for _, s in states)
            entry = self.index.get(key, {"uses": 0})
            entry.update({
                "tokens_len": len(tokens),
                "nbytes": int(nbytes),
                "last_used": time.time(),
                "uses": entry.get("uses", 0) + 1,
            })
            self.index[key] = entry
            self._evict()
            self._save_index()
        except Exception as e:
            logger.warning("kache save %s: %s", key, e)

    def load(self, key: str) -> Optional[Tuple[List[int], List[Tuple[str, Any]]]]:
        f = self.dir / f"{key}.safetensors"
        meta_f = self.dir / f"{key}.json"
        if not f.exists() or not meta_f.exists():
            return None
        try:
            arrays = dict(mx.load(str(f)))
            meta = json.loads(meta_f.read_text())
            tree = meta["tree"]

            def rebuild(pos: int):
                kind, val = tree[pos]
                if kind == "arr":
                    return arrays[val], pos + 1
                if kind in ("seq", "tup"):
                    items = []
                    p = pos + 1
                    for _ in range(int(val)):
                        item, p = rebuild(p)
                        items.append(item)
                    return (tuple(items) if kind == "tup" else items), p
                return val, pos + 1

            n_states = int(tree[0][1])
            states = []
            p = 1
            for _ in range(n_states):
                cls_name = tree[p][1]
                st, p = rebuild(p + 1)
                states.append((cls_name, st))

            entry = self.index.get(key, {})
            entry["last_used"] = time.time()
            entry["uses"] = entry.get("uses", 0) + 1
            self.index[key] = entry
            self._save_index()
            return [int(t) for t in meta["tokens"]], states
        except Exception as e:
            logger.warning("kache load %s: %s", key, e)
            f.unlink(missing_ok=True)
            meta_f.unlink(missing_ok=True)
            self.index.pop(key, None)
            self._save_index()
            return None

    def stats(self) -> Dict:
        return {
            "entries": len(self.index),
            "nbytes": self._total_bytes(),
            "budget_bytes": self.budget,
            "dir": str(self.dir),
        }
