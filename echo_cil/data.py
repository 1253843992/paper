from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


class EchoCILDataset:
    """Manifest-backed dataset with a small shard cache.

    Each item contains signal ``x`` [2048], order spectrum [3, 256], quality
    features [9], class id and source metadata.  The cache avoids loading all
    compressed shards into memory at once.
    """

    def __init__(self, processed_dir: str | Path, split: str,
                 classes: Optional[Sequence[int]] = None, cache_size: int = 3):
        self.root = Path(processed_dir)
        rows = list(csv.DictReader((self.root / "manifest.csv").open(encoding="utf-8-sig")))
        self.rows = [r for r in rows if r["split"] == split and (classes is None or int(r["class_id"]) in classes)]
        self.cache_size = cache_size
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, name: str) -> Dict[str, np.ndarray]:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        with np.load(self.root / name) as z:
            item = {k: z[k] for k in z.files}
        self._cache[name] = item
        self._cache.move_to_end(name)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return item

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        shard = self._load_shard(row["shard"])
        j = int(row["index"])
        return {"x": shard["x"][j].astype(np.float32),
                "order": shard["order"][j].astype(np.float32),
                "quality": shard["quality"][j].astype(np.float32),
                "y": int(row["class_id"]), "rpm": float(row["rpm"]),
                "record_id": int(row["record_id"]), "source": row["source"],
                "start": int(row["start"]), "shard": row["shard"],
                "index": j}


class DictDataset:
    """Torch-compatible wrapper for replay dictionaries and new samples."""
    def __init__(self, items: List[Dict[str, object]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_numpy(items: List[Dict[str, object]]) -> Dict[str, object]:
    return {"x": np.stack([i["x"] for i in items]),
            "order": np.stack([i["order"] for i in items]),
            "quality": np.stack([i["quality"] for i in items]),
            "y": np.asarray([i["y"] for i in items], dtype=np.int64),
            "record_id": np.asarray([i["record_id"] for i in items], dtype=np.int64)}
