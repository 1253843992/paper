from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np


class ReplayBuffer:
    """Class-balanced byte-budget replay buffer storing processed samples."""
    def __init__(self, byte_budget: int = 32 * 1024 * 1024, seed: int = 2027):
        self.byte_budget, self.rng = byte_budget, random.Random(seed)
        self.items: Dict[int, List[dict]] = defaultdict(list)

    def _bytes_per_item(self, item: dict) -> int:
        return int(item["x"].nbytes + item["order"].nbytes + item["quality"].nbytes + 64)

    @staticmethod
    def _sample_key(item: dict):
        """Return a stable key so an exemplar cannot enter memory twice."""
        if "shard" in item and "index" in item:
            return str(item["shard"]), int(item["index"])
        if "start" in item:
            return int(item["record_id"]), int(item["start"]), int(item["y"])
        digest = hashlib.blake2b(np.asarray(item["x"]).tobytes(), digest_size=16).digest()
        return int(item["record_id"]), int(item["y"]), digest

    def update(self, candidates: List[dict], active_classes: List[int]) -> None:
        pool = {c: list(self.items.get(c, [])) for c in active_classes}
        for item in candidates:
            pool[int(item["y"])].append(item)

        # The caller may pass replay items accidentally.  De-duplicate here as
        # a second line of defence before sampling the next memory.
        for c in active_classes:
            unique = {}
            for item in pool.get(c, []):
                unique.setdefault(self._sample_key(item), item)
            pool[c] = list(unique.values())

        available = [len(pool[c]) for c in active_classes if pool.get(c)]
        if not available:
            self.items = defaultdict(list)
            return
        example = next(pool[c][0] for c in active_classes if pool.get(c))
        budget_cap = max(1, self.byte_budget // max(1, len(active_classes))
                         // max(1, self._bytes_per_item(example)))
        # Use the same number of exemplars per class.  Leaving a small part of
        # the byte budget unused is preferable to biasing training to classes
        # that happen to have longer source recordings.
        per_class = min(budget_cap, min(available))
        self.items = defaultdict(list)
        for c in active_classes:
            values = pool.get(c, [])
            self.rng.shuffle(values)
            self.items[c] = values[:per_class]
        while self.bytes_used > self.byte_budget:
            c = max(self.items, key=lambda k: len(self.items[k]))
            if not self.items[c]: break
            self.items[c].pop()

    @property
    def bytes_used(self) -> int:
        return sum(self._bytes_per_item(i) for values in self.items.values() for i in values)

    def all(self) -> List[dict]:
        return [i for values in self.items.values() for i in values]
