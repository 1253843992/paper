from __future__ import annotations

import hashlib
import io
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .physics import WINDOW


MEMORY_DTYPE = np.dtype([
    ('x', '<f4', (WINDOW,)), ('label', '<u2'), ('rpm', '<f4'),
    ('file_id', '<u2'), ('start', '<u4'), ('sample_id', '<u8'),
], align=False)


class ReplayBuffer:
    """Auditable waveform-only replay under an actual serialized-byte cap."""

    def __init__(self, byte_budget: int = 512 * 1024, seed: int = 0):
        self.byte_budget = int(byte_budget)
        self.seed = int(seed)
        self.items: Dict[int, List[dict]] = defaultdict(list)
        if self.byte_budget <= 0:
            raise ValueError('byte_budget must be positive')

    def _priority(self, item: dict) -> bytes:
        payload = f'{self.seed}:{int(item["sample_id"])}'.encode()
        return hashlib.blake2b(payload, digest_size=16).digest()

    @staticmethod
    def _persistent(item: dict) -> dict:
        return {
            'x': np.asarray(item['x'], dtype=np.float32).copy(),
            'y': int(item['y']), 'rpm': float(item['rpm']),
            'record_id': int(item['record_id']), 'start': int(item['start']),
            'sample_id': int(item['sample_id']),
        }

    @staticmethod
    def _array(items: Sequence[dict]) -> np.ndarray:
        array = np.empty(len(items), dtype=MEMORY_DTYPE)
        for index, item in enumerate(items):
            array[index] = (
                np.asarray(item['x'], dtype=np.float32), int(item['y']),
                float(item['rpm']), int(item['record_id']),
                int(item['start']), int(item['sample_id']),
            )
        return array

    @classmethod
    def serialized_size_of(cls, items: Sequence[dict]) -> int:
        stream = io.BytesIO()
        np.save(stream, cls._array(items), allow_pickle=False)
        return stream.tell()

    def _capacity(self) -> int:
        approximate = self.byte_budget // MEMORY_DTYPE.itemsize
        while approximate > 0:
            dummy = np.zeros(approximate, dtype=MEMORY_DTYPE)
            stream = io.BytesIO(); np.save(stream, dummy, allow_pickle=False)
            if stream.tell() <= self.byte_budget:
                return approximate
            approximate -= 1
        return 0

    def update(self, new_candidates: Sequence[dict],
               active_classes: Sequence[int]) -> None:
        classes = list(active_classes)
        pool: Dict[int, Dict[int, dict]] = {c: {} for c in classes}
        for item in self.all() + list(new_candidates):
            label = int(item['y'])
            if label in pool:
                persistent = self._persistent(item)
                pool[label].setdefault(int(persistent['sample_id']), persistent)

        capacity = self._capacity()
        base, remainder = divmod(capacity, max(1, len(classes)))
        self.items = defaultdict(list)
        for position, label in enumerate(classes):
            quota = base + int(position < remainder)
            ranked = sorted(pool[label].values(), key=self._priority)
            self.items[label] = ranked[:quota]

        # The dtype and NumPy header are both included in the check.
        while self.bytes_used > self.byte_budget:
            label = max(classes, key=lambda c: len(self.items[c]))
            self.items[label].pop()

    @property
    def bytes_used(self) -> int:
        return self.serialized_size_of(self.all())

    def all(self) -> List[dict]:
        return [item for label in sorted(self.items)
                for item in self.items[label]]

    def counts(self) -> Dict[str, int]:
        return {str(label): len(self.items[label]) for label in sorted(self.items)}

    def save(self, path: str | Path) -> int:
        path = Path(path)
        np.save(path, self._array(self.all()), allow_pickle=False)
        size = path.stat().st_size
        if size > self.byte_budget:
            raise RuntimeError(f'memory serialization exceeds budget: {size}')
        return size
