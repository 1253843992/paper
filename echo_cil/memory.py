from __future__ import annotations

import hashlib
import io
import math
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
    """Replay capped by both bytes and <=10% of distinct seen training windows."""

    def __init__(self, byte_budget: int = 512 * 1024, seed: int = 0,
                 max_fraction: float = 0.1):
        self.byte_budget = int(byte_budget)
        self.seed = int(seed)
        self.items: Dict[int, List[dict]] = defaultdict(list)
        self.max_fraction = float(max_fraction)
        self.seen_counts: Dict[int, int] = {}
        if not 0.0 < self.max_fraction <= 0.1:
            raise ValueError('max_fraction must be in (0, 0.1]')
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

    def _stratified_select(self, candidates: Sequence[dict], quota: int) -> List[dict]:
        """Deterministically cover records/load conditions before taking repeats."""
        by_record: Dict[int, List[dict]] = defaultdict(list)
        for item in candidates:
            by_record[int(item['record_id'])].append(item)
        for record in by_record:
            by_record[record].sort(key=self._priority)
        selected: List[dict] = []
        depth = 0
        records = sorted(by_record)
        while len(selected) < quota:
            added = False
            for record in records:
                values = by_record[record]
                if depth < len(values):
                    selected.append(values[depth])
                    added = True
                    if len(selected) == quota:
                        break
            if not added:
                break
            depth += 1
        return selected

    def update(self, new_candidates: Sequence[dict],
               active_classes: Sequence[int]) -> None:
        classes = list(active_classes)
        # Only newly introduced classes may contribute fresh samples. Old
        # samples must come exclusively from the previously retained buffer.
        new_by_class: Dict[int, set] = defaultdict(set)
        for item in new_candidates:
            label = int(item['y'])
            if label in self.seen_counts:
                raise ValueError('cannot reload old-class training candidates')
            if label not in classes:
                raise ValueError('candidate class is not active')
            new_by_class[label].add(int(item['sample_id']))
        counts = {**self.seen_counts,
                  **{label: len(ids) for label, ids in new_by_class.items()}}
        if set(classes) != set(counts):
            raise ValueError('active classes must match all seen classes')
        sample_limit = math.floor(sum(counts.values()) * self.max_fraction)
        capacity = min(self._capacity(), sample_limit)
        if capacity < len(classes):
            raise ValueError('replay budget cannot retain one sample per class; '
                             'increase bytes/fraction without exceeding 10%')
        pool: Dict[int, Dict[int, dict]] = {c: {} for c in classes}
        for item in self.all() + list(new_candidates):
            label = int(item['y'])
            if label in pool:
                persistent = self._persistent(item)
                pool[label].setdefault(int(persistent['sample_id']), persistent)

        # Redistribute unfilled quotas without recovering discarded old data.
        quotas = {label: 0 for label in classes}
        remaining = capacity
        while remaining:
            added = False
            for label in classes:
                if quotas[label] < len(pool[label]):
                    quotas[label] += 1
                    remaining -= 1
                    added = True
                    if not remaining:
                        break
            if not added:
                break
        self.seen_counts = counts
        self.items = defaultdict(list)
        for label in classes:
            self.items[label] = self._stratified_select(
                list(pool[label].values()), quotas[label])

        # The dtype and NumPy header are both included in the check.
        while self.bytes_used > self.byte_budget:
            label = max(classes, key=lambda c: len(self.items[c]))
            self.items[label].pop()

    @property
    def seen_training_count(self) -> int:
        return sum(self.seen_counts.values())

    @property
    def sample_limit(self) -> int:
        return math.floor(self.seen_training_count * self.max_fraction)

    @property
    def bytes_used(self) -> int:
        return self.serialized_size_of(self.all())

    def all(self) -> List[dict]:
        return [item for label in sorted(self.items)
                for item in self.items[label]]

    def counts(self) -> Dict[str, int]:
        return {str(label): len(self.items[label]) for label in sorted(self.items)}

    def save(self, path: str | Path) -> int:
        if len(self.all()) > self.sample_limit:
            raise RuntimeError('replay sample fraction exceeded')
        path = Path(path)
        np.save(path, self._array(self.all()), allow_pickle=False)
        size = path.stat().st_size
        if size > self.byte_budget:
            raise RuntimeError(f'memory serialization exceeds budget: {size}')
        return size
