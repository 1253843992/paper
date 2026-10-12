from __future__ import annotations

import csv
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler

from .physics import physics_features


class EchoCILDataset:
    """Manifest-backed split with a bounded compressed-shard cache."""

    def __init__(self, processed_dir: str | Path, split: str,
                 classes: Optional[Sequence[int]] = None, cache_size: int = 3):
        self.root = Path(processed_dir)
        rows = list(csv.DictReader(
            (self.root / 'manifest.csv').open(encoding='utf-8-sig')))
        allowed = set(classes) if classes is not None else None
        self.rows = [row for row in rows if row['split'] == split and
                     (allowed is None or int(row['class_id']) in allowed)]
        self.cache_size = cache_size
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, name: str) -> Dict[str, np.ndarray]:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        with np.load(self.root / name) as archive:
            item = {key: archive[key] for key in archive.files}
        self._cache[name] = item
        self._cache.move_to_end(name)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return item

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        shard = self._load_shard(row['shard'])
        offset = int(row['index'])
        return {
            'x': shard['x'][offset].astype(np.float32),
            'order': shard['order'][offset].astype(np.float32),
            'quality': shard['quality'][offset].astype(np.float32),
            'y': int(row['class_id']), 'rpm': float(row['rpm']),
            'record_id': int(row['record_id']), 'source': row['source'],
            'start': int(row['start']), 'sample_id': int(row['sample_id']),
            'shard': row['shard'], 'index': offset,
        }


class DictDataset:
    """Replay/current item wrapper that reconstructs derived physics on demand."""

    def __init__(self, items: List[Dict[str, object]],
                 physics_stats: Optional[Mapping[str, object]] = None):
        self.items = items
        self.physics_stats = physics_stats

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.items[index]
        if 'order' in item and 'quality' in item:
            return item
        if self.physics_stats is None:
            raise ValueError('physics_stats are required for waveform-only replay')
        order, quality = physics_features(
            np.asarray(item['x'], dtype=np.float32), float(item['rpm']),
            self.physics_stats)
        enriched = dict(item)
        enriched['order'] = order
        enriched['quality'] = quality
        return enriched


class GroupBalancedBatchSampler(Sampler[List[int]]):
    """Exact 32/32 old-new batches with uniform classes inside each group."""

    def __init__(self, dataset: DictDataset, old_classes: Sequence[int],
                 new_classes: Sequence[int], batch_size: int, seed: int,
                 steps: Optional[int] = None):
        if batch_size <= 0 or (old_classes and batch_size % 2):
            raise ValueError('incremental batch size must be positive and even')
        self.dataset = dataset
        self.old_classes = list(old_classes)
        self.new_classes = list(new_classes)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.steps = steps or max(1, int(np.ceil(len(dataset) / batch_size)))
        self.by_class: Dict[int, List[int]] = defaultdict(list)
        for index, item in enumerate(dataset.items):
            self.by_class[int(item['y'])].append(index)
        missing = [c for c in self.old_classes + self.new_classes
                   if not self.by_class[c]]
        if missing:
            raise ValueError(f'classes without training samples: {missing}')

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps

    def _draw_group(self, rng: np.random.Generator,
                    classes: Sequence[int], count: int) -> List[int]:
        labels = rng.choice(np.asarray(classes), size=count, replace=True)
        return [int(rng.choice(self.by_class[int(label)])) for label in labels]

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 104729 * self.epoch)
        for _ in range(self.steps):
            if self.old_classes:
                half = self.batch_size // 2
                indices = self._draw_group(rng, self.old_classes, half)
                indices += self._draw_group(rng, self.new_classes,
                                            self.batch_size - half)
            else:
                indices = self._draw_group(
                    rng, self.new_classes, self.batch_size)
            rng.shuffle(indices)
            yield indices


def collate_numpy(items: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    return {
        'x': np.stack([item['x'] for item in items]).astype(np.float32),
        'order': np.stack([item['order'] for item in items]).astype(np.float32),
        'quality': np.stack([item['quality'] for item in items]).astype(np.float32),
        'y': np.asarray([item['y'] for item in items], dtype=np.int64),
        'rpm': np.asarray([item['rpm'] for item in items], dtype=np.float32),
        'record_id': np.asarray([item['record_id'] for item in items], dtype=np.int64),
        'sample_id': np.asarray([item['sample_id'] for item in items], dtype=np.uint64),
    }
