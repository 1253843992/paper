"""Strict ECHO-CIL preprocessing for the CWRU drive-end dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from echo_cil.physics import (
    BANDS_HZ, FS_TARGET, HARMONIC_ORDERS, ORDER_MAX, ORDER_POINTS, WINDOW,
    apply_physics_stats, raw_physics_features,
)


TRAIN_STRIDE = 1024
EVAL_STRIDE = 2048
CLASS_ORDER = [
    'N', 'IR007', 'OR007@6', 'B007', 'IR014',
    'OR014@6', 'B014', 'IR021', 'OR021@6', 'B021',
]
CLASS_ID = {name: index for index, name in enumerate(CLASS_ORDER)}
TASK_ORDER = {'O0': [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]}
LEGACY_LABEL = {97: 'N', 105: 'IR007', 118: 'IR014', 130: 'IR021',
                169: 'OR007@6', 185: 'OR014@6', 197: 'OR021@6',
                209: 'B007', 222: 'B014', 234: 'B021'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def infer_label(path: Path) -> str:
    match = re.search(r'(?:^|_)(IR|OR|B)(007|014|021)(?:@6)?(?:_|\.)',
                      path.name, re.I)
    if match:
        kind, size = match.group(1).upper(), match.group(2)
        return f'OR{size}@6' if kind == 'OR' else f'{kind}{size}'
    if path.name.lower().startswith('normal'):
        return 'N'
    match = re.match(r'(\d+)\.mat$', path.name, re.I)
    if match and int(match.group(1)) in LEGACY_LABEL:
        return LEGACY_LABEL[int(match.group(1))]
    raise ValueError(f'cannot infer CWRU label from {path}')


def infer_source_fs(path: Path) -> int:
    text = str(path).lower()
    if '48k' in text:
        return 48_000
    # The official CWRU normal-baseline records (97--100) are 48 kHz even
    # when a local mirror omits "48k" from its directory name.
    numeric = re.search(r'(?:^|_)(\d+)(?:\.mat)$', path.name.lower())
    record_number = int(numeric.group(1)) if numeric else None
    if path.name.lower().startswith('normal') or record_number in (97, 98, 99, 100):
        return 48_000
    if '12k' in text or record_number in set(LEGACY_LABEL) - {97}:
        return 12_000
    raise ValueError(
        f'cannot audit source sample rate from path: {path}. '
        'Place records under a 12k/48k-labelled directory.')


def read_record(path: Path) -> Tuple[np.ndarray, float, str, int]:
    data = loadmat(path, squeeze_me=True)
    keys = [key for key in data if not key.startswith('__') and 'DE_time' in key]
    if not keys:
        keys = [key for key in data if not key.startswith('__')
                and 'DE' in key and np.asarray(data[key]).ndim > 0]
    if not keys:
        raise ValueError(f'no drive-end signal in {path}')
    key = sorted(keys, key=lambda value: ('DE_time' not in value, len(value)))[0]
    x = np.asarray(data[key], dtype=np.float64).reshape(-1)
    rpm_keys = [key for key in data if 'RPM' in key.upper()]
    rpm = (float(np.asarray(data[rpm_keys[0]]).reshape(-1)[0])
           if rpm_keys else float('nan'))
    return x, rpm, key, infer_source_fs(path)


def split_blocks(n_samples: int, source_fs: int) -> Dict[str, Tuple[int, int]]:
    first, second = int(n_samples * 0.60), int(n_samples * 0.80)
    guard = int(round(WINDOW * source_fs / FS_TARGET))
    return {
        'train': (0, max(0, first - guard)),
        'val': (min(n_samples, first + guard), max(0, second - guard)),
        'test': (min(n_samples, second + guard), n_samples),
    }


def resample_block(x: np.ndarray, source_fs: int) -> np.ndarray:
    if source_fs == FS_TARGET:
        return np.asarray(x, dtype=np.float64)
    divisor = int(np.gcd(source_fs, FS_TARGET))
    return resample_poly(x, FS_TARGET // divisor, source_fs // divisor)


def starts_for(length: int, stride: int) -> np.ndarray:
    if length < WINDOW:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, length - WINDOW + 1, stride, dtype=np.int64)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path('echo_cil_processed'))
    parser.add_argument('--base-classes', type=int, nargs=4,
                        default=TASK_ORDER['O0'][0],
                        help='four base classes that may define normalization')
    args = parser.parse_args()
    source, output = args.input, args.output
    base_classes = list(args.base_classes)
    if len(set(base_classes)) != 4 or any(c not in range(10) for c in base_classes):
        raise ValueError('--base-classes must contain four distinct IDs from 0 to 9')
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / 'shards'
    shard_dir.mkdir(exist_ok=True)

    records = []
    for path in sorted(source.rglob('*.mat')):
        x, rpm, de_key, source_fs = read_record(path)
        label = infer_label(path)
        records.append({
            'path': path, 'x': x, 'rpm': rpm, 'de_key': de_key,
            'source_fs': source_fs, 'label': label,
            'class_id': CLASS_ID[label], 'load': path.parent.name,
        })
    if len(records) != 40:
        raise ValueError(f'expected 40 audited CWRU records, found {len(records)}')

    load_rpms: Dict[str, List[float]] = {}
    for record in records:
        if np.isfinite(record['rpm']) and record['rpm'] > 0:
            load_rpms.setdefault(record['load'], []).append(record['rpm'])
    for record in records:
        record['rpm_raw'] = record['rpm']
        record['rpm_imputed'] = not np.isfinite(record['rpm']) or record['rpm'] <= 0
        if record['rpm_imputed']:
            candidates = load_rpms.get(record['load'], [])
            if not candidates:
                raise ValueError(f'cannot impute RPM for {record["path"]}')
            record['rpm'] = float(np.median(candidates))

    # Only O0 base classes and only their training blocks determine all fixed
    # normalization statistics.
    base_parts = []
    for record in records:
        if record['class_id'] not in base_classes:
            continue
        lo, hi = split_blocks(len(record['x']), record['source_fs'])['train']
        base_parts.append(resample_block(record['x'][lo:hi], record['source_fs']))
    base = np.concatenate(base_parts)
    waveform_mean = float(base.mean())
    waveform_std = float(base.std() + 1e-12)

    pending = []
    inventory_rows = []
    base_orders, base_quality = [], []
    for record_id, record in enumerate(records):
        blocks = split_blocks(len(record['x']), record['source_fs'])
        inventory_rows.append({
            'record_id': record_id,
            'source': str(record['path'].relative_to(source)),
            'sha256': sha256(record['path']), 'label': record['label'],
            'class_id': record['class_id'], 'load': record['load'],
            'de_key': record['de_key'], 'n_samples': len(record['x']),
            'fs_source': record['source_fs'], 'rpm_raw': record['rpm_raw'],
            'rpm_used': record['rpm'], 'rpm_imputed': record['rpm_imputed'],
            'train_block': blocks['train'], 'val_block': blocks['val'],
            'test_block': blocks['test'],
        })
        for split, (lo, hi) in blocks.items():
            block = resample_block(record['x'][lo:hi], record['source_fs'])
            stride = TRAIN_STRIDE if split == 'train' else EVAL_STRIDE
            starts = starts_for(len(block), stride)
            xs, orders, qualities = [], [], []
            for local_start in starts:
                segment = block[int(local_start):int(local_start) + WINDOW]
                normalized = ((segment - waveform_mean) / waveform_std).astype(np.float32)
                order, quality = raw_physics_features(normalized, record['rpm'])
                xs.append(normalized); orders.append(order); qualities.append(quality)
            if not xs:
                continue
            order_array = np.stack(orders)
            quality_array = np.stack(qualities)
            if split == 'train' and record['class_id'] in base_classes:
                base_orders.append(order_array)
                base_quality.append(quality_array)
            pending.append({
                'record_id': record_id, 'record': record, 'split': split,
                'block_lo': lo, 'starts': starts, 'x': np.stack(xs),
                'order': order_array, 'quality': quality_array,
            })

    order_base = np.concatenate(base_orders)
    quality_base = np.concatenate(base_quality)
    order_mean = order_base.mean(axis=(0, 2))
    order_std = order_base.std(axis=(0, 2)).clip(min=1e-6)
    quality_mean = quality_base.mean(axis=0)
    quality_std = quality_base.std(axis=0).clip(min=1e-6)
    stats = {
        'protocol_version': 'ECHO-CIL-strict',
        'waveform_mean': waveform_mean, 'waveform_std': waveform_std,
        # Legacy aliases retained only for transparent inspection.
        'mean': waveform_mean, 'std': waveform_std,
        'order_mean': order_mean.tolist(), 'order_std': order_std.tolist(),
        'quality_mean': quality_mean.tolist(),
        'quality_std': quality_std.tolist(),
        'source_classes': base_classes, 'target_fs': FS_TARGET,
    }

    manifest_rows = []
    for entry in pending:
        record = entry['record']
        standardized_order, standardized_quality = apply_physics_stats(
            entry['order'], entry['quality'], stats)
        shard_name = f'r{entry["record_id"]:03d}_{entry["split"]}.npz'
        np.savez_compressed(
            shard_dir / shard_name, x=entry['x'], order=standardized_order,
            quality=standardized_quality, start=entry['starts'],
            rpm=np.full(len(entry['starts']), record['rpm'], dtype=np.float32),
            class_id=np.full(len(entry['starts']), record['class_id'], dtype=np.int64),
        )
        for index, local_start in enumerate(entry['starts']):
            source_start = entry['block_lo'] + int(round(
                int(local_start) * record['source_fs'] / FS_TARGET))
            source_end = entry['block_lo'] + int(round(
                (int(local_start) + WINDOW) * record['source_fs'] / FS_TARGET))
            sample_id = int.from_bytes(hashlib.blake2b(
                f'{entry["record_id"]}:{source_start}'.encode(),
                digest_size=8).digest(), 'little')
            manifest_rows.append({
                'record_id': entry['record_id'],
                'source': str(record['path'].relative_to(source)),
                'shard': f'shards/{shard_name}', 'index': index,
                'split': entry['split'], 'class_id': record['class_id'],
                'label': record['label'], 'load': record['load'],
                'rpm': record['rpm'], 'rpm_imputed': record['rpm_imputed'],
                'start': source_start, 'end': source_end,
                'sample_id': sample_id, 'fs_source': record['source_fs'],
                'fs': FS_TARGET, 'window': WINDOW,
            })

    for name, rows in (('manifest.csv', manifest_rows),
                       ('record_inventory.csv', inventory_rows)):
        with (output / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    write_json(output / 'normalization.json', stats)
    write_json(output / 'class_mapping.json', {
        'class_id': CLASS_ID, 'class_order': CLASS_ORDER,
        'primary_order': TASK_ORDER['O0'],
        'units_note': '007/014/021 are inch fault sizes',
    })
    write_json(output / 'preprocess_config.json', {
        'protocol_version': 'ECHO-CIL-strict',
        'split': 'blocked_60_20_20_with_guards',
        'window': WINDOW, 'train_stride': TRAIN_STRIDE,
        'eval_stride': EVAL_STRIDE, 'guard_target_samples': WINDOW,
        'bands_hz': BANDS_HZ, 'filter': 'Butterworth-4 SOS zero-phase',
        'reflect_pad': 256, 'order_grid': [0, ORDER_MAX, ORDER_POINTS],
        'harmonic_queries': HARMONIC_ORDERS.tolist(),
        'quality': 'per-band log-PBR, normalized entropy, RPM +/-1% stability',
    })
    summary = {
        'records': len(records), 'windows': len(manifest_rows),
        'by_split': {split: sum(row['split'] == split for row in manifest_rows)
                     for split in ('train', 'val', 'test')},
        'by_class': {name: sum(row['label'] == name for row in manifest_rows)
                     for name in CLASS_ORDER},
        'rpm_missing': sum(record['rpm_imputed'] for record in records),
    }
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
