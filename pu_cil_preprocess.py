"""PU 15-state protocol. Split whole records before resampling/windowing.

No fitting to validation/test data. Does not modify raw files or CWRU outputs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from collections import Counter
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from echo_cil.physics import FS_TARGET, WINDOW, raw_physics_features, apply_physics_stats

CLASSES = ['K001', 'KA01', 'KA03', 'KI01', 'KI03', 'KA04', 'KI04',
           'KA05', 'KI05', 'KA06', 'KI07', 'KA07', 'KI08', 'KA08', 'KI18']
TASKS = [list(range(5)), [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
CONDITIONS = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10']
# Approximate 6203 nominal anchors, NOT measured specimen-specific geometry.
# NTN 6203 technical sheet: https://docs.rs-online.com/a404/A700000014892791.pdf
BASE_ORDERS = [1., 2., .382, 1.994, 3.987, 3.053, 4.947]
HARMONICS = [b * h for b in BASE_ORDERS for h in (1, 2, 3)]
PATTERN = re.compile(r'^(N\d+_M\d+_F\d+)_(K[AIB]?\d+)_(\d+)\.mat$', re.I)

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def inventory(root, seed):
    groups = {}
    for path in sorted(root.rglob('*.mat')):
        m = PATTERN.fullmatch(path.name)
        if not m or m[2] not in CLASSES or m[1] not in CONDITIONS:
            continue
        groups.setdefault((m[2], m[1]), []).append((int(m[3]), path))
    rows = []
    for label in CLASSES:
        for condition in CONDITIONS:
            entries = sorted(groups.get((label, condition), []))
            if [i for i, _ in entries] != list(range(1, 21)):
                raise ValueError(f'{label}/{condition}: expected unique records 1..20')
            # Stable independently of traversal order and training seed.
            key = hashlib.sha256(f'{seed}/{label}/{condition}'.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
            shuffled = rng.permutation(20)
            splits = {int(i): ('train' if j < 12 else 'val' if j < 16 else 'test')
                      for j, i in enumerate(shuffled)}
            for i, (record_number, path) in enumerate(entries):
                rows.append(dict(record_id=len(rows), source=path.relative_to(root).as_posix(),
                                 class_id=CLASSES.index(label), label=label, load=condition,
                                 record_number=record_number, split=splits[i]))
    return rows

def channel(record, name):
    ys = record['Y'] if isinstance(record['Y'], list) else [record['Y']]
    matches = [y for y in ys if y['Name'] == name]
    if len(matches) != 1:
        raise ValueError(f'expected one {name} channel')
    y = matches[0]
    xs = record['X'] if isinstance(record['X'], list) else [record['X']]
    t = np.asarray(xs[int(y['XIndex']) - 1]['Data'], dtype=np.float64).reshape(-1)
    v = np.asarray(y['Data'], dtype=np.float64).reshape(-1)
    if len(t) != len(v) or len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(v).all():
        raise ValueError(f'invalid {name} arrays')
    if (np.diff(t) <= 0).any():
        raise ValueError(f'non-increasing {name} timestamps')
    return t, v

def read_record(path):
    record = loadmat(path, simplify_cells=True)[path.stem]
    t, x = channel(record, 'vibration_1')
    ts, speed = channel(record, 'speed')
    # Interpolate jittered acquisition timestamps onto an approximately native
    # uniform grid; polyphase low-pass filtering precedes downsampling to 12k.
    fs = int(round((len(t) - 1) / (t[-1] - t[0])))
    if not 40000 <= fs <= 100000 or np.max(np.diff(t)) > .05:
        raise ValueError(f'unexpected sampling rate/gap in {path}: {fs}')
    uniform_t = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * fs)) + 1) / fs
    uniform = np.interp(uniform_t, t, x)
    gcd = int(np.gcd(fs, FS_TARGET))
    signal = resample_poly(uniform, FS_TARGET // gcd, fs // gcd)
    if (speed <= 0).any() or speed.max() > 3000:
        raise ValueError(f'invalid RPM channel: {path}')
    gaps = np.flatnonzero(np.diff(t) > 4 / fs)
    gap_intervals = [(float(t[j]-t[0]), float(t[j+1]-t[0])) for j in gaps]
    return signal, ts, speed, dict(fs_source=fs, n_samples=len(x),
        gap_intervals=gap_intervals,
        time_origin=float(t[0]), duration=float(t[-1] - t[0]),
        sample_interval_min=float(np.diff(t).min()), sample_interval_max=float(np.diff(t).max()))

def window_starts(length, count, gaps=(), allow_fewer=False):
    # Reserve one full window at each end against resampling boundary effects.
    available = (length - 2 * WINDOW) // WINDOW
    candidates = WINDOW + np.arange(max(0, available)) * WINDOW
    # 10 ms on either side also excludes interpolation/filter transients.
    for lo, hi in gaps:
        candidates = candidates[((candidates + WINDOW) / FS_TARGET <= lo - .01) |
                                (candidates / FS_TARGET >= hi + .01)]
    if allow_fewer and len(candidates) >= min(10, count):
        count = min(count, len(candidates))
    if len(candidates) < count:
        raise ValueError(f'record supports only {len(candidates)} clean windows, requested {count}')
    return candidates[np.floor(np.linspace(0, len(candidates) - 1, count)).astype(np.int64)]

def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('pu_cil_processed'))
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--windows-per-record', type=int, default=20)
    parser.add_argument('--base-classes', type=int, nargs='+', default=TASKS[0])
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--skip-invalid-records', action='store_true',
                        help='explicitly exclude unreadable/invalid records and log them; never reassign splits')
    args = parser.parse_args()
    if not 1 <= args.windows_per_record <= 20:
        parser.error('--windows-per-record must be in 1..20')
    if len(set(args.base_classes)) != 5 or not set(args.base_classes) <= set(range(15)):
        parser.error('--base-classes must contain five distinct IDs in 0..14')
    records = inventory(args.input, args.split_seed)
    if args.audit_only:
        print(json.dumps({'records': len(records), 'classes': CLASSES,
                          'records_by_split': dict(Counter(r['split'] for r in records))}, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output is not empty. Use a new directory; no files will be overwritten.')
    args.output.mkdir(parents=True, exist_ok=True)
    shards = args.output / 'shards'
    shards.mkdir()
    write_json(args.output / 'status.json', {'complete': False})
    # Raw selected windows are cached on disk, not all original MAT recordings.
    n = 0
    total = total2 = 0.
    seen_hashes = set()
    accepted, excluded = [], []
    for i, row in enumerate(records):
        path = args.input / row['source']
        checksum = digest(path)
        if checksum in seen_hashes:
            raise ValueError(f'duplicate source content: {path}')
        seen_hashes.add(checksum)
        try:
            signal, ts, speed, metadata = read_record(path)
            starts = window_starts(len(signal), args.windows_per_record, metadata['gap_intervals'], allow_fewer=True)
        except (ValueError, TypeError, OSError) as error:
            if not args.skip_invalid_records:
                raise ValueError(f'Invalid record {path}: {error}') from error
            excluded.append(dict(source=row['source'], split=row['split'], label=row['label'],
                                 sha256=checksum, error=f'{type(error).__name__}: {error}'))
            write_json(args.output / 'excluded_records.json', excluded)
            print(f'EXCLUDED {path.name}: {error}', flush=True)
            continue
        if len(starts) < args.windows_per_record:
            print(f'Quality exclusion: {path.name}: {len(starts)} clean windows', flush=True)
        x = np.stack([signal[s:s + WINDOW] for s in starts]).astype(np.float32)
        rpm = []
        for start in starts:
            lo = metadata['time_origin'] + start / FS_TARGET
            hi = lo + WINDOW / FS_TARGET
            mask = (ts >= lo) & (ts < hi)
            if not mask.any():
                raise ValueError(f'no measured speed for window: {path}')
            rpm.append(float(np.mean(speed[mask])))
        row.update(metadata, sha256=checksum, rpm_used=float(np.mean(rpm)),
                   channel='vibration_1', rpm_source='speed', windows=len(starts))
        accepted.append(row)
        np.savez(shards / f'r{i:04d}_raw.npz', x=x, start=starts, rpm=np.asarray(rpm, dtype=np.float32))
        if row['split'] == 'train' and row['class_id'] in args.base_classes:
            values = x.astype(np.float64)
            n += values.size
            total += values.sum()
            total2 += np.square(values).sum()
        if (i + 1) % 50 == 0:
            print(f'Read {i+1}/{len(records)} records', flush=True)
    records = accepted
    for label in CLASSES:
        for condition in CONDITIONS:
            for split, minimum in [('train', 10), ('val', 3), ('test', 3)]:
                if sum(r['label']==label and r['load']==condition and r['split']==split for r in records) < minimum:
                    raise ValueError(f'Too many excluded records: {label}/{condition}/{split}')
    mean = total / n
    std = max(float(np.sqrt(max(total2 / n - mean * mean, 0))), 1e-12)
    base_order, base_quality = [], []
    for row in records:
        i = row['record_id']
        raw = shards / f'r{i:04d}_raw.npz'
        with np.load(raw) as a:
            x = ((a['x'] - mean) / std).astype(np.float32)
            starts, rpm = a['start'], a['rpm']
        features = [raw_physics_features(v, float(r), harmonic_orders=HARMONICS) for v, r in zip(x, rpm)]
        order = np.stack([f[0] for f in features])
        quality = np.stack([f[1] for f in features])
        if not np.isfinite(order).all() or not np.isfinite(quality).all():
            raise ValueError(f'nonfinite physics features: {row["source"]}')
        if row['split'] == 'train' and row['class_id'] in args.base_classes:
            base_order.append(order)
            base_quality.append(quality)
        np.savez(raw, x=x, start=starts, rpm=rpm, order=order, quality=quality)
        if (i + 1) % 50 == 0:
            print(f'Physics {i+1}/{len(records)} records', flush=True)
    ob, qb = np.concatenate(base_order), np.concatenate(base_quality)
    stats = dict(protocol_version='PU-record-split', waveform_mean=mean, waveform_std=std,
                 mean=mean, std=std, order_mean=ob.mean(axis=(0, 2)).tolist(),
                 order_std=ob.std(axis=(0, 2)).clip(1e-6).tolist(),
                 quality_mean=qb.mean(axis=0).tolist(), quality_std=qb.std(axis=0).clip(1e-6).tolist(),
                 source_classes=args.base_classes, target_fs=FS_TARGET, harmonic_orders=HARMONICS,
                 base_orders=BASE_ORDERS, prior_note='Approximate nominal 6203, not specimen-specific')
    manifest = []
    for row in records:
        i = row['record_id']
        raw = shards / f'r{i:04d}_raw.npz'
        with np.load(raw) as a:
            arrays = {k: a[k] for k in a.files}
        arrays['order'], arrays['quality'] = apply_physics_stats(arrays['order'], arrays['quality'], stats)
        arrays['class_id'] = np.full(len(arrays['x']), row['class_id'], dtype=np.int64)
        name = f'r{i:04d}_{row["split"]}.npz'
        np.savez_compressed(shards / name, **arrays)
        raw.unlink()  # Only this run's explicitly named intermediate shard.
        for j, start in enumerate(arrays['start']):
            token = f'{row["source"]}:{int(start)}:{FS_TARGET}:{WINDOW}'
            manifest.append(dict(record_id=i, source=row['source'], shard=f'shards/{name}', index=j,
                split=row['split'], class_id=row['class_id'], label=row['label'], load=row['load'],
                rpm=float(arrays['rpm'][j]), rpm_imputed=False, start=int(start), end=int(start)+WINDOW,
                sample_id=int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), 'little'),
                fs_source=row['fs_source'], fs=FS_TARGET, window=WINDOW))
    save_csv(args.output / 'manifest.csv', manifest)
    save_csv(args.output / 'record_inventory.csv', records)
    write_json(args.output / 'normalization.json', stats)
    write_json(args.output / 'class_mapping.json', dict(class_id=dict(zip(CLASSES, range(15))),
        class_order=CLASSES, primary_order=TASKS, task_orders={'O0': TASKS},
        units_note='Bearing specimen/state labels, NOT 15 distinct fault mechanisms'))
    write_json(args.output / 'preprocess_config.json', dict(dataset='PU', split_seed=args.split_seed,
        split='whole_record_stratified_12_4_4', target_fs=FS_TARGET, window=WINDOW,
        windows_per_record=args.windows_per_record, conditions=CONDITIONS,
        start_units='target samples from record time_origin', base_classes=args.base_classes,
        resampling='timestamp interpolation to mean-rate grid, then antialiased resample_poly',
        harmonic_queries=HARMONICS, training_adapter_required=True,
        note='Trainer still needs PU task orders, model anchors and label semantics adaptation'))
    write_json(args.output / 'excluded_records.json', excluded)
    summary = dict(records=len(records), excluded_records=len(excluded), windows=len(manifest),
        by_split=dict(Counter(r['split'] for r in manifest)),
        records_by_split=dict(Counter(r['split'] for r in records)),
        by_class=dict(Counter(r['label'] for r in manifest)))
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'status.json', {'complete': True})
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
