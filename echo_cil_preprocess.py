"""ECHO-CIL v2 preprocessing for the CWRU drive-end dataset.

The script deliberately splits each raw record into disjoint continuous blocks
before windowing.  This avoids the leakage introduced by random windows from a
single recording appearing in train and test.  It also produces physics-aware
order spectra and quality features used by the ECHO-CIL model.

Example:
    python echo_cil_preprocess.py --input cwru_data --output echo_cil_processed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt


FS_TARGET = 12_000
WINDOW = 2048
TRAIN_STRIDE = 1024
EVAL_STRIDE = 2048
GUARD = WINDOW
CLASS_ORDER = ["N", "IR007", "OR007@6", "B007", "IR014", "OR014@6", "B014", "IR021", "OR021@6", "B021"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_ORDER)}
TASK_ORDER = {0: [0, 1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}

# Numeric CWRU files in the repository's 0HP directory.
LEGACY_LABEL = {97: "N", 105: "IR007", 118: "IR014", 130: "IR021",
                169: "OR007@6", 185: "OR014@6", 197: "OR021@6",
                209: "B007", 222: "B014", 234: "B021"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def infer_label(path: Path) -> str:
    name = path.name
    m = re.search(r"(?:^|_)(IR|OR|B)(007|014|021)(?:@6)?(?:_|\.)", name, re.I)
    if m:
        prefix, size = m.group(1).upper(), m.group(2)
        if prefix == "OR":
            return f"OR{size}@6"
        return f"{prefix}{size}"
    if name.lower().startswith("normal"):
        return "N"
    m = re.match(r"(\d+)\.mat$", name, re.I)
    if m and int(m.group(1)) in LEGACY_LABEL:
        return LEGACY_LABEL[int(m.group(1))]
    raise ValueError(f"Cannot infer CWRU label from {path}")


def read_record(path: Path) -> Tuple[np.ndarray, float, str]:
    data = loadmat(path, squeeze_me=True)
    de_keys = [k for k in data if not k.startswith("__") and "DE_time" in k]
    if not de_keys:
        de_keys = [k for k in data if not k.startswith("__") and "DE" in k and np.asarray(data[k]).ndim > 0]
    if not de_keys:
        raise ValueError(f"No drive-end signal in {path}")
    key = sorted(de_keys, key=lambda k: ("DE_time" not in k, len(k)))[0]
    x = np.asarray(data[key], dtype=np.float64).reshape(-1)
    rpm_keys = [k for k in data if "RPM" in k.upper()]
    rpm = float(np.asarray(data[rpm_keys[0]]).reshape(-1)[0]) if rpm_keys else float("nan")
    if not np.isfinite(rpm) or rpm <= 0:
        rpm = float("nan")
    return x, rpm, key


def split_blocks(n: int) -> Dict[str, Tuple[int, int]]:
    b1, b2 = int(n * 0.60), int(n * 0.80)
    # [start, end) intervals.  Guard samples are excluded around both cuts.
    return {"train": (0, max(0, b1 - GUARD)),
            "val": (min(n, b1 + GUARD), max(0, b2 - GUARD)),
            "test": (min(n, b2 + GUARD), n)}


def starts_for(lo: int, hi: int, stride: int) -> np.ndarray:
    if hi - lo < WINDOW:
        return np.empty((0,), dtype=np.int64)
    return np.arange(lo, hi - WINDOW + 1, stride, dtype=np.int64)


def standard_moments(x: np.ndarray) -> Tuple[float, float, float, float, float, float, float, float, float]:
    x = x.astype(np.float64, copy=False)
    mu = float(np.mean(x)); sd = float(np.std(x) + 1e-12)
    z = (x - mu) / sd
    m3 = float(np.mean(z ** 3)); m4 = float(np.mean(z ** 4))
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    p2p = float(np.ptp(x))
    shape = rms / (float(np.mean(np.abs(x))) + 1e-12)
    impulse = peak / (float(np.mean(np.abs(x))) + 1e-12)
    margin = peak / (float(np.mean(np.sqrt(np.abs(x))) ** 2) + 1e-12)
    return rms, sd, m4 - 3.0, m3, peak / (rms + 1e-12), p2p, shape, impulse, margin


def order_features(x: np.ndarray, rpm: float, fs: int = FS_TARGET) -> Tuple[np.ndarray, np.ndarray]:
    """Return [3, 256] envelope order spectra and nine quality features."""
    bands = [(500, 1500), (1500, 3000), (3000, 5500)]
    qgrid = np.linspace(0.0, 20.0, 256, dtype=np.float32)
    spectra = []
    if not np.isfinite(rpm) or rpm <= 0:
        rpm = 1797.0
    shaft = rpm / 60.0
    for low, high in bands:
        sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
        band = sosfiltfilt(sos, x).astype(np.float64, copy=False)
        env = np.abs(hilbert(band)); env -= np.mean(env)
        amp = np.abs(np.fft.rfft(env * np.hanning(len(env))))
        freq = np.fft.rfftfreq(len(env), 1.0 / fs)
        q = freq / shaft
        spectra.append(np.interp(qgrid, q, amp, left=0.0, right=0.0).astype(np.float32))
    return np.stack(spectra), np.asarray(standard_moments(x), dtype=np.float32)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    inp, out = args.input, args.output
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / "shards"; shard_dir.mkdir(exist_ok=True)

    records = []
    for path in sorted(inp.rglob("*.mat")):
        x, rpm, de_key = read_record(path)
        label = infer_label(path)
        records.append({"path": path, "x": x, "rpm": rpm, "de_key": de_key,
                        "label": label, "class_id": CLASS_ID[label],
                        "load": path.parent.name})
    if len(records) != 40:
        print(f"WARNING: discovered {len(records)} MAT records; expected 40 for this repository")

    # A few normal CWRU files omit the RPM variable.  Use the median RPM of
    # the same load for the physics branch, while retaining an explicit flag
    # so that the imputation is auditable rather than silently fabricated.
    load_rpms: Dict[str, List[float]] = {}
    for r in records:
        if np.isfinite(r["rpm"]):
            load_rpms.setdefault(r["load"], []).append(r["rpm"])
    for r in records:
        r["rpm_raw"] = r["rpm"]
        r["rpm_imputed"] = not np.isfinite(r["rpm"])
        if r["rpm_imputed"]:
            candidates = load_rpms.get(r["load"], [])
            r["rpm"] = float(np.median(candidates)) if candidates else 1797.0

    # Fixed normalization is estimated only from base-task (0..3) training blocks.
    base_train = [r for r in records if r["class_id"] in TASK_ORDER[0]]
    train_parts = []
    for r in base_train:
        lo, hi = split_blocks(len(r["x"]))["train"]
        train_parts.append(r["x"][lo:hi])
    norm_mean = float(np.mean(np.concatenate(train_parts)))
    norm_std = float(np.std(np.concatenate(train_parts)) + 1e-12)

    manifest_rows: List[dict] = []; inventory_rows: List[dict] = []
    for r_idx, r in enumerate(records):
        raw = r["x"]
        fs_src = 12000  # CWRU files in this repository are 12-kHz DE recordings.
        blocks = split_blocks(len(raw))
        inventory_rows.append({"record_id": r_idx, "source": str(r["path"].relative_to(inp)),
                               "sha256": sha256(r["path"]), "label": r["label"],
                               "class_id": r["class_id"], "load": r["load"],
                               "de_key": r["de_key"], "n_samples": len(raw),
                               "fs_source": fs_src, "rpm_raw": r["rpm_raw"],
                               "rpm_used": r["rpm"], "rpm_imputed": r["rpm_imputed"],
                               "train_block": blocks["train"], "val_block": blocks["val"],
                               "test_block": blocks["test"]})
        for split, (lo, hi) in blocks.items():
            stride = TRAIN_STRIDE if split == "train" else EVAL_STRIDE
            st = starts_for(lo, hi, stride)
            if len(st) == 0:
                continue
            xs, phys, qual = [], [], []
            for start in st:
                segment = raw[int(start):int(start) + WINDOW]
                if fs_src != FS_TARGET:
                    segment = resample_poly(segment, FS_TARGET, fs_src)
                    segment = segment[:WINDOW]
                xz = ((segment - norm_mean) / norm_std).astype(np.float32)
                p, q = order_features(xz, r["rpm"], FS_TARGET)
                xs.append(xz); phys.append(p); qual.append(q)
            shard_name = f"r{r_idx:03d}_{split}.npz"
            np.savez_compressed(shard_dir / shard_name,
                                x=np.stack(xs), order=np.stack(phys), quality=np.stack(qual),
                                start=st, rpm=np.full(len(st), r["rpm"], dtype=np.float32),
                                class_id=np.full(len(st), r["class_id"], dtype=np.int64))
            for j, start in enumerate(st):
                manifest_rows.append({"record_id": r_idx, "source": str(r["path"].relative_to(inp)),
                                      "shard": f"shards/{shard_name}", "index": j,
                                      "split": split, "task": next(t for t, cs in TASK_ORDER.items() if r["class_id"] in cs),
                                      "class_id": r["class_id"], "label": r["label"],
                                      "load": r["load"], "rpm": r["rpm"],
                                      "rpm_imputed": r["rpm_imputed"],
                                      "start": int(start), "end": int(start) + WINDOW,
                                      "fs": FS_TARGET, "window": WINDOW})

    for name, rows in [("manifest.csv", manifest_rows), ("record_inventory.csv", inventory_rows)]:
        with (out / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    write_json(out / "class_mapping.json", {"class_id": CLASS_ID, "task_order": TASK_ORDER,
              "class_order": CLASS_ORDER, "units_note": "007/014/021 are inch fault sizes"})
    write_json(out / "normalization.json", {"protocol": "base_train_fixed", "mean": norm_mean,
              "std": norm_std, "source_classes": TASK_ORDER[0], "target_fs": FS_TARGET})
    write_json(out / "preprocess_config.json", {"protocol": "blocked_60_20_20_with_guards",
              "window": WINDOW, "train_stride": TRAIN_STRIDE, "eval_stride": EVAL_STRIDE,
              "guard": GUARD, "bands_hz": [[500,1500],[1500,3000],[3000,5500]],
              "order_grid": [0,20,256], "features": {"order": "3 bands x 256", "quality": 9}})
    summary = {"records": len(records), "windows": len(manifest_rows),
               "by_split": {s: sum(x["split"] == s for x in manifest_rows) for s in ["train", "val", "test"]},
               "by_class": {c: sum(x["label"] == c for x in manifest_rows) for c in CLASS_ORDER},
               "rpm_missing": sum(r["rpm_imputed"] for r in records),
               "rpm_imputation": "same-load median; see record_inventory.csv"}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
