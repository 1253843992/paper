"""Physics preprocessing specified by the ECHO-CIL protocol."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


FS_TARGET = 12_000
WINDOW = 2048
ORDER_MAX = 20.0
ORDER_POINTS = 256
BANDS_HZ = ((500, 1500), (1500, 3000), (3000, 5500))
REFLECT_PAD = 256

# CWRU 6205-2RS JEM SKF anchors.  Keep BSF and 2BSF as separate types.
BASE_ORDERS = np.asarray(
    [1.0, 2.0, 0.39828, 2.3567, 4.7135, 3.5848, 5.4152],
    dtype=np.float32,
)
HARMONIC_ORDERS = np.asarray(
    [float(base * harmonic) for base in BASE_ORDERS for harmonic in (1, 2, 3)],
    dtype=np.float32,
)
HARMONIC_TYPES = np.repeat(np.arange(7, dtype=np.int64), 3)
HARMONIC_NUMBERS = np.tile(np.arange(3, dtype=np.int64), 7)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / max(denom, 1e-12))


def _map_to_order(power: np.ndarray, freq: np.ndarray, rpm: float) -> np.ndarray:
    shaft_hz = max(float(rpm) / 60.0, 1e-6)
    order = freq / shaft_hz
    grid = np.linspace(0.0, ORDER_MAX, ORDER_POINTS, dtype=np.float64)
    mapped = np.interp(grid, order, power, left=0.0, right=0.0)
    return np.log1p(mapped).astype(np.float32)


def raw_physics_features(x: np.ndarray, rpm: float,
                         fs: int = FS_TARGET, harmonic_orders=None) -> Tuple[np.ndarray, np.ndarray]:
    """Return unstandardized log-order spectra and nine quality features.

    ``x`` is one already base-statistic-normalized waveform.  Filtering and
    Hilbert analysis use reflection padding and are cropped back to the exact
    input window so no neighbouring split is ever accessed.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) != WINDOW:
        raise ValueError(f'expected {WINDOW} waveform points, got {len(x)}')
    if not np.isfinite(rpm) or rpm <= 0:
        raise ValueError('rpm must be a finite positive value')

    padded = np.pad(x, (REFLECT_PAD, REFLECT_PAD), mode='reflect')
    hann = np.hanning(WINDOW)
    freq = np.fft.rfftfreq(WINDOW, 1.0 / fs)
    spectra = []
    quality = []
    shaft_hz = float(rpm) / 60.0
    resolution_floor = max(fs / (WINDOW * shaft_hz),
                           9.0 * ORDER_MAX / (ORDER_POINTS - 1))
    grid = np.linspace(0.0, ORDER_MAX, ORDER_POINTS, dtype=np.float64)

    for low, high in BANDS_HZ:
        sos = butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
        filtered = sosfiltfilt(sos, padded)
        envelope = np.abs(hilbert(filtered))[REFLECT_PAD:-REFLECT_PAD]
        envelope = envelope - envelope.mean()
        power = np.abs(np.fft.rfft(envelope * hann)) ** 2
        power = power / (float(np.sum(hann ** 2)) + 1e-12)
        mapped = _map_to_order(power, freq, rpm)
        spectra.append(mapped)

        anchor_mask = np.zeros(ORDER_POINTS, dtype=bool)
        peaks = []
        for anchor in (HARMONIC_ORDERS if harmonic_orders is None else harmonic_orders):
            mask = np.abs(grid - float(anchor)) <= resolution_floor
            anchor_mask |= mask
            if mask.any():
                peaks.append(float(mapped[mask].max()))
        background = mapped[~anchor_mask]
        pbr = float(np.mean(peaks) / (np.median(background) + 1e-6))
        probability = mapped.astype(np.float64).clip(min=0.0)
        probability /= probability.sum() + 1e-12
        entropy = float(-(probability * np.log(probability + 1e-12)).sum()
                        / np.log(ORDER_POINTS))
        lower = _map_to_order(power, freq, rpm * 0.99)
        upper = _map_to_order(power, freq, rpm * 1.01)
        stability = 0.5 * (_cosine(mapped, lower) + _cosine(mapped, upper))
        quality.extend((np.log1p(max(pbr, 0.0)), entropy, stability))

    return np.stack(spectra).astype(np.float32), np.asarray(quality, dtype=np.float32)


def apply_physics_stats(order: np.ndarray, quality: np.ndarray,
                        stats: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    order_mean = np.asarray(stats['order_mean'], dtype=np.float32).reshape(3, 1)
    order_std = np.asarray(stats['order_std'], dtype=np.float32).reshape(3, 1)
    quality_mean = np.asarray(stats['quality_mean'], dtype=np.float32)
    quality_std = np.asarray(stats['quality_std'], dtype=np.float32)
    order = (np.asarray(order, dtype=np.float32) - order_mean) / order_std
    quality = (np.asarray(quality, dtype=np.float32) - quality_mean) / quality_std
    return order.astype(np.float32), quality.astype(np.float32)


def physics_features(x: np.ndarray, rpm: float, stats: Mapping[str, object],
                     fs: int = FS_TARGET) -> Tuple[np.ndarray, np.ndarray]:
    return apply_physics_stats(*raw_physics_features(
        x, rpm, fs, stats.get('harmonic_orders')), stats)


def load_physics_stats(processed_dir: str | Path) -> dict:
    path = Path(processed_dir) / 'normalization.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    required = ('order_mean', 'order_std', 'quality_mean', 'quality_std')
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(
            f'{path} is from the legacy preprocessor; missing {missing}. '
            'Regenerate the dataset with the strict ECHO-CIL preprocessor.')
    return data
