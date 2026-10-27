import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from pu_cil_preprocess import CLASSES, CONDITIONS, inventory, window_starts, HARMONICS
from echo_cil.physics import raw_physics_features, physics_features


class TestPUPreprocess(unittest.TestCase):
    def test_record_splits_are_fixed_disjoint_and_balanced(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in CLASSES:
                for condition in CONDITIONS:
                    for i in range(1, 21):
                        (root / f'{condition}_{label}_{i}.mat').touch()
            rows = inventory(root, 2026)
            self.assertEqual(rows, inventory(root, 2026))
            self.assertEqual(len({r['source'] for r in rows}), 1200)
            for label in CLASSES:
                for condition in CONDITIONS:
                    subset = [r for r in rows if r['label'] == label and r['load'] == condition]
                    self.assertEqual([sum(r['split'] == s for r in subset) for s in ('train', 'val', 'test')], [12, 4, 4])

    def test_windows_disjoint_and_inside_record(self):
        starts = window_starts(48000, 20)
        self.assertTrue((np.diff(starts) >= 2048).all())
        self.assertGreaterEqual(starts[0], 2048)
        self.assertLessEqual(starts[-1] + 4096, 48000)
        with self.assertRaises(ValueError):
            window_starts(30000, 20)

    def test_replay_uses_same_priors(self):
        x = np.random.default_rng(1).normal(size=2048).astype(np.float32)
        stats = dict(order_mean=[0]*3, order_std=[1]*3,
                     quality_mean=[0]*9, quality_std=[1]*9, harmonic_orders=HARMONICS)
        expected = raw_physics_features(x, 1500, harmonic_orders=HARMONICS)
        actual = physics_features(x, 1500, stats)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    def test_gap_windows_are_excluded(self):
        starts = window_starts(48000, 20, [(1., 1.004)], allow_fewer=True)
        self.assertTrue((((starts+2048)/12000 <= .99) | (starts/12000 >= 1.014)).all())
        self.assertTrue((np.diff(starts) >= 2048).all())

if __name__ == '__main__':
    unittest.main()
