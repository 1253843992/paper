import unittest
import numpy as np
from echo_cil.memory import ReplayBuffer
from echo_cil.physics import WINDOW


def samples(label, count):
    return [dict(x=np.zeros(WINDOW, dtype=np.float32), y=label, rpm=1790,
                 record_id=label * 4 + i % 4, start=i * 512,
                 sample_id=label * 10000 + i) for i in range(count)]


class ReplayTests(unittest.TestCase):
    def test_three_stages(self):
        memory = ReplayBuffer(32 * 1024 * 1024, max_fraction=0.1)
        counts = [234, 273, 272, 272, 272, 272, 272, 272, 272, 272]
        active = []
        for group, expected in zip(([0, 1, 2, 3], [4, 5, 6], [7, 8, 9]),
                                   (105, 186, 268)):
            previous = {item['sample_id'] for item in memory.all()}
            new = [item for label in group for item in samples(label, counts[label])]
            active.extend(group)
            memory.update(new, active)
            self.assertEqual(memory.sample_limit, expected)
            self.assertEqual(len(memory.all()), expected)
            self.assertLessEqual(memory.bytes_used, memory.byte_budget)
            self.assertLessEqual(len(memory.all()) / memory.seen_training_count, 0.1)
            self.assertTrue(all(item['y'] in group or item['sample_id'] in previous
                                for item in memory.all()))
        with self.assertRaises(ValueError):
            memory.update(samples(0, 234), active)

    def test_fraction_bytes_and_duplicates(self):
        for fraction in (0, -0.1, 0.1001, float('nan')):
            with self.assertRaises(ValueError):
                ReplayBuffer(max_fraction=fraction)
        memory = ReplayBuffer(100000, max_fraction=0.05)
        candidates = samples(0, 400)
        memory.update(candidates + candidates, [0])
        self.assertEqual(memory.seen_training_count, 400)
        self.assertEqual(memory.sample_limit, 20)
        self.assertLess(len(memory.all()), 20)
        self.assertLessEqual(memory.bytes_used, 100000)
        with self.assertRaises(ValueError):
            ReplayBuffer(100).update(samples(0, 100), [0])


if __name__ == '__main__':
    unittest.main()
