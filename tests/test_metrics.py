import unittest
from unittest.mock import patch

import numpy as np
import torch

from echo_cil.metrics import classification_metrics, forgetting_metrics, benchmark_inference


class MetricsTests(unittest.TestCase):
    def test_f1_and_directional_rates(self):
        cm = [[8, 1, 1, 0], [0, 7, 2, 1], [1, 0, 8, 1], [2, 1, 0, 7]]
        values = classification_metrics(cm, 2)
        self.assertAlmostEqual(values['macro_f1'], (16/21+14/19+16/21+14/19)/4)
        self.assertAlmostEqual(values['old_to_new_rate'], .2)
        self.assertAlmostEqual(values['new_to_old_rate'], .2)
        self.assertAlmostEqual(values['old_new_harmonic'], .75)
        base = classification_metrics(cm, 0)
        self.assertIsNone(base['old_to_new_rate'])
        self.assertIsNone(base['old_new_harmonic'])
        missing = classification_metrics([[10, 0], [10, 0]], 1)
        self.assertAlmostEqual(missing['macro_f1'], 1/3)
        self.assertEqual(missing['new_to_old_rate'], 1.)

    def test_forgetting_and_positive_transfer(self):
        stages = [{'test': {'task_macro': [.9]}}, {'test': {'task_macro': [.8, .85]}},
                  {'test': {'task_macro': [.7, .9, .95]}}]
        values = forgetting_metrics(stages)
        self.assertAlmostEqual(values['forgetting'], .075)
        self.assertAlmostEqual(values['backward_transfer'], -.075)
        self.assertIsNone(forgetting_metrics(stages[:1])['forgetting'])

    def test_latency_units_and_warmup(self):
        calls = []
        with patch('echo_cil.metrics.time.perf_counter', side_effect=[0, .01, 1, 1.03]):
            values = benchmark_inference(lambda: calls.append(1), 4, torch.device('cpu'), warmup=3, repeats=2)
        self.assertEqual(len(calls), 5)
        self.assertAlmostEqual(values['inference_batch_ms'], 20.)
        self.assertAlmostEqual(values['inference_ms_per_sample'], 5.)
        self.assertAlmostEqual(values['inference_samples_per_second'], 200.)


if __name__ == '__main__':
    unittest.main()
