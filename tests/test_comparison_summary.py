import csv
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from run_cil_comparison import summarize, run_training


class SummaryTests(unittest.TestCase):
    def test_training_automatically_summarizes_each_stage(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.stdout = iter(['training log\n', '{"stage": 0, "epoch": 1}\n',
                               '{"stage": 0, "test_macro": 0.9}\n',
                               '{"stage": 1, "test_macro": 0.8}\n'])
        process.wait.return_value = 0
        with patch('run_cil_comparison.subprocess.Popen', return_value=process), \
             patch('run_cil_comparison.summarize') as summary:
            run_training(['python', 'trainer.py'], Path('runs/example'))
        self.assertEqual(summary.call_count, 2)
        summary.assert_called_with(Path('runs/example'))

    def test_stages_partial_runs_and_seed_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, count in [(0, 3), (1, 2)]:
                path = root/'er'/'O0'/f'seed_{seed}'/'results.json'
                path.parent.mkdir(parents=True)
                stages = [{'stage': s, 'test': {'macro': .8 + seed*.1,
                           'overall': .75, 'old': 0 if s == 0 else .7, 'new': .9},
                           'memory_count': 100+s} for s in range(count)]
                path.write_text(json.dumps({'stages': stages,
                                            'status': 'complete' if count == 3 else 'running'}))
            summarize(root)
            with (root/'comparison_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]['test_old'], '')
            self.assertEqual(rows[0]['test_macro_pct'], '80.0')
            self.assertEqual(rows[-1]['run_complete'], 'False')
            self.assertEqual(rows[2]['stage'], '2')
            aggregate = json.loads((root/'comparison_summary.json').read_text())
            self.assertEqual(len(aggregate), 3)
            self.assertAlmostEqual(aggregate[0]['test_macro_mean'], .85)
            self.assertIsNone(aggregate[0]['test_old_mean'])
            self.assertEqual(aggregate[2]['runs'], 1)
            self.assertEqual(aggregate[1]['completed_runs'], 1)


if __name__ == '__main__':
    unittest.main()
