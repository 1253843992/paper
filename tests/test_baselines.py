import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from echo_cil.baselines import (METHODS, CILModel, objective, pod_loss,
                                effective_weights, restore_model)
from train_cil_baselines import (quotas_for, select_herding, split_bic,
                                 targets_for, replay_size, update_memory)
from run_cil_comparison import completed


def batch(device='cpu'):
    return {'x': torch.randn(8, 2048, device=device),
            'order': torch.randn(8, 3, 256, device=device),
            'quality': torch.randn(8, 9, device=device),
            'rpm': torch.full((8,), 1790., device=device)}


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_methods_losses_expansion_and_checkpoint(self):
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(2)
                model = CILModel(method, [0, 4, 9, 2], 'cnn1d')
                teacher = model.frozen_copy()
                old_weights = model.head.weight.detach().clone()
                model.expand([6, 1, 8])
                data = batch()
                output = model(data)
                output['cosine'] = output['logits'] / model.head.log_scale.exp() if method == 'lucir' else output['logits']
                loss = objective(method, output, torch.tensor([0, 1, 2, 3, 4, 5, 6, 0]), 4,
                                 teacher(data), class_weights=effective_weights([3]*4+[30]*3, .96, 'cpu'))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))
                if method in ('der_expand', 'foster'):
                    self.assertEqual(len(model.encoders), 2)
                    self.assertTrue(all(p.grad is None for p in model.encoders[0].parameters()))
                    self.assertEqual(output['features'].shape, (8, 256))
                metadata = dict(method=method, classes=model.classes, backbone='cnn1d', proxies=10,
                                encoder_count=len(model.encoders), model=model.state_dict(),
                                aux_outputs=model.aux.out_features if model.aux is not None else None)
                restored = restore_model(metadata)
                model.eval()
                torch.testing.assert_close(model(data)['logits'], restored(data)['logits'])
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_echo_encoder_forward_backward(self):
        model = CILModel('er', [0, 1], 'echo')
        output = model(batch())
        F.cross_entropy(output['logits'], torch.arange(8) % 2).backward()
        self.assertEqual(len(output['maps']), 3)
        self.assertIsNotNone(model.encoders[0].evidence.query_projection.weight.grad)

    def test_wa_and_pod(self):
        model = CILModel('wa', [0, 1], 'cnn1d')
        model.expand([2])
        with torch.no_grad():
            model.head.weight[:2].fill_(1)
            model.head.weight[2:].fill_(4)
        self.assertAlmostEqual(model.align(2), .25, places=6)
        torch.testing.assert_close(model.head.weight[0], model.head.weight[2])
        maps = [torch.randn(3, 8, 20), torch.randn(3, 16, 10)]
        self.assertEqual(float(pod_loss(maps, maps)), 0.)
        self.assertGreater(float(pod_loss(maps, [torch.randn_like(x) for x in maps])), 0.)

    def test_mapping_and_herding(self):
        torch.testing.assert_close(targets_for(torch.tensor([9, 0, 4]), [0, 4, 9]), torch.tensor([2, 0, 1]))
        values = F.normalize(torch.randn(20, 8), dim=1)
        chosen = select_herding(values, 5)
        self.assertEqual(len(set(chosen)), 5)
        self.assertEqual(chosen, select_herding(values, 5))

    def test_memory_bic_trajectory_and_caps(self):
        args = Namespace(method='derpp', replay_fraction=.1, memory_bytes=32*1024*1024, seed=0)
        memory, retained_ids, seen = [], set(), 0
        for stage, classes in enumerate(([0, 1, 2, 3], [4, 5, 6], [7, 8, 9])):
            count = 100
            new = [dict(x=np.zeros(2048, np.float32), y=c, rpm=1790., record_id=c,
                        start=i*1024, sample_id=c*1000+i) for c in classes for i in range(count)]
            active = list(range(max(classes)+1))
            trajectory = {x['sample_id']: np.full(len(active), stage+.25, np.float32) for x in new}
            seen += len(new)
            model = Namespace(classes=active)
            updated = update_memory(model, memory, new, seen, trajectory, args, {})
            self.assertLessEqual(len(updated), int(seen*.1))
            self.assertLessEqual(replay_size(updated), args.memory_bytes)
            for item in updated:
                if item['y'] < min(classes):
                    self.assertIn(item['sample_id'], retained_ids)
                    prior = next(x for x in memory if x['sample_id'] == item['sample_id'])
                    np.testing.assert_array_equal(item['dark_logits'], prior['dark_logits'])
                else:
                    self.assertEqual(item['dark_mask'].sum(), len(active))
                    np.testing.assert_array_equal(item['dark_logits'][:len(active)], trajectory[item['sample_id']])
            memory = updated
            retained_ids = {x['sample_id'] for x in memory}
        train, held = split_bic(memory, active, .2, 0)
        self.assertFalse({x['sample_id'] for x in train} & {x['sample_id'] for x in held})
        self.assertEqual(len(held), 20)

    def test_partial_results_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'results.json'
            path.write_text('{"stages": [{"test": {"macro": 1.0}}]}')
            self.assertFalse(completed(path))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_amp(self):
        model = CILModel('podnet', [0, 1], 'echo').cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            output = model(batch('cuda'))
            loss = objective('podnet', output, torch.arange(8, device='cuda') % 2, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
