import io
import unittest

import torch

from echo_cil.model import ECHOModel
from train_echocil import loss_terms


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        self.inputs = (torch.randn(6, 2048), torch.randn(6, 3, 256),
                       torch.randn(6, 9), torch.full((6,), 1790.0))

    def test_incremental_gradients(self):
        model = ECHOModel([0, 2])
        for stage, classes in enumerate(([0, 2], [4, 7], [8, 9])):
            if stage:
                model.begin_increment(classes, stage)
                self.assertFalse(any(p.requires_grad for p in model.evidence.parameters()))
                model.enable_incremental_finetuning()
                model.add_expert(torch.empty(128, 0), classes)
            self.assertEqual(model.class_ids, model.evidence_classifier.class_ids)
            model.train()
            model.zero_grad(set_to_none=True)
            logits, features, diag = model(*self.inputs)
            labels = torch.tensor((model.class_ids * 6)[:6])
            loss, parts = loss_terms(model, logits, {'y': labels}, features,
                                    diag, [], model.class_ids, 0.5, 0, 0,
                                    base_stage=True)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(parts['evidence_ce'], 0)
            loss.backward()
            for module in (model.evidence, model.evidence_classifier, model.spectral_gate):
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
            torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4).step()

    def test_probability_mixture_and_roundtrip(self):
        model = ECHOModel([0, 2]).eval()
        with torch.no_grad():
            original, z, diag = model(*self.inputs)
            model.set_prototypes(z, torch.tensor([0, 2, 0, 2, 0, 2]), self.inputs[3], 0.25)
            baseline, _, diag = model(*self.inputs)
            model.evidence_weight.fill_(0.25)
            mixed, _, _ = model(*self.inputs)
            expected = 0.75 * baseline.softmax(1) + 0.25 * diag['evidence_logits'].softmax(1)
            torch.testing.assert_close(mixed.exp(), expected)
            torch.testing.assert_close(model(*self.inputs)[0], mixed)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = ECHOModel([0, 2]).eval()
            restored.set_prototypes(z, model.prototype_labels, self.inputs[3])
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            torch.testing.assert_close(restored(*self.inputs)[0], mixed)
            restored.begin_increment([7], 1)
            self.assertEqual(float(restored.evidence_weight), 0)
            self.assertEqual(restored.prototype_vectors.numel(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally')
    def test_cuda_amp(self):
        model = ECHOModel([0, 2]).cuda().train()
        with torch.autocast('cuda', dtype=torch.float16):
            logits, _, diag = model(*(x.cuda() for x in self.inputs))
            loss = logits.square().mean() + diag['evidence_logits'].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
