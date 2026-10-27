import argparse
import unittest
from unittest.mock import patch

import torch
from echo_cil.model import ECHOModel
from echo_cil.ablations import apply_ablation, configure_model


class AblationTests(unittest.TestCase):
    def test_core_paths_and_random_basis(self):
        from echo_cil.ablations import random_same_rank_basis, class_mean_prototypes
        torch.set_num_threads(2)
        model = ECHOModel([0, 1]).eval()
        batch = (torch.randn(2, 2048), torch.randn(2, 3, 256), torch.randn(2, 9), torch.full((2,), 1790.))
        args = argparse.Namespace(ablation='no_physical_guidance', lambda_metric=.05)
        apply_ablation(args)
        configure_model(model, args)
        with patch.object(model.evidence, 'forward', side_effect=AssertionError('physical path executed')), \
             patch.object(model.quality_gate, 'forward', side_effect=AssertionError('quality path executed')):
            first = model(*batch)[0]
            second = model(batch[0], batch[1]*3, batch[2]*2, batch[3]+200)[0]
            self.assertTrue(torch.allclose(first, second))
        args.ablation = 'mean_fusion'
        apply_ablation(args)
        configure_model(model, args)
        with patch.object(model.fusion.attention, 'forward', side_effect=AssertionError('attention executed')):
            model(*batch)[0].sum().backward()
        rng = torch.get_rng_state().clone()
        basis = random_same_rank_basis(torch.eye(128)[:, :3], 123)
        self.assertTrue(torch.allclose(basis.T @ basis, torch.eye(3), atol=1e-6))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(basis, random_same_rank_basis(torch.eye(128)[:, :3], 123)))
        vectors, labels, rpms = class_mean_prototypes(torch.tensor([[1., 0.], [0., 1.], [1., 0.]]),
                                                     torch.tensor([0, 0, 1]), torch.tensor([100., 200., 300.]))
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(rpms.tolist(), [150., 300.])
        self.assertTrue(torch.allclose(vectors.norm(dim=1), torch.ones(2)))

    def test_evaluation_does_not_advance_training_rng(self):
        import numpy as np
        from train_echocil import predict_global
        torch.set_num_threads(2)
        model = ECHOModel([0, 1]).eval()
        dataset = [dict(x=np.zeros(2048), order=np.zeros((3, 256)), quality=np.zeros(9),
                        y=0, rpm=1790., record_id=0, sample_id=0)]
        rng = torch.get_rng_state().clone()
        for _ in range(3):
            predict_global(model, dataset, torch.device('cpu'), 1, 0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_projection_executes_and_empty_basis_is_identity(self):
        from echo_cil.model import ProtectedExpert
        basis = torch.eye(128)[:, :2]
        expert = ProtectedExpert(basis, 1, [2])
        torch.nn.init.normal_(expert.up.weight)
        h, gamma = torch.randn(3, 128), torch.ones(3, 1)
        protected, gate, raw = expert(h, gamma)
        self.assertTrue(torch.allclose(protected[:, :2], torch.zeros(3, 2), atol=1e-6))
        expert.basis = basis[:, :0]
        unprotected, gate, raw = expert(h, gamma)
        self.assertTrue(torch.allclose(unprotected, gate * raw))
        self.assertFalse(torch.allclose(protected, unprotected))

    def test_inference_views_restore_state_and_rng(self):
        from echo_cil.ablation_audit import inference_views
        torch.set_num_threads(2)
        model = ECHOModel([0, 1]).train()
        model.prototype_weight.fill_(.75)
        model.evidence_weight.fill_(.25)
        rng = torch.get_rng_state().clone()
        def evaluate():
            torch.rand(1)
            return [float(model.prototype_weight), float(model.evidence_weight)]
        views = inference_views(model, evaluate)
        self.assertEqual(views['fused']['metrics'], [.75, .25])
        self.assertEqual(views['neural_only']['metrics'], [0., 0.])
        self.assertEqual(views['without_prototype']['metrics'], [0., .25])
        self.assertEqual(views['without_evidence']['metrics'], [.75, 0.])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        with self.assertRaises(RuntimeError):
            inference_views(model, lambda: (_ for _ in ()).throw(RuntimeError()))
        self.assertEqual(float(model.prototype_weight), .75)
        self.assertEqual(float(model.evidence_weight), .25)

    def test_fusion_presets(self):
        for name, p, e in [('no_prototype', [0.], [.5]),
                           ('no_evidence_fusion', [1.], [0.]),
                           ('neural_only', [0.], [0.])]:
            args = argparse.Namespace(ablation=name, prototype_weights=[1.], evidence_weights=[.5])
            apply_ablation(args)
            self.assertEqual(args.prototype_weights, p)
            self.assertEqual(args.evidence_weights, e)

    def test_presets_and_no_consolidation_override(self):
        for name, field in [('no_evidence_loss', 'lambda_evidence'),
                            ('no_spectral_dropout', 'spectral_dropout'), ('no_kd', 'lambda_kd')]:
            args = argparse.Namespace(ablation=name)
            setattr(args, field, 1.)
            apply_ablation(args)
            self.assertEqual(getattr(args, field), 0.)
        args = argparse.Namespace(ablation='no_metric_losses', lambda_supcon=.1, lambda_metric=.05)
        apply_ablation(args)
        self.assertEqual((args.lambda_supcon, args.lambda_metric), (0, 0))
        args = argparse.Namespace(ablation='no_consolidation', consolidation_epochs=1)
        apply_ablation(args)
        self.assertEqual(args.consolidation_epochs, 0)
        args = argparse.Namespace(ablation='no_projection')
        apply_ablation(args)
        self.assertFalse(args.projection_enabled)

    def test_spectrum_is_skipped_and_gate_is_constant(self):
        torch.set_num_threads(2)
        model = ECHOModel([0, 1]).eval()
        batch = (torch.randn(2, 2048), torch.randn(2, 3, 256), torch.randn(2, 9), torch.full((2,), 1790.))
        args = argparse.Namespace(ablation='no_spectrum', lambda_metric=.05)
        apply_ablation(args)
        configure_model(model, args)
        with patch.object(model.spectrum, 'forward', side_effect=AssertionError('Spectrum must not execute')):
            features, _, diag = model.base_features(*batch)
        self.assertEqual(features.shape, (2, 128))
        self.assertTrue((diag['spectral_gate'] == 0).all())
        args.ablation = 'no_spectral_gate'
        apply_ablation(args)
        configure_model(model, args)
        with patch.object(model.spectral_gate, 'forward', side_effect=AssertionError('Gate must not execute')):
            _, _, diag = model.base_features(*batch)
        self.assertTrue((diag['spectral_gate'] == 1).all())
        self.assertEqual(model.architecture_metadata()['spectrum_mode'], 'constant')

    def test_metric_weight_reaches_training_loss(self):
        from train_echocil import loss_terms
        model = ECHOModel([0, 1])
        model.lambda_evidence = 0
        labels = torch.tensor([0, 1])
        logits = torch.zeros(2, 2)
        data = {'y': labels}
        diagnostics = {'evidence_logits': logits}
        with patch('train_echocil.hard_metric_loss', return_value=torch.tensor(2.)), \
             patch('train_echocil.supervised_contrastive_loss', return_value=torch.tensor(0.)):
            model.lambda_metric = .05
            full, _ = loss_terms(model, logits, data, torch.randn(2, 128), diagnostics,
                                 [], [0, 1], .5, 0, 0, base_stage=True)
            model.lambda_metric = 0
            removed, _ = loss_terms(model, logits, data, torch.randn(2, 128), diagnostics,
                                    [], [0, 1], .5, 0, 0, base_stage=True)
        self.assertAlmostEqual(float(full-removed), .1, places=6)


if __name__ == '__main__':
    unittest.main()
