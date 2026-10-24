import argparse
import unittest
from unittest.mock import patch

import torch
from echo_cil.model import ECHOModel
from echo_cil.ablations import apply_ablation, configure_model


class AblationTests(unittest.TestCase):
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
