"""Single-factor basic ablations; apply before validation and config export."""
ABLATIONS = ('full', 'no_evidence_loss', 'no_spectrum', 'no_spectral_gate',
             'no_spectral_dropout', 'no_projection', 'no_kd',
             'no_metric_losses', 'no_consolidation', 'no_prototype',
             'no_evidence_fusion', 'neural_only')


def apply_ablation(args):
    changes = {
        'no_evidence_loss': {'lambda_evidence': 0.0},
        'no_spectral_dropout': {'spectral_dropout': 0.0},
        'no_kd': {'lambda_kd': 0.0},
        'no_metric_losses': {'lambda_supcon': 0.0, 'lambda_metric': 0.0},
        'no_consolidation': {'consolidation_epochs': 0},
        'no_prototype': {'prototype_weights': [0.0]},
        'no_evidence_fusion': {'evidence_weights': [0.0]},
        'neural_only': {'prototype_weights': [0.0], 'evidence_weights': [0.0]},
    }.get(args.ablation, {})
    for name, value in changes.items():
        setattr(args, name, value)
    args.spectrum_mode = {'no_spectrum': 'off', 'no_spectral_gate': 'constant'}.get(args.ablation, 'learned')
    args.projection_enabled = args.ablation != 'no_projection'


def configure_model(model, args):
    model.spectrum_mode = args.spectrum_mode
    model.lambda_metric = args.lambda_metric
