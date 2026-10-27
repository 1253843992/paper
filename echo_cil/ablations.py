"""Single-factor basic ablations; apply before validation and config export."""
BASIC_ABLATIONS = ('full', 'no_evidence_loss', 'no_spectrum', 'no_spectral_gate',
             'no_spectral_dropout', 'no_projection', 'no_kd',
             'no_metric_losses', 'no_consolidation', 'no_prototype',
             'no_evidence_fusion', 'neural_only')
CORE_ABLATIONS = ('full', 'no_physical_guidance', 'mean_fusion', 'no_spectrum',
                  'no_experts', 'no_projection', 'random_projection',
                  'no_competition', 'no_amplitude', 'no_consolidation',
                  'no_prototype', 'class_mean_prototype', 'no_rpm_penalty')
ABLATIONS = tuple(dict.fromkeys(BASIC_ABLATIONS + CORE_ABLATIONS))


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
        'no_physical_guidance': {'lambda_evidence': 0.0, 'evidence_weights': [0.0]},
        'no_experts': {'expert_limit': 0},
        'no_competition': {'lambda_competition': 0.0},
        'no_amplitude': {'lambda_amplitude': 0.0},
        'no_rpm_penalty': {'prototype_rpm_penalties': [0.0]},
        'class_mean_prototype': {'prototypes_per_record': 0},
    }.get(args.ablation, {})
    for name, value in changes.items():
        setattr(args, name, value)
    args.spectrum_mode = {'no_spectrum': 'off', 'no_spectral_gate': 'constant'}.get(args.ablation, 'learned')
    args.projection_enabled = args.ablation != 'no_projection'
    args.physical_guidance = args.ablation != 'no_physical_guidance'
    args.fusion_mode = 'mean' if args.ablation == 'mean_fusion' else 'attention'


def configure_model(model, args):
    model.spectrum_mode = args.spectrum_mode
    model.lambda_metric = args.lambda_metric
    model.physical_guidance = args.physical_guidance
    model.fusion.mode = args.fusion_mode


def random_same_rank_basis(basis, seed):
    import torch
    if basis.shape[1] == 0:
        return basis.clone()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    random = torch.randn(basis.shape, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(random, mode='reduced')
    return q.to(basis)


def class_mean_prototypes(vectors, labels, rpms):
    import torch
    classes = labels.unique(sorted=True)
    means = torch.stack([vectors[labels == label].mean(0) for label in classes])
    return (torch.nn.functional.normalize(means, dim=1), classes,
            torch.stack([rpms[labels == label].mean() for label in classes]))
