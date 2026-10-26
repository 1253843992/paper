"""Fixed-checkpoint inference interventions, never used for model selection."""
import csv
import json
from pathlib import Path

import torch


@torch.no_grad()
def inference_views(model, evaluate):
    """Restore weights, train/eval modes and torch RNG even if evaluation fails."""
    prototype = float(model.prototype_weight)
    evidence = float(model.evidence_weight)
    modes = [(module, module.training) for module in model.modules()]
    result = {}
    try:
        devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            model.eval()
            for name, p, e in [('fused', prototype, evidence),
                               ('neural_only', 0., 0.),
                               ('without_prototype', 0., evidence),
                               ('without_evidence', prototype, 0.)]:
                model.prototype_weight.fill_(p)
                model.evidence_weight.fill_(e)
                result[name] = dict(prototype_weight=p, evidence_weight=e,
                                    metrics=evaluate())
    finally:
        model.prototype_weight.fill_(prototype)
        model.evidence_weight.fill_(evidence)
        for module, training in modes:
            module.training = training
    return result


def summarize_audits(root):
    rows = []
    for path in sorted(Path(root).glob('*/O*/seed_*/audit_stage_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        for view, item in data['views'].items():
            metrics = item['metrics']
            rows.append(dict(method=data['ablation'], order=data['order'],
                             seed=data['seed'], stage=data['stage'], view=view,
                             test_macro=metrics['macro'], test_old=metrics['old'] if data['stage'] else None,
                             test_new=metrics['new'], test_macro_f1=metrics['macro_f1'],
                             prototype_weight=item['prototype_weight'], evidence_weight=item['evidence_weight'],
                             expert_ranks=json.dumps(data['expert_ranks']), audit_path=str(path)))
    if rows:
        with (Path(root)/'ablation_diagnostics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
