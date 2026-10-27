"""Class-incremental mechanisms adapted to 1-D ECHO data (see README_CIL_EXPERIMENTS)."""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from .model import ECHOModel, ConvGNAct

METHODS = ('finetune', 'er', 'icarl', 'bic', 'lucir', 'wa', 'podnet',
           'derpp', 'der_expand', 'foster')


class Backbone(nn.Module):
    """Shared encoder; no protected experts, auxiliary head or inference mixing."""
    def __init__(self, kind='echo', harmonic_orders=None):
        super().__init__()
        self.kind = kind
        if kind == 'echo':
            source = ECHOModel([0], harmonic_orders=harmonic_orders)
            for name in ('evidence', 'quality_gate', 'time', 'fusion', 'spectrum',
                         'spectral_fusion', 'evidence_norm', 'spectral_gate'):
                setattr(self, name, getattr(source, name))
        else:
            self.blocks = nn.ModuleList([
                ConvGNAct(1, 32, 15, stride=4, padding=7),
                ConvGNAct(32, 64, 7, stride=2, padding=3),
                ConvGNAct(64, 96, 5, stride=2, padding=2),
                ConvGNAct(96, 128, 5, stride=2, padding=2)])
            self.output = nn.Linear(128, 128)

    def forward(self, batch):
        maps = []
        if self.kind != 'echo':
            x = batch['x'].unsqueeze(1)
            for block in self.blocks:
                x = block(x)
                maps.append(x)
            return self.output(x.mean(-1)), maps
        gamma = self.quality_gate(batch['quality'])
        evidence, _ = self.evidence(batch['order'], batch['rpm'])
        x = self.time.stem(batch['x'].unsqueeze(1))
        for block in self.time.blocks:
            x, _ = block(x, evidence.mean(1), gamma)
            maps.append(x)
        h, _ = self.fusion(x.transpose(1, 2), evidence, gamma)
        physical = self.evidence_norm(evidence.mean(1))
        gate = self.spectral_gate(torch.cat((h, physical, gamma), 1)).sigmoid()
        spectral = self.spectrum(batch['x'])
        return h + gate * self.spectral_fusion(torch.cat((h, spectral), 1)), maps


class CosineHead(nn.Module):
    def __init__(self, classes, proxies=1):
        super().__init__()
        self.proxies = proxies
        self.weight = nn.Parameter(torch.empty(classes, proxies, 128))
        nn.init.normal_(self.weight, std=.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.)))

    def forward(self, features):
        similarity = torch.einsum('bd,cpd->bcp', F.normalize(features.float(), dim=1),
                                  F.normalize(self.weight.float(), dim=2))
        # Local similarity classifier: within-class soft assignment to proxies.
        scores = (similarity.softmax(-1) * similarity).sum(-1)
        return self.log_scale.exp().clamp(max=100) * scores


class CILModel(nn.Module):
    def __init__(self, method, classes, backbone='echo', proxies=10, harmonic_orders=None):
        super().__init__()
        self.method, self.backbone_kind = method, backbone
        self.classes, self.proxies = list(classes), proxies
        self.encoders = nn.ModuleList([Backbone(backbone, harmonic_orders)])
        self.head = self.make_head(len(classes), 128)
        self.aux = None
        self.register_buffer('bias_scale', torch.ones(len(classes)))
        self.register_buffer('bias_shift', torch.zeros(len(classes)))

    def make_head(self, count, dim):
        if self.method in ('lucir', 'podnet'):
            return CosineHead(count, self.proxies if self.method == 'podnet' else 1)
        # WA must use an unnormalised, bias-free head.
        return nn.Linear(dim, count, bias=self.method != 'wa')

    def expand(self, new_classes):
        old_head = self.head
        device = next(self.parameters()).device
        old_count = len(self.classes)
        dynamic = self.method in ('der_expand', 'foster')
        if dynamic:
            for encoder in self.encoders:
                encoder.requires_grad_(False)
            self.encoders.append(copy.deepcopy(self.encoders[-1]).requires_grad_(True))
        self.classes.extend(new_classes)
        self.head = self.make_head(len(self.classes), 128 * len(self.encoders)).to(device)
        with torch.no_grad():
            if isinstance(self.head, CosineHead):
                self.head.weight[:old_count].copy_(old_head.weight)
                self.head.log_scale.copy_(old_head.log_scale)
            else:
                self.head.weight[:old_count].zero_()
                self.head.weight[:old_count, :old_head.in_features].copy_(old_head.weight)
                if self.head.bias is not None:
                    self.head.bias[:old_count].copy_(old_head.bias)
        self.bias_scale = torch.cat((self.bias_scale, torch.ones(len(new_classes), device=device)))
        self.bias_shift = torch.cat((self.bias_shift, torch.zeros(len(new_classes), device=device)))
        if dynamic:
            count = len(new_classes) + 1 if self.method == 'der_expand' else len(self.classes)
            self.aux = nn.Linear(128, count).to(device)

    def train(self, mode=True):
        super().train(mode)
        for encoder in self.encoders:
            if not any(p.requires_grad for p in encoder.parameters()):
                encoder.eval()
        return self

    def forward(self, batch):
        encoded = [encoder(batch) for encoder in self.encoders]
        features = torch.cat([pair[0] for pair in encoded], dim=1)
        logits = self.head(features)
        return {'logits': logits * self.bias_scale + self.bias_shift,
                'features': features, 'maps': encoded[-1][1],
                'aux': self.aux(encoded[-1][0]) if self.aux is not None else None}

    def frozen_copy(self):
        return copy.deepcopy(self).eval().requires_grad_(False)

    @torch.no_grad()
    def align(self, old_count):
        weights = self.head.weight
        ratio = weights[:old_count].norm(dim=1).mean() / weights[old_count:].norm(dim=1).mean().clamp_min(1e-8)
        weights[old_count:].mul_(ratio)
        return float(ratio)


def kd_loss(student, teacher, temperature=2.):
    return F.kl_div(F.log_softmax(student.float() / temperature, 1),
                    F.softmax(teacher.float() / temperature, 1), reduction='batchmean') * temperature**2


def pod_loss(student_maps, teacher_maps):
    """1-D POD: temporal pooling of squared activations, not flattened features."""
    distances = []
    for student, teacher in zip(student_maps, teacher_maps):
        left = F.normalize(student.float().square().sum(-1), dim=1)
        right = F.normalize(teacher.float().square().sum(-1), dim=1)
        distances.append((left - right).norm(dim=1).mean())
    return torch.stack(distances).mean()


def nca_loss(logits, targets, margin=.6):
    positive = logits.gather(1, targets[:, None]).squeeze(1) - margin
    negatives = logits.masked_fill(F.one_hot(targets, logits.shape[1]).bool(), -torch.inf)
    return (torch.logsumexp(negatives, dim=1) - positive).clamp_min(0).mean()


def objective(method, output, targets, old_count, teacher=None, *, temperature=2.,
              kd_weight=1., feature_weight=5., ranking_margin=.5, ranking_k=2,
              pod_spatial=5., pod_flat=1., class_weights=None):
    logits = output['logits'].float()
    ce = F.cross_entropy(logits, targets)
    if method == 'icarl':
        soft = F.one_hot(targets, logits.shape[1]).float()
        if teacher is not None:
            soft[:, :old_count] = teacher['logits'].float().sigmoid()
        return F.binary_cross_entropy_with_logits(logits, soft, reduction='none').sum(1).mean()
    if method == 'podnet':
        loss = nca_loss(logits, targets)
        if teacher is not None:
            factor = math.sqrt(logits.shape[1] / (logits.shape[1] - old_count))
            flat = (F.normalize(output['features'].float(), dim=1) -
                    F.normalize(teacher['features'].float(), dim=1)).square().sum(1).mean()
            loss += factor * (pod_flat * flat + pod_spatial * pod_loss(output['maps'], teacher['maps']))
        return loss
    if teacher is None:
        return ce
    if method in ('bic', 'wa'):
        distill = kd_loss(logits[:, :old_count], teacher['logits'], temperature)
        fraction = old_count / logits.shape[1]
        return (1 - fraction) * ce + fraction * kd_weight * distill
    if method == 'lucir':
        factor = math.sqrt(old_count / (logits.shape[1] - old_count))
        less_forget = (1 - F.cosine_similarity(output['features'].float(), teacher['features'].float())).mean()
        mask = targets < old_count
        rank = logits.sum() * 0
        if mask.any():
            # Margin is defined on unscaled cosine scores.
            scores = output['cosine']
            positives = scores[mask].gather(1, targets[mask, None])
            negatives = scores[mask, old_count:].topk(min(ranking_k, logits.shape[1]-old_count), 1).values
            rank = F.relu(ranking_margin - positives + negatives).mean()
        return ce + feature_weight * factor * less_forget + rank
    if method == 'der_expand':
        auxiliary_targets = torch.where(targets < old_count, 0, targets - old_count + 1)
        return ce + F.cross_entropy(output['aux'].float(), auxiliary_targets)
    if method == 'foster':
        return (F.cross_entropy(logits / class_weights, targets) +
                F.cross_entropy(output['aux'].float(), targets) +
                kd_weight * kd_loss(logits[:, :old_count], teacher['logits'], temperature))
    return ce


def effective_weights(counts, beta, device):
    values = torch.as_tensor(counts, dtype=torch.float32, device=device).clamp_min(1)
    weights = (1-beta) / (1-torch.pow(torch.tensor(beta, device=device), values)).clamp_min(1e-8)
    return weights / weights.mean()


def restore_model(checkpoint, device='cpu'):
    """Restore inference model from our metadata; load trusted local checkpoints only."""
    model = CILModel(checkpoint['method'], checkpoint['classes'], checkpoint['backbone'],
                     checkpoint['proxies'])
    count = checkpoint['encoder_count']
    if count > 1:
        model.encoders.extend(Backbone(checkpoint['backbone']) for _ in range(count-1))
        model.head = model.make_head(len(model.classes), 128*count)
    if checkpoint['aux_outputs'] is not None:
        model.aux = nn.Linear(128, checkpoint['aux_outputs'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval()
