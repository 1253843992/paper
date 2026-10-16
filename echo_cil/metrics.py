"""Shared evaluation definitions for ECHO and all comparison methods."""
import time

import numpy as np
import torch


QUALITY_FIELDS = ('macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1',
                  'old_to_new_rate', 'new_to_old_rate', 'old_new_harmonic')
TIMING_FIELDS = ('inference_batch_ms', 'inference_p50_ms', 'inference_p95_ms',
                 'inference_ms_per_sample', 'inference_samples_per_second')


def classification_metrics(confusion, old_count):
    cm = np.asarray(confusion, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or (cm < 0).any():
        raise ValueError('Expected square nonnegative confusion matrix: true rows, predicted columns')
    if not 0 <= old_count <= len(cm):
        raise ValueError('Invalid old class count')
    support, predicted, correct = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros_like(correct), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
    f1 = np.divide(2*correct, support+predicted, out=np.zeros_like(correct), where=support+predicted > 0)
    result = dict(macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                  macro_f1=float(f1.mean()), weighted_f1=float((f1*support).sum()/max(support.sum(), 1)),
                  old_to_new_rate=None, new_to_old_rate=None, old_new_harmonic=None)
    if 0 < old_count < len(cm):
        old_total, new_total = support[:old_count].sum(), support[old_count:].sum()
        result['old_to_new_rate'] = float(cm[:old_count, old_count:].sum()/old_total) if old_total else None
        result['new_to_old_rate'] = float(cm[old_count:, :old_count].sum()/new_total) if new_total else None
        old, new = recall[:old_count].mean(), recall[old_count:].mean()
        result['old_new_harmonic'] = float(2*old*new/(old+new)) if old+new else 0.
    return result


def forgetting_metrics(stages):
    if len(stages) <= 1:
        return {'forgetting': None, 'backward_transfer': None}
    tasks = [stage['test'].get('task_macro') for stage in stages]
    if any(values is None or len(values) < i+1 for i, values in enumerate(tasks)):
        return {'forgetting': None, 'backward_transfer': None}
    final = tasks[-1]
    forgetting = [max(tasks[j][i] for j in range(i, len(tasks)-1))-final[i] for i in range(len(tasks)-1)]
    backward = [final[i]-tasks[i][i] for i in range(len(tasks)-1)]
    return dict(forgetting=float(np.mean(forgetting)), backward_transfer=float(np.mean(backward)))


@torch.inference_mode()
def benchmark_inference(predict, batch_size, device, warmup=5, repeats=20):
    """Device-resident FP32 input; includes final prediction, excludes data I/O/H2D."""
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    with torch.autocast(device_type=device.type, enabled=False):
        for _ in range(warmup):
            predict()
        sync()
        times = []
        for _ in range(repeats):
            sync()
            started = time.perf_counter()
            predict()
            sync()
            times.append((time.perf_counter()-started)*1000)
    mean = float(np.mean(times))
    return dict(inference_batch_ms=mean, inference_p50_ms=float(np.percentile(times, 50)),
                inference_p95_ms=float(np.percentile(times, 95)),
                inference_ms_per_sample=mean/batch_size,
                inference_samples_per_second=batch_size*1000/mean,
                inference_batch_size=batch_size, inference_warmup=warmup,
                inference_repeats=repeats, inference_precision='float32',
                inference_scope='device_resident_inputs_to_argmax',
                inference_cpu_threads=torch.get_num_threads(),
                inference_device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU')
