"""Sequential, restart-safe runner for baseline and ECHO-CIL experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

from echo_cil.baselines import METHODS
from echo_cil.metrics import QUALITY_FIELDS, TIMING_FIELDS, classification_metrics, forgetting_metrics
from train_echocil import ORDERS


def completed(path):
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        groups = data.get('configuration', {}).get('task_order', data.get('order'))
        expected = len(groups) if isinstance(groups, list) else 3
        return (len(data.get('stages', [])) == expected and
                sorted(stage['stage'] for stage in data['stages']) == list(range(expected)) and
                all('test' in stage and 'macro' in stage['test'] for stage in data['stages']) and
                data.get('status', 'complete') == 'complete' and data.get('complete', True))
    except (ValueError, KeyError, TypeError):
        return False


def summarize(root):
    rows = []
    for path in sorted(root.glob('*/*/seed_*/results.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        method, order, seed = path.relative_to(root).parts[:3]
        history = []
        stage_history = []
        for result in sorted(data.get('stages', []), key=lambda s: s['stage']):
            if 'test' not in result or 'macro' not in result['test']:
                continue
            stage, metrics = result['stage'], result['test']
            history.append(metrics['macro'])
            stage_history.append(result)
            row = {'method': method, 'order': order, 'seed': int(seed.split('_')[-1]),
                   'stage': stage, 'run_complete': completed(path)}
            for name in ('overall', 'macro', 'old', 'new'):
                # Stage 0 has no old classes. ECHO historically writes 0 here;
                # use blank/null for every method rather than a false 0% accuracy.
                value = None if name == 'old' and stage == 0 else metrics.get(name)
                row['test_'+name] = value
                row['test_'+name+'_pct'] = value * 100 if value is not None else None
            quality = {name: metrics.get(name) for name in QUALITY_FIELDS}
            confusion = metrics.get('confusion_matrix', metrics.get('confusion'))
            if confusion is not None:
                groups = data.get('configuration', {}).get('task_order', data.get('order'))
                if not isinstance(groups, list):
                    groups = ORDERS[order]
                quality.update(classification_metrics(confusion, sum(map(len, groups[:stage]))))
            for name in QUALITY_FIELDS:
                value = quality[name]
                row['test_'+name] = value
                row['test_'+name+'_pct'] = value*100 if value is not None else None
            for name, value in forgetting_metrics(stage_history).items():
                row[name] = value
                row[name+'_pp'] = value*100 if value is not None else None
            for name in (*TIMING_FIELDS, 'inference_batch_size', 'inference_warmup',
                         'inference_repeats', 'inference_precision', 'inference_scope',
                         'inference_cpu_threads', 'inference_device_name'):
                row[name] = result.get(name)
            row.update(model_parameters=result.get('model_parameters', result.get('parameters_total')),
                       peak_expanded_model_parameters=result.get('peak_expanded_model_parameters'),
                       optimizer_steps=result.get('optimizer_steps', result.get('optimization_steps')),
                       stage_elapsed_seconds=result.get('elapsed_seconds', result.get('training_seconds')),
                       peak_cuda_allocated_bytes=result.get('cuda_peak_allocated_bytes', result.get('peak_cuda_bytes')),
                       replay_fraction_actual=result.get('replay_fraction_actual'),
                       prototype_tensor_bytes=result.get('prototype_tensor_bytes'))
            row.update(average_incremental_accuracy=statistics.mean(history),
                       memory_count=result.get('memory_count'),
                       memory_bytes=result.get('memory_bytes'),
                       results_path=str(path.resolve()))
            rows.append(row)
    if not rows:
        return
    with (root/'comparison_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    # Keep orders AND stages separate when aggregating random seeds.
    keys = sorted({(r['method'], r['order'], r['stage']) for r in rows})
    for method, order, stage in keys:
        group = [r for r in rows if (r['method'], r['order'], r['stage']) == (method, order, stage)]
        entry = {'method': method, 'order': order, 'stage': stage, 'runs': len(group),
                 'completed_runs': sum(r['run_complete'] for r in group)}
        for name in ('test_overall', 'test_macro', 'test_old', 'test_new', 'average_incremental_accuracy',
                     *('test_'+field for field in QUALITY_FIELDS), 'forgetting', 'backward_transfer',
                     *TIMING_FIELDS, 'model_parameters', 'optimizer_steps', 'stage_elapsed_seconds',
                     'peak_cuda_allocated_bytes', 'memory_count', 'memory_bytes'):
            values = [row[name] for row in group if row[name] is not None]
            entry[name+'_mean'] = statistics.mean(values) if values else None
            entry[name+'_std'] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(entry)
    (root/'comparison_summary.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    print(f'SUMMARY updated: {(root / "comparison_summary.csv").resolve()} ({len(rows)} stage rows)', flush=True)


def run_training(command, output_root):
    """Refresh the comparison table whenever a trainer reports a saved stage."""
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and 'stage' in event and 'test_macro' in event:
                    summarize(output_root)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=(*METHODS, 'echo'), default=[*METHODS, 'echo'])
    parser.add_argument('--orders', nargs='+', choices=ORDERS, default=['O0'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--data', type=Path, help='Single-order preprocessed directory')
    parser.add_argument('--processed-root', type=Path, default=Path('echo_cil_processed_suite'))
    parser.add_argument('--input', type=Path, help='Raw audited data, to preprocess missing orders')
    parser.add_argument('--output-root', type=Path, default=Path('runs/cil_comparison'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--replay-fraction', type=float, default=.1)
    parser.add_argument('--memory-bytes', type=int, default=32*1024*1024)
    parser.add_argument('--backbone', choices=('echo', 'cnn1d'), default='echo')
    parser.add_argument('--selection', choices=('val_macro', 'fixed'), default='val_macro')
    parser.add_argument('--finetune-epochs', type=int, default=20)
    parser.add_argument('--compression-epochs', type=int, default=40)
    parser.add_argument('--calibration-steps', type=int, default=200)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.output_root)
        return
    if args.data and len(args.orders) != 1:
        parser.error('--data requires exactly one order; use --processed-root for multiple orders')
    if not 0 < args.replay_fraction <= .1:
        parser.error('--replay-fraction must be in (0,.1]')
    if 'echo' in args.methods and (args.backbone != 'echo' or args.selection == 'fixed'):
        parser.error('Original ECHO uses its own backbone and validation selection; exclude echo for this setting')
    root = Path(__file__).resolve().parent
    for order in args.orders:
        data = args.data or args.processed_root/order
        if not (data/'manifest.csv').exists():
            if not args.input:
                parser.error(f'{data} is missing; pass --input to preprocess')
            command = [sys.executable, str(root/'echo_cil_preprocess.py'), '--input', str(args.input),
                       '--output', str(data), '--base-classes', *map(str, ORDERS[order][0])]
            print('PREPROCESS', subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
        for seed in args.seeds:
            for method in args.methods:
                output = args.output_root/method/order/f'seed_{seed}'
                command = [sys.executable, str(root/('train_echocil.py' if method == 'echo' else 'train_cil_baselines.py')),
                           '--data', str(data), '--output', str(output), '--order', order, '--seed', str(seed),
                           '--device', args.device, '--num-workers', str(args.num_workers),
                           '--batch-size', str(args.batch_size), '--replay-fraction', str(args.replay_fraction),
                           '--memory-bytes', str(args.memory_bytes), '--amp' if args.amp else '--no-amp']
                if method == 'echo':
                    command += ['--base-epochs', str(args.epochs), '--incremental-epochs', str(args.epochs)]
                    meta = data/'preprocess_config.json'
                    if meta.exists() and json.loads(meta.read_text(encoding='utf-8')).get('dataset') == 'PU':
                        command += ['--config', str(root/'echo_cil_pu_config.json')]
                else:
                    command += ['--method', method, '--epochs', str(args.epochs), '--backbone', args.backbone,
                                '--selection', args.selection, '--finetune-epochs', str(args.finetune_epochs),
                                '--compression-epochs', str(args.compression_epochs),
                                '--calibration-steps', str(args.calibration_steps)]
                print('RUN', subprocess.list2cmdline(command), flush=True)
                if args.dry_run:
                    continue
                files = [root/'train_cil_baselines.py', root/'echo_cil'/'baselines.py',
                         root/'train_echocil.py', root/'echo_cil_config.json',
                         *sorted((root/'echo_cil').glob('*.py')),
                         data/'manifest.csv', data/'normalization.json',
                         data/'class_mapping.json', data/'preprocess_config.json',
                         root/'echo_cil_pu_config.json']
                hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
                signature = hashlib.sha256(json.dumps([command, hashes], sort_keys=True).encode()).hexdigest()
                receipt = output/'suite_receipt.json'
                if output.exists() and any(output.iterdir()):
                    if (completed(output/'results.json') and receipt.exists() and
                            json.loads(receipt.read_text())['signature'] == signature):
                        print('SKIP completed matching configuration:', output, flush=True)
                        continue
                    raise RuntimeError(f'Incomplete or different run at {output}; choose a new --output-root. No files overwritten.')
                # Trainer refuses nonempty output; create receipt only after it exits.
                try:
                    run_training(command, args.output_root)
                finally:
                    if output.exists():
                        receipt.write_text(json.dumps({'signature': signature, 'command': command, 'hashes': hashes}, indent=2), encoding='utf-8')
                    summarize(args.output_root)
    if not args.dry_run:
        summarize(args.output_root)


if __name__ == '__main__':
    main()
