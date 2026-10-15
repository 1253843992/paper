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
from train_echocil import ORDERS


def completed(path):
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return (len(data.get('stages', [])) == 3 and
                all('test' in stage and 'macro' in stage['test'] for stage in data['stages']) and
                data.get('status', 'complete') == 'complete' and data.get('complete', True))
    except (ValueError, KeyError, TypeError):
        return False


def summarize(root):
    rows = []
    for path in sorted(root.glob('*/*/seed_*/results.json')):
        if not completed(path):
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        method, order, seed = path.relative_to(root).parts[:3]
        final = data['stages'][-1]
        row = {'method': method, 'order': order, 'seed': int(seed.split('_')[-1]),
               'final_macro': final['test']['macro'], 'final_old': final['test']['old'],
               'final_new': final['test']['new'],
               'average_incremental_accuracy': statistics.mean(s['test']['macro'] for s in data['stages']),
               'memory_count': final.get('memory_count'),
               'memory_bytes': final.get('memory_bytes'),
               'results_path': str(path.resolve())}
        rows.append(row)
    if not rows:
        return
    with (root/'comparison_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for method in sorted({row['method'] for row in rows}):
        # First aggregate seeds WITHIN each order. Do not pretend orders are seeds.
        for order in sorted({row['order'] for row in rows if row['method'] == method}):
            group = [row for row in rows if row['method'] == method and row['order'] == order]
            entry = {'method': method, 'order': order, 'runs': len(group)}
            for name in ('final_macro', 'final_old', 'final_new', 'average_incremental_accuracy'):
                values = [row[name] for row in group]
                entry[name+'_mean'] = statistics.mean(values)
                entry[name+'_std'] = statistics.stdev(values) if len(values) > 1 else None
            aggregate.append(entry)
    (root/'comparison_summary.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')


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
                         data/'manifest.csv', data/'normalization.json']
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
                    subprocess.run(command, check=True)
                finally:
                    if output.exists():
                        receipt.write_text(json.dumps({'signature': signature, 'command': command, 'hashes': hashes}, indent=2), encoding='utf-8')
                    summarize(args.output_root)
    if not args.dry_run:
        summarize(args.output_root)


if __name__ == '__main__':
    main()
