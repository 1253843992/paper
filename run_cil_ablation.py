"""Run basic ECHO ablations and automatically export per-stage metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from echo_cil.ablations import ABLATIONS
from train_echocil import ORDERS
from run_cil_comparison import completed, run_training, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('echo_cil_processed'))
    parser.add_argument('--order', choices=ORDERS, default='O0')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--ablations', nargs='+', choices=ABLATIONS, default=list(ABLATIONS))
    parser.add_argument('--output-root', type=Path, default=Path('runs/cil_ablation'))
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'echo_cil_config.json')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--replay-fraction', type=float, default=.1)
    parser.add_argument('--smoke', action='store_true', help='One epoch per training phase; functionality only')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.output_root)
        return
    if args.epochs <= 0 or args.batch_size <= 0 or args.batch_size % 2 or args.cpu_threads <= 0 or args.num_workers < 0:
        parser.error('Positive epochs/threads, positive even batch size and nonnegative workers required')
    if not 0 < args.replay_fraction <= .1:
        parser.error('--replay-fraction must be in (0,.1]')
    root = Path(__file__).resolve().parent
    files = [root/'train_echocil.py', Path(__file__), root/'run_cil_comparison.py',
             *sorted((root/'echo_cil').glob('*.py')), args.config,
             args.data/'manifest.csv', args.data/'normalization.json']
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for seed in args.seeds:
        for ablation in args.ablations:
            output = args.output_root/ablation/args.order/f'seed_{seed}'
            command = [sys.executable, '-u', str(root/'train_echocil.py'), '--config', str(args.config),
                       '--data', str(args.data), '--output', str(output), '--order', args.order,
                       '--seed', str(seed), '--ablation', ablation, '--device', args.device,
                       '--base-epochs', str(1 if args.smoke else args.epochs),
                       '--incremental-epochs', str(1 if args.smoke else args.epochs),
                       '--batch-size', str(args.batch_size), '--num-workers', str(args.num_workers),
                       '--cpu-threads', str(args.cpu_threads), '--replay-fraction', str(args.replay_fraction),
                       '--amp' if args.amp else '--no-amp']
            if args.smoke:
                command += ['--head-warmup-epochs', '1', '--expert-warmup-epochs', '1',
                            '--consolidation-epochs', '1', '--prototype-weights', '0',
                            '--prototype-neighbors', '1', '--prototype-rpm-penalties', '0',
                            '--evidence-weights', '0']
            print('RUN', subprocess.list2cmdline(command), flush=True)
            if args.dry_run:
                continue
            signature = hashlib.sha256(json.dumps([command, hashes], sort_keys=True).encode()).hexdigest()
            receipt = output/'suite_receipt.json'
            if output.exists() and any(output.iterdir()):
                if (completed(output/'results.json') and receipt.exists() and
                        json.loads(receipt.read_text())['signature'] == signature):
                    summarize(args.output_root)
                    continue
                raise RuntimeError(f'Incomplete/different run: {output}. Use a new output root.')
            try:
                run_training(command, args.output_root)
            finally:
                if output.exists():
                    receipt.write_text(json.dumps(dict(signature=signature, command=command, hashes=hashes), indent=2), encoding='utf-8')
                summarize(args.output_root)


if __name__ == '__main__':
    main()
