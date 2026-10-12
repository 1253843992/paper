"""Run the pre-registered five-order by three-seed ECHO-CIL suite."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train_echocil import ORDERS


def run(command: list[str]) -> None:
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True,
                        help='directory containing the 40 audited MAT records')
    parser.add_argument('--processed-root', type=Path,
                        default=Path('echo_cil_processed_suite'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('runs/echo_cil_suite'))
    parser.add_argument('--orders', nargs='+', choices=sorted(ORDERS),
                        default=sorted(ORDERS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--preprocess-only', action='store_true')
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    for order_name in args.orders:
        data_dir = args.processed_root / order_name
        if args.force or not (data_dir / 'manifest.csv').exists():
            run([
                sys.executable, str(root / 'echo_cil_preprocess.py'),
                '--input', str(args.input), '--output', str(data_dir),
                '--base-classes', *map(str, ORDERS[order_name][0]),
            ])
        if args.preprocess_only:
            continue
        for seed in args.seeds:
            output = args.output_root / order_name / f'seed_{seed}'
            if not args.force and (output / 'results.json').exists():
                print(f'SKIP completed {order_name} seed {seed}', flush=True)
                continue
            command = [
                sys.executable, str(root / 'train_echocil.py'),
                '--data', str(data_dir), '--output', str(output),
                '--order', order_name, '--seed', str(seed),
                '--device', args.device, '--num-workers', str(args.num_workers),
            ]
            command.append('--no-amp' if args.no_amp else '--amp')
            run(command)


if __name__ == '__main__':
    main()
