"""Six-stage CLI smoke on a tiny PU subset; metrics are NOT experiment results."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    if args.workspace.exists():
        raise ValueError('Use a new smoke workspace')
    data=args.workspace/'data'
    (data/'shards').mkdir(parents=True)
    for name in ('normalization.json','class_mapping.json','preprocess_config.json','status.json'):
        shutil.copy2(args.data/name,data/name)
    with (args.data/'manifest.csv').open(encoding='utf-8-sig') as stream:
        all_rows=list(csv.DictReader(stream))
    counts=Counter()
    rows=[]
    for row in all_rows:
        key=(row['split'],row['class_id'])
        if counts[key] < (20 if row['split']=='train' else 2):
            rows.append(row)
            counts[key]+=1
    for shard in {r['shard'] for r in rows}:
        shutil.copy2(args.data/shard,data/shard)
    with (data/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,'-u',str(root/'train_echocil.py'),'--data',str(data),
             '--config',str(root/'echo_cil_pu_config.json'),'--output',str(args.workspace/'run'),
             '--device','cpu','--no-amp','--cpu-threads','2','--num-workers','0',
             '--batch-size','32','--base-epochs','1','--incremental-epochs','1',
             '--head-warmup-epochs','1','--expert-warmup-epochs','1','--consolidation-epochs','1',
             '--experts-per-stage','1','--expert-limit','5','--prototype-weights','0.5',
             '--prototype-neighbors','1','--prototype-rpm-penalties','0','--evidence-weights','0']
    subprocess.run(command,check=True)
    result=json.loads((args.workspace/'run/results.json').read_text())
    assert result['complete'] and len(result['stages'])==6
    assert result['configuration']['dataset_protocol']['dataset']=='PU'
    print('PU SIX-STAGE CLI SMOKE PASSED (not scientific accuracy results)')

if __name__=='__main__':
    main()
