"""Audited stage-boundary restart for trusted BiC artifacts, not bitwise replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .baselines import restore_model
from .data import EchoCILDataset

def resume_bic_stage0(args,protocol):
    path=Path(args.resume_stage0)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if args.method != 'bic' or checkpoint['method'] != 'bic' or checkpoint['stage'] != 0:
        raise ValueError('--resume-stage0 currently supports BiC stage 0 only')
    if checkpoint['classes'] != protocol['task_order'][0]:
        raise ValueError('Checkpoint base classes do not match dataset')
    original=checkpoint['configuration']
    if original.get('dataset_protocol') != protocol:
        raise ValueError('Checkpoint dataset protocol differs from current dataset')
    for name in ('seed','order','backbone','proxies','replay_fraction','memory_bytes'):
        if original.get(name) != getattr(args,name):
            raise ValueError(f'Restart requires original {name}={original.get(name)}')
    if any(not torch.isfinite(v).all() for v in checkpoint['model'].values()):
        raise ValueError('Checkpoint contains nonfinite tensors')
    with (path.parent/'results.json').open(encoding='utf-8') as stream:
        previous=json.load(stream)
    prefix=[s for s in previous['stages'] if s['stage']==0]
    if len(prefix)!=1 or 'test' not in prefix[0]:
        raise ValueError('Missing unique stage-0 result')
    if previous['configuration'].get('dataset_protocol') != protocol:
        raise ValueError('Results protocol mismatch')
    with np.load(path.parent/'memory_stage_0.npz',allow_pickle=False) as archive:
        memory=[dict(x=r['x'].copy(),y=int(r['label']),rpm=float(r['rpm']),
                     record_id=int(r['file_id']),start=int(r['start']),sample_id=int(r['sample_id']))
                for r in archive['samples']]
    dataset=EchoCILDataset(args.data,'train',checkpoint['classes'])
    lookup={int(r['sample_id']):i for i,r in enumerate(dataset.rows)}
    if len(memory)>int(len(dataset)*args.replay_fraction) or len({x['sample_id'] for x in memory})!=len(memory):
        raise ValueError('Invalid replay quota or duplicate samples')
    if len(memory)!=prefix[0]['memory_count'] or prefix[0]['seen_training_count']!=len(dataset):
        raise ValueError('Replay/results/training-count mismatch')
    if {x['y'] for x in memory} != set(checkpoint['classes']):
        raise ValueError('Replay missing old classes')
    for item in memory:
        if item['sample_id'] not in lookup:
            raise ValueError('Replay sample is absent from current training manifest')
        source=dataset[lookup[item['sample_id']]]
        if (source['y']!=item['y'] or source['record_id']!=item['record_id'] or
            source['start']!=item['start'] or not np.isclose(source['rpm'],item['rpm']) or
            not np.array_equal(source['x'],item['x'])):
            raise ValueError('Replay differs from original training data')
    model=restore_model(checkpoint,args.device)
    model.requires_grad_(True)
    provenance=dict(checkpoint=str(path.resolve()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    memory_sha256=hashlib.sha256((path.parent/'memory_stage_0.npz').read_bytes()).hexdigest(),
                    source_configuration=original,bitwise_resume=False,
                    note='Stage-boundary restart with fresh RNG/optimizer; inherited stage-0 metrics remain unchanged')
    print('Restarting BiC at Stage 1; original Stage 0 retained. Not an exact RNG replay.',flush=True)
    return model,memory,len(dataset),prefix,provenance
