"""Dataset-specific tasks and semantics; never infer severity from PU IDs."""
import csv
import json
from pathlib import Path
import numpy as np
from .physics import HARMONIC_ORDERS

CWRU_ORDERS = {
    'O0': [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]],
    'O1': [[0, 7, 8, 9], [4, 5, 6], [1, 2, 3]],
    'O2': [[0, 4, 9, 2], [6, 1, 8], [3, 7, 5]],
    'O3': [[0, 8, 3, 4], [2, 7, 6], [9, 5, 1]],
    'O4': [[0, 6, 7, 5], [1, 9, 2], [8, 4, 3]],
}
CWRU_SEMANTICS = {str(c): {'family': 0 if c == 0 else 1+(c-1)%3,
                          'severity': 0 if c == 0 else 1+(c-1)//3} for c in range(10)}

def load_protocol(root, order_name, stats):
    root = Path(root)
    mapping = json.loads((root/'class_mapping.json').read_text(encoding='utf-8'))
    config = json.loads((root/'preprocess_config.json').read_text(encoding='utf-8'))
    dataset = config.get('dataset', 'CWRU')
    if dataset not in ('CWRU', 'PU'):
        raise ValueError(f'Unsupported dataset: {dataset}')
    labels = mapping['class_order']
    if stats.get('target_fs') != 12000 or config.get('window') != 2048:
        raise ValueError('Model requires 12 kHz / 2048-sample preprocessing')
    if mapping['class_id'] != {label: i for i, label in enumerate(labels)}:
        raise ValueError('Inconsistent class ID mapping')
    if dataset == 'PU':
        if not json.loads((root/'status.json').read_text())['complete']:
            raise ValueError('PU preprocessing is incomplete')
        orders = mapping['task_orders']
        anchors = stats['harmonic_orders']
        if not np.allclose(anchors, config['harmonic_queries']):
            raise ValueError('PU preprocessing/normalization anchors differ')
        semantics = {}
        for i, name in enumerate(labels):
            family = 'healthy' if name.startswith('K0') else 'outer' if name.startswith('KA') else 'inner' if name.startswith('KI') else None
            if family is None:
                raise ValueError(f'No explicit PU family definition: {name}')
            semantics[str(i)] = dict(family=family, severity=None)
    else:
        if len(labels) != 10:
            raise ValueError('Legacy CWRU protocol requires 10 classes')
        orders, anchors, semantics = CWRU_ORDERS, HARMONIC_ORDERS.tolist(), CWRU_SEMANTICS
    if order_name not in orders:
        raise ValueError(f'{dataset} has no order {order_name}; available: {list(orders)}')
    groups = orders[order_name]
    flat = [c for group in groups for c in group]
    if not all(groups) or sorted(flat) != list(range(len(labels))):
        raise ValueError('Tasks must cover every class exactly once')
    if list(stats['source_classes']) != groups[0]:
        raise ValueError('Normalization base classes do not match task order; reprocess this order')
    anchors = np.asarray(anchors, dtype=np.float32)
    if anchors.shape != (21,) or not np.isfinite(anchors).all() or (anchors <= 0).any():
        raise ValueError('Expected 21 finite positive harmonic anchors')
    with (root/'manifest.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if {int(r['class_id']) for r in rows} != set(flat):
        raise ValueError('Manifest classes differ from task configuration')
    for split in ('train','val','test'):
        if {int(r['class_id']) for r in rows if r['split']==split} != set(flat):
            raise ValueError(f'Missing classes in {split}')
    return dict(dataset=dataset, task_order=groups, class_names=labels,
                harmonic_orders=anchors.tolist(), class_semantics=semantics,
                severity_supported=dataset=='CWRU')
