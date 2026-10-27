"""Read-only structural validation of completed PU preprocessing."""
import argparse
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from echo_cil.data import EchoCILDataset
from echo_cil.physics import physics_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    root = parser.parse_args().data
    assert json.loads((root/'status.json').read_text())['complete']
    with (root/'manifest.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    by_record = defaultdict(list)
    for row in rows:
        by_record[row['source']].append(row)
    assert len({r['sample_id'] for r in rows}) == len(rows)
    for entries in by_record.values():
        assert len({r['split'] for r in entries}) == 1
        ordered = sorted(entries, key=lambda r: int(r['start']))
        assert all(int(a['end']) <= int(b['start']) for a,b in zip(ordered, ordered[1:]))
        with np.load(root/entries[0]['shard']) as a:
            assert a['x'].shape == (len(entries),2048)
            assert a['order'].shape == (len(entries),3,256)
            assert a['quality'].shape == (len(entries),9)
            for key in ('x','order','quality','rpm'):
                assert np.isfinite(a[key]).all(), (entries[0]['source'],key)
    stats = json.loads((root/'normalization.json').read_text())
    for split in ('train','val','test'):
        dataset = EchoCILDataset(root,split)
        for i in (0,len(dataset)//2,len(dataset)-1):
            item = dataset[i]
            order,quality = physics_features(item['x'],item['rpm'],stats)
            np.testing.assert_allclose(order,item['order'],rtol=1e-5,atol=1e-5)
            np.testing.assert_allclose(quality,item['quality'],rtol=1e-5,atol=1e-5)
    print(json.dumps(dict(valid=True,records=len(by_record),windows=len(rows),
                         by_split=dict(Counter(r['split'] for r in rows))),indent=2))

if __name__ == '__main__':
    main()
