import copy
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from echo_cil.model import ECHOModel, restore_echo_model
from echo_cil.baselines import CILModel, restore_model
from echo_cil.protocol import load_protocol, CWRU_SEMANTICS
from echo_cil.physics import HARMONIC_ORDERS
from pu_cil_preprocess import HARMONICS, TASKS, CLASSES
from run_cil_comparison import completed, summarize


def semantics():
    return {str(i): dict(family='healthy' if i==0 else 'outer' if n.startswith('KA') else 'inner',
                         severity=None) for i,n in enumerate(CLASSES)}

class PUTrainingTests(unittest.TestCase):
    def test_six_stages_and_checkpoint(self):
        torch.set_num_threads(2)
        model = ECHOModel(TASKS[0], expert_limit=5, harmonic_orders=HARMONICS, class_semantics=semantics())
        batch = (torch.randn(2,2048),torch.randn(2,3,256),torch.randn(2,9),torch.ones(2)*1500)
        for stage, group in enumerate(TASKS):
            if stage:
                model.begin_increment(group,stage)
                model.add_expert(torch.empty(128,0),group)
            logits,_,_ = model(*batch)
            self.assertEqual(logits.shape,(2,5+stage*2))
            logits.sum().backward()
            model.zero_grad(set_to_none=True)
        model.set_prototypes(torch.randn(15,128),torch.arange(15),torch.ones(15)*1500,.5)
        model.set_prototype_options(1,0,.1,0)
        model.eval()
        checkpoint = dict(stage=5,architecture=model.architecture_metadata(),model=model.state_dict())
        restored = restore_echo_model(checkpoint)
        torch.testing.assert_close(model(*batch)[0],restored(*batch)[0])
        self.assertEqual(restored.class_semantics,semantics())
        self.assertEqual(restored.class_ids,list(range(15)))
        torch.testing.assert_close(restored.evidence.anchor_orders,torch.tensor(HARMONICS))

    def test_semantics_and_legacy_defaults(self):
        model = ECHOModel([0,1,2,3,4],class_semantics=semantics())
        model.prototype_labels = torch.arange(15)
        self.assertEqual(model.semantic_mask(1,'family').nonzero().flatten().tolist(),[1,2,5,7,9,11,13])
        self.assertEqual(model.semantic_mask(3,'severity').nonzero().flatten().tolist(),[3])
        with self.assertRaises(ValueError):
            model.set_prototype_options(1,0,0,.2)
        legacy = ECHOModel([0,1,2,3])
        self.assertEqual(legacy.class_semantics,CWRU_SEMANTICS)
        torch.testing.assert_close(legacy.evidence.anchor_orders,torch.from_numpy(HARMONIC_ORDERS))

    def test_dynamic_backbone_keeps_pu_anchors(self):
        model = CILModel('der_expand',TASKS[0],harmonic_orders=HARMONICS)
        model.expand(TASKS[1])
        for encoder in model.encoders:
            torch.testing.assert_close(encoder.evidence.anchor_orders,torch.tensor(HARMONICS))

    def test_completion_respects_six_stages(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'results.json'
            data=dict(configuration=dict(task_order=TASKS),complete=True,
                      stages=[dict(stage=i,test=dict(macro=.5)) for i in range(3)])
            path.write_text(json.dumps(data))
            self.assertFalse(completed(path))
            data['stages']=[dict(stage=i,test=dict(macro=.5)) for i in range(6)]
            path.write_text(json.dumps(data))
            self.assertTrue(completed(path))

    def test_protocol_rejects_wrong_order_and_normalization(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            mapping=dict(class_order=CLASSES,class_id=dict(zip(CLASSES,range(15))),task_orders={'O0':TASKS})
            config=dict(dataset='PU',window=2048,harmonic_queries=HARMONICS)
            for name,value in [('class_mapping.json',mapping),('preprocess_config.json',config),('status.json',dict(complete=True))]:
                (root/name).write_text(json.dumps(value))
            (root/'manifest.csv').write_text('class_id,split\n'+''.join(f'{c},{s}\n' for c in range(15) for s in ('train','val','test')))
            stats=dict(target_fs=12000,source_classes=TASKS[0],harmonic_orders=HARMONICS)
            self.assertEqual(load_protocol(root,'O0',stats)['task_order'],TASKS)
            with self.assertRaises(ValueError):
                load_protocol(root,'O1',stats)
            with self.assertRaises(ValueError):
                load_protocol(root,'O0',dict(stats,source_classes=[0,1,2,3]))

    def test_summary_emits_all_six_stages(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            out=root/'echo/O0/seed_0'
            out.mkdir(parents=True)
            stages=[]
            for i in range(6):
                n=5+i*2
                confusion=torch.eye(n,dtype=torch.int64).tolist()
                stages.append(dict(stage=i,test=dict(macro=1.,overall=1.,old=1.,new=1.,
                    task_macro=[1.]*(i+1),confusion_matrix=confusion)))
            (out/'results.json').write_text(json.dumps(dict(configuration=dict(task_order=TASKS),complete=True,stages=stages)))
            summarize(root)
            with (root/'comparison_summary.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),6)
            self.assertTrue(all(r['run_complete']=='True' for r in rows))

if __name__=='__main__':
    unittest.main()
