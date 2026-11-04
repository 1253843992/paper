import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from echo_cil.numerics import NumericalWatch
from echo_cil.baselines import CILModel,objective
from train_cil_baselines import update_memory

class NumericalTests(unittest.TestCase):
    def test_failure_writes_context_and_snapshot_without_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=CILModel('bic',[0,1],'cnn1d')
            original=model.head.weight.detach().clone()
            args=SimpleNamespace(output=Path(tmp),method='bic',device=torch.device('cpu'),amp=False,diagnose_nonfinite=True)
            optimizer=torch.optim.Adam(model.parameters())
            watch=NumericalWatch(args,model,None,optimizer,None,1,'train')
            watch.begin(3,7,dict(sample_id=np.array([123],dtype=np.uint64),y=np.array([1])))
            with self.assertRaisesRegex(FloatingPointError,'teacher.logits'):
                watch.check('teacher',dict(logits=torch.tensor([[0.,float('nan')]])))
            report=json.loads(next((Path(tmp)/'diagnostics').glob('*.json')).read_text())
            self.assertEqual(report['first_nonfinite'],'teacher.logits')
            self.assertEqual(report['sample_ids'],[123])
            self.assertEqual(report['epoch'],3)
            snapshot=torch.load(next((Path(tmp)/'diagnostics').glob('*.pt')),weights_only=False)
            self.assertIsNotNone(snapshot['rng_before'])
            torch.testing.assert_close(model.head.weight,original)

    def test_bic_components_do_not_change_objective(self):
        out={'logits':torch.randn(4,7,requires_grad=True)}
        teacher={'logits':torch.randn(4,5)}
        parts={}
        loss=objective('bic',out,torch.arange(4),5,teacher,components=parts)
        torch.testing.assert_close(loss,objective('bic',out,torch.arange(4),5,teacher))
        self.assertEqual(set(parts),{'classification','distillation'})

    def test_derpp_fifteen_class_history_and_mask(self):
        args=SimpleNamespace(method='derpp',total_classes=15,replay_fraction=.1,memory_bytes=33554432,seed=0)
        old=[];seen=0;classes=[]
        for group in ([0,1,2,3,4],[5,6],[7,8],[9,10],[11,12],[13,14]):
            classes+=group
            new=[dict(x=np.ones(2048,dtype=np.float32),y=c,rpm=1500.,record_id=c,start=i*2048,sample_id=c*100+i)
                 for c in group for i in range(20)]
            trajectory={x['sample_id']:np.arange(len(classes),dtype=np.float32) for x in new}
            seen+=len(new)
            old=update_memory(SimpleNamespace(classes=classes),old,new,seen,trajectory,args,{})
            self.assertTrue(all(len(x['dark_logits'])==15 for x in old))
            self.assertTrue(all(not x['dark_mask'][len(classes):].any() for x in old))
            for x in old:
                if x['y'] in group:
                    self.assertEqual(int(x['dark_mask'].sum()),len(classes))

if __name__=='__main__':
    unittest.main()
