"""Fail-fast numerical diagnostics. Never replaces NaNs or skips batches."""
import json
import random
from pathlib import Path
import numpy as np
import torch

def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [cpu_tree(v) for v in value]
    return value

def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def tensor_summary(x):
    x=x.detach()
    finite=torch.isfinite(x)
    values=x[finite]
    return dict(shape=list(x.shape), dtype=str(x.dtype), nonfinite=int((~finite).sum()),
                minimum=float(values.min()) if values.numel() else None,
                maximum=float(values.max()) if values.numel() else None)

class NumericalWatch:
    def __init__(self,args,model,teacher,optimizer,scaler,stage,phase):
        self.args,self.model,self.teacher=args,model,teacher
        self.optimizer,self.scaler=optimizer,scaler
        self.context=dict(stage=stage,phase=phase,method=args.method,device=str(args.device),
                          amp=args.amp,torch_version=torch.__version__)
        self.raw=None
        self.observed={}
        self.before_rng=None

    def begin(self,epoch,batch,raw):
        self.context.update(epoch=epoch,batch=batch)
        self.raw=raw
        self.observed={}
        self.before_rng=rng_state() if getattr(self.args,'diagnose_nonfinite',False) else None

    def check(self,name,value):
        if isinstance(value,dict):
            for key,x in value.items():
                self.check(f'{name}.{key}',x)
        elif isinstance(value,(list,tuple)):
            for i,x in enumerate(value):
                self.check(f'{name}.{i}',x)
        elif torch.is_tensor(value):
            self.observed[name]=value.detach()
            if not torch.isfinite(value).all():
                self.fail(name)

    def fail(self,where):
        directory=Path(self.args.output)/'diagnostics'
        directory.mkdir(parents=True,exist_ok=True)
        stem=f'stage_{self.context["stage"]}_{self.context["phase"]}_epoch_{self.context["epoch"]}_batch_{self.context["batch"]}'
        report=dict(self.context,first_nonfinite=where,
                    sample_ids=[int(v) for v in self.raw.get('sample_id',[])],
                    labels=[int(v) for v in self.raw.get('y',[])],
                    tensors={k:tensor_summary(v) for k,v in self.observed.items()},
                    rng_before_available=self.before_rng is not None,
                    note='Snapshot is diagnostic, not a complete training-resume checkpoint')
        report_path=directory/(stem+'.json')
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        torch.save(dict(report=report,batch=self.raw,observed=cpu_tree(self.observed),model=cpu_tree(self.model.state_dict()),
                        teacher=cpu_tree(self.teacher.state_dict()) if self.teacher else None,
                        gradients={n:cpu_tree(p.grad) for n,p in self.model.named_parameters() if p.grad is not None},
                        optimizer=cpu_tree(self.optimizer.state_dict()) if self.optimizer else None,
                        scaler=self.scaler.state_dict() if self.scaler else None,
                        configuration={k:str(v) if isinstance(v,(Path,torch.device)) else v for k,v in vars(self.args).items()},
                        backbone=self.model.backbone_kind,proxies=self.model.proxies,
                        encoder_count=len(self.model.encoders),
                        class_ids=self.model.classes,rng_before=self.before_rng,rng_after=rng_state()),
                   directory/(stem+'.pt'))
        raise FloatingPointError(f'{where}: stage={self.context["stage"]}, epoch={self.context["epoch"]}, '
                                 f'batch={self.context["batch"]}; diagnostic: {report_path}')
