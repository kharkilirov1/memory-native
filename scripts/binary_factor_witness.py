"""Paired CPU regression: test geometry and learning rules separately.
No claim of reproducing LittleBit-2 model-scale training or GPU throughput.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from memory_native.research.binary_factor import *


def dataset(seed, family='spectral', n=48, k=64, rank=8):
    gen=torch.Generator().manual_seed(seed+132401)
    if family=='binary':
        a=torch.randint(0,2,(n,rank),generator=gen).float()*2-1
        b=torch.randint(0,2,(rank,k),generator=gen).float()*2-1
        h=torch.rand(n,generator=gen)*.8+.2
        l=torch.rand(rank,generator=gen)*.8+.2
        g=torch.rand(k,generator=gen)*.8+.2
        w=(h[:,None]*a*l)@(b*g)/math.sqrt(k*rank)
    else:
        a,_=torch.linalg.qr(torch.randn(n,n,generator=gen))
        b,_=torch.linalg.qr(torch.randn(k,n,generator=gen))
        s=torch.arange(1,n+1).float().pow(-.7)
        # Deliberately heterogeneous row/column magnitudes, specified in advance.
        h=torch.exp(.6*torch.randn(n,generator=gen))
        g=torch.exp(.6*torch.randn(k,generator=gen))
        w=(h[:,None]*a*s)@b.T*g
    x=torch.randn(2304,k,generator=gen)
    y=x@w.T
    return w,(x[:1536],y[:1536]),(x[1536:1792],y[1536:1792]),(x[1792:],y[1792:])


def create(initial,mode,lr,scale_lr,seed):
    if mode in ('native','frozen'):
        model=NativeBinaryFactorLinear(initial,lr=lr,seed=seed+371)
        if mode=='frozen':
            model.left.update_enabled=model.right.update_enabled=False
        optimizer=torch.optim.AdamW(model.parameters(),lr=scale_lr,weight_decay=0)
    else:
        model=QATBinaryFactorLinear(initial)
        if mode=='qat_adam':
            optimizer=torch.optim.AdamW([{'params':[model.left,model.right],'lr':lr},
                {'params':model.scale_parameters(),'lr':scale_lr}],weight_decay=0)
        else:
            optimizer=torch.optim.AdamW(model.scale_parameters(),lr=scale_lr,weight_decay=0)
    return model,optimizer


def run(seed,mode,method,lr,steps=600,scale_lr=.003,family='spectral',test=True):
    w,train,val,held=dataset(seed,family)
    init=initialize_binary_factors(w,8,method='joint_itq' if method=='guarded' else method,
                                   guarded=method=='guarded',iterations=30,seed=seed+915)
    model,optimizer=create(init,mode,lr,scale_lr,seed)
    with torch.no_grad():
        warm=float((model(val[0])-val[1]).square().mean()/val[1].square().mean())
    stream=torch.Generator().manual_seed(seed+55201)
    denom=train[1].square().mean()
    seconds=0.
    for _ in range(steps):
        ids=torch.randint(1536,(64,),generator=stream)
        x,y=train[0][ids],train[1][ids]
        t=time.perf_counter()
        model.zero_grad(set_to_none=True)
        loss=(model(x)-y).square().mean()/denom
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward()
        if mode=='qat_rms':row_rms_step(model,lr)
        optimizer.step()
        if mode=='qat_adam':model.clamp_latents()
        seconds+=time.perf_counter()-t
    with torch.no_grad():
        finalval=float((model(val[0])-val[1]).square().mean()/val[1].square().mean())
        finaltest=float((model(held[0])-held[1]).square().mean()/held[1].square().mean()) if test else None
    record=dict(seed=seed,family=family,mode=mode,method=method,lr=lr,scale_lr=scale_lr,
        steps=steps,warm_validation=warm,validation=finalval,test=finaltest,seconds=seconds,
        state=tensor_state_bytes(model,optimizer),init=init.metadata,
        weight_sha256=hashlib.sha256(w.numpy().tobytes()).hexdigest())
    if mode in ('native','frozen'):
        record['updates']=[int(model.left.steps),int(model.right.steps)]
        record['flips']=int(model.left.flips+model.right.flips)
    return record


def save(path,record):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['tune','confirm'],required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--selection',type=Path)
    p.add_argument('--family',choices=['spectral','binary'],default='spectral')
    args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=='tune':
        arms=[dict(mode=mode,method=method,lr=lr,scale_lr=.003)
              for method in ('svd','guarded') for mode in ('native','qat_adam','qat_rms')
              for lr in (.001,.005,.02)]
        arms += [dict(mode='frozen',method=m,lr=0.,scale_lr=.003) for m in ('svd','guarded')]
    else:
        arms=json.loads(args.selection.read_text())['arms']
    for seed in range(args.start,args.start+args.seeds):
        for arm in arms:
            name=f"{args.family}_{arm['mode']}_{arm['method']}_lr{arm['lr']}_s{seed}.json"
            path=args.output/name
            if path.exists():
                old=json.loads(path.read_text())
                if old['steps']!=args.steps or old['scale_lr']!=arm['scale_lr']:
                    raise ValueError('Existing record has a different configuration; use a new output directory')
                print('EXISTS',name,flush=True);continue
            rec=run(seed=seed,steps=args.steps,family=args.family,test=args.phase=='confirm',**arm)
            save(path,rec)
            print(name,round(rec['validation'],6),rec['test'],round(rec['seconds'],3),flush=True)

if __name__=='__main__':main()
