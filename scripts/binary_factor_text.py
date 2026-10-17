"""Small pretrained byte-GPT recovery, not a LittleBit paper reproduction.
Data: pinned local Python stdlib sources, whole-file train/val/test split.
Only FFN matrices are factorized. Attention, embeddings and norms are FP controls.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, sys, sysconfig, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from memory_native.research.binary_factor import *

class Block(nn.Module):
    def __init__(self,d=64,heads=4):
        super().__init__();self.heads=heads
        self.ln1=nn.LayerNorm(d);self.ln2=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,d*3,bias=False);self.proj=nn.Linear(d,d,bias=False)
        self.fc=nn.Linear(d,4*d,bias=False);self.fc2=nn.Linear(4*d,d,bias=False)
    def forward(self,x):
        b,t,d=x.shape;q,k,v=self.qkv(self.ln1(x)).chunk(3,-1)
        def heads(z):return z.reshape(b,t,self.heads,d//self.heads).transpose(1,2)
        a=F.scaled_dot_product_attention(heads(q),heads(k),heads(v),is_causal=True)
        x=x+self.proj(a.transpose(1,2).reshape(b,t,d))
        return x+self.fc2(F.gelu(self.fc(self.ln2(x))))

class ByteGPT(nn.Module):
    def __init__(self,d=64,context=64):
        super().__init__();self.d=d;self.context=context
        self.tok=nn.Embedding(256,d);self.pos=nn.Embedding(context,d)
        self.blocks=nn.ModuleList([Block(d),Block(d)])
        self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,256,bias=False)
        self.head.weight=self.tok.weight
        with torch.no_grad():
            self.tok.weight.normal_(std=.02);self.pos.weight.normal_(std=.02)
    def forward(self,x):
        h=self.tok(x)+self.pos(torch.arange(x.shape[1]))
        for b in self.blocks:h=b(h)
        return self.head(self.norm(h))


def prepare_data(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():return
    lib=Path(sysconfig.get_paths()['stdlib'])
    paths=sorted(lib.glob('*.py'))
    splits={s:[] for s in ['train','val','test']};manifest=[]
    for p in paths:
        raw=p.read_bytes()
        if len(raw)<1024:continue
        bucket=int(hashlib.sha256(p.name.encode()).hexdigest()[:8],16)%10
        split='train' if bucket<8 else 'val' if bucket==8 else 'test'
        splits[split].append(raw+b'\n\n')
        manifest.append({'name':p.name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'split':split})
    for split,parts in splits.items():
        # Avoid sample windows crossing file boundaries: sampling uses valid starts.
        stream=bytearray();starts=[]
        for raw in parts:
            base=len(stream);stream.extend(raw);starts.extend(range(base,base+len(raw)-65))
        torch.save({'bytes':torch.tensor(list(stream),dtype=torch.uint8),'starts':torch.tensor(starts,dtype=torch.int32)},root/f'{split}.pt')
    (root/'manifest.json').write_text(json.dumps({'source':'local Python standard library, top-level .py',
        'split_rule':'sha256(filename) first8hex modulo10: train<8,val==8,test==9','files':manifest},indent=2))


def load_data(root):
    items={s:torch.load(root/f'{s}.pt',weights_only=True) for s in ['train','val','test']}
    for item in items.values():item['bytes']=item['bytes'].long()
    return items


def batch(split,gen,batch_size=8,context=64):
    valid=split['starts'];starts=valid[torch.randint(len(valid),(batch_size,),generator=gen)].long()
    ids=starts[:,None]+torch.arange(context)[None,:]
    raw=split['bytes']
    return raw[ids],raw[ids+1]


def fixed(split,seed,n=32):
    gen=torch.Generator().manual_seed(seed)
    return [batch(split,gen) for _ in range(n)]

@torch.no_grad()
def evaluate(model,batches):
    training=model.training;model.eval();losses=[]
    for x,y in batches:losses.append(float(F.cross_entropy(model(x).flatten(0,1),y.flatten())))
    model.train(training)
    return sum(losses)/len(losses)


def write(path,data):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def train_teacher(root,data,steps=1200):
    torch.manual_seed(1234);m=ByteGPT();opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01)
    gen=torch.Generator().manual_seed(14159);val=fixed(data['val'],900,16);curves=[]
    start=time.perf_counter()
    for step in range(steps):
        x,y=batch(data['train'],gen)
        lr=.0003+.5*(.003-.0003)*(1+math.cos(math.pi*step/max(1,steps-1)))
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(m(x).flatten(0,1),y.flatten());loss.backward();opt.step()
        if step%200==0 or step==steps-1:
            curves.append({'step':step+1,'validation':evaluate(m,val)});print('teacher',curves[-1],flush=True)
    torch.save({'state':m.state_dict(),'steps':steps,'seed':1234},root/'teacher.pt')
    write(root/'teacher.json',{'steps':steps,'training_positions':steps*512,'seconds':time.perf_counter()-start,'curve':curves})


def converted(teacher,mode,method,lr,seed):
    model=copy.deepcopy(teacher);model.train();initializers=[];layers=[]
    if mode=='dense':return model,layers,initializers
    for i,b in enumerate(model.blocks):
        for j,name in enumerate(('fc','fc2')):
            orig=getattr(b,name)
            init=initialize_binary_factors(orig.weight,16,method='joint_itq' if method=='guarded' else method,
                guarded=method=='guarded',iterations=30,seed=seed+7919*(2*i+j))
            cls=NativeBinaryFactorLinear if mode in ('native','frozen') else QATBinaryFactorLinear
            layer=cls(init,lr=lr,seed=seed+123*(2*i+j)) if mode in ('native','frozen') else cls(init)
            if mode=='frozen':layer.left.update_enabled=layer.right.update_enabled=False
            setattr(b,name,layer);layers.append(layer);initializers.append(init.metadata)
    return model,layers,initializers


def recover(teacher,data,seed,mode,method,lr,steps=600,test=True,save_to=None):
    model,layers,inits=converted(teacher,mode,method,lr,seed)
    qparams=[p for m in layers if isinstance(m,QATBinaryFactorLinear) for p in [m.left,m.right]]
    scaleparams=[p for m in layers for p in [m.h,m.ell,m.g]]
    used={id(p) for p in qparams+scaleparams}
    rest=[p for p in model.parameters() if id(p) not in used]
    groups=[{'params':rest,'lr':.0003},{'params':scaleparams,'lr':.003}]
    if qparams:groups.append({'params':qparams,'lr':lr})
    opt=torch.optim.AdamW(groups,weight_decay=0.)
    val=fixed(data['val'],901,16);held=fixed(data['test'],902,64) if test else None
    before=evaluate(model,val)
    gen=torch.Generator().manual_seed(100000+seed);elapsed=0.;curve=[]
    teacher.eval()
    for step in range(steps):
        x,y=batch(data['train'],gen)
        t=time.perf_counter()
        with torch.no_grad():logp=F.log_softmax(teacher(x),-1)
        opt.zero_grad(set_to_none=True)
        logits=model(x);logq=F.log_softmax(logits,-1)
        kd=(logp.exp()*(logp-logq)).sum(-1).mean()
        ce=F.cross_entropy(logits.flatten(0,1),y.flatten())
        loss=kd+.3*ce
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();opt.step()
        for layer in layers:
            if isinstance(layer,QATBinaryFactorLinear):layer.clamp_latents()
        elapsed+=time.perf_counter()-t
        if (step+1)%200==0 or step+1==steps:
            curve.append({'step':step+1,'validation':evaluate(model,val)})
    score=evaluate(model,held) if test else None
    record={'mode':mode,'method':method,'lr':lr,'seed':seed,'steps':steps,'training_positions':steps*512,
            'warm_validation':before,'validation':curve[-1]['validation'],'test':score,
            'perplexity':None if score is None else math.exp(score),'seconds':elapsed,'includes_teacher_forward_time':True,
            'state':tensor_state_bytes(model,opt),'init':inits,'curve':curve,
            'scope':'small pretrained byte GPT; only 4 FFN matrices replaced; shared FP shell adapts; not full LittleBit-2'}
    if mode in ('native','frozen'):
        record['flips']=sum(int(l.left.flips+l.right.flips) for l in layers)
        record['updates']=[[int(l.left.steps),int(l.right.steps)] for l in layers]
    if save_to:
        torch.save({'record':record,'model':model.state_dict(),'optimizer':opt.state_dict(),'data_rng':gen.get_state()},save_to)
        # Reconstruct factory and confirm exact checkpoint roundtrip outputs.
        clone,_,_=converted(teacher,mode,method,lr,seed)
        clone.load_state_dict(torch.load(save_to,weights_only=True)['model'])
        assert evaluate(clone,held)==score
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=['teacher','tune','confirm'],required=True)
    p.add_argument('--start',type=int,default=40);p.add_argument('--seeds',type=int,default=1)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--selection',type=Path)
    args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    prepare_data(root/'data');data=load_data(root/'data')
    if args.phase=='teacher':train_teacher(root,data,args.steps);return
    teacher=ByteGPT();teacher.load_state_dict(torch.load(root/'teacher.pt',weights_only=True)['state']);teacher.eval()
    if args.phase=='tune':
        arms=[dict(mode=m,method=init,lr=lr) for m in ['native','qat_adam'] for init in ['svd','guarded'] for lr in [.001,.005,.02]]
    else:arms=json.loads(args.selection.read_text())['arms']
    out=root/args.phase;out.mkdir(exist_ok=True)
    for seed in range(args.start,args.start+args.seeds):
        for arm in arms:
            path=out/f"{arm['mode']}_{arm['method']}_lr{arm['lr']}_s{seed}.json"
            if path.exists():
                if json.loads(path.read_text())['steps']!=args.steps:
                    raise ValueError('Existing record has a different step budget; use a new directory')
                continue
            save_to=root/'native_seed40.pt' if args.phase=='confirm' and seed==40 and arm['mode']=='native' and arm['method']=='guarded' else None
            r=recover(teacher,data,seed,steps=args.steps,test=args.phase=='confirm',save_to=save_to,**arm)
            write(path,r);print(path.name,r['validation'],r['test'],r['seconds'],flush=True)
    if args.phase=='confirm' and not (root/'teacher_test.json').exists():
        score=evaluate(teacher,fixed(data['test'],902,64));write(root/'teacher_test.json',{'ce':score,'perplexity':math.exp(score)})
if __name__=='__main__':main()
