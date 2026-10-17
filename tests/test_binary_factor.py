import copy
import io
import math
import pytest
import torch
from memory_native.research.binary_factor import (
    initialize_binary_factors, NativeBinaryFactorLinear, QATBinaryFactorLinear,
    PackedBinaryLinear, pack6, unpack6, row_rms_step, tensor_state_bytes,
    BinaryFactorInference, export_binary_factors,
)

torch.set_num_threads(1)

def init(seed=1, n=12, k=16, r=4, **kw):
    w=torch.randn(n,k,generator=torch.Generator().manual_seed(seed)) / math.sqrt(k)
    return initialize_binary_factors(w,r,seed=seed,**kw)

@pytest.mark.parametrize('width',[4,8,16,64,132])
def test_pack_roundtrip_and_bytes(width):
    z=torch.randint(0,64,(7,width),generator=torch.Generator().manual_seed(width),dtype=torch.uint8)
    assert torch.equal(unpack6(pack6(z)),z)
    assert pack6(z).numel()==z.numel()*3//4

def test_every_code_and_lane():
    z=torch.arange(64,dtype=torch.uint8)[:,None].expand(64,4)
    assert torch.equal(unpack6(pack6(z)),z)

@pytest.mark.parametrize('bad',[torch.zeros(2,5),torch.ones(2,4)*2,torch.full((2,4),64,dtype=torch.uint8),torch.full((2,4),-1)])
def test_bad_pack(bad):
    with pytest.raises(ValueError):pack6(bad)

@pytest.mark.parametrize('bad',[torch.zeros(2,4,dtype=torch.uint8),torch.zeros(2,3),torch.zeros(0,dtype=torch.uint8)])
def test_bad_unpack(bad):
    with pytest.raises(ValueError):unpack6(bad)

@pytest.mark.parametrize('method',['svd','random','joint_itq'])
def test_initialization_is_deterministic_preserves_source_and_rng(method):
    w=torch.randn(12,16); original=w.clone(); rng=torch.random.get_rng_state().clone()
    a=initialize_binary_factors(w,4,method=method,seed=23)
    b=initialize_binary_factors(w,4,method=method,seed=23)
    assert torch.equal(w,original) and torch.equal(rng,torch.random.get_rng_state())
    assert torch.equal(a.dense_weight(),b.dense_weight())
    assert a.left_latent.shape==(12,4) and a.right_latent.shape==(4,16)

def test_itq_surrogate_decreases_not_claiming_operator_monotonicity():
    i=init(iterations=30)
    v=i.metadata['itq_objective']
    assert all(b<=a+1e-9 for a,b in zip(v,v[1:]))

@pytest.mark.parametrize('calibrated',[False,True])
def test_guarded_init_uses_only_supplied_objective(calibrated):
    calibration=torch.randn(40,16) if calibrated else None
    i=init(guarded=True,calibration=calibration)
    rows=i.metadata['candidates']
    selected=next(a for a in rows if a['name']==i.metadata['selected'])
    assert selected['relative_mse']<=rows[0]['relative_mse']+1e-12

@pytest.mark.parametrize('kw',[{'rank':3},{'rank':20},{'method':'bad'},{'iterations':-1},{'seed':True},{'method':'svd','guarded':True}])
def test_invalid_init(kw):
    args={'rank':4,**kw}
    with pytest.raises(ValueError):initialize_binary_factors(torch.randn(12,16),**args)

@pytest.mark.parametrize('shape',[(3,16),(2,3,16),(16,)])
def test_forward_and_all_gradients_match_qat_at_common_state(shape):
    i=init(); a=NativeBinaryFactorLinear(i,lr=.2); b=QATBinaryFactorLinear(i)
    x=torch.randn(shape,requires_grad=True); xb=x.detach().clone().requires_grad_()
    ya,yb=a(x),b(xb); upstream=torch.randn_like(ya)
    ya.backward(upstream); yb.backward(upstream)
    torch.testing.assert_close(ya,yb,atol=1e-6,rtol=2e-5)
    torch.testing.assert_close(x.grad,xb.grad,atol=1e-6,rtol=2e-5)
    for key in ('h','g','ell'):
        torch.testing.assert_close(getattr(a,key).grad,getattr(b,key).grad,atol=1e-6,rtol=2e-5)
    assert int(a.left.steps)==int(a.right.steps)==1
    assert not a.left._outstanding_forward and not a.right._outstanding_forward

@pytest.mark.parametrize('seed',range(5))
def test_bank_transition_matches_independent_formula(seed):
    q=torch.randn(5,12); b=PackedBinaryLinear(q,lr=.017,seed=seed)
    g=torch.randn(5,12); old=unpack6(b.codes).float()
    v=.1*g.square().mean(1,keepdim=True)
    z=old-16*.017*g/v.sqrt().clamp_min(.001)
    lo=z.floor(); u=torch.rand(z.shape,generator=torch.Generator().manual_seed(seed))
    expected=(lo+(u<z-lo)).clamp(0,63).to(torch.uint8)
    b.update_from_gradient(g)
    assert torch.equal(unpack6(b.codes),expected)
    torch.testing.assert_close(b.rms,v)

def test_update_never_uses_global_rng():
    b=PackedBinaryLinear(torch.ones(8,8),seed=9)
    state=torch.random.get_rng_state().clone()
    b.update_from_gradient(torch.ones(8,8))
    assert torch.equal(state,torch.random.get_rng_state())

@pytest.mark.parametrize('gradient',[100.,-100.,0.])
def test_saturation_and_zero(gradient):
    b=PackedBinaryLinear(torch.ones(8,8),lr=100)
    old=b.codes.clone(); b.update_from_gradient(torch.full((8,8),gradient))
    z=unpack6(b.codes)
    assert int(z.min())>=0 and int(z.max())<=63
    if gradient>0:assert bool((z==0).all())
    elif gradient<0:assert bool((z==63).all())
    else:assert torch.equal(old,b.codes)

def test_reuse_rejected_then_first_graph_completes():
    a=NativeBinaryFactorLinear(init()); x=torch.randn(3,16); y=a(x)
    with pytest.raises(RuntimeError):a(x)
    with pytest.raises(RuntimeError):a.state_dict()
    with pytest.raises(RuntimeError):a.set_lr(.01)
    y.sum().backward()
    a(x).sum().backward();assert int(a.left.steps)==2

@pytest.mark.parametrize('mode',['eval','disabled','no_grad'])
def test_read_only_does_not_modify_state(mode):
    a=NativeBinaryFactorLinear(init()); old=copy.deepcopy(a.state_dict())
    if mode=='eval':a.eval()
    if mode=='disabled':a.left.update_enabled=a.right.update_enabled=False
    with torch.no_grad() if mode=='no_grad' else torch.enable_grad():
        y=a(torch.randn(3,16,requires_grad=True))
        if mode!='no_grad':y.sum().backward()
    for name,value in old.items():
        if torch.is_tensor(value):assert torch.equal(value,a.state_dict()[name])

def test_gradient_before_mutation():
    b=PackedBinaryLinear(torch.ones(8,8),lr=100)
    x=torch.ones(2,8,requires_grad=True);old=b.visible().clone()
    b(x).sum().backward()
    torch.testing.assert_close(x.grad,torch.ones(2,8)@old)
    assert not torch.equal(old,b.visible())

def test_nonfinite_proposal_does_not_mutate_bank():
    b=PackedBinaryLinear(torch.ones(8,8)); state=copy.deepcopy(b.state_dict())
    with pytest.raises(ValueError):b.update_from_gradient(torch.full((8,8),float('nan')))
    for k,v in state.items():
        if torch.is_tensor(v):assert torch.equal(v,b.state_dict()[k])

def test_shared_scale_and_counter_checkpoint_continuation():
    a=NativeBinaryFactorLinear(init(),lr=.015,seed=46); opt=torch.optim.AdamW(a.parameters(),lr=.003,weight_decay=0)
    data=[torch.randn(7,16,generator=torch.Generator().manual_seed(30+k)) for k in range(9)]
    def step(m,o,x):
        o.zero_grad(set_to_none=True);loss=m(x).square().mean();loss.backward();o.step()
    for x in data[:4]:step(a,opt,x)
    file=io.BytesIO();torch.save({'m':a.state_dict(),'o':opt.state_dict()},file);file.seek(0)
    saved=torch.load(file,weights_only=True)
    b=NativeBinaryFactorLinear(init(),lr=99,seed=5);ob=torch.optim.AdamW(b.parameters(),lr=1)
    b.load_state_dict(saved['m']);ob.load_state_dict(saved['o'])
    for x in data[4:]:step(a,opt,x);step(b,ob,x)
    for k,v in a.state_dict().items():
        if torch.is_tensor(v):assert torch.equal(v,b.state_dict()[k])
        else:assert v==b.state_dict()[k]
    assert int(a.left.steps)==9

@pytest.mark.parametrize('key,value',[('version',9),('lr',float('nan')),('beta',1.2),('eps',0.),('seed',True),('update_enabled',None)])
def test_refuses_bad_checkpoint_metadata(key,value):
    b=PackedBinaryLinear(torch.ones(8,8));s=copy.deepcopy(b.state_dict());s['_extra_state'][key]=value
    with pytest.raises(RuntimeError):b.load_state_dict(s)

def test_no_full_factor_parameters_and_memory_accounting():
    a=NativeBinaryFactorLinear(init(n=32,k=64,r=8));b=QATBinaryFactorLinear(init(n=32,k=64,r=8))
    assert list(a.left.parameters())==list(a.right.parameters())==[]
    assert sum(p.numel() for p in a.parameters())==32+64+8
    assert a.left.codes.numel()+a.right.codes.numel()==8*(32+64)*3//4
    assert tensor_state_bytes(a)['persistent_total']<tensor_state_bytes(b)['persistent_total']

@pytest.mark.parametrize('rank',[4,8,12])
def test_binary_export_equivalence(rank):
    a=NativeBinaryFactorLinear(init(n=16,k=20,r=rank),bias=torch.randn(16));a.eval()
    artifact=export_binary_factors(a);inf=BinaryFactorInference(artifact)
    x=torch.randn(2,3,20)
    torch.testing.assert_close(a(x),inf(x),atol=1e-6,rtol=1e-5)
    assert list(inf.parameters())==[]
    assert 'rms' not in artifact and 'codes' not in artifact
    assert artifact['left_bits'].numel()==16*math.ceil(rank/8)

def test_row_rms_qat_update_matches_formula():
    b=QATBinaryFactorLinear(init());x=torch.randn(4,16);b(x).square().mean().backward()
    old=b.left.clone();v=.1*b.left.grad.square().mean(1,keepdim=True)
    expected=(old-.01*b.left.grad/v.sqrt().clamp_min(.001)).clamp(-31.5/16,31.5/16)
    row_rms_step(b,.01)
    torch.testing.assert_close(b.left,expected)
