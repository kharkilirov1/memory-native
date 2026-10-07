import itertools
import pytest
import torch
from memory_native.research.visible2_reference import (
    pack_visible2, unpack_visible2, patch_visible2_cpu, visible2_matmul_reference,
)


def test_all_81_four_ternary_combinations():
    src = torch.tensor(list(itertools.product((-1,0,1), repeat=4)), dtype=torch.int8)
    codes = pack_visible2(src)
    assert codes.dtype == torch.uint8
    assert codes.numel() == 81
    assert torch.equal(unpack_visible2(codes), src)


@pytest.mark.parametrize("n,k,group", (
    (3,12,4), (13,128,32), (7,128,128), (8,256,64),
))
def test_grouped_forward_permuted_parity(n,k,group):
    gen = torch.Generator().manual_seed(123+n)
    t = torch.randint(-1,2,(n,k),generator=gen,dtype=torch.int8)
    x = torch.randn(5,k,generator=gen)
    scale = torch.rand(n,k//group,generator=gen)*0.3+0.01
    perm = torch.randperm(k,generator=gen).to(torch.int32)
    y = visible2_matmul_reference(x,pack_visible2(t),scale,perm,group)
    Wp = t.float()*scale.repeat_interleave(group,dim=1)
    W = torch.empty_like(Wp)
    W[:,perm.long()] = Wp
    expected = x@W.T
    torch.testing.assert_close(y,expected,atol=1e-5,rtol=1e-5)


def test_two_flips_in_one_byte_are_preserved():
    t=torch.tensor([[-1,0,1,-1,1,1,-1,0]],dtype=torch.int8)
    pack=pack_visible2(t)
    changed=patch_visible2_cpu(
        pack,torch.tensor([0,1,4,7]),torch.tensor([1,-1,0,1]),
    )
    expected=t.clone()
    expected.reshape(-1)[[0,1,4,7]]=torch.tensor([1,-1,0,1],dtype=torch.int8)
    assert torch.equal(unpack_visible2(changed),expected)
    assert torch.equal(pack,pack_visible2(t))


def test_invalid_code_or_out_of_range_rejected():
    with pytest.raises(ValueError):
        unpack_visible2(torch.tensor([[255]],dtype=torch.uint8))
    with pytest.raises(ValueError):
        pack_visible2(torch.tensor([[0,2,0,0]],dtype=torch.int8))
    with pytest.raises(ValueError):
        patch_visible2_cpu(torch.tensor([[0]],dtype=torch.uint8),torch.tensor([4]),torch.tensor([1]))


def test_bytes_budget():
    t=torch.randint(-1,2,(128,512),dtype=torch.int8)
    assert pack_visible2(t).numel() == t.numel()//4


from memory_native.research.visible2_reference import derive_visible2_from_packed6


@pytest.mark.parametrize("C", [1,2,4,7,11])
def test_direct6_to_visible2_matches_ternary_oracle(C):
    gen = torch.Generator().manual_seed(C + 99)
    t = torch.randint(-1,2,(6,128),dtype=torch.int16,generator=gen)
    c = torch.randint(-(C-1),C,(6,128),dtype=torch.int16,generator=gen)
    codes = ((t+1)*(2*C-1)+(c+C-1)).to(torch.int32).reshape(6,-1,4)
    b0 = codes[...,0] | ((codes[...,1] & 3) << 6)
    b1 = (codes[...,1] >> 2) | ((codes[...,2] & 15) << 4)
    b2 = (codes[...,2] >> 4) | (codes[...,3] << 2)
    packed6 = torch.stack((b0,b1,b2),dim=-1).reshape(6,-1).to(torch.uint8)
    assert torch.equal(derive_visible2_from_packed6(packed6,C),pack_visible2(t))
