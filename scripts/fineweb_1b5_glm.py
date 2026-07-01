"""MN-GLM (GLM-5.2-class) ~1.5-2B on ONE GPU: FineWeb-Edu BPE, reversible + Counter-MoE, checkpointed.

The runnable artifact for a real-corpus MN-GLM run (blueprint: results/MN_GLM_1B5.md). Everything is
set to what the runs in this repo showed:

  architecture : RMSNorm + GQA(+RoPE, QK-norm) + SwiGLU Counter-MoE  (glm.ReversibleMNGLM)
  memory       : reversible O(1) activations, anchor_every=2   (+35% tok/s vs pure reversible)
  weights      : counter_packed (0.75 B/weight), optimizer-in-state (no fp master, no Adam moments)
  int8         : forward + update on the Tensor Cores  (d>=768 -> int8 WINS: x2.05 fwd, x1.45-2.16 upd)
  act_save_bits: 8  (int8 saved activation reused by the int8 update -- no re-quant of X)
  MoE          : grouped + stacked kernels, ZERO per-expert python loops (x6.2 vs the naive loop)

Single-GPU by design: the counter memory is tiny (the enwik8 1B fit a 14.6 GiB T4 in 2.25 GiB), so a
~2B MN-GLM fits one mid-GPU with room to spare -- and single-process is trivially DDP-correct. NOTE
for multi-GPU scaling: the base counter linears all-reduce their grad_w inside backward (states stay
bit-identical), but the grouped MoE stacked update does NOT yet all-reduce its grad_w -- add that
before running data-parallel or the experts diverge across ranks. (single GPU: nothing to reduce.)

Resume: loads a mounted ckpt.pt if present, else fresh; saves model+opt+step every CKPT_EVERY.
Run:  python scripts/fineweb_1b5_glm.py
"""
import glob
import os
import sys
import time

import torch

# ---------------- config (~1.9B total / ~0.6B active; scale via L / E) ----------------
D, L, NH, NKV, BLK = 1536, 24, 12, 2, 1024      # d, layers, query-heads, kv-heads (GQA 6x), seq
E, TOP_K = 8, 2                                  # MoE experts / top-k  (capacity ~ E/top_k x dense)
LR, LR_SCALE, C, ABITS = 2e-3, 2e-4, 11, 8
ANCHOR = 2                                       # reversible speed/memory knob (2-4 sweet spot)
INT8 = True                                      # d=1536 >= 768 -> int8 wins; set False on narrow models
BATCH = 16
LOSS_CHUNK = 4096                                # chunk the vocab-50257 head so [B,T,V] is never built
VAL_TOKENS = 600_000
CKPT_EVERY, STEP_CAP, TIME_CAP = 200, 200_000, 38_000
CKPT_OUT = "ckpt_glm.pt"


def get_tokenizer():
    try:
        import tiktoken
    except Exception:
        os.system("pip -q install tiktoken >/dev/null 2>&1"); import tiktoken
    return tiktoken.get_encoding("gpt2")         # vocab 50257, eot 50256


class FWStream:
    """FineWeb-Edu (sample-10BT) streamed + GPT-2 BPE as a rolling token buffer."""
    def __init__(self, enc):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        self.it = iter(ds); self.enc = enc; self.eot = enc.eot_token; self.buf = []

    def _fill(self, n):
        while len(self.buf) < n:
            self.buf.extend(self.enc.encode_ordinary(next(self.it)["text"])); self.buf.append(self.eot)

    def pull(self, n_tokens, device):
        self._fill(n_tokens); out = torch.tensor(self.buf[:n_tokens], dtype=torch.long, device=device)
        self.buf = self.buf[n_tokens:]; return out

    def batch(self, B, T, device):
        need = B * (T + 1); self._fill(need)
        chunk = torch.tensor(self.buf[:need], dtype=torch.long, device=device).view(B, T + 1)
        self.buf = self.buf[need:]
        return chunk[:, :T].contiguous(), chunk[:, 1:].contiguous()


def main():
    assert torch.cuda.is_available(), "need a CUDA GPU"
    dev = "cuda"
    from memory_native import ReversibleMNGLM, fmt_bytes, peak_training_memory
    from memory_native.optimizers import build_optimizer

    enc = get_tokenizer(); vocab = enc.n_vocab
    stream = FWStream(enc)
    val_ids = stream.pull(VAL_TOKENS, dev)       # held out BEFORE training tokens (no overlap)
    print(f"tokenizer gpt2 vocab {vocab}; val {val_ids.numel()/1e6:.2f}M tok", flush=True)

    torch.manual_seed(0)
    ckw = dict(lr=LR, lr_scale=LR_SCALE, C=C, act_save_bits=ABITS)
    if INT8:
        ckw.update(forward_compute="int8", update_compute="int8")
    m = ReversibleMNGLM(vocab, D, L, NH, NKV, BLK, kind="counter_packed", n_experts=E, top_k=TOP_K,
                        qk_norm=True, grouped=True, swiglu=True, anchor_every=ANCHOR, **ckw).to(dev).train()
    opt = build_optimizer("adamw", m.trainable_parameters(), LR)

    start = 0
    ck = sorted(glob.glob("/kaggle/input/**/ckpt_glm.pt", recursive=True)) or (
        [CKPT_OUT] if os.path.exists(CKPT_OUT) else [])
    if ck:
        st = torch.load(ck[0], map_location=dev)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
        print(f"RESUMED from {ck[0]} at step {start}", flush=True)
    else:
        print("fresh start", flush=True)

    cc = sum(c.in_features * c.out_features for c in m.counter_layers())
    print(f"MN-GLM d={D} L={L} GQA {NH}/{NKV} E={E}/{TOP_K} swiglu anchor={ANCHOR} int8={INT8}: "
          f"{cc/1e9:.2f}B counter coeffs (persistent ~{fmt_bytes(cc*3//4)} vs fp32+Adam {fmt_bytes(cc*16)})",
          flush=True)

    xb, yb = stream.batch(BATCH, BLK, dev)
    def onestep():
        opt.zero_grad(set_to_none=True); _, l = m(xb, yb, loss_chunk=LOSS_CHUNK); l.backward(); opt.step()
    print(f"one-step peak (B={BATCH}): {fmt_bytes(peak_training_memory(onestep, torch.device(dev)))}", flush=True)

    @torch.no_grad()
    def evaluate(n=20):
        m.eval(); tot = 0.0; g = torch.Generator(device=dev).manual_seed(0)
        for _ in range(n):
            i = torch.randint(0, val_ids.numel() - BLK - 1, (BATCH,), generator=g, device=dev)
            x = torch.stack([val_ids[j:j+BLK] for j in i]); y = torch.stack([val_ids[j+1:j+1+BLK] for j in i])
            tot += m(x, y, loss_chunk=LOSS_CHUNK)[1].item()
        m.train(); return tot / n

    def save(step):
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step}, CKPT_OUT)

    print(f"\n==== TRAIN MN-GLM  batch {BATCH} x seq {BLK}  from step {start} ====", flush=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); s = start
    toks = BATCH * BLK
    while s < STEP_CAP and time.time() - t0 < TIME_CAP:
        x, y = stream.batch(BATCH, BLK, dev); _, loss = m(x, y, loss_chunk=LOSS_CHUNK)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); s += 1
        vloss = evaluate() if s % 200 == 0 else None
        if s % 25 == 0:
            el = time.time() - t0; tps = (s - start) * toks / el
            msg = f"  step {s:6d}  train {loss.item():.4f}  {tps:,.0f} tok/s  {el:.0f}s"
            if vloss is not None: msg += f"  val {vloss:.4f}"
            print(msg, flush=True)
        if s % CKPT_EVERY == 0:
            save(s)
    fval = evaluate(); save(s)
    el = time.time() - t0
    print(f"\nFINAL: step {s}  val {fval:.4f}  train {loss.item():.4f}  "
          f"peak {fmt_bytes(torch.cuda.max_memory_allocated())}  "
          f"{(s-start)*toks/el:,.0f} tok/s  {el:.0f}s  ckpt -> {CKPT_OUT}", flush=True)


if __name__ == "__main__":
    print("torch", torch.__version__, "| GPU",
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE", flush=True)
    main()
    print("=== DONE ===", flush=True)
