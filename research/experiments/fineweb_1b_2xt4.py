"""Full-method 1B run on 2x T4 (Kaggle): FineWeb-Edu BPE, data-parallel, checkpointed.

The body of the Kaggle kernel script. At push time a PKG_B64 line (the packaged memory_native
source tar) is prepended and the per-rank tar is unpacked into sys.path inside each spawned worker.

Both T4s run data-parallel: each rank trains its own micro-batch, the conventional-parameter
grads are all-reduced in the loop, and the counter weight-gradient is all-reduced *inside* backward
(the counter optimizer is an in-place state update, so there is no Parameter .grad for DDP) -- this
keeps the two replicas' packed states bit-identical (validated: 0 bytes differ across ranks).

NCCL note: every collective (the counter grad_w all-reduce in backward, and evaluate()'s loss
all-reduce) must be called symmetrically by ALL ranks -- gating any of them under `if rank==0`
deadlocks the group. The one-step peak probe therefore runs on all ranks; only rank0 prints.

Resume: loads /kaggle/input/**/ckpt.pt if a checkpoint dataset is mounted, else starts fresh; saves
model+opt+step+per-layer SR seeds to /kaggle/working/ckpt.pt every CKPT_EVERY steps so training
continues across Kaggle sessions.
"""
import base64, io, tarfile, os, sys, time, glob
import torch, torch.multiprocessing as mp, torch.distributed as dist, torch.nn.functional as Fnn

# ---------------- config ----------------
D, L, H, BLK = 2048, 24, 16, 256        # 1.21B counter coeffs (same body as the enwik8 1B run)
BATCH_GLOBAL = 128                       # effective batch; split across the 2 T4s
LR, LR_SCALE, C, ABITS = 2e-3, 2e-4, 11, 8   # int8 saved activation -> reusable by the int8 update
# int8 Tensor-Core levers (the big win at d=2048): forward_compute=int8 runs Y=XT^T on the int8
# cache (x2.05 isolated forward); update_compute=int8 + act_save_bits=8 reuses the saved activation
# for the grad_w correlation (x1.45-2.16). Deterministic int8 forward keeps the reversible recompute
# valid. Set INT8=False to fall back to the fp path (better loss curve, slower).
INT8 = True
LOSS_CHUNK = 4096                        # chunk the vocab-50257 head so [B,T,V] is never built
VAL_TOKENS_PER_RANK = 600_000
CKPT_EVERY = 100
STEP_CAP = 200_000
TIME_CAP = 38_000                        # ~10.5h, leaves room for final ckpt under Kaggle's 12h
CKPT_OUT = "/kaggle/working/ckpt.pt"

def setup_pkg(rank):
    d = f"/kaggle/working/pkg{rank}"; os.makedirs(d, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(base64.b64decode(PKG_B64))).extractall(d)
    sys.path.insert(0, d + "/src")

def get_tokenizer():
    try:
        import tiktoken
    except Exception:
        os.system("pip -q install tiktoken >/dev/null 2>&1")
        import tiktoken
    return tiktoken.get_encoding("gpt2")   # vocab 50257, eot id 50256

class FWStream:
    """FineWeb-Edu (sample-10BT) streamed + GPT-2 BPE, sharded per rank, as a rolling token buffer."""
    def __init__(self, rank, world, enc):
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        ds = split_dataset_by_node(ds, rank=rank, world_size=world)
        self.it = iter(ds); self.enc = enc; self.eot = enc.eot_token; self.buf = []
    def _fill(self, n):
        while len(self.buf) < n:
            doc = next(self.it)
            self.buf.extend(self.enc.encode_ordinary(doc["text"])); self.buf.append(self.eot)
    def pull(self, n_tokens, device):
        self._fill(n_tokens); out = torch.tensor(self.buf[:n_tokens], dtype=torch.long, device=device)
        self.buf = self.buf[n_tokens:]; return out
    def batch(self, B, T, device):
        need = B * (T + 1); self._fill(need)
        chunk = torch.tensor(self.buf[:need], dtype=torch.long, device=device).view(B, T + 1)
        self.buf = self.buf[need:]
        return chunk[:, :T].contiguous(), chunk[:, 1:].contiguous()

def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29531")
    setup_pkg(rank); torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    from memory_native import ReversibleGPT, GPTConfig, fmt_bytes, memory_report, peak_training_memory
    from memory_native.optimizers import build_optimizer
    dev = f"cuda:{rank}"; log = (rank == 0)
    Bloc = BATCH_GLOBAL // world

    enc = get_tokenizer(); vocab = enc.n_vocab
    stream = FWStream(rank, world, enc)
    val_ids = stream.pull(VAL_TOKENS_PER_RANK, dev)        # held out BEFORE training tokens (no overlap)
    if log: print(f"tokenizer gpt2 vocab {vocab}; val {val_ids.numel()/1e6:.2f}M tok/rank", flush=True)

    torch.manual_seed(0)                                   # identical init on every rank
    cfg = GPTConfig(vocab, BLK, L, H, D)
    ckw = dict(lr=LR, lr_scale=LR_SCALE, C=C, act_save_bits=ABITS)
    if INT8:
        ckw.update(forward_compute="int8", update_compute="int8")   # cache_mode int8 is auto-set
    m = ReversibleGPT(cfg, "counter_packed", **ckw).to(dev).train()
    opt = build_optimizer("adamw", m.trainable_parameters(), LR)

    start_step = 0
    ck = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
    if ck:
        st = torch.load(ck[0], map_location=dev)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start_step = st["step"]
        for lay, s in zip(m.counter_layers(), st.get("sr_steps", [])): lay._sr_step = s
        if log: print(f"RESUMED from {ck[0]} at step {start_step}", flush=True)
    elif log:
        print("no checkpoint found -> fresh start", flush=True)

    if log:
        cc = sum(c.in_features * c.out_features for c in m.counter_layers())
        print(f"1B FULL METHOD: {cc/1e9:.2f}B counter coeffs, persistent "
              f"{fmt_bytes(memory_report(m)['persistent_bytes'])} (vs fp32+Adam {fmt_bytes(cc*16)})", flush=True)

    # one-step peak probe -- ALL ranks run it: the counter grad_w all-reduce inside backward is a
    # collective, so it must be called symmetrically or NCCL deadlocks. Only rank0 prints.
    xb, yb = stream.batch(Bloc, BLK, dev)
    def onestep():
        opt.zero_grad(set_to_none=True); _, l = m(xb, yb, loss_chunk=LOSS_CHUNK); l.backward()
        for p in m.trainable_parameters():       # keep conventional params synced through the probe too
            if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        opt.step()
    pk = peak_training_memory(onestep, torch.device(dev))
    if log: print(f"per-GPU one-step peak (B/gpu={Bloc}): {fmt_bytes(pk)}", flush=True)

    @torch.no_grad()
    def evaluate(n=20):
        # collective (all_reduce) -> must be called by ALL ranks, never under `if log`.
        m.eval(); tot = torch.zeros((), device=dev)
        g = torch.Generator(device=dev).manual_seed(0)
        for _ in range(n):
            i = torch.randint(0, val_ids.numel() - BLK - 1, (Bloc,), generator=g, device=dev)
            x = torch.stack([val_ids[j:j+BLK] for j in i]); y = torch.stack([val_ids[j+1:j+1+BLK] for j in i])
            tot += m(x, y, loss_chunk=LOSS_CHUNK)[1]
        m.train(); dist.all_reduce(tot, op=dist.ReduceOp.AVG); return (tot / n).item()

    def save_ckpt(step):
        if not log: return
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                    "sr_steps": [lay._sr_step for lay in m.counter_layers()]}, CKPT_OUT)

    if log: print(f"\n==== TRAIN 1B FineWeb-Edu  global-batch {BATCH_GLOBAL} ({Bloc}/gpu x{world})  "
                  f"from step {start_step} ====", flush=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); s = start_step
    toks = BATCH_GLOBAL * BLK
    while s < STEP_CAP and time.time() - t0 < TIME_CAP:
        x, y = stream.batch(Bloc, BLK, dev); _, loss = m(x, y, loss_chunk=LOSS_CHUNK)
        opt.zero_grad(set_to_none=True); loss.backward()
        for p in m.trainable_parameters():
            if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        opt.step(); s += 1
        vloss = evaluate() if s % 200 == 0 else None   # all ranks (collective inside)
        if log and s % 25 == 0:
            el = time.time() - t0; tps = (s - start_step) * toks / el
            msg = f"  step {s:6d}  train {loss.item():.4f}  {tps:,.0f} tok/s  {el:.0f}s"
            if vloss is not None: msg += f"  val {vloss:.4f}"
            print(msg, flush=True)
        if s % CKPT_EVERY == 0:
            save_ckpt(s); dist.barrier()
    fval = evaluate()                                   # all ranks
    save_ckpt(s); dist.barrier()
    if log:
        el = time.time() - t0
        print(f"\nFINAL: step {s}  val {fval:.4f}  train {loss.item():.4f}  "
              f"peak {fmt_bytes(torch.cuda.max_memory_allocated())}  "
              f"{(s-start_step)*toks/el:,.0f} tok/s  {el:.0f}s  ckpt -> {CKPT_OUT}", flush=True)
    dist.destroy_process_group()
    sys.stdout.flush(); sys.stderr.flush()
    # Hard-exit: the HF streaming dataset's background threads race the GIL during normal Python
    # finalization (PyGILState_Release fatal error), which flips the kernel to ERROR *after* the
    # checkpoint is already written -- risking the output not being published. os._exit skips that
    # finalization entirely; the work and the ckpt are done.
    os._exit(0)

if __name__ == "__main__":
    print("torch", torch.__version__, "GPUs", torch.cuda.device_count(),
          [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
    assert torch.cuda.device_count() >= 2, "need the T4 x2 machine"
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    print("=== DONE ===", flush=True)
