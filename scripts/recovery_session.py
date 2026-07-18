"""Resumable recovery-distillation session for a counter-swapped donor.

The engine behind the Colab 1.5B run (notebooks/qwen15_recovery_colab.ipynb) and the local
smoke test. Deliberately colab-free: paths and config come in, a witness dict comes out.

- Corpus: per-domain uint32 token bins (scripts/build_mix_corpus.py). Each training sequence
  samples its domain by weight from a step-seeded RNG, so the mix is stable AND resume-exact.
- Loop: KD(teacher||student, T=2) + ce_alpha*CE, AdamW on the fp slice, counter body
  self-updates in backward (fused Triton path on CUDA via counter_packed).
- Checkpoints: student state_dict + AdamW state + step, atomic write; resume picks up the
  corpus cursors deterministically from the step number.
- Witness: per-domain val PPL + fixed generations, logged to jsonl after every eval.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

EVAL_PROMPTS = [
    ("en", "The most important discovery in modern physics was"),
    ("en", "To train a neural network efficiently, you should"),
    ("ru", "Столица России — это город"),
    ("ru", "Чтобы приготовить борщ, нужно сначала"),
    ("code", "def fibonacci(n):\n    "),
    ("code", "import numpy as np\n\ndef softmax(x):\n    "),
]


class DomainMix:
    """Per-domain memmapped token streams with weight-sampled, step-deterministic batches."""

    def __init__(self, corpus_dir: str, seq: int, batch: int, seed: int = 0):
        with open(os.path.join(corpus_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.seq, self.batch, self.seed = seq, batch, seed
        self.train, self.val, self.weights, self.names = {}, {}, [], []
        for name, info in self.manifest["domains"].items():
            self.train[name] = np.memmap(os.path.join(corpus_dir, f"train_{name}.bin"),
                                         dtype=np.uint32, mode="r")
            vp = os.path.join(corpus_dir, f"val_{name}.bin")
            if os.path.exists(vp):
                self.val[name] = np.memmap(vp, dtype=np.uint32, mode="r")
            self.names.append(name)
            self.weights.append(info["share"])
        w = np.asarray(self.weights, dtype=np.float64)
        self.weights = w / w.sum()
        self._next_step = None                 # sequential-consumption cursor state

    def epoch_steps(self) -> int:
        total = sum(len(t) for t in self.train.values())
        return total // (self.seq * self.batch)

    def _seek(self, step: int) -> None:
        """Rebuild RNG + per-domain cursors as if steps 0..step-1 were consumed (one
        vectorized replay), so a resumed run reads exactly the untouched remainder."""
        self._rng = np.random.default_rng(self.seed)
        self._cursor = {n: 0 for n in self.names}
        if step:
            draws = self._rng.choice(len(self.names), size=step * self.batch, p=self.weights)
            for i, n in enumerate(self.names):
                self._cursor[n] = int((draws == i).sum())
        self._next_step = step

    def batch_at(self, step: int, device) -> torch.Tensor:
        if self._next_step != step:
            self._seek(step)
        rows = []
        for i in self._rng.choice(len(self.names), size=self.batch, p=self.weights):
            dom = self.names[int(i)]
            stream = self.train[dom]
            n_seq = max(len(stream) // self.seq, 1)
            pos = (self._cursor[dom] % n_seq) * self.seq
            rows.append(torch.from_numpy(stream[pos:pos + self.seq].astype(np.int64)))
            self._cursor[dom] += 1
        self._next_step = step + 1
        return torch.stack(rows).to(device)

    def val_batches(self, device, max_tokens: int = 120_000):
        out = {}
        for name, stream in self.val.items():
            n = min(len(stream), max_tokens) // self.seq
            rows = [torch.from_numpy(stream[i * self.seq:(i + 1) * self.seq].astype(np.int64))
                    for i in range(n)]
            bs = [torch.stack(rows[i:i + 4]).to(device) for i in range(0, len(rows), 4)]
            out[name] = bs
        return out


@torch.no_grad()
def generations(model, tokenizer, device, max_new: int = 40) -> dict:
    model.eval()
    out = {}
    for dom, prompt in EVAL_PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        out[f"{dom}:{prompt[:28]}"] = tokenizer.decode(gen[0][ids.shape[1]:],
                                                       skip_special_tokens=True)
    model.train()
    return out


def eval_all(student, tokenizer, val, device) -> dict:
    from memory_native.eval import perplexity
    res = {f"ppl_{n}": perplexity(student, b) for n, b in val.items()}
    res["gen"] = generations(student, tokenizer, device)
    return res


def save_ckpt(path: str, student, opt, step: int):
    tmp = path + ".tmp"
    torch.save({"step": step, "student": student.state_dict(), "opt": opt.state_dict()}, tmp)
    os.replace(tmp, path)


def run_session(*, student, teacher, tokenizer, mix: DomainMix, device,
                steps: int, ckpt_path: str, log_path: str,
                kd_alpha=1.0, ce_alpha=0.3, temperature=2.0,
                lr=3e-4, grad_clip=1.0, eval_every=500, ckpt_every=500,
                max_hours: float = 0.0) -> dict:
    """max_hours>0: stop cleanly (checkpoint + final eval) when the wall budget runs out --
    the Colab session ends on OUR schedule, not the platform's."""
    from memory_native.recovery.distill import kd_divergence

    student.train()
    fp = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(fp, lr=lr)

    start = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(ck["student"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
        print(f"[resume] from step {start}", flush=True)

    val = mix.val_batches(device)
    log = open(log_path, "a", encoding="utf-8")

    def emit(step, payload):
        payload = {"step": step, "t": time.time(), **payload}
        log.write(json.dumps(payload, ensure_ascii=False) + "\n")
        log.flush()

    if start == 0:
        base = eval_all(student, tokenizer, val, device)
        print("[warm-start]", {k: round(v, 1) for k, v in base.items() if k.startswith("ppl")},
              flush=True)
        emit(0, {"phase": "warm", **base})

    t_start = time.perf_counter()
    t_last = t_start
    stopped_at = steps
    for step in range(start, steps):
        if max_hours and (time.perf_counter() - t_start) > max_hours * 3600:
            print(f"[deadline] wall budget reached at step {step}", flush=True)
            stopped_at = step
            break
        ids = mix.batch_at(step, device)
        with torch.no_grad():
            t_logits = teacher(ids).logits
        out = student(ids, labels=ids if ce_alpha else None)
        loss = kd_alpha * kd_divergence(out.logits, t_logits, temperature)
        if ce_alpha:
            loss = loss + ce_alpha * out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(fp, grad_clip)
        opt.step()

        if (step + 1) % 50 == 0:
            dt = (time.perf_counter() - t_last) / 50
            t_last = time.perf_counter()
            lv = float(loss.detach())          # detach: no requires_grad->scalar warning
            print(f"step {step+1:6d}/{steps}  loss {lv:.3f}  {dt:.2f}s/step", flush=True)
            emit(step + 1, {"loss": lv, "s_per_step": dt})
        if (step + 1) % eval_every == 0:
            r = eval_all(student, tokenizer, val, device)
            print("  eval", {k: round(v, 1) for k, v in r.items() if k.startswith("ppl")},
                  flush=True)
            emit(step + 1, {"phase": "eval", **r})
        if (step + 1) % ckpt_every == 0:
            save_ckpt(ckpt_path, student, opt, step + 1)

    save_ckpt(ckpt_path, student, opt, stopped_at)
    final = eval_all(student, tokenizer, val, device)
    emit(stopped_at, {"phase": "final", **final})
    log.close()
    return final
