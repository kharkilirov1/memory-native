# TASK 05 — GRPO loop for the desktop env, with the mx6 optimizer-free method (same rules: only research/agent-rl/, no git)

Goal: fine-tune a HF causal LM (dev: `Qwen/Qwen2.5-0.5B-Instruct` on CPU; final: Qwen3.8-27B on a GPU) with RL on the
desktop tasks, where every nn.Linear of the transformer blocks is replaced by mx6 `LatentGridLinear` (no FP32 master
weights, no Adam: the update happens inside backward), plus an `--optim adamw` baseline with plain nn.Linear.

mx6 lives in `research/mxfp4-native-no-state/mxfp4-native/mx6` (add `research/mxfp4-native-no-state/mxfp4-native` to
sys.path). READ these before coding: `mx6/latent_grid.py` (`LatentGridLinear`, `from_dense`, `lg_layers`; it is
BIAS-FREE), `mx6/lg_accum.py` (`set_momentum(model, mu, dtype, n=N)` + `lg_accumulate(model, N)`: N micro-batch
backwards per optimizer step, exactly N per layer), and `probes/lg_ab.py` for how the arm `lg6_fp4_h2` maps to
constructor args (k, headroom, scale format). Best known recipe: lg6_fp4_h2, momentum 0.9, fp8 buffer, lr ~3x the
no-momentum lr.

## Files
1. `rl/convert.py` — `convert_to_lg(model, k=6, headroom=2.0, lr=..., momentum=0.9, mom_dtype="fp8", n_micro=N,
   skip=("lm_head","embed"))`: replace each nn.Linear inside the decoder layers by `LatentGridLinear.from_dense(...)`;
   if the Linear has a bias (Qwen q/k/v do), wrap: `LGLinearBias(lg, bias_param)` where the bias stays an nn.Parameter
   trained by AdamW (tiny). Embeddings, norms, lm_head: frozen by default (`--train-norms` makes norms AdamW). Return a
   report (n layers converted, params in lg, resident bytes). Must work on CPU (torch backend) and CUDA.
2. `rl/policy.py` — `LMPolicy(model, tokenizer, max_new_tokens=96, temperature=1.0)`: builds a chat prompt
   (system prompt = how to read the tree + the ACTION grammar from deskenv/actions.py + "end your reply with one line
   ACTION: {...}", brief reasoning allowed), user = task instruction + history (previous actions + their results, last
   4) + the CURRENT rendered tree only. `act(obs, history) -> (text, prompt_ids, completion_ids)` with sampling;
   `model.eval()` + `torch.no_grad()` during generation (LatentGridLinear must not update in eval/no_grad).
3. `rl/grpo.py` — one step:
   - sample B task seeds; for each, run G episodes of the SAME task (same seed -> same instruction/files) with the
     policy (runner.run_episode-like, but it must record every turn's (prompt_ids, completion_ids));
   - reward R per episode = env reward; advantage A = R - mean(R over the G of that task) (Dr.GRPO: no std division);
     skip groups with all-equal R;
   - loss for a turn = -A * sum(logprob of completion tokens) / T_total, T_total = completion tokens of all kept turns
     in the step (token-level normalisation); one optimizer step per GRPO step (strictly on-policy, no PPO ratio/clip,
     no reference model / KL);
   - lg arm: pack all kept turns into EXACTLY N micro-batches (`n_micro`, balanced by token count, sequences in a
     micro-batch right-padded with attention mask); if fewer turns than N, add zero-loss dummy micro-batches (a zero
     loss gives a zero grad, harmless for the EMA); run them inside `with lg_accumulate(model, N) as acc:
     for i in acc.micro_batches(): loss_i.backward()`; then AdamW step for the bias/norm params.
   - adamw arm: same packing + standard grad accumulation, AdamW on all trainable params, grad clip 1.0.
   - log per step: mean/max reward, success rate (R==1), mean turns, malformed rate, tokens, step time.
4. `rl/train.py` — CLI: `--model`, `--optim lg|adamw`, `--lr`, `--steps`, `--B 2 --G 4`, `--n-micro 4`, `--kinds`,
   `--max-turns 8`, `--ad-bin`, `--display`, `--seed`, `--out` (jsonl log + final summary), `--eval-every K` (greedy eval
   on a fixed set of 8 task seeds per kind, reports success rate), `--dry-env` (uses FakeAD-based scripted env so the
   loop runs on Windows without a desktop).
5. `tests/` — on CPU with a TINY random model (`transformers` LlamaConfig/Qwen2Config, hidden 64, 2 layers, vocab of the
   real tokenizer is not needed: build a tiny tokenizer-free path or use `AutoTokenizer` only if available offline —
   prefer a minimal fake tokenizer class in tests): convert_to_lg replaces the right layers and keeps outputs close to
   the dense model (relative error < 0.15); advantage computation; packing into exactly N micro-batches (incl. dummy
   padding); one grpo step with --dry-env changes lg weights (idx) only when some advantage != 0; adamw arm runs.

Keep it compact. Reply with: files + line counts, pytest summary line, doubts.
