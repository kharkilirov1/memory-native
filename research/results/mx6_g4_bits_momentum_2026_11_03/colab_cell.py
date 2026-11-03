import subprocess, time, os
t0 = time.time()
def sh(cmd, timeout=None, show=40):
    print(f"\n$ {cmd[:200]}", flush=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).splitlines()
        print("\n".join(l[:300] for l in out[-show:] if "Warning" not in l), flush=True)
        print(f"[rc={r.returncode}]", flush=True)
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]", flush=True)
R = "/content/mn/research/mxfp4-native-no-state/mxfp4-native"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
sh(f"rm -rf /content/mn && git clone -q --depth 1 --branch research/mxfp4-native-no-state --filter=blob:none --sparse https://github.com/kharkilirov1/memory-native /content/mn && cd /content/mn && git sparse-checkout set research/mxfp4-native-no-state/mxfp4-native && git log --oneline -1")
os.makedirs("/content/q5", exist_ok=True)
# ---- 1) speed + memory at 1.21B (alone on the GPU), batch 8x512, no grad checkpoint
T = f"cd {R} && PYTHONPATH=. python probes/train_1b.py --probe-batch --batches 8 --probe-n 25 --no-grad-checkpoint --out /content/q5/speed "
for tag, a in [("dense", "--arm dense"), ("lg6", "--arm lg6_fp4_h2"),
               ("lg6_fp8", "--arm lg6_fp4_h2 --momentum 0.9 --momentum-dtype fp8"),
               ("lg6_int4", "--arm lg6_fp4_h2 --momentum 0.9 --momentum-dtype int4"),
               ("lg6_bf16", "--arm lg6_fp4_h2 --momentum 0.9 --momentum-dtype bf16")]:
    sh(f"{T}{a} 2>&1 | grep -E 'batch +8:|ERROR|Error' | sed 's/^/{tag}: /'", timeout=900, show=5)
print("speed done", round((time.time() - t0) / 60, 1), "min", flush=True)
# ---- 2) quality d512 enwik8 (parallel)
B = (f"cd {R} && PYTHONPATH=. python probes/lg_ab.py --dataset enwik8 --chars 20000000 --d 512 --layers 4 --heads 8 --steps 3000 "
     "--precision bf16 --sdpa --compile --sched cosine --adam-lr 1e-3 --scale-format nvfp4 --eval-every 250 --log-curve --batch 128 ")
sh(f"cd {R} && PYTHONPATH=. python probes/lg_ab.py --dataset enwik8 --chars 20000000 --d 64 --layers 1 --heads 1 --steps 1 --scale-format nvfp4 --arms dense_nobias --seeds 0 > /dev/null 2>&1; ls -la {R}/data | head", timeout=900)
M = "--momentum 0.9 --momentum-dtype"
runs = {
    "dense_s0": "--arms dense_nobias --seeds 0",
    "lg6_fp8_s1": f"--arms lg6_fp4_h2 --seeds 1 --lr 0.03 {M} fp8",
    "lg5_fp8_s0": f"--arms lg5_fp4_h2 --seeds 0 --lr 0.03 {M} fp8",
    "lg5_fp8_s1": f"--arms lg5_fp4_h2 --seeds 1 --lr 0.03 {M} fp8",
    "lg5_fp8_lr05": f"--arms lg5_fp4_h2 --seeds 0 --lr 0.05 {M} fp8",
    "lg4_fp8_s0": f"--arms lg4_fp4_h2 --seeds 0 --lr 0.03 {M} fp8",
    "lg6_int4_s0": f"--arms lg6_fp4_h2 --seeds 0 --lr 0.03 {M} int4",
    "lg6_int4_s1": f"--arms lg6_fp4_h2 --seeds 1 --lr 0.03 {M} int4",
    "lg5_int4_s0": f"--arms lg5_fp4_h2 --seeds 0 --lr 0.03 {M} int4",
    "lg5_int4_s1": f"--arms lg5_fp4_h2 --seeds 1 --lr 0.03 {M} int4",
}
procs = {k: subprocess.Popen(f"{B}{a} > /content/q5/{k}.log 2>&1", shell=True) for k, a in runs.items()}
while any(p.poll() is None for p in procs.values()):
    time.sleep(30)
    if time.time() - t0 > 85 * 60:
        print("TIME BUDGET HIT", flush=True); break
for k in runs:
    sh(f"grep -E 'momentum:|seed=|memory:|Error|error' /content/q5/{k}.log | sed 's/^/{k}: /'", show=6)
print("ALL_DONE", round((time.time() - t0) / 60, 1), "min", flush=True)
