import subprocess, time, os
t0 = time.time()
def sh(cmd, timeout=None, show=40):
    print(f"\n$ {cmd[:200]}", flush=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).splitlines()
        print("\n".join(l[:300] for l in out[-show:] if "Warning" not in l), flush=True)
        print(f"[rc={r.returncode}]", flush=True)
        return r.returncode
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]", flush=True)
        return -1
R = "/content/mn/research/mxfp4-native-no-state/mxfp4-native"
os.makedirs("/content/q6", exist_ok=True)
sh(f"rm -rf /content/mn && git clone -q --depth 1 --branch research/mxfp4-native-no-state --filter=blob:none --sparse https://github.com/kharkilirov1/memory-native /content/mn && cd /content/mn && git sparse-checkout set research/mxfp4-native-no-state/mxfp4-native && git log --oneline -1")
# ---- 0) fused-kernel tests ON CUDA (gate: stop if they fail)
rc = sh(f"cd {R} && PYTHONPATH=. python -m pytest -q tests/test_lg_fused_m4.py tests/test_lg_momentum.py 2>&1 | tail -15", timeout=900)
if rc != 0:
    print("FUSED TESTS FAILED -> stop", flush=True)
    raise SystemExit
# data for the 1B run (background, while the rest runs)
dp = subprocess.Popen(f"cd {R} && PYTHONPATH=. python probes/train_1b.py --prepare-data --data-dir /content/data > /content/q6/prep.log 2>&1", shell=True)
# ---- 1) speed at 1.21B, batch 8x512, no grad checkpoint
T = f"cd {R} && PYTHONPATH=. python probes/train_1b.py --probe-batch --batches 8 --probe-n 25 --no-grad-checkpoint --out /content/q6/speed "
for tag, a in [("lg6", "--arm lg6_fp4_h2"),
               ("lg6_int4_fused", "--arm lg6_fp4_h2 --momentum 0.9 --momentum-dtype int4"),
               ("lg6_int4_unfused", "--arm lg6_fp4_h2 --momentum 0.9 --momentum-dtype int4", ),
               ("dense", "--arm dense")]:
    env = "MX6_FUSE_M4=0 " if tag.endswith("unfused") else ""
    sh(f"{env}{T}{a} 2>&1 | grep -E 'batch +8:|ERROR|Error' | sed 's/^/{tag}: /'", timeout=900, show=5)
print("speed done", round((time.time() - t0) / 60, 1), "min", flush=True)
# ---- 2) d512 quality (parallel): fused int4 sanity + 5-bit variants
B = (f"cd {R} && PYTHONPATH=. python probes/lg_ab.py --dataset enwik8 --chars 20000000 --d 512 --layers 4 --heads 8 --steps 3000 "
     "--precision bf16 --sdpa --compile --sched cosine --adam-lr 1e-3 --scale-format nvfp4 --eval-every 250 --log-curve --batch 128 ")
sh(f"cd {R} && PYTHONPATH=. python probes/lg_ab.py --dataset enwik8 --chars 20000000 --d 64 --layers 1 --heads 1 --steps 1 --scale-format nvfp4 --arms dense_nobias --seeds 0 > /dev/null 2>&1; ls {R}/data", timeout=900)
M = "--momentum-dtype int4"
runs = {
    "lg6_int4f_s0": f"--arms lg6_fp4_h2 --seeds 0 --lr 0.03 --momentum 0.9 {M}",
    "lg5_int4_m095": f"--arms lg5_fp4_h2 --seeds 0 --lr 0.03 --momentum 0.95 {M}",
    "lg5_int4_h3": f"--arms lg5_fp4_h3 --seeds 0 --lr 0.03 --momentum 0.9 {M}",
    "lg5_int4_h1.5": f"--arms lg5_fp4_h1.5 --seeds 0 --lr 0.03 --momentum 0.9 {M}",
    "lg5_int4_lr02": f"--arms lg5_fp4_h2 --seeds 0 --lr 0.02 --momentum 0.9 {M}",
    "lg6_int4_m095": f"--arms lg6_fp4_h2 --seeds 0 --lr 0.03 --momentum 0.95 {M}",
}
procs = {k: subprocess.Popen(f"{B}{a} > /content/q6/{k}.log 2>&1", shell=True) for k, a in runs.items()}
while any(p.poll() is None for p in procs.values()):
    time.sleep(20)
for k in runs:
    sh(f"grep -E 'momentum:|seed=|Error|error' /content/q6/{k}.log | sed 's/^/{k}: /'", show=4)
print("d512 done", round((time.time() - t0) / 60, 1), "min", flush=True)
# ---- 3) 1.21B TinyStories: lg6 + fused int4 momentum, SAME config as the 11-01 runs (5000-step cosine), stop at 1000
dp.wait()
sh("tail -3 /content/q6/prep.log", show=3)
sh(f"cd {R} && PYTHONPATH=. python probes/train_1b.py --arm lg6_fp4_h2 --adam-lr 1e-4 --lr 0.01 --momentum 0.9 --momentum-dtype int4 "
   f"--steps 5000 --stop-after 1000 --data-dir /content/data --out /content/q6/run1b --ckpt-every 100000 > /content/q6/run1b.log 2>&1", timeout=60 * 60)
sh("grep -E 'EVAL|step (50|100|250|500|750|1000)/|optim|Error|error' /content/q6/run1b.log | cut -c1-220", show=30)
print("ALL_DONE", round((time.time() - t0) / 60, 1), "min", flush=True)
