# R4 arm-1: cached-KD recovery gemma-12b ternary — COUNTER_LR_START halved 0.002 -> 0.001
# (ranked cause #1 in results/GEMMA12B_CACHED_KD.md: lr too hot for 12B group+dec4)
import os, subprocess, sys
os.system("nvidia-smi -L")
assert os.system(f"{sys.executable} -m pip install -q -U transformers") == 0
MODEL_DIR = REPO_SRC = MIX = STATE = CACHE = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "model.safetensors" in files and "config.json" in files and "gemma-4" in root:
        MODEL_DIR = root
    if "kd_cached_recovery.py" in files and root.endswith("scripts") and "mn-g12" not in root:
        REPO_SRC = os.path.dirname(root)
    if "manifest.json" in files and "train_en.bin" in files and root.endswith("mix_gemma"):
        MIX = root
    if "block_0000.pt" in files and "manifest.json" in files:
        STATE = root
    if "cache_manifest.json" in files:
        CACHE = root
assert all((MODEL_DIR, REPO_SRC, MIX, STATE, CACHE)), (MODEL_DIR, REPO_SRC, MIX, STATE, CACHE)
os.makedirs("/kaggle/working/repo", exist_ok=True)
assert os.system(f"cp -r {REPO_SRC}/. /kaggle/working/repo/") == 0
REPO = "/kaggle/working/repo"
print("model:", MODEL_DIR, "\nmix:", MIX, "\nstate:", STATE, "\ncache:", CACHE, flush=True)
ENV = dict(os.environ, PYTHONPATH=f"{REPO}/src",
    MODEL=MODEL_DIR, STATE_DIR=STATE, DATA_DIR=MIX, CACHE=CACHE,
    CKPT_DIR="/kaggle/working/ckpt", CKPT_TMP="/kaggle/tmp",
    STEPS="750", KD_T="2.0", CE_ALPHA="0.3",
    COUNTER_LR_START="0.001", COUNTER_LR_END="0.0001", FP_LR="0.0001",
    STATS_SCOPE="group", DECIMATION="4",
    GRAD_CKPT="0", FREEZE_EMBED="1", SPLIT_GPUS="1", EVAL_AT_START="1",
    EVAL_EVERY="250", EVAL_MAX_TOKENS="12000", LOG_EVERY="25",
    DEVICE="cuda", PYTORCH_ALLOC_CONF="expandable_segments:True",
    HF_HUB_DISABLE_PROGRESS_BARS="1")
try:
    r = subprocess.run([sys.executable, "scripts/kd_cached_recovery.py"],
                       env=ENV, cwd=REPO, timeout=3600 * 11)
    code = r.returncode
except subprocess.TimeoutExpired:
    print("recovery timed out at 11h -- keeping the periodic checkpoints", flush=True)
    code = 0
print(f"recovery exit {code}")
sys.exit(code)
