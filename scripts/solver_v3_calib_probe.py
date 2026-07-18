"""Calibration probe for the layerwise witness: runs in a subprocess so the page cache
and allocator pressure of the model die with the process. Saves {path: (W fp32, H fp32)}
for the sampled layers to OUT (torch.save)."""
import gc
import os

import torch
try:
    from datasets import load_dataset
except ImportError:  # lean fallback: direct parquet fetch (fastparquet), same corpus
    load_dataset = None
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_native.donor.ptq import collect_hessians

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
SEQ = int(os.environ.get("SEQ", "256"))
BATCH = int(os.environ.get("BATCH", "4"))
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "4"))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,23").split(",")]
OUT = os.environ.get("OUT", "/tmp/solver_v3_calib.pt")

torch.manual_seed(0)
torch.set_num_threads(os.cpu_count())
print(f"calib probe: {MODEL} layers={LAYERS} {CALIB_BATCHES}x{BATCH}x{SEQ}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if load_dataset is not None:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(t for t in ds["train"]["text"] if t.strip())
else:
    import fastparquet
    parquet = os.environ.get(
        "WIKITEXT_PARQUET", "/tmp/wikitext2_train.txt")
    if not os.path.exists(parquet):
        import urllib.request
        url = ("https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/"
               "wikitext-2-raw-v1/train-00000-of-00001.parquet")
        urllib.request.urlretrieve(url, parquet)
    pf = fastparquet.ParquetFile(parquet)
    text = "\n\n".join(t for t in pf.to_pandas()["text"].tolist() if t.strip())
    del pf
    ds = None
ids = tokenizer(text, return_tensors="pt").input_ids[0]
n_seq = ids.numel() // SEQ
ids = ids[: n_seq * SEQ].view(n_seq, SEQ)
calib = [ids[i: i + BATCH] for i in range(0, CALIB_BATCHES * BATCH, BATCH)]
del ds, text, tokenizer
gc.collect()

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()

targets, weights = [], {}
for li in LAYERS:
    for name, mod in model.model.layers[li].named_modules():
        if isinstance(mod, torch.nn.Linear):
            path = f"model.layers.{li}.{name}"
            targets.append(path)
            weights[path] = mod.weight.detach().clone()        # keep bf16: cast in the witness
print(f"{len(targets)} target linears; collecting hessians...", flush=True)
hessians = collect_hessians(model, targets, calib)
weights = {p: w.to(torch.float32) for p, w in weights.items()}
torch.save({"W": weights, "H": hessians, "layers": LAYERS,
            "model": MODEL, "seq": SEQ, "batch": BATCH, "calib_batches": CALIB_BATCHES}, OUT)
print(f"saved -> {OUT}", flush=True)
