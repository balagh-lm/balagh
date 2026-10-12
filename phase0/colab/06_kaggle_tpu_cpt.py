# Balagh-nano continued pretraining on Kaggle TPU v5e-8 (PyTorch/XLA, SPMD FSDP), one 9-hour session at a time.
#
# Design for Kaggle's limits: every session (batch "Save & Run All") resumes from the newest checkpoint on the
# Hub (balagh-lm/balagh-nano-ckpt), trains until a wall-clock guard (default 8h30m) or the step budget, saves
# bf16 weights + optimizer state to /kaggle/working and uploads to the Hub, then exits cleanly.
# Data: packed uint32 shards (2048 tokens) from a Kaggle dataset (fast) or streamed from the Hub (fallback).
# SMOKE=1 runs 50 steps on a tiny slice, saves and uploads a checkpoint, then exits: run that first.
#
# Kaggle Secrets needed: HF_TOKEN (write). Accelerator: TPU v5e-8. Internet: on.
# Base: google/gemma-4-E2B (text-only load). Fallback base: Qwen/Qwen3.5-2B-Base (set BASE).

import os, sys, time, json, math, glob, gc
SMOKE = os.environ.get("SMOKE", "0") == "1"
BASE = os.environ.get("BASE", "google/gemma-4-E2B")
CKPT_REPO = os.environ.get("CKPT_REPO", "balagh-lm/balagh-nano-ckpt")
TOKENS_REPO = os.environ.get("TOKENS_REPO", "balagh-lm/tokens-gemma4-v1")
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input/balagh-tokens-gemma4-v1")   # Kaggle dataset mount
SEQ = 2048
GLOBAL_BATCH = int(os.environ.get("GLOBAL_BATCH", 256))          # 256 x 2048 = 0.5M tokens/step
PER_CHIP = int(os.environ.get("PER_CHIP", 4))                     # micro-batch per chip; grad accumulation fills GLOBAL_BATCH
LR, MIN_LR, WARMUP = float(os.environ.get("LR", 2e-5)), 2e-6, int(os.environ.get("WARMUP", 200))
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 20000))           # 20k x 0.5M = 10B tokens
WALL_SECONDS = int(os.environ.get("WALL_SECONDS", 8 * 3600 + 30 * 60))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 1000))
REPLAY_SOURCES = {"en_edu", "code", "math"}
if SMOKE: TOTAL_STEPS, SAVE_EVERY, WALL_SECONDS, GLOBAL_BATCH = 50, 25, 20 * 60, 32

def secret(k):
    try:
        from kaggle_secrets import UserSecretsClient; return UserSecretsClient().get_secret(k)
    except Exception: return os.environ.get(k)
HF = secret("HF_TOKEN")
from huggingface_hub import HfApi, snapshot_download, list_repo_files
api = HfApi(token=HF); api.create_repo(CKPT_REPO, repo_type="model", exist_ok=True)

# ---------------------------------------------------------------- TPU / XLA setup
import torch, numpy as np
import torch_xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
xr.use_spmd()
dev = torch_xla.device(); n_dev = xr.global_runtime_device_count()
mesh = Mesh(np.arange(n_dev), (n_dev, 1), ("fsdp", "model"))
print("devices:", n_dev, "mesh:", mesh.shape(), flush=True)

# ---------------------------------------------------------------- model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
tok = AutoTokenizer.from_pretrained(BASE, token=HF)
t0 = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, token=HF)
except Exception as e:
    print("AutoModelForCausalLM failed, trying ImageTextToText text tower:", str(e)[:120])
    from transformers import AutoModelForImageTextToText
    full = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.bfloat16, token=HF)
    model = full.language_model if hasattr(full, "language_model") else full
model.config.use_cache = False
if hasattr(model, "gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
n_params = sum(p.numel() for p in model.parameters()); print(f"{BASE}: {n_params/1e9:.2f}B params, loaded in {time.time()-t0:.0f}s", flush=True)
# freeze per-layer embeddings (Gemma 4 E-series) to keep optimizer state small; everything else trains
frozen = 0
for name, p in model.named_parameters():
    if "per_layer" in name or "embed_tokens_per_layer" in name: p.requires_grad_(False); frozen += p.numel()
print(f"frozen per-layer embedding params: {frozen/1e9:.2f}B", flush=True)
model.to(dev)
# FSDP-style sharding of every large parameter along the fsdp axis
for name, p in model.named_parameters():
    if p.dim() >= 2 and p.numel() > 1_000_000:
        xs.mark_sharding(p, mesh, ("fsdp", None) if p.shape[0] % n_dev == 0 else (None, "fsdp") if p.shape[1] % n_dev == 0 else (None, None))

params = [p for p in model.parameters() if p.requires_grad]
try:
    import bitsandbytes as bnb; opt = bnb.optim.AdamW8bit(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.1); print("optimizer: AdamW8bit")
except Exception:
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.1, foreach=False); print("optimizer: AdamW (fp32 state)")

# ---------------------------------------------------------------- checkpoint resume
def latest_step():
    files = list_repo_files(CKPT_REPO, token=HF)
    steps = sorted({int(f.split("/")[0].split("-")[1]) for f in files if f.startswith("step-")})
    return steps[-1] if steps else 0
start_step = latest_step()
if start_step:
    local = snapshot_download(CKPT_REPO, allow_patterns=[f"step-{start_step}/*"], token=HF, local_dir="/kaggle/working/resume")
    sd = torch.load(f"{local}/step-{start_step}/model.pt", map_location="cpu")
    model.load_state_dict(sd, strict=False); del sd
    try: opt.load_state_dict(torch.load(f"{local}/step-{start_step}/optim.pt", map_location="cpu"))
    except Exception as e: print("optimizer state not restored:", str(e)[:100])
    print(f"resumed from step {start_step}", flush=True)
step = start_step

# ---------------------------------------------------------------- data
def shard_files():
    if os.path.isdir(DATA_DIR):
        fs = sorted(glob.glob(f"{DATA_DIR}/**/*.npy", recursive=True))
        if fs: return [("local", f) for f in fs]
    fs = [f for f in list_repo_files(TOKENS_REPO, repo_type="dataset", token=HF) if f.endswith(".npy")]
    return [("hub", f) for f in sorted(fs)]
files = shard_files(); print(f"{len(files)} token shards ({files[0][0]})", flush=True)
rng = np.random.default_rng(1234 + step)   # per-session shuffle; sources are interleaved by weight below
by_src = {}
for kind, f in files: by_src.setdefault(f.split("/")[-2], []).append((kind, f))
weights = {s: len(v) for s, v in by_src.items()}
order = []
total = sum(weights.values())
for s, v in by_src.items():
    rng.shuffle(v)
for i in range(total):
    s = rng.choice(list(weights), p=np.array(list(weights.values())) / total)
    if by_src[s]: order.append(by_src[s].pop())
def load_shard(kind, f):
    if kind == "local": return np.load(f, mmap_mode="r")
    from huggingface_hub import hf_hub_download
    return np.load(hf_hub_download(TOKENS_REPO, f, repo_type="dataset", token=HF))
# consume shards from the position implied by step (each shard = SHARD_SEQS sequences)
seqs_per_step = GLOBAL_BATCH
seq_cursor = step * seqs_per_step
def batches():
    pos = 0
    for kind, f in order:
        arr = load_shard(kind, f); n = arr.shape[0]
        if pos + n <= seq_cursor: pos += n; continue
        start = max(0, seq_cursor - pos)
        for i in range(start, n - seqs_per_step + 1, seqs_per_step):
            yield torch.from_numpy(np.asarray(arr[i:i + seqs_per_step]).astype(np.int64))
        pos += n
    while True:   # wrap around (second epoch of the shuffled order)
        for kind, f in order:
            arr = load_shard(kind, f)
            for i in range(0, arr.shape[0] - seqs_per_step + 1, seqs_per_step):
                yield torch.from_numpy(np.asarray(arr[i:i + seqs_per_step]).astype(np.int64))

# ---------------------------------------------------------------- train
def lr_at(s):
    if s < WARMUP: return LR * s / max(1, WARMUP)
    p = min(1.0, (s - WARMUP) / max(1, TOTAL_STEPS - WARMUP)); return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * p))
micro = PER_CHIP * n_dev; accum = max(1, GLOBAL_BATCH // micro)
print(f"global batch {GLOBAL_BATCH} seqs = {GLOBAL_BATCH*SEQ/1e6:.2f}M tokens/step; micro {micro} x accum {accum}; steps {step}->{TOTAL_STEPS}", flush=True)
def save(step, final=False):
    out = f"/kaggle/working/step-{step}"; os.makedirs(out, exist_ok=True)
    sd = {k: v.to("cpu") for k, v in model.state_dict().items()}
    torch.save(sd, f"{out}/model.pt"); del sd
    torch.save(opt.state_dict(), f"{out}/optim.pt")
    json.dump({"step": step, "base": BASE, "tokens_seen": step * GLOBAL_BATCH * SEQ, "lr": lr_at(step), "final": final}, open(f"{out}/meta.json", "w"))
    t1 = time.time(); api.upload_folder(folder_path=out, path_in_repo=f"step-{step}", repo_id=CKPT_REPO, commit_message=f"step {step}{' final' if final else ''}")
    print(f"checkpoint step {step} uploaded in {time.time()-t1:.0f}s", flush=True)
    import shutil; shutil.rmtree(out, ignore_errors=True)
model.train(); t_start = time.time(); it = batches(); losses = []
while step < TOTAL_STEPS:
    for g in opt.param_groups: g["lr"] = lr_at(step)
    opt.zero_grad(set_to_none=True); loss_acc = 0.0
    for a in range(accum):
        toks = next(it)
        x = toks[:micro] if toks.shape[0] >= micro else toks
        x = x.to(dev); xs.mark_sharding(x, mesh, ("fsdp", None))
        out = model(input_ids=x, labels=x)
        (out.loss / accum).backward(); loss_acc += float(out.loss.detach().cpu()) / accum if (a == accum - 1 and step % 10 == 0) else 0.0
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); xm.mark_step(); step += 1
    if step % 10 == 0:
        losses.append(loss_acc); el = time.time() - t_start
        print(f"step {step} loss {loss_acc:.3f} lr {lr_at(step):.2e} {(step-start_step)*GLOBAL_BATCH*SEQ/el:,.0f} tok/s elapsed {el/60:.0f}m", flush=True)
    if step % SAVE_EVERY == 0: save(step)
    if time.time() - t_start > WALL_SECONDS:
        print("wall-clock guard reached; saving and exiting", flush=True); save(step); break
if step >= TOTAL_STEPS: save(step, final=True); print("TRAINING COMPLETE", flush=True)
json.dump({"session_end_step": step, "losses_last": losses[-20:], "seconds": int(time.time() - t_start)}, open("/kaggle/working/session_summary.json", "w"))
print("SESSION DONE at step", step)
