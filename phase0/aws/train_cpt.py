# Balagh-nano continued pretraining of Qwen/Qwen3.5-2B-Base on one NVIDIA GPU (torchrun DDP also works).
#
# Built for a single AWS p5.4xlarge (1x H100 80GB) on Spot: resumes from the newest local or S3 checkpoint,
# checkpoints every SAVE_EVERY_MIN minutes and immediately on a Spot interruption notice, SIGTERM or a stop file,
# pushes bf16 Hugging Face-format weights at every MILESTONE_TOKENS, tracks held-out loss per language, stops at
# MAX_HOURS, and never writes a checkpoint after a non-finite step. Data = packed uint32 shards from
# balagh-lm/tokens-qwen35-v1 (05_tokenize_export).
#
# Precision: fp32 master weights + bf16 autocast (Gated DeltaNet A_log / norm weights are stored fp32).
# Kernels: flash-linear-attention + causal-conv1d (PyTorch fallback is ~2x slower); REQUIRE_KERNELS=1 aborts
# without them, USE_HUB_KERNELS=1 loads them from the Hugging Face Hub instead. Loss: chunked cross-entropy so
# the 248k-vocab logits never materialise for a whole micro-batch. Packing: documents concatenated with EOS.
#
# Exit codes (bootstrap.sh relies on them): 0 done and persisted; 3 stopped (Spot/signal/stop file), checkpoint
# saved; 4 MAX_HOURS reached, checkpoint saved; 5 final artefacts could not be persisted; other = crash.
#
# Env (defaults): BASE [Qwen/Qwen3.5-2B-Base], DATA_DIR [./tokens], RUN [balagh-nano-v0], OUT_DIR [./runs], TOTAL_TOKENS [8e9],
# LR [6e-5], MIN_LR_RATIO [0.1], WARMUP_FRAC [0.01], SCHEDULE [cosine|constant], BATCH_SEQS [256], MICRO_SEQS [8],
# SAVE_EVERY_MIN [30], EVAL_EVERY [1000], EVAL_SEQS [32], MILESTONE_TOKENS [1e9], MAX_HOURS [1e9], S3_URI, HF_CKPT_REPO
# [balagh-lm/balagh-nano-ckpt], HF_PUSH [1], SEED [1234], EXCLUDE_SOURCES, REQUIRE_KERNELS [1], USE_HUB_KERNELS [0],
# GRAD_CKPT [1], EXPECT_RESUME [0], ALLOW_CONFIG_CHANGE [0], SMOKE [0].

import os, sys, json, time, math, glob, signal, shutil, threading, subprocess, urllib.request, urllib.error, contextlib, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

E = os.environ.get
BASE = E("BASE", "Qwen/Qwen3.5-2B-Base")
DATA_DIR = E("DATA_DIR", "./tokens")
RUN = E("RUN", "balagh-nano-v0")
OUT_DIR = os.path.join(E("OUT_DIR", "./runs"), RUN)
SMOKE = E("SMOKE", "0") == "1"
TOTAL_TOKENS = float(E("TOTAL_TOKENS", 8e9))
LR = float(E("LR", 6e-5)); MIN_LR_RATIO = float(E("MIN_LR_RATIO", 0.1)); WARMUP_FRAC = float(E("WARMUP_FRAC", 0.01))
SCHEDULE = E("SCHEDULE", "cosine")
BATCH_SEQS = int(E("BATCH_SEQS", 256)); MICRO_SEQS = int(E("MICRO_SEQS", 8))
SAVE_EVERY_MIN = float(E("SAVE_EVERY_MIN", 30)); EVAL_EVERY = int(E("EVAL_EVERY", 1000)); EVAL_SEQS = int(E("EVAL_SEQS", 32))
MILESTONE_TOKENS = float(E("MILESTONE_TOKENS", 1e9))
MAX_HOURS = float(E("MAX_HOURS", 1e9))
S3_URI = (E("S3_URI") or "").rstrip("/")
HF_CKPT_REPO = E("HF_CKPT_REPO", "balagh-lm/balagh-nano-ckpt")
HF_PUSH = E("HF_PUSH", "1") == "1"
SEED = int(E("SEED", 1234))
EXCLUDE = set(filter(None, E("EXCLUDE_SOURCES", "").split(",")))
REQUIRE_KERNELS = E("REQUIRE_KERNELS", "1") == "1"
USE_HUB_KERNELS = E("USE_HUB_KERNELS", "0") == "1"
GRAD_CKPT = E("GRAD_CKPT", "1") == "1"
EXPECT_RESUME = E("EXPECT_RESUME", "0") == "1"
ALLOW_CONFIG_CHANGE = E("ALLOW_CONFIG_CHANGE", "0") == "1"
CE_CHUNK = int(E("CE_CHUNK", 4096))

T0 = time.time()
WORLD = int(E("WORLD_SIZE", 1)); RANK = int(E("RANK", 0)); LOCAL_RANK = int(E("LOCAL_RANK", 0))
if WORLD > 1:
    import datetime
    torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
torch.cuda.set_device(LOCAL_RANK); DEV = torch.device("cuda", LOCAL_RANK)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
IS_MAIN = RANK == 0
os.makedirs(OUT_DIR, exist_ok=True)
METRICS = os.path.join(OUT_DIR, "metrics.jsonl")

def log(*a, **k):
    if IS_MAIN: print(time.strftime("%H:%M:%S"), *a, flush=True, **k)

def metric(obj):
    if not IS_MAIN: return
    with open(METRICS, "a") as f: f.write(json.dumps({"t": round(time.time() - T0, 1), **obj}) + "\n")

def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode: raise RuntimeError(f"{cmd}: {r.stderr[-500:]}")
    return r

def barrier():
    if WORLD > 1: torch.distributed.barrier()

# ------------------------------------------------------------------ interruption handling
STOP = threading.Event(); STOP_REASON = []
def _on_signal(sig, frame):
    STOP_REASON.append(f"signal {sig}"); STOP.set()
signal.signal(signal.SIGTERM, _on_signal); signal.signal(signal.SIGINT, _on_signal)

def spot_watcher():
    """Poll IMDSv2 every 5 s for a Spot interruption notice (404 until one is scheduled). Gives up only if IMDS
    never answered (not on EC2); transient errors after that are retried."""
    reached = False; fails = 0
    while not STOP.is_set():
        try:
            tok = urllib.request.urlopen(urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                  headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"}), timeout=2).read().decode()
            reached = True
            try:
                body = urllib.request.urlopen(urllib.request.Request("http://169.254.169.254/latest/meta-data/spot/instance-action",
                       headers={"X-aws-ec2-metadata-token": tok}), timeout=2).read().decode()
                STOP_REASON.append("spot interruption: " + body); STOP.set(); return
            except urllib.error.HTTPError:
                pass
            fails = 0
        except Exception:
            fails += 1
            if not reached and fails >= 3: return   # not on EC2 or IMDS disabled
        time.sleep(5)
if IS_MAIN: threading.Thread(target=spot_watcher, daemon=True).start()

def stop_file():
    return os.path.exists(os.path.join(OUT_DIR, "STOP_REQUESTED"))

# ------------------------------------------------------------------ data
man = json.load(open(os.path.join(DATA_DIR, "manifest.json")))
SEQ = int(man["seq_len"])
shard_files = sorted(f for f in glob.glob(os.path.join(DATA_DIR, "*", "shard-*.npy")) if os.path.basename(os.path.dirname(f)) not in EXCLUDE | {"val"})
assert shard_files, f"no shards under {DATA_DIR}"
_mm = {}
def shard(i):
    if i not in _mm: _mm[i] = np.load(shard_files[i], mmap_mode="r")
    return _mm[i]
rows = np.array([shard(i).shape[0] for i in range(len(shard_files))], dtype=np.int64)
for i in range(len(shard_files)): assert shard(i).ndim == 2 and shard(i).shape[1] == SEQ and shard(i).dtype == np.uint32, shard_files[i]
FILE_OF = np.repeat(np.arange(len(shard_files), dtype=np.int32), rows)
ROW_OF = np.concatenate([np.arange(r, dtype=np.int32) for r in rows])
N_SEQS = len(FILE_OF)
DATA_FP = hashlib.sha1(json.dumps([[os.path.relpath(f, DATA_DIR).replace("\\", "/"), int(r)] for f, r in zip(shard_files, rows)]).encode()).hexdigest()[:16]
by_src = {}
for i, f in enumerate(shard_files): by_src[os.path.basename(os.path.dirname(f))] = by_src.get(os.path.basename(os.path.dirname(f)), 0) + int(rows[i])
log(f"data: {len(shard_files)} shards, {N_SEQS:,} sequences x {SEQ} = {N_SEQS * SEQ / 1e9:.2f}B tokens (fingerprint {DATA_FP}); by source (B tokens):",
    {k: round(v * SEQ / 1e9, 3) for k, v in sorted(by_src.items())})
_perm = {}
def perm(epoch):
    if epoch not in _perm:
        _perm.clear(); _perm[epoch] = np.random.default_rng(SEED + epoch).permutation(N_SEQS)
    return _perm[epoch]

def batch_indices(step):
    """Global batch for an optimizer step: BATCH_SEQS sequences of a seeded permutation, so resuming at a step
    needs no data-loader state. Wraps into a fresh permutation per epoch."""
    start = step * BATCH_SEQS; out = []
    while len(out) < BATCH_SEQS:
        ep, off = divmod(start + len(out), N_SEQS)
        take = min(BATCH_SEQS - len(out), N_SEQS - off)
        out.extend(perm(ep)[off:off + take].tolist())
    return out

def load_seqs(idx):
    return torch.from_numpy(np.stack([shard(int(FILE_OF[i]))[int(ROW_OF[i])] for i in idx]).astype(np.int64))

val_sets = {}
for f in sorted(glob.glob(os.path.join(DATA_DIR, "val", "*.npy"))):
    v = np.load(f)[:EVAL_SEQS]
    if len(v): val_sets[os.path.basename(f)[:-4]] = torch.from_numpy(v.astype(np.int64))
log("validation sets:", {k: v.shape[0] for k, v in val_sets.items()})
assert val_sets, "no held-out sets under DATA_DIR/val: per-language loss and the pilot report depend on them"
missing_val = sorted(set(by_src) - set(val_sets))
if missing_val: log("WARNING: no held-out set for", missing_val)

if SMOKE: TOTAL_TOKENS = min(TOTAL_TOKENS, 40 * BATCH_SEQS * SEQ); EVAL_EVERY = 10; SAVE_EVERY_MIN = 3; MILESTONE_TOKENS = 1e18
TOTAL_STEPS = max(1, int(TOTAL_TOKENS // (BATCH_SEQS * SEQ)))
WARMUP = max(1, int(TOTAL_STEPS * WARMUP_FRAC))
def lr_at(step):
    if step < WARMUP: return LR * (step + 1) / WARMUP
    if SCHEDULE == "constant": return LR
    p = min(1.0, (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP))
    return LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * p)))

# ------------------------------------------------------------------ model
def kernels_status():
    st = {}
    for mod in ["fla", "causal_conv1d", "flash_attn"]:
        try: m = __import__(mod); st[mod] = getattr(m, "__version__", "ok")
        except Exception: st[mod] = None
    return st
ks = kernels_status(); log("kernels:", ks, "| hub kernels" if USE_HUB_KERNELS else "")
if REQUIRE_KERNELS and not (ks["fla"] and (ks["causal_conv1d"] or USE_HUB_KERNELS)):
    raise SystemExit("flash-linear-attention and causal-conv1d (or USE_HUB_KERNELS=1) are required; REQUIRE_KERNELS=0 runs the ~2x slower PyTorch fallback")

from transformers import AutoModelForCausalLM, AutoTokenizer
HF_TOKEN = E("HF_TOKEN")
tok = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, token=HF_TOKEN, attn_implementation=E("ATTN", "sdpa"),
                                             **({"use_kernels": True} if USE_HUB_KERNELS else {}))
log("model class:", type(model).__name__)
model.config.use_cache = False
if GRAD_CKPT: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.to(DEV)
N_PARAMS = sum(p.numel() for p in model.parameters())
N_EMB = model.get_input_embeddings().weight.numel()
TIED = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
FLOPS_PER_TOKEN = 6 * (N_PARAMS - N_EMB + (N_EMB if TIED else 0))   # matmul FLOPs incl. the LM head, excl. the lookup
log(f"params {N_PARAMS/1e9:.3f}B (embeddings {N_EMB/1e6:.0f}M, tied={TIED}); ~{FLOPS_PER_TOKEN/1e9:.1f} GFLOP/token; grad checkpointing {GRAD_CKPT}")

class LossModel(nn.Module):
    """Decoder hidden states, then cross-entropy in recomputed chunks: peak memory holds CE_CHUNK x vocab logits."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.m.model(input_ids=ids).last_hidden_state
        W = self.m.get_output_embeddings().weight
        h = h[:, :-1].reshape(-1, h.shape[-1]); y = ids[:, 1:].reshape(-1)
        total = h.new_zeros((), dtype=torch.float32)
        def ce(hc, yc):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = F.linear(hc, W)
            return F.cross_entropy(logits.float(), yc, reduction="sum")
        for i in range(0, h.shape[0], CE_CHUNK):
            total = total + torch.utils.checkpoint.checkpoint(ce, h[i:i + CE_CHUNK], y[i:i + CE_CHUNK], use_reentrant=False)
        return total / y.numel()

lm = LossModel(model)
if WORLD > 1: lm = nn.parallel.DistributedDataParallel(lm, device_ids=[LOCAL_RANK])
core = lm.module if WORLD > 1 else lm

decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    (no_decay if p.ndim < 2 or "norm" in n.lower() or "A_log" in n or "dt_bias" in n else decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.95), eps=1e-8, fused=True)
CONFIG = {"run": RUN, "base": BASE, "seed": SEED, "total_steps": TOTAL_STEPS, "batch_seqs": BATCH_SEQS, "seq": SEQ, "lr": LR,
          "schedule": SCHEDULE, "n_seqs": int(N_SEQS), "data_fp": DATA_FP}

# ------------------------------------------------------------------ checkpoints
CK = os.path.join(OUT_DIR, "ckpt"); os.makedirs(CK, exist_ok=True)
_up = {"thread": None, "ok": True}

def s3_prune(keep=2):
    pre = f"{S3_URI}/{RUN}/ckpt/"
    r = sh(f"aws s3 ls {pre}")
    if r.returncode: return
    names = sorted(l.split()[-1] for l in r.stdout.splitlines() if l.split() and l.split()[-1].startswith("step-") and l.split()[-1].endswith(".pt"))
    for n in names[:-keep]: sh(f"aws s3 rm --only-show-errors {pre}{n}")

def s3_publish(local_pt, meta):
    """Upload the checkpoint, then (only if that worked) a latest.json naming it, then prune old checkpoints.
    latest.json in S3 therefore never points at an object that is not there."""
    t = time.time(); size = os.path.getsize(local_pt) / 2**30
    r = sh(f"aws s3 cp --only-show-errors {local_pt} {S3_URI}/{RUN}/ckpt/{os.path.basename(local_pt)}")
    if r.returncode: log("   S3 checkpoint upload FAILED:", r.stderr[-300:]); _up["ok"] = False; return
    mp = local_pt + ".latest.json"; json.dump(meta, open(mp, "w"))
    r = sh(f"aws s3 cp --only-show-errors {mp} {S3_URI}/{RUN}/ckpt/latest.json"); os.remove(mp)
    if r.returncode: log("   S3 latest.json upload FAILED:", r.stderr[-300:]); _up["ok"] = False; return
    _up["ok"] = True; s3_prune(2)
    log(f"   S3: {os.path.basename(local_pt)} ({size:.1f} GiB) in {time.time()-t:.0f}s")

def save_ckpt(step, reason, block_upload=False):
    if IS_MAIN:
        t = time.time(); name = f"step-{step:07d}.pt"; path = os.path.join(CK, name)
        if not os.path.exists(path):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": CONFIG}, path + ".tmp")
            os.replace(path + ".tmp", path)
        th = _up["thread"]
        if th and th.is_alive(): th.join()   # the previous upload publishes its own pointer before we move ours
        meta = {"file": name, "step": step, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason}
        json.dump(meta, open(os.path.join(CK, "latest.json.tmp"), "w")); os.replace(os.path.join(CK, "latest.json.tmp"), os.path.join(CK, "latest.json"))
        for old in sorted(glob.glob(os.path.join(CK, "step-*.pt")))[:-2]: os.remove(old)
        log(f"checkpoint step {step} ({reason}) in {time.time()-t:.0f}s")
        if S3_URI:
            th = threading.Thread(target=s3_publish, args=(path, meta)); th.start(); _up["thread"] = th
            if block_upload:
                th.join()
                if not _up["ok"]: raise RuntimeError("checkpoint upload to S3 failed")
    barrier()

def s3_exists(key):
    bucket, *pre = S3_URI[5:].split("/")
    r = sh(f"aws s3api head-object --bucket {bucket} --key {'/'.join(pre + [key])}")
    if r.returncode == 0: return True
    if "404" in r.stderr or "Not Found" in r.stderr: return False
    raise RuntimeError(f"cannot check s3 {key}: {r.stderr[-200:]}")

def find_resume():
    """Newest usable checkpoint, local first, then S3 (latest.json, else the newest step-*.pt listed)."""
    local = os.path.join(CK, "latest.json")
    if os.path.exists(local):
        p = os.path.join(CK, json.load(open(local))["file"])
        if os.path.exists(p): return p
    if not S3_URI: return None
    for attempt in range(5):
        try:
            exists = s3_exists(f"{RUN}/ckpt/latest.json"); break
        except RuntimeError as e: log("   ", e); time.sleep(10 * (attempt + 1))
    else: raise SystemExit("S3 state unreadable: refusing to restart from step 0")
    cands = []
    if exists and sh(f"aws s3 cp --only-show-errors {S3_URI}/{RUN}/ckpt/latest.json {local}").returncode == 0 and os.path.exists(local):
        cands.append(json.load(open(local))["file"])
    r = sh(f"aws s3 ls {S3_URI}/{RUN}/ckpt/")
    cands += sorted((l.split()[-1] for l in r.stdout.splitlines() if l.split() and l.split()[-1].startswith("step-") and l.split()[-1].endswith(".pt")), reverse=True)
    for name in dict.fromkeys(cands):
        p = os.path.join(CK, name); log("downloading checkpoint", name, "from S3 ...")
        if sh(f"aws s3 cp --only-show-errors {S3_URI}/{RUN}/ckpt/{name} {p}").returncode == 0 and os.path.exists(p): return p
        log("   could not fetch", name, "- trying an older one")
    return None

def trim_metrics(step):
    """Drop metric lines from steps that will be replayed after a resume (the S3 copy can be ahead)."""
    if not (IS_MAIN and os.path.exists(METRICS)): return
    keep = []
    for line in open(METRICS):
        try: m = json.loads(line)
        except Exception: continue
        if "step" not in m or m["step"] <= step: keep.append(json.dumps(m))
    open(METRICS + ".tmp", "w").write("\n".join(keep) + ("\n" if keep else "")); os.replace(METRICS + ".tmp", METRICS)

step = 0
rp = find_resume() if IS_MAIN else None
if WORLD > 1:
    obj = [rp]; torch.distributed.broadcast_object_list(obj, 0); rp = obj[0]
if rp:
    st = torch.load(rp, map_location="cpu", weights_only=False)
    cfg = st.get("config", {})
    diff = {k: (cfg.get(k), v) for k, v in CONFIG.items() if k in cfg and cfg.get(k) != v}
    if diff and not ALLOW_CONFIG_CHANGE:
        raise SystemExit(f"checkpoint config differs from this run: {diff} (ALLOW_CONFIG_CHANGE=1 to continue anyway)")
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = int(st["step"])
    del st; torch.cuda.empty_cache(); trim_metrics(step)
    log(f"resumed from {os.path.basename(rp)} at step {step}" + (f" (config changes allowed: {diff})" if diff else ""))
elif EXPECT_RESUME:
    raise SystemExit("EXPECT_RESUME=1 but no checkpoint was found: refusing to restart from step 0")
if WORLD > 1:
    for p in model.parameters(): torch.distributed.broadcast(p.data, 0)

def push_milestone(step, tokens_seen, final=False):
    """bf16 HF-format text model -> HF_CKPT_REPO/<RUN>/<tag>/ (about 3.8 GB). Returns True when pushed."""
    if not (IS_MAIN and HF_PUSH): return True
    from huggingface_hub import HfApi
    tag = "final" if final else f"tokens-{tokens_seen/1e9:.1f}B"
    d = os.path.join(OUT_DIR, "export", tag); os.makedirs(d, exist_ok=True)
    sd = {k: v.detach().to("cpu", torch.bfloat16) for k, v in model.state_dict().items()}
    if TIED: sd.pop("lm_head.weight", None)
    model.save_pretrained(d, state_dict=sd, safe_serialization=True); tok.save_pretrained(d)
    cp = os.path.join(d, "config.json"); c = json.load(open(cp))   # weights are bf16 and inference wants the cache
    c["dtype"] = "bfloat16"
    if "torch_dtype" in c: c["torch_dtype"] = "bfloat16"
    c["use_cache"] = True
    if isinstance(c.get("text_config"), dict): c["text_config"]["use_cache"] = True
    json.dump(c, open(cp, "w"), indent=2)
    json.dump({**CONFIG, "step": step, "tokens_seen": tokens_seen}, open(os.path.join(d, "balagh_meta.json"), "w"), indent=1)
    if os.path.exists(METRICS): shutil.copyfile(METRICS, os.path.join(d, "metrics.jsonl"))
    api = HfApi(token=HF_TOKEN)
    for attempt in range(4 if final else 2):
        try:
            api.create_repo(HF_CKPT_REPO, private=True, exist_ok=True)
            api.upload_folder(folder_path=d, repo_id=HF_CKPT_REPO, path_in_repo=f"{RUN}/{tag}", commit_message=f"{RUN} {tag} (step {step})")
            log(f"pushed {RUN}/{tag} to {HF_CKPT_REPO}"); shutil.rmtree(d, ignore_errors=True); return True
        except Exception as e:
            log(f"   HF push attempt {attempt + 1} failed:", str(e)[:300]); time.sleep(20 * 3 ** attempt)
    if S3_URI: sh(f"aws s3 cp --recursive --only-show-errors {d} {S3_URI}/{RUN}/export/{tag}/")   # a copy that survives the instance
    return False

@torch.no_grad()
def evaluate(step):
    model.eval(); res = {}
    for name, v in val_sets.items():
        tot = 0.0; n = 0
        for i in range(0, v.shape[0], MICRO_SEQS):
            x = v[i:i + MICRO_SEQS].to(DEV); tot += float(core(x)) * x.shape[0]; n += x.shape[0]
        res[name] = round(tot / max(1, n), 4)
    model.train(); metric({"step": step, "val": res}); log("val loss:", res)
    return res

# ------------------------------------------------------------------ train
tokens_per_step = BATCH_SEQS * SEQ
log(f"run {RUN}: {TOTAL_STEPS} steps x {tokens_per_step/1e6:.2f}M tokens = {TOTAL_STEPS*tokens_per_step/1e9:.2f}B tokens; "
    f"LR {LR} {SCHEDULE}, warmup {WARMUP}; micro {MICRO_SEQS} x accum {BATCH_SEQS // (MICRO_SEQS * WORLD)}; world {WORLD}")
assert BATCH_SEQS % (MICRO_SEQS * WORLD) == 0, "BATCH_SEQS must be divisible by MICRO_SEQS x WORLD_SIZE"
had_eval = os.path.exists(METRICS) and any('"val"' in l for l in open(METRICS))
if step == 0 or not had_eval: evaluate(step)
model.train(); last_save = time.time(); t_win = time.time(); tok_win = 0; bad_steps = 0
next_milestone = (math.floor(step * tokens_per_step / MILESTONE_TOKENS) + 1) * MILESTONE_TOKENS

def stop_now(step, why):
    save_ckpt(step, why, block_upload=True)
    log("stopping:", why); metric({"event": "stopped", "step": step, "reason": why[:200]}); sys.exit(3)

def run_step(step, micro):
    """Mean loss, or None if a stop was requested mid-step (single GPU: the partial step is discarded; the batch
    is a function of `step`, so resuming replays it exactly)."""
    idx = batch_indices(step)[RANK::WORLD]
    accum = len(idx) // micro; loss_sum = 0.0
    for a in range(accum):
        if WORLD == 1 and STOP.is_set():
            opt.zero_grad(set_to_none=True); return None
        x = load_seqs(idx[a * micro:(a + 1) * micro]).to(DEV, non_blocking=True)
        ctx = lm.no_sync() if (WORLD > 1 and a < accum - 1) else contextlib.nullcontext()
        with ctx:
            loss = lm(x) / accum
            loss.backward()
        loss_sum += float(loss.detach())
    return loss_sum

while step < TOTAL_STEPS:
    if IS_MAIN and stop_file() and not STOP.is_set(): STOP_REASON.append("stop file (cost cap or operator)"); STOP.set()
    if WORLD > 1:
        flag = torch.tensor([1.0 if STOP.is_set() else 0.0], device=DEV); torch.distributed.all_reduce(flag)
        if flag.item() > 0: STOP.set()
    if STOP.is_set(): stop_now(step, "stop: " + "; ".join(STOP_REASON)[:150])
    for g in opt.param_groups: g["lr"] = lr_at(step)
    try:
        loss = run_step(step, MICRO_SEQS)
    except torch.OutOfMemoryError:
        if WORLD > 1: raise
        loss = "oom"
    if loss == "oom":
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        if MICRO_SEQS == 1: raise SystemExit("out of memory at MICRO_SEQS=1")
        MICRO_SEQS //= 2; log(f"OOM: retrying step {step} with MICRO_SEQS={MICRO_SEQS}"); continue
    if loss is None: stop_now(step, "stop: " + "; ".join(STOP_REASON)[:150])
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if not (math.isfinite(loss) and math.isfinite(gn)):
        opt.zero_grad(set_to_none=True); bad_steps += 1
        log(f"non-finite loss/grad at step {step} (loss {loss}, grad norm {gn}): update skipped ({bad_steps} in a row)")
        metric({"event": "skipped_step", "step": step, "loss": str(loss), "gn": str(gn)})
        if bad_steps >= 3: raise SystemExit(f"{bad_steps} consecutive non-finite steps at step {step}: no checkpoint written")
        step += 1; continue
    bad_steps = 0
    opt.step(); opt.zero_grad(set_to_none=True); step += 1; tok_win += tokens_per_step
    if step % 10 == 0 or step == 1:
        dt = time.time() - t_win; tps = tok_win / max(dt, 1e-6); mfu = tps * FLOPS_PER_TOKEN / (989e12 * WORLD)
        metric({"step": step, "tokens": step * tokens_per_step, "loss": round(loss, 4), "lr": lr_at(step), "grad_norm": round(gn, 3),
                "tok_s": round(tps), "mfu_h100": round(mfu, 3), "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "micro": MICRO_SEQS})
        log(f"step {step}/{TOTAL_STEPS} loss {loss:.4f} lr {lr_at(step):.2e} gn {gn:.2f} {tps:,.0f} tok/s mfu(H100) {mfu:.1%} "
            f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}G elapsed {(time.time()-T0)/3600:.2f}h")
        t_win = time.time(); tok_win = 0
    if STOP.is_set(): continue   # skip eval/milestone work: the top of the loop saves and exits
    if step % EVAL_EVERY == 0: evaluate(step)
    if step * tokens_per_step >= next_milestone:
        save_ckpt(step, "milestone"); push_milestone(step, step * tokens_per_step); next_milestone += MILESTONE_TOKENS
    elif time.time() - last_save > SAVE_EVERY_MIN * 60:
        save_ckpt(step, "periodic"); last_save = time.time()
    hours_flag = torch.tensor([1.0 if (time.time() - T0) / 3600 > MAX_HOURS else 0.0], device=DEV)
    if WORLD > 1: torch.distributed.all_reduce(hours_flag)
    if hours_flag.item() > 0:
        save_ckpt(step, "max_hours", block_upload=True)
        log(f"MAX_HOURS={MAX_HOURS} reached at step {step}"); metric({"event": "paused", "step": step}); sys.exit(4)

evaluate(step)
try:
    save_ckpt(step, "final", block_upload=True)
except RuntimeError as e:
    log("final checkpoint not persisted:", e); sys.exit(5)
if not push_milestone(step, step * tokens_per_step, final=True):
    log("final model could not be pushed to the Hub (a copy went to S3 export/final if S3 works)"); sys.exit(5)
metric({"event": "done", "step": step})
if IS_MAIN: open(os.path.join(OUT_DIR, "DONE"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
barrier(); log("TRAINING COMPLETE")
