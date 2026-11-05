# Balagh test t01 (Kaggle GPU T4): train_cpt.py with OPTIM=adamw8bit on Qwen3.5-0.8B-Base and a few small
# token shards: 20 steps, checkpoint, then resume to 40 steps. Checks the bitsandbytes path and that its optimizer
# state survives save/resume before any paid AWS run. Prints "T01 PASSED" or "T01 FAILED: ...".
import os, subprocess, sys, glob, json, urllib.request

REF = os.environ.get("BALAGH_REF", "7570c53484d8c6e8fefd1840ba579f46dcd42832")
subprocess.run("pip -q install 'transformers==5.20.1' 'bitsandbytes==0.50.2' 'huggingface_hub[hf_xet]'", shell=True, check=True)

from huggingface_hub import snapshot_download
data = snapshot_download("balagh-lm/tokens-qwen35-v1", repo_type="dataset", local_dir="/tmp/tokens",
                         allow_patterns=["manifest.json", "val/*", "hno_Arab/*", "phr_Arab/*", "kas_Arab/*"])
src = urllib.request.urlopen(f"https://raw.githubusercontent.com/balagh-lm/balagh/{REF}/phase0/aws/train_cpt.py").read()
open("/tmp/train_cpt.py", "wb").write(src)

import torch
print("torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0), "| bf16 ok", torch.cuda.is_bf16_supported())
seq = json.load(open(f"{data}/manifest.json"))["seq_len"]
env = dict(os.environ, BASE="Qwen/Qwen3.5-0.8B-Base", OPTIM="adamw8bit", REQUIRE_KERNELS="0", SMOKE="1", HF_PUSH="0",
           BATCH_SEQS="4", MICRO_SEQS="1", EVAL_SEQS="2", DATA_DIR=data, OUT_DIR="/tmp/runs", RUN="t01",
           CE_CHUNK="512", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")   # T4 has 15 GB; H100 keeps the 4096 default

def run(steps, extra):
    e = dict(env, TOTAL_TOKENS=str(steps * 4 * seq), **extra)
    p = subprocess.run([sys.executable, "-u", "/tmp/train_cpt.py"], env=e, capture_output=True, text=True)
    print(p.stdout[-6000:]); print(p.stderr[-4000:]); return p.returncode, p.stdout

rc1, out1 = run(20, {})
rc2, out2 = run(40, {"ALLOW_CONFIG_CHANGE": "1"}) if rc1 == 0 else (None, "")
problems = []
if rc1 != 0: problems.append(f"first run exit {rc1}")
if rc1 == 0 and rc2 != 0: problems.append(f"resume run exit {rc2}")
if "AdamW8bit" not in out1: problems.append("optimizer is not AdamW8bit")
if rc2 == 0 and "resumed from" not in out2: problems.append("second run did not resume")
print("T01 PASSED" if not problems else "T01 FAILED: " + "; ".join(problems))
