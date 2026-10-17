# Builds Kaggle/Colab notebooks from the pipeline scripts: one setup cell (pip + knobs + secret check) and one
# cell with the script. Run: python build_notebooks.py   (writes .ipynb next to each script)
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SECRET_CHECK = [
    "def _tok():\n",
    "    try:\n",
    "        from google.colab import userdata; return userdata.get('HF_TOKEN')\n",
    "    except Exception: pass\n",
    "    try:\n",
    "        from kaggle_secrets import UserSecretsClient; return UserSecretsClient().get_secret('HF_TOKEN')\n",
    "    except Exception: return os.environ.get('HF_TOKEN')\n",
]
SPECS = {
    "01_build_corpus": dict(dir=HERE, pip="datasets huggingface_hub pyarrow datasketch xxhash scikit-learn numpy joblib",
        title="Balagh corpus builder: PakLID v1, routing, dedup, decontamination", need_token=True,
        note=("Kaggle: Accelerator **None** (CPU), Internet **on**, Add-ons > Secrets: `HF_TOKEN` (write, balagh-lm). Save & Run All (batch) so it runs with your PC off.\n\n"
              "Order: `MODE='paklid'` once; then `MODE='route'` in five copies with `SOURCES` = `group:urdu-fw2`, `group:urdu-hplt`, `group:urdu-sangraha`, `group:pak-mid`, `group:pak-small`; "
              "then `MODE='dedup'` with `LANGS` = `urd_Arab` | `pbt_Arab,snd_Arab` | `urd_Latn,pnb_Arab,skr_Arab,bal_Arab,brh_Arab,kas_Arab,hno_Arab,phr_Arab`; then `MODE='card'`. Every mode resumes where it stopped."),
        knobs={"MODE": "paklid", "SOURCES": "all", "LANGS": "", "MAX_UNITS": "0", "PAKLID_PER_SOURCE": "0"}),
    "03_sft_data": dict(dir=HERE, pip="datasets huggingface_hub pyarrow", title="Balagh SFT data: native instructions -> balagh-lm/sft-v0", need_token=True,
        note="Kaggle CPU or Colab CPU. Needs `HF_TOKEN` (write). About 20 minutes.", knobs={}),
    "05_tokenize_export": dict(dir=HERE, pip="'transformers>=5.20' huggingface_hub pyarrow numpy", title="Balagh tokenisation: corpus-v1 + replay -> packed Qwen3.5 token shards", need_token=True,
        note=("Kaggle CPU (or Colab CPU High-RAM). Needs `HF_TOKEN` (write). Run after corpus-v1 is complete. Parallel copies: `SOURCES='urd_Arab'` | `SOURCES='en_edu,code,math'` | "
              "`SOURCES='pbt_Arab,snd_Arab,pnb_Arab,skr_Arab,bal_Arab,brh_Arab,kas_Arab,hno_Arab,phr_Arab,urd_Latn'`; then one run with `MODE='manifest'`. "
              "Set `TOTAL_TOKENS` from the AWS pilot's measured throughput (plan section 3)."),
        knobs={"MODE": "tokenize", "SOURCES": "", "TOTAL_TOKENS": "8e9"}),
    "07_sft": dict(dir=HERE, pip="'transformers==5.20.1' 'flash-linear-attention==0.5.2' bitsandbytes datasets huggingface_hub accelerate && pip -q install --no-build-isolation 'causal-conv1d==1.7.0' || echo 'causal-conv1d unavailable: slower fallback'",
        title="Balagh SFT: chat fine-tuning + generation-based UrduMMLU gate", need_token=True,
        note=("Colab **A100** (40 GB is enough) or L4. Needs `HF_TOKEN`. Path A: `BASE_REPO='balagh-lm/balagh-nano-ckpt'`, `BASE_SUBFOLDER='balagh-nano-v0/final'`. "
              "Path B: `BASE_REPO='Qwen/Qwen3.5-2B-Base'`, `BASE_SUBFOLDER=''`. Smoke test first with `SMOKE='1'` (400 examples, ~10 min)."),
        knobs={"BASE_REPO": "balagh-lm/balagh-nano-ckpt", "BASE_SUBFOLDER": "balagh-nano-v0/final", "SMOKE": "1"}),
    "balagh_eval_baselines": dict(dir=os.path.dirname(HERE), pip="'transformers==5.20.1' bitsandbytes accelerate datasets huggingface_hub && python -c \"import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else 1)\" && pip -q install 'flash-linear-attention==0.5.2' && pip -q install --no-build-isolation 'causal-conv1d==1.7.0' || echo 'no linear-attention kernels (T4 or build failed): PyTorch fallback'",
        title="Balagh evaluation: baselines and checkpoints, one protocol", need_token=False,
        note=("Colab A100/L4 (bf16), or Kaggle GPU T4 x2 (Qwen3.5 runs in fp32 there, slower). `HF_TOKEN` only for private checkpoints. "
              "Baselines: leave knobs empty. A training checkpoint: `CKPT_REPO='balagh-lm/balagh-nano-ckpt'`, `CKPT_SUBFOLDER='balagh-nano-v0/tokens-1.0B'`."),
        knobs={"ONLY": "", "CKPT_REPO": "", "CKPT_SUBFOLDER": "", "N_MMLU": "1000"}),
}
for name, sp in SPECS.items():
    src = open(os.path.join(sp["dir"], name + ".py"), encoding="utf-8").read()
    setup = [f"!pip -q install -U {sp['pip']}\n", "import os\n"]
    for k, v in sp["knobs"].items(): setup.append(f"os.environ['{k}'] = '{v}'\n")
    setup += SECRET_CHECK
    setup.append("assert _tok(), 'add HF_TOKEN in Kaggle Add-ons > Secrets or Colab Secrets (key icon) and enable access'\nprint('HF token found')" if sp["need_token"]
                 else "print('HF token found' if _tok() else 'no HF token (fine for public models)')")
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "colab": {"provenance": [], "name": name + ".ipynb"}},
          "cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"# {sp['title']}\n", "\n", sp["note"] + "\n", "\n", "Source: https://github.com/balagh-lm/balagh/tree/main/phase0"]},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": setup},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [src]}]}
    out = os.path.join(sp["dir"], name + ".ipynb"); json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False); print("wrote", out)

# ------------------------------------------------------------------ ready-made Kaggle job notebooks (knobs preset, import by URL, Save & Run All)
KDIR = os.path.join(os.path.dirname(HERE), "kaggle"); os.makedirs(KDIR, exist_ok=True)
JOBS = {   # job name -> (script spec, knob overrides, one-line purpose)
    "k01_paklid":          ("01_build_corpus", {"MODE": "paklid"}, "Train PakLID v1 (CPU, ~1 h)"),
    "k02_route_urdu_fw2":  ("01_build_corpus", {"MODE": "route", "SOURCES": "group:urdu-fw2", "THEN_DEDUP": "1"}, "Route FineWeb-2 Urdu (waits for PakLID)"),
    "k03_route_urdu_hplt": ("01_build_corpus", {"MODE": "route", "SOURCES": "group:urdu-hplt", "THEN_DEDUP": "1"}, "Route HPLT 2.0 + GlotCC Urdu"),
    "k04_route_urdu_sangraha": ("01_build_corpus", {"MODE": "route", "SOURCES": "group:urdu-sangraha", "THEN_DEDUP": "1"}, "Route Sangraha verified Urdu"),
    "k05_route_pak_mid":   ("01_build_corpus", {"MODE": "route", "SOURCES": "group:pak-mid", "THEN_DEDUP": "1"}, "Route Pashto + Sindhi sources"),
    "k06_route_pak_small": ("01_build_corpus", {"MODE": "route", "SOURCES": "group:pak-small", "THEN_DEDUP": "1"}, "Route Wikipedia, transcripts, GlotCC and small-language buckets"),
    "k07_dedup_all":       ("01_build_corpus", {"MODE": "dedup", "LANGS": ""}, "Dedup + decontaminate every language and write the card (rerun if a routing job could not)"),
    "k08_sft_data":        ("03_sft_data", {}, "Build balagh-lm/sft-v0"),
    "k09_eval_baselines_t4": ("balagh_eval_baselines", {}, "Baseline evals on Kaggle GPU T4 x2 (set Accelerator: GPU T4 x2)"),
    "k10_tok_urdu":        ("05_tokenize_export", {"SOURCES": "urd_Arab", "TOTAL_TOKENS": "8e9"}, "Tokenise Urdu (after corpus-v1 exists)"),
    "k11_tok_replay":      ("05_tokenize_export", {"SOURCES": "en_edu,code,math", "TOTAL_TOKENS": "8e9"}, "Tokenise English/code/maths replay"),
    "k12_tok_rest":        ("05_tokenize_export", {"SOURCES": "pbt_Arab,snd_Arab,pnb_Arab,skr_Arab,bal_Arab,brh_Arab,kas_Arab,hno_Arab,phr_Arab,urd_Latn", "TOTAL_TOKENS": "8e9"}, "Tokenise the other languages"),
    "k13_tok_manifest":    ("05_tokenize_export", {"MODE": "manifest", "TOTAL_TOKENS": "8e9"}, "Write the token manifest (after k10-k12)"),
}
for job, (spec_name, knobs, purpose) in JOBS.items():
    sp = SPECS[spec_name]
    src = open(os.path.join(sp["dir"], spec_name + ".py"), encoding="utf-8").read()
    allk = dict(sp["knobs"]); allk.update(knobs)
    setup = [f"!pip -q install -U {sp['pip']}\n", "import os\n"] + [f"os.environ['{k}'] = '{v}'\n" for k, v in allk.items()] + SECRET_CHECK
    setup.append("assert _tok(), 'attach the HF_TOKEN secret: Add-ons > Secrets'\nprint('HF token found')" if sp["need_token"] else "print('HF token found' if _tok() else 'no HF token')")
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"# Balagh job {job}: {purpose}\n", "\n", sp["note"] + "\n", "\n", f"Knobs preset: `{json.dumps(knobs)}`. Generated from `phase0/colab/{spec_name}.py`."]},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": setup},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [src]}]}
    json.dump(nb, open(os.path.join(KDIR, job + ".ipynb"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", len(JOBS), "Kaggle job notebooks to", KDIR)
