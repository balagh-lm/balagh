# Builds Colab/Kaggle notebooks from the pipeline scripts in this folder: one setup cell (pip + secrets
# check), one cell with the script. Run: python build_notebooks.py
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SPECS = {
    "01_fetch_corpora": dict(pip="datasets huggingface_hub pyarrow zstandard", title="Balagh data pipeline 1: fetch open corpora to the Hub", platform="colab",
        note="Runtime: CPU. Needs `HF_TOKEN` (write) in Colab Secrets. Resumable. Smoke test: `ONLY='wiki:skr,smol:smolsent__en_ur'`, `MAX_DOCS='2000'`.",
        knobs={"ONLY": "", "MAX_DOCS": "0"}),
    "02_clean_corpus": dict(pip="datasets huggingface_hub pyarrow zstandard datasketch xxhash scikit-learn", title="Balagh data pipeline 2: clean, PakLID, dedup, decontaminate", platform="colab",
        note="Runtime: CPU High-RAM. Needs `HF_TOKEN`. Resumable per language. Smoke test: `LANGS='skr_Arab'`, `MAX_DOCS='20000'`.",
        knobs={"LANGS": "", "MAX_DOCS": "0", "FORCE": "0"}),
    "03_sft_data": dict(pip="datasets huggingface_hub pyarrow", title="Balagh data pipeline 3: native SFT data", platform="colab",
        note="Runtime: CPU. Needs `HF_TOKEN`. Builds balagh-lm/sft-v0 (train/dev JSONL + data card).", knobs={}),
    "05_tokenize_export": dict(pip="datasets huggingface_hub transformers numpy kaggle", title="Balagh data pipeline 5: tokenise the mix and export to the Hub + a Kaggle dataset", platform="colab",
        note="Runtime: CPU High-RAM. Needs `HF_TOKEN`; for the Kaggle export also `KAGGLE_USERNAME` and `KAGGLE_KEY` in Secrets. Smoke test: `MAX_TOKENS_TOTAL='20000000'`.",
        knobs={"MAX_TOKENS_TOTAL": "0", "TOKENIZER": "google/gemma-4-E2B"}),
    "06_kaggle_tpu_cpt": dict(pip="transformers accelerate huggingface_hub bitsandbytes", title="Balagh-nano continued pretraining: one Kaggle TPU v5e-8 session", platform="kaggle",
        note="Kaggle: Accelerator TPU v5e-8, Internet on, Secret `HF_TOKEN` (write), input dataset = the tokens Kaggle dataset. First run with `SMOKE='1'` (50 steps, ~20 min). Then submit as Save & Run All with SMOKE='0'; repeat each session, it resumes from the Hub.",
        knobs={"SMOKE": "1", "BASE": "google/gemma-4-E2B", "DATA_DIR": "/kaggle/input/balagh-tokens-gemma4-v1"}),
}
for name, sp in SPECS.items():
    src = open(os.path.join(HERE, name + ".py"), encoding="utf-8").read()
    setup = [f"!pip -q install -U {sp['pip']}\n", "import os\n"]
    for k, v in sp["knobs"].items(): setup.append(f"os.environ['{k}'] = '{v}'\n")
    if sp["platform"] == "colab":
        setup += ["from google.colab import userdata\n", "assert userdata.get('HF_TOKEN'), 'add HF_TOKEN (write) in Colab Secrets and enable notebook access'\n", "print('token ok')"]
        meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "colab": {"provenance": [], "name": name + ".ipynb"}}
    else:
        setup += ["from kaggle_secrets import UserSecretsClient\n", "assert UserSecretsClient().get_secret('HF_TOKEN'), 'add HF_TOKEN in Kaggle Secrets (Add-ons -> Secrets)'\n", "print('token ok')"]
        meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "kaggle": {"accelerator": "tpu1vmV5e8"}}
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [f"# {sp['title']}\n", "\n", sp["note"] + "\n", "\n", "Source: https://github.com/balagh-lm/balagh/tree/main/phase0/colab"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": setup},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [src]}]}
    out = os.path.join(HERE, name + ".ipynb"); json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False); print("wrote", out)
