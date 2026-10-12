# Builds Colab notebooks from the pipeline scripts in this folder: one setup cell (pip + HF token
# from Colab Secrets), one cell with the script. Run: python build_notebooks.py
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PIP = {
    "01_fetch_corpora": "datasets huggingface_hub pyarrow zstandard",
    "02_clean_corpus": "datasets huggingface_hub pyarrow zstandard datasketch xxhash scikit-learn",
}
TITLES = {
    "01_fetch_corpora": "Balagh data pipeline 1/2: fetch open corpora to the Hub",
    "02_clean_corpus": "Balagh data pipeline 2/2: clean, PakLID, dedup, decontaminate",
}
NOTES = {
    "01_fetch_corpora": "Runtime: CPU (High-RAM not needed). Needs `HF_TOKEN` in Colab Secrets with write access to balagh-lm. Resumable; re-run after a timeout. Set `ONLY` / `MAX_DOCS` env vars in the setup cell for a smoke test, e.g. `ONLY=wiki:skr`, `MAX_DOCS=2000`.",
    "02_clean_corpus": "Runtime: CPU High-RAM (Urdu has ~5M docs). Needs `HF_TOKEN`. Resumable per language. Smoke test: `LANGS=skr_Arab`, `MAX_DOCS=20000`.",
}
for name, pip in PIP.items():
    src = open(os.path.join(HERE, name + ".py"), encoding="utf-8").read()
    setup = [
        f"!pip -q install -U {pip}\n",
        "import os\n",
        "# optional knobs for smoke tests (leave empty for the full run)\n",
        "os.environ['ONLY'] = ''        # 01 only: comma-separated source keys, e.g. 'wiki:skr,smol:smolsent__en_ur'\n",
        "os.environ['LANGS'] = ''       # 02 only: comma-separated languages; empty = all\n",
        "os.environ['MAX_DOCS'] = '0'   # cap per source/language for smoke tests; 0 = no cap\n",
        "from google.colab import userdata\n",
        "assert userdata.get('HF_TOKEN'), 'add HF_TOKEN (write) in Colab Secrets and enable notebook access'\n",
        "print('token ok')",
    ]
    if name == "02_clean_corpus":
        setup = [s for s in setup if not s.startswith("os.environ['ONLY']")]
    else:
        setup = [s for s in setup if not s.startswith("os.environ['LANGS']")]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "colab": {"provenance": [], "name": name + ".ipynb"}},
          "cells": [
              {"cell_type": "markdown", "metadata": {}, "source": [f"# {TITLES[name]}\n", "\n", NOTES[name] + "\n", "\n", "Source: https://github.com/balagh-lm/balagh/tree/main/phase0/colab"]},
              {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": setup},
              {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [src]},
          ]}
    out = os.path.join(HERE, name + ".ipynb")
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out)
