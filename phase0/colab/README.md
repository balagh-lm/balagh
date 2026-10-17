# Balagh data and fine-tuning notebooks

Data jobs run on **Kaggle CPU** as batch runs ("Save & Run All"), so they keep going with your PC off. Several can run at once. Fine-tuning and evaluation run on **Colab A100/L4**. Training itself runs on AWS (see `../aws/README.md`). Every output lives on the Hugging Face Hub under `balagh-lm`. Every job resumes where it stopped, so a timeout only loses the unit in flight.

## One-time setup

1. Hugging Face: accept the FLORES+ terms at huggingface.co/datasets/openlanguagedata/flores_plus (decontamination refuses to run without it). Then create a **fine-grained token** with write access to the `balagh-lm` org repos **and** the "read access to public gated repos you can access" permission, so the same token can read FLORES+.
2. Kaggle: Add-ons > Secrets > add `HF_TOKEN`. Colab: key icon > add `HF_TOKEN`, enable notebook access. Never paste the token into a cell.

## Run order

| Step | Notebook | Where | Output on the Hub |
|---|---|---|---|
| 1a | `01_build_corpus` with `MODE='paklid'` | Kaggle CPU, ~1 h | `balagh-lm/paklid` (model + held-out report) |
| 1b | `01_build_corpus` with `MODE='route'`, five copies: `SOURCES='group:urdu-fw2'`, `'group:urdu-hplt'`, `'group:urdu-sangraha'`, `'group:pak-mid'`, `'group:pak-small'` | Kaggle CPU, 2-6 h each, in parallel | `balagh-lm/corpus-v1-routed` |
| 1c | `01_build_corpus` with `MODE='dedup'`, three copies: `LANGS='urd_Arab'`, `'pbt_Arab,snd_Arab'`, `'urd_Latn,pnb_Arab,skr_Arab,bal_Arab,brh_Arab,kas_Arab,hno_Arab,phr_Arab'` | Kaggle CPU, in parallel | `balagh-lm/corpus-v1` (+ `stats.json` per language) |
| 1d | `01_build_corpus` with `MODE='card'` | Kaggle CPU, minutes | data card with the routing matrix |
| 2 | `03_sft_data` | Kaggle CPU | `balagh-lm/sft-v0` |
| 3 | `../balagh_eval_baselines` | Colab A100 or L4 | baseline table (Qwen3.5-2B-Base is the gate) |
| 4 | `05_tokenize_export`, three copies: `SOURCES='urd_Arab'`, `'en_edu,code,math'`, `'pbt_Arab,snd_Arab,pnb_Arab,skr_Arab,bal_Arab,brh_Arab,kas_Arab,hno_Arab,phr_Arab,urd_Latn'`, then `MODE='manifest'` | Kaggle CPU | `balagh-lm/tokens-qwen35-v1` |
| 5 | AWS smoke, pilot, main run | AWS p5.4xlarge | `balagh-lm/balagh-nano-ckpt` (private) |
| 6 | `07_sft` (`SMOKE='1'` first) | Colab A100 | `balagh-lm/balagh-nano-v0-instruct` (private until release) |

Importing on Kaggle: Create > New Notebook > File > Import Notebook > Link, then paste the GitHub URL of the `.ipynb` (for example `https://github.com/balagh-lm/balagh/blob/main/phase0/colab/01_build_corpus.ipynb`). Set the knobs in the first code cell, then Save Version > Save & Run All.

## Smoke tests (minutes, no GPU)

- `01_build_corpus`: `MODE='paklid'` with `PAKLID_PER_SOURCE='300'`, then `MODE='route'` with `SOURCES='wiki:skr'` and `MAX_UNITS='1'`, then `MODE='dedup'` with `LANGS='skr_Arab'`. Use a scratch repo by also setting `ROUTED_REPO`, `OUT_REPO` and `PAKLID_REPO` to names under your own account.
- `05_tokenize_export`: `TOTAL_TOKENS='2e7'` with `LOCAL_OUT='/kaggle/working/tok'` (it refuses small budgets against the real repo).

## What the corpus build does

Per-language normalisation without global NFKC (Arabic yeh/kaf mapped to Urdu forms; Sindhi keeps Arabic yeh; Pashto keeps both yeh letters and only unifies kaf; digits to ASCII, tatweel and zero-width characters removed, PDF presentation forms decomposed). Quality filters cover length, script share, word length, repetition, boilerplate, duplicate lines and digit ratio. **PakLID v1** then routes every document to the language it is really in. This matters because the upstream buckets bleed: Saraiki Wikipedia holds Urdu and Punjabi pages, GlotCC's Balochi bucket holds Urdu, and FineWeb-2's rejected Saraiki bucket holds hundreds of real Saraiki documents. Then come exact dedup (SHA-1) and MinHash near-dedup (5-word shingles, 128 permutations, 16x8 bands; the higher-priority source wins), plus 13-word-gram decontamination against UrduMMLU, Belebele, SIB-200, Global PIQA, FLORES+ and SMOL. Upstream revisions are pinned per source.

Sources: FineWeb-2 (incl. its rejected buckets for minority languages), HPLT 2.0 cleaned, GlotCC-V1, Sangraha verified, Wikipedia 2023-11, Omnilingual-ASR transcripts (Pashto, Shahmukhi, Brahui, Kashmiri, Hindko, Pothwari).
