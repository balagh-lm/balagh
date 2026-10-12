# Balagh on Colab: data preparation

Everything here runs on Google Colab (Pro / AI Pro, 200 compute units a month) and stores results on the Hugging Face Hub under the `balagh-lm` org. Nothing is kept on Colab or on the laptop.

## One-time setup (5 minutes)

1. Hugging Face: create a **write** token at huggingface.co/settings/tokens scoped to the `balagh-lm` org (or your account with org write rights).
2. Colab: open any notebook, click the key icon (Secrets), add `HF_TOKEN` with that value, enable "Notebook access". Never paste the token into a cell.
3. Accept FLORES+ terms once at huggingface.co/datasets/openlanguagedata/flores_plus (click-through), so decontamination can use it.

## Run order

| Step | Notebook | Where | Units (approx.) | Output on the Hub |
|---|---|---|---|---|
| 1 | [01_fetch_corpora.ipynb](https://colab.research.google.com/github/balagh-lm/balagh/blob/main/phase0/colab/01_fetch_corpora.ipynb) | Colab CPU | 1 to 3 (CPU runtimes cost almost nothing; a few hours wall-clock) | `balagh-lm/raw-corpora-v0`: parquet shards per source/config + `manifest.json` |
| 2 | [02_clean_corpus.ipynb](https://colab.research.google.com/github/balagh-lm/balagh/blob/main/phase0/colab/02_clean_corpus.ipynb) | Colab CPU High-RAM | 2 to 6 | `balagh-lm/corpus-v1`: cleaned shards per language, `stats.json` per language, data card |
| 3 | [03_sft_data.ipynb](https://colab.research.google.com/github/balagh-lm/balagh/blob/main/phase0/colab/03_sft_data.ipynb) | Colab CPU | 1 | `balagh-lm/sft-v0`: train/dev chat JSONL + data card |
| 4 | [balagh_eval_baselines.ipynb](https://colab.research.google.com/github/balagh-lm/balagh/blob/main/phase0/balagh_eval_baselines.ipynb) (same notebook as on Kaggle) | Colab A100/L4 or Kaggle T4 x2 | 20 to 40 | results JSON + leaderboard rows |
| 5 | [05_tokenize_export.ipynb](https://colab.research.google.com/github/balagh-lm/balagh/blob/main/phase0/colab/05_tokenize_export.ipynb) | Colab CPU High-RAM | 2 to 4 | `balagh-lm/tokens-gemma4-v1` (packed uint32 shards, 10B-token mix) + a Kaggle dataset copy |
| 6 | [06_kaggle_tpu_cpt.ipynb](https://github.com/balagh-lm/balagh/blob/main/phase0/colab/06_kaggle_tpu_cpt.ipynb) (import into Kaggle by URL) | Kaggle TPU v5e-8, batch Save & Run All, one 9 h session at a time | Kaggle TPU quota, 20 h/week/account | `balagh-lm/balagh-nano-ckpt`: a checkpoint every 1,000 steps, resumes itself |
| 7 | 07_sft_train (next: QLoRA/LoRA on the step-20000 checkpoint, or on Gemma 4 E4B for the fallback path) | Colab A100 | 100 to 140 | `balagh-lm/balagh-nano-v0` adapter + merged weights |

Steps 1, 2 and 5 are resumable: 1 skips shards already on the Hub, 2 skips languages whose `stats.json` exists, 5 skips token shards already uploaded. Step 6 resumes from the newest checkpoint on the Hub every session and stops itself after 8 h 30 min. A Colab or Kaggle timeout costs only the shard or the steps in flight.

Step 6 on Kaggle: File > Import Notebook > GitHub URL of `06_kaggle_tpu_cpt.ipynb`; Settings: Accelerator TPU v5e-8, Internet on, Add-ons > Secrets `HF_TOKEN`; Add Input > the Kaggle dataset produced by step 5. First run with `SMOKE='1'` (50 steps, ~20 min, uploads a checkpoint); then set `SMOKE='0'` and Save & Run All. Re-submit the same notebook after each session ends.

## Smoke test first (10 minutes, ~0 units)

In the setup cell of notebook 1 set `ONLY='wiki:skr,smol:smolsent__en_ur'` and `MAX_DOCS='2000'`, run, confirm two shard folders appear in `balagh-lm/raw-corpora-v0`. Then in notebook 2 set `LANGS='skr_Arab'` and `MAX_DOCS='5000'`, run, confirm `skr_Arab/stats.json` and shards appear in `balagh-lm/corpus-v1`. Then clear the knobs and run both fully. Notebook 5 has its own smoke knob, `MAX_TOKENS_TOTAL='20000000'`, which produces one small shard per source.

## What the cleaning does, in one paragraph

Per-language normalisation (digits, Arabic-vs-Urdu letter forms, tatweel, zero-width characters, PDF presentation forms; no global NFKC, which corrupts Nastaliq code points), quality filters (length, script share, average word length, boilerplate lines, repetition, duplicate lines, digit ratio), a PakLID language check that re-sorts documents to the language the classifier predicts (the raw buckets bleed: Urdu in the Punjabi bucket, Roman Urdu in the Balochi bucket), exact dedup by SHA-1, MinHash near-dedup (128 permutations, 16 bands x 8 rows, about Jaccard 0.75), and 13-word-gram decontamination against UrduMMLU, Belebele, SIB-200, Global PIQA and FLORES+. Every document keeps its source and licence.

## Sizes to expect

Raw mirror: roughly 30 to 50 GB across sources (Urdu dominates). Cleaned corpus: 5 to 8 billion words of Urdu, under 1 billion each for Pashto and Sindhi, tens of millions for Shahmukhi and Saraiki, a few million for Balochi and Brahui. Exact numbers land in each language's `stats.json` and the data card.
