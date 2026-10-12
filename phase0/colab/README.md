# Balagh on Colab: data preparation

Everything here runs on Google Colab (Pro / AI Pro, 200 compute units a month) and stores results on the Hugging Face Hub under the `balagh-lm` org. Nothing is kept on Colab or on the laptop.

## One-time setup (5 minutes)

1. Hugging Face: create a **write** token at huggingface.co/settings/tokens scoped to the `balagh-lm` org (or your account with org write rights).
2. Colab: open any notebook, click the key icon (Secrets), add `HF_TOKEN` with that value, enable "Notebook access". Never paste the token into a cell.
3. Accept FLORES+ terms once at huggingface.co/datasets/openlanguagedata/flores_plus (click-through), so decontamination can use it.

## Run order

| Step | Notebook | Runtime | Units (approx.) | Output on the Hub |
|---|---|---|---|---|
| 1 | [01_fetch_corpora.ipynb](https://colab.research.google.com/github/balagh-lm/balagh/blob/main/phase0/colab/01_fetch_corpora.ipynb) | CPU | 1 to 3 (CPU runtimes cost almost nothing; a few hours wall-clock) | `balagh-lm/raw-corpora-v0`: parquet shards per source/config + `manifest.json` |
| 2 | [02_clean_corpus.ipynb](https://colab.research.google.com/github/balagh-lm/balagh/blob/main/phase0/colab/02_clean_corpus.ipynb) | CPU High-RAM | 2 to 6 | `balagh-lm/corpus-v1`: cleaned shards per language, `stats.json` per language, data card |
| 3 | 03_sft_data (next) | CPU | 1 | `balagh-lm/sft-v0` |
| 4 | 04_eval_baselines (port of the Kaggle notebook) | A100 or L4 | 20 to 40 | results JSON + leaderboard rows |
| 5 | 05_sft_train (QLoRA/LoRA on Gemma 4 E4B or 12B) | A100 | 100 to 140 | `balagh-lm/balagh-nano-v0` adapter + merged weights |

Both pipeline notebooks are resumable: step 1 skips shards already on the Hub, step 2 skips languages whose `stats.json` exists. A Colab timeout costs only the shard in flight.

## Smoke test first (10 minutes, ~0 units)

In the setup cell of notebook 1 set `ONLY='wiki:skr,smol:smolsent__en_ur'` and `MAX_DOCS='2000'`, run, confirm two shard folders appear in `balagh-lm/raw-corpora-v0`. Then in notebook 2 set `LANGS='skr_Arab'` and `MAX_DOCS='5000'`, run, confirm `skr_Arab/stats.json` and shards appear in `balagh-lm/corpus-v1`. Then clear the knobs and run both fully.

## What the cleaning does, in one paragraph

Per-language normalisation (digits, Arabic-vs-Urdu letter forms, tatweel, zero-width characters, PDF presentation forms; no global NFKC, which corrupts Nastaliq code points), quality filters (length, script share, average word length, boilerplate lines, repetition, duplicate lines, digit ratio), a PakLID language check that re-sorts documents to the language the classifier predicts (the raw buckets bleed: Urdu in the Punjabi bucket, Roman Urdu in the Balochi bucket), exact dedup by SHA-1, MinHash near-dedup (128 permutations, 16 bands x 8 rows, about Jaccard 0.75), and 13-word-gram decontamination against UrduMMLU, Belebele, SIB-200, Global PIQA and FLORES+. Every document keeps its source and licence.

## Sizes to expect

Raw mirror: roughly 30 to 50 GB across sources (Urdu dominates). Cleaned corpus: 5 to 8 billion words of Urdu, under 1 billion each for Pashto and Sindhi, tens of millions for Shahmukhi and Saraiki, a few million for Balochi and Brahui. Exact numbers land in each language's `stats.json` and the data card.
