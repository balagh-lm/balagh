# What 200 Colab compute units a month can and cannot do

**Compiled:** 12 October 2026. Colab does not publish a rate table; the per-hour burn appears in the runtime's Resources panel after you connect. Read it once per GPU type and correct the table below. Third-party measurements (March 2026) are used as the planning figures.

## Budget by GPU

| Colab GPU | Burn (units/hour, planning figure) | Hours per 200 units | bf16 peak | Realistic training throughput at ~35% MFU |
|---|---|---|---|---|
| A100 40 GB | ~5.4 (McCormick, Mar 2026) to ~13 (older guides) | 15 to 37 h | 312 TFLOP/s | ~110 TFLOP/s |
| L4 24 GB | ~1.7 to 4.8 | 40 to 115 h | 121 TFLOP/s (fp8 242) | ~40 TFLOP/s |
| T4 16 GB | ~1.2 to 1.8 | 110 to 165 h | no bf16 cores; fp16 65 | ~15 TFLOP/s fp16, ~6 bf16 |

Session caps: 24 hours per session on Pro tiers, background execution on Pro+. Checkpoint to the Hugging Face Hub every 500 steps regardless.

## Translating hours into tokens (6 × params × tokens, divided by throughput)

Full-parameter training, A100 at ~110 TFLOP/s:

| Model | Tokens per A100-hour | Tokens per month at 200 units (15 to 37 h) |
|---|---|---|
| 2B class (Gemma 4 E2B, Qwen3.5-2B) | ~29M | 0.4 to 1.1B |
| 4B class (Gemma 4 E4B, Qwen3.5-4B) | ~15M | 0.2 to 0.55B |

Continued pretraining of a 2B model on 10B tokens therefore needs 9 to 23 months of the Colab budget. **Colab cannot be the pretraining engine.** That stays with TPU Research Cloud, the AMD credit, or Kaggle's TPU when it allocates.

## What fits, and how big a model

Memory decides the ceiling; A100 40 GB is the useful Colab GPU.

| Method | Memory rule of thumb | Largest Apache-2.0 candidate that fits on A100 40 GB | Tokens per A100-hour (approx.) |
|---|---|---|---|
| Full fine-tune, bf16 weights + 8-bit Adam | ~8 bytes/param + activations | ~3B (Gemma 4 E2B comfortably, E4B tight with gradient checkpointing and short sequences) | 25 to 30M (2B) |
| LoRA, bf16 base | ~2.2 bytes/param + activations | ~12B (Gemma 4 12B, Qwen3.5-9B) | 12 to 18M (9B to 12B) |
| QLoRA, 4-bit base (NF4) | ~0.7 bytes/param + activations | ~31B (Gemma 4 31B at ~18 GB weights; 26B-A4B similar) | 3 to 6M (27B to 31B; dequant overhead) |

Memory arithmetic for the 31B case: 31B × 0.5 byte (NF4) ≈ 16 GB, plus LoRA adapters, optimizer and activations with checkpointing at 2K context ≈ 8 to 12 GB. Fits with headroom on 40 GB. 80 GB A100s, when Colab assigns them, allow 4K context or larger batches.

## Where "higher than 2B" is real and where it is not

- **Supervised fine-tuning and preference tuning (SFT, DPO) at 9B to 31B: yes.** The native SFT set is 300K to 500K examples, about 150M tokens. QLoRA on Gemma 4 12B at ~12M tokens/hour needs ~13 A100-hours per epoch: one month's budget covers one epoch at 12B, or three epochs at E4B, or one epoch at 31B with a 50K-example subset. This is how Balagh-nano-v0 beats Qalb without any pretraining: Gemma 4 E4B-it already scores far above Qalb's 31.5 to 35.5 on UrduMMLU, and a native SFT on top is the first release.
- **Continued pretraining at any size: no**, not on Colab alone. Even with LoRA (which trains ~1% of the weights but still pays the full forward and backward cost through the frozen base), 10B tokens on a 2B model is years of this budget.
- **4-bit tricks do not change the token math.** NF4 and 8-bit optimizers cut memory, not FLOPs. QLoRA at 31B actually runs slower per token than full bf16 at 2B because of dequantization. If "collapsed-order computing" meant something other than low-precision and low-rank methods, say what you had in mind and I'll size it; nothing in common use lets 200 units pretrain a 2B model.

## Recommended split of the 200 units

| Use | Units per month | Why |
|---|---|---|
| Data pipeline: download, PakLID filtering, dedup, decontamination, tokenizer stats | 10 to 20 (CPU-heavy, run on the cheapest GPU or CPU runtime) | one-time, repeated per corpus version |
| Baseline and gate evaluations (UrduMMLU, Belebele, FLORES+) on 2B to 12B models | 20 to 40 | each candidate and each checkpoint |
| SFT/QLoRA training of Balagh-nano-v0 (Gemma 4 E4B full LoRA, or 12B QLoRA) | 100 to 140 | the release that beats Qalb |
| Synthetic-data generation with an open teacher on Colab (optional; free APIs are cheaper) | 0 to 20 | only if API rate limits bite |

Pretraining compute is tracked separately in `../PLAN.md` section 6.

## Storage on Colab

Colab's disk (about 100 to 200 GB) is scratch only. Every corpus shard and checkpoint is pushed to the Hugging Face Hub (`balagh-lm` org) as it is produced, and the notebooks are written to resume from the Hub. Your HF write token goes in Colab's Secrets panel as `HF_TOKEN`; it is never written into a notebook.
