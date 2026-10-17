# Balagh-nano-v0: release plan v2, target 12 November 2026

**Revised 17 Oct 2026** after the founder chose a Qwen 2B base and secured $1,000 of AWS credits. Replaces v1 (Gemma 4 E2B on Kaggle TPU), which is in git history. Research behind every number: `research/qwen-aws-2026-10-17.md`.

## 1. Decisions

**Base model: `Qwen/Qwen3.5-2B-Base`** (Apache-2.0, released 2 Mar 2026).
- There is no official 2B Qwen 3.8. Qwen's Qwen 3.8 open weights are 27B dense (Apache-2.0), a 180B MoE and a 2.4T MoE. Repos named "Qwen3.8-2B" on Hugging Face (for example `empero-ai/Qwen3.8-2B-Distill`) are community instruction fine-tunes of Qwen3.5-2B distilled from a Qwen 3.8 teacher, not base models, so they are unsuitable for continued pretraining.
- Qwen3.5-2B-Base shares Qwen 3.8's BPE vocabulary byte for byte (`vocab.json` and `merges.txt` hashes identical). Two consequences: Qwen3.8-27B (Apache-2.0) can serve as a same-tokenizer teacher for SFT data, and if Qwen ever ships a small 3.8 base on this vocabulary, our token shards carry over unchanged.
- We train the text model only: 1.88B parameters (509M tied embeddings + 1.37B decoder; 18 Gated DeltaNet linear-attention layers + 6 full-attention layers). The checkpoint's vision tower (331M) and MTP head (61M) are dropped at load.

**Known costs of this choice, measured or published:**
- The Qwen tokenizer spends about 30% more tokens than Gemma 4 on the same text: Urdu 1.97 vs 1.51 tokens/word, Pashto 2.37 vs 1.87, Sindhi 2.82 vs 2.16 (our FLORES measurement, 11 Oct). A fixed token budget covers about 23% fewer words.
- Qwen small models start weak on Pashto (Belebele pbt: Qwen3-4B 37.9 vs Gemma-3n-E4B 64.6). Qwen3.5 has not been measured on Urdu by anyone; our baseline eval is the first.
- In its favour: under the same continued-pretraining recipe, the Qwen family forgot the least English of any base (AfriqueLLM: Qwen3-8B -1.6%, Gemma3-4B -6.2%, Llama3.1-8B -10.2%).

**Training method:** full-parameter continued pretraining, then SFT. No LoRA for the pretraining stage: LoRA at 2e-4 with 5% replay collapsed English on this exact base in the one published run (Tamil, MMLU 55.2 to 32.3). No vocabulary extension in v0: published evidence says it buys efficiency, not quality, and adds instability risk.

## 2. Where each job runs

| Job | Where | Why |
|---|---|---|
| Corpus build: PakLID v1, routing, dedup (`01_build_corpus`) | Kaggle CPU, batch "Save & Run All", 5 sessions in parallel | free, no GPU needed, PC can be off |
| SFT data (`03_sft_data`), tokenisation (`05_tokenize_export`) | Kaggle CPU | free |
| Baseline and checkpoint evals | Colab A100/L4 (units), or the AWS instance after training | T4 has no bf16 and no working linear-attention kernels |
| LR/replay pilot, 3 arms x 150M tokens | AWS p5.4xlarge (1x H100 80GB) | also measures real throughput |
| Main continued pretraining, 8-10B tokens | AWS p5.4xlarge, Spot if quota allows, else On-Demand | $1,000 credits |
| SFT, about 0.2B tokens | Colab A100 or AWS | |
| Kaggle TPU | dropped for this release | torch_xla is frozen (last release Nov 2025) and HF's Gated DeltaNet fallback has no XLA lowering; the JAX/MaxText route lacks a 2B config and packing isolation. Revisit for the Gemma-vs-Qwen bake-off paper. |

## 3. AWS budget ($1,000 credits)

Prices are AWS public list prices of 15 Oct 2026. On-Demand is the same in us-east-1, us-east-2 and us-west-2 ($6.88/h for p5.4xlarge). Spot is not: Spot Advisor shows p5.4xlarge about 68% below On-Demand in **us-west-2** but only 9-15% in us-east-1/2, so Spot runs go to us-west-2. Throughput is unmeasured for a 2B Gated DeltaNet model. The only published 2B data point implies about 16-19% MFU, not the 30% textbook figure, so the table plans for both.

| Item | At 30% MFU (26k tokens/s) | At 18% MFU (16k tokens/s) |
|---|---|---|
| Smoke test, 1 h | $7 | $7 |
| Pilot, 3 arms x 150M tokens | 5 h, $33 | 8 h, $54 |
| 8B tokens On-Demand ($6.88/h) | 85 h, $585 | 139 h, $955 (over budget) |
| 8B tokens Spot us-west-2 (about $2.2-3.7/h) | $190-315 | $305-515 |
| Evals, S3, egress | about $40 | about $40 |

**Budget rule:** the pilot measures tokens/s, then `TOTAL_TOKENS = $650 / price per hour x tokens/s x 3600`. On Spot in us-west-2 that is 8-10B tokens. On-Demand at the slow end it is about 5.5B tokens. The smoke test also tries `GRAD_CKPT=0` (no activation recomputation, about 25% faster if it fits in 80 GB).

**Expectations, from precedents:** 8-10B tokens is the low end of published adaptation runs that showed clear gains (Qwen-based precedents used 15-130B). Expect large gains on reading comprehension and translation (AfriqueLLM on Qwen3.5-4B: Belebele +10, FLORES +31) and small gains on knowledge exams (+3 to +9 on MMLU-style sets). Qwen3.5 forgot more English than Qwen3 under the same recipe, which is why replay stays at about a third and English is a release gate.

Not used: Capacity Blocks (upfront fee, likely not payable from credits, minimum 24 h = $1,146 for 8x H100), SageMaker (15% dearer, separate quotas), multi-GPU nodes (192-vCPU quota asks are rarely granted to new accounts).

**Quota requests**, filed the day AWS is logged in, in us-east-1, us-east-2 and us-west-2 (small asks are often auto-approved):

| Quota | Code | Ask |
|---|---|---|
| Running On-Demand P instances | L-417A185B | 16 vCPU (one p5.4xlarge) |
| P Spot instance requests | L-7212CCBC, or L-C4BD4855 if listed for P5 | 16 vCPU |
| Running On-Demand G and VT instances | L-DB2E81BA | 8 vCPU (g6e.2xlarge for evals and fallback) |

File in us-west-2 first (cheap Spot), then us-east-2 and us-east-1 as backups.

**Account checks first:** Paid plan (Activate credits require it), credit "Applicable products" include EC2, credit expiry date, and whether the account was created region-locked (Pakistan maps to Sydney under the new sign-up flow).

**Guardrails:** quotas cap the hardware; the training instance stops itself at `MAX_HOURS` and on completion; an Auto Scaling group of size 1 replaces Spot interruptions but fails closed after 6 launches; a cost budget alerts at $700 gross and at $5 of real card charges. Budget data lags 8-12 h, so the in-instance limits are the real brake.

## 4. Data mix and hyperparameters

Token shares of `TOTAL_TOKENS` (set in `05_tokenize_export.py`): Urdu 40%, Pashto 10%, Sindhi 6%, Roman Urdu 3%, Shahmukhi 2.5%, Saraiki 1%, Balochi 0.8%, Kashmiri 0.4%, Brahui 0.3%, Hindko/Pothwari 0.2%; replay 35.8% (FineWeb-Edu English 25.8%, code-heavy web 6%, FineMath 4%). Small languages repeat up to 4 epochs. Replay at about a third follows Replay-on-Demand (Qwen3.5-4B: 17-32% replay cuts forgetting 12-26x for about 0.02 nats of target loss). Languages without their own tokens do not improve (AfriqueLLM), so every language keeps its own share.

| Setting | Value | Source |
|---|---|---|
| Peak LR | pilot picks from 3e-5, 6e-5, 1e-4 | full-FT precedents span 1e-5 to 1.2e-4 |
| Schedule | 1% warmup, cosine to 10% | |
| Optimiser | AdamW, betas 0.9/0.95, weight decay 0.1 (not on norms, biases, A_log), clip 1.0 | Replay-on-Demand, AfriqueLLM |
| Batch | 256 x 4096 = 1.05M tokens per step | |
| Precision | fp32 master weights, bf16 autocast | Gated DeltaNet A_log and gated norm are stored fp32 |
| Kernels | flash-linear-attention 0.5.2, causal-conv1d 1.7.0, transformers 5.20.1 | the PyTorch fallback is about 2x slower |
| Loss | chunked cross-entropy over the 248k vocab | a single 4096-token sequence has 4 GB of fp32 logits |

## 5. Release gates

1. The CPT base beats Qwen3.5-2B-Base on UrduMMLU and Belebele urd_Arab, urd_Latn, pbt_Arab and snd_Arab, under one log-probability protocol.
2. English holds: Belebele eng_Latn and an MMLU subset drop by no more than 3 points versus Qwen3.5-2B-Base.
3. Held-out loss improves for every Tier-1 language.
4. The instruct model has under 2% invalid answers on generation-based UrduMMLU and beats Qalb (31.5-35.5).

## 6. Timeline

| Dates | Data (Kaggle) | Training (AWS) | Other |
|---|---|---|---|
| Fri 17 to Sun 19 Oct | PakLID v1, then routing in 5 parallel sessions | log in, account checks, quota requests filed | HF token in Kaggle/Colab secrets; FLORES+ terms |
| Mon 20 to Tue 21 Oct | dedup per language, data card, SFT data, tokenisation | S3 bucket, role, launch template; smoke test once quota lands | baseline evals on Colab A100 |
| Wed 22 to Thu 23 Oct | | pilot, then pick the LR | |
| Fri 24 to Thu 30 Oct | | main run, about 85-106 h on 1x H100; evals at every 1B tokens | **decision 26 Oct:** if no AWS quota by then, take Path B |
| Fri 31 Oct to Tue 4 Nov | | | SFT on Colab A100; optional Qwen3.8-27B-distilled instructions |
| Wed 5 to Sat 8 Nov | | | gates, model card, GGUF |
| Sun 9 to Wed 12 Nov | | | release, blog post, P1 tokenizer and P3 adaptation preprints |

**If the Gated DeltaNet stack proves too fragile** (kernel installs, NaNs, throughput far below 16k tokens/s): fall back to `Qwen/Qwen3-1.7B-Base`, a standard-attention model with no special kernels, accepting its worse tokenizer (Urdu 3.17 tokens/word vs 1.97).

**Path B (fallback, guaranteed):** native SFT only, on Qwen3.5-2B-Base, on Colab, released as `balagh-nano-v0-sft` and described honestly as instruction-tuned, not pretrained. Continued pretraining follows when quota or TRC arrives.

## 7. What the founder must do (only these need a human)

1. **Kaggle:** sign in inside Claude's built-in browser pane, so batch notebooks can be imported and run.
2. **Hugging Face:** create a fine-grained write token for the `balagh-lm` org and add it as a Kaggle secret and a Colab secret named `HF_TOKEN`. Accept the FLORES+ terms.
3. **AWS:** log in on this PC (`aws login` or `aws configure sso`) under a profile named `ownova`. Claude then runs the read-only checks, and files quota requests, the S3 bucket, the role, the budget and the launch only after the founder approves each one.
4. **AWS secret:** store the HF token in SSM Parameter Store yourself (`aws ssm put-parameter --name /balagh/hf_token --type SecureString --value <token> --profile ownova`), so the training instance can read it without the token passing through Claude.
