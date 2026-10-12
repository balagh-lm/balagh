# Balagh-nano-v0: release plan, target 12 November 2026

**Decided 12 Oct 2026.** Use every free resource at once: Kaggle TPU v5e-8 on both accounts for pretraining, Colab (200 units) for data, evaluation and SFT, Colab TPU v6e-1 or the AMD $100 credit for the bake-off, Kaggle T4 x2 for extra evals. Release gate unchanged: beats its own base on UrduMMLU and Belebele urd/pbt/snd, keeps English, under 2% invalid outputs, and beats Qalb (31.5 to 35.5 on UrduMMLU).

## Two release paths, decided by the TPU queue on 26 October

- **Path A (full):** Gemma 4 E2B continued-pretrained on ~10B tokens of the cleaned corpus on Kaggle TPUs, then native SFT on Colab. Needs Kaggle batch TPU jobs to start within hours, not days.
- **Path B (fallback, guaranteed):** Gemma 4 E4B with native SFT only (QLoRA or LoRA on Colab A100), released as `balagh-nano-v0-sft`, honestly described as instruction-tuned, not pretrained. Still beats Qalb by a wide margin, since Gemma 4's own instruct model already does.

Both paths share weeks 1 to 2 entirely. The decision point is whether the first real pretraining session has run by 26 October.

## Week by week

| Week | Dates | Kaggle TPU (both accounts) | Colab (200 units) | Other |
|---|---|---|---|---|
| 1 | 12 to 18 Oct | TPU feasibility batch (queued); pretraining smoke test (30 min) as soon as the trainer notebook lands | Run 01 fetch and 02 clean (CPU, ~5 units); run 04 baseline evals on A100 (~25 units); read TPU v6e-1 burn rate | HF token in Colab Secrets; Kaggle API token in Colab Secrets (for dataset export); AMD program joined |
| 2 | 19 to 25 Oct | Session 1 to 4 of pretraining (9 h each, ~2.5B tokens/week/account) | 03 SFT data build; 05 tokenise + export corpus to a Kaggle dataset; bake-off arm 1 on v6e-1 if burn rate allows (~70 units) | PakLID v1 trained on independent sources (paper P2 data) |
| 3 | 26 Oct to 1 Nov | **Decision point 26 Oct.** Sessions 5 to 8; mid-run gate eval at ~5B tokens | Bake-off arm 2 or AMD run; eval harness finalised; SFT dry run on E2B (small) | Wikimedia Rapid Fund (community applicant) by 1 Nov |
| 4 | 2 to 8 Nov | Sessions 9 to 12, reaching ~10B tokens; final base checkpoint to the Hub | SFT on the pretrained checkpoint (Path A) or on E4B (Path B), ~100 units; gate evals; DPO if time allows | Model card, data card, blog draft |
| 5 | 9 to 12 Nov | Buffer for failed sessions | Final evals, GGUF export, Hub release of base + instruct | Release post; P1 tokenizer paper and P3 bake-off note to arXiv |

## Compute ledger (what each resource must deliver)

| Resource | Allowance | Assigned | Tokens or outputs |
|---|---|---|---|
| Kaggle TPU, account 1 | 20 h/week | pretraining, 9+9+2 h sessions | ~2.5B tokens/week at 30% MFU |
| Kaggle TPU, account 2 (Ali) | 20 h/week | pretraining, alternate data halves or a second seed for the bake-off | ~2.5B tokens/week |
| Kaggle T4 x2, both accounts | 30 h/week each | evals of checkpoints, small SFT | ~10 eval runs/week |
| Colab 200 units | per month | data 10 to 20, evals 20 to 40, SFT 100 to 140, bake-off remainder on v6e-1 | one SFT epoch at 12B or three at E4B |
| AMD Developer Cloud | $100 once | bake-off if v6e-1 burn rate is bad | 2 arms × ~3B tokens at 2B |
| Hugging Face Hub | unlimited public | all corpora, checkpoints, models, leaderboard | storage for ~1 TB of checkpoints |

## Plumbing rules that make 9-hour sessions survivable

1. Tokenised training data lives in a Kaggle dataset (fast local reads), produced once by notebook 05 from `balagh-lm/corpus-v1`.
2. Every pretraining session resumes from the latest checkpoint on the Hub and saves one every 1,000 steps plus at exit; a session is scheduled as a batch "Save & Run All" with a wall-clock guard that stops at 8h40m and uploads.
3. Checkpoints are bf16 weights plus 8-bit optimizer state (~14 GB for a 2B model) to keep uploads under 10 minutes.
4. Every checkpoint gets the gate evaluation on Kaggle T4 x2 or Colab before the next session starts; a regression stops the run and we fall back to the last good checkpoint with a lower learning rate.

## Risks that would move the date

- Kaggle TPU never allocates: Path B ships on time; pretraining continues on TRC when granted.
- Gemma 4 E2B unsupported by the TPU stack (torch_xla or JAX): switch the trainer to Qwen3.5-2B-Base, which has the same memory profile; the bake-off becomes the release model's selection.
- Data pipeline overruns (Urdu dedup is the long pole): ship with whatever languages have `stats.json`, Urdu first.
- Session failures: the buffer week absorbs up to four lost sessions.

## Paper outputs by 12 Nov

P1 tokenizer study (data complete), P3 bake-off note (if run), model card plus a short technical report for the release. P2 PakLID and P4 corpus card follow in December.
