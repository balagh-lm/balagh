# Balagh (بلاغ)

**Open language models for the languages of Pakistan.** Apache-2.0 weights, open data, open benchmarks, reproducible training. Built by [Ownova](https://ownova.org), Lahore.

Pakistan has over 240 million people and some 70 languages. No open model serves more than Urdu, and the Urdu models that exist score below their own base model on native knowledge tests. Balagh is an attempt to fix that in the open.

## Languages

| Tier | Languages | What we aim for |
|---|---|---|
| 1 | Urdu, Punjabi (Shahmukhi), Pashto, Sindhi, Saraiki, Balochi | full chat and generation quality, Perso-Arabic and Roman script, code-switching |
| 2 | Brahui, Hindko, Kashmiri, Pahari-Pothwari | understanding, translation, best-effort generation |
| 3 | ~30 northern and smaller languages recorded in Common Voice | identification and translation to Urdu and English |

## What is here, and what is coming

| Artefact | Status |
|---|---|
| Tokenizer study: 45 tokenizer files measured on 10 Pakistani-language samples | done, paper in preparation |
| PakLID: a language and dialect classifier for Perso-Arabic Pakistani languages, with an audit of language-ID bleed in web corpora | v0 in progress |
| Open pretraining corpus with data cards and benchmark decontamination | in progress |
| Balagh-nano: a 2B-class model continued-pretrained on Pakistani languages | training on free compute |
| Controlled base-model bake-off (Gemma 4 vs Qwen3.5) on equal word budgets | planned |
| Native benchmarks for Shahmukhi, Saraiki, Balochi, Hindko, Brahui | planned |
| Balagh 12B flagship | gated on compute grants |

## Principles

- **No regression.** Nothing is released unless it beats the base model it started from on native benchmarks and keeps its English.
- **Native evaluation.** Judged on UrduMMLU, Belebele, FLORES+ and our own native suites, not on an LLM judge scoring the same categories as the training data.
- **Clean provenance.** Every training source is listed with its licence. Synthetic data comes only from Apache-2.0 teachers.
- **Everything public.** Weights, data cards, the classifier, evaluation outputs, training logs and configs.

## Team and contact

Muneeb ur Rehman (AI Engineer, Ownova) and [CO-FOUNDER]. Contact: muneeb@ownova.org. Native-speaker annotators and community contributors are credited in each release.

## Licence

Code and weights: Apache-2.0. Datasets: CC-BY-4.0 where rights allow; per-source licences in each data card.
