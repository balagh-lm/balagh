# Tokenizer cost of Pakistan's languages across widely used LLM tokenizers (Balagh-Eval / paper 2).
#
# Parity (Petrov et al. 2023): on parallel sentences, tokens(language) / tokens(English), averaged over pairs.
#   Parallel sources: FLORES+ devtest (urd, pbt, snd, kas Arabic script; pan_Guru as a Gurmukhi contrast) and
#   SMOL smolsent (ur, pa-Arab = Shahmukhi, brh, ks). Needs the FLORES+ terms accepted on the Hub.
# Fertility: tokens per whitespace word and characters per token on held-out corpus-v1 documents for every language
#   (the only option for Saraiki, Balochi, Hindko, Pothwari and Roman Urdu, which have no parallel set).
# Output: tokenizer_study.json + tokenizer_study.md in OUT (default ./). CPU only.

import os, json, random, statistics, traceback
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

OUT = os.environ.get("OUT", ".")
N_DOCS = int(os.environ.get("N_DOCS", 300)); SEED = 0

TOKENIZERS = [   # (label, kind, id)
    ("Qwen3.5 (Balagh base)", "hf", "Qwen/Qwen3.5-2B-Base"),
    ("Gemma 4", "hf", "google/gemma-4-E2B"),
    ("Llama 3.1 (Qalb, Alif)", "hf", "enstazao/Qalb-1.0-8B-Instruct"),
    ("GPT-4o o200k", "tiktoken", "o200k_base"),
    ("GPT-4 cl100k", "tiktoken", "cl100k_base"),
    ("BLOOM", "hf", "bigscience/bloom-560m"),
    ("mT5", "hf", "google/mt5-base"),
    ("XLM-R", "hf", "FacebookAI/xlm-roberta-base"),
]

def load_tok(kind, ident):
    if kind == "tiktoken":
        import tiktoken; enc = tiktoken.get_encoding(ident)
        return lambda t: len(enc.encode(t, disallowed_special=()))
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(ident)
    return lambda t: len(tk(t, add_special_tokens=False)["input_ids"])

def parallel_sets():
    from datasets import load_dataset
    sets = {}
    eng = {r["id"]: r["text"] for r in load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")}
    for code in ["urd_Arab", "pbt_Arab", "snd_Arab", "kas_Arab", "pan_Guru"]:
        d = load_dataset("openlanguagedata/flores_plus", code, split="devtest")
        sets[f"FLORES {code}"] = [(eng[r["id"]], r["text"]) for r in d if r["id"] in eng]
    for code, name in [("ur", "urd_Arab"), ("pa-Arab", "pnb_Arab (Shahmukhi)"), ("brh", "brh_Arab"), ("ks", "kas_Arab")]:
        p = hf_hub_download("google/smol", f"smolsent/en_{code}.jsonl", repo_type="dataset")
        pairs = []
        for line in open(p, encoding="utf-8"):
            r = json.loads(line); src, trg = r.get("src"), r.get("trg")
            if isinstance(src, str) and isinstance(trg, str) and src.strip() and trg.strip(): pairs.append((src, trg))
        sets[f"SMOL {name}"] = pairs
    return sets

def corpus_docs():
    out = {}
    for lang in ["urd_Arab", "urd_Latn", "pbt_Arab", "snd_Arab", "pnb_Arab", "skr_Arab", "bal_Arab", "brh_Arab", "kas_Arab", "hno_Arab", "phr_Arab"]:
        p = hf_hub_download("balagh-lm/corpus-v1", f"{lang}/shard-00000.parquet", repo_type="dataset")
        pf = pq.ParquetFile(p)
        texts = pf.read_row_group(0, columns=["text"]).column("text").to_pylist()
        random.Random(SEED).shuffle(texts)
        out[lang] = [t[:3000] for t in texts[:N_DOCS]]
    return out

if __name__ == "__main__":
    par = parallel_sets(); docs = corpus_docs()
    print({k: len(v) for k, v in par.items()}, {k: len(v) for k, v in docs.items()}, flush=True)
    res = {}
    for label, kind, ident in TOKENIZERS:
        try: n = load_tok(kind, ident)
        except Exception as e: print("skip", label, str(e)[:120]); res[label] = {"error": str(e)[:200]}; continue
        r = {"id": ident, "parity": {}, "fertility": {}}
        for k, pairs in par.items():
            ratios = [n(x) / max(1, n(e)) for e, x in pairs]
            r["parity"][k] = {"mean": round(statistics.mean(ratios), 3), "pairs": len(pairs)}
        for lang, ts in docs.items():
            tok = sum(n(t) for t in ts); words = sum(len(t.split()) for t in ts); chars = sum(len(t) for t in ts)
            r["fertility"][lang] = {"tokens_per_word": round(tok / words, 3), "chars_per_token": round(chars / tok, 3), "docs": len(ts)}
        res[label] = r; print("done", label, flush=True)
    json.dump(res, open(os.path.join(OUT, "tokenizer_study.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    ok = [k for k, v in res.items() if "error" not in v]
    pk = list(par); fl = list(docs)
    md = ["## Parity: tokens relative to the same English sentence (1.0 = as cheap as English)", "",
          "| tokenizer | " + " | ".join(pk) + " |", "|---|" + "---|" * len(pk)]
    md += [f"| {k} | " + " | ".join(f"{res[k]['parity'][p]['mean']:.2f}" for p in pk) + " |" for k in ok]
    md += ["", "## Fertility: tokens per word on corpus-v1 documents", "", "| tokenizer | " + " | ".join(fl) + " |", "|---|" + "---|" * len(fl)]
    md += [f"| {k} | " + " | ".join(f"{res[k]['fertility'][l]['tokens_per_word']:.2f}" for l in fl) + " |" for k in ok]
    md += [f"", "Skipped: " + ", ".join(f"{k} ({v['error'][:60]})" for k, v in res.items() if "error" in v)] if len(ok) < len(res) else []
    open(os.path.join(OUT, "tokenizer_study.md"), "w", encoding="utf-8").write("\n".join(md)); print("\n".join(md))
