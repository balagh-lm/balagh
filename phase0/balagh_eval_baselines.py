# Balagh Phase 0: baseline native-benchmark evaluation on Kaggle GPU T4 x2
#
# Purpose: the "before" numbers for the no-regression gate and the bake-off paper.
# Scores every candidate base/instruct model and Qalb under ONE protocol:
#   zero-shot multiple choice, English instruction + Urdu content, answer chosen by the
#   log-probability of the letter tokens " A"/" B"/" C"/" D" after "Answer:" (MMLU-style).
# This never produces "invalid outputs", so numbers are not directly comparable with the UrduMMLU
# paper's generation-based scores (which count invalid outputs as wrong), but the ranking is, and
# Qalb is scored under the same protocol so the comparison with Balagh is apples to apples.
#
# Datasets: MBZUAI/UrduMMLU (26,431 items; a fixed random subset of N_MMLU is used per run),
#           facebook/belebele urd_Arab / pbt_Arab / snd_Arab (900 each; N_BELE per language).
# Kaggle: Settings -> Accelerator GPU T4 x2, Internet ON. First cell installs deps.

import os, json, time, random, math, sys, gc, traceback
import torch
from datasets import load_dataset

N_MMLU = int(os.environ.get("N_MMLU", 1000))      # items from UrduMMLU (seeded sample)
N_BELE = int(os.environ.get("N_BELE", 300))       # items per Belebele language
SEED = 0
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
LETTERS = ["A", "B", "C", "D"]
ALL_LETTERS = ["A", "B", "C", "D", "E", "F"]

# (name, repo, quant) ; quant in {None, "8bit", "4bit"}
MODELS = [
    ("gemma-4-E2B-base",     "google/gemma-4-E2B",            None),
    ("gemma-4-E2B-it",       "google/gemma-4-E2B-it",         None),
    ("gemma-4-E4B-base",     "google/gemma-4-E4B",            "8bit"),
    ("gemma-4-E4B-it",       "google/gemma-4-E4B-it",         "8bit"),
    ("qwen3.5-2B-base",      "Qwen/Qwen3.5-2B-Base",          None),
    ("qwen3.5-2B-instruct",  "Qwen/Qwen3.5-2B",               None),
    ("qwen3.5-4B-base",      "Qwen/Qwen3.5-4B-Base",          None),
    ("qwen3.5-4B-instruct",  "Qwen/Qwen3.5-4B",               None),
    ("qalb-1.0-8B-instruct", "enstazao/Qalb-1.0-8B-Instruct", "4bit"),
]
ONLY = os.environ.get("ONLY")  # comma-separated model names to restrict a run
if ONLY:
    keep = set(ONLY.split(","))
    MODELS = [m for m in MODELS if m[0] in keep]

# ------------------------------------------------------------------ data
def load_items():
    random.seed(SEED)
    items = {"urdummlu": [], "belebele_urd": [], "belebele_pbt": [], "belebele_snd": []}
    mm = load_dataset("MBZUAI/UrduMMLU", split="test")
    idx = list(range(len(mm))); random.shuffle(idx)
    for i in idx[:N_MMLU]:
        r = mm[i]; opts = r["options"] or {}
        keys = sorted(k for k, v in opts.items() if v)          # some items have 5 options (A-E) or blanks
        if r["correct_key"] not in keys or len(keys) < 2: continue
        items["urdummlu"].append({"id": r["id"], "q": r["question"], "letters": keys, "choices": [opts[k] for k in keys], "gold": keys.index(r["correct_key"]), "domain": r["domain"]})
    for code, key in [("urd_Arab", "belebele_urd"), ("pbt_Arab", "belebele_pbt"), ("snd_Arab", "belebele_snd")]:
        b = load_dataset("facebook/belebele", code, split="test")
        idx = list(range(len(b))); random.seed(SEED); random.shuffle(idx)
        for i in idx[:N_BELE]:
            r = b[i]
            items[key].append({"id": f"{code}-{r['question_number']}-{i}", "passage": r["flores_passage"], "q": r["question"], "letters": LETTERS, "choices": [r[f"mc_answer{k}"] for k in range(1, 5)], "gold": int(r["correct_answer_num"]) - 1})
    return items

def prompt_of(it):
    head = "The following is a multiple choice question. Answer with the letter of the correct option.\n\n"
    if "passage" in it:
        head += "Passage: " + it["passage"].strip() + "\n\n"
    body = "Question: " + it["q"].strip() + "\n" + "\n".join(f"{L}. {c}" for L, c in zip(it["letters"], it["choices"])) + "\nAnswer:"
    return head + body

# ------------------------------------------------------------------ model loading
def load_model(repo, quant):
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(repo)
    cfg = AutoConfig.from_pretrained(repo)
    kw = {"device_map": "auto", "low_cpu_mem_usage": True}
    if quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True) if quant == "8bit" else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["torch_dtype"] = torch.float16
    model = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(repo, **kw); break
        except Exception as e:
            print(f"  {cls_name} failed for {repo}: {str(e)[:160]}")
    if model is None: raise RuntimeError("could not load " + repo)
    model.eval()
    return tok, model

def letter_token_ids(tok):
    ids = {}
    for L in ALL_LETTERS:
        cand = tok.encode(" " + L, add_special_tokens=False)
        if len(cand) != 1: cand = tok.encode(L, add_special_tokens=False)
        ids[L] = cand[-1]
    return ids

@torch.no_grad()
def score_items(tok, model, items, lids, max_len=1536, tag=""):
    correct = 0; n = 0; nonfinite = 0; per_domain = {}
    dev = next(model.parameters()).device
    t0 = time.time()
    for k, it in enumerate(items):
        enc = tok(prompt_of(it), return_tensors="pt", truncation=True, max_length=max_len)
        enc = {a: b.to(dev) for a, b in enc.items() if a in ("input_ids", "attention_mask")}
        logits = model(**enc).logits[0, -1].float()
        if not torch.isfinite(logits).all(): nonfinite += 1
        lid = [lids[L] for L in it["letters"]]
        pick = int(torch.argmax(logits[lid]))
        ok = int(pick == it["gold"]); correct += ok; n += 1
        d = it.get("domain", "all"); per_domain.setdefault(d, [0, 0]); per_domain[d][0] += ok; per_domain[d][1] += 1
        if (k + 1) % 200 == 0: print(f"    {tag} {k+1}/{len(items)} acc={correct/n:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return {"acc": correct / max(1, n), "n": n, "nonfinite_logits": nonfinite, "per_domain": {d: round(v[0] / v[1], 3) for d, v in per_domain.items()}, "seconds": round(time.time() - t0)}

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    items = load_items()
    print({k: len(v) for k, v in items.items()})
    results_path = os.path.join(OUT_DIR, "balagh_eval_results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    for name, repo, quant in MODELS:
        if name in results and all(ds in results[name] for ds in items):
            print("skip (done):", name); continue
        print(f"\n=== {name} ({repo}, quant={quant}) ===", flush=True)
        try:
            tok, model = load_model(repo, quant)
            lids = letter_token_ids(tok)
            results.setdefault(name, {"repo": repo, "quant": quant})
            for ds, its in items.items():
                if ds in results[name]: continue
                results[name][ds] = score_items(tok, model, its, lids, tag=f"{name}/{ds}")
                print(f"  {ds}: acc={results[name][ds]['acc']:.3f} n={results[name][ds]['n']} nonfinite={results[name][ds]['nonfinite_logits']}", flush=True)
                json.dump(results, open(results_path, "w"), indent=1)
            del model, tok; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            results.setdefault(name, {"repo": repo, "quant": quant})["error"] = str(e)[:400]
            print("  FAILED:", str(e)[:400]); traceback.print_exc(limit=2)
            json.dump(results, open(results_path, "w"), indent=1)
            gc.collect(); torch.cuda.empty_cache()
    # report
    rows = ["| model | UrduMMLU (n) | Belebele urd | Belebele pbt | Belebele snd | note |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        if "error" in r and "urdummlu" not in r: rows.append(f"| {name} | error | | | | {r['error'][:80]} |"); continue
        g = lambda ds: (f"{r[ds]['acc']*100:.1f} ({r[ds]['n']})" if ds in r else "-")
        note = "nonfinite logits!" if any(r.get(ds, {}).get("nonfinite_logits", 0) for ds in items) else ""
        rows.append(f"| {name} | {g('urdummlu')} | {g('belebele_urd')} | {g('belebele_pbt')} | {g('belebele_snd')} | {note} |")
    report = "\n".join(rows)
    open(os.path.join(OUT_DIR, "balagh_eval_report.md"), "w").write(report)
    print("\n================ REPORT (paste into the Phase 0 log) ================")
    print(f"protocol: zero-shot MC, English instruction, letter log-prob; N_MMLU={N_MMLU} seed={SEED}; N_BELE={N_BELE}/language\n")
    print(report)
    print("\nchance = 25.0. Qalb's published generation-based UrduMMLU: 31.5-35.5; Llama-3.1-8B 43.3-43.9; Gemma-3-4B-IT 45.0; Gemma-4-31B-IT 76.5.")
