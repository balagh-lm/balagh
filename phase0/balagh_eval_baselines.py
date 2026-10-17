# Balagh evaluation harness: baselines and checkpoints under ONE protocol (Colab A100/L4, Kaggle T4 x2, AWS).
#
# Protocol: zero-shot multiple choice, English instruction + native content, answer = argmax of the
# log-probability of the option-letter token after "Answer:" (MMLU-style). This never produces invalid
# outputs, so numbers are not directly comparable with generation-based UrduMMLU scores (which count
# invalid outputs as wrong), but rankings are, and every model here (Qalb included) gets the same protocol.
#
# Sets: UrduMMLU (seeded N_MMLU subset), Belebele urd_Arab / urd_Latn (Roman Urdu) / pbt_Arab / snd_Arab,
# and two English forgetting checks: Belebele eng_Latn and an MMLU (cais/mmlu) seeded subset.
# Release gates use: native sets must beat Qwen3.5-2B-Base; English sets may drop at most 3 points.
#
# Models: MODELS below (baselines), or a checkpoint pushed by phase0/aws/train_cpt.py:
#   CKPT_REPO=balagh-lm/balagh-nano-ckpt CKPT_SUBFOLDER=balagh-nano-v0/tokens-1.0B  (name it with CKPT_NAME)
# Precision: bf16 on Ampere or newer; fp32 for models up to 5B on older GPUs (T4), because Qwen3.5's
# Gated DeltaNet is unreliable in fp16; fp16 otherwise. ONLY=name1,name2 restricts the baseline list.
# Needs HF_TOKEN only for private checkpoints (Colab/Kaggle secrets or env).

import os, json, time, random, math, sys, gc, traceback
import torch
from datasets import load_dataset

N_MMLU = int(os.environ.get("N_MMLU", 1000)); N_BELE = int(os.environ.get("N_BELE", 300)); N_ENG = int(os.environ.get("N_ENG", 500))
SEED = 0
OUT_DIR = next((d for d in ["/kaggle/working", "/content"] if os.path.isdir(d)), ".")
LETTERS = ["A", "B", "C", "D"]; ALL_LETTERS = ["A", "B", "C", "D", "E", "F"]

def secret(k):
    for get in (lambda: __import__("google.colab.userdata", fromlist=["get"]).get(k),
                lambda: __import__("kaggle_secrets").UserSecretsClient().get_secret(k)):
        try:
            v = get()
            if v: return v
        except Exception: pass
    return os.environ.get(k)
HF_TOKEN = secret("HF_TOKEN")

# (name, repo, quant, subfolder) ; quant in {None, "8bit", "4bit"}
MODELS = [
    ("qwen3.5-2B-base",          "Qwen/Qwen3.5-2B-Base",          None,   None),   # the Balagh-nano base: the gate
    ("qwen3.5-2B-instruct",      "Qwen/Qwen3.5-2B",               None,   None),
    ("qwen3.8-2B-distill (community)", "empero-ai/Qwen3.8-2B-Distill", None, None),   # what "Qwen 3.8 2B" on HF actually is
    ("qwen3.5-4B-base",          "Qwen/Qwen3.5-4B-Base",          None,   None),
    ("gemma-4-E2B-base",         "google/gemma-4-E2B",            None,   None),   # reference: best tokenizer on our languages
    ("gemma-4-E2B-it",           "google/gemma-4-E2B-it",         None,   None),
    ("qalb-1.0-8B-instruct",     "enstazao/Qalb-1.0-8B-Instruct", "4bit", None),   # the model to beat
]
if os.environ.get("CKPT_REPO"):
    MODELS = [(os.environ.get("CKPT_NAME", os.environ["CKPT_REPO"] + "/" + os.environ.get("CKPT_SUBFOLDER", "")), os.environ["CKPT_REPO"], None, os.environ.get("CKPT_SUBFOLDER") or None)]
elif os.environ.get("ONLY"):
    keep = set(os.environ["ONLY"].split(",")); MODELS = [m for m in MODELS if m[0] in keep]

# ------------------------------------------------------------------ data
def load_items():
    items = {"urdummlu": [], "belebele_urd": [], "belebele_urd_latn": [], "belebele_pbt": [], "belebele_snd": [], "belebele_eng": [], "mmlu_eng": []}
    mm = load_dataset("MBZUAI/UrduMMLU", split="test")
    idx = list(range(len(mm))); random.Random(SEED).shuffle(idx)
    for i in idx[:N_MMLU]:
        r = mm[i]; opts = r["options"] or {}
        keys = sorted(k for k, v in opts.items() if v)          # some items have 5 options (A-E) or blanks
        if r["correct_key"] not in keys or len(keys) < 2: continue
        items["urdummlu"].append({"id": r["id"], "q": r["question"], "letters": keys, "choices": [opts[k] for k in keys], "gold": keys.index(r["correct_key"]), "domain": r["domain"]})
    for code, key, n in [("urd_Arab", "belebele_urd", N_BELE), ("urd_Latn", "belebele_urd_latn", N_BELE), ("pbt_Arab", "belebele_pbt", N_BELE),
                         ("snd_Arab", "belebele_snd", N_BELE), ("eng_Latn", "belebele_eng", N_ENG)]:
        b = load_dataset("facebook/belebele", code, split="test")
        idx = list(range(len(b))); random.Random(SEED).shuffle(idx)
        for i in idx[:n]:
            r = b[i]
            items[key].append({"id": f"{code}-{r['question_number']}-{i}", "passage": r["flores_passage"], "q": r["question"], "letters": LETTERS,
                               "choices": [r[f"mc_answer{k}"] for k in range(1, 5)], "gold": int(r["correct_answer_num"]) - 1})
    try:
        en = load_dataset("cais/mmlu", "all", split="test")
        idx = list(range(len(en))); random.Random(SEED).shuffle(idx)
        for i in idx[:N_ENG]:
            r = en[i]
            items["mmlu_eng"].append({"id": f"mmlu-{i}", "q": r["question"], "letters": LETTERS, "choices": list(r["choices"]), "gold": int(r["answer"]), "domain": r["subject"]})
    except Exception as e:
        print("English MMLU unavailable:", str(e)[:120]); items.pop("mmlu_eng")
    return items

def prompt_of(it):
    head = "The following is a multiple choice question. Answer with the letter of the correct option.\n\n"
    if "passage" in it: head += "Passage: " + it["passage"].strip() + "\n\n"
    return head + "Question: " + it["q"].strip() + "\n" + "\n".join(f"{L}. {c}" for L, c in zip(it["letters"], it["choices"])) + "\nAnswer:"

# ------------------------------------------------------------------ model loading
def pick_dtype(n_params_guess):
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8: return torch.bfloat16
    return torch.float32 if n_params_guess <= 5e9 else torch.float16

def load_model(repo, quant, subfolder=None):
    import transformers
    from transformers import AutoTokenizer
    kw_common = {"token": HF_TOKEN}
    if subfolder: kw_common["subfolder"] = subfolder
    tok = AutoTokenizer.from_pretrained(repo, **kw_common)
    kw = {"device_map": "auto", "low_cpu_mem_usage": True, **kw_common}
    size_guess = 8e9 if "8B" in repo else (4.7e9 if "4B" in repo or "E4B" in repo else 2.5e9)
    if quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        cdt = pick_dtype(size_guess) if pick_dtype(size_guess) != torch.float32 else torch.float16
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True) if quant == "8bit" else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=cdt)
    else:
        kw["dtype"] = pick_dtype(size_guess)
    model = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            model = getattr(transformers, cls_name).from_pretrained(repo, **kw); break
        except Exception as e:
            print(f"  {cls_name} failed for {repo}: {str(e)[:160]}")
    if model is None: raise RuntimeError("could not load " + repo)
    model.eval()
    print(f"  loaded {type(model).__name__} dtype={next(model.parameters()).dtype} transformers={transformers.__version__}")
    return tok, model

def letter_token_ids(tok):
    ids = {}
    for L in ALL_LETTERS:
        cand = tok.encode(" " + L, add_special_tokens=False)
        if len(cand) != 1: cand = tok.encode(L, add_special_tokens=False)
        ids[L] = cand[-1]
    return ids

@torch.no_grad()
def score_items(tok, model, items, lids, max_len=2048, tag=""):
    correct = 0; n = 0; nonfinite = 0; per_domain = {}
    dev = next(model.parameters()).device; t0 = time.time()
    for k, it in enumerate(items):
        enc = tok(prompt_of(it), return_tensors="pt", truncation=True, max_length=max_len)
        enc = {a: b.to(dev) for a, b in enc.items() if a in ("input_ids", "attention_mask")}
        logits = model(**enc).logits[0, -1].float()
        if not torch.isfinite(logits).all(): nonfinite += 1
        pick = int(torch.argmax(logits[[lids[L] for L in it["letters"]]]))
        ok = int(pick == it["gold"]); correct += ok; n += 1
        d = it.get("domain", "all"); per_domain.setdefault(d, [0, 0]); per_domain[d][0] += ok; per_domain[d][1] += 1
        if (k + 1) % 200 == 0: print(f"    {tag} {k+1}/{len(items)} acc={correct/n:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return {"acc": correct / max(1, n), "n": n, "nonfinite_logits": nonfinite, "per_domain": {d: round(v[0] / v[1], 3) for d, v in per_domain.items()}, "seconds": round(time.time() - t0)}

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    items = load_items(); print({k: len(v) for k, v in items.items()})
    results_path = os.path.join(OUT_DIR, "balagh_eval_results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    fresh_ckpt = bool(os.environ.get("CKPT_REPO"))   # a checkpoint name can be re-pushed with new weights: never reuse its cache
    for name, repo, quant, sub in MODELS:
        prev = results.get(name)
        if prev and (fresh_ckpt or (prev.get("repo"), prev.get("subfolder"), prev.get("quant")) != (repo, sub, quant)): results.pop(name); prev = None
        if prev and all(ds in prev and prev[ds].get("n") == len(its) for ds, its in items.items()): print("skip (done):", name); continue
        print(f"\n=== {name} ({repo}{'/' + sub if sub else ''}, quant={quant}) ===", flush=True)
        tok = model = None
        try:
            tok, model = load_model(repo, quant, sub); lids = letter_token_ids(tok)
            entry = results.setdefault(name, {}); entry.update({"repo": repo, "subfolder": sub, "quant": quant, "dtype": str(next(model.parameters()).dtype)}); entry.pop("error", None)
            for ds, its in items.items():
                if ds in entry and entry[ds].get("n") == len(its): continue   # sample sizes must match the current knobs
                entry[ds] = score_items(tok, model, its, lids, tag=f"{name}/{ds}")
                print(f"  {ds}: acc={entry[ds]['acc']:.3f} n={entry[ds]['n']} nonfinite={entry[ds]['nonfinite_logits']}", flush=True)
                json.dump(results, open(results_path, "w"), indent=1)
        except Exception as e:
            results.setdefault(name, {"repo": repo, "subfolder": sub, "quant": quant})["error"] = str(e)[:400]
            print("  FAILED:", str(e)[:400]); traceback.print_exc(limit=2)
            json.dump(results, open(results_path, "w"), indent=1)
        finally:
            tok = model = None; gc.collect(); torch.cuda.empty_cache()   # release after the except frame is gone
    cols = [("urdummlu", "UrduMMLU"), ("belebele_urd", "Bele urd"), ("belebele_urd_latn", "Bele Roman Urdu"), ("belebele_pbt", "Bele pbt"),
            ("belebele_snd", "Bele snd"), ("belebele_eng", "Bele eng"), ("mmlu_eng", "MMLU eng")]
    rows = ["| model | " + " | ".join(c[1] for c in cols) + " | note |", "|---|" + "---|" * (len(cols) + 1)]
    for name, r in results.items():
        if "error" in r and "urdummlu" not in r: rows.append(f"| {name} | error |" + " |" * (len(cols) - 1) + f" {r['error'][:80]} |"); continue
        g = lambda ds: (f"{r[ds]['acc']*100:.1f}" if ds in r else "-")
        note = "nonfinite logits!" if any(r.get(ds, {}).get("nonfinite_logits", 0) for ds, _ in cols) else r.get("dtype", "")
        rows.append(f"| {name} | " + " | ".join(g(c[0]) for c in cols) + f" | {note} |")
    report = "\n".join(rows); open(os.path.join(OUT_DIR, "balagh_eval_report.md"), "w").write(report)
    print("\n================ REPORT ================")
    print(f"protocol: zero-shot MC, English instruction, letter log-prob; N_MMLU={N_MMLU}, N_BELE={N_BELE}/language, N_ENG={N_ENG}, seed={SEED}\n")
    print(report)
    print("\nchance = 25.0 (UrduMMLU has some 5-option items). Qalb's published generation-based UrduMMLU: 31.5-35.5; Llama-3.1-8B 43.3-43.9; Qwen3-4B-Instruct 50.9.")
