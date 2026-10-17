# Balagh data pipeline, step 3: native instruction data for SFT -> balagh-lm/sft-v0
#
# Seeds (all open): Alif Urdu-Instruct (large-traversaal/urdu-instruct, CC-BY-SA-4.0, 51,686 rows),
# Aya Dataset Urdu/Punjabi/Pashto/Sindhi rows (CohereLabs/aya_dataset, Apache-2.0),
# Sindhi reasoning chat (nassimjp/Sindhi-Reasoning-Chat-Dataset, CC-BY-4.0, LLM-generated: flagged),
# Roman-Urdu-Parl pairs (Mavkif/Roman-Urdu-Parl-split, Apache-2.0) turned into transliteration tasks,
# SMOL sentence/document translation pairs (google/smol, CC-BY-4.0) turned into translation tasks,
# Belebele/FLORES are NEVER used here (evaluation only).
# Output: chat-format JSONL shards with a `source`, `licence`, `lang`, `register` (native|roman|mixed|translation)
# and `synthetic` flag per row, deduplicated, decontaminated, with a train/dev split. CPU runtime.

import os, io, re, json, random, hashlib, collections
import pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi

OUT = os.environ.get("BALAGH_SFT_REPO", "balagh-lm/sft-v0")
random.seed(0)
def secret(k):
    """Colab Secrets, then Kaggle Secrets, then the environment, then a local `hf auth login`."""
    try:
        from google.colab import userdata
        v = userdata.get(k)
        if v: return v
    except Exception: pass
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret(k)
        if v: return v
    except Exception: pass
    if os.environ.get(k): return os.environ[k]
    if k == "HF_TOKEN":
        try:
            from huggingface_hub import get_token
            return get_token()
        except Exception: return None
    return None
TOKEN = secret("HF_TOKEN"); api = HfApi(token=TOKEN); api.create_repo(OUT, repo_type="dataset", exist_ok=True)

DIG = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}; DIG.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
def norm(t): return re.sub(r"[ \t]+", " ", t.translate(DIG).replace("ـ", "").replace("‌", "")).strip()

rows = []
def add(messages, source, licence, lang, register, synthetic=False):
    if any(len(m["content"]) < 2 for m in messages): return
    rows.append({"messages": messages, "source": source, "licence": licence, "lang": lang, "register": register, "synthetic": synthetic})

# 1. Alif Urdu-Instruct
try:
    d = load_dataset("large-traversaal/urdu-instruct", split="train", token=TOKEN)
    for r in d:
        ins, out = norm(r["instruction"]), norm(r["output"])
        add([{"role": "user", "content": ins}, {"role": "assistant", "content": out}], "alif-urdu-instruct", "CC-BY-SA-4.0", "urd", "native", synthetic=True)  # Alif's set is partly LLM-generated/translated
    print("alif:", len(d))
except Exception as e: print("alif skip", str(e)[:100])

# 2. Aya dataset (human-written) for Pakistani languages
try:
    d = load_dataset("CohereLabs/aya_dataset", split="train", token=TOKEN)
    keep = {"Urdu": "urd", "Punjabi": "pan", "Pashto": "pbt", "Sindhi": "snd", "Persian": None}
    n = collections.Counter()
    for r in d:
        lang = keep.get(r.get("language"))
        if not lang: continue
        add([{"role": "user", "content": norm(r["inputs"])}, {"role": "assistant", "content": norm(r["targets"])}], "aya-dataset", "Apache-2.0", lang, "native"); n[lang] += 1
    print("aya:", dict(n))
except Exception as e: print("aya skip", str(e)[:100])

# 3. Sindhi reasoning chat (LLM-generated; keep but flag)
try:
    d = load_dataset("nassimjp/Sindhi-Reasoning-Chat-Dataset", split="train", token=TOKEN)
    cols = d.column_names; n = 0
    for r in d:
        if "messages" in cols and isinstance(r["messages"], list):
            msgs = [{"role": m.get("role", "user"), "content": norm(str(m.get("content", "")))} for m in r["messages"]]
        else:
            q = r.get("question") or r.get("instruction") or r.get("prompt"); a = r.get("answer") or r.get("response") or r.get("output")
            if not (q and a): continue
            msgs = [{"role": "user", "content": norm(str(q))}, {"role": "assistant", "content": norm(str(a))}]
        add(msgs, "sindhi-reasoning-chat", "CC-BY-4.0", "snd", "native", synthetic=True); n += 1
    print("sindhi:", n)
except Exception as e: print("sindhi skip", str(e)[:100])

# 4. Roman-Urdu-Parl -> transliteration both ways (sampled; the full set is 6.3M pairs)
try:
    d = load_dataset("Mavkif/Roman-Urdu-Parl-split", split="train", streaming=True, token=TOKEN)
    n = 0
    PROMPTS_R2U = ["Convert this Roman Urdu to Urdu script:\n{x}", "یہ رومن اردو ہے، اسے اردو رسم الخط میں لکھیں:\n{x}", "Roman Urdu ko Urdu script mein likhein:\n{x}"]
    PROMPTS_U2R = ["Write this Urdu in Roman Urdu:\n{x}", "اسے رومن اردو میں لکھیں:\n{x}", "Is Urdu ko Roman Urdu mein likhein:\n{x}"]
    for r in d:
        u, ro = norm(r.get("Urdu text", "")), norm(r.get("Roman-Urdu text", ""))
        if len(u) < 15 or len(ro) < 15 or len(u) > 400: continue
        if random.random() < 0.5: add([{"role": "user", "content": random.choice(PROMPTS_R2U).format(x=ro)}, {"role": "assistant", "content": u}], "roman-urdu-parl", "Apache-2.0", "urd", "roman")
        else: add([{"role": "user", "content": random.choice(PROMPTS_U2R).format(x=u)}, {"role": "assistant", "content": ro}], "roman-urdu-parl", "Apache-2.0", "urd", "roman")
        n += 1
        if n >= 40000: break
    print("roman-urdu-parl:", n)
except Exception as e: print("roman-urdu-parl skip", str(e)[:100])

# 5. SMOL translation pairs (en <-> ur/ps/brh/pa-Arab/ks/bal/skr), sentence and document level
LANGNAME = {"ur": ("Urdu", "urd"), "ps": ("Pashto", "pbt"), "brh": ("Brahui", "brh"), "pa-Arab": ("Punjabi (Shahmukhi)", "pnb"), "pa": ("Punjabi", "pan"), "ks": ("Kashmiri", "kas"), "bal": ("Balochi", "bal"), "skr": ("Saraiki", "skr"), "sd-Deva": ("Sindhi (Devanagari)", "snd")}
for cfg in ["smoldoc__en_brh", "smoldoc__en_ps", "smoldoc__en_ur", "smolsent__en_brh", "smolsent__en_ks", "smolsent__en_pa-Arab", "smolsent__en_ur", "gatitos__en_bal", "gatitos__en_ks", "gatitos__en_pa-Arab", "gatitos__en_skr"]:
    try:
        d = load_dataset("google/smol", cfg, split="train", token=TOKEN); code = cfg.split("__en_")[-1]; name, lang = LANGNAME.get(code, (code, code)); n = 0
        for r in d:
            src = r.get("src") or r.get("source") or r.get("en") or ""; tgt = r.get("trg") or r.get("target") or r.get(code) or ""
            if isinstance(src, list): src = " ".join(src)
            if isinstance(tgt, list): tgt = " ".join(tgt)
            src, tgt = norm(str(src)), norm(str(tgt))
            if len(src) < 3 or len(tgt) < 3: continue
            if cfg.startswith("gatitos") and random.random() < 0.7: continue   # word lists: keep a small share
            if random.random() < 0.5: add([{"role": "user", "content": f"Translate to {name}:\n{src}"}, {"role": "assistant", "content": tgt}], f"smol-{cfg}", "CC-BY-4.0", lang, "translation")
            else: add([{"role": "user", "content": f"Translate to English:\n{tgt}"}, {"role": "assistant", "content": src}], f"smol-{cfg}", "CC-BY-4.0", lang, "translation")
            n += 1
        print(cfg, n)
    except Exception as e: print("smol skip", cfg, str(e)[:80])

# dedup + decontamination (question text against eval sets)
seen = set(); dedup = []
for r in rows:
    key = hashlib.sha1((r["messages"][0]["content"] + "|" + r["messages"][-1]["content"]).encode()).hexdigest()
    if key in seen: continue
    seen.add(key); dedup.append(r)
print("rows:", len(rows), "after dedup:", len(dedup))
evalq = set()
try:
    for r in load_dataset("MBZUAI/UrduMMLU", split="test", token=TOKEN): evalq.add(norm(r["question"])[:80])
    for cfg in ["urd_Arab", "pbt_Arab", "snd_Arab"]:
        for r in load_dataset("facebook/belebele", cfg, split="test", token=TOKEN): evalq.add(norm(r["question"])[:80])
except Exception as e: print("decontam partial:", str(e)[:80])
clean = [r for r in dedup if norm(r["messages"][0]["content"])[:80] not in evalq]
print("after decontam:", len(clean))
random.shuffle(clean); dev = clean[:2000]; train = clean[2000:]
stats = collections.Counter((r["source"], r["lang"], r["register"]) for r in clean)

def push(name, rs):
    buf = io.BytesIO(); buf.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rs).encode("utf-8")); buf.seek(0)
    api.upload_file(path_or_fileobj=buf, path_in_repo=f"{name}.jsonl", repo_id=OUT, repo_type="dataset", commit_message=f"{name} {len(rs)} rows")
push("train", train); push("dev", dev)
card = "---\nlicense: other\nlicense_name: mixed-open\npretty_name: Balagh SFT v0\n---\n# Balagh SFT v0\n\nNative-language instruction data for the languages of Pakistan, assembled from openly licensed sources. Chat format (`messages`), with `source`, `licence`, `lang`, `register` and `synthetic` per row. Belebele and FLORES are never included (evaluation only); rows whose prompt matches an UrduMMLU or Belebele question are removed.\n\n| source | lang | register | rows |\n|---|---|---|---|\n" + "\n".join(f"| {s} | {l} | {g} | {n:,} |" for (s, l, g), n in sorted(stats.items())) + f"\n\nTrain {len(train):,}, dev {len(dev):,}. Built by `phase0/colab/03_sft_data.ipynb` in the Balagh repo.\n"
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=OUT, repo_type="dataset", commit_message="data card")
print("DONE", dict(stats))
