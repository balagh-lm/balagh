# Balagh data pipeline, step 2: turn balagh-lm/raw-corpora-v0 into balagh-lm/corpus-v1.
#
# Per language: per-language normalisation (no global NFKC), quality filters, PakLID language check
# (the raw buckets bleed: FineWeb-2's "pnb" holds Urdu, GlotCC's "bal" holds Roman Urdu, etc.),
# exact dedup (sha1), near-dedup (MinHash, 128 perms, 16 bands x 8 rows ~= Jaccard 0.75),
# benchmark decontamination (13-word-gram overlap with UrduMMLU, Belebele, SIB-200, Global PIQA,
# FLORES+ if your token has accepted its terms), then parquet shards + stats + a data card to the Hub.
#
# Runs on a Colab High-RAM CPU runtime (no GPU needed); Urdu is the big one (~5M docs). Resumable per
# language: a language whose stats.json exists in the output repo is skipped unless FORCE=1.
# Needs HF_TOKEN in Colab Secrets. pip: datasets huggingface_hub pyarrow datasketch xxhash scikit-learn

import os, io, re, json, time, hashlib, gc, collections, random, traceback
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

RAW = os.environ.get("BALAGH_RAW_REPO", "balagh-lm/raw-corpora-v0")
OUT = os.environ.get("BALAGH_OUT_REPO", "balagh-lm/corpus-v1")
WORK = "/content/work" if os.path.isdir("/content") else "./work"
LANGS = [l for l in os.environ.get("LANGS", "pnb_Arab,skr_Arab,bal_Arab,brh_Arab,kas_Arab,snd_Arab,pbt_Arab,urd_Latn,urd_Arab").split(",") if l]
FORCE = os.environ.get("FORCE") == "1"
MAX_DOCS = int(os.environ.get("MAX_DOCS", 0)) or None
os.makedirs(WORK, exist_ok=True)

def hf_token():
    try:
        from google.colab import userdata; return userdata.get("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")
TOKEN = hf_token(); api = HfApi(token=TOKEN)
api.create_repo(OUT, repo_type="dataset", exist_ok=True)

# which raw sources feed which output language (source prefixes in the raw repo)
RAW_SOURCES = {
    "urd_Arab": ["fineweb2/urd_Arab", "hplt2/urd_Arab", "sangraha/urd", "wiki/ur", "omniasr/urd_Arab"],
    "urd_Latn": ["fineweb2/urd_Latn", "glotcc/urd-Latn"],
    "pbt_Arab": ["fineweb2/pbt_Arab", "hplt2/pbt_Arab", "wiki/ps", "omniasr/pbt_Arab"],
    "snd_Arab": ["fineweb2/snd_Arab", "hplt2/snd_Arab", "sangraha/snd", "wiki/sd", "omniasr/snd_Arab"],
    "pnb_Arab": ["fineweb2/pnb_Arab", "fineweb2removed/pnb_Arab", "glotcc/pnb-Arab", "wiki/pnb", "omniasr/pnb_Arab"],
    "skr_Arab": ["fineweb2/skr_Arab", "fineweb2removed/skr_Arab", "glotcc/skr-Arab", "wiki/skr", "omniasr/skr_Arab"],
    "bal_Arab": ["glotcc/bal-Arab", "omniasr/bal_Arab"],
    "brh_Arab": ["fineweb2/brh_Arab", "fineweb2removed/brh_Arab", "glotcc/brh-Arab", "omniasr/brh_Arab"],
    "kas_Arab": ["fineweb2/kas_Arab", "fineweb2removed/kas_Arab", "hplt2/kas_Arab", "sangraha/kas", "wiki/ks", "omniasr/kas_Arab"],
}
# the raw "removed" buckets and glotcc buckets are re-sorted by PakLID: a doc goes to the language
# PakLID predicts, so Urdu hiding in the Punjabi bucket ends up in Urdu, and vice versa.

# ------------------------------------------------------------------ normalisation (per language, no NFKC)
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}; DIGITS.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
PRESENTATION = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
COMMON_MAP = {"ي": "ی", "ك": "ک", "ـ": "", "‌": "", "‍": "", "‎": "", "‏": "", "﻿": ""}   # Arabic yeh/kaf -> Urdu-script forms; drop tatweel, ZW chars
LANG_MAP = {
    "snd_Arab": {"ی": "ي"},        # Sindhi orthography uses Arabic yeh; keep its own letters untouched
    "urd_Latn": {},
}
def normalise(text, lang):
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.translate(DIGITS)
    m = dict(COMMON_MAP); m.update(LANG_MAP.get(lang, {}))
    if lang.endswith("_Arab"):
        t = "".join(m.get(ch, ch) for ch in t)
        if PRESENTATION.search(t):               # PDF presentation forms -> decomposed letters (NFKC only on those spans)
            import unicodedata
            t = PRESENTATION.sub(lambda x: unicodedata.normalize("NFKC", x.group(0)), t)
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# ------------------------------------------------------------------ quality filters
BOILER = re.compile(r"(cookie|javascript|all rights reserved|click here|subscribe|login|password|©|\bhttp[s]?://)", re.I)
def quality(text, lang):
    n = len(text)
    if n < 200 or n > 200_000: return "length"
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 100: return "few_letters"
    if lang.endswith("_Arab"):
        share = sum(1 for c in letters if ARABIC.match(c)) / len(letters)
        if share < 0.80: return "script_share"
    else:
        share = sum(1 for c in letters if c.isascii()) / len(letters)
        if share < 0.90: return "script_share"
    words = text.split()
    if len(words) < 30: return "few_words"
    if sum(len(w) for w in words) / len(words) > 14: return "long_words"
    lines = [l for l in text.split("\n") if l.strip()]
    if lines and sum(1 for l in lines if BOILER.search(l)) / len(lines) > 0.3: return "boilerplate"
    top = collections.Counter(words).most_common(1)[0][1] / len(words)
    if top > 0.20: return "repetition"
    dup_lines = 1 - len(set(lines)) / max(1, len(lines))
    if dup_lines > 0.5: return "dup_lines"
    digits = sum(c.isdigit() for c in text) / n
    if digits > 0.25: return "digits"
    return None

# ------------------------------------------------------------------ PakLID (trained in-notebook on independent sources + raw samples)
def train_paklid():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    import joblib
    cache = os.path.join(WORK, "paklid.joblib")
    if os.path.exists(cache): return joblib.load(cache)
    print("training PakLID from seed samples ...")
    seeds = {  # label -> list of (raw repo prefix, docs to sample); prefer sources NOT produced by GlotLID where they exist
        "urd_Arab": [("wiki/ur", 3000), ("omniasr/urd_Arab", 3000), ("sangraha/urd", 3000), ("fineweb2/urd_Arab", 2000)],
        "urd_Latn": [("fineweb2/urd_Latn", 4000), ("glotcc/urd-Latn", 2000)],
        "pbt_Arab": [("wiki/ps", 3000), ("omniasr/pbt_Arab", 2000), ("hplt2/pbt_Arab", 3000)],
        "snd_Arab": [("wiki/sd", 3000), ("omniasr/snd_Arab", 2000), ("sangraha/snd", 3000)],
        "pnb_Arab": [("wiki/pnb", 4000), ("omniasr/pnb_Arab", 3000), ("fineweb2/pnb_Arab", 2000)],
        "skr_Arab": [("wiki/skr", 3000), ("omniasr/skr_Arab", 2000), ("fineweb2/skr_Arab", 2000)],
        "bal_Arab": [("omniasr/bal_Arab", 3000), ("glotcc/bal-Arab", 3000)],
        "brh_Arab": [("omniasr/brh_Arab", 3000), ("glotcc/brh-Arab", 2000), ("fineweb2/brh_Arab", 1000)],
        "kas_Arab": [("wiki/ks", 2000), ("omniasr/kas_Arab", 3000), ("sangraha/kas", 2000)],
        "hno_Arab": [("omniasr/hno_Arab", 4000)],
        "phr_Arab": [("omniasr/phr_Arab", 4000)],
    }
    X, y = [], []
    for lab, srcs in seeds.items():
        for prefix, n in srcs:
            try:
                ds = load_dataset(RAW, data_files=f"{prefix}/*.parquet", split="train", streaming=True, token=TOKEN)
                got = 0
                for row in ds:
                    t = normalise(row["text"], lab)
                    for chunk in re.split(r"\n+", t):
                        chunk = chunk.strip()
                        if 40 <= len(chunk) <= 600:
                            X.append(chunk.lower()); y.append(lab); got += 1
                            if got >= n: break
                    if got >= n: break
            except Exception as e:
                print("   seed skip", prefix, str(e)[:80])
    # confusers: Persian and Arabic from FineWeb-2 raw (not mirrored) via streaming
    for lab, cfg in [("fas_Arab", "fas_Arab"), ("arb_Arab", "arb_Arab")]:
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-2", cfg, split="train", streaming=True, token=TOKEN); got = 0
            for row in ds:
                for chunk in re.split(r"\n+", normalise(row["text"], lab)):
                    if 40 <= len(chunk.strip()) <= 600: X.append(chunk.strip().lower()); y.append(lab); got += 1
                    if got >= 3000: break
                if got >= 3000: break
        except Exception as e: print("   confuser skip", cfg, str(e)[:80])
    print("   PakLID training chunks:", len(X), collections.Counter(y))
    clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=3, sublinear_tf=True, max_features=800_000),
                        CalibratedClassifierCV(LinearSVC(C=0.5), cv=3))
    clf.fit(X, y); joblib.dump(clf, cache)
    return clf

def lid_batch(clf, texts):
    heads = [t[:600].lower() for t in texts]
    proba = clf.predict_proba(heads); labs = list(clf.classes_)
    idx = proba.argmax(1)
    return [labs[i] for i in idx], proba.max(1)

# ------------------------------------------------------------------ decontamination n-grams
def eval_ngrams(n=13):
    grams = set()
    def add(text):
        w = normalise(text, "urd_Arab").split()
        for i in range(len(w) - n + 1): grams.add(" ".join(w[i:i + n]))
    srcs = []
    try:
        d = load_dataset("MBZUAI/UrduMMLU", split="test", token=TOKEN); srcs.append(("urdummlu", [r["question"] + " " + " ".join((r["options"] or {}).values()) for r in d]))
    except Exception as e: print("decontam skip UrduMMLU", str(e)[:80])
    for cfg in ["urd_Arab", "urd_Latn", "pbt_Arab", "snd_Arab", "pan_Guru"]:
        try:
            d = load_dataset("facebook/belebele", cfg, split="test", token=TOKEN); srcs.append((f"belebele_{cfg}", [r["flores_passage"] + " " + r["question"] for r in d]))
        except Exception as e: print("decontam skip belebele", cfg, str(e)[:60])
    for cfg in ["urd_Arab", "pbt_Arab", "snd_Arab", "kas_Arab", "pan_Guru"]:
        try:
            d = load_dataset("Davlan/sib200", cfg, token=TOKEN); srcs.append((f"sib200_{cfg}", [r["text"] for s in d for r in d[s]]))
        except Exception as e: print("decontam skip sib200", cfg, str(e)[:60])
    for cfg in ["urd_arab", "urd_latn", "snd_arab", "bcc_arab", "bsk_arab"]:
        try:
            d = load_dataset("mrlbenchmarks/global-piqa-nonparallel", cfg, token=TOKEN); srcs.append((f"piqa_{cfg}", [" ".join(str(v) for v in r.values() if isinstance(v, str)) for s in d for r in d[s]]))
        except Exception as e: print("decontam skip piqa", cfg, str(e)[:60])
    for cfg in ["urd_Arab", "pbt_Arab", "snd_Arab", "kas_Arab"]:
        try:
            d = load_dataset("openlanguagedata/flores_plus", cfg, token=TOKEN); srcs.append((f"flores_{cfg}", [r["text"] for s in d for r in d[s]]))
        except Exception as e: print("decontam skip flores+ (accept terms on HF for your token)", cfg, str(e)[:60])
    for name, texts in srcs:
        for t in texts: add(t)
    print(f"decontamination: {len(grams):,} {n}-grams from {len(srcs)} eval sets")
    return grams

def contaminated(text, grams, n=13):
    w = text.split()
    for i in range(0, max(1, len(w) - n + 1)):
        if " ".join(w[i:i + n]) in grams: return True
    return False

# ------------------------------------------------------------------ MinHash near-dedup
import xxhash
from datasketch import MinHash
NUM_PERM, BANDS, ROWS = 128, 16, 8
def signature(text):
    w = text.split(); shingles = {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}
    m = MinHash(num_perm=NUM_PERM, hashfunc=lambda b: xxhash.xxh32(b).intdigest())
    m.update_batch([s.encode("utf-8") for s in shingles])
    return m.hashvalues.astype(np.uint32)

def lsh_dedup(sigs):
    """sigs: (N, NUM_PERM) uint32. Returns boolean keep mask (first occurrence kept)."""
    N = sigs.shape[0]; keep = np.ones(N, dtype=bool)
    for b in range(BANDS):
        band = np.ascontiguousarray(sigs[:, b * ROWS:(b + 1) * ROWS]).view(np.dtype((np.void, ROWS * 4))).ravel()
        seen = {}
        for i in range(N):
            if not keep[i]: continue
            k = band[i].tobytes()
            j = seen.get(k)
            if j is None: seen[k] = i
            else: keep[i] = False
        del seen; gc.collect()
    return keep

# ------------------------------------------------------------------ main per-language pass
def process_language(lang, clf, grams, done_files):
    if not FORCE and f"{lang}/stats.json" in done_files:
        print(f"skip {lang} (stats.json present)"); return
    t0 = time.time(); stats = collections.Counter(); seen_sha = set(); texts_meta = []; sig_list = []
    reassigned = collections.Counter()
    tmp = os.path.join(WORK, lang); os.makedirs(tmp, exist_ok=True)
    for prefix in RAW_SOURCES[lang]:
        try:
            ds = load_dataset(RAW, data_files=f"{prefix}/*.parquet", split="train", streaming=True, token=TOKEN)
        except Exception as e:
            print(f"   no raw shards for {prefix}: {str(e)[:80]}"); continue
        batch = []
        def flush(batch):
            labs, confs = lid_batch(clf, [b[0] for b in batch])
            for (text, meta, src), lab, conf in zip(batch, labs, confs):
                if lab != lang or conf < 0.6:
                    stats["lid_other"] += 1; reassigned[lab] += 1; continue
                sha = hashlib.sha1(text.encode()).hexdigest()
                if sha in seen_sha: stats["exact_dup"] += 1; continue
                seen_sha.add(sha)
                if contaminated(text, grams): stats["contaminated"] += 1; continue
                sig_list.append(signature(text)); texts_meta.append((text, meta, src, sha)); stats["kept_pre_minhash"] += 1
        for row in ds:
            stats["raw"] += 1
            text = normalise(row["text"], lang); meta = row.get("meta", "{}")
            reason = quality(text, lang)
            if reason: stats["filtered_" + reason] += 1; continue
            batch.append((text, meta, prefix))
            if len(batch) >= 2000: flush(batch); batch = []
            if stats["raw"] % 100000 == 0: print(f"   {lang} {prefix}: raw {stats['raw']:,} kept {stats['kept_pre_minhash']:,} ({time.time()-t0:.0f}s)", flush=True)
            if MAX_DOCS and stats["raw"] >= MAX_DOCS: break
        if batch: flush(batch)
    if not texts_meta:
        print(f"   {lang}: nothing kept"); return
    sigs = np.stack(sig_list); del sig_list; gc.collect()
    keep = lsh_dedup(sigs); stats["near_dup"] = int((~keep).sum()); stats["kept"] = int(keep.sum())
    # write shards
    SH = 50000; n = 0; words = 0; chars = 0
    kept_rows = [texts_meta[i] for i in range(len(texts_meta)) if keep[i]]
    del texts_meta; gc.collect()
    for s in range(0, len(kept_rows), SH):
        chunk = kept_rows[s:s + SH]
        table = pa.table({"text": [r[0] for r in chunk], "meta": [r[1] for r in chunk], "source": [r[2] for r in chunk], "sha1": [r[3] for r in chunk]})
        words += sum(len(r[0].split()) for r in chunk); chars += sum(len(r[0]) for r in chunk)
        buf = io.BytesIO(); pq.write_table(table, buf, compression="zstd"); buf.seek(0)
        api.upload_file(path_or_fileobj=buf, path_in_repo=f"{lang}/shard-{n:05d}.parquet", repo_id=OUT, repo_type="dataset", commit_message=f"{lang} shard {n}")
        n += 1
    stats.update({"shards": n, "words": words, "chars": chars, "seconds": int(time.time() - t0), "lid_reassigned_to": dict(reassigned.most_common(8))})
    api.upload_file(path_or_fileobj=json.dumps(dict(stats), indent=1).encode(), path_in_repo=f"{lang}/stats.json", repo_id=OUT, repo_type="dataset", commit_message=f"{lang} stats")
    print(f"   {lang} DONE: raw {stats['raw']:,} -> kept {stats['kept']:,} docs, {words/1e6:.1f}M words, {n} shards, {stats['seconds']}s; LID reassigned {dict(reassigned.most_common(5))}")
    import shutil; shutil.rmtree(tmp, ignore_errors=True)

def write_data_card():
    files = list_repo_files(OUT, repo_type="dataset", token=TOKEN)
    rows = []
    for lang in LANGS:
        if f"{lang}/stats.json" in files:
            st = json.load(open(hf_hub_download(OUT, f"{lang}/stats.json", repo_type="dataset", token=TOKEN)))
            rows.append(f"| {lang} | {st.get('raw',0):,} | {st.get('kept',0):,} | {st.get('words',0)/1e6:.1f}M | {st.get('lid_other',0):,} | {st.get('exact_dup',0)+st.get('near_dup',0):,} | {st.get('contaminated',0):,} |")
    card = f"""---
license: other
license_name: mixed-open
license_link: https://github.com/balagh-lm/balagh
language: [ur, pa, ps, sd, skr, bal, brh, ks]
pretty_name: Balagh corpus v1 (languages of Pakistan)
---
# Balagh corpus v1

Cleaned, deduplicated, benchmark-decontaminated web and reference text for the languages of Pakistan, built by [Balagh](https://github.com/balagh-lm/balagh) (Ownova, Lahore) from openly packaged sources. Language labels come from PakLID, a Perso-Arabic language/dialect classifier trained for this project; the raw buckets of FineWeb-2, HPLT and GlotCC were re-sorted by it, which is why counts differ from the source corpora.

| language | raw docs | kept docs | words | dropped: other language | dropped: duplicates | dropped: benchmark overlap |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Sources and licences (packaging licences; underlying page copyrights are not cleared by them)
FineWeb-2 (ODC-By-1.0, Common Crawl ToU), HPLT 2.0 cleaned (CC0-1.0), GlotCC-V1 (CC0-1.0), Sangraha verified (CC-BY-4.0), Wikipedia 2023-11-01 (CC-BY-SA-3.0), Omnilingual-ASR transcripts (CC-BY-4.0). The `source` column records the origin of every document. Takedown: muneeb@ownova.org.

## Processing
Per-language normalisation without global NFKC (Nastaliq-specific code points preserved), quality filters (length, script share, repetition, boilerplate, digit ratio), PakLID check (confidence >= 0.6), exact dedup, MinHash near-dedup (128 permutations, 16 bands x 8 rows), 13-word-gram decontamination against UrduMMLU, Belebele, SIB-200, Global PIQA and FLORES+. Code: `phase0/colab/02_clean_corpus.ipynb` in the Balagh repo.
"""
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=OUT, repo_type="dataset", commit_message="data card")

if __name__ == "__main__":
    done_files = set(list_repo_files(OUT, repo_type="dataset", token=TOKEN))
    clf = train_paklid()
    grams = eval_ngrams()
    for lang in LANGS:
        print(f"\n== {lang}", flush=True)
        try: process_language(lang, clf, grams, done_files)
        except Exception as e: print("   FAILED", lang, str(e)[:300]); traceback.print_exc(limit=2)
        gc.collect()
    write_data_card()
    print("DONE")
