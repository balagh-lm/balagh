# Paper 1: how much of what public web corpora label as each Pakistani language really is that language?
#
# For every (corpus, claimed language) pair: N_SAMPLE documents spread across the upstream files (row groups at an
# even stride for parquet; the first N_SCAN lines for MADLAD's jsonl.gz), then automatic signals per document:
#   PakLID v1 label + confidence, Urdu function-word rate (01_build_corpus.urdu_rate), Arabic/Latin script share, length.
# Writes audit_auto.json (per-pair distributions) and annotation/{lang}.csv: a stratified sample per language across
# corpora for native speakers (blind to corpus and to the automatic labels), whose labels calibrate the signals.
# CPU only; reads 01_build_corpus.py for the shared normalisation, quality and LID code.

import os, io, re, csv, json, gzip, random, collections, hashlib
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", "audit_out")
N_SAMPLE = int(os.environ.get("N_SAMPLE", 500)); N_ANNOT = int(os.environ.get("N_ANNOT", 150)); N_SCAN = int(os.environ.get("N_SCAN", 20000))
SEED = 20261107

ns = {"__name__": "corpus_audit", "__file__": os.path.join(HERE, "..", "colab", "01_build_corpus.py")}
os.environ.setdefault("SCRATCH", os.path.join(OUT, "scratch"))
exec(compile(open(ns["__file__"], encoding="utf-8").read(), "01_build_corpus.py", "exec"), ns)
base_norm, quality, lid_predict, urdu_rate, script_of = ns["base_norm"], ns["quality"], ns["lid_predict"], ns["urdu_rate"], ns["script_of"]
ns["CLF"] = ns["load_paklid"]()

# (corpus, claimed language as our code, repo, kind, path glob or file list, text field)
PAIRS = []
def add(corpus, lang, repo, kind, pattern, field): PAIRS.append(dict(corpus=corpus, lang=lang, repo=repo, kind=kind, pattern=pattern, field=field))
for l in ["urd_Arab", "urd_Latn", "pbt_Arab", "snd_Arab", "pnb_Arab", "skr_Arab", "brh_Arab", "kas_Arab"]:
    add("FineWeb-2", l, "HuggingFaceFW/fineweb-2", "parquet", f"data/{l}/train/*.parquet", "text")
for l in ["pnb_Arab", "skr_Arab", "brh_Arab", "kas_Arab"]:
    add("FineWeb-2 removed", l, "HuggingFaceFW/fineweb-2", "parquet", f"data/{l}_removed/train/*.parquet", "text")
for l in ["urd_Arab", "pbt_Arab", "snd_Arab", "kas_Arab"]:
    add("HPLT 2.0", l, "HPLT/HPLT2.0_cleaned", "parquet", f"{l}/*.parquet", "text")
for g, l in [("urd-Arab", "urd_Arab"), ("urd-Latn", "urd_Latn"), ("pbt-Arab", "pbt_Arab"), ("snd-Arab", "snd_Arab"), ("pnb-Arab", "pnb_Arab"),
             ("skr-Arab", "skr_Arab"), ("bal-Arab", "bal_Arab"), ("brh-Arab", "brh_Arab"), ("kas-Arab", "kas_Arab")]:
    add("GlotCC-V1", l, "cis-lmu/GlotCC-V1", "parquet", f"v1.0/{g}/*.parquet", "content")
for m, l in [("ur", "urd_Arab"), ("ps", "pbt_Arab"), ("sd", "snd_Arab"), ("skr", "skr_Arab"), ("bgp", "bal_Arab")]:
    add("MADLAD-400 clean", l, "allenai/MADLAD-400", "jsonl.gz", f"data/{m}/{m}_clean_*.jsonl.gz", "text")
for c, l in [("ur", "urd_Arab"), ("ps", "pbt_Arab"), ("sd", "snd_Arab"), ("pnb", "pnb_Arab")]:
    add("CulturaX", l, "uonlp/CulturaX", "parquet", f"{c}/*.parquet", "text")

api = HfApi(); fs = HfFileSystem()
def files(repo, pattern):
    import fnmatch
    return sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, pattern) and f.count("/") == pattern.count("/"))

def sample_parquet(p):
    fl = files(p["repo"], p["pattern"]); plan = []
    for f in fl:
        md = pq.ParquetFile(fs.open(f"datasets/{p['repo']}/{f}")).metadata
        plan += [(f, rg) for rg in range(md.num_row_groups)]
    rng = random.Random(SEED); rng.shuffle(plan); out = []
    for f, rg in plan:
        t = pq.ParquetFile(fs.open(f"datasets/{p['repo']}/{f}")).read_row_group(rg, columns=[p["field"]]).column(p["field"]).to_pylist()
        rng.shuffle(t); out += [x for x in t[:max(1, N_SAMPLE // 20)] if isinstance(x, str) and x.strip()]
        if len(out) >= N_SAMPLE: break
    return out[:N_SAMPLE], len(fl)

def sample_jsonl_gz(p):
    fl = files(p["repo"], p["pattern"]); rows = []
    for f in fl:
        with fs.open(f"datasets/{p['repo']}/{f}") as fh, gzip.open(fh, "rt", encoding="utf-8") as g:
            for i, line in enumerate(g):
                if i >= N_SCAN: break
                try: t = json.loads(line).get(p["field"])
                except Exception: continue
                if isinstance(t, str) and t.strip(): rows.append(t)
        if len(rows) >= N_SCAN: break
    random.Random(SEED).shuffle(rows); return rows[:N_SAMPLE], len(fl)

def analyse(lang, texts):
    sc = script_of(lang); docs = []
    norm = [base_norm(t) for t in texts]
    q = [quality(b, sc) for b in norm]
    labs, confs = lid_predict(norm) if norm else ([], [])
    for t, b, qq, lab, conf in zip(texts, norm, q, labs, confs):
        letters = [c for c in b if c.isalpha()]; arab = sum(1 for c in letters if ns["ARABIC_RE"].match(c))
        docs.append({"sha": hashlib.sha1(t.encode("utf-8")).hexdigest()[:12], "paklid": lab, "conf": round(float(conf), 3), "quality": qq or "ok",
                     "urdu_rate": round(urdu_rate(b), 2) if sc == "Arab" else None, "arabic_share": round(arab / max(1, len(letters)), 3), "chars": len(b), "text": b})
    return docs

def summary(lang, docs):
    n = len(docs); c = collections.Counter(d["paklid"] for d in docs)
    return {"n": n, "paklid_claimed_share": round(c[lang] / max(1, n), 3), "paklid_top": dict(c.most_common(5)),
            "paklid_claimed_conf>=0.7": round(sum(d["paklid"] == lang and d["conf"] >= 0.7 for d in docs) / max(1, n), 3),
            "urdu_like_share(>=8)": round(sum((d["urdu_rate"] or 0) >= 8 for d in docs) / max(1, n), 3) if script_of(lang) == "Arab" else None,
            "quality_fail_share": round(sum(d["quality"] != "ok" for d in docs) / max(1, n), 3)}

if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "annotation"), exist_ok=True)
    res_path = os.path.join(OUT, "audit_auto.json")
    res = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else {}
    pool = collections.defaultdict(list)
    for p in PAIRS:
        key = f"{p['corpus']} | {p['lang']}"
        cache = os.path.join(OUT, "docs", re.sub(r"[^A-Za-z0-9_.-]+", "_", key) + ".jsonl")
        try:
            if os.path.exists(cache): docs = [json.loads(l) for l in open(cache, encoding="utf-8")]
            else:
                texts, nfiles = (sample_parquet if p["kind"] == "parquet" else sample_jsonl_gz)(p)
                docs = analyse(p["lang"], texts); os.makedirs(os.path.dirname(cache), exist_ok=True)
                with open(cache, "w", encoding="utf-8") as f:
                    for d in docs: f.write(json.dumps(d, ensure_ascii=False) + "\n")
            res[key] = {**p, **summary(p["lang"], docs)}; print(key, {k: res[key][k] for k in ("n", "paklid_claimed_share", "urdu_like_share(>=8)", "quality_fail_share")}, flush=True)
            for d in docs: pool[p["lang"]].append((p["corpus"], d))
        except Exception as e:
            res[key] = {**p, "error": str(e)[:300]}; print(key, "ERROR", str(e)[:200], flush=True)
        json.dump(res, open(res_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    # annotation sheets: per language, N_ANNOT docs stratified over corpora, shuffled, no corpus or LID shown
    key_rows = []
    for lang, items in pool.items():
        by = collections.defaultdict(list)
        for corpus, d in items: by[corpus].append(d)
        rng = random.Random(SEED); pick = []; k = max(1, N_ANNOT // len(by))
        for corpus, ds in by.items(): rng.shuffle(ds); pick += [(corpus, d) for d in ds[:k]]
        rng.shuffle(pick)
        with open(os.path.join(OUT, "annotation", f"{lang}.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(["item", "text (first 700 characters)", f"Is this text written in {lang}? (yes / mostly / no)", "If not, which language? (Urdu / English / Persian / Punjabi / other)", "Is it real running text? (yes / no: list, menu, spam, machine-translated)", "notes"])
            for i, (corpus, d) in enumerate(pick):
                item = f"{lang}-{i:04d}"; w.writerow([item, d["text"][:700].replace("\n", " / "), "", "", "", ""])
                key_rows.append({"item": item, "lang": lang, "corpus": corpus, "sha": d["sha"], "paklid": d["paklid"], "conf": d["conf"], "urdu_rate": d["urdu_rate"]})
    with open(os.path.join(OUT, "annotation_key.json"), "w", encoding="utf-8") as f: json.dump(key_rows, f, indent=1, ensure_ascii=False)
    print("annotation sheets:", {l: min(N_ANNOT, len(v)) for l, v in pool.items()})
