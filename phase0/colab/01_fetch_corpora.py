# Balagh data pipeline, step 1: fetch every openly licensed corpus for Pakistan's languages and
# mirror it, shard by shard, into the Hugging Face dataset repo balagh-lm/raw-corpora-v0.
#
# Runs on a Colab CPU or any GPU runtime (no GPU needed). Resumable: shards already in the Hub repo
# are skipped, so a session timeout costs nothing. Needs HF_TOKEN in Colab Secrets (write access to
# the balagh-lm org). Nothing is stored on Colab beyond one shard at a time.
#
# Licences mirrored: FineWeb-2 (ODC-By), HPLT 2.0 cleaned (CC0), GlotCC (CC0), Sangraha verified
# (CC-BY-4.0), Wikipedia (CC-BY-SA-3.0), Omnilingual-ASR transcripts (CC-BY-4.0), SMOL (CC-BY-4.0).
# Each shard carries its source, config and licence in the parquet metadata and in the manifest.

import os, io, json, time, hashlib, itertools, traceback
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
import pyarrow as pa, pyarrow.parquet as pq

REPO = os.environ.get("BALAGH_RAW_REPO", "balagh-lm/raw-corpora-v0")
SHARD_DOCS = int(os.environ.get("SHARD_DOCS", 50000))
MAX_DOCS = int(os.environ.get("MAX_DOCS", 0)) or None          # cap per config for test runs
ONLY = set(filter(None, os.environ.get("ONLY", "").split(",")))  # e.g. "fineweb2:urd_Arab,wiki:ur"

def hf_token():
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")

TOKEN = hf_token()
api = HfApi(token=TOKEN)
api.create_repo(REPO, repo_type="dataset", exist_ok=True)

# key -> (hf dataset, config, split, text field, extra fields to keep, licence, language tag)
SOURCES = {}
for lang in ["urd_Arab", "urd_Latn", "pbt_Arab", "snd_Arab", "pnb_Arab", "skr_Arab", "brh_Arab", "kas_Arab"]:
    SOURCES[f"fineweb2:{lang}"] = ("HuggingFaceFW/fineweb-2", lang, "train", "text", ["id", "url", "date", "language_score"], "ODC-By-1.0", lang)
for lang in ["pnb_Arab", "skr_Arab", "brh_Arab", "kas_Arab"]:   # rejected-by-LID buckets: PakLID will re-sort them
    SOURCES[f"fineweb2removed:{lang}"] = ("HuggingFaceFW/fineweb-2", f"{lang}_removed", "train", "text", ["id", "url", "date", "language_score"], "ODC-By-1.0", lang)
for lang in ["urd_Arab", "pbt_Arab", "snd_Arab", "kas_Arab"]:
    SOURCES[f"hplt2:{lang}"] = ("HPLT/HPLT2.0_cleaned", lang, "train", "text", ["id", "u", "lang", "seg_langs"], "CC0-1.0", lang)
for lang in ["bal-Arab", "skr-Arab", "brh-Arab", "pnb-Arab", "urd-Latn"]:
    SOURCES[f"glotcc:{lang}"] = ("cis-lmu/GlotCC-V1", lang, "train", "content", ["url", "warc-date", "lid-script"], "CC0-1.0", lang.replace("-", "_"))
for lang in ["urd", "snd", "kas"]:
    SOURCES[f"sangraha:{lang}"] = ("ai4bharat/sangraha", "verified", lang, "text", ["doc_id", "type"], "CC-BY-4.0", lang + "_Arab")
for wl, lang in [("ur", "urd_Arab"), ("pnb", "pnb_Arab"), ("skr", "skr_Arab"), ("sd", "snd_Arab"), ("ps", "pbt_Arab"), ("ks", "kas_Arab")]:
    SOURCES[f"wiki:{wl}"] = ("wikimedia/wikipedia", f"20231101.{wl}", "train", "text", ["id", "url", "title"], "CC-BY-SA-3.0", lang)
for lang in ["hno_Arab", "phr_Arab", "pnb_Arab", "brh_Arab", "kas_Arab", "pbt_Arab", "urd_Arab", "snd_Arab", "bal_Arab", "skr_Arab"]:
    SOURCES[f"omniasr:{lang}"] = ("facebook/omnilingual-asr-corpus", lang, None, "raw_text", ["id"], "CC-BY-4.0", lang)   # split resolved at run time; audio column dropped
for cfg in ["smoldoc__en_brh", "smoldoc__en_ps", "smoldoc__en_ur", "smolsent__en_brh", "smolsent__en_ks", "smolsent__en_pa-Arab", "smolsent__en_ur", "gatitos__en_bal", "gatitos__en_ks", "gatitos__en_pa-Arab", "gatitos__en_skr"]:
    SOURCES[f"smol:{cfg}"] = ("google/smol", cfg, "train", None, None, "CC-BY-4.0", cfg.split("__en_")[-1])

if ONLY:
    SOURCES = {k: v for k, v in SOURCES.items() if k in ONLY}

def existing_shards():
    files = api.list_repo_files(REPO, repo_type="dataset")
    return set(files)

def shard_path(key, n):
    src, cfg = key.split(":", 1)
    return f"{src}/{cfg}/shard-{n:05d}.parquet"

def resolve_split(ds_name, cfg, split):
    if split: return split
    from datasets import get_dataset_split_names
    names = get_dataset_split_names(ds_name, cfg)
    return "train" if "train" in names else names[0]

def row_to_record(row, text_field, keep):
    if text_field is None:                       # SMOL: keep the whole row as JSON
        return {"text": json.dumps(row, ensure_ascii=False), "meta": "{}"}
    txt = row.get(text_field)
    if not isinstance(txt, str) or not txt.strip(): return None
    meta = {k: (row[k] if not isinstance(row[k], (bytes, bytearray)) else None) for k in (keep or []) if k in row}
    for k, v in list(meta.items()):
        if not isinstance(v, (str, int, float, bool, type(None))): meta[k] = str(v)[:200]
    return {"text": txt, "meta": json.dumps(meta, ensure_ascii=False)}

def write_shard(records, key, n, src_info):
    ds_name, cfg, split, _, _, lic, lang = src_info
    table = pa.table({"text": [r["text"] for r in records], "meta": [r["meta"] for r in records],
                      "sha1": [hashlib.sha1(r["text"].encode("utf-8")).hexdigest() for r in records]})
    table = table.replace_schema_metadata({"source": ds_name, "config": cfg, "split": split or "", "licence": lic, "language": lang, "shard": str(n), "fetched": time.strftime("%Y-%m-%d")})
    buf = io.BytesIO(); pq.write_table(table, buf, compression="zstd"); buf.seek(0)
    path = shard_path(key, n)
    for attempt in range(5):
        try:
            api.upload_file(path_or_fileobj=buf, path_in_repo=path, repo_id=REPO, repo_type="dataset", commit_message=f"{key} shard {n} ({len(records)} docs)")
            return path
        except Exception as e:
            print("   upload retry", attempt, str(e)[:120]); time.sleep(10 * (attempt + 1)); buf.seek(0)
    raise RuntimeError("upload failed: " + path)

def fetch(key, src_info, done):
    ds_name, cfg, split, text_field, keep, lic, lang = src_info
    split = resolve_split(ds_name, cfg, split)
    kw = {"streaming": True, "token": TOKEN}
    ds = load_dataset(ds_name, cfg, split=split, **kw)
    if "omnilingual" in ds_name:
        ds = ds.remove_columns([c for c in ds.column_names if c in ("audio",)]) if ds.column_names else ds
    n_shard, buf, n_docs, n_bytes = 0, [], 0, 0
    # skip shards already uploaded by consuming the stream only when needed
    while shard_path(key, n_shard) in done: n_shard += 1
    skip_docs = n_shard * SHARD_DOCS
    it = iter(ds)
    if skip_docs:
        print(f"   resuming after {skip_docs} docs ({n_shard} shards present)")
        it = itertools.islice(it, skip_docs, None)
    t0 = time.time()
    for row in it:
        rec = row_to_record(row, text_field, keep)
        if rec is None: continue
        buf.append(rec); n_docs += 1; n_bytes += len(rec["text"])
        if len(buf) >= SHARD_DOCS:
            write_shard(buf, key, n_shard, src_info); n_shard += 1; buf = []
            print(f"   {key}: {n_docs + skip_docs:,} docs, {n_bytes/1e9:.2f} GB text, {time.time()-t0:.0f}s", flush=True)
        if MAX_DOCS and n_docs >= MAX_DOCS: break
    if buf:
        write_shard(buf, key, n_shard, src_info); n_shard += 1
    return {"shards": n_shard, "docs_this_run": n_docs, "text_bytes_this_run": n_bytes, "licence": lic, "language": lang, "source": ds_name, "config": cfg, "split": split}

if __name__ == "__main__":
    done = existing_shards()
    manifest_path = "manifest.json"
    try:
        manifest = json.load(open(hf_hub_download(REPO, manifest_path, repo_type="dataset", token=TOKEN)))
    except Exception:
        manifest = {}
    print(f"{len(SOURCES)} sources; {len(done)} shards already in {REPO}")
    for key, src_info in SOURCES.items():
        print(f"\n== {key}  ({src_info[0]} / {src_info[1]} / {src_info[2] or 'auto'})", flush=True)
        try:
            manifest[key] = fetch(key, src_info, done)
            manifest[key]["status"] = "ok"
        except Exception as e:
            manifest[key] = {"status": "error", "error": str(e)[:300]}
            print("   FAILED:", str(e)[:300]); traceback.print_exc(limit=1)
        api.upload_file(path_or_fileobj=json.dumps(manifest, indent=1, ensure_ascii=False).encode(), path_in_repo=manifest_path, repo_id=REPO, repo_type="dataset", commit_message=f"manifest after {key}")
    print("\nDONE. Manifest:")
    for k, v in manifest.items(): print(f"  {k:32s} {v.get('status'):6s} shards={v.get('shards','-')} err={v.get('error','')[:60]}")
