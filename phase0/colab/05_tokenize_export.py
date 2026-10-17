# Balagh data pipeline, step 5: tokenise balagh-lm/corpus-v1 plus English/code/math replay into packed uint32
# token shards for continued pretraining, and push them to the Hub (balagh-lm/tokens-qwen35-v1).
#
# Tokenizer: Qwen/Qwen3.5-2B-Base. Its vocab.json and merges.txt are byte-identical to Qwen3.6 and Qwen3.8, so these
# shards stay valid if a small Qwen3.8 base is released and swapped in later.
# Layout: {source}/shard-NNNNN.npy, each (rows, SEQ) uint32, documents packed back to back and separated by EOS;
# val/{source}.npy = held-out sequences made from ~1% of documents chosen by a hash of their text, which are never
# put in any training shard in any epoch; _state/{source}.json makes every source resumable at the exact
# file/row/carry where the last commit ended; manifest.json records tokens, words, epochs and tokens per word.
#
# Mix = shares of TOTAL_TOKENS (default 8e9), each capped by the data available x its max epochs. Input files are
# visited in a fixed hash order, so a budget cut takes a representative sample rather than the first sources.
# Runs on Kaggle CPU or Colab CPU. Parallel sessions: SOURCES=urd_Arab | SOURCES=en_edu,code,math | the rest; then
# MODE=manifest once. Needs HF_TOKEN (write). Smoke test: TOTAL_TOKENS=2e7 LOCAL_OUT=/kaggle/working/tok.

import os, io, json, time, math, glob, shutil, hashlib, collections, traceback
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

MODE = os.environ.get("MODE", "tokenize")              # tokenize | manifest
TOKENIZER = os.environ.get("TOKENIZER", "Qwen/Qwen3.5-2B-Base")
CORPUS = os.environ.get("CORPUS_REPO", "balagh-lm/corpus-v1")
OUT = os.environ.get("TOKENS_REPO", "balagh-lm/tokens-qwen35-v1")
TOTAL = float(os.environ.get("TOTAL_TOKENS", 8e9))
SEQ = int(os.environ.get("SEQ", 4096)); SHARD_SEQS = int(os.environ.get("SHARD_SEQS", 8192))   # 8192 x 4096 x 4 B = 128 MiB
VAL_SEQS = int(os.environ.get("VAL_SEQS", 64)); VAL_PERMILLE = int(os.environ.get("VAL_PERMILLE", 10))
SOURCES_SEL = [x.strip() for x in os.environ.get("SOURCES", "").split(",") if x.strip()]
LOCAL_OUT = os.environ.get("LOCAL_OUT")
CORPUS_DIR = os.environ.get("CORPUS_DIR")      # dry run: read corpus-v1 from a local folder (01_build_corpus LOCAL_OUT layout)
WALL_SECONDS = int(os.environ.get("WALL_SECONDS", 11 * 3600 + 15 * 60))
COMMIT_EVERY_S = int(os.environ.get("COMMIT_EVERY_S", 300))
SCRATCH = os.environ.get("SCRATCH") or ("/content/scratch" if os.path.isdir("/content") else "/tmp/balagh-tok" if os.name == "posix" else "./scratch-tok")
os.makedirs(SCRATCH, exist_ok=True); T0 = time.time()

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
TOKEN = secret("HF_TOKEN"); api = HfApi(token=TOKEN)
if not LOCAL_OUT:
    if not TOKEN: raise SystemExit("HF_TOKEN missing (or set LOCAL_OUT for a dry run)")
    if TOTAL < 1e9 and os.environ.get("ALLOW_SMALL") != "1":
        raise SystemExit("TOTAL_TOKENS < 1e9 would write a toy run into the real repo: set LOCAL_OUT=<dir> or a scratch TOKENS_REPO (ALLOW_SMALL=1)")
    api.create_repo(OUT, repo_type="dataset", exist_ok=True)

# share of TOTAL, max epochs over the available data, (repo, file glob, text column)
MIX = {
    "urd_Arab": (0.400, 1, (CORPUS, "urd_Arab/shard-*.parquet", "text")),
    "pbt_Arab": (0.100, 2, (CORPUS, "pbt_Arab/shard-*.parquet", "text")),
    "snd_Arab": (0.060, 3, (CORPUS, "snd_Arab/shard-*.parquet", "text")),
    "pnb_Arab": (0.025, 4, (CORPUS, "pnb_Arab/shard-*.parquet", "text")),
    "skr_Arab": (0.010, 4, (CORPUS, "skr_Arab/shard-*.parquet", "text")),
    "bal_Arab": (0.008, 4, (CORPUS, "bal_Arab/shard-*.parquet", "text")),
    "brh_Arab": (0.003, 4, (CORPUS, "brh_Arab/shard-*.parquet", "text")),
    "kas_Arab": (0.004, 4, (CORPUS, "kas_Arab/shard-*.parquet", "text")),
    "hno_Arab": (0.001, 4, (CORPUS, "hno_Arab/shard-*.parquet", "text")),
    "phr_Arab": (0.001, 4, (CORPUS, "phr_Arab/shard-*.parquet", "text")),
    "urd_Latn": (0.030, 2, (CORPUS, "urd_Latn/shard-*.parquet", "text")),
    # replay (keeps English, code and maths from eroding during adaptation)
    "en_edu":   (0.258, 1, ("HuggingFaceFW/fineweb-edu", "sample/10BT/*.parquet", "text")),
    "code":     (0.060, 1, ("OpenCoder-LLM/opc-fineweb-code-corpus", "data/*.parquet", "text")),   # MIT, code-heavy web pages
    "math":     (0.040, 1, ("HuggingFaceTB/finemath", "finemath-4plus/*.parquet", "text")),
}
if os.environ.get("MIX_JSON"): MIX.update({k: tuple(v) for k, v in json.loads(os.environ["MIX_JSON"]).items()})
unknown = [s for s in SOURCES_SEL if s not in MIX]
if unknown: raise SystemExit(f"unknown SOURCES {unknown}; known: {list(MIX)}")

def files_of(repo, pattern):
    """Matching files in a fixed pseudo-random order (md5 of the path): a budget cut-off then samples every
    source in proportion instead of taking the first ones in priority order."""
    import fnmatch
    if CORPUS_DIR and repo == CORPUS:
        fs = [os.path.relpath(os.path.join(d, f), CORPUS_DIR).replace("\\", "/") for d, _, ff in os.walk(CORPUS_DIR) for f in ff]
    else:
        for attempt in range(6):
            try: fs = api.list_repo_files(repo, repo_type="dataset"); break
            except Exception as e:
                if attempt == 5: raise
                print("   list retry", attempt, str(e)[:100], flush=True); time.sleep(min(300, 20 * 2 ** attempt))
    return sorted((f for f in fs if fnmatch.fnmatch(f, pattern) and f.count("/") == pattern.count("/")), key=lambda f: hashlib.md5(f.encode()).hexdigest())

def out_root(): return os.path.join(LOCAL_OUT, OUT.replace("/", "__"))

def fetch_out(path):
    """Local copy of a file in OUT, None if it does not exist; other errors are retried, then raised (never
    mistaken for 'no state', which would restart a source from zero)."""
    if LOCAL_OUT:
        p = os.path.join(out_root(), path); return p if os.path.exists(p) else None
    from huggingface_hub.errors import EntryNotFoundError
    for attempt in range(6):
        try: return hf_hub_download(OUT, path, repo_type="dataset", token=TOKEN, local_dir=os.path.join(SCRATCH, "state"), force_download=True)
        except EntryNotFoundError: return None
        except Exception as e:
            if attempt == 5: raise
            print("   state read retry", path, str(e)[:120], flush=True); time.sleep(min(300, 20 * 2 ** attempt))

def read_json(path):
    p = fetch_out(path)
    return json.load(open(p, encoding="utf-8")) if p else None

def commit(ops, message):
    """ops: list of (path_in_repo, local_file); atomic on the Hub."""
    if LOCAL_OUT:
        for p, l in ops:
            d = os.path.join(out_root(), p); os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copyfile(l, d)
    else:
        for attempt in range(10):
            try:
                api.create_commit(OUT, repo_type="dataset", operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=l) for p, l in ops], commit_message=message); break
            except Exception as e:
                print("   commit retry", attempt, str(e)[:160], flush=True); time.sleep(min(600, 30 * 2 ** attempt))
        else: raise RuntimeError("commit failed: " + message)
    for _, l in ops:
        try: os.remove(l)
        except OSError: pass

def save_npy(arr, name):
    p = os.path.join(SCRATCH, name.replace("/", "__")); np.save(p, arr); return p if p.endswith(".npy") else p + ".npy"

def is_val_doc(t):
    return int(hashlib.md5(t[:4000].encode("utf-8")).hexdigest()[:8], 16) % 1000 < VAL_PERMILLE

def tokenize_source(name, tok, eos):
    share, max_ep, (repo, pattern, field) = MIX[name]
    budget = int(TOTAL * share)
    files = files_of(repo, pattern)
    if not files: raise RuntimeError(f"{name}: no files for {repo}/{pattern}")
    fp = hashlib.sha1("\n".join(files).encode()).hexdigest()[:16]
    st = read_json(f"_state/{name}.json")
    fresh = {"epoch": 0, "file": 0, "row": 0, "carry": [], "vcarry": [], "val_count": 0, "shards": 0, "tokens": 0, "words": 0, "docs": 0,
             "val_docs": 0, "finished": False, "budget": budget, "seq": SEQ, "tokenizer": TOKENIZER, "files_fp": fp, "max_ep": max_ep}
    if st is None: st = fresh
    else:
        mism = {k: (st.get(k), fresh[k]) for k in ("seq", "tokenizer", "files_fp") if st.get(k) != fresh[k]}
        if mism: raise RuntimeError(f"{name}: saved state was made with different settings or input {mism}; delete _state/{name}.json, "
                                    f"val/{name}.npy and {name}/ in {OUT} to rebuild this source")
        if st.get("finished"):
            if st.get("budget", 0) >= budget or st["epoch"] >= max_ep:
                print(f"== {name}: finished earlier ({st['tokens']/1e9:.3f}B tokens)"); return st
            print(f"== {name}: extending budget {st.get('budget', 0)/1e9:.3f}B -> {budget/1e9:.3f}B"); st["finished"] = False
        st["budget"] = budget
    val = []
    if st["val_count"]:
        vp = fetch_out(f"val/{name}.npy")
        if vp is None: raise RuntimeError(f"{name}: state says {st['val_count']} val rows but val/{name}.npy is missing")
        val = [list(map(int, r)) for r in np.load(vp)]
    carry, vcarry = list(st["carry"]), list(st["vcarry"]); seqs = []
    pending = []; last_commit = time.time(); t0 = time.time()
    print(f"== {name}: budget {budget/1e9:.3f}B tokens, {len(files)} files, resume epoch {st['epoch']} file {st['file']} row {st['row']} shards {st['shards']}", flush=True)
    full = lambda: st["tokens"] + len(seqs) * SEQ >= budget

    def flush(final=False):
        nonlocal seqs, pending, last_commit
        if seqs:
            arr = np.asarray(seqs, dtype=np.uint32); seqs = []
            pending.append((f"{name}/shard-{st['shards']:05d}.npy", save_npy(arr, f"{name}-{st['shards']}.npy")))
            st["shards"] += 1; st["tokens"] += int(arr.size)
        if pending and (final or time.time() - last_commit >= COMMIT_EVERY_S):
            ops = list(pending)
            if val: ops.append((f"val/{name}.npy", save_npy(np.asarray(val, dtype=np.uint32), f"val-{name}.npy")))
            st["val_count"] = len(val)
            sp = os.path.join(SCRATCH, f"state-{name}.json"); json.dump({**st, "carry": carry, "vcarry": vcarry}, open(sp, "w"))
            commit(ops + [(f"_state/{name}.json", sp)], f"{name}: {st['shards']} shards, {st['tokens']/1e9:.3f}B tokens"); pending = []; last_commit = time.time()
            print(f"   {name}: {st['tokens']/1e9:.3f}B / {budget/1e9:.3f}B tokens, {st['shards']} shards, {st['docs']:,} docs, val {len(val)} rows, {(time.time()-t0)/60:.0f} min", flush=True)

    while st["epoch"] < max_ep and not full():
        while st["file"] < len(files) and not full():
            if CORPUS_DIR and repo == CORPUS:
                local = os.path.join(SCRATCH, "src-copy.parquet"); shutil.copyfile(os.path.join(CORPUS_DIR, files[st["file"]]), local)
            else:
                local = hf_hub_download(repo, files[st["file"]], repo_type="dataset", token=TOKEN, local_dir=os.path.join(SCRATCH, "src"))
            col = pq.read_table(local, columns=[field]).column(field).to_pylist()
            r = st["row"]
            while r < len(col) and not full():
                batch = [t for t in col[r:r + 256] if isinstance(t, str) and len(t) >= 50]
                r += 256
                train_b = []
                for t in batch:
                    if is_val_doc(t):
                        if st["epoch"] == 0 and len(val) < VAL_SEQS:   # val docs never enter training, in any epoch
                            vcarry.extend(tok(t, add_special_tokens=False)["input_ids"]); vcarry.append(eos); st["val_docs"] += 1
                            while len(vcarry) >= SEQ and len(val) < VAL_SEQS: val.append(vcarry[:SEQ]); vcarry = vcarry[SEQ:]
                    else: train_b.append(t)
                if train_b:
                    for t, ids in zip(train_b, tok(train_b, add_special_tokens=False)["input_ids"]):
                        carry.extend(ids); carry.append(eos); st["docs"] += 1; st["words"] += len(t.split())
                while len(carry) >= SEQ and not full():
                    seqs.append(carry[:SEQ]); carry = carry[SEQ:]
                    if len(seqs) >= SHARD_SEQS: st["row"] = r; flush()
                if time.time() - T0 > WALL_SECONDS:
                    st["row"] = r; flush(final=True); print(f"   wall-clock guard: {name} paused"); return st
            if r >= len(col): st["file"] += 1; st["row"] = 0
            else: st["row"] = r
            try: os.remove(local)
            except OSError: pass
        if full(): break
        st["epoch"] += 1; st["file"] = 0; st["row"] = 0; carry = []   # no document spans two epochs
        print(f"   {name}: data exhausted after {st['epoch']} of max {max_ep} epochs", flush=True)
    if not val and vcarry:   # tiny source: one padded held-out sequence is better than none
        val.append((vcarry + [eos] * SEQ)[:SEQ]); vcarry = []
    st["epochs_used"] = min(max_ep, st["epoch"] + 1); st["finished"] = True; flush(final=True)
    if not val: print(f"   WARNING {name}: no held-out sequence (too few documents)")
    print(f"   {name} DONE: {st['tokens']/1e9:.3f}B tokens ({st['tokens']/max(1, st['words']):.2f} tokens/word incl. EOS), {st['shards']} shards, "
          f"{st['epochs_used']} epoch(s), val {len(val)} rows from {st['val_docs']} documents", flush=True)
    return st

def write_manifest():
    rows = {}
    for name in MIX:
        st = read_json(f"_state/{name}.json")
        if st: rows[name] = {k: st.get(k) for k in ("tokens", "words", "docs", "shards", "epochs_used", "finished", "val_count", "val_docs")} | {"tokens_per_word": round(st["tokens"] / max(1, st["words"]), 3), "share_target": MIX[name][0]}
    total = sum(r["tokens"] for r in rows.values())
    unfinished = [k for k, v in rows.items() if not v["finished"]] + [k for k in MIX if k not in rows]
    man = {"tokenizer": TOKENIZER, "seq_len": SEQ, "dtype": "uint32", "shard_seqs": SHARD_SEQS, "total_tokens": total, "sources": rows, "unfinished": unfinished,
           "note": "documents packed with EOS; val/{source}.npy holds ~1% of documents (hash-selected) that never appear in training shards; shares are of TOTAL_TOKENS, capped by data x max epochs"}
    mp = os.path.join(SCRATCH, "manifest.json"); json.dump(man, open(mp, "w"), indent=1)
    lines = ["---", "license: other", "license_name: mixed-open", "pretty_name: Balagh token shards (Qwen3.5 tokenizer)", "---", "# Balagh token shards v1",
             f"Packed uint32 token sequences ({SEQ} tokens) of [balagh-lm/corpus-v1](https://huggingface.co/datasets/balagh-lm/corpus-v1) plus English/code/maths replay, tokenised with `{TOKENIZER}`. Total **{total/1e9:.2f}B tokens**.", "",
             "| source | tokens | share | tokens/word | epochs | held-out rows |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['tokens']/1e9:.3f}B | {v['tokens']/max(1,total):.1%} | {v['tokens_per_word']} | {v.get('epochs_used') or '-'} | {v.get('val_count') or 0} |" for k, v in rows.items()]
    if unfinished: lines += ["", f"Unfinished sources: {', '.join(unfinished)}"]
    cp = os.path.join(SCRATCH, "README.md"); open(cp, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    commit([("manifest.json", mp), ("README.md", cp)], f"manifest: {total/1e9:.2f}B tokens")
    print(json.dumps(man, indent=1)[:3000])
    if unfinished: print("WARNING: unfinished sources:", unfinished)

if __name__ == "__main__":
    if MODE == "manifest": write_manifest()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER, token=TOKEN)
        eos = tok.eos_token_id if tok.eos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
        assert eos is not None and len(tok) < 2**32
        print(f"tokenizer {TOKENIZER}: vocab {len(tok):,}, eos {eos} ({tok.convert_ids_to_tokens(eos)}), seq {SEQ}, total budget {TOTAL/1e9:.2f}B", flush=True)
        failed = []
        for name in (SOURCES_SEL or list(MIX)):
            ok = False
            for attempt in range(6):
                try: st = tokenize_source(name, tok, eos); ok = bool(st.get("finished")); break
                except Exception as e:
                    print(f"   FAILED {name} (attempt {attempt + 1}/6): {str(e)[:300]}"); traceback.print_exc(limit=3)
                    if "saved state was made with different settings" in str(e) or time.time() - T0 > WALL_SECONDS: break
                    time.sleep(min(600, 30 * 2 ** attempt))
            if not ok: failed.append(name)
            if time.time() - T0 > WALL_SECONDS: break
        if not SOURCES_SEL and not failed: write_manifest()
        if failed: raise SystemExit(f"INCOMPLETE sources (rerun to resume): {failed}")
    print(f"DONE in {(time.time()-T0)/60:.1f} min")
