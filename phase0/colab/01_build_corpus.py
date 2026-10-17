# Balagh corpus builder: upstream open corpora -> balagh-lm/corpus-v1 (cleaned, language-checked,
# deduplicated, benchmark-decontaminated), in four resumable modes. Runs on Kaggle CPU (preferred: free,
# server-side, 4 vCPU / 30 GB RAM, 12 h per session, several sessions in parallel), Colab CPU, or any
# Linux box. No GPU. Every output goes to the Hugging Face Hub, so a dead session only loses its unit.
#
#   MODE=paklid  train PakLID v1 (Perso-Arabic language/dialect ID) on sources that were NOT labelled by
#                GlotLID wherever such sources exist; upload to balagh-lm/paklid with a held-out report.
#   MODE=route   stream upstream parquet files unit by unit (a unit = a block of row groups, ~100k docs),
#                normalise, quality-filter, run PakLID, and write each document to the language PakLID
#                predicts -> balagh-lm/corpus-v1-routed/{lang}/{source}/u*.parquet, plus per-unit stats.
#                Re-sorting matters: FineWeb-2's Shahmukhi bucket holds Urdu, its Urdu bucket holds
#                Punjabi and Saraiki, GlotCC's Balochi bucket holds Roman Urdu.
#   MODE=dedup   per language: exact dedup (sha1), 13-gram decontamination against every eval set we report
#                (normalised with the same per-language rules), MinHash near-dedup (5-word shingles, 128
#                permutations, 16 bands x 8 rows ~ Jaccard 0.75; higher-priority sources win), final
#                shards -> balagh-lm/corpus-v1/{lang}/shard-*.parquet + stats.json.
#   MODE=card    data card for corpus-v1 with the routing matrix (bucket -> predicted language).
#
# Order: paklid (one session, ~30-60 min) -> route (parallel sessions: SOURCES=group:urdu-fw2, group:urdu-hplt,
# group:urdu-sangraha, group:pak-mid, group:pak-small) -> dedup (parallel: LANGS=urd_Arab; LANGS=pbt_Arab,snd_Arab;
# LANGS=<the rest>) -> card. Smoke test: MODE=paklid PAKLID_PER_SOURCE=300, then MODE=route SOURCES=wiki:skr
# MAX_UNITS=1, then MODE=dedup LANGS=skr_Arab. LOCAL_OUT=/some/dir writes to disk instead of the Hub (dry run).
#
# Secrets: HF_TOKEN with write access to the balagh-lm org (Kaggle: Add-ons > Secrets; Colab: key icon).

import os, io, re, sys, json, time, math, glob, gc, random, shutil, hashlib, collections, traceback, unicodedata
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

MODE = os.environ.get("MODE", "route")
ROUTED_REPO = os.environ.get("ROUTED_REPO", "balagh-lm/corpus-v1-routed")
OUT_REPO = os.environ.get("OUT_REPO", "balagh-lm/corpus-v1")
PAKLID_REPO = os.environ.get("PAKLID_REPO", "balagh-lm/paklid")
LOCAL_OUT = os.environ.get("LOCAL_OUT")                       # dry run: mirror the Hub layout under this folder
SOURCES_SEL = os.environ.get("SOURCES", "all")
LANGS_SEL = os.environ.get("LANGS", "")
MAX_UNITS = int(os.environ.get("MAX_UNITS", 0)) or None       # per source, for smoke tests
UNIT_ROWS = int(os.environ.get("UNIT_ROWS", 100_000))
WORKERS = int(os.environ.get("WORKERS", 0)) or (os.cpu_count() or 2)
PAKLID_PER_SOURCE = int(os.environ.get("PAKLID_PER_SOURCE", 0)) or None   # cap seeds per source (smoke)
MAX_CHUNKS_PER_DOC = int(os.environ.get("MAX_CHUNKS_PER_DOC", 4))
WALL_SECONDS = int(os.environ.get("WALL_SECONDS", 11 * 3600 + 15 * 60))   # stop starting new units after this
SCRATCH = os.environ.get("SCRATCH") or ("/content/scratch" if os.path.isdir("/content") else "/tmp/balagh" if os.name == "posix" else "./scratch")
COMMIT_EVERY_S = int(os.environ.get("COMMIT_EVERY_S", 480))   # at most one commit per 8 min per session (Hub commit rate limits)
os.makedirs(SCRATCH, exist_ok=True)
T_START = time.time()

def secret(k):
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

TOKEN = secret("HF_TOKEN")
api = HfApi(token=TOKEN)
if not LOCAL_OUT and not TOKEN:
    raise SystemExit("HF_TOKEN missing: add it to Kaggle/Colab secrets, or set LOCAL_OUT for a dry run")

# ------------------------------------------------------------------------------------------ Hub store
class Store:
    """A Hub dataset repo (or a local folder in dry-run mode) with batched commits."""
    def __init__(self, repo, repo_type="dataset"):
        self.repo, self.rt, self.pending = repo, repo_type, []
        if LOCAL_OUT:
            self.root = os.path.join(LOCAL_OUT, repo.replace("/", "__")); os.makedirs(self.root, exist_ok=True)
        else:
            api.create_repo(repo, repo_type=repo_type, exist_ok=True)
    def files(self, prefix=""):
        if LOCAL_OUT:
            out = set()
            for d, _, fs in os.walk(self.root):
                for f in fs: out.add(os.path.relpath(os.path.join(d, f), self.root).replace("\\", "/"))
        else:
            for attempt in range(5):
                try: out = set(api.list_repo_files(self.repo, repo_type=self.rt)); break
                except Exception as e: print("   list retry", str(e)[:100]); time.sleep(15 * (attempt + 1))
            else: raise RuntimeError("cannot list " + self.repo)
        return {f for f in out if f.startswith(prefix)}
    def add_bytes(self, path, data):
        tmp = os.path.join(SCRATCH, "pending-" + hashlib.md5(path.encode()).hexdigest())
        with open(tmp, "wb") as f: f.write(data)
        self.pending.append((path, tmp))
    def add_file(self, path, local): self.pending.append((path, local))
    def add_json(self, path, obj): self.add_bytes(path, json.dumps(obj, indent=1, ensure_ascii=False, default=str).encode("utf-8"))
    def maybe_flush(self, message):
        """Commit only if the last commit was long enough ago (keeps parallel sessions under Hub rate limits)."""
        if time.time() - getattr(self, "last_commit", 0) >= COMMIT_EVERY_S: self.flush(message)
    def flush(self, message):
        if not self.pending: return
        self.last_commit = time.time()
        if LOCAL_OUT:
            for path, local in self.pending:
                dst = os.path.join(self.root, path); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(local, dst)
        else:
            ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=l) for p, l in self.pending]
            for attempt in range(10):
                try: api.create_commit(self.repo, repo_type=self.rt, operations=ops, commit_message=message); break
                except Exception as e:
                    print("   commit retry", attempt, str(e)[:160], flush=True); time.sleep(min(600, 30 * 2 ** attempt))
            else: raise RuntimeError("commit failed: " + message)
        for _, local in self.pending:
            if os.path.basename(local).startswith("pending-") or local.startswith(SCRATCH):
                try: os.remove(local)
                except OSError: pass
        self.pending = []
    def download(self, path):
        if LOCAL_OUT: return os.path.join(self.root, path)
        for attempt in range(5):
            try: return hf_hub_download(self.repo, path, repo_type=self.rt, token=TOKEN, local_dir=os.path.join(SCRATCH, "store", self.repo.replace("/", "__")))
            except Exception as e: print("   download retry", path, str(e)[:100]); time.sleep(15 * (attempt + 1))
        raise RuntimeError("download failed: " + path)
    def read_json(self, path, default=None):
        try:
            with open(self.download(path), encoding="utf-8") as f: return json.load(f)
        except Exception: return default

# ------------------------------------------------------------------------------------------ sources
# key -> upstream repo, path glob inside the repo, text column, metadata columns kept, licence, bucket
# language (what upstream claims), dedup priority (lower wins when two documents are near-duplicates).
S = {}
def src(key, repo, glob_, field, keep, lic, bucket, prio): S[key] = dict(repo=repo, glob=glob_, field=field, keep=keep, licence=lic, bucket=bucket, prio=prio)
for wl, lang in [("ur", "urd_Arab"), ("pnb", "pnb_Arab"), ("skr", "skr_Arab"), ("sd", "snd_Arab"), ("ps", "pbt_Arab"), ("ks", "kas_Arab")]:
    src(f"wiki:{wl}", "wikimedia/wikipedia", f"20231101.{wl}/*.parquet", "text", ["id", "url", "title"], "CC-BY-SA-3.0", lang, 0)
for lang in ["pbt_Arab", "pnb_Arab", "brh_Arab", "kas_Arab", "hno_Arab", "phr_Arab"]:
    src(f"omniasr:{lang}", "facebook/omnilingual-asr-corpus", f"data/{lang}/train-*.parquet", "raw_text", ["segment_id", "speaker_id"], "CC-BY-4.0", lang, 0)
for code, lang in [("urd", "urd_Arab"), ("snd", "snd_Arab"), ("kas", "kas_Arab")]:
    src(f"sangraha:{code}", "ai4bharat/sangraha", f"verified/{code}/*.parquet", "text", ["doc_id", "type"], "CC-BY-4.0", lang, 1)
for lang in ["urd_Arab", "pbt_Arab", "snd_Arab", "kas_Arab"]:
    src(f"hplt2:{lang}", "HPLT/HPLT2.0_cleaned", f"{lang}/*.parquet", "text", ["id", "u", "collection"], "CC0-1.0", lang, 2)
for lang in ["urd_Arab", "urd_Latn", "pbt_Arab", "snd_Arab", "pnb_Arab", "skr_Arab", "brh_Arab", "kas_Arab"]:
    src(f"fw2:{lang}", "HuggingFaceFW/fineweb-2", f"data/{lang}/train/*.parquet", "text", ["id", "url", "date", "language_score"], "ODC-By-1.0", lang, 3)
for g, lang in [("urd-Arab", "urd_Arab"), ("urd-Latn", "urd_Latn"), ("pnb-Arab", "pnb_Arab"), ("skr-Arab", "skr_Arab"), ("bal-Arab", "bal_Arab"), ("brh-Arab", "brh_Arab"), ("kas-Arab", "kas_Arab")]:
    src(f"glotcc:{g}", "cis-lmu/GlotCC-V1", f"v1.0/{g}/*.parquet", "content", ["warc-target-uri", "warc-date"], "CC0-1.0", lang, 4)
for lang in ["pnb_Arab", "skr_Arab", "brh_Arab", "kas_Arab", "snd_Arab", "pbt_Arab"]:
    src(f"fw2rm:{lang}", "HuggingFaceFW/fineweb-2", f"data/{lang}_removed/train/*.parquet", "text", ["id", "url", "date", "language_score"], "ODC-By-1.0", lang, 5)

GROUPS = {   # balanced for parallel Kaggle sessions (upstream compressed sizes, Oct 2026)
    "urdu-fw2": ["fw2:urd_Arab"],                                                      # 6.9 GB, 4.8M docs
    "urdu-hplt": ["hplt2:urd_Arab", "glotcc:urd-Arab"],                                # 8.8 + 1.0 GB
    "urdu-sangraha": ["sangraha:urd"],                                                 # 12.8 GB
    "pak-mid": ["fw2:pbt_Arab", "hplt2:pbt_Arab", "fw2rm:pbt_Arab", "fw2:snd_Arab", "hplt2:snd_Arab", "sangraha:snd", "fw2rm:snd_Arab"],
    "pak-small": [k for k in S if k.startswith(("wiki:", "omniasr:", "glotcc:", "fw2rm:", "fw2:", "hplt2:kas", "sangraha:kas"))
                  and k not in ("fw2:urd_Arab", "glotcc:urd-Arab", "fw2:pbt_Arab", "fw2rm:pbt_Arab", "fw2:snd_Arab", "fw2rm:snd_Arab")],
}
def selected_sources():
    if SOURCES_SEL in ("", "all"): return list(S)
    out = []
    for item in SOURCES_SEL.split(","):
        item = item.strip()
        if item.startswith("group:"): out += GROUPS[item[6:]]
        elif item in S: out.append(item)
        else: raise SystemExit(f"unknown source {item}; known: {sorted(S)} or group:{sorted(GROUPS)}")
    return list(dict.fromkeys(out))

TIER_LANGS = ["urd_Arab", "urd_Latn", "pbt_Arab", "snd_Arab", "pnb_Arab", "skr_Arab", "bal_Arab", "brh_Arab", "kas_Arab", "hno_Arab", "phr_Arab"]
CONFUSERS = ["fas_Arab", "arb_Arab", "eng_Latn"]
LID_MIN = {l: 0.6 for l in TIER_LANGS}; LID_MIN.update({"hno_Arab": 0.8, "phr_Arab": 0.8, "bal_Arab": 0.7, "brh_Arab": 0.7})

# ------------------------------------------------------------------------------------------ normalisation
ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
DIGITS.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
PRESENTATION = re.compile(r"[ﭐ-﷿ﹰ-﻿]+")
ZW = {ord(c): None for c in "ـ‌‍‎‏﻿"}          # tatweel, ZWNJ, ZWJ, LRM, RLM, BOM
YEH_KAF = {ord("ي"): "ی", ord("ك"): "ک", ord("ى"): "ی"}   # Arabic yeh/kaf/alef maksura -> Urdu-script forms
SINDHI = {ord("ی"): "ي", ord("ى"): "ي"}                   # Sindhi writes Arabic yeh; its kaf letters are left alone
PASHTO = {ord("ك"): "ک"}                                  # Pashto keeps ي (-i, plural) and ی (-ay) as distinct letters; unify kaf only

def base_norm(text):
    t = text.replace("\r\n", "\n").replace("\r", "\n").translate(DIGITS).translate(ZW)
    if PRESENTATION.search(t): t = PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), t)
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()

def lang_norm(t, lang):
    if lang == "snd_Arab": return t.translate(SINDHI)
    if lang == "pbt_Arab": return t.translate(PASHTO)
    if lang.endswith("_Arab"): return t.translate(YEH_KAF)
    return t

def script_of(lang): return lang.split("_")[1]

def lid_norm(t):
    """The ONE transform PakLID sees, in training and in serving: base_norm + Arabic yeh/kaf folded to Urdu forms,
    lower-cased. Label-independent, so the classifier cannot learn the per-language output normalisation."""
    return t.translate(YEH_KAF).lower()

# ------------------------------------------------------------------------------------------ quality
BOILER = re.compile(r"(cookie|javascript|all rights reserved|click here|subscribe|log ?in|password|©|https?://)", re.I)
def quality(text, script):
    n = len(text)
    if n < 200: return "short"
    if n > 300_000: return "long"
    letters = 0; arabic = 0; ascii_ = 0; digits = 0
    for c in text:
        if c.isalpha():
            letters += 1
            if c.isascii(): ascii_ += 1
            elif ARABIC_RE.match(c): arabic += 1
        elif c.isdigit(): digits += 1
    if letters < 100: return "few_letters"
    if script == "Arab" and arabic / letters < 0.80: return "script_share"
    if script == "Latn" and ascii_ / letters < 0.90: return "script_share"
    if digits / n > 0.25: return "digits"
    words = text.split()
    if len(words) < 30: return "few_words"
    if sum(map(len, words)) / len(words) > 14: return "long_words"
    if collections.Counter(words).most_common(1)[0][1] / len(words) > 0.20: return "repetition"
    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        if sum(1 for l in lines if BOILER.search(l)) / len(lines) > 0.30: return "boilerplate"
        if 1 - len(set(lines)) / len(lines) > 0.50: return "dup_lines"
    return None

# ------------------------------------------------------------------------------------------ PakLID
CLF = None
def lid_windows(t):
    """Two 500-char windows (head and middle) so a mixed page is judged on more than its header."""
    t = lid_norm(t)
    if len(t) <= 1000: return [t[:500], t[500:1000] or t[:500]]
    mid = len(t) // 2
    return [t[:500], t[mid:mid + 500]]

def lid_predict(texts):
    xs = []
    for t in texts: xs += lid_windows(t)
    proba = CLF.predict_proba(xs)
    proba = (proba[0::2] + proba[1::2]) / 2
    labs = CLF.classes_
    i = proba.argmax(1)
    return [labs[j] for j in i], proba.max(1)

def iter_parquet_rows(repo, glob_, field, keep, max_rows=None):
    """Yield (text, meta) samples spread across an upstream parquet glob (row groups at an even stride),
    read remotely with column projection, so a 5 GB file costs only the row groups we touch.
    glob_ = "stream:<split>" uses datasets streaming instead (for CSV/JSON repos)."""
    if glob_.startswith("stream:"):
        from datasets import load_dataset
        n = 0
        for row in load_dataset(repo, split=glob_[7:], streaming=True, token=TOKEN):
            t = row.get(field)
            if isinstance(t, str) and t.strip():
                n += 1; yield t, {}
                if max_rows and n >= max_rows: return
        return
    files = sorted(f for f in hub_files(repo) if glob_match(f, glob_))
    if not files: raise RuntimeError(f"no files match {repo}/{glob_}")
    plan = []
    for f in files:
        md = remote_parquet(repo, f).metadata
        plan += [(f, rg, md.row_group(rg).num_rows) for rg in range(md.num_row_groups)]
    total = sum(r for _, _, r in plan); want = max_rows or total
    need_groups = max(1, math.ceil(len(plan) * min(1.0, 1.5 * want / max(1, total))))
    stride = max(1, len(plan) // need_groups)
    n = 0; cur = (None, None)
    for f, rg, _ in plan[::stride]:
        if cur[0] != f: cur = (f, remote_parquet(repo, f))
        pf = cur[1]
        cols = [c for c in [field] + keep if c in pf.schema_arrow.names]
        tbl = pf.read_row_group(rg, columns=cols).to_pydict()
        for i, t in enumerate(tbl[field]):
            if not isinstance(t, str) or not t.strip(): continue
            n += 1
            yield t, {k: tbl[k][i] for k in cols if k != field}
            if max_rows and n >= max_rows: return

_hub_cache = {}
def hub_files(repo, revision=None):
    k = (repo, revision)
    if k not in _hub_cache: _hub_cache[k] = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    return _hub_cache[k]

_fs = None
def hffs():
    global _fs
    if _fs is None:
        from huggingface_hub import HfFileSystem
        _fs = HfFileSystem(token=TOKEN)
    return _fs

def remote_parquet(repo, path, revision=None):
    """Open an upstream parquet file over HTTP range requests (footer + requested column chunks only)."""
    rev = f"@{revision}" if revision else ""
    return pq.ParquetFile(hffs().open(f"datasets/{repo}{rev}/{path}", "rb", block_size=8 * 2**20))

def fetch_local(repo, path, revision=None):
    """Download one upstream file into SCRATCH (not the shared cache), so deleting it really frees disk."""
    d = os.path.join(SCRATCH, "upstream", repo.replace("/", "__"))
    for attempt in range(5):
        try: return hf_hub_download(repo, path, repo_type="dataset", token=TOKEN, revision=revision, local_dir=d)
        except Exception as e: print("   upstream download retry", path, str(e)[:100]); time.sleep(20 * (attempt + 1))
    raise RuntimeError("upstream download failed: " + path)
def glob_match(path, pattern):
    import fnmatch
    return fnmatch.fnmatch(path, pattern) and path.count("/") == pattern.count("/")

def chunks_for_lid(text, lo=60, hi=500):
    out = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        while len(para) > hi:
            cut = max(para.rfind("۔", 0, hi), para.rfind(".", 0, hi), para.rfind(" ", 0, hi))
            cut = cut if cut > lo else hi
            out.append(para[:cut + 1].strip()); para = para[cut + 1:].strip()
        if len(para) >= lo: out.append(para)
    return out

def train_paklid():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    cap = PAKLID_PER_SOURCE
    # (label, upstream repo, glob, field, max docs, independent of GlotLID?)
    seeds = [
        ("urd_Arab", "wikimedia/wikipedia", "20231101.ur/*.parquet", "text", 3000, True),
        ("urd_Arab", "ai4bharat/sangraha", "verified/urd/data-0.parquet", "text", 2000, True),
        ("urd_Latn", "Mavkif/Roman-Urdu-Parl-split", "stream:train", "Roman-Urdu text", 6000, True),
        ("urd_Latn", "HuggingFaceFW/fineweb-2", "data/urd_Latn/train/*.parquet", "text", 1500, False),
        ("pbt_Arab", "wikimedia/wikipedia", "20231101.ps/*.parquet", "text", 3000, True),
        ("pbt_Arab", "facebook/omnilingual-asr-corpus", "data/pbt_Arab/train-*.parquet", "raw_text", 3000, True),
        ("snd_Arab", "wikimedia/wikipedia", "20231101.sd/*.parquet", "text", 3000, True),
        ("snd_Arab", "ai4bharat/sangraha", "verified/snd/data-1.parquet", "text", 2000, True),
        ("pnb_Arab", "wikimedia/wikipedia", "20231101.pnb/*.parquet", "text", 4000, True),
        ("pnb_Arab", "facebook/omnilingual-asr-corpus", "data/pnb_Arab/train-*.parquet", "raw_text", 3000, True),
        ("skr_Arab", "wikimedia/wikipedia", "20231101.skr/*.parquet", "text", 4000, True),
        ("skr_Arab", "HuggingFaceFW/fineweb-2", "data/skr_Arab/train/*.parquet", "text", 500, False),
        ("bal_Arab", "cis-lmu/GlotCC-V1", "v1.0/bal-Arab/*.parquet", "content", 3000, False),
        ("brh_Arab", "facebook/omnilingual-asr-corpus", "data/brh_Arab/train-*.parquet", "raw_text", 3000, True),
        ("brh_Arab", "HuggingFaceFW/fineweb-2", "data/brh_Arab/train/*.parquet", "text", 1500, False),
        ("kas_Arab", "wikimedia/wikipedia", "20231101.ks/*.parquet", "text", 2000, True),
        ("kas_Arab", "facebook/omnilingual-asr-corpus", "data/kas_Arab/train-*.parquet", "raw_text", 3000, True),
        ("hno_Arab", "facebook/omnilingual-asr-corpus", "data/hno_Arab/train-*.parquet", "raw_text", 4000, True),
        ("phr_Arab", "facebook/omnilingual-asr-corpus", "data/phr_Arab/train-*.parquet", "raw_text", 4000, True),
        ("fas_Arab", "HuggingFaceFW/fineweb-2", "data/fas_Arab/train/000_00000.parquet", "text", 3000, False),
        ("arb_Arab", "HuggingFaceFW/fineweb-2", "data/arb_Arab/train/000_00000.parquet", "text", 3000, False),
        ("eng_Latn", "wikimedia/wikipedia", "20231101.en/train-00000-of-*.parquet", "text", 3000, True),
    ]
    rows = []   # (label, group id, source, chunk)
    counts = collections.Counter()
    for lab, repo, g, field, n, indep in seeds:
        n = min(n, cap) if cap else n
        src_name = f"{repo.split('/')[-1]}:{g.split('/')[0] if '/' in g else g}"
        got_docs = 0
        for attempt in range(3):
            seed_rows, got_docs = [], 0
            try:
                for doc_i, (t, _) in enumerate(iter_parquet_rows(repo, g, field, [], max_rows=n * 2)):
                    t = base_norm(t)
                    cs = [c for c in chunks_for_lid(t) if not (script_of(lab) == "Arab" and len(ARABIC_RE.findall(c)) < 0.7 * sum(ch.isalpha() for ch in c))]
                    if len(cs) > MAX_CHUNKS_PER_DOC:   # spread seeds over many documents, not a few long articles
                        cs = [cs[i] for i in sorted(random.Random(doc_i).sample(range(len(cs)), MAX_CHUNKS_PER_DOC))]
                    seed_rows += [(lab, f"{src_name}#{doc_i}", src_name, lid_norm(c)) for c in cs]
                    got_docs += 1
                    if len(seed_rows) >= n: break
                break
            except Exception as e:
                print(f"   seed retry {attempt + 1}/3 {lab} {repo} {g}: {str(e)[:120]}", flush=True); seed_rows = []; time.sleep(20 * 3 ** attempt)
        rows += seed_rows; counts[(lab, src_name)] = len(seed_rows)
        print(f"   seed {lab:9s} {src_name:45s} chunks={counts[(lab, src_name)]:,} docs={got_docs:,} {'independent' if indep else 'GlotLID-labelled'}", flush=True)
    min_chunks, min_docs = (50, 5) if cap else (300, 20)   # every label must have real data before anything is fitted or uploaded
    need = TIER_LANGS + CONFUSERS
    have = {l: (sum(1 for r in rows if r[0] == l), len({r[1] for r in rows if r[0] == l})) for l in need}
    bad = {l: v for l, v in have.items() if v[0] < min_chunks or v[1] < min_docs}
    if bad: raise SystemExit(f"PakLID aborted: labels with too little seed data (chunks, documents): {bad}")
    # balance: at most 8k chunks per label, document-grouped 85/15 split
    by_lab = collections.defaultdict(list)
    for r in rows: by_lab[r[0]].append(r)
    rng = random.Random(13); train, test = [], []
    for lab, rs in by_lab.items():
        rng.shuffle(rs); rs = rs[:8000]
        groups = sorted({r[1] for r in rs}); rng.shuffle(groups); test_g = set(groups[: max(1, len(groups) * 15 // 100)])
        for r in rs: (test if r[1] in test_g else train).append(r)
    print(f"   PakLID train {len(train):,} chunks, test {len(test):,}; labels {sorted(by_lab)}", flush=True)
    clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True, max_features=600_000, dtype=np.float32),
                        CalibratedClassifierCV(LinearSVC(C=0.5), cv=3))
    clf.fit([r[3] for r in train], [r[0] for r in train])
    pred = clf.predict([r[3] for r in test])
    rep = classification_report([r[0] for r in test], pred, digits=3, zero_division=0, output_dict=True)
    labs = sorted(by_lab); cm = confusion_matrix([r[0] for r in test], pred, labels=labs).tolist()
    per_source = collections.defaultdict(lambda: [0, 0])
    for r, p in zip(test, pred): per_source[r[2] + "|" + r[0]][0] += int(p == r[0]); per_source[r[2] + "|" + r[0]][1] += 1
    report = {"version": "v1", "trained": time.strftime("%Y-%m-%d"), "train_chunks": len(train), "test_chunks": len(test),
              "seed_counts": {f"{a}|{b}": c for (a, b), c in counts.items()}, "classification_report": rep, "labels": labs, "confusion_matrix": cm,
              "per_source_accuracy": {k: round(a / b, 4) for k, (a, b) in per_source.items()},
              "note": "document-grouped held-out split; FineWeb-2/GlotCC seeds were labelled by GlotLID upstream (flagged), all others are independent"}
    path = os.path.join(SCRATCH, "paklid-v1.joblib"); joblib.dump(clf, path, compress=3)
    store = Store(PAKLID_REPO, "model")
    store.add_file("paklid-v1.joblib", path); store.add_json("paklid-v1-report.json", report)
    md = ["---", "license: apache-2.0", "tags: [language-identification, urdu, punjabi, pashto, sindhi, saraiki, balochi, brahui, kashmiri]", "---",
          "# PakLID v1", "", "Character 1-5-gram TF-IDF + calibrated linear SVM that identifies the Perso-Arabic-script languages of Pakistan (plus Roman Urdu, Persian, Arabic and English as confusers). Built by Balagh (Ownova) to re-sort web-corpus buckets whose upstream language labels are unreliable.", "",
          f"Held-out (document-grouped) macro F1: **{rep['macro avg']['f1-score']:.3f}**, accuracy {rep['accuracy']:.3f} on {len(test):,} chunks.", "",
          "| label | precision | recall | f1 | support |", "|---|---|---|---|---|"]
    md += [f"| {l} | {rep[l]['precision']:.3f} | {rep[l]['recall']:.3f} | {rep[l]['f1-score']:.3f} | {int(rep[l]['support'])} |" for l in labs if l in rep]
    md += ["", "Usage: normalise exactly as in training (digits to ASCII, tatweel/zero-width removed, Arabic yeh/kaf folded to Urdu forms, lower-cased), then `joblib.load(hf_hub_download('balagh-lm/paklid', 'paklid-v1.joblib')).predict_proba([text[:500]])`. "
           "Code: `lid_norm` and `base_norm` in `phase0/colab/01_build_corpus.py` at github.com/balagh-lm/balagh."]
    store.add_bytes("README.md", "\n".join(md).encode()); store.flush("PakLID v1")
    print(f"   PakLID v1 macro-F1 {rep['macro avg']['f1-score']:.3f} acc {rep['accuracy']:.3f}; uploaded to {PAKLID_REPO}")
    return clf

def load_paklid():
    import joblib
    store = Store(PAKLID_REPO, "model")
    if "paklid-v1.joblib" not in store.files(): raise SystemExit("PakLID v1 not found: run MODE=paklid first")
    clf = joblib.load(store.download("paklid-v1.joblib"))
    missing = set(TIER_LANGS + CONFUSERS) - set(clf.classes_)
    if missing: raise SystemExit(f"PakLID model lacks labels {sorted(missing)}: retrain with MODE=paklid")
    return clf

# ------------------------------------------------------------------------------------------ route
def route_batch(batch):
    """Worker: batch of (text, meta) -> list of (pred|None, reason, conf, text, meta)."""
    bucket = batch[0]; items = batch[1]
    sc = script_of(bucket)
    out = [None] * len(items); cand = []; cand_i = []
    for i, (t, meta) in enumerate(items):
        b = base_norm(t)
        r = quality(b, sc)
        if r: out[i] = (None, "q_" + r, 0.0, None, None)
        else: cand.append(b); cand_i.append(i)
    if cand:
        labs, confs = lid_predict(cand)
        for i, b, lab, conf in zip(cand_i, cand, labs, confs):
            meta = items[i][1]
            if lab in CONFUSERS: out[i] = (None, "lid_" + lab, float(conf), None, None)
            elif conf < LID_MIN.get(lab, 0.6): out[i] = (None, "lid_lowconf", float(conf), None, None)
            elif script_of(lab) != sc: out[i] = (None, "lid_script_mismatch", float(conf), None, None)
            else: out[i] = (lab, "ok", float(conf), lang_norm(b, lab), meta)
    return out

POOL_TIMEOUT = int(os.environ.get("POOL_TIMEOUT", 3 * 3600))
def pool():
    if WORKERS <= 1: return None
    import multiprocessing as mp
    return mp.get_context("fork").Pool(WORKERS)

def units_of(repo, glob_, revision=None):
    """Deterministic units: (file, first row group, last row group + 1, rows), from parquet footers only."""
    files = sorted(f for f in hub_files(repo, revision) if glob_match(f, glob_))
    units = []
    for f in files:
        md = remote_parquet(repo, f, revision).metadata
        start, rows = 0, 0
        for rg in range(md.num_row_groups):
            rows += md.row_group(rg).num_rows
            if rows >= UNIT_ROWS or rg == md.num_row_groups - 1:
                units.append((f, start, rg + 1, rows)); start, rows = rg + 1, 0
    return units

def route_source(key, routed, done):
    s = S[key]; ks = key.replace(":", "__")
    rev_path = f"_progress/{ks}/revision.json"
    if rev_path in done:
        rev = routed.read_json(rev_path)
        if not rev or not rev.get("sha"): raise RuntimeError(f"{key}: revision.json exists but is unreadable or unpinned; refusing to re-pin")
    else:
        rev = None
    if not rev:
        sha = None
        for attempt in range(5):
            try: sha = api.dataset_info(s["repo"]).sha; break
            except Exception as e: print("   pin retry", attempt, str(e)[:100], flush=True); time.sleep(15 * (attempt + 1))
        if not sha: raise RuntimeError(f"cannot pin the upstream revision for {key}")
        rev = {"repo": s["repo"], "sha": sha, "glob": s["glob"], "pinned": time.strftime("%Y-%m-%d")}
        routed.add_json(rev_path, rev); routed.flush(f"{key}: pin upstream revision")
    units = units_of(s["repo"], s["glob"], rev.get("sha"))
    if not units: raise RuntimeError(f"{key}: no upstream files match {s['repo']}/{s['glob']}")
    print(f"== {key}: {len(units)} units, {sum(u[3] for u in units):,} rows, upstream {s['repo']}@{(rev.get('sha') or '?')[:8]}", flush=True)
    p = pool(); n_done = 0; finished = {ui for ui in range(len(units)) if f"_progress/{ks}/u{ui:05d}.json" in done}
    try:
        for ui, (f, rg0, rg1, nrows) in enumerate(units):
            marker = f"_progress/{ks}/u{ui:05d}.json"
            if marker in done: continue
            if MAX_UNITS and n_done >= MAX_UNITS: break
            if time.time() - T_START > WALL_SECONDS:
                print("   wall-clock guard: stopping before unit", ui); routed.flush(f"{key} (session end)"); return False
            t0 = time.time()
            local = fetch_local(s["repo"], f, rev.get("sha"))
            pf = pq.ParquetFile(local); cols = [c for c in [s["field"]] + s["keep"] if c in pf.schema_arrow.names]
            tbl = pf.read_row_groups(list(range(rg0, rg1)), columns=cols).to_pydict()
            items = []
            for i, t in enumerate(tbl[s["field"]]):
                if isinstance(t, str) and t.strip():
                    meta = {k: tbl[k][i] for k in cols if k != s["field"]}
                    items.append((t, json.dumps(meta, ensure_ascii=False, default=str)))
            del tbl
            batches = [(s["bucket"], items[i:i + 1000]) for i in range(0, len(items), 1000)]
            res = (p.map_async(route_batch, batches, chunksize=1).get(timeout=POOL_TIMEOUT) if p else [route_batch(b) for b in batches])
            buckets = collections.defaultdict(lambda: {"text": [], "meta": [], "sha1": [], "lid_conf": []})
            stats = collections.Counter(); conf_sum = collections.Counter()
            for r in (x for rr in res for x in rr):
                lab, reason, conf, text, meta = r
                if lab is None: stats[reason] += 1; continue
                b = buckets[lab]; b["text"].append(text); b["meta"].append(meta); b["sha1"].append(hashlib.sha1(text.encode("utf-8")).hexdigest()); b["lid_conf"].append(round(conf, 4))
                stats["routed_" + lab] += 1; conf_sum[lab] += conf
            for lab, b in buckets.items():
                t = pa.table(b).replace_schema_metadata({"source": key, "upstream": s["repo"], "revision": rev.get("sha") or "", "licence": s["licence"], "bucket": s["bucket"]})
                tmp = os.path.join(SCRATCH, f"route-{ks}-{ui}-{lab}.parquet"); pq.write_table(t, tmp, compression="zstd")
                routed.add_file(f"{lab}/{ks}/u{ui:05d}.parquet", tmp)
            unit_stats = {"source": key, "unit": ui, "file": f, "row_groups": [rg0, rg1], "rows": len(items), "bucket": s["bucket"], "counts": dict(stats),
                          "mean_conf": {l: round(conf_sum[l] / stats["routed_" + l], 4) for l in conf_sum}, "seconds": round(time.time() - t0, 1)}
            routed.add_json(marker, unit_stats); routed.maybe_flush(f"{key} up to unit {ui}")
            n_done += 1; finished.add(ui)
            top = ", ".join(f"{k[7:]}={v:,}" for k, v in stats.most_common() if k.startswith("routed_"))
            print(f"   {key} u{ui}/{len(units)}: {len(items):,} docs in {time.time()-t0:.0f}s -> {top}; dropped {sum(v for k, v in stats.items() if not k.startswith('routed_')):,}", flush=True)
            if ui + 1 == len(units) or units[ui + 1][0] != f:
                try: os.remove(local)
                except OSError: pass
    finally:
        if p: p.close(); p.join()
    if len(finished) == len(units) and f"_progress/{ks}/complete.json" not in done:
        routed.add_json(f"_progress/{ks}/complete.json", {"source": key, "units": len(units), "revision": rev.get("sha"), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    routed.flush(f"{key} done")
    return True

# ------------------------------------------------------------------------------------------ dedup + decontam
import xxhash
P31 = np.uint64(2**31 - 1)
_rng = np.random.RandomState(20261017)
MH_A = _rng.randint(1, 2**31 - 1, size=128).astype(np.uint64)
MH_B = _rng.randint(0, 2**31 - 1, size=128).astype(np.uint64)
GRAMS = None   # sorted uint64 array of eval 13-gram hashes for the language being processed

def word_hashes(text):
    w = text.split()
    return np.fromiter((xxhash.xxh64_intdigest(x.encode("utf-8")) for x in w), dtype=np.uint64, count=len(w))

def ngram_hashes(wh, n):
    m = len(wh) - n + 1
    if m <= 0: return np.empty(0, dtype=np.uint64)
    h = wh[:m].copy()
    for j in range(1, n):
        x = wh[j:j + m]
        h ^= (x << np.uint64(j)) | (x >> np.uint64(64 - j))
    return h

def minhash_bands(wh):
    sh = np.unique(ngram_hashes(wh, 5) if len(wh) >= 5 else wh)
    s32 = ((sh ^ (sh >> np.uint64(32))) & np.uint64(0xFFFFFFFF))
    sig = np.full(128, np.iinfo(np.uint64).max, dtype=np.uint64)
    for i in range(0, len(s32), 2048):
        blk = s32[i:i + 2048, None]
        sig = np.minimum(sig, ((blk * MH_A[None, :] + MH_B[None, :]) % P31).min(0))
    sig32 = sig.astype(np.uint32)
    return np.array([xxhash.xxh64_intdigest(sig32[b * 8:(b + 1) * 8].tobytes()) for b in range(16)], dtype=np.uint64)

def dedup_batch(texts):
    out = []
    for t in texts:
        wh = word_hashes(t)
        g = ngram_hashes(wh, 13)
        contaminated = False
        if len(g) and GRAMS is not None and len(GRAMS):   # GRAMS is sorted (np.unique): binary search, O(n log m) per doc
            pos = np.searchsorted(GRAMS, g)
            contaminated = bool((GRAMS[np.minimum(pos, len(GRAMS) - 1)] == g).any())
        out.append((contaminated, None if contaminated else minhash_bands(wh), len(wh)))
    return out

EVAL_SETS = [   # (hf id, config, splits or None for all, text builder) -- everything we report, plus FLORES+ dev/devtest
    ("MBZUAI/UrduMMLU", None, ["test"], lambda r: [r.get("question", "")] + [str(v) for v in (r.get("options") or {}).values()] if isinstance(r.get("options"), dict) else [r.get("question", "")] + [str(v) for v in (r.get("options") or [])]),
] + [("facebook/belebele", c, ["test"], lambda r: [r["flores_passage"], r["question"], r["mc_answer1"], r["mc_answer2"], r["mc_answer3"], r["mc_answer4"]]) for c in ["urd_Arab", "urd_Latn", "pbt_Arab", "snd_Arab", "pan_Guru", "eng_Latn"]] \
  + [("Davlan/sib200", c, None, lambda r: [r["text"]]) for c in ["urd_Arab", "pbt_Arab", "snd_Arab", "kas_Arab", "pan_Guru"]] \
  + [("mrlbenchmarks/global-piqa-nonparallel", c, None, lambda r: [str(v) for v in r.values() if isinstance(v, str)]) for c in ["urd_arab", "urd_latn", "snd_arab", "bcc_arab", "bsk_arab"]] \
  + [("openlanguagedata/flores_plus", c, None, lambda r: [r["text"]]) for c in ["urd_Arab", "pbt_Arab", "snd_Arab", "kas_Arab"]] \
  + [("google/smol", c, None, lambda r: [str(x) for x in (r.get("srcs") or [r.get("src")]) if x] + [str(x) for x in (r.get("trgs") or [r.get("trg")]) if x]) for c in ["smolsent__en_ur", "smolsent__en_pa-Arab", "smolsent__en_brh", "smolsent__en_ks", "smoldoc__en_ur", "smoldoc__en_ps", "smoldoc__en_brh"]]
# SMOL is in the decontamination list because the SFT set uses SMOL and evaluation may report SMOL-style translation.

_eval_cache = None
def eval_texts():
    global _eval_cache
    if _eval_cache is not None: return _eval_cache
    from datasets import load_dataset
    texts, used, failed = [], [], []
    for ds_id, cfg, splits, fn in EVAL_SETS:
        err = None
        for attempt in range(3):
            try:
                d = load_dataset(ds_id, cfg, token=TOKEN) if cfg else load_dataset(ds_id, token=TOKEN)
                got = []
                for sp in (splits or list(d)):
                    if sp not in d: continue
                    for r in d[sp]: got += [x for x in fn(r) if isinstance(x, str) and len(x.split()) >= 13]
                texts += got; used.append(f"{ds_id}/{cfg or '-'}: {len(got):,} texts"); err = None; break
            except Exception as e:
                err = e; time.sleep(20 * (attempt + 1))
        if err is not None: failed.append(f"{ds_id}/{cfg or '-'}: {str(err)[:110]}")
    print("   eval sets for decontamination:\n     " + "\n     ".join(used + ["FAILED " + f for f in failed]), flush=True)
    if (failed or len(texts) < 5000) and not (LOCAL_OUT or os.environ.get("ALLOW_PARTIAL_DECONTAM") == "1"):
        raise SystemExit("decontamination sets unavailable (accept the FLORES+ terms on the Hub and use a token that can read gated repos), "
                         f"refusing to build a corpus that may contain benchmark text: {failed}")
    _eval_cache = (texts, used); return _eval_cache

def eval_grams(lang):
    texts, used = eval_texts()
    hs = [ngram_hashes(word_hashes(lang_norm(base_norm(t), lang)), 13) for t in texts]
    g = np.unique(np.concatenate(hs)) if hs else np.empty(0, dtype=np.uint64)
    return g, used

def dedup_language(lang, routed, out, done_out):
    global GRAMS
    if f"{lang}/stats.json" in done_out and os.environ.get("FORCE") != "1":
        print(f"skip {lang}: stats.json present (FORCE=1 to redo)"); return
    files = sorted(routed.files(f"{lang}/"), key=lambda f: (S.get(f.split("/")[1].replace("__", ":"), {"prio": 9})["prio"], f))
    files = [f for f in files if f.endswith(".parquet")]
    if not files: print(f"== {lang}: no routed files"); return
    t0 = time.time(); GRAMS, used = eval_grams(lang)
    print(f"== {lang}: {len(files)} routed files; {len(GRAMS):,} eval 13-grams", flush=True)
    p = pool()   # forked after GRAMS is set, so workers see it
    seen = set(); bands = []; spill = []; stats = collections.Counter(); by_source = collections.Counter()
    spill_dir = os.path.join(SCRATCH, f"spill-{lang}"); shutil.rmtree(spill_dir, ignore_errors=True); os.makedirs(spill_dir)
    try:
        for fi, f in enumerate(files):
            local = routed.download(f); t = pq.read_table(local)
            src_key = f.split("/")[1].replace("__", ":")
            texts = t.column("text").to_pylist(); sha = t.column("sha1").to_pylist()
            keep_i = []
            for i, h in enumerate(sha):
                k = int(h[:16], 16)
                if k in seen: stats["exact_dup"] += 1; continue
                seen.add(k); keep_i.append(i)
            sub = [texts[i] for i in keep_i]
            chunks = [sub[i:i + 500] for i in range(0, len(sub), 500)]
            res = (p.map_async(dedup_batch, chunks, chunksize=1).get(timeout=POOL_TIMEOUT) if p else [dedup_batch(c) for c in chunks])
            res = [x for r in res for x in r]
            rows_keep = []
            for i, (contam, b, nw) in zip(keep_i, res):
                if contam: stats["contaminated"] += 1; continue
                rows_keep.append(i); bands.append(b)
            if rows_keep:
                st = t.take(pa.array(rows_keep)).append_column("source", pa.array([src_key] * len(rows_keep)))
                sp = os.path.join(spill_dir, f"{fi:05d}.parquet"); pq.write_table(st, sp, compression="zstd"); spill.append((sp, len(rows_keep)))
            stats["in"] += len(texts)
            if not LOCAL_OUT and local.startswith(SCRATCH):
                try: os.remove(local)
                except OSError: pass
            if fi % 20 == 0: print(f"   {lang} {fi+1}/{len(files)} files: in {stats['in']:,}, exact dup {stats['exact_dup']:,}, contaminated {stats['contaminated']:,} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        if p: p.close(); p.join()
    if not bands: print(f"   {lang}: nothing left"); return
    B = np.stack(bands); N = len(B); dup = np.zeros(N, dtype=bool)
    for b in range(16):
        _, first, inv = np.unique(B[:, b], return_index=True, return_inverse=True)
        dup |= first[inv.ravel()] != np.arange(N)
    del B, bands; gc.collect()
    stats["near_dup"] = int(dup.sum())
    # write final shards
    SH = 50_000; buf = []; n_sh = 0; words = 0; chars = 0; gi = 0
    def emit(rows, n_sh):
        tbl = pa.concat_tables(rows).replace_schema_metadata({"corpus": "balagh-lm/corpus-v1", "language": lang})
        tmp = os.path.join(SCRATCH, f"final-{lang}-{n_sh}.parquet"); pq.write_table(tbl, tmp, compression="zstd")
        out.add_file(f"{lang}/shard-{n_sh:05d}.parquet", tmp)
        if len(out.pending) >= 8: out.maybe_flush(f"{lang} shards up to {n_sh}")
    pending_rows = []; pending_n = 0
    for sp, n in spill:
        t = pq.read_table(sp); mask = ~dup[gi:gi + n]; gi += n; os.remove(sp)
        t = t.filter(pa.array(mask))
        if t.num_rows == 0: continue
        for txt, s_ in zip(t.column("text").to_pylist(), t.column("source").to_pylist()):
            words += len(txt.split()); chars += len(txt); by_source[s_] += 1
        pending_rows.append(t); pending_n += t.num_rows
        while pending_n >= SH:
            tbl = pa.concat_tables(pending_rows); emit([tbl.slice(0, SH)], n_sh); n_sh += 1
            rest = tbl.slice(SH); pending_rows = [rest] if rest.num_rows else []; pending_n = rest.num_rows
    if pending_n: emit(pending_rows, n_sh); n_sh += 1
    stats.update({"kept": int((~dup).sum()), "shards": n_sh, "words": words, "chars": chars, "seconds": int(time.time() - t0)})
    out.add_json(f"{lang}/stats.json", {**dict(stats), "kept_by_source": dict(by_source), "eval_sets": used, "minhash": "5-word shingles, 128 perms, 16x8 bands"})
    out.flush(f"{lang} final shards + stats")
    shutil.rmtree(spill_dir, ignore_errors=True)
    print(f"   {lang} DONE: in {stats['in']:,} -> kept {stats['kept']:,} docs ({words/1e6:.1f}M words); exact {stats['exact_dup']:,}, near {stats['near_dup']:,}, contaminated {stats['contaminated']:,}; {n_sh} shards, {stats['seconds']}s", flush=True)

# ------------------------------------------------------------------------------------------ card
def write_card(routed, out):
    files = out.files()
    rows = []; tot_words = 0
    for lang in TIER_LANGS:
        if f"{lang}/stats.json" not in files: continue
        st = out.read_json(f"{lang}/stats.json", {})
        tot_words += st.get("words", 0)
        rows.append(f"| {lang} | {st.get('in', 0):,} | {st.get('kept', 0):,} | {st.get('words', 0)/1e6:,.1f}M | {st.get('exact_dup', 0) + st.get('near_dup', 0):,} | {st.get('contaminated', 0):,} |")
    matrix = collections.defaultdict(collections.Counter)
    for f in routed.files("_progress/"):
        if f.endswith(".json") and "/u" in f:
            u = routed.read_json(f, {})
            for k, v in u.get("counts", {}).items():
                if k.startswith("routed_"): matrix[u.get("source", "?")][k[7:]] += v
    langs = [l for l in TIER_LANGS if any(l in c for c in matrix.values())]
    mrows = [f"| {s} | " + " | ".join(f"{matrix[s][l]:,}" if matrix[s][l] else "" for l in langs) + " |" for s in sorted(matrix)]
    card = f"""---
license: other
license_name: mixed-open
license_link: https://github.com/balagh-lm/balagh
language: [ur, pa, ps, sd, skr, bal, brh, ks]
pretty_name: Balagh corpus v1 (languages of Pakistan)
---
# Balagh corpus v1

Cleaned, language-checked, deduplicated and benchmark-decontaminated text for the languages of Pakistan, built by [Balagh](https://github.com/balagh-lm/balagh) (Ownova) from openly packaged corpora. Total: **{tot_words/1e9:.2f}B words**.

| language | routed docs in | kept docs | words | duplicates removed | eval-overlap removed |
|---|---|---|---|---|---|
{chr(10).join(rows)}

## Where documents really belong (routing matrix)
Each upstream bucket was re-labelled with [PakLID v1](https://huggingface.co/balagh-lm/paklid). Rows are upstream sources, columns the language PakLID assigned (documents passing quality filters and confidence thresholds).

| source | {' | '.join(langs)} |
|---|{'---|' * len(langs)}
{chr(10).join(mrows)}

## Sources and licences
FineWeb-2 (ODC-By-1.0 + Common Crawl ToU), HPLT 2.0 cleaned (CC0-1.0 packaging), GlotCC-V1 (CC0-1.0 packaging), Sangraha verified (CC-BY-4.0), Wikipedia 2023-11-01 (CC-BY-SA-3.0), Omnilingual-ASR transcripts (CC-BY-4.0). Packaging licences do not clear the copyright of the underlying pages; every document keeps its `source` and upstream metadata, and upstream revisions are pinned in `balagh-lm/corpus-v1-routed/_progress/*/revision.json`. Takedown requests: muneeb@ownova.org.

## Processing
Per-language normalisation without global NFKC (Arabic yeh/kaf mapped to Urdu forms except in Sindhi, digits to ASCII, tatweel and zero-width characters removed, presentation forms decomposed), quality filters (length, script share, word length, repetition, boilerplate, duplicate lines, digit ratio), PakLID v1 routing (confidence >= 0.6; 0.7 for Balochi/Brahui; 0.8 for Hindko/Pothwari), exact dedup (SHA-1), MinHash near-dedup (5-word shingles, 128 permutations, 16 bands x 8 rows; on collision the higher-priority source is kept: Wikipedia/transcripts > Sangraha > HPLT > FineWeb-2 > GlotCC > FineWeb-2 removed), and 13-word-gram decontamination against UrduMMLU, Belebele, SIB-200, Global PIQA, FLORES+ and SMOL. Code: `phase0/colab/01_build_corpus.py`.
"""
    out.add_bytes("README.md", card.encode("utf-8")); out.flush("data card")
    print(card[:3000])

# ------------------------------------------------------------------------------------------ main
if __name__ == "__main__":
    print(f"MODE={MODE} workers={WORKERS} scratch={SCRATCH} {'DRY RUN -> ' + LOCAL_OUT if LOCAL_OUT else 'Hub'}", flush=True)
    if MODE == "paklid":
        train_paklid()
    elif MODE == "route":
        CLF = load_paklid()
        routed = Store(ROUTED_REPO); done = routed.files("_progress/"); failed = []
        try:
            for key in selected_sources():
                try:
                    if route_source(key, routed, done) is False: failed.append(key + " (paused by the wall-clock guard)"); break
                except Exception as e:
                    failed.append(key); print(f"   FAILED {key}: {str(e)[:300]}"); traceback.print_exc(limit=3)
        finally:
            try: routed.flush("session end")
            except Exception as e: print("   final flush failed:", str(e)[:200])
        if failed: raise SystemExit(f"route incomplete, rerun this notebook to resume: {failed}")
    elif MODE == "dedup":
        routed = Store(ROUTED_REPO); out = Store(OUT_REPO); done_out = out.files()
        langs = [l.strip() for l in (LANGS_SEL.split(",") if LANGS_SEL else TIER_LANGS) if l.strip()]
        unknown = [l for l in langs if l not in TIER_LANGS]
        if unknown: raise SystemExit(f"unknown LANGS {unknown}; known: {TIER_LANGS}")
        prog = routed.files("_progress/")
        incomplete = [k for k in S if f"_progress/{k.replace(':', '__')}/complete.json" not in prog]
        if incomplete and os.environ.get("ALLOW_INCOMPLETE") != "1" and not LOCAL_OUT:
            raise SystemExit(f"routing is not complete for {incomplete}: finish MODE=route first (ALLOW_INCOMPLETE=1 to dedup anyway)")
        for lang in langs:
            try: dedup_language(lang, routed, out, done_out)
            except Exception as e: print(f"   FAILED {lang}: {str(e)[:300]}"); traceback.print_exc(limit=3)
            gc.collect()
    elif MODE == "card":
        write_card(Store(ROUTED_REPO), Store(OUT_REPO))
    else:
        raise SystemExit("MODE must be paklid, route, dedup or card")
    print(f"DONE in {(time.time() - T_START)/60:.1f} min")
