# Balagh data pipeline, step 5: tokenise balagh-lm/corpus-v1 with the Gemma 4 tokenizer into packed
# uint32 shards (fixed-length sequences with document boundaries marked by EOS), build the training mix,
# and export it (a) to the Hub as balagh-lm/tokens-gemma4-v1 and (b) to a Kaggle dataset so TPU
# sessions read it from local disk. CPU runtime; needs HF_TOKEN and, for (b), KAGGLE_USERNAME +
# KAGGLE_KEY in Colab Secrets (your Kaggle API token from kaggle.com/settings; never paste in a cell).
#
# Mix for the 10B-token Balagh-nano run (tokens): Urdu 4.0B, Pashto 1.0B (<=2 epochs), Sindhi 0.6B
# (<=3 epochs), Shahmukhi 0.25B (<=4 epochs), Saraiki 0.1B, Balochi 0.08B, Brahui 0.03B, Kashmiri 0.04B,
# Roman Urdu 0.3B, replay: English 2.6B + code 0.6B + math 0.4B from FineWeb-Edu / StarCoder / FineMath
# samples (streamed here, permissive licences). Edit MIX to change. Sequences of 2048 tokens.

import os, io, json, time, math, random, collections
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

TOK = os.environ.get("TOKENIZER", "google/gemma-4-E2B")
CORPUS = os.environ.get("BALAGH_OUT_REPO", "balagh-lm/corpus-v1")
OUT = os.environ.get("BALAGH_TOKENS_REPO", "balagh-lm/tokens-gemma4-v1")
KAGGLE_DS = os.environ.get("KAGGLE_DATASET", "balagh-tokens-gemma4-v1")
SEQ = 2048; SHARD_SEQS = 8192           # 8192 x 2048 x 4 bytes = 64 MB per shard
MAX_TOKENS_TOTAL = float(os.environ.get("MAX_TOKENS_TOTAL", 0)) or None   # smoke test: e.g. 20e6
random.seed(0)

def secret(k):
    try:
        from google.colab import userdata; return userdata.get(k)
    except Exception: return os.environ.get(k)
TOKEN = secret("HF_TOKEN"); api = HfApi(token=TOKEN); api.create_repo(OUT, repo_type="dataset", exist_ok=True)
tok = AutoTokenizer.from_pretrained(TOK, token=TOKEN); EOS = tok.eos_token_id or tok.pad_token_id
assert tok.vocab_size < 2**32

# name -> (loader kwargs for load_dataset, text field, token budget, max epochs)
MIX = {
    "urd_Arab":  (dict(path=CORPUS, data_files="urd_Arab/*.parquet"), "text", 4.0e9, 1),
    "pbt_Arab":  (dict(path=CORPUS, data_files="pbt_Arab/*.parquet"), "text", 1.0e9, 2),
    "snd_Arab":  (dict(path=CORPUS, data_files="snd_Arab/*.parquet"), "text", 0.6e9, 3),
    "pnb_Arab":  (dict(path=CORPUS, data_files="pnb_Arab/*.parquet"), "text", 0.25e9, 4),
    "skr_Arab":  (dict(path=CORPUS, data_files="skr_Arab/*.parquet"), "text", 0.10e9, 4),
    "bal_Arab":  (dict(path=CORPUS, data_files="bal_Arab/*.parquet"), "text", 0.08e9, 4),
    "brh_Arab":  (dict(path=CORPUS, data_files="brh_Arab/*.parquet"), "text", 0.03e9, 4),
    "kas_Arab":  (dict(path=CORPUS, data_files="kas_Arab/*.parquet"), "text", 0.04e9, 4),
    "urd_Latn":  (dict(path=CORPUS, data_files="urd_Latn/*.parquet"), "text", 0.30e9, 2),
    "en_edu":    (dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT"), "text", 2.6e9, 1),
    "code":      (dict(path="bigcode/starcoderdata", data_dir="python"), "content", 0.6e9, 1),
    "math":      (dict(path="HuggingFaceTB/finemath", name="finemath-4plus"), "text", 0.4e9, 1),
}
if MAX_TOKENS_TOTAL:
    total = sum(v[2] for v in MIX.values()); MIX = {k: (a, b, c * MAX_TOKENS_TOTAL / total, d) for k, (a, b, c, d) in MIX.items()}

def shard_name(src, n): return f"{src}/shard-{n:05d}.npy"
done = set(api.list_repo_files(OUT, repo_type="dataset"))
manifest = {}

def tokenise_source(src, loader, field, budget, max_epochs):
    produced = 0; n_shard = 0; buf = []; carry = []
    while shard_name(src, n_shard) in done: n_shard += 1; produced += SHARD_SEQS * SEQ
    if produced >= budget: return produced, n_shard
    for epoch in range(max_epochs):
        try: ds = load_dataset(split="train", streaming=True, token=TOKEN, **loader)
        except Exception as e: print(f"   {src}: cannot load ({str(e)[:100]})"); return produced, n_shard
        if epoch == 0 and n_shard: ds = ds.skip(0)   # resumption is at shard granularity only; later epochs re-read from the start
        texts = []
        for row in ds:
            t = row.get(field)
            if not isinstance(t, str) or len(t) < 50: continue
            texts.append(t)
            if len(texts) >= 512:
                for ids in tok(texts, add_special_tokens=False)["input_ids"]:
                    carry.extend(ids); carry.append(EOS)
                texts = []
                while len(carry) >= SEQ:
                    buf.append(np.array(carry[:SEQ], dtype=np.uint32)); carry = carry[SEQ:]
                    if len(buf) >= SHARD_SEQS:
                        arr = np.stack(buf); bio = io.BytesIO(); np.save(bio, arr); bio.seek(0)
                        api.upload_file(path_or_fileobj=bio, path_in_repo=shard_name(src, n_shard), repo_id=OUT, repo_type="dataset", commit_message=f"{src} shard {n_shard}")
                        n_shard += 1; produced += arr.size; buf = []
                        print(f"   {src}: {produced/1e9:.3f}B tokens, shard {n_shard}", flush=True)
                        if produced >= budget: return produced, n_shard
        print(f"   {src}: epoch {epoch+1} exhausted at {produced/1e9:.3f}B (budget {budget/1e9:.2f}B)")
    return produced, n_shard

for src, (loader, field, budget, max_epochs) in MIX.items():
    print(f"== {src} budget {budget/1e9:.2f}B tokens", flush=True)
    t0 = time.time()
    try: produced, n = tokenise_source(src, loader, field, budget, max_epochs)
    except Exception as e: print("   FAILED", src, str(e)[:200]); produced, n = 0, 0
    manifest[src] = {"tokens": produced, "shards": n, "budget": budget, "seconds": int(time.time() - t0)}
    api.upload_file(path_or_fileobj=json.dumps({"tokenizer": TOK, "seq_len": SEQ, "eos": EOS, "sources": manifest}, indent=1).encode(), path_in_repo="manifest.json", repo_id=OUT, repo_type="dataset", commit_message="manifest")
print("tokens total:", sum(m["tokens"] for m in manifest.values()) / 1e9, "B")

# ---- Kaggle dataset export (so TPU sessions read from local disk) ----
KU, KK = secret("KAGGLE_USERNAME"), secret("KAGGLE_KEY")
if KU and KK:
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = KU, KK
    os.system("pip -q install -U kaggle")
    from huggingface_hub import snapshot_download
    local = snapshot_download(OUT, repo_type="dataset", token=TOKEN, local_dir="/content/tokens")
    meta = {"title": KAGGLE_DS, "id": f"{KU}/{KAGGLE_DS}", "licenses": [{"name": "other"}]}
    json.dump(meta, open(os.path.join(local, "dataset-metadata.json"), "w"))
    rc = os.system(f"kaggle datasets create -p {local} --dir-mode zip -q || kaggle datasets version -p {local} --dir-mode zip -m 'update' -q")
    print("kaggle export rc:", rc, "->", f"{KU}/{KAGGLE_DS}")
else:
    print("KAGGLE_USERNAME/KAGGLE_KEY not set: skipped Kaggle export (TPU sessions can stream from the Hub instead, slower)")
