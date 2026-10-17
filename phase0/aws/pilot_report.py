# Compare pilot arms (one run per learning rate) and recommend the main-run LR.
# Usage: python pilot_report.py <runs_dir> <lr1> <lr2> ...   (run directories are <anything>-lr<lr>/metrics.jsonl)
# Rule: lowest mean held-out loss over the Pakistani-language sets among arms whose English held-out loss rose by
# at most ENG_TOL nats above its own step-0 value. Arms are comparable only if each has a step-0 evaluation and
# all end at the same step; otherwise the report says INCOMPLETE and exits 1 without recommending anything.
import sys, os, json, glob

runs_dir, lrs = sys.argv[1], sys.argv[2:]
ENG_TOL = float(os.environ.get("ENG_TOL", 0.05))
TARGET = ["urd_Arab", "pbt_Arab", "snd_Arab", "pnb_Arab", "skr_Arab", "urd_Latn", "bal_Arab", "brh_Arab", "kas_Arab"]
REPLAY_EN = "en_edu"

rows, problems = [], []
for lr in lrs:
    arms = glob.glob(os.path.join(runs_dir, f"*-lr{lr}", "metrics.jsonl"))
    if not arms: rows.append({"lr": lr, "missing": "no metrics.jsonl"}); problems.append(f"LR {lr}: no metrics"); continue
    evals, steps = {}, []
    for line in open(arms[0]):
        try: m = json.loads(line)
        except Exception: continue
        if "val" in m: evals[m["step"]] = m["val"]          # duplicates after a resume collapse onto one step
        if "tok_s" in m: steps.append(m)
    if 0 not in evals: rows.append({"lr": lr, "missing": "no step-0 evaluation"}); problems.append(f"LR {lr}: no step-0 baseline"); continue
    last_step = max(evals); first, last = evals[0], evals[last_step]
    tgt = [k for k in TARGET if k in last and k in first]
    if not tgt or REPLAY_EN not in last: rows.append({"lr": lr, "missing": "held-out sets missing"}); problems.append(f"LR {lr}: no target/English held-out sets"); continue
    mean_t = sum(last[k] for k in tgt) / len(tgt)
    d_t = mean_t - sum(first[k] for k in tgt) / len(tgt)
    d_en = last[REPLAY_EN] - first[REPLAY_EN]
    tps = sorted(s["tok_s"] for s in steps[2:])[len(steps[2:]) // 2] if len(steps) > 4 else None
    rows.append({"lr": lr, "steps": last_step, "target_mean": round(mean_t, 4), "target_delta": round(d_t, 4), "en_delta": round(d_en, 4),
                 "per_lang": {k: round(last[k] - first[k], 4) for k in tgt}, "median_tok_s": tps, "ok": d_en <= ENG_TOL})

done = [r for r in rows if not r.get("missing")]
if done and len({r["steps"] for r in done}) > 1: problems.append("arms end at different steps: " + ", ".join(f"{r['lr']}@{r['steps']}" for r in done))
out = ["# Balagh pilot report", "", f"English tolerance: held-out loss may rise by at most {ENG_TOL} nats.", "",
       "| LR | last step | target mean loss | change | English change | median tokens/s | eligible |", "|---|---|---|---|---|---|---|"]
for r in rows:
    if r.get("missing"): out.append(f"| {r['lr']} | {r['missing']} | | | | | |"); continue
    out.append(f"| {r['lr']} | {r['steps']} | {r['target_mean']} | {r['target_delta']:+} | {r['en_delta']:+} | {r['median_tok_s']} | {'yes' if r['ok'] else 'no'} |")
out += ["", "Per-language change in held-out loss (last eval minus step 0):", ""]
out += [f"- LR {r['lr']}: " + ", ".join(f"{k} {v:+}" for k, v in r["per_lang"].items()) for r in done]
if problems:
    out += ["", "**INCOMPLETE: arms are not comparable, no recommendation.**"] + [f"- {p}" for p in problems]
    print("\n".join(out)); sys.exit(1)
ok = [r for r in done if r["ok"]]
best = min(ok, key=lambda r: r["target_mean"]) if ok else None
out += ["", f"**Recommended main-run LR: {best['lr']}**" if best else "**No eligible arm: every LR raised English loss too much; lower the range and rerun.**"]
if best and best.get("median_tok_s"):
    tps = best["median_tok_s"]
    out.append(f"Measured throughput {tps:,} tokens/s = {tps*3600/1e6:.0f}M tokens/hour; 8B tokens = {8e9/tps/3600:.0f} h, 10B = {10e9/tps/3600:.0f} h. "
               f"Budget rule: TOTAL_TOKENS = 650 / price_per_hour x {tps:,} x 3600.")
print("\n".join(out))
sys.exit(0 if best else 1)
