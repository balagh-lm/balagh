# Paper 1 annotation: short CSV packs (same packs as the Google Forms: 5 shared anchors + 25 unique texts).
# Reads research/audit/forms/packs.json and research/audit/annotation/{lang}.csv; writes
# Documents/Balagh-annotation-to-send/packs/{lang}-packN.csv with the same columns as the full sheets.
import csv, json, os

AUDIT = r"C:\Users\Muneeb\Documents\pakistan-languages-llm\research\audit"
OUT = r"C:\Users\Muneeb\Documents\Balagh-annotation-to-send\packs"
os.makedirs(OUT, exist_ok=True)
manifest = json.load(open(os.path.join(AUDIT, "forms", "packs.json"), encoding="utf-8"))
for lang, m in manifest.items():
    rows = list(csv.reader(open(os.path.join(AUDIT, "annotation", f"{lang}.csv"), encoding="utf-8-sig")))
    header, by_id = rows[0], {r[0]: r for r in rows[1:]}
    for name, ids in m["packs"].items():
        with open(os.path.join(OUT, f"{name}.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(header)
            for i in ids: w.writerow(by_id[i])
    print(lang, {n: len(v) for n, v in m["packs"].items()})
