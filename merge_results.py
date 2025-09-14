import glob, json, hashlib, pandas as pd, os

# 1) raw + stable id
raw = pd.read_csv("data/questions.csv")
raw["id"] = raw["text"].apply(lambda t: hashlib.sha1((t or "").encode("utf-8")).hexdigest())

# 2) collect JSONL from both output/ and replicated/
rows = []
for root in ["output","replicated"]:
    for path in glob.glob(os.path.join(root, "labels_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
lab = pd.DataFrame(rows)

# 3) dedup: keep latest by timestamp (or swap for highest confidence)
if not lab.empty:
    lab = lab.sort_values("ts").drop_duplicates("id", keep="last")

# 4) join & save
gold = raw.merge(lab[["id","label","confidence"]], on="id", how="left")
gold.to_csv("output/gold.csv", index=False)
print("Wrote output/gold.csv with", gold["label"].notna().sum(), "labels of", len(gold))