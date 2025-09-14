import glob
import json
import os
import hashlib
import pandas as pd

# Where shards are located
SHARD_DIRS = ["output", "replicated"]
OUTPUT_PATH = "output/gold.jsonl"

def main():
    rows = []

    # Load all shard files
    for root in SHARD_DIRS:
        for path in glob.glob(os.path.join(root, "labels_*.jsonl")):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue

    if not rows:
        print("⚠️ No shard files found.")
        return

    # Deduplicate by id: keep the most recent (highest ts)
    df = pd.DataFrame(rows)
    if "id" not in df.columns:
        raise RuntimeError("Shard entries must include an 'id' field.")

    # Sort so latest ts per id is last
    if "ts" in df.columns:
        df = df.sort_values("ts")

    # Drop duplicates, keeping last
    df = df.drop_duplicates("id", keep="last")

    # Keep only relevant columns (expand if you want more)
    keep_cols = [
        "id", "idx", "text", "domain",
        "label", "label_id", "confidence",
        "worker", "ts"
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None

    # Write JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            record = {c: row[c] for c in keep_cols}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(df)} unique rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()