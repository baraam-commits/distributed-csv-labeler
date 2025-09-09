#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Labeler Script — Quick Reference
--------------------------------

Usage:
  python labeler.py <input_file> [options]

Supported input formats (pick one; default is CSV if none given):
  --csv                     Treat input as CSV (default if none specified)
  --tsv                     Treat input as TSV (tab-delimited)
  --jsonl                   Treat input as JSONL (one JSON object per line)

Selecting the text field/column:
  --text-col NAME           (CSV/TSV) Column to label (default: text)
  --text-field NAME         (JSONL)   Field to label (default: text)

Output:
  -o PATH, --output-path PATH
                            Custom output path (default: <input>.labeled.{csv|jsonl})
  --new-file                Start a fresh labeled file; if output exists, timestamp a new file
  --force-csv               Always export as CSV, even if input is JSONL/TSV

Skip behavior:
  --skip-default-no-search  When you press SKIP, auto-label as No-Search(0) with confidence 1.0
  (Default) Consecutive duplicate rows (same text as previous) are auto-skipped
  --keep-consecutive-duplicates   Disable auto-skip of consecutive duplicates

Targets (optional; shown every row):
  --target-total N          Stop prompt after labeling N items (unless you continue)
  --target-search N         Target for "Search(1)" labels
  --target-no-search N      Target for "No-Search(0)" labels
  --auto-stop               Exit immediately when all specified targets are met

Controls (while labeling):
  A = No-Search(0)  → then enter confidence (default 1.0 if you just press Enter)
  D = Search(1)     → then enter confidence
  J = Quick No-Search(0) with confidence 1.0 (no prompt)
  K = Quick Search(1)    with confidence 1.0 (no prompt)
  B = Back/Edit previous row
  S = Skip (optionally default to No-Search(0) with --skip-default-no-search)
  Q = Quit

Notes:
- Prints metrics every row: labeled, remaining, Search/No-Search counts, and targets (if any).
- Saves after each label (atomic write). Safe to stop anytime; just rerun to resume.
- Works per-file; run separately for each domain/dataset file.
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime

# =========================
# Keybindings (customize me)
# =========================
KEY_NO_SEARCH       = "a"   # label 0 (then ask for confidence)
KEY_SEARCH          = "d"   # label 1 (then ask for confidence)
KEY_QUICK_NOSEARCH  = "j"   # label 0, confidence 1.0 (no prompt)
KEY_QUICK_SEARCH    = "k"   # label 1, confidence 1.0 (no prompt)
KEY_BACK            = "b"   # go back one row to edit/overwrite
KEY_SKIP            = "s"   # skip current row
KEY_QUIT            = "q"   # quit cleanly

# Only these three fields matter for labeling output
FIELDS_OUT = ["text", "gold_search_needed", "gold_confidence"]

# ---------- Utilities ----------

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def resume_index(rows):
    """First index where either label or confidence is missing."""
    for i, r in enumerate(rows):
        if r.get("gold_search_needed") in (None, "") or r.get("gold_confidence") in (None, ""):
            return i
    return len(rows)

def validate_conf(s):
    s = (s or "").strip()
    if s == "":
        return 1.0
    try:
        v = float(s)
        if 0.0 <= v <= 1.0:
            return v
    except ValueError:
        pass
    return None

def compute_metrics(rows):
    total = len(rows)
    labeled = search = nosearch = 0
    for r in rows:
        gl, gc = r.get("gold_search_needed"), r.get("gold_confidence")
        if gl not in ("", None) and gc not in ("", None):
            labeled += 1
            if gl == "1":
                search += 1
            elif gl == "0":
                nosearch += 1
    return {"total": total, "labeled": labeled, "remaining": total - labeled,
            "search": search, "nosearch": nosearch}

def targets_met(m, tgt_total, tgt_search, tgt_nosearch):
    if tgt_total    is not None and m["labeled"]  < tgt_total:     return False
    if tgt_search   is not None and m["search"]   < tgt_search:    return False
    if tgt_nosearch is not None and m["nosearch"] < tgt_nosearch:  return False
    return True

def render(i, row, m, targets):
    clear()
    print(f"[{i+1}/{m['total']}]")
    print("-" * 80)
    print(row["text"])
    print("-" * 80)
    gl = row.get("gold_search_needed", "")
    gc = row.get("gold_confidence", "")
    label_str = {"1": "Search(1)", "0": "No-Search(0)"}.get(gl, "—")
    print(f"Current row: label={label_str} | confidence={gc if gc else '—'}\n")
    print(f"Progress: labeled {m['labeled']}/{m['total']} | remaining {m['remaining']}")
    print(f"Counts:   Search(1)={m['search']} | No-Search(0)={m['nosearch']}")
    tparts = []
    if targets['total']    is not None: tparts.append(f"Target total={targets['total']}")
    if targets['search']   is not None: tparts.append(f"Target Search(1)={targets['search']}")
    if targets['nosearch'] is not None: tparts.append(f"Target No-Search(0)={targets['nosearch']}")
    if tparts: print(" | ".join(tparts))
    print("\nControls:")
    print(f"  {KEY_NO_SEARCH.upper()} = No-Search(0)   {KEY_SEARCH.upper()} = Search(1)")
    print(f"  {KEY_QUICK_NOSEARCH.upper()} = Quick No-Search(0) [conf=1.0]   {KEY_QUICK_SEARCH.upper()} = Quick Search(1) [conf=1.0]")
    print(f"  {KEY_BACK.upper()} = Back/Edit           {KEY_SKIP.upper()} = Skip")
    print(f"  {KEY_QUIT.upper()} = Quit")

# --- Dupe detection (consecutive) ---

_ws_re = re.compile(r"\s+")

def _norm_text_for_dupe(s: str) -> str:
    if s is None:
        return ""
    # trim, collapse whitespace, lowercase for robust equality on near-identical lines
    return _ws_re.sub(" ", s.strip()).lower()

def is_consecutive_duplicate(rows, i) -> bool:
    if i <= 0:
        return False
    return _norm_text_for_dupe(rows[i]["text"]) == _norm_text_for_dupe(rows[i-1]["text"])

# ---------- CSV/TSV IO ----------

def load_input_rows_csv(path, text_col="text", delimiter=","):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=delimiter)
        if text_col not in (r.fieldnames or []):
            raise ValueError(f"Input must have a '{text_col}' column (got: {r.fieldnames})")
        return [{"text": row.get(text_col, ""), "gold_search_needed": "", "gold_confidence": ""} for row in r]

def load_output_rows_csv(path, delimiter=","):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for row in r:
            rows.append({"text": row.get("text", ""),
                         "gold_search_needed": row.get("gold_search_needed", ""),
                         "gold_confidence": row.get("gold_confidence", "")})
    return rows

def atomic_save_csv(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_OUT)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS_OUT})
    os.replace(tmp, path)

# ---------- JSONL IO ----------

def load_input_rows_jsonl(path, text_field="text"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if text_field not in obj:
                raise ValueError(f"JSONL objects must have '{text_field}' (got keys: {list(obj.keys())})")
            rows.append({"text": obj[text_field], "gold_search_needed": "", "gold_confidence": ""})
    return rows

def load_output_rows_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            rows.append({"text": obj.get("text",""),
                         "gold_search_needed": obj.get("gold_search_needed",""),
                         "gold_confidence": obj.get("gold_confidence","")})
    return rows

def atomic_save_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k, "") for k in FIELDS_OUT}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description="Search/No-Search labeler with quick keys, metrics, resume, and per-file text field/column.")
    p.add_argument("input_path", help="Input file (CSV/TSV/JSONL).")
    # input format flags
    p.add_argument("--csv", action="store_true", help="Treat input as CSV (default if none specified).")
    p.add_argument("--tsv", action="store_true", help="Treat input as TSV (tab-delimited).")
    p.add_argument("--jsonl", action="store_true", help="Treat input as JSONL.")
    # where to read text from
    p.add_argument("--text-col", default="text", help="(CSV/TSV) Column to label (default: text)")
    p.add_argument("--text-field", default="text", help="(JSONL) Field to label (default: text)")
    # output & session mgmt
    p.add_argument("-o", "--output-path", default=None, help="Output path (default: <input>.labeled.{csv|jsonl})")
    p.add_argument("--new-file", action="store_true", help="Start fresh; if output exists, timestamp a new file.")
    p.add_argument("--force-csv", action="store_true", help="Always export as CSV, even if input is JSONL/TSV.")
    p.add_argument("--skip-default-no-search", action="store_true",
                   help="Treat skipped rows as No-Search(0) with confidence 1.0")
    # dupes
    p.add_argument("--keep-consecutive-duplicates", action="store_true",
                   help="Do NOT auto-skip consecutive duplicate texts (default is to skip).")
    # targets
    p.add_argument("--target-total", type=int, default=None)
    p.add_argument("--target-search", type=int, default=None)
    p.add_argument("--target-no-search", type=int, default=None)
    p.add_argument("--auto-stop", action="store_true")
    args = p.parse_args()

    # Determine input format
    in_fmt = "csv"
    if args.tsv: in_fmt = "tsv"
    if args.jsonl: in_fmt = "jsonl"

    base, _ext = os.path.splitext(args.input_path)

    # Decide default output path based on format (unless forced CSV)
    if args.force_csv:
        default_out = base + ".labeled.csv"
    else:
        default_out = base + (".labeled.jsonl" if in_fmt == "jsonl" else ".labeled.csv")
    out_path = args.output_path or default_out

    # Load rows (resume from existing output unless --new-file)
    if not args.new_file and os.path.exists(out_path):
        if out_path.lower().endswith(".jsonl"):
            rows = load_output_rows_jsonl(out_path)
        else:
            rows = load_output_rows_csv(out_path, delimiter="\t" if out_path.lower().endswith(".tsv") else ",")
    else:
        if in_fmt == "jsonl":
            rows = load_input_rows_jsonl(args.input_path, text_field=args.text_field)
        elif in_fmt == "tsv":
            rows = load_input_rows_csv(args.input_path, text_col=args.text_col, delimiter="\t")
        else:
            rows = load_input_rows_csv(args.input_path, text_col=args.text_col, delimiter=",")
        if args.new_file and os.path.exists(out_path):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            root, ext = os.path.splitext(out_path)
            out_path = f"{root}.{ts}{ext or '.csv'}"

    if not rows:
        print("No rows found.")
        return

    targets = {"total": args.target_total, "search": args.target_search, "nosearch": args.target_no_search}

    # If resuming and targets already satisfied
    m = compute_metrics(rows)
    if targets_met(m, targets["total"], targets["search"], targets["nosearch"]):
        print("Targets already met on this file.")
        if args.auto_stop:
            return
        input("Press Enter to continue labeling anyway…")

    i = resume_index(rows)

    while 0 <= i < len(rows):
        # Auto-skip consecutive duplicates unless disabled
        if not args.keep_consecutive_duplicates and is_consecutive_duplicate(rows, i):
            # (Optional) print a one-line note; then skip forward
            # print(f"Skipped duplicate row {i+1} (same as previous).")
            i += 1
            continue

        m = compute_metrics(rows)
        row = rows[i]
        render(i, row, m, targets)

        # Check targets mid-session
        if targets_met(m, targets["total"], targets["search"], targets["nosearch"]):
            if args.auto_stop:
                break
            ans = input("🎯 Targets met. Enter=continue, 'q'=quit: ").strip().lower()
            if ans == "q":
                break

        ch = input("Choice: ").strip().lower()

        if ch == KEY_QUIT:
            break
        if ch == KEY_BACK:
            i = max(0, i - 1)
            continue

        if ch == KEY_SKIP:
            if args.skip_default_no_search:
                row["gold_search_needed"] = "0"
                row["gold_confidence"] = "1.0"
                # Save immediately
                if args.force_csv or out_path.lower().endswith(".csv") or out_path.lower().endswith(".tsv"):
                    atomic_save_csv(out_path, rows)
                else:
                    atomic_save_jsonl(out_path, rows)
            i += 1
            continue

        # Quick 1.0 confidence keys
        if ch == KEY_QUICK_NOSEARCH:
            row["gold_search_needed"] = "0"
            row["gold_confidence"] = "1.0"
            if args.force_csv or out_path.lower().endswith(".csv") or out_path.lower().endswith(".tsv"):
                atomic_save_csv(out_path, rows)
            else:
                atomic_save_jsonl(out_path, rows)
            i += 1
            continue

        if ch == KEY_QUICK_SEARCH:
            row["gold_search_needed"] = "1"
            row["gold_confidence"] = "1.0"
            if args.force_csv or out_path.lower().endswith(".csv") or out_path.lower().endswith(".tsv"):
                atomic_save_csv(out_path, rows)
            else:
                atomic_save_jsonl(out_path, rows)
            i += 1
            continue

        # Standard keys (ask for confidence)
        if ch == KEY_NO_SEARCH:
            row["gold_search_needed"] = "0"
        elif ch == KEY_SEARCH:
            row["gold_search_needed"] = "1"
        else:
            input("Invalid key. Press Enter to continue.")
            continue

        # Confidence input (only for standard keys)
        while True:
            cin = input("Confidence [0..1, Enter=1.0]: ")
            conf = validate_conf(cin)
            if conf is not None:
                break
            print("Invalid confidence. Enter a number between 0 and 1.")
        row["gold_confidence"] = f"{conf:.6f}"

        # Save after each annotation (honor --force-csv)
        if args.force_csv or out_path.lower().endswith(".csv") or out_path.lower().endswith(".tsv"):
            atomic_save_csv(out_path, rows)
        else:
            atomic_save_jsonl(out_path, rows)

        i += 1

    # Final save
    if args.force_csv or out_path.lower().endswith(".csv") or out_path.lower().endswith(".tsv"):
        atomic_save_csv(out_path, rows)
    else:
        atomic_save_jsonl(out_path, rows)

    print(f"\n✅ Saved progress to: {out_path}")

if __name__ == "__main__":
    main()
