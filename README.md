# Distributed LLM Labeler & Offline Chatbot

A distributed, fault-tolerant labeling system and an offline retrieval-augmented chatbot.
Built with FastAPI + Ollama + confidence calibration. Runs locally, scales across nodes, and
emits append-only JSONL shards you can merge into a single dataset.

---

## Features

- Distributed labeling with leader election and auto failover
- Shard-based outputs with peer replication
- Ollama-backed classifier that returns strict JSON: {"search_needed": 0|1, "confidence": float}
- Confidence calibration (temperature scaling / Platt, per-domain shrink)
- Input preprocessing: slang/abbrev expansion, emoji demojize, NER hints (<ENT>…</ENT>)


---

## Project Structure

app/
  Llm_classifer_script.py      # Core classifier (Ollama + prompt assembly + calibration hook)
  Prosses_user_input.py        # Tokenize/standardize + NER + ENT tagging
  calibration_api.py           # Calibrators + manager + auto-fit/save
  node.py                      # FastAPI node (leader election, claims, shard streaming, replication)
  app_data/
    Calibration_data/          # CSVs and saved calibrators.json
    Abbreviations and Slang.csv
data/
  questions.csv                # Input (must have "text"; optional "domain")
output/                        # Local append-only JSONL shards
replicated/                    # Peer shards replicated here
state/                         # Node state (progress, epoch, leader flag)
merge_results.py               # Dedup by SHA1(id) → final CSV
Dockerfile
docker-compose.yml
README.md (this file)
Engineering a Robust Search Classification Pipeline for an Offline Retrieval-Augmented Chatbot.pdf

---

## Prerequisites

- Python 3.11+
- Ollama installed and running
- A local Ollama model pulled (default used here: qwen2.5:0.5b-instruct)
- spaCy English model (en_core_web_sm)

---

## Install

$ pip install -r requirements.txt
$ python -m spacy download en_core_web_sm
$ ollama pull qwen2.5:0.5b-instruct

---

## Quickstart

Single node:

$ python app/node.py --csv data/questions.csv --port 8001 --mode server

Three nodes on one machine (example):

$ python app/node.py --csv data/questions.csv --port 8001 --peers 127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003 --mode auto
$ python app/node.py --csv data/questions.csv --port 8002 --peers 127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003 --mode auto
$ python app/node.py --csv data/questions.csv --port 8003 --peers 127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003 --mode auto

Merge all labeled shards:

$ python merge_results.py

Use the classifier directly in Python:

from app.Llm_classifer_script import llmClassifier
clf = llmClassifier(gpu=True)
print(clf.classify("who is the ceo of openai?"))
# -> {"search_needed": 1, "confidence": 0.xx}

---

## Data Format

- Required CSV column: text
- Optional CSV column: domain  (e.g., general, programming)
- Output JSONL fields per row: id, idx, text, domain, label, label_id, confidence, worker, ts

---

## CLI Options (node.py)

--csv PATH           Input CSV path
--port INT           HTTP port for this node
--peers LIST         Comma-separated peer addresses (host:port)
--mode MODE          auto | server | client
--batch INT          Claim batch size
--prefer-leader      In auto mode, bias this node to lead

---

## HTTP Endpoints (node)

GET  /status       Current node status
GET  /progress     Rows processed + % done (approx)
GET  /peers        Known peers + health
GET  /ping         Liveness probe
POST /claim        (Leader only) Assign a [start,end] work range
GET  /shards       List completed local shard files
GET  /pull?name=   Stream a specific shard file

---

## Notes

- If you maintain your own model/decoding settings, update DEFAULT_MODEL and DEFAULT_OPTIONS
  in Llm_classifer_script.py.
- Calibration expects labeled CSVs with columns: confidence (float), search_needed (0/1).
- The repo includes a detailed PDF write-up of the pipeline and design decisions.

---

## Project Report

./Engineering a Robust Search Classification Pipeline for an Offline Retrieval-Augmented Chatbot.pdf

---

## License

Copyright (c) 2025 Baraa Mohaisen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use, 
copy, modify, and distribute the Software, subject to the following conditions:

1. Attribution Required  
   Any use of the Software, in whole or in part, must include proper credit to  
   the original author: "Copyright (c) 2025 Baraa Mohaisen".  

2. Non-Commercial Use  
   The Software may not be sold, licensed, or otherwise monetized in any form  
   without prior written permission from the copyright holder.  

3. No Warranty  
   The Software is provided "AS IS", without warranty of any kind, express or  
   implied, including but not limited to the warranties of merchantability,  
   fitness for a particular purpose and noninfringement. In no event shall the  
   author or copyright holder be liable for any claim, damages, or other  
   liability, whether in an action of contract, tort, or otherwise, arising from,  
   out of, or in connection with the Software or the use or other dealings in  
   the Software.

---

## Author

Baraa Mohaisen
University of Washington Bothell — Electrical Engineering
GitHub: https://github.com/baraam-commits
LinkedIn: https://www.linkedin.com/in/baraa-mohaisen-46522b30a/
