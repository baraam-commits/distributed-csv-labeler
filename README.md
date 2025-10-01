# Distributed LLM Labeler 

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

Creative Commons Attribution-NonCommercial 4.0 International Public License

By exercising the Licensed Rights (defined below), You accept and agree to be bound 
by the terms and conditions of this Creative Commons Attribution-NonCommercial 4.0 
International Public License ("Public License"). To the extent this Public License 
may be interpreted as a contract, You are granted the Licensed Rights in consideration 
of Your acceptance of these terms and conditions, and the Licensor grants You such 
rights in consideration of benefits the Licensor receives from making the Licensed 
Material available under these terms and conditions.

Section 1 – Definitions.
a. Adapted Material means material subject to Copyright and Similar Rights that is 
   derived from or based upon the Licensed Material and in which the Licensed Material 
   is translated, altered, arranged, transformed, or otherwise modified in a manner 
   requiring permission under the Copyright and Similar Rights held by the Licensor.
b. Licensed Material means the artistic or literary work, database, or other material 
   to which the Licensor applied this Public License.
c. Licensed Rights means the rights granted to You subject to the terms and conditions 
   of this Public License, which are limited to all Copyright and Similar Rights that 
   apply to Your use of the Licensed Material and that the Licensor has authority to license.
d. Licensor means the individual(s) or entity(ies) granting rights under this Public License.
e. You means the individual or entity exercising the Licensed Rights under this Public License.

Section 2 – Scope.
a. License grant. Subject to the terms and conditions of this Public License, the Licensor 
   hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable 
   license to exercise the Licensed Rights in the Licensed Material to:
   i. reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and
   ii. produce, reproduce, and Share Adapted Material for NonCommercial purposes only.
b. Attribution. If You Share the Licensed Material (including in modified form), You must:
   i. give appropriate credit to the Licensor, provide a link to the license, and indicate if changes were made; and
   ii. not in any way suggest the Licensor endorses You or Your use.
c. NonCommercial. You may not exercise the Licensed Rights in any manner that is primarily intended 
   for or directed toward commercial advantage or monetary compensation.

Section 3 – Disclaimer of Warranties and Limitation of Liability.
THE LICENSED MATERIAL IS PROVIDED “AS-IS” AND WITHOUT WARRANTIES OF ANY KIND. 
TO THE GREATEST EXTENT PERMISSIBLE BY LAW, NEITHER THE LICENSOR NOR ITS AFFILIATES 
SHALL BE LIABLE FOR ANY DAMAGES ARISING OUT OF THE USE OF THE LICENSED MATERIAL.

Section 4 – Term and Termination.
This Public License is effective as long as You comply with its terms. The Licensor 
may terminate this license if You fail to do so. 

For the full license text, see: https://creativecommons.org/licenses/by-nc/4.0/

---

## Author

Baraa Mohaisen
University of Washington Bothell — Electrical Engineering
GitHub: https://github.com/baraam-commits
LinkedIn: https://www.linkedin.com/in/baraa-mohaisen-46522b30a/
