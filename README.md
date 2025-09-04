# distributed-csv-labeler
# Distributed BERT Labeler

A distributed, fault-tolerant pipeline for labeling large datasets to train a BERT classifier.  
Built with **FastAPI**, **Docker**, and **peer-to-peer shard replication**, this system can scale across multiple machines on a local network and automatically recover from node crashes.

---

## 🚀 Features
- **Distributed Processing**: Multiple identical nodes (no separate server/client image).  
- **Leader Election**: Nodes elect a leader based on the most regressed index.  
- **Task Distribution**: Leader hands out contiguous CSV ranges for labeling.  
- **Shard Replication**: Each node writes append-only JSONL shards and pulls completed shards from peers.  
- **Fault Tolerance**: If the leader crashes, another node promotes itself and resumes without data loss.  
- **Final Merge**: All shards are deduplicated (by a stable SHA1 ID) into one gold CSV.  
- **Dockerized**: Build once, run anywhere (Linux, Windows, macOS).

---

## 📂 Project Structure

app/
├─ node.py # Core node logic (FastAPI + worker + replication)
├─ requirements.txt # Python dependencies
data/
└─ questions.csv # Input data (MUST have 'text' column)
output/ # Local labeled shard files
replicated/ # Shards pulled from peers
state/ # Node state (progress, leader info, CSV index)
Dockerfile # Container definition
docker-compose.yml # Multi-node local test harness
merge_results.py # Final dedup + merge script
