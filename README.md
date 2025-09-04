# distributed-csv-labeler
A distributed, fault-tolerant labeling pipeline for preparing large datasets for BERT fine-tuning.  Built with FastAPI, Docker, and peer-to-peer shard replication.  Nodes elect leaders automatically, distribute CSV ranges, replicate results, and merge into a final training dataset.
