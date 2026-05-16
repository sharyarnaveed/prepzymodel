# prepzymodel

Science education dataset preparation and model training utilities for Prepzy.

## Contents

- `dataset/` — Raw source text files (physics, chemistry, biology)
- `conversiontojsonl.py` — Convert raw texts to JSONL training format
- `datacompletion.py` — Complete missing outputs using a local LLM (CUDA)
- `dataset_train.jsonl` — Training dataset
- `clean_science_dataset.jsonl` — Cleaned science dataset

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers
```
