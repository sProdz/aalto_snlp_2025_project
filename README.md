# aalto_snlp_2025_project
A minimal end‑to‑end project for the Statistical Natural Language Processing (SNLP 2025) course. It demonstrates how to index ~345 K famous quotes in Qdrant and retrieve the most similar ones to a user prompt with TF‑IDF (BOW), Word2Vec, and BERT embeddings.

⸻

1. Install Python dependencies with Poetry

Prerequisites: Python ≥ 3.10 and Poetry ≥ 1.7 installed on your system.
```bash
# Clone the repo and enter it
$ git clone https://github.com/<your‑org>/aalto‑snlp‑2025‑project.git
$ cd aalto‑snlp‑2025‑project

# Install the exact versions from pyproject.toml (plus dev deps)
$ poetry install --with dev

# Activate the virtual‑env for ad‑hoc commands
$ poetry shell
```


⸻

2. Download pre‑computed data from Google Drive

https://drive.google.com/drive/folders/1mEty_7JNAwxzjGzm3VtP-bBTNfn3VXdX?usp=sharing

```bash
# Install gdown once (outside the Poetry env is fine)
$ pip install --upgrade gdown

# Replace YOUR_LINK with the *folder* share link
$ gdown --folder "https://drive.google.com/drive/folders/1mEty_7JNAwxzjGzm3VtP-bBTNfn3VXdX?usp=sharing" -O .
```

gdown downloads the folder structure into the current directory. Make sure the final tree looks like:
```
project_root/
├── data/
│   ├── BERT/
│   │   ├── embeddings/
│   │   ├── models/
│   │       ├── finetuned/
│   ├── BOW/
│   ├── quotes/
│   └── word2vec/
│       ├── model/
├── docker-compose.yml
└── …
```

(Manual download + unzip works the same—just place data/ in the repo root.)

⸻

3. Run the stack with Docker Compose

```bash
# Compose the docker-compose.yml using
$ docker compose up

# Run it down when you are done the demo using
$ docker compose down
```

**NOTE: In case you are experiencing memory issues with Qdrant (docker), which is usually indicated by errors 137, you can limit the number of vectors being uploaded or run the database on disk.**

The ingest service reads the MAX_POINTS environment variable declared in docker‑compose.yml:

```
services:
  ingest:
    …
    environment:
      MAX_POINTS: 100      # ← change this number (max 345000)
```

Edit it before docker compose up

Stopping the services
```bash
# Gracefully stop and keep the Qdrant data volume
$ docker compose down

Wiping Qdrant for a fresh start

# Stop everything *and* delete the named volume `qdrant_storage`
$ docker compose down -v
# (re)start to ingest from scratch
$ docker compose up
```


⸻

4. Interactive search CLI

Run from inside the Poetry environment after the ingest job has finished:

```bash
$ poetry run python main.py \
    --collections BERT,W2V,BOW \
    --top_k 5 \
    --qdrant_url http://localhost:6333
```
```
Argument	Short	Default	Description
--collections	-c	(required)	Comma‑separated list of BERT, W2V, BOW specifying which vector collections to query.
--top_k	-k	5	Number of results per collection.
--qdrant_url	-u	http://localhost:6333	Address of the Qdrant HTTP endpoint.
````

The script prints an ASCII banner describing which collections are loaded. Type a sentence or fragment at the >> prompt and press Enter; the top‑k most similar quotes for each collection are printed. Enter exit or Ctrl‑C to quit.

⸻

## Troubleshooting & tips ##
- Ingest job finished too quickly? Check container logs: docker compose logs -f ingest.
- Missing model/vectorizer files → verify you downloaded the data/ folder into the repo root before running docker compose up.
- Docker failing mysteriously when uploading vectors (points) and seeing error codes 137? → Most probably out of memory, lower the number of points by following instructions in section 3



Happy quoting 📚✨