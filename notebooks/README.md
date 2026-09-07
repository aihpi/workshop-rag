# Workshop Notebooks

This folder contains notebooks for two workshops. Setup is shared — pick your workshop below:

| Workshop | Topic | Jump to |
|----------|-------|---------|
| **Workshop 2** | RAG Fundamentals — chunking, retrieval, embeddings, OCR | [Workshop 2 Notebooks](#workshop-2--rag-fundamentals) |
| **Workshop 3** | RAG Evaluation — RAGAS metrics, retrieval & generation evaluation | [Workshop 3 Notebooks](#workshop-3--rag-evaluation-with-ragas) |

---

## Setup (both workshops)

### Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Docker (running)

### 1) Prepare the notebooks environment

From the repository root:

```bash
./scripts/setup_notebooks_kernel.sh
```

Optional extras for OCR/VLM (Workshop 2 only):

```bash
# VLM support:
./scripts/setup_notebooks_kernel.sh --extra vlm

# Mac users (Apple Silicon, OCRMac + VLM):
./scripts/setup_notebooks_kernel.sh --extra mac_vlm
```

This will:
1. Create/use `./.venv-notebooks` in the repository root.
2. Install dependencies from `notebooks/pyproject.toml` into that root environment.
3. Register a Jupyter kernel named `workshop-ragv2` (display name `Python (workshop-ragV2)`).

Then open any notebook in VS Code and select the kernel. In the top-right of the notebook:
1. Click `Select Kernel`.
2. Click `Jupyter Kernel...`.
3. Pick `Python (workshop-ragV2)` (`.venv-notebooks/bin/python`).

### 2) Start Qdrant

Qdrant must be running **before** you open the notebooks. The notebooks connect to
it on port `6333`, but they do not start it for you.

**Docker Desktop**: under `Images`, click `Run` on `qdrant/qdrant`, expand
`Optional settings`, name the container `qdrant` and map host port `6333` to
container port `6333`. (If you do not have the image yet, search `qdrant/qdrant`
in the top search bar and pull it first.)

**Command line**, equivalent and keeps the data in `qdrant_storage/`:

```bash
docker run -d --name qdrant -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Or use the compose file from the repository root:

```bash
docker compose up -d
```

Verify it is up before continuing:

```bash
curl http://localhost:6333
```

The dashboard is at http://localhost:6333/dashboard.

### 3) Configure your API key

Copy the template and fill in your own credentials:

```bash
cd notebooks
cp .env.example .env
```

Open `notebooks/.env` and replace `your_openai_api_key_here` with your real API key:

```env
OPENAI_API_KEY=your_real_key_here
OPENAI_API_BASE=https://api.aisc.hpi.de/
```

`.env` is gitignored and must never be committed. `.env.example` is the tracked
template, so keep it free of secrets. Do not put your key into a notebook cell.

---

## Workshop 2 — RAG Fundamentals

After completing the [Setup](#setup-both-workshops), work through these notebooks in order:

| # | Notebook | Topic |
|---|----------|-------|
| 01 | `w2_01_chunking_and_retrieval.ipynb` | Chunking strategies and retrieval |
| 02 | `w2_02_embedding_models.ipynb` | Embedding model comparison |
| 03 | `w2_03_real_world_datentypen.ipynb` | Real-world data types |
| 04 | `w2_04_ocr_docling_vlm_comparison.ipynb` | OCR: Docling vs VLM comparison |

---

## Workshop 3 — RAG Evaluation with RAGAS

After completing the [Setup](#setup-both-workshops), work through these notebooks in order:

| # | Notebook | Topic | Duration |
|---|----------|-------|----------|
| 01 | `w3_01_intro_end_to_end.ipynb` | End-to-end RAG + RAGAS intro on a simple website | 30 min |
| 02 | `w3_02_ingestion.ipynb` | PDF → Docling → Chunks → Embeddings → Qdrant | 20 min |
| 03 | `w3_03_retrieval_evaluation.ipynb` | Context Precision & Context Recall, TOP_K experiment | 40 min |
| 04 | `w3_04_generation_evaluation.ipynb` | Answer Correctness & Faithfulness, prompt experiment | 40 min |

Notebooks 02–04 share settings via `ragkit/config.py`. See the config file for available parameters (dataset, chunking mode, embedding model, etc.).

---

## The `ragkit` package

Helpers that more than one notebook needs live in `notebooks/ragkit/`, so the
cells import them instead of redefining them:

```python
from ragkit.embed import embed, cached_embed
from ragkit.chunk import chunk_by_paragraph, normalize_text
from ragkit.search import rag_search, entropy
```

| Module | Holds |
|--------|-------|
| `config.py` | Paths, model names, chunking parameters, Qdrant settings. The one place to change an experiment. |
| `embed.py` | `embed()` plus the model-specific backends, retries, batching, the on-disk vector cache, image encoding. |
| `chunk.py` | Text normalisation (including the German umlaut repair) and every chunking strategy. |
| `search.py` | Qdrant access, cosine similarity, entropy, MRR/nDCG, answer generation from retrieved context. |
| `viz.py` | Kernel density estimate and figure-to-base64 encoding; the plots themselves stay in the notebooks. |

No installation step: the notebooks run with `notebooks/` as the working
directory, so the package is importable as-is.

Code that *is* the lesson stays in the notebooks, so the chunking-comparison
loop, the end-to-end RAG loop in w3_01 and the one-off figures are still there to
read and edit.

Embeddings are cached under `notebooks/embedding_cache/`. To force
re-computation:

```python
import ragkit.embed as rk_embed
rk_embed.RECREATE = True
```

Tests for the pure functions:

```bash
python tests/test_ragkit.py
```

---

## Qdrant

Dashboard: http://localhost:6333/dashboard

Useful Docker commands:

```bash
docker ps --filter name=qdrant
docker logs qdrant --tail 100
docker start qdrant
```

## Troubleshooting

**Kernel/interpreter does not appear:**

```bash
./scripts/setup_notebooks_kernel.sh --skip-sync
```

Then reload VS Code and select kernel `Python (workshop-ragV2)`.

**`uv: command not found`:**
- Install uv, then restart your terminal.

**Qdrant start fails:**
- Confirm Docker is running.
- Confirm port `6333` is free.
- Check existing container state with `docker ps -a --filter name=qdrant`.

**RAGAS `InstructorRetryException` / `max_tokens` errors:**
- The evaluator LLM may need a higher `max_tokens` — especially reasoning models like `gpt-oss-120b`.
- Increase `max_tokens` in the `llm_factory()` call (already set to 8192–65536 in notebooks 03/04).

**Embedding `batch_size` errors (413):**
- `minilm-embedding` has a max batch size of 32 (configured in `ragkit/config.py`).
- If you add a new embedding model, add its constraints to the `_EMBED_MODELS` registry in `ragkit/config.py`.
