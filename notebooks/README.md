# Workshop notebooks

Learning material for two workshops of the HPI AI Service Centre. The notebooks are [marimo](https://marimo.io) apps: participants see explanations, figures and interactive controls, and can reveal the code behind every cell when they want to.

| Workshop | Topic | Notebooks |
|---|---|---|
| RAG II | Fundamentals: chunking, embeddings, retrieval, real-world document types, OCR | `w2_01` to `w2_04` |
| RAG III | Evaluation with RAGAS: ingestion, retrieval metrics, generation metrics | `w3_01` to `w3_04` |

## Setup

Prerequisites: Python 3.10 or newer, [`uv`](https://docs.astral.sh/uv/getting-started/installation/), Docker, and an API key for the HPI AISC inference API.

1. Install the environment (from this folder):

   ```bash
   cd notebooks
   uv sync
   ```

2. Start Qdrant (from the repository root; keeps its data in `qdrant_storage/`):

   ```bash
   docker compose up -d
   ```

   The dashboard is at http://localhost:6333/dashboard.

3. Add your API key:

   ```bash
   cp .env.example .env
   ```

   Replace `your_openai_api_key_here` in `.env` with your key. `.env` is gitignored; never paste a key into a notebook cell.

## Running a notebook

App view (what participants use; code hidden):

```bash
uv run marimo run w2_01_chunking_and_retrieval.py
```

Edit view (every cell's code can be expanded, edited and re-run):

```bash
uv run marimo edit w2_01_chunking_and_retrieval.py
```

Most notebooks start with a setup cell that checks the API key, the Qdrant connection and the data files, and tells you how to fix whatever is missing. **`w2_00` and `w2_01` are the exception: they embed nothing at runtime.** Every number in them was measured once, offline, by `tools/run_grid.py` and read back from `data/grid/`, so they open instantly, work without a key and without a network, and cannot disagree with each other.

### Rebuilding the measured tables

Only needed when the corpus, the question sets or the configuration grid change. From `notebooks/`:

```bash
uv run python -m tools.run_grid --stage all
uv run python -m tools.run_extras
uv run python -m tools.make_metric_animations
```

The grid takes about three to four hours the first time and is free afterwards, because every vector lands in `embedding_cache/`. Check the schema first with `uv run python -m tools.run_grid --stage 1 --sections 120`, which finishes in a couple of minutes.

### Collecting scores from a session

`w2_00` submits one GitHub issue per score to the public repository named in `ragkit/submit.py` (`SCORES_REPO`). Participants need a GitHub account and nothing else: no write access is granted, and no two submissions can conflict. Each notebook draws an anonymous handle once and keeps it in the gitignored `.playground/`, which is what pairs a participant's two attempts without identifying them.

As instructor, confirm a session by applying a `session-YYYY-MM-DD` label to its issues. Unlabelled issues are ignored, so a score submitted after the workshop cannot change a past session. The results view at the bottom of `w2_00` reads the labelled issues back and plots the two rounds against the best known configuration.

## Notebooks

| Notebook | Topic |
|---|---|
| `w2_00_chunking_playground` | The five-minute opening and closing exercise: tune a chunking and embedding configuration against a precomputed score table, submit the score anonymously, and see how the room did |
| `w2_01_chunking_and_retrieval` | Preprocessing, chunking strategies, text embeddings, retrieval metrics, model recommendation |
| `w2_02_embedding_models` | Image and text embeddings: image-to-image, text-to-image, image-to-text retrieval |
| `w2_03_real_world_datentypen` | IT-Grundschutz PDF with Docling: structure-aware chunking, Qdrant, RAG answers |
| `w2_04_ocr_docling_vlm_comparison` | OCR: Docling OCR vs Docling VLM vs external VLM |
| `w3_01_intro_end_to_end` | End-to-end RAG and RAGAS introduction |
| `w3_02_ingestion` | PDF → Docling → chunks → embeddings → Qdrant |
| `w3_03_retrieval_evaluation` | Context precision and recall, top-k experiment |
| `w3_04_generation_evaluation` | Answer correctness and faithfulness, prompt experiment |

## Data

| Folder | Content | Licence |
|---|---|---|
| `data/` | BSI IT-Grundschutz-Kompendium 2023 (PDF, Docling JSON, Markdown) and three human question sets with gold passages (GSKI). `data/grid/` holds the measured result tables `w2_00` and `w2_01` read | BSI publications, free to use with source; question sets from the GSKI pilot project |
| `raw_data/birds/` | 26 European bird species: photographs from Wikimedia Commons (CC0, CC BY, CC BY-SA; see `attribution.csv`) and the German Wikipedia article per species (CC BY-SA 4.0) | as listed per file |
| `raw_data/*.pdf` | BSI Standard 200-1 and a table-heavy paper for the OCR notebook | BSI; publisher terms |
| `processed/` | Pre-computed Docling outputs and chunk files so notebooks run without the slow conversion steps | derived |

The bird set was fetched once with `uv run python -m ragkit.crawl` (see that module for how such data can be collected). Participants never crawl.

## The `ragkit` package

Helpers that more than one notebook needs live in `ragkit/`; the notebooks import them instead of redefining them. No install step: marimo puts the notebook directory on `sys.path`.

| Module | Holds |
|---|---|
| `config.py` | Paths, model names, chunking parameters, Qdrant settings, and `setup()`, which validates key, Qdrant and data files and returns the environment |
| `theme.py` | HPI palette, grey-first matplotlib style, notebook header; `theme.css` and `head.html` are passed to `marimo.App` |
| `embed.py` | `embed()` with the model-specific backends, retries, batching, on-disk vector cache, image encoding |
| `chunk.py` | Text normalisation (German umlaut repair) and every chunking strategy |
| `search.py` | Qdrant access, cosine similarity, the rank-1 to rank-2 gap, entropy, MRR, nDCG (binary and graded), answer generation from retrieved context |
| `submit.py` | Anonymous score submission for `w2_00`: handles, the issue payload, reading a confirmed session back |
| `viz.py` | Kernel density estimate and figure encoding |
| `crawl.py` | Fetches the bird dataset from Wikimedia Commons and Wikipedia |

Embeddings are cached under `embedding_cache/`. Force recomputation with `ragkit.embed.RECREATE = True`.

Tests and lint:

```bash
uv run pytest -q
uv run ruff check .
```

| Script in `tools/` | Purpose |
|---|---|
| `run_grid.py` | Measures every chunking and embedding configuration over the full corpus and writes `data/grid/*.parquet` |
| `run_extras.py` | The few measured facts a figure needs but a table cannot hold: the header projection, the token-limit error, the score densities, the Matryoshka curve |
| `make_metric_animations.py` | Renders the three metric GIFs in `public/img/` |
| `convert_nb.py` | Converts a Jupyter notebook to a themed marimo notebook with all code hidden and reports what still needs manual attention |

## Troubleshooting

`uv: command not found`: install uv, then restart the terminal.

Qdrant does not start: confirm Docker is running and port 6333 is free (`docker ps -a --filter name=qdrant`).

RAGAS `InstructorRetryException` or `max_tokens` errors: reasoning models such as `gpt-oss-120b` need a higher `max_tokens` in `llm_factory()`; notebooks w3_03 and w3_04 already set 8192 to 65536.

Embedding batch errors (HTTP 413): `minilm-embedding` accepts at most 32 texts per batch; constraints per model live in the `_EMBED_MODELS` registry in `ragkit/config.py`.
