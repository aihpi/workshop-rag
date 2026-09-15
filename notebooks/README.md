# Workshop guide

Follow this page from top to bottom. It takes you from an empty machine to the last notebook, in the order the two workshops use them.

The notebooks are [marimo](https://marimo.io) apps. You see explanations, figures and controls, and you can open the code behind any cell when you want to.

| Workshop | Notebooks | Topic |
|---|---|---|
| RAG II | `w2_00` to `w2_04` | chunking, embeddings, retrieval, real-world document types, OCR |
| RAG III | `w3_01` to `w3_04` | evaluation with RAGAS: ingestion, retrieval metrics, generation metrics |

Each step below ends with a way to check it worked. If a check fails, fix it before moving on; every later step assumes the earlier ones.

---

## Step 1: Setup

This guide assumes you have worked through [`workshop-getting-started`](https://github.com/aihpi/workshop-getting-started), so `uv`, Docker and `git` are installed and working. Your API key comes later, on the day.

Clone the repository and install the environment:

```bash
git clone https://github.com/aihpi/workshop-rag.git
```

```bash
cd workshop-rag/notebooks
```

```bash
uv sync
```

The clone carries about 100 MB of data (a PDF corpus, bird photographs, pre-computed results), so give it a minute. `uv sync` reads `pyproject.toml` and builds an isolated environment in `.venv/`; it does not touch any other Python on your machine.

**Check it worked:**

```bash
uv run marimo --version
```

A version number means step 1 is done.

---

## Step 2: your first notebook

This is the opening exercise of RAG II. Run it from `notebooks/`:

```bash
uv run marimo run w2_00_chunking_playground.py
```

Your browser opens the playground. **This notebook needs no API key, no Docker and no network**, so you can do this step even if the rest of your setup is not finished yet.

A vector store does not retrieve documents, it retrieves *chunks*: the pieces of text that were cut out, embedded and indexed. The playground lets you cut a 2.5 million character corpus in different ways and see how often the passage that answers a question lands in the top five hits.

- Pick a **strategy**, a **model**, a **size**, an **overlap**, and whether to prepend the section title.
- Press **Evaluate**. Your score is Recall@5, the share of questions whose answer passage was retrieved.
- You have **eight evaluations** per round, because in real life every experiment costs time.

The controls are chained: each one offers only the values the ones before it leave available. So every configuration you can build has already been measured, and you never waste an evaluation on a combination that has no answer. Changing the strategy changes which sizes are on offer, which is the point.

You run this notebook **twice**: once at the start of the workshop knowing nothing, and once at the end. Choose `round` accordingly.

**Check it worked:** press Evaluate once and a Recall@5 figure appears with the number of evaluations left.

> **`marimo run` against `marimo edit`.** `run` gives the app view with the code hidden, which is what you want while following along. `edit` opens the same notebook with every cell expandable, editable and re-runnable. Everything below works with either.

---

## Step 3: submitting your score

At the bottom of `w2_00`, tick the consent box and press **Submit**. The notebook sends your best configuration of the round.

You are identified only by a handle such as `teal-otter-41`, drawn once and kept in `.playground/state.json`, which is gitignored and never leaves your machine. That handle is all anyone sees. It is what lets your first and second round be compared without anyone knowing whose they are.

Submission needs a **GitHub account**, and nothing else. No access has to be granted to you. One of two things happens:

- If you have the [GitHub CLI](https://cli.github.com/) installed and logged in (`gh auth login`), the notebook submits directly.
- Otherwise it shows a link. Open it, and GitHub's new-issue form appears already filled in. Press the green button.

If neither works, the notebook prints the lines to paste manually.

**One submission per round.** Before it sends anything the notebook looks for an issue you already opened for this round, in your local state first and on GitHub second. If it finds one it creates nothing and offers **Overwrite my submission** instead, which replaces the body of that same issue. So a second press cannot put a second score in the room, and a better configuration found late still gets in. The pre-filled link is the exception: without `gh` nothing can be looked up, so open that form once.

**Check it worked:** the notebook confirms the submission, or hands you a link that opens a pre-filled issue.

---

## Step 4: add your API key

From here on the notebooks call the AISC inference API. You get the key at the workshop.

From `notebooks/`:

```bash
cp .env.example .env
```

Open `.env` in any editor and replace `your_openai_api_key_here` with the key you were given. Leave the other lines alone.

`.env` is gitignored. Never paste a key into a notebook cell and never commit one.

**Check it worked:** open the next notebook and the table at the top shows a tick next to the API key.

---

## Step 5: start Qdrant

Qdrant is the vector database the notebooks store embeddings in. It runs in Docker. From the **repository root**, one level above `notebooks/`:

```bash
docker compose up -d
```

It keeps its data in `qdrant_storage/`, so it survives a restart. Leave it running for the rest of the workshop.

**Check it worked:** open http://localhost:6333/dashboard and you see the Qdrant interface.

To stop it later: `docker compose down`.

---

## Step 6: RAG II, the remaining notebooks

Run each the same way, from `notebooks/`:

```bash
uv run marimo run w2_01_chunking_and_retrieval.py
```

Take them in order. Each starts with a table showing whether the key, Qdrant and its data files are in place, so you can see what is missing before anything fails.

| Notebook | What it teaches | Needs |
|---|---|---|
| `w2_01_chunking_and_retrieval` | Preprocessing, chunking strategies, text embeddings, retrieval metrics, and which model to choose. The long version of what you did by feel in `w2_00` | nothing; reads pre-computed tables |
| `w2_02_embedding_models` | Image and text embeddings over 26 bird species: image-to-image retrieval, text search over the articles, and the shared multimodal space | key |
| `w2_03_real_world_datentypen` | A real PDF through Docling: structure-aware chunking, storing it in Qdrant, answering questions from it | key, Qdrant |
| `w2_04_ocr_docling_vlm_comparison` | OCR compared three ways: Docling OCR, Docling's vision model, an external vision model | key, Qdrant |

Two notes on speed. `w2_03` downloads Docling's layout models the first time it converts a PDF, which takes a few minutes and only happens once. `w2_04` ships every result pre-computed and loads it instantly; set `RERUN_OCR = True` in its first section only if you want to watch it work for real.

---

## Step 7: RAG III

Same command, `w3_01` first:

```bash
uv run marimo run w3_01_intro_end_to_end.py
```

**Order matters here.** `w3_02` is what fills the Qdrant collection that `w3_03` and `w3_04` then query. Run it before either of them, or they will find nothing to evaluate.

| Notebook | What it teaches | Needs |
|---|---|---|
| `w3_01_intro_end_to_end` | A whole RAG pipeline end to end, and what RAGAS measures | key |
| `w3_02_ingestion` | PDF to Docling to chunks to embeddings to Qdrant. **Run this before w3_03 and w3_04** | key, Qdrant |
| `w3_03_retrieval_evaluation` | Context precision and recall, and what changing top-k does | key, Qdrant, w3_02 |
| `w3_04_generation_evaluation` | Answer correctness and faithfulness, and what changing the prompt does | key, Qdrant, w3_02 |

---

## Troubleshooting

**The table at the top of a notebook shows a cross.** It names the fix underneath: a missing key points you at step 4, an unreachable Qdrant at step 5, a missing file at your clone. Fix it and re-run the cell.

**`uv: command not found`.** Open a new terminal, so it picks up the command. If it is still missing, go back to [`workshop-getting-started`](https://github.com/aihpi/workshop-getting-started).

**Qdrant does not start.** Check Docker Desktop is running, then that nothing else holds port 6333:

```bash
docker ps -a --filter name=qdrant
```

**`w3_03` or `w3_04` find nothing.** Run `w3_02` first; it creates the collection they read.

**No GitHub account at submission time.** The playground prints your score as plain text at the end. Keep it and submit later, or give it to the instructor.

**RAGAS `InstructorRetryException` or `max_tokens` errors.** Reasoning models such as `gpt-oss-120b` need a higher `max_tokens` in `llm_factory()`; `w3_03` and `w3_04` already raise it from 8192 to 65536.

**Embedding batch errors (HTTP 413).** `minilm-embedding` takes at most 32 texts per batch. The per-model limits live in the `_EMBED_MODELS` registry in `ragkit/config.py`.

---

## Data

| Folder | Content | Licence |
|---|---|---|
| `data/` | BSI IT-Grundschutz-Kompendium 2023 (PDF, Docling JSON, Markdown) and three human question sets with gold passages (GSKI). `data/grid/` holds the measured tables `w2_00` and `w2_01` read | BSI publications, free to use with source; question sets from the GSKI pilot project |
| `raw_data/birds/` | 26 European bird species: photographs from Wikimedia Commons (CC0, CC BY, CC BY-SA; see `attribution.csv`) and the German Wikipedia article per species (CC BY-SA 4.0) | as listed per file |
| `raw_data/*.pdf` | BSI Standard 200-1 and two papers for the OCR notebook | BSI; publisher terms |
| `processed/` | Pre-computed Docling outputs and chunk files, so the notebooks run without the slow conversion steps | derived |

The bird set was fetched once with `uv run python -m ragkit.crawl`. Participants never crawl.

---

## For instructors

`w2_00` submits one GitHub issue per score to the public repository named by `SCORES_REPO` in `ragkit/submit.py`. Participants need an account and nothing more: no write access is granted, and no two submissions can collide.

Confirm a session by applying a `session-YYYY-MM-DD` label to its issues. Unlabelled issues are ignored, so a score submitted after the workshop cannot change a past session.

The labelling and the room plot live in the scores repository, not here. Clone [`aihpi/workshop-rag-scores`](https://github.com/aihpi/workshop-rag-scores) and run its instructor notebook:

```bash
uv run marimo run instructor.py
```

It lists every submission, flags anything that does not match the template, creates and applies the session label, and plots the two rounds against the best known configuration. Labelling needs `gh` logged in as someone with triage rights on that repository; reading needs nothing. Project it at the end of the session: participants no longer see the plot in their own copy of `w2_00`.

---

## For maintainers

Helpers that more than one notebook needs live in `ragkit/`. There is no install step: marimo puts the notebook directory on `sys.path`.

| Module | Holds |
|---|---|
| `config.py` | Paths, model names, chunking parameters, Qdrant settings, and `setup()`, which validates key, Qdrant and data files and returns the environment |
| `theme.py` | HPI palette, grey-first matplotlib style, notebook header; `theme.css` and `head.html` are passed to `marimo.App` |
| `embed.py` | `embed()` with the model-specific backends, retries, batching, on-disk vector cache, image encoding |
| `chunk.py` | Text normalisation (German umlaut repair) and every chunking strategy |
| `search.py` | Qdrant access, cosine similarity, the rank-1 to rank-2 gap, entropy, MRR, nDCG (binary and graded), answer generation from retrieved context |
| `submit.py` | Anonymous score submission for `w2_00`: handles, the issue payload, reading a confirmed session back |
| `viz.py` | Kernel density estimate, image grids, figure encoding |
| `crawl.py` | Fetches the bird dataset from Wikimedia Commons and Wikipedia |

Embeddings are cached under `embedding_cache/`. Force recomputation with `ragkit.embed.RECREATE = True`.

```bash
uv run pytest -q
```

```bash
uv run ruff check .
```

| Script in `tools/` | Purpose |
|---|---|
| `run_grid.py` | Measures chunking and embedding configurations over the full corpus and writes `data/grid/*.parquet` |
| `run_extras.py` | The few measured facts a figure needs but a table cannot hold: the header projection, the token-limit error, the score densities, the Matryoshka curve |
| `make_metric_animations.py` | Renders the three metric GIFs in `public/img/` |
| `convert_nb.py`, `preconvert.py` | Convert a Jupyter notebook to a themed marimo notebook and report what still needs attention |

### Rebuilding the measured tables

Only needed when the corpus, the question sets or the grid change. From `notebooks/`:

```bash
uv run python -m tools.run_grid --stage all
```

```bash
uv run python -m tools.run_extras
```

```bash
uv run python -m tools.make_metric_animations
```

Check the schema first with `uv run python -m tools.run_grid --stage 1 --sections 120`, which finishes in a couple of minutes.

`w2_00` offers exactly the configurations the tables hold, so widening the grid widens what participants can try. `--stage dense` measures every combination its controls can express:

```bash
uv run python -m tools.run_grid --stage dense --resume
```

The tables are rewritten after every single configuration, so the run can be stopped at any point and `--resume` picks it up, measuring only what is missing. A partly finished run leaves the playground correct, just with fewer options. Add `--dry-run` to list what is planned and what is already done.
