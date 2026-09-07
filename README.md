# RAG workshop materials

Teaching material of the HPI AI Service Centre (AISC) for a workshop series on retrieval-augmented generation (RAG). All material is in [`notebooks/`](notebooks/README.md): marimo notebooks that show explanations, figures and interactive controls first and reveal the code on demand.

| Workshop | Notebooks | Content |
|---|---|---|
| RAG II | `w2_01` to `w2_04` | chunking strategies and retrieval metrics, image and text embeddings, real-world document types with Docling, OCR compared with vision-language models |
| RAG III | `w3_01` to `w3_04` | evaluation with RAGAS: ingestion, retrieval metrics, generation metrics |

The setup (Python environment with `uv`, Qdrant via `docker compose up -d`, API key) is described in [notebooks/README.md](notebooks/README.md). The notebooks talk to the AISC inference API; an API key is required.

## Data and licences

- BSI IT-Grundschutz-Kompendium 2023 and BSI Standard 200-1 (Bundesamt für Sicherheit in der Informationstechnik), with question sets from the GSKI pilot project, under `notebooks/data/` and `notebooks/raw_data/`.
- 104 bird photographs from Wikimedia Commons under CC0, CC BY and CC BY-SA licences and German Wikipedia extracts (CC BY-SA 4.0), under `notebooks/raw_data/birds/` with an attribution list per file.
- One table-heavy paper for the OCR notebook under `notebooks/raw_data/`.

## About the repository name

The repository was created as `workshop-ragV2` while the earlier `workshop-rag` (now archived as `workshop-rag-legacy`) was still in use. It is a single workshop series, not a second part.

## Licence

See [LICENSE](LICENSE).
