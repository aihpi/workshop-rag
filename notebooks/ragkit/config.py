"""Shared configuration for the workshop notebooks.

Loads environment variables from .env (if present), defines path constants,
model names, and infrastructure settings. Call `setup()` to create the
evaluation cache directory and print a configuration summary.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

# ragkit lives in <repo>/notebooks/ragkit, so the workshop directory is one level up.
WORKSHOP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSHOP_DIR.parent
DATA_DIR = WORKSHOP_DIR / 'data'
ENV_PATH = WORKSHOP_DIR / '.env'

# Quelldokument für RAG
PDF_PATH = DATA_DIR / 'IT_Grundschutz_Kompendium_Edition2023.pdf'

# --- Ground-Truth-Datensatz ---
# Verfügbare Datensätze:
#   '40_einfach'  — 40 einfache Fragen (sep=',')
#   '43_komplex'  — 43 komplexe Fragen (sep=';')
#   '123_einfach' — 123 einfache Fragen (sep=';')
DATASET = '40_einfach'

_DATASETS = {
    '40_einfach':  ('GSKI_Fragen-Antworten-Fundstellen_40_Einfach.csv',  ','),
    '43_komplex':  ('GSKI_Fragen-Antworten-Fundstellen_43_Komplex.csv',  ';'),
    '123_einfach': ('GSKI_Fragen-Antworten-Fundstellen_123_Einfach.csv', ';'),
}

_csv_file, CSV_SEP = _DATASETS[DATASET]  # Dateiname + Trennzeichen aus dem gewählten Datensatz
CSV_PATH = DATA_DIR / _csv_file


# ---------------------------------------------------------------------------
# .env laden (einfacher Parser ohne externe Abhängigkeit)
# ---------------------------------------------------------------------------
# setdefault sorgt dafür, dass bereits gesetzte Umgebungsvariablen
# (z.B. aus der Shell) Vorrang haben.

if ENV_PATH.exists():
    for _line in ENV_PATH.read_text(encoding='utf-8').splitlines():
        _line = _line.strip()
        if not _line or _line.startswith('#') or '=' not in _line:
            continue
        _key, _value = _line.split('=', 1)
        os.environ.setdefault(_key.strip(), _value.strip())


# ---------------------------------------------------------------------------
# Modellkonfiguration
# ---------------------------------------------------------------------------
# Drei Modell-Rollen:
#   1. Embedding-Modell:  Text → Vektoren (für Qdrant)
#   2. RAG-Generierung:   Beantwortet Fragen basierend auf Kontexten
#   3. RAGAS-Evaluator:   Bewertet die Qualität des RAG-Systems
#
# LiteLLM benötigt das Provider-Präfix 'openai/' im Modellnamen,
# LangChain ChatOpenAI (für RAGAS) erwartet nur den Modellnamen ohne Präfix.

EMBED_MODEL_NAME = 'openai/octen-embedding-8b'       # Embedding model e.g. 'openai/octen-embedding-8b' or 'openai/minilm-embedding' 
RAG_MODEL_NAME = 'openai/gpt-oss-120b'               # RAG-Generierung via LiteLLM
EVALUATOR_MODEL_NAME = 'gpt-oss-120b'  # 'llama-3-3-70b' or 'openai/gpt-oss-120b' # RAGAS-Evaluator via LangChain
API_BASE_URL = os.getenv('OPENAI_API_BASE', 'https://api.aisc.hpi.de/')

# --- Embedding-Modell-Registry ---
# Jedes Modell hat spezifische Einschränkungen (z.B. Token-Limits).
# max_chars: Maximale Zeichenanzahl pro Text vor dem Embedding (None = kein Limit)
_EMBED_MODELS = {
    'openai/octen-embedding-8b': {'max_chars': None, 'batch_size': 64},
    'openai/minilm-embedding':   {'max_chars': 350,  'batch_size': 32},  # miniLM: 256-Token-Limit, max batch 32
}

EMBED_MAX_CHARS = _EMBED_MODELS[EMBED_MODEL_NAME]['max_chars']
EMBED_BATCH_SIZE = _EMBED_MODELS[EMBED_MODEL_NAME]['batch_size']
EMBED_SHORT = EMBED_MODEL_NAME.split('/')[-1]  # 'openai/octen-embedding-8b' → 'octen-embedding-8b'


# ---------------------------------------------------------------------------
# Qdrant Vektordatenbank
# ---------------------------------------------------------------------------

QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))


# ---------------------------------------------------------------------------
# Chunking-Parameter
# ---------------------------------------------------------------------------

MAX_CHUNK = 1200   # Maximale Zeichenanzahl pro Chunk
OVERLAP = 200      # Überlappung in Zeichen zwischen Chunks
TOP_K = 5          # Anzahl abzurufender Chunks pro Frage

# --- Chunking-Modus ---
# 'markdown_headers': Markdown-Export, Split an Überschriften (mit max_chunk + overlap)
# 'json_structured_sections': JSON-Export, Split an Heading-Labels (ohne max_chunk)
CHUNKING_MODE = 'json_structured_sections'

# Collection-Name wird aus dem Chunking-Modus abgeleitet
COLLECTION_NAME = f'grundschutz_chunks_{CHUNKING_MODE}__{EMBED_SHORT}'

# Cache-Verzeichnis für Evaluationsergebnisse (pro Datensatz + Chunking + Embedding + Evaluator)
EVALUATOR_SHORT = EVALUATOR_MODEL_NAME.split('/')[-1]
CACHE_DIR = DATA_DIR / 'cache' / f'{DATASET}__{CHUNKING_MODE}__{EMBED_SHORT}__{EVALUATOR_SHORT}'

# Figures-Verzeichnis für gespeicherte Plots
FIGURES_DIR = WORKSHOP_DIR / 'figures'


# ---------------------------------------------------------------------------
# Setup für die Workshop-3-Notebooks (explizit aufrufen, kein Import-Effekt)
# ---------------------------------------------------------------------------

def setup() -> None:
    """Create CACHE_DIR and print the configuration summary.

    Called by the Workshop 3 notebooks right after `from ragkit.config import *`.
    Kept out of import time so that Workshop 2 notebooks, which only need
    `ragkit.embed` or `ragkit.search`, neither print this banner nor create
    the evaluation cache directory.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print('=== Workshop-Konfiguration ===')
    print(f'  PDF:             {PDF_PATH.name} | exists: {PDF_PATH.exists()}')
    print(f'  Datensatz:       {DATASET} ({CSV_PATH.name}, sep="{CSV_SEP}") | exists: {CSV_PATH.exists()}')
    print(f'  Embedding:       {EMBED_MODEL_NAME} (max_chars={EMBED_MAX_CHARS})')
    print(f'  RAG-Modell:      {RAG_MODEL_NAME}')
    print(f'  Evaluator:       {EVALUATOR_MODEL_NAME}')
    print(f'  API Base URL:    {API_BASE_URL}')
    print(f'  API Key gesetzt: {bool(os.getenv("OPENAI_API_KEY"))}')
    print(f'  Qdrant:          {QDRANT_HOST}:{QDRANT_PORT}')
    print(f'  Chunking-Modus:  {CHUNKING_MODE}')
    print(f'  Collection:      {COLLECTION_NAME}')
    print(f'  Chunking:        MAX_CHUNK={MAX_CHUNK}, OVERLAP={OVERLAP}, TOP_K={TOP_K}')
    print('=' * 30)
