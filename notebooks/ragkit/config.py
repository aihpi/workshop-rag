"""Shared configuration for the workshop notebooks.

Loads environment variables from .env (if present), defines path constants,
model names, and infrastructure settings. Call `setup()` to create the
evaluation cache directory and print a configuration summary.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

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
# .env laden (bereits gesetzte Umgebungsvariablen haben Vorrang)
# ---------------------------------------------------------------------------

load_dotenv(ENV_PATH, override=False)


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
# Setup (explizit aufrufen, kein Import-Effekt)
# ---------------------------------------------------------------------------

@dataclass
class WorkshopEnv:
    """What every notebook needs before the first cell of real work."""

    api_key: str | None
    api_base: str
    qdrant_host: str
    qdrant_port: int
    qdrant_ok: bool
    missing_files: list[Path] = field(default_factory=list)

    @property
    def problems(self) -> list[str]:
        out = []
        if not self.api_key:
            out.append(f'OPENAI_API_KEY is missing: copy {ENV_PATH.parent / ".env.example"} to '
                       f'{ENV_PATH} and insert your key.')
        if not self.qdrant_ok:
            out.append(f'Qdrant does not answer on {self.qdrant_host}:{self.qdrant_port}: '
                       f'run `docker compose up -d` in {REPO_ROOT}.')
        for path in self.missing_files:
            out.append(f'Missing data file: {path}')
        return out

    @property
    def ok(self) -> bool:
        return not self.problems

    def openai(self):
        from .embed import client
        return client()

    def qdrant(self):
        from .search import client
        return client(self.qdrant_host, self.qdrant_port)

    def summary_md(self) -> str:
        """Markdown table for display in a notebook."""
        mark = lambda good: '✓' if good else '✗'
        rows = [
            ('API base', self.api_base, mark(bool(self.api_key))),
            ('API key', 'set' if self.api_key else 'missing', mark(bool(self.api_key))),
            ('Qdrant', f'{self.qdrant_host}:{self.qdrant_port}', mark(self.qdrant_ok)),
            ('Data files', 'all present' if not self.missing_files else
             ', '.join(p.name for p in self.missing_files), mark(not self.missing_files)),
        ]
        table = '| | | |\n|---|---|---|\n' + '\n'.join(f'| {a} | {b} | {c} |' for a, b, c in rows)
        fixes = ''.join(f'\n\n> {p}' for p in self.problems)
        return table + fixes


def _qdrant_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(host=host, port=port, timeout=timeout).get_collections()
        return True
    except Exception:  # noqa: BLE001 - any failure means 'not reachable'
        return False


def setup(*, require_qdrant: bool = True, required_files: tuple[Path, ...] = (),
          strict: bool = True) -> WorkshopEnv:
    """Validate credentials, Qdrant and data files; return the environment.

    Args:
        require_qdrant: Probe the Qdrant server (skip for notebooks without a vector store).
        required_files: Paths that must exist before the notebook can run.
        strict: Raise `RuntimeError` listing every problem instead of returning them.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    env = WorkshopEnv(
        api_key=os.getenv('OPENAI_API_KEY') or None,
        api_base=API_BASE_URL,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        qdrant_ok=_qdrant_reachable(QDRANT_HOST, QDRANT_PORT) if require_qdrant else True,
        missing_files=[Path(p) for p in required_files if not Path(p).exists()],
    )
    if strict and not env.ok:
        raise RuntimeError('Workshop setup incomplete:\n- ' + '\n- '.join(env.problems))
    return env
