"""Gold data, caches and judge calls shared by the w3 evaluation notebooks (w3_02, w3_03).

The notebooks always work on the 43 complex GSKI questions (C9 of the RAG III
plan), independent of `DATASET` in config.py. `tools/precompute_w3_02.py`
writes the cache once; the notebooks read it and call the judge live only when
a participant presses a button.

    gold = load_gold()                         # 43 dicts: question, reference, fundstelle, baustein
    ret = load_retrieval()                     # top-10 chunks per question (text, rank, score)
    run = await judge_once(q, ref, contexts)   # one judge run: verdicts, claims, both metrics
"""

import asyncio
import json
import os
import re

import pandas as pd

from .config import API_BASE_URL, DATA_DIR, QDRANT_HOST, QDRANT_PORT

# ---------------------------------------------------------------------------
# Paths and the standard configuration
# ---------------------------------------------------------------------------

GOLD_CSV = DATA_DIR / 'GSKI_Fragen-Antworten-Fundstellen_43_Komplex.csv'
GOLD_SEP = ';'
CHUNKING = 'json_structured_sections'
EMBED_MODEL = 'openai/octen-embedding-8b'
JUDGE_MODEL = 'gpt-oss-120b'
COLLECTION = f'grundschutz_chunks_{CHUNKING}__{EMBED_MODEL.split("/")[-1]}'
TOP_K_VALUES = [1, 3, 5, 10]

CACHE = DATA_DIR / 'cache' / 'w3_02'
RETRIEVAL_FILE = CACHE / 'retrieval_top10.json'    # {'questions': [{'chunks': [{rank, score, text, chunk_id, pages}]}]}
DEMO_FILE = CACHE / 'demo_questions.json'          # {'demos': [{index, runs: [run, ...]}], 'candidates': [...]}
TRANSLATIONS_FILE = CACHE / 'translations.json'    # {'en': {German: English}, 'de': {English: German}}
# Module A on the website of the KI-Servicezentrum (C27): lines and vectors come from the w3_01 cache.
KISZ_FILE = CACHE / 'kisz' / 'kisz_eval.json'      # {'questions': [{question, reference, line_ids, source_ids, runs, second}], 'demos': [...]}
KISZ_TRANSLATIONS_FILE = CACHE / 'kisz' / 'translations.json'

# The four configurations with cached TOP_K experiments (dataset 43_komplex).
CONFIGS = {
    'standard': ('json_structured_sections', 'octen-embedding-8b', 'gpt-oss-120b'),
    'minilm': ('json_structured_sections', 'minilm-embedding', 'gpt-oss-120b'),
    'markdown': ('markdown_headers', 'octen-embedding-8b', 'gpt-oss-120b'),
    'llama': ('json_structured_sections', 'octen-embedding-8b', 'llama-3-3-70b'),
}


def config_dir(name: str):
    chunking, embed, judge = CONFIGS[name]
    return DATA_DIR / 'cache' / f'43_komplex__{chunking}__{embed}__{judge}'


# ---------------------------------------------------------------------------
# Gold data
# ---------------------------------------------------------------------------

_PART_SPLIT = re.compile(r'"\s*,\s*\n\s*"')
_BAUSTEIN = re.compile(r'\b([A-Z]{2,4}(?:\.\d+)+)')


def clean_fundstelle(raw: str) -> str:
    """The passage column without CSV quote artifacts and without dropped requirements (ENTFALLEN)."""
    parts = [p.strip().strip('"').strip() for p in _PART_SPLIT.split(str(raw))]
    parts = [p for p in parts if p and 'ENTFALLEN' not in p and 'Diese Anforderung ist entfallen' not in p]
    # One line per paragraph in Markdown; a single line break inside a part becomes a hard break.
    return '\n\n'.join(re.sub(r'\s*\n\s*', '  \n', p) for p in parts)


def baustein(fundstelle: str) -> str | None:
    """The first module ID (e.g. 'OPS.2.2') named in the passage, or None."""
    m = _BAUSTEIN.search(fundstelle)
    return m.group(1) if m else None


def load_gold() -> list[dict]:
    df = pd.read_csv(GOLD_CSV, sep=GOLD_SEP)
    out = []
    for i, row in df.iterrows():
        fund = clean_fundstelle(row.iloc[2])
        out.append({'index': int(i), 'question': str(row['Frage']).strip(), 'reference': str(row['Antwort']).strip(),
                    'fundstelle': fund, 'baustein': baustein(fund) or baustein(str(row['Frage']))})
    return out


def words(text: str) -> int:
    return len(str(text).split())


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_retrieval() -> list[dict]:
    return _read(RETRIEVAL_FILE)['questions']


def load_demos() -> list[dict]:
    return _read(DEMO_FILE)['demos']


def load_translations() -> dict[str, dict[str, str]]:
    if not TRANSLATIONS_FILE.exists():
        return {'en': {}, 'de': {}}
    return _read(TRANSLATIONS_FILE)


def load_config(name: str) -> dict:
    """Cached scores of one configuration: {'top5': {metric: [43]}, 'topk': {'1': {metric: [43]}, ...}}."""
    d = config_dir(name)
    return {'top5': _read(d / 'retrieval_eval_top5.json'), 'topk': _read(d / 'retrieval_experiment_topk.json')}


def load_kisz() -> dict:
    """Module A: the KISZ questions with their top-5 lines, five judge runs and the second judge's decompositions."""
    return _read(KISZ_FILE)


def load_kisz_translations() -> dict[str, dict[str, str]]:
    if not KISZ_TRANSLATIONS_FILE.exists():
        return {'en': {}, 'de': {}}
    return _read(KISZ_TRANSLATIONS_FILE)


def cache_complete() -> bool:
    return all(p.exists() for p in (RETRIEVAL_FILE, DEMO_FILE, TRANSLATIONS_FILE, KISZ_FILE, KISZ_TRANSLATIONS_FILE)) and all(
        (config_dir(n) / f).exists() for n in CONFIGS
        for f in ('retrieval_eval_top5.json', 'retrieval_experiment_topk.json'))


# ---------------------------------------------------------------------------
# Qdrant check
# ---------------------------------------------------------------------------

def collection_size(timeout: float = 2.0) -> int | None:
    """Number of points in the standard collection, or None when Qdrant or the collection is missing."""
    try:
        from qdrant_client import QdrantClient
        n = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=timeout).count(COLLECTION).count
    except Exception:  # noqa: BLE001 - any failure means 'not available'
        return None
    return int(n) or None


def retrieve(questions: list[str], top_k: int = 10) -> list[list[dict]]:
    """Top-k chunks per question from the standard collection."""
    from qdrant_client import QdrantClient

    from .embed import embed

    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vecs = embed(questions, model=EMBED_MODEL, batch_size=64, max_chars=None)
    out = []
    for v in vecs:
        hits = qc.query_points(collection_name=COLLECTION, query=v.tolist(), limit=top_k, with_payload=True).points
        out.append([{'rank': r + 1, 'score': float(h.score), 'text': h.payload['text'],
                     'chunk_id': str(h.payload.get('chunk_id')), 'pages': h.payload.get('citation_hint')}
                    for r, h in enumerate(hits)])
    return out


# ---------------------------------------------------------------------------
# The judge (RAGAS prompts, called directly so the verdicts and claims stay visible)
# ---------------------------------------------------------------------------

def judge_llm(model: str = JUDGE_MODEL):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=API_BASE_URL)
    return llm_factory(model, client=client, max_tokens=16384)


def average_precision(verdicts: list[int]) -> float:
    """Context Precision of one question, the same formula as RAGAS."""
    hits, total = 0, 0.0
    for i, v in enumerate(verdicts):
        if v:
            hits += 1
            total += hits / (i + 1)
    return total / hits if hits else 0.0


def recall(claims: list[dict]) -> float:
    return sum(c['attributed'] for c in claims) / len(claims) if claims else 0.0


async def judge_chunks(llm, question: str, reference: str, contexts: list[str]) -> list[dict]:
    """One Context Precision verdict (1/0) with reason per chunk, in rank order."""
    from ragas.metrics.collections.context_precision.util import (
        ContextPrecisionInput,
        ContextPrecisionOutput,
        ContextPrecisionPrompt,
    )

    prompt = ContextPrecisionPrompt()

    async def one(ctx):
        r = await llm.agenerate(prompt.to_string(ContextPrecisionInput(question=question, context=ctx, answer=reference)),
                                ContextPrecisionOutput)
        return {'verdict': int(bool(r.verdict)), 'reason': r.reason}

    return list(await asyncio.gather(*(one(c) for c in contexts)))


async def judge_claims(llm, question: str, reference: str, contexts: list[str]) -> list[dict]:
    """The reference split into claims, each with attributed (1/0) and reason (Context Recall)."""
    from ragas.metrics.collections.context_recall.util import (
        ContextRecallInput,
        ContextRecallOutput,
        ContextRecallPrompt,
    )

    r = await llm.agenerate(
        ContextRecallPrompt().to_string(ContextRecallInput(question=question, context='\n'.join(contexts), answer=reference)),
        ContextRecallOutput)
    return [{'statement': c.statement, 'attributed': int(bool(c.attributed)), 'reason': c.reason}
            for c in r.classifications]


async def judge_once(question: str, reference: str, contexts: list[str], llm=None) -> dict:
    """One full judge run for one question: both metrics with all verdicts. Raises on any API error."""
    llm = llm or judge_llm()
    chunks, claims = await asyncio.gather(judge_chunks(llm, question, reference, contexts),
                                          judge_claims(llm, question, reference, contexts))
    verdicts = [c['verdict'] for c in chunks]
    return {'chunks': chunks, 'claims': claims, 'context_precision': average_precision(verdicts),
            'context_recall': recall(claims), 'n_claims': len(claims)}


def judge_prompt_text() -> tuple[str, str]:
    """The instructions of both RAGAS prompts, verbatim, for display."""
    from ragas.metrics.collections.context_precision.util import ContextPrecisionPrompt
    from ragas.metrics.collections.context_recall.util import ContextRecallPrompt

    return ContextPrecisionPrompt.instruction, ContextRecallPrompt.instruction
