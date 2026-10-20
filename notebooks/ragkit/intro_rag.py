"""The mini RAG pipeline of w3_01 and its two RAGAS metrics.

The pipeline answers questions about the website of the KI-Servicezentrum:
every non-empty line of the page is a chunk, the TOP_K most similar chunks go
to the answer model, and RAGAS scores the result with Context Precision and
Answer Correctness. `tools/precompute_w3_01.py` runs it once for the prepared
questions and writes the cache; the notebook calls `aevaluate_one` live only
when a participant presses a button.
"""

import json
import os

import numpy as np

from .config import (
    API_BASE_URL,
    DATA_DIR,
    EMBED_MODEL_NAME,
    EVALUATOR_MODEL_NAME,
    RAG_MODEL_NAME,
)

URL = 'https://hpi.de/ki-servicezentrum/'
TOP_K = 3
CACHE = DATA_DIR / 'cache' / 'w3_01'
CHUNKS_FILE = CACHE / 'chunks.json'          # {'url', 'crawled', 'chunks': [...]}
VECTORS_FILE = CACHE / 'chunk_vectors.npy'   # one row per chunk, same order
RESULTS_FILE = CACHE / 'results.json'        # {'manual': [...], 'synthetic': [...]}
TRANSLATIONS_FILE = CACHE / 'translations_en.json'  # {German text: English text} for the en version

SYSTEM_PROMPT = (
    'Beantworte die Frage nur auf Basis des Kontexts. Antworte kurz und präzise. '
    'Wenn die Informationen im Kontext nicht ausreichen, um die Frage zu beantworten, '
    'gib das klar an. Verwende keine Informationen, die nicht im Kontext stehen.'
)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def crawl(url: str = URL) -> list[str]:
    """Convert the page with Docling and return its non-empty lines as chunks."""
    from docling.document_converter import DocumentConverter

    markdown = DocumentConverter().convert(url).document.export_to_markdown()
    return [line.strip() for line in markdown.split('\n') if line.strip()]


def load_corpus() -> tuple[list[str], np.ndarray]:
    """Cached chunks and their embeddings."""
    chunks = json.loads(CHUNKS_FILE.read_text(encoding='utf-8'))['chunks']
    return chunks, np.load(VECTORS_FILE)


def load_results() -> dict:
    return json.loads(RESULTS_FILE.read_text(encoding='utf-8'))


def load_translations() -> dict[str, str]:
    """English translations of the cached corpus texts; empty when the file is missing."""
    if not TRANSLATIONS_FILE.exists():
        return {}
    return json.loads(TRANSLATIONS_FILE.read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def retrieve(question: str, chunks: list[str], vectors: np.ndarray, top_k: int = TOP_K) -> list[int]:
    """Indices of the top_k chunks by cosine similarity to the question."""
    from .embed import embed

    q = np.asarray(embed([question])[0], dtype=np.float32)
    sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    return [int(i) for i in np.argsort(-sims)[:top_k]]


def generate(question: str, contexts: list[str]) -> str:
    """Answer from the retrieved chunks with the answer model."""
    from litellm import completion

    block = '\n\n'.join(f'[Quelle {i + 1}]\n{c}' for i, c in enumerate(contexts))
    resp = completion(
        model=RAG_MODEL_NAME,
        messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                  {'role': 'user', 'content': f'Frage: {question}\n\nKontext:\n{block}'}],
        api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.2, max_tokens=4000,
    )
    return (resp.choices[0].message.content or '').strip()


def _metrics():
    """Context Precision and Answer Correctness with the judge model of config.py."""
    from openai import AsyncOpenAI
    from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerCorrectness, ContextPrecision

    key = os.getenv('OPENAI_API_KEY')
    judge = llm_factory(EVALUATOR_MODEL_NAME, client=AsyncOpenAI(api_key=key, base_url=API_BASE_URL),
                        max_tokens=16384)
    emb = LiteLLMEmbeddings(model=EMBED_MODEL_NAME, api_key=key, api_base=API_BASE_URL,
                            encoding_format='float')
    return ContextPrecision(llm=judge), AnswerCorrectness(llm=judge, embeddings=emb)


async def score_many(rows: list[dict]) -> list[dict]:
    """Add 'context_precision' and 'answer_correctness' to rows with question, reference, contexts, answer."""
    cp, ac = _metrics()
    cp_res = await cp.abatch_score([{'user_input': r['question'], 'reference': r['reference'],
                                     'retrieved_contexts': r['contexts']} for r in rows])
    ac_res = await ac.abatch_score([{'user_input': r['question'], 'response': r['answer'],
                                     'reference': r['reference']} for r in rows])
    return [{**r, 'context_precision': float(c.value), 'answer_correctness': float(a.value)}
            for r, c, a in zip(rows, cp_res, ac_res)]


async def aevaluate_one(question: str, reference: str, chunks: list[str], vectors: np.ndarray) -> dict:
    """Retrieval, answer and both metrics for one question, from a running event loop (marimo cells); raises on any API error."""
    idx = retrieve(question, chunks, vectors)
    contexts = [chunks[i] for i in idx]
    row = {'question': question, 'reference': reference, 'chunk_ids': idx, 'contexts': contexts,
           'answer': generate(question, contexts)}
    return (await score_many([row]))[0]
