"""Indexing, retrieval, similarity, ranking metrics and answer generation.

The Qdrant side of the workshop, plus the numbers used to judge whether a
retriever is doing its job.
"""

import hashlib
import os
import uuid

import numpy as np

from .config import (
    API_BASE_URL,
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    RAG_MODEL_NAME,
)
from .embed import embed

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

_client = None


def client(host: str = QDRANT_HOST, port: int = QDRANT_PORT):
    """Return the shared Qdrant client, created on first use."""
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        _client = QdrantClient(host=host, port=port)
    return _client


def make_point_id(doc_name: str, chunk_idx: int) -> str:
    """Create a deterministic UUID from document name + chunk index.

    Deterministic so that re-indexing overwrites the same points instead of
    piling up duplicates.
    """
    raw = f'{doc_name}_{chunk_idx}'
    hash_hex = hashlib.sha256(raw.encode()).hexdigest()
    return str(uuid.UUID(hash_hex[:32]))


def rag_search(query: str, top_k: int = 5, collection: str = COLLECTION_NAME,
               model: str | None = None):
    """Embed a query and return the top-k hits as plain dicts."""
    q_vec = np.asarray(embed(query, **({'model': model} if model else {}))[0]).tolist()
    response = client().query_points(
        collection_name=collection,
        query=q_vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []
    for h in response.points:
        payload = h.payload or {}
        results.append({
            'score': h.score,
            'chunk_id': payload.get('chunk_id'),
            'text': payload.get('text', ''),
            'source_file': payload.get('source_file'),
            'page_numbers': payload.get('page_numbers', []),
            'citation_hint': payload.get('citation_hint'),
        })
    return results


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between all rows."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-10)
    return normed @ normed.T


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

def entropy(scores, top_n=None, shift_min: bool = False,
            temperature: float | None = None) -> float:
    """Shannon entropy (bits) of similarity scores normalised to probabilities.

    Low entropy means the retriever is decisive: a few chunks dominate. High
    entropy means the scores are flat and it cannot discriminate.

    Scores are sorted internally, so `top_n` means the top n regardless of the
    order they arrive in.

    Default (`shift_min=False`) is the formula taught in w2_01: p_i = s_i / sum(s),
    with non-positive scores dropped. `shift_min=True` first subtracts the
    minimum score, which makes the result offset-invariant (only the shape of
    the score curve counts) and sends the lowest score to p=0. w2_02 uses this
    variant for the Matryoshka experiment, where absolute cosine levels differ
    between vector widths. `temperature` switches to a softmax over the scores
    (p_i = exp(s_i/T) / sum exp(s_j/T)); cosine scores differ by hundredths, so
    T around 0.05 is needed before the entropy separates peaked from flat
    curves. w2_01 teaches this variant.
    """
    if top_n is not None:
        scores = sorted(scores, reverse=True)[:top_n]
    arr = np.asarray(scores, dtype=float)
    if temperature is not None:
        z = (arr - arr.max()) / temperature
        p = np.exp(z) / np.exp(z).sum()
        return float(-np.sum(p * np.log2(p + 1e-12)))
    if shift_min:
        arr = arr - arr.min() + 1e-9
    else:
        arr = arr[arr > 0]
    p = arr / arr.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def rank_gap(scores) -> float:
    """Distance between the best and the second-best score.

    Discriminability without labels: a large gap means one chunk clearly won, a
    gap near zero means the order of the hit list is noise. It says nothing
    about whether the winner is the right chunk.
    """
    top = sorted(scores, reverse=True)[:2]
    return float(top[0] - top[1]) if len(top) == 2 else float('nan')


def ndcg_graded(gains: list[float], k: int = 5) -> float:
    """nDCG@k over relevance grades given in rank order, with gain 2**g - 1.

    `gains[i]` is the grade of the chunk at rank i+1, so 0 is irrelevant and a
    higher grade is more relevant. Binary grades reproduce `ndcg_at_k`.
    """
    def dcg(values):
        return sum((2.0 ** g - 1) / np.log2(i + 2) for i, g in enumerate(values[:k]))

    ideal = dcg(sorted(gains, reverse=True))
    return dcg(gains) / ideal if ideal > 0 else 0.0


def reciprocal_rank(ranked_docs: list, relevant: set) -> float:
    """Return 1/rank of the first relevant doc, or 0 if none found.

    Averaged over a query set, this is MRR.
    """
    for rank, doc in enumerate(ranked_docs, 1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_docs: list, relevant: set, k: int = 5) -> float:
    """Normalised Discounted Cumulative Gain at k.

    Rewards relevant hits near the top of the ranking, not just their presence.
    """
    dcg = 0.0
    for i, doc in enumerate(ranked_docs[:k]):
        if doc in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 1
    # Ideal DCG: all relevant docs at the top
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_docs: list, relevant: set, k: int = 5) -> float:
    """Share of the relevant docs that appear in the top k (0 when nothing is relevant)."""
    if not relevant:
        return 0.0
    return len(set(ranked_docs[:k]) & set(relevant)) / len(relevant)


def precision_at_k(ranked_docs: list, relevant: set, k: int = 5) -> float:
    """Share of the top k that is relevant."""
    return len(set(ranked_docs[:k]) & set(relevant)) / k


# ---------------------------------------------------------------------------
# Generation from retrieved context
# ---------------------------------------------------------------------------

def build_rag_context(hits, max_chars_per_chunk: int = 1400) -> str:
    """Assemble retrieved chunks into a prompt-sized, cited context block.

    max_chars_per_chunk keeps the prompt inside the LLM's practical context
    window; over-stuffing the context can actually degrade answer quality.
    """
    blocks = []
    for i, h in enumerate(hits, start=1):
        pages = h.get('page_numbers') or []
        cite = h.get('citation_hint') or '-'
        pages_text = ', '.join(map(str, pages)) if pages else '-'
        snippet = (h.get('text') or '')[:max_chars_per_chunk]
        blocks.append(
            f"[Source {i}] score={h['score']:.4f} | pages={pages_text} | cite={cite}\n{snippet}"
        )
    return '\n\n'.join(blocks)


def answer_with_llm(query: str, hits, model: str | None = None,
                    system_prompt: str | None = None) -> str:
    """Answer `query` from the retrieved `hits`, refusing to go beyond them."""
    from litellm import completion

    model = model or RAG_MODEL_NAME
    system_prompt = system_prompt or (
        'You are a RAG assistant for IT-Grundschutz. '
        'Answer only based on the provided context. '
        'If the information is missing, say so clearly. '
        'List the sources with page numbers at the end. '
        'Answer in the language of the question.'
    )

    resp = completion(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',
             'content': f'Question:\n{query}\n\nContext:\n{build_rag_context(hits)}'},
        ],
        api_base=API_BASE_URL,
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.2,
    )
    return resp.choices[0].message.content
