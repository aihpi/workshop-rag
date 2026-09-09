"""Gold-set loading and relevance labels for retrieval evaluation.

The GSKI gold sets give a question, an answer and a Fundstelle, the passage of
the Kompendium that answers it. Retrieval metrics need that passage as a set of
section keys, and every chunk needs to know which sections it covers. This
module does both mappings; the metric primitives live in `search`.
"""

import csv
import io
import re
from bisect import bisect_right
from itertools import accumulate
from typing import Any

from rapidfuzz import fuzz, process

from .chunk import REQ_ID_RE, normalize_text
from .config import _DATASETS, DATA_DIR
from .search import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

FUNDSTELLE_COL = 'Fundstellen im IT-Grundschutz-Kompendium 2023'
FUZZY_CUTOFF = 99
_WS_RE = re.compile(r'\s+')
_QUOTE_RE = re.compile('[„“”‚‘’«»"\']')


# ---------------------------------------------------------------------------
# Gold set
# ---------------------------------------------------------------------------

def load_gold(dataset: str) -> list[dict[str, str]]:
    """Rows of a gold CSV named in `config._DATASETS` as question / answer / fundstelle."""
    filename, sep = _DATASETS[dataset]
    raw = (DATA_DIR / filename).read_bytes()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('cp1252')
    rows = csv.DictReader(io.StringIO(text, newline=''), delimiter=sep)
    return [{'question': r['Frage'], 'answer': r['Antwort'], 'fundstelle': r[FUNDSTELLE_COL]}
            for r in rows]


def section_key(section: dict[str, Any]) -> str:
    """Stable label of a section: its requirement ID, else '<baustein or front>::<title>'."""
    return section['req_id'] or f"{section['baustein_id'] or 'front'}::{section['title']}"


def _flat(text: str) -> str:
    """Lower-cased text with whitespace runs collapsed and quotation marks unified.

    Those two are what differs between a passage pasted from the PDF and the
    Docling export („…“ against '…', single against double line breaks).
    ponytail: hyphenation and dash variants are not touched.
    """
    return _QUOTE_RE.sub("'", _WS_RE.sub(' ', text)).lower()


def section_texts(sections: list[dict[str, Any]]) -> list[str]:
    """Flattened section texts for `gold_targets`; compute once per set of sections."""
    return [_flat(s['text']) for s in sections]


def _passage_needles(fundstelle: str) -> list[str]:
    """The first 200 characters of the passage, and of the passage without its first line.

    Gold Fundstellen often open with a citation line ('OPS.2.3, Kap. 1.1') that
    is not corpus text and would cost the fuzzy score five points.
    """
    text = normalize_text(fundstelle)
    return [_flat(part)[:200] for part in [text] + text.split('\n', 1)[1:]]


def gold_targets(fundstelle: str, sections: list[dict[str, Any]],
                 texts: list[str] | None = None) -> set[str]:
    """Section keys a Fundstelle points at; empty when nothing in `sections` resolves.

    Requirement IDs in the Fundstelle that exist in `sections` win. Otherwise
    the passage start (see `_passage_needles`) is matched against every section
    text with rapidfuzz partial_ratio and the best section is accepted only at
    >= FUZZY_CUTOFF. Pass `texts=section_texts(sections)` when labelling many
    rows so the flattening happens once.
    """
    ids = {f'{baustein}.A{n}' for baustein, n in REQ_ID_RE.findall(fundstelle)}
    known = ids & {s['req_id'] for s in sections}
    if known:
        return known
    if texts is None:
        texts = section_texts(sections)
    hits = [process.extractOne(needle, texts, scorer=fuzz.partial_ratio,
                               score_cutoff=FUZZY_CUTOFF) for needle in _passage_needles(fundstelle)]
    hits = [h for h in hits if h]
    if not hits:
        return set()
    best = max(hits, key=lambda h: h[1])
    return {section_key(sections[best[2]])}


# ---------------------------------------------------------------------------
# Chunk labels
# ---------------------------------------------------------------------------

def attach_section_keys(chunks: list[str], sections: list[dict[str, Any]],
                        probe: int = 80) -> list[dict[str, Any]]:
    """Chunk records carrying the keys of every section a chunk overlaps.

    Chunks must be slices of `sections_to_text(sections)` in document order,
    verbatim up to whitespace (what every chunker in `chunk` produces). The
    first and last `probe` characters of each chunk are located in the joined
    text with whitespace collapsed and mapped to section offsets by bisection,
    so a fixed-size chunk spanning several sections gets all of their keys.
    ponytail: a duplicate removed by `dedupe` keeps the keys of its first
    occurrence only, and `probe` must not exceed the chunker's overlap.
    """
    texts = [_WS_RE.sub(' ', s['text']) for s in sections]
    joined = ' '.join(texts)
    starts = [0] + list(accumulate(len(t) + 1 for t in texts))[:-1]
    keys = [section_key(s) for s in sections]
    records, cursor = [], 0
    for chunk in chunks:
        flat = _WS_RE.sub(' ', chunk)
        head = joined.find(flat[:probe], cursor)
        tail = joined.find(flat[-probe:], max(head, 0))
        if min(head, tail) < 0:
            raise ValueError(f'chunk is not a slice of the section text: {flat[:probe]!r}')
        first = bisect_right(starts, head) - 1
        last = bisect_right(starts, tail + len(flat[-probe:]) - 1) - 1
        records.append({'text': chunk, 'section_keys': keys[first:last + 1]})
        cursor = head
    return records


def relevant_chunks(chunk_records: list[dict[str, Any]], target_keys: set[str]) -> set[int]:
    """Indices of the chunk records whose `section_keys` overlap `target_keys`."""
    return {i for i, rec in enumerate(chunk_records)
            if target_keys.intersection(rec['section_keys'])}


# ---------------------------------------------------------------------------
# Metrics for one ranking
# ---------------------------------------------------------------------------

def evaluate_ranking(ranked: list, relevant: set, k: int = 5) -> dict[str, Any]:
    """MRR, nDCG@k, recall@k, precision@k and the rank of the first hit (None if none)."""
    first_hit = next((rank for rank, doc in enumerate(ranked, 1) if doc in relevant), None)
    return {
        'mrr': reciprocal_rank(ranked, relevant),
        'ndcg': ndcg_at_k(ranked, relevant, k),
        'recall': recall_at_k(ranked, relevant, k),
        'precision': precision_at_k(ranked, relevant, k),
        'first_hit_rank': first_hit,
    }
