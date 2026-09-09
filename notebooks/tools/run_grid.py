"""Search the chunking and embedding grid that w2_00 and w2_01 read back.

Run once, offline, and commit the tables in `data/grid/`. The notebooks never
embed anything at runtime: they look configurations up in `grid_scores.parquet`
and show retrieved chunks from `grid_hits.parquet`.

    uv run python -m tools.run_grid --stage 1 --sections 300   # schema smoke test
    uv run python -m tools.run_grid --stage all                # the real run, ~3.5 h

Run it from `notebooks/`, as a module, so that `ragkit` is importable.

Every configuration embeds the whole corpus once (about 2.45 M characters, some
eight minutes on octen), so the cache in `embedding_cache/` is what makes a
re-run free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ragkit import chunk as C
from ragkit import evaluate as E
from ragkit.config import DATA_DIR
from ragkit.embed import CACHE_DIR, cached_embed, embed
from ragkit.search import l2_normalise

MD_PATH = DATA_DIR / 'IT_Grundschutz_Kompendium_Edition2023.md'
OUT_DIR = DATA_DIR / 'grid'
GOLD_SETS = ('123_einfach', '43_komplex')
DIFFICULTY = {'123_einfach': 'easy', '43_komplex': 'complex'}
MODELS = {
    'octen': {'model': 'openai/octen-embedding-8b', 'batch_size': 64, 'max_chars': None},
    'miniLM': {'model': 'openai/minilm-embedding', 'batch_size': 32, 'max_chars': 350},
}
K = 5
TOP_N_HITS = 10  # ranks kept per question for the notebooks' inspector


@dataclass(frozen=True)
class Config:
    strategy: str            # paragraph | words | chars | section
    size: int | None = None  # words or characters; max chunk for section
    overlap: int = 0
    prepend_title: bool = False
    model: str = 'octen'

    @property
    def chunking_id(self) -> str:
        parts = [self.strategy]
        if self.size:
            parts.append(str(self.size))
        if self.overlap:
            parts.append(f'ov{self.overlap}')
        if self.prepend_title:
            parts.append('title')
        return '_'.join(parts)

    @property
    def id(self) -> str:
        return f'{self.chunking_id}__{self.model}'


# --- the grid ---------------------------------------------------------------

#: the six configurations w2_01 walks through chapter by chapter; both models must cover them, so
#: the notebook's tables never have to say "not measured".
TEACHING = [
    Config('paragraph'),
    Config('words', 100),
    Config('chars', 300),
    Config('chars', 1200),
    Config('chars', 1200, overlap=200),
    Config('section', 1200),
]

STAGE1 = [
    Config('paragraph'),
    *[Config('words', w) for w in (60, 100, 200, 400)],
    *[Config('chars', c) for c in (300, 600, 900, 1200, 1600, 2000)],
    Config('chars', 1200, overlap=200),
    Config('section', 1200),
]


def _dedupe_configs(configs: list[Config]) -> list[Config]:
    seen, unique = set(), []
    for cfg in configs:
        if cfg.id not in seen:
            seen.add(cfg.id)
            unique.append(cfg)
    return unique


def stage2(best: list[Config]) -> list[Config]:
    """Refinements around the stage-1 winners, plus miniLM over the whole teaching set."""
    out: list[Config] = []
    for cfg in best[:2]:
        if cfg.strategy in ('chars', 'section') and cfg.size:
            out += [replace(cfg, overlap=int(cfg.size * f)) for f in (0.1, 0.2)]
    out += [replace(cfg, prepend_title=True) for cfg in best[:3]]
    out += [Config('section', s) for s in (800, 1600)]
    out += [replace(cfg, model='miniLM') for cfg in TEACHING]
    return _dedupe_configs(out)


# --- corpus, gold and chunks ------------------------------------------------


def load_corpus(n_sections: int | None):
    sections = C.build_sections(C.strip_furniture(C.normalize_text(MD_PATH.read_text(encoding='utf-8'))))
    if n_sections:
        sections = sections[:n_sections]
    return sections, C.sections_to_text(sections)


def load_questions(sections, texts):
    """Gold rows that resolve to a section, plus the rows dropped and why."""
    rows, dropped = [], []
    for name in GOLD_SETS:
        for i, row in enumerate(E.load_gold(name)):
            targets = E.gold_targets(row['fundstelle'], sections, texts)
            record = {
                'qid': f'{name}_{i:03d}',
                'question': row['question'],
                'difficulty': DIFFICULTY[name],
                'gold_set': name,
                'fundstelle': row['fundstelle'][:300],
            }
            (rows if targets else dropped).append({**record, 'targets': sorted(targets)})
    return rows, dropped


def title_of(sections_by_key, keys):
    for key in keys:
        section = sections_by_key.get(key)
        if section and section.get('baustein_id'):
            return f"{section['baustein_id']} {section['baustein_title']}"
    return ''


def build_records(cfg: Config, sections, text, sections_by_key):
    """Chunk records as {'text', 'section_keys'}, deduplicated on text."""
    if cfg.strategy == 'section':
        records = []
        for s in sections:
            prefix = f"{s['baustein_id']} {s['baustein_title']}\n" if s['baustein_id'] and s['kind'] != 'baustein' else ''
            body = prefix + s['text']
            size = cfg.size or 1200
            parts = C.parser_aware_split(body, size, cfg.overlap) if len(body) > size else [body]
            records += [{'text': p, 'section_keys': [E.section_key(s)]} for p in parts]
    else:
        raw = {
            'paragraph': lambda: C.chunk_by_paragraph(text),
            'words': lambda: C.chunk_by_words(text, cfg.size),
            'chars': lambda: C.chunk_by_chars(text, cfg.size, cfg.overlap),
        }[cfg.strategy]()
        probe = min(80, cfg.overlap) if cfg.overlap else 80
        records = E.attach_section_keys(raw, sections, probe=probe)

    seen, unique = set(), []
    for rec in records:
        if rec['text'] not in seen:
            seen.add(rec['text'])
            unique.append(rec)
    if cfg.prepend_title and cfg.strategy != 'section':
        unique = [{**r, 'text': f"{title_of(sections_by_key, r['section_keys'])}\n{r['text']}".lstrip()}
                  for r in unique]
    return unique


# --- embedding and scoring --------------------------------------------------


def fingerprint(texts) -> str:
    return hashlib.md5('\n'.join(texts).encode('utf-8')).hexdigest()[:8]  # cache key, not security


def embed_cached(key, texts, model):
    t0 = time.perf_counter()
    fresh = not (CACHE_DIR / f'{key}.npy').exists()
    vectors = l2_normalise(cached_embed(key, lambda: embed(texts, **MODELS[model])))
    return vectors, (round(time.perf_counter() - t0, 1) if fresh else None)


def score_config(cfg, records, questions, query_vectors, chunk_vectors, embed_seconds):
    scores = query_vectors @ chunk_vectors.T
    per_q, hits = [], []
    for qi, q in enumerate(questions):
        relevant = E.relevant_chunks(records, set(q['targets']))
        order = np.argsort(-scores[qi])
        metrics = E.evaluate_ranking(order[:50].tolist(), relevant, K)
        top = scores[qi][order[:2]]
        metrics['gap'] = float(top[0] - top[1]) if len(top) > 1 else float('nan')
        metrics['difficulty'] = q['difficulty']
        per_q.append(metrics)
        for rank, idx in enumerate(order[:TOP_N_HITS], 1):
            hits.append({'config_id': cfg.id, 'qid': q['qid'], 'rank': rank, 'chunk_idx': int(idx),
                         'score': float(scores[qi][idx]), 'relevant': bool(idx in relevant),
                         'text': records[idx]['text']})

    lengths = [len(r['text']) for r in records]
    row = {'config_id': cfg.id, 'chunking_id': cfg.chunking_id, 'strategy': cfg.strategy,
           'size': cfg.size, 'overlap': cfg.overlap, 'prepend_title': cfg.prepend_title,
           'model': cfg.model, 'chunks': len(records),
           'median_chars': int(np.median(lengths)), 'p95_chars': int(np.percentile(lengths, 95)),
           'over_350_chars': float(np.mean([n > 350 for n in lengths])),
           'embed_seconds': embed_seconds}
    for label, subset in (('', per_q),
                          ('_easy', [m for m in per_q if m['difficulty'] == 'easy']),
                          ('_complex', [m for m in per_q if m['difficulty'] == 'complex'])):
        for name, key in (('MRR', 'mrr'), ('nDCG@5', 'ndcg'), ('Recall@5', 'recall'),
                          ('Precision@5', 'precision'), ('gap', 'gap')):
            row[name + label] = float(np.mean([m[key] for m in subset])) if subset else float('nan')
    return row, hits, lengths


# --- driver -----------------------------------------------------------------


def run(configs, sections, text, sections_by_key, questions, out_dir, state):
    scores, hits, lengths = state['scores'], state['hits'], state['lengths']
    records_cache, query_cache = {}, {}
    for i, cfg in enumerate(configs, 1):
        if cfg.id in {r['config_id'] for r in scores}:
            continue
        if cfg.chunking_id not in records_cache:
            records_cache[cfg.chunking_id] = build_records(cfg, sections, text, sections_by_key)
        records = records_cache[cfg.chunking_id]
        texts = [r['text'] for r in records]
        print(f'[{i}/{len(configs)}] {cfg.id}: {len(records)} chunks', flush=True)

        chunk_vectors, seconds = embed_cached(f'w201grid_{cfg.chunking_id}_{cfg.model}_{fingerprint(texts)}',
                                              texts, cfg.model)
        if cfg.model not in query_cache:
            qtexts = [q['question'] for q in questions]
            query_cache[cfg.model], _ = embed_cached(
                f'w201grid_queries_{cfg.model}_{fingerprint(qtexts)}', qtexts, cfg.model)

        row, config_hits, config_lengths = score_config(
            cfg, records, questions, query_cache[cfg.model], chunk_vectors, seconds)
        scores.append(row)
        hits += config_hits
        counts, edges = np.histogram(config_lengths, bins=40)
        lengths += [{'config_id': cfg.id, 'bin_left': float(a), 'bin_right': float(b), 'count': int(n)}
                    for a, b, n in zip(edges[:-1], edges[1:], counts)]
        print(f'    Recall@5 {row["Recall@5"]:.3f}  MRR {row["MRR"]:.3f}  nDCG@5 {row["nDCG@5"]:.3f}', flush=True)
        write(out_dir, state)
    return scores


def write(out_dir, state):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state['scores']).to_parquet(out_dir / 'grid_scores.parquet', index=False)
    pd.DataFrame(state['hits']).to_parquet(out_dir / 'grid_hits.parquet', index=False, compression='zstd')
    pd.DataFrame(state['lengths']).to_parquet(out_dir / 'grid_lengths.parquet', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['1', '2', 'all'], default='all')
    parser.add_argument('--sections', type=int, help='truncate the corpus, for a schema smoke test')
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    parser.add_argument('--dry-run', action='store_true', help='list the configurations and stop')
    args = parser.parse_args()

    sections, text = load_corpus(args.sections)
    sections_by_key = {E.section_key(s): s for s in sections}
    texts = E.section_texts(sections)
    questions, dropped = load_questions(sections, texts)
    print(f'{len(sections)} sections, {len(text):,} chars, {len(questions)} questions, '
          f'{len(dropped)} dropped', flush=True)

    if args.dry_run:
        for cfg in STAGE1:
            print(' ', cfg.id)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'grid_questions.parquet').parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame([{k: v for k, v in q.items() if k != 'targets'} for q in questions]).to_parquet(
        args.out / 'grid_questions.parquet', index=False)
    (args.out / 'dropped_questions.json').write_text(json.dumps(
        [{k: v for k, v in q.items() if k != 'targets'} for q in dropped], indent=1, ensure_ascii=False))

    state = {'scores': [], 'hits': [], 'lengths': []}
    if args.stage in ('1', 'all'):
        run(STAGE1, sections, text, sections_by_key, questions, args.out, state)
    if args.stage in ('2', 'all'):
        ranked = sorted(state['scores'], key=lambda r: -r['Recall@5'])
        best = [Config(r['strategy'], r['size'], r['overlap'], r['prepend_title'], r['model'])
                for r in ranked]
        run(stage2(best), sections, text, sections_by_key, questions, args.out, state)

    table = pd.DataFrame(state['scores']).sort_values('Recall@5', ascending=False)
    print(table[['config_id', 'chunks', 'Recall@5', 'MRR', 'nDCG@5', 'Recall@5_easy',
                 'Recall@5_complex']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
