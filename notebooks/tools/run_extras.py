"""Produce `data/grid/extras.json`, the small measured facts w2_01 shows but cannot compute.

The notebook must run without an API key, so anything that needs the embedding API is measured here,
once, and committed. Run it after `tools.run_grid`, from `notebooks/`:

    uv run python -m tools.run_extras

It reuses the vector cache the grid run filled, so it costs a few API calls, not another hour.
"""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ragkit import theme
from ragkit.embed import embed
from ragkit.search import cosine_similarity_matrix, l2_normalise
from ragkit.viz import kde_plot
from tools.run_grid import (
    MODELS,
    OUT_DIR,
    Config,
    build_records,
    embed_cached,
    fingerprint,
    load_corpus,
    load_questions,
)

IMG_DIR = OUT_DIR.parent.parent / 'public' / 'img'   # served by marimo as /public/img/
N_HEADERS = 9   # header-only chunks to project
N_BODIES = 3    # chunks with a body, for contrast
OTHER_SAMPLE = 4000  # non-relevant scores kept per configuration, enough for a density


def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    similarity = cosine_similarity_matrix(vectors)
    upper = similarity[np.triu_indices(len(vectors), k=1)]
    return float(upper.mean())


def project_2d(vectors: np.ndarray) -> np.ndarray:
    """Two principal components, by SVD. No new dependency for a scatter plot."""
    centred = vectors - vectors.mean(axis=0)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    return centred @ right[:2].T


def header_projection(sections, text):
    """Requirement headers of one Baustein against bodies from unrelated ones.

    The point of the figure is that header-only chunks from the same family are
    hard to tell apart, so the headers must come from one family: picking the
    first header-only chunks in the file gives front matter (Vorwort,
    Dankesworte), which is diverse and shows the opposite.
    """
    records = build_records(Config('paragraph'), sections, text, {})
    texts = [r['text'] for r in records]
    header_only = [t.strip() for t in texts if t.strip().startswith('## ') and '\n' not in t.strip()]
    headers = [t for t in header_only if re.match(r'## ORP\.4\.A\d+ ', t)][:N_HEADERS]

    seen, bodies = set(), []
    for record in records:
        body = record['text'].strip()
        family = (record['section_keys'] or [''])[0].split('.')[0]
        if len(body) > 600 and not body.startswith('## ') and family not in seen:
            seen.add(family)
            bodies.append(body)
        if len(bodies) == N_BODIES:
            break

    vectors, _ = embed_cached(f'w201extras_projection_octen_{fingerprint(headers + bodies)}',
                              headers + bodies, 'octen')
    coordinates = project_2d(vectors)
    def _label(raw, limit):
        return re.sub(r'\s+', ' ', raw).strip()[:limit]

    # Short labels: the header points land almost on top of each other, which is the whole message,
    # so long titles would just collide into mush.
    labels = [_label(h.removeprefix('## ').split(' ', 1)[0], 12) for h in headers] + \
             [f'{_label(b, 20)}…' for b in bodies]
    return {
        'labels': labels,
        'kind': ['header'] * len(headers) + ['body'] * len(bodies),
        'x': [round(float(v), 4) for v in coordinates[:, 0]],
        'y': [round(float(v), 4) for v in coordinates[:, 1]],
        'pairwise_cosine_headers': round(mean_pairwise_cosine(vectors[:len(headers)]), 4),
        'pairwise_cosine_distinct': round(mean_pairwise_cosine(vectors[len(headers):]), 4),
    }


def token_limit_error(text: str):
    """The verbatim API error for an input past the small model's limit, so the notebook can show it."""
    long_text = text[:40000]
    for model, chars in (('miniLM', len(long_text)), ('octen', len(long_text))):
        try:
            embed([long_text], **{**MODELS[model], 'max_chars': None})
        except Exception as exc:  # noqa: BLE001  the message is the artefact
            return {'model': MODELS[model]['model'], 'chars': chars,
                    'error': f'{type(exc).__name__}: {exc}'[:600]}
    return {'model': MODELS['miniLM']['model'], 'chars': len(long_text),
            'error': 'no error was raised; the API truncated silently'}


def score_distribution(cfg: Config, sections, text, questions):
    from ragkit.evaluate import relevant_chunks

    records = build_records(cfg, sections, text, {})
    texts = [r['text'] for r in records]
    chunk_vectors, _ = embed_cached(f'w201grid_{cfg.chunking_id}_{cfg.model}_{fingerprint(texts)}',
                                    texts, cfg.model)
    query_texts = [q['question'] for q in questions]
    query_vectors, _ = embed_cached(f'w201grid_queries_{cfg.model}_{fingerprint(query_texts)}',
                                    query_texts, cfg.model)
    scores = query_vectors @ chunk_vectors.T

    relevant_scores, other_scores = [], []
    rng = np.random.default_rng(0)
    for qi, q in enumerate(questions):
        relevant = sorted(relevant_chunks(records, set(q['targets'])))
        relevant_scores += scores[qi][relevant].tolist()
        mask = np.ones(scores.shape[1], dtype=bool)
        mask[relevant] = False
        other_scores += rng.choice(scores[qi][mask], size=min(24, mask.sum()), replace=False).tolist()
    return {
        'config_id': cfg.id,
        'relevant': [round(float(v), 4) for v in relevant_scores],
        'other': [round(float(v), 4) for v in rng.choice(other_scores,
                                                         size=min(OTHER_SAMPLE, len(other_scores)),
                                                         replace=False)],
    }


def matryoshka(cfg: Config, sections, text, questions, dims=(4096, 2048, 1024, 512, 256, 128, 64)):
    """Retrieval quality when only the first d dimensions of each vector are kept."""
    from ragkit.evaluate import evaluate_ranking, relevant_chunks

    records = build_records(cfg, sections, text, {})
    texts = [r['text'] for r in records]
    chunk_vectors, _ = embed_cached(f'w201grid_{cfg.chunking_id}_{cfg.model}_{fingerprint(texts)}',
                                    texts, cfg.model)
    query_texts = [q['question'] for q in questions]
    query_vectors, _ = embed_cached(f'w201grid_queries_{cfg.model}_{fingerprint(query_texts)}',
                                    query_texts, cfg.model)
    relevant = [relevant_chunks(records, set(q['targets'])) for q in questions]

    usable = [d for d in dims if d <= chunk_vectors.shape[1]]
    mrr, recall = [], []
    for d in usable:
        scores = l2_normalise(query_vectors[:, :d]) @ l2_normalise(chunk_vectors[:, :d]).T
        measured = [evaluate_ranking(np.argsort(-scores[qi])[:50].tolist(), relevant[qi], 5)
                    for qi in range(len(questions))]
        mrr.append(round(float(np.mean([m['mrr'] for m in measured])), 4))
        recall.append(round(float(np.mean([m['recall'] for m in measured])), 4))
    return {'config_id': cfg.id, 'dims': list(usable), 'mrr': mrr, 'recall': recall}


# --- the figures, rendered once and committed ---------------------------------
#
# These three need the embedding API, so the notebook cannot draw them: it runs offline. They are
# rendered here as PNGs under `public/img/` and the notebook embeds the files.

def render_projection(extras, path):
    data = extras['header_chunk_projection']
    fig, ax = plt.subplots(figsize=(6.0, 4.2), layout='constrained')
    for x, y, label, kind in zip(data['x'], data['y'], data['labels'], data['kind']):
        filled = kind == 'header'
        ax.scatter(x, y, s=34, facecolor=theme.INK if filled else 'white',
                   edgecolor=theme.INK, linewidth=0.9, zorder=3)
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(6, 3), fontsize=7.5,
                    color=theme.INK if filled else theme.GREY)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.18)  # room for the labels, which otherwise run off the right edge
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_density(extras, path):
    fig, ax = plt.subplots(figsize=(5.4, 2.6), layout='constrained')
    kde_plot(extras['score_distribution']['relevant'], extras['score_distribution']['other'], ax=ax)
    ax.set_xlabel('cosine similarity')
    ax.set_ylabel('density')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_matryoshka(extras, path):
    data = extras['matryoshka']
    fig, ax = plt.subplots(figsize=(4.6, 2.6), layout='constrained')
    ax.plot(data['dims'], data['mrr'], color=theme.INK, marker='o', markersize=4)
    ax.plot(data['dims'], data['recall'], color=theme.GREY, marker='s', markersize=4, linestyle='--')
    ax.annotate('MRR', (data['dims'][-1], data['mrr'][-1]), xytext=(8, 0), textcoords='offset points',
                fontsize=8.5, va='center', color=theme.INK)
    ax.annotate('Recall@5', (data['dims'][-1], data['recall'][-1]), xytext=(8, 0),
                textcoords='offset points', fontsize=8.5, va='center', color=theme.GREY)
    ax.set_xscale('log', base=2)
    ax.set_xticks(data['dims'], [str(d) for d in data['dims']], fontsize=7.5)
    ax.set_xlabel('kept dimensions')
    ax.set_ylim(0, 1)
    ax.margins(x=0.18)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_figures(extras):
    theme.apply_mpl()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for name, render in (('w201_header_projection.png', render_projection),
                         ('w201_score_density.png', render_density),
                         ('w201_matryoshka.png', render_matryoshka)):
        render(extras, IMG_DIR / name)
        print(f'wrote {IMG_DIR / name}', flush=True)


def main():
    sections, text = load_corpus(None)
    questions, dropped = load_questions(sections, [s['text'] for s in sections])

    scores = pd.read_parquet(OUT_DIR / 'grid_scores.parquet')
    best = scores.sort_values('Recall@5', ascending=False).iloc[0]
    winner = Config(best.strategy, None if pd.isna(best['size']) else int(best['size']),
                    int(best.overlap), bool(best.prepend_title), best.model)
    print(f'winner: {winner.id}  Recall@5 {best["Recall@5"]:.3f}', flush=True)

    extras = {
        'gold_standard': {k: (None if pd.isna(v) else v.item() if hasattr(v, 'item') else v)
                          for k, v in best.items()},
        'questions': {'rows': len(questions) + len(dropped), 'evaluated': len(questions),
                      'dropped': [{'question': d['question'], 'fundstelle': d['fundstelle'][:200]}
                                  for d in dropped]},
        'header_chunk_projection': header_projection(sections, text),
        'token_limit': token_limit_error(text),
        'score_distribution': score_distribution(winner, sections, text, questions),
        'matryoshka': matryoshka(winner, sections, text, questions),
    }
    (OUT_DIR / 'extras.json').write_text(json.dumps(extras, indent=1, ensure_ascii=False),
                                         encoding='utf-8')
    print(f'wrote {OUT_DIR / "extras.json"}', flush=True)
    render_figures(extras)


if __name__ == '__main__':
    main()
