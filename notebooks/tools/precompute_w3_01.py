"""Build the cache of w3_01: website chunks, embeddings and RAGAS results.

Run once from `notebooks/`:

    uv run python -m tools.precompute_w3_01            # all stages, reusing what exists
    uv run python -m tools.precompute_w3_01 --recrawl  # crawl the website again first

Writes to data/cache/w3_01/: chunks.json (one chunk per non-empty line of the
page), chunk_vectors.npy (their embeddings) and results.json (per question the
top-3 chunks, the answer, Context Precision and Answer Correctness, and for the
manual questions the website lines that answer them) and translations_en.json
(an English translation of every cached German text, shown in the English
version of the notebook).

    uv run python -m tools.precompute_w3_01 --stage translate  # translations only
    uv run python -m tools.precompute_w3_01 --stage rescore    # changed references only
    uv run python -m tools.precompute_w3_01 --stage more-metrics  # Context Recall, Faithfulness (median of 3)
    uv run python -m tools.precompute_w3_01 --stage synth-sources  # source lines of the RAGAS question in w3_01

API calls: one embedding batch for the chunks, then per question one query
embedding, one answer and the judge calls of both metrics (10 questions), then
one translation call per 15 texts. The rescore stage keeps chunks and answer of
a manual question whose reference in MANUAL changed, runs only the judge calls
of both metrics for it and translates the new reference.
"""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ragkit import intro_rag as rag
from ragkit.config import (
    API_BASE_URL,
    DATA_DIR,
    EMBED_MODEL_NAME,
    EVALUATOR_MODEL_NAME,
    RAG_MODEL_NAME,
)
from ragkit.embed import embed

TODAY = datetime.now(UTC).date().isoformat()

# The five manual test questions of the former w3_01, with their reference answers.
# `source` holds a fragment of each website line that answers the question; the
# script resolves it against the crawl and fails loudly when the page changed.
MANUAL = [
    {'question': 'Welche Angebote bietet das KI-Servicezentrum an?',
     'reference': 'Das KI-Servicezentrum bietet kostenlose Beratung, Weiterbildung und Rechenleistung (Infrastruktur) an.',
     'source': ['Wir bieten Beratung, Weiterbildung und Infrastruktur']},
    {'question': 'Sind alle Services kostenlos?',
     'reference': 'Ja, alle Services sind kostenlos, da das KI-Servicezentrum vom Bundesministerium für Forschung, Technologie und Raumfahrt gefördert wird.',
     'source': ['Wir bieten Beratung, Weiterbildung und Infrastruktur', 'Das zugrunde liegende Vorhaben wurde mit Mitteln']},
    {'question': 'Welche Beratungsangebote bietet das KI-Servicezentrum?',
     'reference': 'Das KI-Servicezentrum bietet Gruppensprechstunden und individuelle Beratung zum Einsatz von KI. Zudem können dreimonatige Pilotprojekte gemeinsam durchgeführt werden.',
     'source': ['Unsere Sprechstunden finden online', 'Bewerbungen für ein begleitetes KI-Pilotprojekt']},
    {'question': 'Was passiert in einem Pilotprojekt?',
     'reference': 'Unternehmen können sich für Pilotprojekte bewerben und erhalten 2-3 Tage pro Woche aktive Projektbegleitung, um gemeinsam einen Prototypen zu entwickeln. Dabei können sie auch die Recheninfrastruktur nutzen.',
     'source': ['In unseren KI-Pilotprojekten entwickeln', '2-3 Tage pro Woche', 'Nutzung unserer Infrastruktur für Training', 'Bewerbungen für ein begleitetes KI-Pilotprojekt']},
    {'question': 'Muss man sich auf ein Pilotprojekt bewerben, um die Recheninfrastruktur nutzen zu können?',
     'reference': 'Nein. Wenn Sie die Recheninfrastruktur nutzen möchten, reichen Sie über das Portal des KI-Servicezentrums eine Beschreibung Ihres Vorhabens ein. Ein Produktivbetrieb wird nicht angeboten.',
     'source': ['reichen Sie eine Beschreibung für Ihr Vorhaben']},
]


# The RAGAS question of the "Schreiben Sie selbst" task in w3_01, with fragments of the website
# lines that answer them (resolved against the crawl like `source` in MANUAL: first line that
# contains the fragment; the script fails loudly when the page changed).
SYNTH_SOURCES = {
    'How can I stay informed about upcoming KI‑Pilotprojekt application phases through the Newsletter?':
        ['Bewerbungen für ein begleitetes KI-Pilotprojekt', '[Newsletter abonnieren]'],
}


def stage_synth_sources(chunks: list[str]) -> None:
    """Store the website lines that answer the write-it-yourself question; drop them elsewhere; no API calls."""
    results = rag.load_results()
    for row in results['synthetic']:
        frags = next((v for k, v in SYNTH_SOURCES.items() if row['question'].startswith(k)), None)
        if frags is None:
            row.pop('source_ids', None)
            row.pop('source_lines', None)
            continue
        ids = []
        for frag in frags:
            hits = [i for i, c in enumerate(chunks) if frag in c]
            if not hits:
                raise SystemExit(f'source fragment not on the page any more: {frag!r}')
            ids.append(hits[0])
        row['source_ids'] = sorted(set(ids))
        row['source_lines'] = [chunks[i] for i in row['source_ids']]
        print(f"  {row['question'][:60]}: lines {row['source_ids']}")
    rag.RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')


def stage_crawl(recrawl: bool) -> list[str]:
    if rag.CHUNKS_FILE.exists() and not recrawl:
        return json.loads(rag.CHUNKS_FILE.read_text(encoding='utf-8'))['chunks']
    chunks = rag.crawl()
    rag.CACHE.mkdir(parents=True, exist_ok=True)
    rag.CHUNKS_FILE.write_text(json.dumps(
        {'url': rag.URL, 'crawled': TODAY, 'chunks': chunks},
        ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(chunks)} chunks from {rag.URL}')
    return chunks


def stage_embed(chunks: list[str], fresh: bool) -> np.ndarray:
    if rag.VECTORS_FILE.exists() and not fresh:
        vectors = np.load(rag.VECTORS_FILE)
        if len(vectors) == len(chunks):
            return vectors
    vectors = np.vstack(embed(chunks)).astype(np.float32)
    np.save(rag.VECTORS_FILE, vectors)
    return vectors


def resolve_sources(chunks: list[str]) -> list[list[int]]:
    out = []
    for item in MANUAL:
        ids = []
        for frag in item['source']:
            hits = [i for i, c in enumerate(chunks) if frag in c]
            if not hits:
                raise SystemExit(f'source fragment not on the page any more: {frag!r}')
            ids.append(hits[0])
        out.append(ids)
    return out


def run_pipeline(questions: list[str], references: list[str], chunks, vectors) -> list[dict]:
    rows = []
    for q, ref in zip(questions, references):
        idx = rag.retrieve(q, chunks, vectors)
        ctx = [chunks[i] for i in idx]
        rows.append({'question': q, 'reference': ref, 'chunk_ids': idx, 'contexts': ctx,
                     'answer': rag.generate(q, ctx)})
        print(f'  answered: {q[:70]}')
    return asyncio.run(rag.score_many(rows))


TRANSLATE_PROMPT = (
    'Translate each text in the JSON array into American English. Keep Markdown, links, '
    'numbers and product names as they are. If a text is already English, return it unchanged. '
    'Reply with a JSON array of strings only, same length and order as the input.'
)


def _translate_batch(texts: list[str]) -> list[str]:
    from litellm import completion

    for _ in range(3):
        resp = completion(
            model=RAG_MODEL_NAME,
            messages=[{'role': 'system', 'content': TRANSLATE_PROMPT},
                      {'role': 'user', 'content': json.dumps(texts, ensure_ascii=False)}],
            api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0, max_tokens=16000,
        )
        raw = (resp.choices[0].message.content or '').strip()
        raw = raw[raw.find('['):raw.rfind(']') + 1]
        try:
            out = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(out, list) and len(out) == len(texts) and all(isinstance(x, str) for x in out):
            return out
    raise SystemExit(f'translation failed for a batch starting with {texts[0][:60]!r}')


def stage_translate(chunks: list[str], batch: int = 15) -> None:
    """Translate every cached corpus text once; answers and scores stay as they are."""
    results = rag.load_results()
    rows = results['manual'] + results['synthetic']
    texts = list(chunks)
    for r in rows:
        texts += [r['question'], r['reference'], r['answer']]
    done = rag.load_translations()
    todo = [x for x in dict.fromkeys(texts) if x not in done and x != '<!-- image -->']
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        done.update(zip(part, _translate_batch(part)))
        print(f'  translated {min(i + batch, len(todo))}/{len(todo)}')
    done.update(TRANSLATION_FIXES)
    rag.TRANSLATIONS_FILE.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding='utf-8')


# Hand fixes applied after every translation run. The typo card of w3_01 is about the typos,
# so its translation must not hide them.
TRANSLATION_FIXES = {
    'Wenr und wiie kann ich d KI-Servicenzentrum Berlin Brandenbarg kontaktieren, um ein pilotprojekt zu starten und die Infrastruktur für Training zu nutzten?':
        'When and how can I contact the KI-Servicezentrum Berlin Brandenburg to start a pilot project and use the training infrastructure? (The original contains typos.)',
    # Same name for the center as in the notebook text.
    'Nein. Wenn Sie die Recheninfrastruktur nutzen möchten, reichen Sie über das Portal des KI-Servicezentrums eine Beschreibung Ihres Vorhabens ein. Ein Produktivbetrieb wird nicht angeboten.':
        'No. If you would like to use the computing infrastructure, submit a description of your project via the AI Service Center portal. Production operation is not offered.',
}


def stage_rescore(chunks: list[str]) -> None:
    """Rescore the manual questions whose reference in MANUAL differs from the cache.

    Retrieval and answer do not depend on the reference, so chunks and answer stay;
    only Context Precision and Answer Correctness are judged again.
    """
    results = rag.load_results()
    changed = [(row, item) for row, item in zip(results['manual'], MANUAL)
               if row['question'] == item['question'] and row['reference'] != item['reference']]
    if not changed:
        print('no changed references')
        return
    old_refs = [row['reference'] for row, _ in changed]
    rows = [{**row, 'reference': item['reference']} for row, item in changed]
    scored = asyncio.run(rag.score_many(rows))
    by_q = {r['question']: r for r in scored}
    results['manual'] = [by_q.get(r['question'], r) for r in results['manual']]
    for r in scored:
        print(f"  rescored: {r['question'][:70]}: context_precision {r['context_precision']:.3f}, "
              f"answer_correctness {r['answer_correctness']:.3f}")
    rag.RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    # Drop the translations of the replaced references, then translate the new ones.
    still_used = {r['reference'] for r in results['manual'] + results['synthetic']}
    done = rag.load_translations()
    for ref in old_refs:
        if ref not in still_used:
            done.pop(ref, None)
    rag.TRANSLATIONS_FILE.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding='utf-8')
    stage_translate(chunks)


JUDGE_RUNS = 3  # Context Recall and Faithfulness: median of this many judge runs per question


def stage_more_metrics(runs: int = JUDGE_RUNS) -> None:
    """Context Recall and Faithfulness for every cached row, as the median of `runs` judge runs.

    Uses the cached chunks, answers and references; answers and the other two scores
    stay untouched. Every run is kept in 'context_recall_runs' and 'faithfulness_runs'
    (a single value cached before this stage existed counts as the first run). Only
    the missing runs are computed, so a second call makes no API calls.
    """
    import statistics

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall, Faithfulness

    results = rag.load_results()
    rows = results['manual'] + results['synthetic']
    for r in rows:  # rows are the dicts inside `results`
        for key in ('context_recall', 'faithfulness'):
            if f'{key}_runs' not in r:
                r[f'{key}_runs'] = [r[key]] if key in r else []
    missing = max(runs - min(len(r[f'{k}_runs']) for r in rows for k in ('context_recall', 'faithfulness')), 0)
    if missing:
        judge = llm_factory(EVALUATOR_MODEL_NAME,
                            client=AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=API_BASE_URL),
                            max_tokens=16384)
        cr, fa = ContextRecall(llm=judge), Faithfulness(llm=judge)

        async def _one_round(todo_cr, todo_fa):
            cr_res = await cr.abatch_score([{'user_input': r['question'], 'retrieved_contexts': r['contexts'],
                                             'reference': r['reference']} for r in todo_cr]) if todo_cr else []
            fa_res = await fa.abatch_score([{'user_input': r['question'], 'response': r['answer'],
                                             'retrieved_contexts': r['contexts']} for r in todo_fa]) if todo_fa else []
            return cr_res, fa_res

        for n in range(missing):
            todo_cr = [r for r in rows if len(r['context_recall_runs']) < runs]
            todo_fa = [r for r in rows if len(r['faithfulness_runs']) < runs]
            cr_res, fa_res = asyncio.run(_one_round(todo_cr, todo_fa))
            for r, c in zip(todo_cr, cr_res):
                r['context_recall_runs'].append(float(c.value))
            for r, f in zip(todo_fa, fa_res):
                r['faithfulness_runs'].append(float(f.value))
            print(f'  judge round {n + 1} of {missing}: {len(todo_cr)} context recall, {len(todo_fa)} faithfulness')
    for r in rows:
        r['context_recall'] = float(statistics.median(r['context_recall_runs']))
        r['faithfulness'] = float(statistics.median(r['faithfulness_runs']))
        print(f"  {r['question'][:55]}: context_recall {r['context_recall_runs']} -> {r['context_recall']:.3f}, "
              f"faithfulness {r['faithfulness_runs']} -> {r['faithfulness']:.3f}")
    results['context_recall_faithfulness'] = f'median of {runs} judge runs'
    rag.RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--recrawl', action='store_true', help='crawl the website again')
    ap.add_argument('--stage', choices=['crawl', 'translate', 'rescore', 'more-metrics', 'synth-sources', 'all'], default='all')
    args = ap.parse_args()

    chunks = stage_crawl(args.recrawl)
    if args.stage == 'crawl':
        for i, c in enumerate(chunks):
            print(f'[{i}] {c}')
        return
    if args.stage == 'translate':
        stage_translate(chunks)
        return
    if args.stage == 'rescore':
        stage_rescore(chunks)
        return
    if args.stage == 'more-metrics':
        stage_more_metrics()
        return
    if args.stage == 'synth-sources':
        stage_synth_sources(chunks)
        return
    sources = resolve_sources(chunks)
    vectors = stage_embed(chunks, fresh=args.recrawl)

    synth = pd.read_csv(DATA_DIR / 'synthetic_testset_intro.csv')
    print('manual questions')
    manual = run_pipeline([m['question'] for m in MANUAL], [m['reference'] for m in MANUAL], chunks, vectors)
    for row, ids in zip(manual, sources):
        row['source_ids'] = ids
        row['source_lines'] = [chunks[i] for i in ids]
    print('synthetic questions')
    synthetic = run_pipeline(synth['user_input'].tolist(), synth['reference'].tolist(), chunks, vectors)
    for row, kind in zip(synthetic, synth['synthesizer_name']):
        row['synthesizer_name'] = kind

    rag.RESULTS_FILE.write_text(json.dumps({
        'top_k': rag.TOP_K, 'embedding': EMBED_MODEL_NAME, 'answer_model': RAG_MODEL_NAME,
        'judge': EVALUATOR_MODEL_NAME, 'computed': TODAY,
        'manual': manual, 'synthetic': synthetic,
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    for name, rows in (('manual', manual), ('synthetic', synthetic)):
        cp = np.mean([r['context_precision'] for r in rows])
        ac = np.mean([r['answer_correctness'] for r in rows])
        print(f'{name}: context_precision {cp:.3f}, answer_correctness {ac:.3f}')
    print('context recall and faithfulness')
    stage_more_metrics()
    print('sources of the write-it-yourself questions')
    stage_synth_sources(chunks)
    print('translations')
    stage_translate(chunks)


if __name__ == '__main__':
    main()
