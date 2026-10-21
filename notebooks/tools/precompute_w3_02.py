"""Build the cache of w3_02: top-10 chunks, demo questions with judge verdicts, translations.

Run once from `notebooks/` (Qdrant with the standard collection must be up):

    uv run python -m tools.precompute_w3_02                 # all stages, reusing what exists
    uv run python -m tools.precompute_w3_02 --stage translate

Writes to data/cache/w3_02/:

- retrieval_top10.json: for all 43 questions of 43_komplex the top-10 chunks
  (text, rank, similarity) from grundschutz_chunks_json_structured_sections__octen-embedding-8b.
- lmm.json: mixed models score ~ TOP_K + (1 | question) and configuration vs standard (stage lmm, no API).
- demo_questions.json: the candidates for the demo questions with five judge
  runs each at TOP_K 5 (per-chunk verdicts with reasons, the claims of the
  reference with attributed and reason, Context Precision, Context Recall,
  number of claims), and the three chosen demo questions. Run 1 is the one the
  notebook shows; the participant recomputes both metrics from it.
- translations.json: English for every German corpus text the notebook shows
  (questions, references, passages, chunks of the top 5, claims) and German for
  the judge's English reasons.

- kisz/kisz_eval.json (stage kisz, module A of plan S1.3 section 3f, C27): the
  five manual questions of w3_01 about the website of the KI-Servicezentrum, their
  top-5 lines (question embedding against the cached line vectors of w3_01), one
  judge run each and four more for those with useful and useless lines and both
  metrics strictly between 0 and 1 (the demo questions), plus three
  decompositions of the reference by the second judge (llama-3-3-70b).
- kisz/translations.json: English for the shown website texts (from the w3_01
  cache where it exists) and the judge output, German for English reasons.

    uv run python -m tools.precompute_w3_02 --stage kisz

- diagnosis.json (stage diagnosis, extra task part b): the question with the lowest
  Context Recall at TOP_K 5 in the standard configuration and the judge's verdict
  per claim of its reference (run 1 from demo_questions.json when cached), plus
  English for those claims in translations.json.

    uv run python -m tools.precompute_w3_02 --stage diagnosis

Selection of the demo questions (plan S1.3, sections 3 and 3a): passage of at
most MAX_PASSAGE_WORDS words (plan: about 120), Context Precision and Context Recall of run 1 strictly between
0 and 1, two to four useful chunks (plan: two or three, see MAX_USEFUL) in run 1, at least two different values of
a metric over the five runs, three different modules (Bausteine). Among those,
shorter passages first.

API calls: one embedding batch (43 questions); per candidate and run five
Context Precision calls and one Context Recall call (a first run for every
candidate, four more only for those that pass the run-1 criteria); one
translation call per 12 texts.
"""

import argparse
import asyncio
import json
import os

from ragkit import w3_eval as ev
from ragkit.config import API_BASE_URL, RAG_MODEL_NAME

# The plan asks for "unter etwa 120 Wörtern". Under 130 words no question of 43_komplex has Context
# Precision and Context Recall both strictly between 0 and 1 (run of 2026-10-15), and with 200 words only
# two modules qualify, so the pool goes up to 280 words; shorter passages win.
MAX_PASSAGE_WORDS = 280
N_RUNS = 5
CONCURRENCY = 4


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')


def stage_retrieve(gold, fresh=False):
    if ev.RETRIEVAL_FILE.exists() and not fresh:
        return ev.load_retrieval()
    print('retrieval: 43 questions, top 10')
    hits = ev.retrieve([g['question'] for g in gold], top_k=10)
    questions = [{'index': g['index'], 'question': g['question'], 'chunks': h} for g, h in zip(gold, hits)]
    _save(ev.RETRIEVAL_FILE, {'collection': ev.COLLECTION, 'embedding': ev.EMBED_MODEL, 'top_k': 10,
                              'questions': questions})
    return questions


# The plan asks for two or three useful chunks. With that, only question 8 passes (run of 2026-10-15);
# four useful chunks keep the hand computation short and give three different modules.
MAX_USEFUL = 4


def _run1_ok(run):
    useful = sum(c['verdict'] for c in run['chunks'])
    return 2 <= useful <= MAX_USEFUL and 0 < run['context_precision'] < 1 and 0 < run['context_recall'] < 1


def _varies(runs):
    return any(len({round(r[m], 2) for r in runs}) >= 2 for m in ('context_precision', 'context_recall'))


async def _runs(llm, sem, g, contexts, n):
    async def one():
        async with sem:
            return await ev.judge_once(g['question'], g['reference'], contexts, llm=llm)
    return list(await asyncio.gather(*(one() for _ in range(n))))


async def stage_judge(gold, retrieval):
    state = json.loads(ev.DEMO_FILE.read_text(encoding='utf-8')) if ev.DEMO_FILE.exists() else {'candidates': []}
    by_idx = {c['index']: c for c in state['candidates']}
    llm, sem = ev.judge_llm(), asyncio.Semaphore(CONCURRENCY)
    pool = [g for g in gold if ev.words(g['fundstelle']) <= MAX_PASSAGE_WORDS]
    print(f'candidates: {len(pool)} questions with a passage of at most {MAX_PASSAGE_WORDS} words')
    for g in pool:
        ctx = [c['text'] for c in retrieval[g['index']]['chunks'][:5]]
        cand = by_idx.setdefault(g['index'], {'index': g['index'], 'runs': []})
        if not cand['runs']:
            cand['runs'] = await _runs(llm, sem, g, ctx, 1)
        if _run1_ok(cand['runs'][0]) and len(cand['runs']) < N_RUNS:
            cand['runs'] += await _runs(llm, sem, g, ctx, N_RUNS - len(cand['runs']))
        r = cand['runs'][0]
        print(f"  #{g['index']:2d} {g['baustein']}: useful {[c['verdict'] for c in r['chunks']]}, "
              f"CP {r['context_precision']:.2f}, CR {r['context_recall']:.2f} ({r['n_claims']} claims), "
              f"runs {[(round(x['context_precision'], 2), round(x['context_recall'], 2)) for x in cand['runs']]}")
        state['candidates'] = sorted(by_idx.values(), key=lambda c: c['index'])
        _save(ev.DEMO_FILE, {**state, 'demos': state.get('demos', [])})
    return state


def stage_select(gold, state):
    ok = [c for c in state['candidates']
          if len(c['runs']) >= N_RUNS and _run1_ok(c['runs'][0]) and _varies(c['runs'])]
    # Fewest useful chunks first (easiest hand computation, preset question), then the shorter passage.
    ok.sort(key=lambda c: (sum(x['verdict'] for x in c['runs'][0]['chunks']), ev.words(gold[c['index']]['fundstelle'])))
    demos, seen = [], set()
    for c in ok:
        b = gold[c['index']]['baustein']
        if b in seen:
            continue
        seen.add(b)
        demos.append(c)
        if len(demos) == 3:
            break
    print('demo questions:', [(d['index'], gold[d['index']]['baustein']) for d in demos])
    if len(demos) < 3:
        print(f'WARNING: only {len(demos)} demo questions meet all criteria')
    _save(ev.DEMO_FILE, {**state, 'demos': demos,
                         'criteria': {'max_passage_words': MAX_PASSAGE_WORDS, 'runs': N_RUNS, 'top_k': 5,
                                      'judge': ev.JUDGE_MODEL}})
    return demos


PROMPT = {
    'en': ('Translate each text in the JSON array into American English. Keep Markdown, line breaks, IDs such as '
           'OPS.2.2.A2, numbers and the capitalized modal verbs (MUSS, SOLLTE) as English equivalents in capitals '
           '(MUST, SHOULD). If a text is already English, return it unchanged. '
           'Reply with a JSON array of strings only, same length and order as the input.'),
    'de': ('Übersetze jeden Text im JSON-Array ins Deutsche. Behalte Markdown, IDs wie OPS.2.2.A2 und Zahlen bei. '
           'Ist ein Text schon deutsch, gib ihn unverändert zurück. '
           'Antworte nur mit einem JSON-Array aus Strings, gleiche Länge und Reihenfolge wie die Eingabe.'),
}


def _translate_batch(texts, target):
    from litellm import completion

    for _ in range(2):
        resp = completion(model=RAG_MODEL_NAME,
                          messages=[{'role': 'system', 'content': PROMPT[target]},
                                    {'role': 'user', 'content': json.dumps(texts, ensure_ascii=False)}],
                          api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'), temperature=0, max_tokens=32000,
                          timeout=300)
        raw = (resp.choices[0].message.content or '').strip()
        raw = raw[raw.find('['):raw.rfind(']') + 1]
        try:
            out = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(out, list) and len(out) == len(texts) and all(isinstance(x, str) for x in out):
            return out
    if len(texts) > 1:  # a long batch failed: one text at a time
        return [_translate_batch([x], target)[0] for x in texts]
    raise SystemExit(f'translation failed for {texts[0][:60]!r}')


def _translate_all(tr, target, texts, batch=6, workers=4, path=None):
    """Translate what is missing, saving after every batch so a failure loses nothing."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    path = path or ev.TRANSLATIONS_FILE
    done = tr[target]
    todo = [x for x in dict.fromkeys(texts) if x and x not in done]
    parts = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_translate_batch, p, target): p for p in parts}
        for n, f in enumerate(as_completed(futures), 1):
            done.update(zip(futures[f], f.result()))
            _save(path, tr)
            print(f'  {target}: batch {n}/{len(parts)}')


def stage_translate(gold, retrieval, demos):
    tr = ev.load_translations()
    to_en = [g[k] for g in gold for k in ('question', 'reference', 'fundstelle')]
    to_en += [c['text'] for q in retrieval for c in q['chunks'][:5]]
    to_de = []
    for d in demos:
        run = d['runs'][0]
        to_en += [c['statement'] for c in run['claims']]
        to_de += [c['reason'] for c in run['chunks']] + [c['reason'] for c in run['claims']]
        # The reasons go both ways: the judge may write German or English.
        to_en += [c['reason'] for c in run['chunks']] + [c['reason'] for c in run['claims']]
    _translate_all(tr, 'de', to_de)
    _translate_all(tr, 'en', to_en)


# ---------------------------------------------------------------------------
# Module A on the website of the KI-Servicezentrum (plan S1.3, section 3f; C27)
# ---------------------------------------------------------------------------

KISZ_TOP_K = 5
SECOND_JUDGE = 'llama-3-3-70b'
N_SECOND = 3  # decompositions of the reference by the second judge; the notebook shows the first


def _kisz_ok(run):
    """Useful and useless lines at TOP_K 5, Context Precision and Context Recall both strictly between 0 and 1."""
    useful = sum(c['verdict'] for c in run['chunks'])
    return 0 < useful < KISZ_TOP_K and 0 < run['context_precision'] < 1 and 0 < run['context_recall'] < 1


async def _kisz_judge(state, manual, lines):
    llm, sem = ev.judge_llm(), asyncio.Semaphore(CONCURRENCY)
    for q in state['questions']:
        ctx = [lines[i] for i in q['line_ids']]
        g = {'question': q['question'], 'reference': q['reference']}
        if not q['runs']:
            q['runs'] = await _runs(llm, sem, g, ctx, 1)
        if _kisz_ok(q['runs'][0]) and len(q['runs']) < N_RUNS:
            q['runs'] += await _runs(llm, sem, g, ctx, N_RUNS - len(q['runs']))
        r = q['runs'][0]
        print(f"  {q['question'][:60]}: lines {q['line_ids']}, useful {[c['verdict'] for c in r['chunks']]}, "
              f"CP {r['context_precision']:.2f}, CR {r['context_recall']:.2f} ({r['n_claims']} claims), "
              f"runs {[(round(x['context_precision'], 2), round(x['context_recall'], 2), x['n_claims']) for x in q['runs']]}")
        _save(ev.KISZ_FILE, state)
    second = ev.judge_llm(SECOND_JUDGE)
    for q in state['questions']:
        if not (len(q['runs']) >= N_RUNS and _kisz_ok(q['runs'][0])) or len(q.get('second', [])) >= N_SECOND:
            continue
        ctx = [lines[i] for i in q['line_ids']]
        q['second'] = [{'claims': c, 'context_recall': ev.recall(c), 'n_claims': len(c)} for c in await asyncio.gather(
            *(ev.judge_claims(second, q['question'], q['reference'], ctx) for _ in range(N_SECOND)))]
        print(f"  {SECOND_JUDGE} on {q['question'][:50]}: "
              f"{[(s['n_claims'], round(s['context_recall'], 2)) for s in q['second']]}")
        _save(ev.KISZ_FILE, state)
    state['demos'] = [i for i, q in enumerate(state['questions']) if len(q['runs']) >= N_RUNS and _kisz_ok(q['runs'][0])]
    print('KISZ demo questions:', state['demos'])
    _save(ev.KISZ_FILE, state)
    return state


def stage_kisz():
    """Top-5 lines of the five manual questions of w3_01, five judge runs, the second judge's decomposition."""
    import numpy as np

    from ragkit import intro_rag as rag
    from ragkit.embed import embed

    lines, vectors = rag.load_corpus()
    manual = rag.load_results()['manual']
    if ev.KISZ_FILE.exists():
        state = json.loads(ev.KISZ_FILE.read_text(encoding='utf-8'))
    else:
        print('KISZ: embedding the five questions, top 5 from the cached line vectors')
        qv = np.vstack(embed([m['question'] for m in manual])).astype(np.float32)
        sims = (qv @ vectors.T) / (np.linalg.norm(qv, axis=1)[:, None] * np.linalg.norm(vectors, axis=1)[None, :])
        state = {'judge': ev.JUDGE_MODEL, 'second_judge': SECOND_JUDGE, 'top_k': KISZ_TOP_K, 'questions': []}
        for m, s in zip(manual, sims):
            ids = [int(i) for i in np.argsort(-s)[:KISZ_TOP_K]]
            if ids[:3] != m['chunk_ids']:
                print(f"  note: top 3 {ids[:3]} differ from w3_01 {m['chunk_ids']} for {m['question'][:50]}")
            state['questions'].append({'question': m['question'], 'reference': m['reference'], 'line_ids': ids,
                                       'similarity': [round(float(s[i]), 4) for i in ids],
                                       'source_ids': m['source_ids'], 'runs': []})
        _save(ev.KISZ_FILE, state)
    state = asyncio.run(_kisz_judge(state, manual, lines))
    _kisz_translate(state, lines)


def _kisz_translate(state, lines):
    """English for the corpus texts (taken from w3_01 where it exists) and the judge output; German for English reasons."""
    from ragkit import intro_rag as rag

    w301 = rag.load_translations()
    tr = ev.load_kisz_translations()
    to_en, to_de = [], []
    for i in state['demos']:
        q = state['questions'][i]
        texts = [q['question'], q['reference'], *(lines[j] for j in q['line_ids'] + q['source_ids'])]
        for x in texts:
            if x in w301:
                tr['en'][x] = w301[x]
            else:
                to_en.append(x)
        for run in q['runs'][:1] + q.get('second', [])[:1]:
            reasons = [c['reason'] for c in run.get('chunks', [])] + [c['reason'] for c in run['claims']]
            to_en += [c['statement'] for c in run['claims']] + reasons
            to_de += reasons
    _translate_all(tr, 'de', to_de, path=ev.KISZ_TRANSLATIONS_FILE)
    _translate_all(tr, 'en', to_en, path=ev.KISZ_TRANSLATIONS_FILE)
    tr['en'].update(KISZ_EN_FIXES)
    _save(ev.KISZ_TRANSLATIONS_FILE, tr)


# Translations checked by hand (2026-10-19); they replace what the model wrote.
KISZ_EN_FIXES = {
    'KI-Experte und KI-Expertin am Tisch mit Teilnehmenden eines Pilotprojekts, vor ihnen Laptops.':
        'A male and a female AI expert at a table with participants of a pilot project, laptops in front of them.',
}


# ---------------------------------------------------------------------------
# Extra task, part b: diagnosis of the question with the lowest Context Recall (plan S1.3, row 18b)
# ---------------------------------------------------------------------------

DIAGNOSIS_FILE = ev.CACHE / 'diagnosis.json'


def stage_diagnosis(gold, retrieval):
    """The question with the lowest Context Recall at TOP_K 5 (standard configuration, ties: lowest index) and the
    judge's verdict per claim. Reuses run 1 of demo_questions.json when it exists (same top-5 chunks); otherwise one
    judge run (six calls). Then English for the claims (one translation call)."""
    cr = ev.load_config('standard')['top5']['context_recall']
    i = min(range(len(cr)), key=lambda k: (cr[k], k))
    cands = {c['index']: c for c in json.loads(ev.DEMO_FILE.read_text(encoding='utf-8'))['candidates']} \
        if ev.DEMO_FILE.exists() else {}
    if i in cands and cands[i]['runs']:
        run = cands[i]['runs'][0]
        print(f'diagnosis: question {i + 1}, run 1 from {ev.DEMO_FILE.name}')
    else:
        print(f'diagnosis: question {i + 1}, one judge run')
        run = asyncio.run(ev.judge_once(gold[i]['question'], gold[i]['reference'],
                                        [c['text'] for c in retrieval[i]['chunks'][:5]]))
    _save(DIAGNOSIS_FILE, {'index': i, 'top_k': 5, 'judge': ev.JUDGE_MODEL, 'context_recall_config': cr[i],
                           'claims': run['claims'], 'context_recall': run['context_recall']})
    tr = ev.load_translations()
    _translate_all(tr, 'en', [c['statement'] for c in run['claims']])


def stage_lmm():
    """Mixed models score ~ TOP_K + (1 | question) on the cached scores (no API, no Qdrant), see ragkit/w3_lmm.py."""
    from ragkit import w3_lmm

    out = w3_lmm.fit_all()
    _save(ev.LMM_FILE, out)
    for m, model in out['topk'].items():
        print(f"lmm {m}: " + ', '.join(f"{c['a']} vs {c['b']} {c['estimate']:+.3f} (p_holm {c['p_holm']:.3f})" for c in model['contrasts']))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stage', choices=['retrieve', 'judge', 'select', 'translate', 'kisz', 'diagnosis', 'lmm', 'all'],
                    default='all')
    ap.add_argument('--fresh', action='store_true', help='retrieve again even if the cache exists')
    args = ap.parse_args()

    if args.stage in ('kisz', 'all'):
        stage_kisz()
        if args.stage == 'kisz':
            return
    if args.stage in ('lmm', 'all'):
        stage_lmm()
        if args.stage == 'lmm':
            return
    gold = ev.load_gold()
    retrieval = stage_retrieve(gold, fresh=args.fresh)
    if args.stage == 'retrieve':
        return
    if args.stage in ('judge', 'all'):
        state = asyncio.run(stage_judge(gold, retrieval))
    else:
        state = json.loads(ev.DEMO_FILE.read_text(encoding='utf-8'))
    if args.stage in ('judge', 'select', 'all'):
        demos = stage_select(gold, state)
    else:
        demos = state['demos']
    if args.stage in ('translate', 'all'):
        stage_translate(gold, retrieval, demos)
    if args.stage in ('diagnosis', 'all'):
        stage_diagnosis(gold, retrieval)


if __name__ == '__main__':
    main()
