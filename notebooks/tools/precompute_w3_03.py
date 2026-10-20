"""Build the cache of w3_03: claim decompositions, example questions, prompt experiment, overview.

Run from `notebooks/`, stage by stage or all at once (each stage reuses what exists):

    uv run python -m tools.precompute_w3_03 --stage candidates  # decompose every candidate question
    uv run python -m tools.precompute_w3_03 --stage examples    # chosen examples (ROLES) with chunks
    uv run python -m tools.precompute_w3_03 --stage reference   # question 38 against the expert reference
    uv run python -m tools.precompute_w3_03 --stage experiment  # 3 prompts x 3 questions x 3 judge runs
    uv run python -m tools.precompute_w3_03 --stage experiment_claims  # TP/FP/FN of original and short answers
    uv run python -m tools.precompute_w3_03 --stage overview    # all four metrics for all 43 questions
    uv run python -m tools.precompute_w3_03 --stage translate   # English for displayed corpus text
    uv run python -m tools.precompute_w3_03 --stage kisz_candidates  # KISZ answers (also fuller ones) decomposed
    uv run python -m tools.precompute_w3_03 --stage kisz_examples    # chosen KISZ examples (KISZ_ROLES)
    uv run python -m tools.precompute_w3_03 --stage kisz_translate   # English for the new KISZ texts
    uv run python -m tools.precompute_w3_03                     # all stages

The examples are chosen by reading the candidates against their Fundstelle; the
choice and its reasons are written down in ROLES below, not computed.

API calls: per candidate question one retrieval embedding, three judge calls and
two embeddings for Answer Correctness, two judge calls for Faithfulness (12
candidates). The experiment generates six answers (prompts kurz and offen on
three questions; the original answers come from the cache) and scores all nine
answers three times with both metrics. The translation stage calls the answer
model once per 15 texts. The KISZ stages generate five fuller answers from the cached chunks of
data/cache/w3_01 and decompose 14 answers (two judge calls each for both metrics, two embeddings).
"""

import argparse
import asyncio
import json

from ragkit import w3_03_gen as gen
from tools.precompute_w3_01 import _translate_batch

# Candidates per role (section 3a of the component plan), plus 21 for the Faithfulness-0 check
# and three more for role A, whose planned example 32 the review doubted.
CANDIDATES = {
    'A': [32, 33, 28, 27],
    'B': [8, 5],
    'C': [6],
    'D': [4, 9, 0],
    'E': [3],
    'check': [21],
}

# Chosen after reading the decompositions in candidates.json against the Fundstelle; see the report.
ROLES = {'A': 32, 'B': 8, 'C': 6, 'D': 0, 'E': 3}

# Faithfulness 0 at question 21 is a misjudgment, reproduced in candidates.json: the judge rejects
# every claim because the answer states SOLLTE requirements as facts ('wird aktualisiert').
NOT_SCORABLE = {21}

# Three experiment questions: short reference, middle values, understandable on their own.
EXPERIMENT = [24, 22, 26]

# Task 'same answer, other reference' (section 3d): question 38 has a short gold passage (88 words) and an
# answer of 109 words that fits on one screen. The expert reference is only requirement SYS.1.2.3.A8; the
# passage and the answer also describe the threat (processes no longer accessible, memory dumps that cannot
# be evaluated), so a reference written from the passage differs visibly from the expert one.
REFERENCE_Q = 38
REFERENCE_KEY = 'R'
JUDGE_RUNS = 3


def _write(path, data) -> None:
    gen.CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')


async def _gather_limited(coros, limit: int = 6):
    sem = asyncio.Semaphore(limit)

    async def _one(c):
        async with sem:
            return await c
    return await asyncio.gather(*(_one(c) for c in coros))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_candidates() -> None:
    gold, src = gen.load_gold(), gen.load_source()
    done = gen.load_json_or_empty(gen.CANDIDATES_FILE)
    todo = [i for ids in CANDIDATES.values() for i in ids if str(i) not in done]
    if not todo:
        return
    ac, faith = gen.metrics()
    chunks = {i: gen.retrieve(gold[i]['question']) for i in todo}

    async def _one(i):
        q, a, ref = gold[i]['question'], src['answers'][i], gold[i]['answer']
        ctx = [c['text'] for c in chunks[i]]
        d_ac, d_f = await asyncio.gather(gen.adecompose_ac(ac, q, a, ref), gen.adecompose_faith(faith, q, a, ctx))
        print(f'  {i}: AC {d_ac["score"]:.2f} (cache {src["answer_correctness"][i]:.2f}), '
              f'F {d_f["score"]} (cache {src["faithfulness"][i]:.2f})')
        return i, {'ac': d_ac, 'faithfulness': d_f, 'chunks': chunks[i]}

    for i, rec in asyncio.run(_gather_limited([_one(i) for i in todo])):
        done[str(i)] = rec
    _write(gen.CANDIDATES_FILE, done)


def _words(text: str) -> int:
    return len(text.split())


def stage_examples() -> None:
    gold, src = gen.load_gold(), gen.load_source()
    cand = gen.load(gen.CANDIDATES_FILE)
    out = {}
    for role, i in ROLES.items():
        c = cand[str(i)]
        out[role] = {
            'index': i, 'question': gold[i]['question'], 'reference': gold[i]['answer'],
            'fundstelle': gold[i]['fundstelle'], 'answer': src['answers'][i], 'chunks': c['chunks'],
            'context_precision': src['context_precision'][i], 'context_recall': src['context_recall'][i],
            'ac': c['ac'], 'faithfulness': c['faithfulness'],
            'words_answer': _words(src['answers'][i]), 'words_reference': _words(gold[i]['answer']),
        }
    out['check21'] = cand['21']
    old = gen.load_json_or_empty(gen.EXAMPLES_FILE)
    if REFERENCE_KEY in old:  # written by stage_reference
        out[REFERENCE_KEY] = old[REFERENCE_KEY]
    _write(gen.EXAMPLES_FILE, out)


def stage_reference() -> None:
    """Question REFERENCE_Q against the expert reference, for the task 'same answer, other reference'.

    Participants write their own reference live; the notebook then reuses these answer claims, so
    only the reference changes. Nothing a participant writes is cached.
    """
    ex = gen.load(gen.EXAMPLES_FILE)
    if REFERENCE_KEY in ex:
        return
    gold, src = gen.load_gold(), gen.load_source()
    i = REFERENCE_Q
    q, a, ref = gold[i]['question'], src['answers'][i], gold[i]['answer']
    ac, _ = gen.metrics()
    d_ac = asyncio.run(gen.adecompose_ac(ac, q, a, ref))
    print(f'  {i}: AC {d_ac["score"]:.2f} (cache {src["answer_correctness"][i]:.2f}), '
          f'TP {len(d_ac["TP"])}, FP {len(d_ac["FP"])}, FN {len(d_ac["FN"])}')
    ex[REFERENCE_KEY] = {
        'index': i, 'question': q, 'reference': ref, 'fundstelle': gold[i]['fundstelle'], 'answer': a,
        'context_precision': src['context_precision'][i], 'context_recall': src['context_recall'][i],
        'ac': d_ac, 'words_answer': _words(a), 'words_reference': _words(ref),
    }
    _write(gen.EXAMPLES_FILE, ex)


def stage_experiment() -> None:
    gold, src = gen.load_gold(), gen.load_source()
    done = gen.load_json_or_empty(gen.EXPERIMENT_FILE)
    questions = done.get('questions') or [
        {'index': i, 'question': gold[i]['question'], 'reference': gold[i]['answer'],
         'fundstelle': gold[i]['fundstelle'], 'chunks': gen.retrieve(gold[i]['question'])}
        for i in EXPERIMENT]
    results = done.get('results', {})
    ac, faith = gen.metrics()

    async def _prompt(name):
        rows = []
        for i, q in zip(EXPERIMENT, questions):
            ctx = [c['text'] for c in q['chunks']]
            answer = src['answers'][i] if name == 'original' else await gen.agenerate(q['question'], ctx, gen.PROMPTS[name])
            runs = await _gather_limited([gen.ascore(ac, faith, q['question'], answer, q['reference'], ctx)
                                          for _ in range(JUDGE_RUNS)], limit=3)
            print(f'  {name} / {i}: AC {[round(r["ac"], 2) for r in runs]}, F {[round(r["faithfulness"], 2) for r in runs]}')
            rows.append({'answer': answer, 'runs': runs})
        return name, rows

    for name in gen.PROMPTS:
        if name not in results:
            results[name] = asyncio.run(_prompt(name))[1]
            _write(gen.EXPERIMENT_FILE, {'questions': questions, 'prompts': gen.PROMPTS, 'results': results})


def stage_experiment_claims() -> None:
    """Answer Correctness claims (TP, FP, FN) for the original and the short answers, one judge run each."""
    exp = gen.load(gen.EXPERIMENT_FILE)
    if 'claims' in exp:
        return
    ac, _ = gen.metrics()

    async def _all():
        out = {}
        for name in ('original', 'kurz'):
            rows = await _gather_limited([gen.adecompose_ac(ac, q['question'], r['answer'], q['reference'])
                                          for q, r in zip(exp['questions'], exp['results'][name])], limit=3)
            out[name] = [{'TP': len(d['TP']), 'FP': len(d['FP']), 'FN': len(d['FN']), 'f1': d['f1'],
                          'similarity': d['similarity'], 'score': d['score']} for d in rows]
            print(name, [(d['TP'], d['FP'], d['FN'], round(d['f1'], 2)) for d in out[name]])
        return out
    exp['claims'] = asyncio.run(_all())
    _write(gen.EXPERIMENT_FILE, exp)


def stage_overview() -> None:
    """All four metrics per question, as the TOP_K 5 run of the gold set produced them."""
    gold, src = gen.load_gold(), gen.load_source()
    rows = [{
        'index': i, 'question': g['question'],
        'context_precision': src['context_precision'][i], 'context_recall': src['context_recall'][i],
        'answer_correctness': src['answer_correctness'][i], 'faithfulness': src['faithfulness'][i],
        'words_answer': _words(src['answers'][i]), 'words_reference': _words(g['answer']),
        'not_scorable': i in NOT_SCORABLE,
    } for i, g in enumerate(gold)]
    _write(gen.OVERVIEW_FILE, rows)


def _texts_to_translate() -> tuple[list[str], list[str]]:
    """German texts the English version shows, and English judge reasons the German version shows."""
    gold, src = gen.load_gold(), gen.load_source()
    ex, exp, ov = gen.load(gen.EXAMPLES_FILE), gen.load(gen.EXPERIMENT_FILE), gen.load(gen.OVERVIEW_FILE)
    de, en = [], []
    for k, e in ex.items():
        if k not in ROLES:
            continue
        de += [e['question'], e['reference'], e['fundstelle'], e['answer'], *(c['text'] for c in e['chunks'])]
        if k != 'A':  # only the example of rows 6 to 8 shows its claims
            continue
        for kind in ('TP', 'FP', 'FN'):
            de += [s['statement'] for s in e['ac'][kind]]
            en += [s['reason'] for s in e['ac'][kind]]
        de += [s['statement'] for s in e['faithfulness']['claims']]
        en += [s['reason'] for s in e['faithfulness']['claims']]
    if REFERENCE_KEY in ex:  # task 'same answer, other reference': texts and the claims too many
        r = ex[REFERENCE_KEY]
        de += [r['question'], r['reference'], r['fundstelle'], r['answer'], *(s['statement'] for s in r['ac']['FP'])]
    for q in exp['questions']:
        de += [q['question'], q['reference'], q['fundstelle']]
    for rows in exp['results'].values():
        de += [r['answer'] for r in rows]
    de += [r['question'] for r in ov]
    # The cause task of the extra part shows every question at the lower right in full.
    for r in ov:
        if r['context_recall'] >= 0.5 and r['answer_correctness'] < 0.5:
            i = r['index']
            de += [gold[i]['answer'], gold[i]['fundstelle'], src['answers'][i]]
    return list(dict.fromkeys(de)), list(dict.fromkeys(en))


TO_GERMAN = (
    'Translate each text in the JSON array into German. Keep Markdown, numbers, requirement IDs and '
    'product names as they are. If a text is already German, return it unchanged. Reply with a JSON '
    'array of strings only, same length and order as the input.'
)


def _to_german_batch(texts: list[str]) -> list[str]:
    import os

    from litellm import completion

    from ragkit.config import API_BASE_URL, RAG_MODEL_NAME
    for _ in range(3):
        resp = completion(model=RAG_MODEL_NAME, api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'),
                          messages=[{'role': 'system', 'content': TO_GERMAN},
                                    {'role': 'user', 'content': json.dumps(texts, ensure_ascii=False)}],
                          temperature=0, max_tokens=16000)
        raw = (resp.choices[0].message.content or '').strip()
        raw = raw[raw.find('['):raw.rfind(']') + 1]
        try:
            out = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(out, list) and len(out) == len(texts) and all(isinstance(x, str) for x in out):
            return out
    raise SystemExit(f'translation failed for a batch starting with {texts[0][:60]!r}')


PLAIN_TO_ENGLISH = (
    'Translate the text into American English. Keep Markdown, tables, links, numbers, requirement IDs '
    'and product names as they are. Reply with the translation only.'
)


def _plain_to_english(texts: list[str]) -> list[str]:
    """One long text as plain text: answers and Fundstellen break the JSON-array format."""
    import os

    from litellm import completion

    from ragkit.config import API_BASE_URL, RAG_MODEL_NAME
    resp = completion(model=RAG_MODEL_NAME, api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'),
                      messages=[{'role': 'system', 'content': PLAIN_TO_ENGLISH},
                                {'role': 'user', 'content': texts[0]}],
                      temperature=0, max_tokens=20000, timeout=300, num_retries=2)
    out = (resp.choices[0].message.content or '').strip()
    if not out:
        raise RuntimeError(f'empty translation for {texts[0][:60]!r}')
    return [out]


def _translate(path, texts: list[str], fn, batch: int, long_fn=None) -> None:
    """Translate what is missing, six requests at a time; long texts one by one with `long_fn`."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    done = gen.load_json_or_empty(path)
    todo = [t for t in texts if t not in done]
    long_, short = [t for t in todo if len(t) > 1500], [t for t in todo if len(t) <= 1500]
    jobs = [([t], long_fn or fn) for t in long_] + [(short[k:k + batch], fn) for k in range(0, len(short), batch)]
    with ThreadPoolExecutor(6) as pool:
        futures = {pool.submit(f, b): b for b, f in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            b = futures[fut]
            try:
                done.update(zip(b, fut.result()))
            except (Exception, SystemExit) as e:  # noqa: BLE001 - fall back to one text at a time
                if long_fn is None:
                    print(f'  failed: {b[0][:60]!r} ({e})', flush=True)
                else:
                    done.update((x, long_fn([x])[0]) for x in b)
            _write(path, done)
            print(f'  {path.name}: {n} of {len(jobs)}', flush=True)


def stage_translate() -> None:
    de, en = _texts_to_translate()
    _translate(gen.TRANSLATIONS_FILE, de, _translate_batch, 15, long_fn=_plain_to_english)
    _translate(gen.REASONS_DE_FILE, en, _to_german_batch, 20)


# ---------------------------------------------------------------------------
# KISZ website (section 3e and C28): judging single answers by hand uses the short
# texts of the w3_01 pipeline; the gold set stays for measuring (sections F to H).
# ---------------------------------------------------------------------------

KISZ_DIR = gen.KISZ_DIR
KISZ_CANDIDATES_FILE = KISZ_DIR / 'candidates.json'

# Candidates as (set, index in data/cache/w3_01/results.json, prompt). 'pipeline' keeps the cached
# answer of w3_01; a prompt name of gen.PROMPTS generates a second answer from the same three chunks.
KISZ_CANDIDATES = [
    ('manual', 0, 'pipeline'), ('manual', 0, 'original'), ('manual', 0, 'offen'),
    ('manual', 1, 'pipeline'), ('manual', 1, 'original'), ('manual', 1, 'offen'),
    ('manual', 2, 'pipeline'), ('manual', 3, 'pipeline'), ('manual', 4, 'pipeline'),
    ('synthetic', 0, 'pipeline'), ('synthetic', 1, 'pipeline'),
    ('manual', 2, 'original'), ('manual', 3, 'original'), ('manual', 4, 'original'),
]


def _kisz_key(kind: str, i: int, prompt: str) -> str:
    return f'{kind}/{i}/{prompt}'


def stage_kisz_candidates() -> None:
    """Answers and claim decompositions (Answer Correctness, Faithfulness) for every KISZ candidate."""
    from ragkit import intro_rag as rag

    res = rag.load_results()
    done = gen.load_json_or_empty(KISZ_CANDIDATES_FILE)
    todo = [c for c in KISZ_CANDIDATES if _kisz_key(*c) not in done]
    if not todo:
        return
    ac, faith = gen.metrics()

    async def _one(kind, i, prompt):
        r = res[kind][i]
        answer = r['answer'] if prompt == 'pipeline' else await gen.agenerate(r['question'], r['contexts'], gen.PROMPTS[prompt])
        d_ac, d_f = await asyncio.gather(gen.adecompose_ac(ac, r['question'], answer, r['reference']),
                                         gen.adecompose_faith(faith, r['question'], answer, r['contexts']))
        print(f'  {kind}/{i}/{prompt}: AC {d_ac["score"]:.2f} (TP {len(d_ac["TP"])}, FP {len(d_ac["FP"])}, '
              f'FN {len(d_ac["FN"])}), F {d_f["score"]}, {_words(answer)} words', flush=True)
        return _kisz_key(kind, i, prompt), {'kind': kind, 'i': i, 'prompt': prompt, 'answer': answer,
                                             'ac': d_ac, 'faithfulness': d_f}

    for k, rec in asyncio.run(_gather_limited([_one(*c) for c in todo], limit=4)):
        done[k] = rec
    KISZ_DIR.mkdir(parents=True, exist_ok=True)
    _write(KISZ_CANDIDATES_FILE, done)


# Chosen after reading the candidates against the website lines (see the report of 2026-10-19).
# B, C, D: the fuller answer to manual 0 (original prompt of section I, no brevity): all three offers
# of the reference are hits, five explanations count as too many, two of three Faithfulness claims add
# details that are not in the chunks. The short answer of w3_01 to the same question is card 2 of the
# matrix. Card 1: manual 2, the search missed the lines on consultation hours (Context Recall 0).
# Extra task: part a manual 3 and 1, part d manual 4 ("Ja" instead of "Nein", Faithfulness 1).
# Same answer, other reference: the newsletter question of w3_01 section E with the RAGAS reference.
KISZ_ROLES = {
    'B': ('manual', 0, 'original'),
    'cards': [('manual', 2, 'pipeline'), ('manual', 0, 'pipeline')],
    'extra_a': [('manual', 3, 'pipeline'), ('manual', 1, 'pipeline')],
    'extra_d': ('manual', 4, 'pipeline'),
    'ref': ('synthetic', 0, 'pipeline'),
}
KISZ_EXAMPLES_FILE = gen.KISZ_EXAMPLES_FILE
KISZ_TRANSLATIONS_FILE = gen.KISZ_TRANSLATIONS_FILE


def _kisz_row(res, cand, kind, i, prompt, decomposed: bool) -> dict:
    """One KISZ example: texts and source lines of w3_01; scores of w3_01 for its own answers,
    of the decomposition when the notebook shows the calculation (`decomposed`)."""
    r = res[kind][i]
    c = cand[_kisz_key(kind, i, prompt)]
    row = {
        'kind': kind, 'i': i, 'prompt': prompt, 'question': r['question'], 'reference': r['reference'],
        'answer': c['answer'], 'chunk_ids': r['chunk_ids'], 'contexts': r['contexts'],
        'source_ids': r.get('source_ids', []), 'source_lines': r.get('source_lines', []),
        'context_precision': r['context_precision'], 'context_recall': r['context_recall'],
        'words_answer': _words(c['answer']), 'words_reference': _words(r['reference']),
    }
    if decomposed:
        row |= {'ac': c['ac'], 'faithfulness': c['faithfulness'],
                'answer_correctness': c['ac']['score'], 'faithfulness_score': c['faithfulness']['score']}
    else:  # the values participants saw in w3_01 (Faithfulness: median of three judge runs)
        row |= {'answer_correctness': r['answer_correctness'], 'faithfulness_score': r['faithfulness']}
    return row


def stage_kisz_examples() -> None:
    from ragkit import intro_rag as rag

    res, cand = rag.load_results(), gen.load(KISZ_CANDIDATES_FILE)
    b = _kisz_row(res, cand, *KISZ_ROLES['B'], decomposed=True)
    short = res['manual'][KISZ_ROLES['B'][1]]
    b['short'] = {'answer': short['answer'], 'answer_correctness': short['answer_correctness'],
                  'faithfulness_score': short['faithfulness'], 'words_answer': _words(short['answer'])}
    out = {
        'B': b,
        'cards': [_kisz_row(res, cand, *c, decomposed=False) for c in KISZ_ROLES['cards']],
        'extra_a': [_kisz_row(res, cand, *c, decomposed=False) for c in KISZ_ROLES['extra_a']],
        'extra_d': _kisz_row(res, cand, *KISZ_ROLES['extra_d'], decomposed=False),
        'ref': _kisz_row(res, cand, *KISZ_ROLES['ref'], decomposed=True)
        | {'answer_correctness_w3_01': res['synthetic'][0]['answer_correctness']},
    }
    _write(KISZ_EXAMPLES_FILE, out)


def stage_kisz_translate() -> None:
    """English for the new German texts (the fuller answer, the claims); w3_01 already covers the rest."""
    from ragkit import intro_rag as rag

    ex, have = gen.load(KISZ_EXAMPLES_FILE), rag.load_translations()
    b, ref = ex['B'], ex['ref']
    de = [b['answer'], *(s['statement'] for k in ('TP', 'FP', 'FN') for s in b['ac'][k]),
          *(c['statement'] for c in b['faithfulness']['claims']),
          *(s['statement'] for k in ('TP', 'FP', 'FN') for s in ref['ac'][k])]
    _translate(KISZ_TRANSLATIONS_FILE, [x for x in dict.fromkeys(de) if x not in have],
               _translate_batch, 15, long_fn=_plain_to_english)


STAGES = {'candidates': stage_candidates, 'examples': stage_examples, 'reference': stage_reference,
          'kisz_candidates': stage_kisz_candidates, 'kisz_examples': stage_kisz_examples,
          'kisz_translate': stage_kisz_translate,
          'experiment': stage_experiment,
          'experiment_claims': stage_experiment_claims,
          'overview': stage_overview, 'translate': stage_translate}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=list(STAGES), help='run one stage only')
    args = parser.parse_args()
    for name, fn in STAGES.items():
        if args.stage in (None, name):
            print(f'== {name}')
            fn()
