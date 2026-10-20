"""Build the cache of w3_04: pairwise preferences of the judge for one KISZ preference pair.

Run from `notebooks/`, stage by stage or all at once (each stage reuses what exists):

    uv run python -m tools.precompute_w3_04 --stage pair        # copy the pair from data/cache/w3_03/kisz
    uv run python -m tools.precompute_w3_04 --stage plain       # judge without principles, both orders, RUNS each
    uv run python -m tools.precompute_w3_04 --stage principle   # judge with the example principles, both orders
    uv run python -m tools.precompute_w3_04 --stage translate   # English for the judge's reasons
    uv run python -m tools.precompute_w3_04                     # all stages

The pair is the KISZ question "Welche Angebote bietet das KI-Servicezentrum an?": the short answer
of notebook 1 (13 words) and the fuller answer of notebook 3, section B (57 words), both from the
same three chunks. The judge (gpt-oss-120b, AISC API) sees what participants see: the question,
the line of the website and the two answers, labeled A and B. It picks the better one and one
reason from the same list participants use. Every run is stored, also when the judge contradicts
itself across orders; the notebook shows the counts as they are.

API calls: plain 2 orders x 5 runs = 10, principle 2 orders x 3 runs = 6, translation 1 to 2.
The key comes from notebooks/.env (via ragkit.config) and is never printed.
"""

import argparse
import asyncio
import json
import os

from ragkit.config import API_BASE_URL, DATA_DIR, RAG_MODEL_NAME
from tools.precompute_w3_01 import _translate_batch

CACHE = DATA_DIR / 'cache' / 'w3_04'
PAIR_FILE = CACHE / 'pair.json'
VERDICTS_FILE = CACHE / 'verdicts.json'
TRANSLATIONS_FILE = CACHE / 'translations_en.json'
W3_03_KISZ = DATA_DIR / 'cache' / 'w3_03' / 'kisz' / 'examples.json'

RUNS_PLAIN = 5
RUNS_PRINCIPLE = 3
TEMPERATURE = 0.7  # the judge's sampling temperature; stored in the cache and named in the notebook's FAQ

REASONS = ['richtiger', 'vollständiger', 'kürzer', 'belegter']

SYSTEM_PLAIN = (
    'Sie beurteilen Antworten eines Assistenten. Sie sehen eine Frage, die passende Stelle der Webseite '
    'und zwei Antworten, A und B. Entscheiden Sie, welche Antwort besser ist.\n\n'
    'Antworten Sie nur mit einem JSON-Objekt, ohne weiteren Text:\n'
    '{"besser": "A" oder "B", "grund": einer von "richtiger", "vollständiger", "kürzer", "belegter", '
    '"begruendung": ein bis zwei Sätze auf Deutsch}'
)

# Example principles for the extra task, in the style of Constitutional AI (two short rules).
PRINCIPLES_DE = [
    'Bevorzugen Sie die Antwort, die nur sagt, was in der Quelle steht.',
    'Bevorzugen Sie bei gleichem Inhalt die kürzere Antwort.',
]
PRINCIPLES_EN = [
    'Prefer the answer that says only what the source says.',
    'With the same content, prefer the shorter answer.',
]


def _system_principle() -> str:
    rules = '\n'.join(f'{i + 1}. {p}' for i, p in enumerate(PRINCIPLES_DE))
    return SYSTEM_PLAIN.replace('Entscheiden Sie, welche Antwort besser ist.',
                                f'Entscheiden Sie nach diesen Grundsätzen, welche Antwort besser ist:\n{rules}')


def _write(path, data) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# Stage pair
# ---------------------------------------------------------------------------

def stage_pair() -> None:
    """The preference pair from the w3_03 KISZ cache, with the scores RAGAS gave both answers there."""
    b = _read(W3_03_KISZ)['B']
    pair = {
        'question': b['question'],
        'reference': b['reference'],
        'source_ids': b['source_ids'],
        'source_lines': b['source_lines'],
        'short': {'answer': b['short']['answer'], 'words': b['short']['words_answer'],
                  'answer_correctness': b['short']['answer_correctness'],
                  'faithfulness': b['short']['faithfulness_score'], 'origin': 'w3_01'},
        'full': {'answer': b['answer'], 'words': b['words_answer'],
                 'answer_correctness': b['answer_correctness'],
                 'faithfulness': b['faithfulness_score'], 'origin': 'w3_03 B'},
    }
    _write(PAIR_FILE, pair)
    print(f'pair: {pair["short"]["words"]} vs {pair["full"]["words"]} words')


# ---------------------------------------------------------------------------
# Judge stages
# ---------------------------------------------------------------------------

def _user_message(pair: dict, first: str) -> str:
    second = 'full' if first == 'short' else 'short'
    source = '\n'.join(pair['source_lines'])
    return (f'Frage:\n{pair["question"]}\n\nStelle auf der Webseite:\n{source}\n\n'
            f'Antwort A:\n{pair[first]["answer"]}\n\nAntwort B:\n{pair[second]["answer"]}')


def _parse(raw: str) -> dict | None:
    raw = raw[raw.find('{'):raw.rfind('}') + 1]
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if out.get('besser') not in ('A', 'B') or out.get('grund') not in REASONS:
        return None
    # the judge writes the key as 'begruendung' or 'begründung'
    text = next((str(v) for k, v in out.items() if k.lower().startswith('begr')), '')
    if not text.strip():
        return None
    out['begruendung'] = text
    return out


async def _judge_once(system: str, user: str) -> dict:
    from litellm import acompletion

    for _ in range(3):
        resp = await acompletion(
            model=RAG_MODEL_NAME,
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'),
            temperature=TEMPERATURE, max_tokens=4000, timeout=180,
        )
        out = _parse((resp.choices[0].message.content or '').strip())
        if out is not None:
            return out
    raise SystemExit('the judge did not return valid JSON three times')


async def _judge_stage(system: str, runs: int) -> dict:
    """`runs` verdicts per order; each stored with the winner as 'short' or 'full'."""
    pair = _read(PAIR_FILE)
    out = {}
    for first in ('short', 'full'):
        user = _user_message(pair, first)
        second = 'full' if first == 'short' else 'short'
        rows = await asyncio.gather(*(_judge_once(system, user) for _ in range(runs)))
        out[f'{first}_first'] = [{'label': r['besser'], 'winner': first if r['besser'] == 'A' else second,
                                  'reason': r['grund'], 'text': r['begruendung'].strip()} for r in rows]
        print(f'{first} as A:', [r['winner'] for r in out[f'{first}_first']])
    return out


def stage_plain() -> None:
    data = _read(VERDICTS_FILE) if VERDICTS_FILE.exists() else {}
    data.update({'model': RAG_MODEL_NAME.split('/')[-1], 'temperature': TEMPERATURE,
                 'prompt_plain': SYSTEM_PLAIN, 'plain': asyncio.run(_judge_stage(SYSTEM_PLAIN, RUNS_PLAIN))})
    _write(VERDICTS_FILE, data)


def stage_principle() -> None:
    data = _read(VERDICTS_FILE) if VERDICTS_FILE.exists() else {}
    system = _system_principle()
    data.update({'principles_de': PRINCIPLES_DE, 'principles_en': PRINCIPLES_EN, 'prompt_principle': system,
                 'principle': asyncio.run(_judge_stage(system, RUNS_PRINCIPLE))})
    _write(VERDICTS_FILE, data)


def stage_translate() -> None:
    """English for every judge reason; the German text stays as the judge wrote it."""
    data = _read(VERDICTS_FILE)
    texts = [v['text'] for part in ('plain', 'principle') for rows in data[part].values() for v in rows]
    done = _read(TRANSLATIONS_FILE) if TRANSLATIONS_FILE.exists() else {}
    todo = [x for x in dict.fromkeys(texts) if x not in done]
    for i in range(0, len(todo), 15):
        batch = todo[i:i + 15]
        done.update(zip(batch, _translate_batch(batch)))
    _write(TRANSLATIONS_FILE, done)
    print(f'translations: {len(done)}')


STAGES = {'pair': stage_pair, 'plain': stage_plain, 'principle': stage_principle, 'translate': stage_translate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--stage', choices=list(STAGES))
    args = parser.parse_args()
    for name in ([args.stage] if args.stage else STAGES):
        STAGES[name]()


if __name__ == '__main__':
    main()
