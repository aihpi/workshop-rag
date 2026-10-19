"""Answer evaluation of w3_03: Answer Correctness and Faithfulness, opened up.

RAGAS returns only the final score. This module calls the same prompts the two
metrics use internally (claim decomposition, TP/FP/FN classification,
faithfulness verdicts) and keeps every intermediate step, so the notebook can
show a calculation that adds up to the score it shows. It also generates
answers with a given system prompt for the prompt experiment of the notebook.

`tools/precompute_w3_03.py` runs everything once and writes data/cache/w3_03/ (gold set)
and data/cache/w3_03/kisz/ (answers about the KISZ website for judging by hand);
the notebook calls `arun_prompt` and `ascore_reference` live only when a participant
presses a button.
"""

import asyncio
import json
import os

from .config import (
    API_BASE_URL,
    DATA_DIR,
    EMBED_MODEL_NAME,
    EVALUATOR_MODEL_NAME,
    RAG_MODEL_NAME,
)

DATASET = '43_komplex'
TOP_K = 5
COLLECTION = 'grundschutz_chunks_json_structured_sections__octen-embedding-8b'
SOURCE = DATA_DIR / 'cache' / f'{DATASET}__json_structured_sections__octen-embedding-8b__gpt-oss-120b'
CACHE = DATA_DIR / 'cache' / 'w3_03'
CANDIDATES_FILE = CACHE / 'candidates.json'     # decompositions of every candidate question
EXAMPLES_FILE = CACHE / 'examples.json'         # the chosen example per role, with chunks; 'R': other-reference task
EXPERIMENT_FILE = CACHE / 'experiment.json'     # three prompts x three questions x three judge runs
OVERVIEW_FILE = CACHE / 'overview.json'         # all four metrics for all 43 questions
TRANSLATIONS_FILE = CACHE / 'translations_en.json'  # German corpus text -> English
REASONS_DE_FILE = CACHE / 'reasons_de.json'     # English judge reasons -> German
KISZ_DIR = CACHE / 'kisz'                        # KISZ website examples for judging by hand (section 3e)
KISZ_EXAMPLES_FILE = KISZ_DIR / 'examples.json'
KISZ_TRANSLATIONS_FILE = KISZ_DIR / 'translations_en.json'

# The system prompt of the cached answers (former version of the generation notebook), and the two prepared alternatives.
PROMPTS = {
    'original': (
        'Du bist ein RAG-Assistent für IT-Grundschutz. Beantworte die Frage nur auf Basis des '
        'bereitgestellten Kontexts. Wenn die Information im Kontext nicht enthalten ist, sage das klar.'
    ),
    'kurz': (
        'Du bist ein RAG-Assistent für IT-Grundschutz. Beantworte die Frage nur auf Basis des '
        'bereitgestellten Kontexts. Wenn die Information im Kontext nicht enthalten ist, sage das klar. '
        'Beantworte die Frage in kurzen Sätzen. Wenn möglich, beantworte die Frage in maximal 1 bis 5 '
        'Sätzen. Je kürzer und klarer deine Antwort ist, desto besser. Gib nur Informationen an, die '
        'relevant für die Frage sind.'
    ),
    'offen': (
        'Du bist ein Experte für IT-Sicherheit und IT-Grundschutz. Beantworte die Frage ausführlich und '
        'umfassend. Nutze den bereitgestellten Kontext als Ausgangspunkt, aber ergänze dein '
        'Expertenwissen, um eine möglichst vollständige Antwort zu geben. Nenne auch Best Practices und '
        'Empfehlungen, die über den Kontext hinausgehen.'
    ),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_gold() -> list[dict[str, str]]:
    """The 43 questions with reference answer ('answer') and Fundstelle."""
    from .evaluate import load_gold as _load

    return _load(DATASET)


def load_source() -> dict:
    """The cached answers and the four metric lists of the 43_komplex run at TOP_K 5."""
    def _read(name):
        return json.loads((SOURCE / name).read_text(encoding='utf-8'))

    retrieval = _read(f'retrieval_eval_top{TOP_K}.json')
    return {
        'answers': _read(f'generation_responses_top{TOP_K}.json'),
        'answer_correctness': _read(f'answer_correctness_top{TOP_K}.json'),
        'faithfulness': _read(f'faithfulness_top{TOP_K}.json'),
        'context_precision': retrieval['context_precision'],
        'context_recall': retrieval['context_recall'],
    }


def load(path) -> dict | list:
    return json.loads(path.read_text(encoding='utf-8'))


def load_json_or_empty(path) -> dict:
    return load(path) if path.exists() else {}


def cache_ok() -> bool:
    return all(p.exists() for p in (EXAMPLES_FILE, EXPERIMENT_FILE, OVERVIEW_FILE, KISZ_EXAMPLES_FILE))


# ---------------------------------------------------------------------------
# Pipeline: retrieval and answer
# ---------------------------------------------------------------------------

def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """The top_k chunks of the Kompendium collection, with page hint."""
    from .search import rag_search

    hits = rag_search(question, top_k=top_k, collection=COLLECTION, model=EMBED_MODEL_NAME)
    return [{'text': h['text'], 'pages': h['citation_hint'], 'chunk_id': h['chunk_id']} for h in hits]


async def agenerate(question: str, contexts: list[str], system_prompt: str) -> str:
    """Answer from the chunks, with the user message format of the cached answers."""
    from litellm import acompletion

    block = '\n\n'.join(f'[Quelle {i + 1}]\n{c}' for i, c in enumerate(contexts))
    resp = await acompletion(
        model=RAG_MODEL_NAME,
        messages=[{'role': 'system', 'content': system_prompt},
                  {'role': 'user', 'content': f'Frage:\n{question}\n\nKontext:\n{block}'}],
        api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.2, max_tokens=10000, timeout=180,
    )
    return (resp.choices[0].message.content or '').strip()


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def metrics():
    """Answer Correctness and Faithfulness with the judge model of config.py."""
    from openai import AsyncOpenAI
    from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerCorrectness, Faithfulness

    key = os.getenv('OPENAI_API_KEY')
    judge = llm_factory(EVALUATOR_MODEL_NAME, client=AsyncOpenAI(api_key=key, base_url=API_BASE_URL),
                        max_tokens=32768)
    emb = LiteLLMEmbeddings(model=EMBED_MODEL_NAME, api_key=key, api_base=API_BASE_URL,
                            encoding_format='float')
    return AnswerCorrectness(llm=judge, embeddings=emb), Faithfulness(llm=judge)


async def adecompose_ac(ac, question: str, answer: str, reference: str) -> dict:
    """Answer Correctness step by step: claims, TP/FP/FN with reasons, F1, similarity, score."""
    ans_claims, ref_claims = await asyncio.gather(ac._generate_statements(question, answer),
                                                  ac._generate_statements(question, reference))
    cls = await ac._classify_statements(question, ans_claims, ref_claims)
    f1 = ac._compute_f1_score(cls)
    sim = await ac._calculate_similarity(answer, reference)
    w_f1, w_sim = ac.weights
    return {
        'answer_claims': list(ans_claims), 'reference_claims': list(ref_claims),
        'TP': [{'statement': s.statement, 'reason': s.reason} for s in cls.TP],
        'FP': [{'statement': s.statement, 'reason': s.reason} for s in cls.FP],
        'FN': [{'statement': s.statement, 'reason': s.reason} for s in cls.FN],
        'f1': f1, 'similarity': sim, 'score': (w_f1 * f1 + w_sim * sim) / (w_f1 + w_sim),
    }


async def ascore_reference(question: str, answer: str, answer_claims: list[str], reference: str) -> dict:
    """Answer Correctness of one answer against a new reference, live.

    The claims of the answer come from the cached decomposition, so only the reference changes:
    the judge splits the new reference, sorts the claims into TP, FP and FN, and the embeddings
    give the similarity. Raises on any API error, so the caller can show its sentence.
    """
    ac, _ = metrics()
    ref_claims = await ac._generate_statements(question, reference)
    if not ref_claims:  # RAGAS counts this as a perfect match; say so instead of hiding it
        cls, f1 = None, 1.0
    else:
        cls = await ac._classify_statements(question, answer_claims, ref_claims)
        f1 = ac._compute_f1_score(cls)
    sim = await ac._calculate_similarity(answer, reference)
    w_f1, w_sim = ac.weights
    return {
        'reference_claims': list(ref_claims),
        'TP': len(cls.TP) if cls else 0, 'FP': len(cls.FP) if cls else 0, 'FN': len(cls.FN) if cls else 0,
        'f1': f1, 'similarity': sim, 'score': (w_f1 * f1 + w_sim * sim) / (w_f1 + w_sim),
    }


async def adecompose_faith(faith, question: str, answer: str, contexts: list[str]) -> dict:
    """Faithfulness step by step: claims of the answer, verdict and reason per claim, score."""
    claims = await faith._create_statements(question, answer)
    if not claims:
        return {'claims': [], 'score': None}
    verdicts = await faith._create_verdicts(claims, '\n'.join(contexts))
    rows = [{'statement': v.statement, 'verdict': int(v.verdict), 'reason': v.reason}
            for v in verdicts.statements]
    return {'claims': rows, 'score': sum(r['verdict'] for r in rows) / len(rows) if rows else None}


async def ascore(ac, faith, question: str, answer: str, reference: str, contexts: list[str]) -> dict:
    """Both scores as RAGAS computes them (one judge run)."""
    a, f = await asyncio.gather(ac.ascore(user_input=question, response=answer, reference=reference),
                                faith.ascore(user_input=question, response=answer, retrieved_contexts=contexts))
    return {'ac': float(a.value), 'faithfulness': float(f.value)}


async def arun_prompt(system_prompt: str, rows: list[dict], progress=None) -> list[dict]:
    """One live run: an answer per question with the prompt, then both scores once.

    `rows` carry question, reference and contexts. `progress(i)` is called after
    each question. Raises on any API error, so the caller can show its sentence.
    """
    ac, faith = metrics()
    out = []
    for i, r in enumerate(rows):
        answer = await agenerate(r['question'], r['contexts'], system_prompt)
        out.append({'answer': answer, **await ascore(ac, faith, r['question'], answer, r['reference'], r['contexts'])})
        if progress is not None:
            progress(i + 1)
    return out
