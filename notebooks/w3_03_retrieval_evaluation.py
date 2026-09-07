# ruff: noqa: PLR1711,ASYNC230,SIM118  marimo returns; remaining codes are legacy notebook code kept as is
import marimo

__generated_with = "0.24.0"
app = marimo.App(css_file="ragkit/theme.css", html_head_file="ragkit/head.html")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo, theme):
    mo.Html(theme.header_html('RAG III'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 — Retrieval Evaluation

    Dieses Notebook evaluiert die **Retrieval-Qualität** unseres RAG-Systems in Isolation —
    bevor eine Antwort generiert wird.

    1. Ground-Truth-Daten laden (CSV mit kuratierten Fragen & Referenzantworten)
    2. Fragen einbetten und relevante Chunks aus Qdrant abrufen
    3. **Context Precision** messen — Stehen relevante Chunks oben im Ranking?
    4. **Context Recall** messen — Deckt das Retrieval alle Informationen der Referenzantwort ab?
    5. Ergebnisse inspizieren
    6. Experiment: Vergleich verschiedener `TOP_K`-Werte
    7. Optional: Experimente mit verschiedenen RAG Parametern (Embedding Modell, Chunking Methode, ...)

    > **Keine Antwortgenerierung** in diesem Notebook — wir bewerten nur das Retrieval.

    **Voraussetzung:** `w3_02_ingestion.ipynb` muss erfolgreich durchgelaufen sein (Qdrant-Collection befüllt).
    """)
    return


@app.cell(hide_code=True)
def _():
    # Gemeinsame Konfiguration laden (Pfade, Modellnamen, Qdrant-Einstellungen)
    from ragkit import theme
    from ragkit.config import (
        API_BASE_URL,
        CACHE_DIR,
        CHUNKING_MODE,
        COLLECTION_NAME,
        CSV_PATH,
        CSV_SEP,
        DATASET,
        EMBED_SHORT,
        EVALUATOR_MODEL_NAME,
        EVALUATOR_SHORT,
        FIGURES_DIR,
        QDRANT_HOST,
        QDRANT_PORT,
        TOP_K,
        setup,
    )
    env = setup(strict=False)
    theme.apply_mpl()

    # --- Imports ---
    import json
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    from openai import AsyncOpenAI
    from qdrant_client import QdrantClient
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision, ContextRecall
    from ragas.metrics.collections.context_precision.metric import (
        ContextPrecisionInput,
        ContextPrecisionOutput,
    )
    from ragas.metrics.collections.context_recall.util import (
        ContextRecallInput,
        ContextRecallOutput,
    )

    return (
        API_BASE_URL,
        AsyncOpenAI,
        CACHE_DIR,
        CHUNKING_MODE,
        COLLECTION_NAME,
        CSV_PATH,
        CSV_SEP,
        ContextPrecision,
        ContextPrecisionInput,
        ContextPrecisionOutput,
        ContextRecall,
        ContextRecallInput,
        ContextRecallOutput,
        DATASET,
        EMBED_SHORT,
        EVALUATOR_MODEL_NAME,
        EVALUATOR_SHORT,
        FIGURES_DIR,
        QDRANT_HOST,
        QDRANT_PORT,
        QdrantClient,
        TOP_K,
        env,
        json,
        llm_factory,
        os,
        pd,
        plt,
        theme,
    )


@app.cell(hide_code=True)
def _(env, mo):
    mo.md(env.summary_md())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Ground Truth laden

    Wir verwenden einen von Fachexperten kuratierten Datensatz mit rund 40 Fragen zum IT-Grundschutz-Kompendium.
    Jede Frage hat:
    - **Frage** → `user_input` — Was wird an das System gestellt?
    - **Antwort** → `reference` — Die korrekte Referenzantwort
    - **Fundstellen** → Die relevanten Passagen im Originaldokument (für manuelle Inspektion)
    """)
    return


@app.cell(hide_code=True)
def _(CSV_PATH, CSV_SEP, DATASET, pd):
    # CSV einlesen (Datensatz und Trennzeichen aus ragkit/config.py)
    df = pd.read_csv(CSV_PATH, sep=CSV_SEP)

    questions = df['Frage'].tolist()       # Alle Fragen als Liste von Strings
    references = df['Antwort'].tolist()    # Zugehörige Referenzantworten als Liste von Strings

    print(f'{len(df)} Fragen geladen (Datensatz: {DATASET}).')
    print(f'Spalten: {list(df.columns)}\n')
    df.head(3)
    return questions, references


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Retrieval: Kontexte aus Qdrant abrufen

    Für jede Frage wird:
    1. Die Frage mit dem Embedding-Modell in einen Vektor umgewandelt
    2. Per Cosine Similarity die `TOP_K` ähnlichsten Chunks aus Qdrant abgerufen

    Die Qdrant-Collection wurde in `w3_02_ingestion.ipynb` befüllt.
    """)
    return


@app.cell(hide_code=True)
def _(
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QdrantClient,
    TOP_K,
    questions,
):
    from ragkit.embed import embed
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # Verbindung zu Qdrant herstellen (Collection wurde in Notebook 02 befüllt)
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f'Qdrant-Collection "{COLLECTION_NAME}": {info.points_count} Vektoren\n')
    print(f'Embedding {len(questions)} Fragen...')
    question_vectors = embed(questions)
    # Fragen in Vektoren umwandeln (gleicher Embedding-Raum wie die Chunks)
    print(f'Retrieving top-{TOP_K} Kontexte pro Frage...')
    retrieved_contexts = []
    for _q_vec in question_vectors:
    # Für jede Frage die TOP_K ähnlichsten Chunks aus Qdrant per Cosine-Similarity abrufen
        _response = qdrant.query_points(collection_name=COLLECTION_NAME, query=_q_vec, limit=TOP_K, with_payload=True)
        contexts = [hit.payload['text'] for hit in _response.points]
        retrieved_contexts.append(contexts)
    print(f'Fertig. {len(retrieved_contexts)} Fragen mit je {TOP_K} Kontexten.')  # Frage-Vektor als Suchanfrage  # Anzahl der Treffer  # Chunk-Text im Ergebnis mitliefern  # Nur den Textinhalt der Treffer extrahieren (aus dem Payload)
    return qdrant, question_vectors, retrieved_contexts


@app.cell(hide_code=True)
def _(questions, references, retrieved_contexts):
    # Sanity Check: erste Frage und ihre Top-Treffer
    print('=' * 120 + '')
    print(f'Frage: {questions[0]}')
    print(f'Referenzantwort: {references[0][:120]}...')
    print('=' * 120 + '\n')
    for _i, _ctx in enumerate(retrieved_contexts[0]):
        print(f'--------------- Top-{_i + 1} Chunk: ---------------')
        print(f"{_ctx[:120]}{('...' if len(_ctx) > 120 else '')}")
        print()
        if _i == 2:  # only print out the first three 
            break
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wie bewertet RAGAS? — LLM-as-Judge

    Bevor wir die einzelnen Metriken betrachten, ist es wichtig zu verstehen, **wie** RAGAS
    seine Bewertungen durchführt. RAGAS verwendet ein **Evaluator-LLM als Richter**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Context Precision — Sind die abgerufenen Chunks relevant?

    Misst, ob relevante Chunks weiter **oben** im Ranking stehen als irrelevante.
    Für jeden abgerufenen Chunk fragt das Evaluator-LLM: *„Ist dieser Chunk nützlich,
    um die Frage zu beantworten?"* (Verdict: 1=Ja, 0=Nein)

    Daraus wird **Mean Average Precision** berechnet (relevante Chunks weiter oben = höherer Score):

    | Position | Relevant? | Precision@K |
    |----------|-----------|-------------|
    | 1        | Ja        | 1/1 = 1.0   |
    | 2        | Nein      | —           |
    | 3        | Ja        | 2/3 = 0.67  |

    → Context Precision = (1.0 + 0.67) / 2 = **0.84**

    **Benötigte Felder:** `user_input`, `retrieved_contexts`, `reference`
    """)
    return


@app.cell(hide_code=True)
async def _(
    API_BASE_URL,
    AsyncOpenAI,
    ContextPrecision,
    ContextPrecisionInput,
    ContextPrecisionOutput,
    ContextRecall,
    EVALUATOR_MODEL_NAME,
    llm_factory,
    os,
    questions,
    references,
    retrieved_contexts,
):
    # Evaluator-LLM einrichten
    evaluator_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=API_BASE_URL)
    evaluator_llm = llm_factory(EVALUATOR_MODEL_NAME, client=evaluator_client, max_tokens=8192)
    context_precision = ContextPrecision(llm=evaluator_llm)
    context_recall = ContextRecall(llm=evaluator_llm)
    _demo_idx = 3
    print(f'Frage:    {questions[_demo_idx]}')
    print(f'Referenz: {references[_demo_idx][:120]}...\n')
    verdicts = []
    for _i, _ctx in enumerate(retrieved_contexts[_demo_idx]):
        _input_data = ContextPrecisionInput(question=questions[_demo_idx], context=_ctx, answer=references[_demo_idx])
    # Metriken initialisieren
        _prompt_str = context_precision.prompt.to_string(_input_data)
        _result = await evaluator_llm.agenerate(_prompt_str, ContextPrecisionOutput)
        verdicts.append(_result.verdict)
    # ---------------------------------------------------------------------------
    # Demo: Context Precision für eine einzelne Frage — Blick unter die Haube
        _icon = '✓ relevant' if _result.verdict else '✗ irrelevant'
    # Für jeden abgerufenen Chunk fragt das Evaluator-LLM:
    # "Ist dieser Chunk nützlich, um die Frage zu beantworten?" (Verdict: 1=Ja, 0=Nein)
        print(100 * '-')
        print(f'  Chunk {_i + 1}: [{_icon}]')
        print(f'           Grund: {_result.reason}')
    print('=' * 100)
    print(f'Verdicts: {verdicts}\n')
    relevant_precisions = []
    # Jeden Chunk einzeln bewerten
    for k in range(len(verdicts)):
        if verdicts[k] == 1:
            precision_at_k = sum(verdicts[:k + 1]) / (k + 1)
            relevant_precisions.append(precision_at_k)
            print(f'  Position {k + 1}: relevant → Precision@{k + 1} = {sum(verdicts[:k + 1])}/{k + 1} = {precision_at_k:.3f}')
        else:
            print(f'  Position {k + 1}: irrelevant → wird übersprungen')
    if relevant_precisions:
        _avg_precision = sum(relevant_precisions) / len(relevant_precisions)
        terms = ' + '.join(f'{p:.2f}' for p in relevant_precisions)
        print(f'\n  Context Precision = ({terms}) / {len(relevant_precisions)} = {_avg_precision:.2f}')
    else:
        print('\n  Kein relevanter Chunk gefunden → Context Precision = 0.0')
    score_automated = context_precision._calculate_average_precision(verdicts)
    # --- Average Precision Schritt für Schritt berechnen ---
    # Formel: Für jede Position k mit Verdict=1 berechne Precision@k,
    # dann Durchschnitt über alle relevanten Positionen.
    # calculate context precision manually
    # Zur Kontrolle: RAGAS-interne Berechnung mit der gleichen Funktion
    print(f'  Kontrolle (RAGAS intern): {score_automated:.2f}')
    return context_precision, context_recall, evaluator_llm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Bevor wir zur nächsten Metrik übergehen können, müssen wir zunächst erklären was "claims" in RAGAS sind.

    ### Claims: Die Grundeinheit der Bewertung

    Fast alle RAGAS-Metriken basieren auf dem Konzept von **Claims** (Aussagen).
    Das Evaluator-LLM zerlegt einen Text in atomare, eigenständige Aussagen. Jede Aussage sagt möglichst
    einen isolierten Fakt aus, sodass sie einzeln verständlich ist.

    **Beispiel:** Der Evaluator erhält folgenden Text:

    > *„Der ISB erstellt die Sicherheitsleitlinie und koordiniert Maßnahmen mit den Fachabteilungen."*

    Daraus erzeugt er zwei Claims:
    1. *„Der ISB erstellt die Sicherheitsleitlinie."*
    2. *„Der ISB koordiniert Maßnahmen mit den Fachabteilungen."*

    Anschließend wird für jede Aussage geprüft, ob sie durch einen anderen Text belegt ist
    (z.B. durch die abgerufenen Chunks oder die Referenzantwort — je nach Metrik).

    ### Warum das wichtig ist

    - Es gibt **keinen Algorithmus** hinter der Zerlegung — das Evaluator-LLM entscheidet,
      was eine „atomare Aussage" ist und ob sie belegt ist.
    - **Verschiedene Evaluator-Modelle liefern verschiedene Ergebnisse:** Ein stärkeres Modell
      erkennt feinere Unterschiede und bewertet strenger. Ein schwächeres Modell ist unkritischer und produziert daher eher höhere Scores.
    - Die **Anzahl der Claims** beeinflusst den Score direkt: Wenn das LLM einen Satz in 3 statt 5
      Claims zerlegt, hat eine einzelne unbelegte Aussage mehr Gewicht (1/3 vs. 1/5).
    - RAGAS-Scores sind daher **keine objektiven Messwerte**, sondern **LLM-basierte Urteile** —
      vergleichbar nur, wenn dasselbe Evaluator-Modell verwendet wurde. Und selbst dann unterscheiden sich Scores bei erneuter Berechnung da die Identifikation der claims nicht deterministisch ist.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Context Recall — Deckt das Retrieval alle Informationen ab?

    Misst, ob die abgerufenen Chunks genug Information enthalten, um die Referenzantwort zu belegen.

    **So funktioniert die Metrik:**
    1. Das Evaluator-LLM zerlegt die **Referenzantwort** in atomare Claims
    2. Für jeden Claim prüft es: *„Kann diese Aussage den abgerufenen Chunks zugeordnet werden?"*
    3. Score = Anzahl belegter Claims / Gesamtanzahl Claims

    $$ \text{Context Recall} = \frac{\text{Anzahl Referenz-Claims, die in den Chunks belegt sind}}{\text{Gesamtanzahl Referenz-Claims}} $$

    **Benötigte Felder:** `user_input`, `retrieved_contexts`, `reference`

    ### Zusammenspiel von Context Precision und Context Recall

    | | Precision hoch | Precision niedrig |
    |---|---|---|
    | **Recall hoch** | Ideal: relevante Chunks, gut gerankt | Viele Chunks nötig, aber alles abgedeckt |
    | **Recall niedrig** | Wenige, gute Chunks — aber Info fehlt | Schlechtes Retrieval insgesamt |
    """)
    return


@app.cell(hide_code=True)
async def _(
    ContextRecallInput,
    ContextRecallOutput,
    context_recall,
    evaluator_llm,
    questions,
    references,
    retrieved_contexts,
):
    _demo_idx = 4
    context_str = '\n'.join(retrieved_contexts[_demo_idx])
    print(f'Frage:    {questions[_demo_idx]}')
    print(f'Referenz: {references[_demo_idx][:120]}...\n')
    _input_data = ContextRecallInput(question=questions[_demo_idx], context=context_str, answer=references[_demo_idx])
    _prompt_str = context_recall.prompt.to_string(_input_data)
    _result = await evaluator_llm.agenerate(_prompt_str, ContextRecallOutput)
    n_attributed = sum(1 for c in _result.classifications if c.attributed)
    print(f'Die Referenzantwort wurde in {len(_result.classifications)} Claims zerlegt:\n')
    for _i, c in enumerate(_result.classifications):
        _icon = '✓ belegt' if c.attributed else '✗ NICHT belegt'
        print(f'  {_i + 1}. [{_icon}] {c.statement}')
        print(f'     Grund: {c.reason}')
        print()
    print(f'Context Recall = {n_attributed}/{len(_result.classifications)} = {n_attributed / len(_result.classifications):.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Was passiert wenn wir die obrige Codezelle erneut ausführen? Bleibt die Anzahl der claims gleich?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nun können wir Context Precision und Recall für alle Fragen im Datensatz berechnen.
    """)
    return


@app.cell(hide_code=True)
async def _(
    CACHE_DIR,
    TOP_K,
    context_precision,
    context_recall,
    json,
    pd,
    questions,
    references,
    retrieved_contexts,
):
    # ---------------------------------------------------------------------------
    # Batch-Evaluation: Context Precision + Context Recall für alle Fragen
    LOAD_FROM_CACHE = True
    # Evaluator-LLM und Metriken sind bereits oben initialisiert.
    CACHE_PATH = CACHE_DIR / f'retrieval_eval_top{TOP_K}.json'
    if LOAD_FROM_CACHE and CACHE_PATH.exists():
        with open(CACHE_PATH, 'r') as _f:
            cached = json.load(_f)
        _cp_scores = cached['context_precision']
        _cr_scores = cached['context_recall']
        print(f'Ergebnisse aus Cache geladen: {CACHE_PATH.name}')
    else:
        _inputs = [{'user_input': questions[_i], 'reference': references[_i], 'retrieved_contexts': retrieved_contexts[_i]} for _i in range(len(questions))]
        print(f'Evaluiere Context Precision für {len(questions)} Fragen (parallel)...')
        _cp_results = await context_precision.abatch_score(_inputs)
        _cp_scores = [float(r.value) for r in _cp_results]
        print(f'Evaluiere Context Recall für {len(questions)} Fragen (parallel)...')
        _cr_results = await context_recall.abatch_score(_inputs)
        _cr_scores = [float(r.value) for r in _cr_results]
        with open(CACHE_PATH, 'w') as _f:
            json.dump({'context_precision': _cp_scores, 'context_recall': _cr_scores}, _f)
        print(f'Ergebnisse gespeichert: {CACHE_PATH.name}')
    result_df = pd.DataFrame({'user_input': questions, 'reference': references, 'context_precision': _cp_scores, 'context_recall': _cr_scores})
    result_df['retrieved_contexts'] = retrieved_contexts
    print(f"\nContext Precision (Durchschnitt): {result_df['context_precision'].mean():.4f}")
    # DataFrame aufbauen
    print(f"Context Recall (Durchschnitt):    {result_df['context_recall'].mean():.4f}")
    return LOAD_FROM_CACHE, result_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Ergebnisse inspizieren
    """)
    return


@app.cell(hide_code=True)
def _(
    CHUNKING_MODE,
    DATASET,
    EMBED_SHORT,
    EVALUATOR_SHORT,
    FIGURES_DIR,
    TOP_K,
    plt,
    result_df,
):
    # Histogramme für beide Metriken nebeneinander
    metric_cols = ['context_precision', 'context_recall']
    _fig, _axes = plt.subplots(2, 1, figsize=(9, 5))
    for _ax, _col in zip(_axes, metric_cols):
        _ax.hist(result_df[_col], bins=40, edgecolor='black')
        _ax.set_xlabel('Score')
        _ax.set_ylabel('Count')
        mean_score = result_df[_col].mean()
        _ax.set_title(f'{_col}')
        _ax.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.3f}')
        _ax.legend()
    _fig.suptitle(f'{EMBED_SHORT} | {CHUNKING_MODE} | {DATASET} | top_k={TOP_K} | eval={EVALUATOR_SHORT}', fontsize=13, y=1.03)
    plt.tight_layout()
    _fig_dir = FIGURES_DIR / '03_retrieval_evaluation'
    _fig_dir.mkdir(parents=True, exist_ok=True)
    _fig_path = _fig_dir / f'retrieval_hist__{EMBED_SHORT}__{CHUNKING_MODE}__{DATASET}__top{TOP_K}__{EVALUATOR_SHORT}.png'
    _fig.savefig(_fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {_fig_path}')
    # Figur speichern
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) Experiment: TOP_K systematisch variieren

    Was passiert mit Context Precision und Context Recall, wenn wir `TOP_K` verändern?

    - **Precision** sinkt tendenziell mit höherem TOP_K (mehr irrelevante Chunks im Ranking)
    - **Recall** steigt tendenziell mit höherem TOP_K (mehr Kontext = mehr Informationsabdeckung)

    Die optimale Wahl von TOP_K balanciert beide Metriken.

    > **Laufzeit:** ~2 Minuten pro TOP_K-Wert (4 Werte × 2 Metriken). Mit `LOAD_FROM_CACHE = True` sofort.
    """)
    return


@app.cell(hide_code=True)
async def _(
    CACHE_DIR,
    COLLECTION_NAME,
    LOAD_FROM_CACHE,
    context_precision,
    context_recall,
    json,
    qdrant,
    question_vectors,
    questions,
    references,
):
    # --- TOP_K-Werte für das Experiment ---
    TOP_K_VALUES = [1, 3, 5, 10]
    EXPERIMENT_CACHE = CACHE_DIR / 'retrieval_experiment_topk.json'  # CACHE_DIR enthält bereits Datensatz + Chunking + Embedding + Evaluator
    if LOAD_FROM_CACHE and EXPERIMENT_CACHE.exists():
        with open(EXPERIMENT_CACHE, 'r') as _f:
            experiment_results = json.load(_f)  # Gespeicherte Experiment-Ergebnisse laden
        print(f'Experiment-Ergebnisse aus Cache geladen: {EXPERIMENT_CACHE.name}')
    else:
        experiment_results = {}
        for _top_k in TOP_K_VALUES:
            print(f'=== TOP_K={_top_k} ===')
            print('  Retrieval...')
            exp_contexts = []
            for _q_vec in question_vectors:
                _response = qdrant.query_points(collection_name=COLLECTION_NAME, query=_q_vec, limit=_top_k, with_payload=True)
                exp_contexts.append([hit.payload['text'] for hit in _response.points])  # Retrieval mit aktuellem TOP_K
            _inputs = [{'user_input': questions[_i], 'reference': references[_i], 'retrieved_contexts': exp_contexts[_i]} for _i in range(len(questions))]
            print('  Context Precision...')
            _cp_results = await context_precision.abatch_score(_inputs)
            _cp_scores = [float(r.value) for r in _cp_results]
            print('  Context Recall...')
            _cr_results = await context_recall.abatch_score(_inputs)
            _cr_scores = [float(r.value) for r in _cr_results]
            experiment_results[str(_top_k)] = {'context_precision': _cp_scores, 'context_recall': _cr_scores}
            avg_cp = sum(_cp_scores) / len(_cp_scores)
            avg_cr = sum(_cr_scores) / len(_cr_scores)
            print(f'  → Precision: {avg_cp:.4f}, Recall: {avg_cr:.4f}\n')
        with open(EXPERIMENT_CACHE, 'w') as _f:  # Eingaben für Batch-Evaluation aufbauen
            json.dump(experiment_results, _f)
        print(f'Experiment-Ergebnisse gespeichert: {EXPERIMENT_CACHE.name}')  # Context Precision bewerten  # Context Recall bewerten  # Ergebnisse speichern
    return TOP_K_VALUES, experiment_results


@app.cell(hide_code=True)
def _(
    CHUNKING_MODE,
    DATASET,
    EMBED_SHORT,
    EVALUATOR_SHORT,
    FIGURES_DIR,
    experiment_results,
    plt,
):
    # Liniendiagramm: Context Precision und Context Recall vs TOP_K
    _fig, _ax = plt.subplots(figsize=(8, 5))
    top_k_vals = [int(k) for k in experiment_results.keys()]
    _avg_precision = [sum(v['context_precision']) / len(v['context_precision']) for v in experiment_results.values()]  # String-Keys → int
    avg_recall = [sum(v['context_recall']) / len(v['context_recall']) for v in experiment_results.values()]
    _ax.plot(top_k_vals, _avg_precision, 'o-', label='Context Precision', linewidth=2, markersize=8)
    _ax.plot(top_k_vals, avg_recall, 's-', label='Context Recall', linewidth=2, markersize=8)
    _ax.set_xlabel('TOP_K')
    _ax.set_ylabel('Mean Score')
    _ax.set_title(f'Context Precision vs. Recall \n {EMBED_SHORT} | {CHUNKING_MODE} | {DATASET} | eval={EVALUATOR_SHORT}')
    _ax.set_xticks(top_k_vals)
    _ax.set_ylim(0, 1)
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig_dir = FIGURES_DIR / '03_retrieval_evaluation'
    _fig_dir.mkdir(parents=True, exist_ok=True)
    _fig_path = _fig_dir / f'topk_precision_recall__{EMBED_SHORT}__{CHUNKING_MODE}__{DATASET}__{EVALUATOR_SHORT}.png'
    _fig.savefig(_fig_path, dpi=150, bbox_inches='tight')
    # Figur speichern
    print(f'Saved: {_fig_path}')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(
    CHUNKING_MODE,
    DATASET,
    EMBED_SHORT,
    EVALUATOR_SHORT,
    FIGURES_DIR,
    TOP_K_VALUES,
    experiment_results,
    plt,
):
    # Histogramm-Grid: eine Zeile pro Metrik, eine Spalte pro TOP_K
    metrics = ['context_precision', 'context_recall']
    n_topk = len(TOP_K_VALUES)
    _fig, _axes = plt.subplots(2, n_topk, figsize=(3 * n_topk, 5))
    for row, metric in enumerate(metrics):
        for _col, _top_k in enumerate(TOP_K_VALUES):
            _ax = _axes[row, _col]
            scores = experiment_results[str(_top_k)][metric]
            _ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
            avg = sum(scores) / len(scores)
            _ax.axvline(avg, color='red', linestyle='--', linewidth=1, label=f'Mean: {avg:.2f}')
            _ax.set_xlim(0, 1.05)
            _ax.set_title(f'TOP_K={_top_k}', fontsize=10)
            _ax.legend()
            if _col == 0:
                _ax.set_ylabel(metric.replace('_', '\n'), fontsize=9)
            if row == 1:
                _ax.set_xlabel('Score', fontsize=9)
    _fig.suptitle(f'Score Distributions \n {EMBED_SHORT} | {CHUNKING_MODE} | {DATASET} | eval={EVALUATOR_SHORT}')
    plt.tight_layout()
    _fig_dir = FIGURES_DIR / '03_retrieval_evaluation'
    _fig_dir.mkdir(parents=True, exist_ok=True)
    _fig_path = _fig_dir / f'topk_hist_grid__{EMBED_SHORT}__{CHUNKING_MODE}__{DATASET}__{EVALUATOR_SHORT}.png'
    _fig.savefig(_fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {_fig_path}')
    # Figur speichern
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diskussion

    - Welcher TOP_K-Wert balanciert Precision und Recall am besten?
    - Welche Aussagekraft hat Context Precision bei TOP_K = 1?
    - Welche Aussagekraft hat Context Recall bei einem hohen TOP_K Wert?
    - Wie verändert sich die Score-Verteilung (nicht nur der Durchschnitt) mit steigendem TOP_K?

    > **Tipp:** In `ragkit/config.py` können die Parameter umgeschaltet werden: Embedding Modell, Chunking Methode, Dataset, Evaluator LLM.

    - was hat den größten Effekt auf die Scores? Embedding Modell? Chunking Methode? Dataset? Evaluator LLM?
    - Was könnte man am Retrieval verbessern?

    **Weiter geht es mit:**
    - `w3_04_generation_evaluation.ipynb` — Generations-Metriken (Faithfulness und Noise Sensitivity)
    """)
    return


if __name__ == "__main__":
    app.run()
