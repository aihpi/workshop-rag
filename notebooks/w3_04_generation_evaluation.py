# ruff: noqa: PLR1711,ASYNC230,F841  marimo returns; remaining codes are legacy notebook code kept as is
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
    # 04 — Generation Evaluation

    Dieses Notebook evaluiert die **Generierungsqualität** unseres RAG-Systems:

    1. Ground-Truth-Daten laden und Kontexte aus Qdrant abrufen
    2. Antworten generieren (RAG-Pipeline)
    3. **Answer Correctness** messen — Ist die Antwort inhaltlich korrekt?
    4. **Faithfulness** messen — Ist die Antwort durch die Kontexte belegt?
    5. Ergebnisse inspizieren und Schwachstellen identifizieren
    6. Experiment: Was passiert, wenn der System-Prompt geändert wird?

    > **Wichtig:** Eine Antwort kann **faithful** sein (alle Aussagen sind im Kontext belegt)
    > und trotzdem **falsch** (die Kontexte enthielten nicht die richtige Information).
    > Answer Correctness und Faithfulness messen daher unterschiedliche Fehlerarten.

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
        EMBED_MODEL_NAME,
        EMBED_SHORT,
        EVALUATOR_MODEL_NAME,
        EVALUATOR_SHORT,
        FIGURES_DIR,
        QDRANT_HOST,
        QDRANT_PORT,
        RAG_MODEL_NAME,
        TOP_K,
        setup,
    )
    env = setup(strict=False)
    theme.apply_mpl()
    import asyncio

    # --- Imports ---
    import json
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython.display import Markdown, display
    from litellm import acompletion
    from openai import AsyncOpenAI
    from qdrant_client import QdrantClient
    from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerCorrectness, Faithfulness
    from ragas.metrics.collections.answer_correctness.metric import (
        StatementGeneratorInput,
        StatementGeneratorOutput,
    )
    # --- Toggle: auf True setzen, um gespeicherte Ergebnisse zu laden ---
    LOAD_FROM_CACHE = True
    return (
        StatementGeneratorOutput,
        API_BASE_URL,
        AnswerCorrectness,
        AsyncOpenAI,
        CACHE_DIR,
        CHUNKING_MODE,
        COLLECTION_NAME,
        CSV_PATH,
        CSV_SEP,
        DATASET,
        EMBED_MODEL_NAME,
        EMBED_SHORT,
        EVALUATOR_MODEL_NAME,
        EVALUATOR_SHORT,
        FIGURES_DIR,
        Faithfulness,
        LOAD_FROM_CACHE,
        list,
        LiteLLMEmbeddings,
        Markdown,
        QDRANT_HOST,
        QDRANT_PORT,
        QdrantClient,
        RAG_MODEL_NAME,
        StatementGeneratorInput,
        TOP_K,
        acompletion,
        asyncio,
        display,
        env,
        json,
        llm_factory,
        np,
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
    ## 1) Ground Truth laden und Kontexte abrufen

    Wir laden den Datensatz wie in Notebook 03 und rufen die Kontexte erneut aus Qdrant ab.
    """)
    return


@app.cell(hide_code=True)
def _(
    COLLECTION_NAME,
    CSV_PATH,
    CSV_SEP,
    QDRANT_HOST,
    QDRANT_PORT,
    QdrantClient,
    TOP_K,
    pd,
):
    from ragkit.embed import embed

    # CSV laden
    df = pd.read_csv(CSV_PATH, sep=CSV_SEP)
    questions = df['Frage'].tolist()       # Alle Fragen als Liste von Strings
    references = df['Antwort'].tolist()    # Zugehörige Referenzantworten
    print(f'{len(df)} Fragen geladen.')

    # Verbindung zu Qdrant herstellen (Collection wurde in Notebook 02 befüllt)
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f'Qdrant-Collection "{COLLECTION_NAME}": {info.points_count} Vektoren\n')

    # Fragen in Vektoren umwandeln (gleicher Embedding-Raum wie die Chunks)
    print(f'Embedding {len(questions)} Fragen...')
    question_vectors = embed(questions)

    # Für jede Frage die TOP_K ähnlichsten Chunks aus Qdrant per Cosine-Similarity abrufen
    print(f'Retrieving top-{TOP_K} Kontexte pro Frage...')
    retrieved_contexts = []
    for q_vec in question_vectors:
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,                    # Frage-Vektor als Suchanfrage
            limit=TOP_K,                    # Anzahl der Treffer
            with_payload=True,              # Chunk-Text im Ergebnis mitliefern
        )
        # Nur den Textinhalt der Treffer extrahieren (aus dem Payload)
        retrieved_contexts.append([hit.payload['text'] for hit in response.points])

    print(f'Fertig. {len(retrieved_contexts)} Fragen mit je {TOP_K} Kontexten.')
    return embed, qdrant, questions, references, retrieved_contexts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Antworten generieren (RAG Generation)
    """)
    return


@app.cell(hide_code=True)
async def _(
    API_BASE_URL,
    COLLECTION_NAME,
    List,
    Markdown,
    RAG_MODEL_NAME,
    TOP_K,
    acompletion,
    display,
    embed,
    os,
    qdrant,
):
    # Das LLM soll **nur** die Informationen aus dem Kontext verwenden.
    from typing import List  # noqa: UP035 - kept for the original annotations
    SYSTEM_PROMPT = 'Du bist ein RAG-Assistent für IT-Grundschutz. Beantworte die Frage nur auf Basis des bereitgestellten Kontexts. Wenn die Information im Kontext nicht enthalten ist, sage das klar.'

    async def generate_rag_answer(question: str, contexts: List[str], system_prompt: str=SYSTEM_PROMPT) -> str:
        """Generate an answer to a question using retrieved contexts via LiteLLM.

        Uses 'async def' so that multiple calls can run concurrently via
        asyncio.gather() — instead of waiting for each API call sequentially,
        all requests are sent at once and processed in parallel.

        The 'await' keyword pauses this function at the API call, allowing
        other concurrent tasks to proceed while waiting for the response.

        Args:
            question: The user's question.
            contexts: List of retrieved text chunks to use as context.
            system_prompt: System prompt for the LLM.

        Returns:
            The generated answer as a string.
        """
        context_block = '\n\n'.join((f'[Quelle {_i + 1}]\n{ctx}' for _i, ctx in enumerate(contexts)))
        resp = await acompletion(model=RAG_MODEL_NAME, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f'Frage:\n{question}\n\nKontext:\n{context_block}'}], api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'), temperature=0.2, max_tokens=10000)
        return resp.choices[0].message.content
    CUSTOM_QUESTION = 'Was ist der IT Grundschutz des BSI?'
    custom_question_vector = embed([CUSTOM_QUESTION])[0]  # Kontexte nummeriert zusammenfügen
    custom_response = qdrant.query_points(collection_name=COLLECTION_NAME, query=custom_question_vector, limit=TOP_K, with_payload=True)
    custom_contexts = [hit.payload['text'] for hit in custom_response.points]
    answer_to_custom_question = await generate_rag_answer(CUSTOM_QUESTION, custom_contexts)
    print()
    print(f'Frage: {CUSTOM_QUESTION}')  # 'await' pausiert hier, bis die API antwortet — andere Tasks laufen weiter
    print('=' * 100)
    display(Markdown(answer_to_custom_question))
    # ---------------------------------------------------------------------------
    # Eigene Frage an das RAG-System stellen
    # Zeigt den kompletten RAG-Ablauf: Frage → Embedding → Retrieval → Generierung
    # Schritt 1: Frage in einen Vektor umwandeln (gleicher Embedding-Raum wie die Chunks)
    # embed() gibt eine Liste von Vektoren zurück — [0] extrahiert den einzelnen Vektor
    # Schritt 2: Die TOP_K ähnlichsten Chunks aus Qdrant per Cosine-Similarity abrufen
    # Schritt 3: Antwort generieren
    # 'await' ist nötig, da generate_rag_answer eine async-Funktion ist —
    # ohne await wird die Funktion nicht ausgeführt, sondern nur ein Coroutine-Objekt erzeugt
    # Schritt 4: Antwort als formatiertes Markdown anzeigen
    print('=' * 100)  # Was passiert bei der Frage "Wofür steht BSI?" ?  # Nur Texte extrahieren
    return (generate_rag_answer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Für jede Frage generieren wir nun eine Antwort auf Basis der abgerufenen Kontexte.
    Das LLM soll **nur** die Informationen aus dem Kontext verwenden.

    > **Laufzeit:** Bei 40 Fragen dauert die Generierung ca. 5–10 Minuten
    > (ein LLM-Aufruf pro Frage, Reasoning-Modell). Daher laden wir wenn möglich aus dem Cache.
    """)
    return


@app.cell(hide_code=True)
async def _(
    CACHE_DIR,
    LOAD_FROM_CACHE,
    Markdown,
    RAG_MODEL_NAME,
    TOP_K,
    asyncio,
    display,
    generate_rag_answer,
    json,
    questions,
    references,
    retrieved_contexts,
):
    # --- Antworten generieren oder aus Cache laden ---
    RESPONSE_CACHE = CACHE_DIR / f'generation_responses_top{TOP_K}.json'
    if LOAD_FROM_CACHE and RESPONSE_CACHE.exists():
        with open(RESPONSE_CACHE, 'r') as _f:
            responses = json.load(_f)
        print(f'Antworten aus Cache geladen: {RESPONSE_CACHE.name} ({len(responses)} Antworten)')
    else:
        print(f'Generiere Antworten für {len(questions)} Fragen (Modell: {RAG_MODEL_NAME}, parallel)...')
        tasks = [generate_rag_answer(q, ctxs) for q, ctxs in zip(questions, retrieved_contexts)]  # Antworten für alle Fragen parallel generieren:
        responses = list(await asyncio.gather(*tasks))  # asyncio.gather() startet alle API-Aufrufe gleichzeitig statt nacheinander.
        with open(RESPONSE_CACHE, 'w') as _f:
            json.dump(responses, _f, ensure_ascii=False)
        print(f'Antworten gespeichert: {RESPONSE_CACHE.name}')
    print()
    question_number = 3  # Antworten speichern
    print(f'Frage: {questions[question_number]}')
    print('=' * 50, 'reference answer', '=' * 50)
    display(Markdown(references[question_number]))
    print('=' * 50, 'generated answer', '=' * 50)
    # Beispiel: Frage, Referenzantwort und generierte Antwort
    display(Markdown(responses[question_number]))
    return question_number, responses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Answer Correctness

    **Answer Correctness** misst, ob die generierte Antwort **inhaltlich korrekt** ist —
    verglichen mit der Referenzantwort aus dem Ground-Truth-Datensatz.

    Die Metrik kombiniert zwei Komponenten (gewichtet 75/25):
    $$ AC = 0.75 \cdot F1 + 0.25 \cdot CS $$

    #### 1. F1 score (75%)
    Das Evaluator-LLM zerlegt sowohl die generierte Antwort als auch die Referenzantwort
    in atomare Aussagen (Claims) und klassifiziert sie:

    - **TP (True Positive)** — Aussage in Antwort UND in Referenz enthalten
    - **FP (False Positive)** — Aussage in Antwort, aber NICHT in Referenz (falsch/extra)
    - **FN (False Negative)** — Aussage in Referenz, aber NICHT in Antwort (fehlend)

    $$F1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

    #### 2. Semantic Similarity CS (25%)
    Embedding-basierte Cosine Similarity zwischen der gesamten Antwort und der gesamten Referenz.
    """)
    return


@app.cell(hide_code=True)
async def _(
    API_BASE_URL,
    AnswerCorrectness,
    AsyncOpenAI,
    EMBED_MODEL_NAME,
    EVALUATOR_MODEL_NAME,
    LiteLLMEmbeddings,
    llm_factory,
    os,
    question_number,
    questions,
    references,
    responses,
):
    # Evaluator-LLM und Embeddings einrichten
    evaluator_client = AsyncOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=API_BASE_URL,
    )
    evaluator_llm = llm_factory(
        EVALUATOR_MODEL_NAME,
        client=evaluator_client,
        max_tokens=65536,              # Sehr hoch, da Reasoning-Modell ~20-30k Tokens für internes Denken braucht
    )

    # Embeddings für die Semantic-Similarity-Komponente von Answer Correctness
    evaluator_embeddings = LiteLLMEmbeddings(
        model=EMBED_MODEL_NAME,
        api_key=os.getenv('OPENAI_API_KEY'),
        api_base=API_BASE_URL,
        encoding_format='float',
    )

    # Answer Correctness initialisieren (braucht LLM + Embeddings)
    answer_correctness = AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings)

    # --- Demo: Answer Correctness für eine einzelne Frage berechnen ---
    # Zeigt den Score für das Beispiel aus der Erklärung oben 
    demo_question = questions[question_number]
    demo_reference = references[question_number]
    demo_response = responses[question_number]

    print(f'Frage:     {demo_question}')
    print(f'Referenz:  {demo_reference[:100]}...')
    print(f'Antwort:   {demo_response[:100]}...')
    print()

    # Einzelne Frage evaluieren (abatch_score erwartet eine Liste)
    demo_result = await answer_correctness.abatch_score([{
        'user_input': demo_question,
        'response': demo_response,
        'reference': demo_reference,
    }])

    print(f'Answer Correctness Score: {float(demo_result[0].value):.4f}')
    return (
        answer_correctness,
        demo_question,
        demo_reference,
        demo_response,
        evaluator_llm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Was passiert wenn wir die obrige Codezelle erneut ausführen? <br>
    - Ändert sich der Wert des Answer Correctness Score? <br>
    - Falls ja, warum?
    """)
    return


@app.cell(hide_code=True)
async def _(
    StatementGeneratorInput,
    StatementGeneratorOutput,
    answer_correctness,
    demo_question,
    demo_reference,
    demo_response,
    embed,
    evaluator_llm,
    np,
):
    # ---------------------------------------------------------------------------
    # Blick hinter die Kulissen: Wie zerlegt das Evaluator-LLM Texte in Claims?
    print('Claims aus der Referenzantwort:')
    # RAGAS gibt über die API nur den finalen Score zurück, nicht die intern
    # erzeugten Claims. Hier rufen wir den gleichen Prompts manuell auf, die
    # AnswerCorrectness intern für Claim-Zerlegung und Klassifikation verwendet.
    #
    # WICHTIG: Da jeder LLM-Aufruf leicht unterschiedliche Ergebnisse liefert
    # (auch bei temperature ≈ 0), sind die hier angezeigten Claims NICHT exakt
    # dieselben, die in der Bewertung oben verwendet wurden — sie zeigen aber
    # den gleichen Mechanismus und sind qualitativ vergleichbar.
    ref_prompt = answer_correctness.statement_generator_prompt.to_string(StatementGeneratorInput(question=demo_question, answer=demo_reference))
    ref_claims = await evaluator_llm.agenerate(ref_prompt, StatementGeneratorOutput)
    # --- Schritt 1: Claims aus beiden Texten extrahieren ---
    for _i, stmt in enumerate(ref_claims.statements):
        print(f'  {_i + 1}. {stmt}')
    print()
    print('Claims aus der generierten Antwort:')
    resp_prompt = answer_correctness.statement_generator_prompt.to_string(StatementGeneratorInput(question=demo_question, answer=demo_response))
    resp_claims = await evaluator_llm.agenerate(resp_prompt, StatementGeneratorOutput)
    for _i, stmt in enumerate(resp_claims.statements):
        print(f'  {_i + 1}. {stmt}')
    print(f'\n→ Referenz: {len(ref_claims.statements)} Claims, Antwort: {len(resp_claims.statements)} Claims')
    classification = await answer_correctness._classify_statements(demo_question, resp_claims.statements, ref_claims.statements)
    tp, fp, fn = (len(classification.TP), len(classification.FP), len(classification.FN))
    f1 = answer_correctness._compute_f1_score(classification)
    print(f'\nKlassifikation: TP={tp}, FP={fp}, FN={fn}')
    print(f'F1-Score (Factual):  {f1:.3f}')
    for _label, items in [('TP', classification.TP), ('FP', classification.FP), ('FN', classification.FN)]:
        if items:
            print(f'\n  {_label}:')
            for item in items:
                print(f'    • {item.statement}')
                print(f'      Grund: {item.reason}')
    # --- Schritt 2: TP/FP/FN Klassifikation ---
    ref_emb = embed([demo_reference])[0]
    resp_emb = embed([demo_response])[0]
    cos_sim = np.dot(ref_emb, resp_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(resp_emb))
    print(f'\nSemantic Similarity (Cosine): {cos_sim:.3f}')
    final = 0.75 * f1 + 0.25 * cos_sim
    # TP, FP, FN im Detail anzeigen
    # --- Schritt 3: Semantic Similarity ---
    # --- Schritt 4: Gewichteter Gesamtscore ---
    print(f'\nAnswer Correctness = 0.75 · {f1:.3f} + 0.25 · {cos_sim:.3f} = {final:.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Was passiert wenn wir die obrige Codezelle erneut ausführen? <br>
    - Ändert sich die Anzahl der Claims?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Beispiel

    **Frage:** *„Welche Sicherheitsanforderungen gelten für Datei-Uploads und -Downloads über einen Webserver?"*

    **Referenzantwort (kurz, 2 Sätze):**
    > *„Alle mithilfe des Webservers veröffentlichten Dateien müssen vorher auf Schadsoftware
    > geprüft werden. Es muss eine Maximalgröße für Datei-Uploads festgelegt und ausreichend
    > Speicherplatz reserviert sein."*

    **Generierte Antwort (ausführlich, mit Quellenverweisen und Tabelle):**
    > *„Prüfung auf Schadprogramme — Alle über den Webserver veröffentlichten Dateien müssen
    > vor der Bereitstellung auf Schadsoftware geprüft werden (APP.3.2.A3).
    > Maximale Dateigröße festlegen — Für Upload-Vorgänge muss eine maximale Dateigröße definiert
    > werden (APP.3.2.A3, APP.3.1.A4).
    > Speicherplatz reservieren — ausreichend Speicherplatz für erwartete Uploads bereitstellen.
    > Einschränkung der Upload-Funktion — zulässige Dateitypen, Speicherorte und Clients festlegen.
    > Restriktive Zugriffs- und Ausführungsrechte für Upload-Verzeichnisse.
    > Kontrollierter Ablageort — Benutzer dürfen nur in vorab definierten Verzeichnissen speichern.
    > ..."* (gekürzt)

    **Schritt 1 — Claim-Zerlegung** (durch das Evaluator-LLM):

    Das Evaluator-LLM zerlegt die **Referenz in 3 Claims** und die **generierte Antwort in 22 Claims**:

    | # | Claims aus Referenz (3) | Claims aus Antwort (Auswahl von 22) |
    |---|---|---|
    | 1 | Alle veröffentlichten Dateien müssen auf Schadsoftware geprüft werden | Dateien müssen vor Bereitstellung auf Schadsoftware geprüft werden |
    | 2 | Eine Maximalgröße für Datei-Uploads muss festgelegt werden | Für Uploads muss eine maximale Dateigröße definiert werden |
    | 3 | Ausreichend Speicherplatz muss reserviert werden | Ausreichend Speicherplatz für erwartete Uploads bereitstellen |
    | 4 | | Zulässige Dateigrößen müssen festgelegt werden *(Duplikat von #2)* |
    | 5 | | Vorab-Prüfung auf Schadsoftware muss durchgeführt werden *(Duplikat von #1)* |
    | 6 | | Die Upload-Funktion soll eingeschränkt werden |
    | 7 | | Zulässige Dateitypen müssen festgelegt werden |
    | 8 | | Restriktive Rechte für Upload-Verzeichnisse |
    | ... | | *(14 weitere Claims zu Speicherorten, Pfad-Manipulation, Verzeichnis-Listing, etc.)* |

    **Schritt 2 — Klassifikation:**

    | Typ | Anzahl | Bedeutung |
    |---|---|---|
    | **TP** | 5 | 5 Antwort-Claims sind durch die Referenz belegt |
    | **FP** | 17 | 17 Antwort-Claims stehen nicht in der Referenz |
    | **FN** | 0 | Keine Referenz-Claims fehlen in der Antwort |

    → F1 = 2·5 / (2·5 + 17 + 0) = **0.37**

    **Warum 5 TP bei nur 3 Referenz-Claims?** Die generierte Antwort wiederholt Fakten in
    unterschiedlicher Formulierung (z.B. „Maximale Dateigröße definiert werden" und „Zulässige
    Dateigrößen festlegen"). Beide werden als TP gewertet, weil sie denselben Referenz-Claim
    matchen. TP kann also **größer** sein als die Anzahl der Referenz-Claims — Duplikate zählen.

    **Schritt 3 — Semantic Similarity:**

    Cosine Similarity der Embeddings von Antwort und Referenz: **0.85**

    **Schritt 4 — Gewichteter Gesamtscore:**

    Answer Correctness = 0.75 · 0.37 + 0.25 · 0.85 = **0.49**

    > **Wichtig:** Die generierte Antwort enthält **alle** Fakten der Referenz (FN=0)
    > und ist fachlich ausgezeichnet — aber der Score ist nur 0.49, weil 17 von 22 Claims
    > als FP gewertet werden. Answer Correctness misst die **Übereinstimmung mit der Referenz**,
    > nicht die fachliche Qualität. Je kürzer die Referenz und je ausführlicher die Antwort, desto niedriger der Score.

    > **Wichtig:** Die Identifizierung der claims durch das Evaluator-LLM ist **nicht** deterministisch, und kann sich pro Aufruf ändern (und somit auch die Answer Correctness)!
    """)
    return


@app.cell(hide_code=True)
async def _(
    CACHE_DIR,
    LOAD_FROM_CACHE,
    TOP_K,
    answer_correctness,
    json,
    pd,
    questions,
    references,
    responses,
):
    # Nun evaluieren wir den gesamtes Datensatz
    AC_CACHE = CACHE_DIR / f'answer_correctness_top{TOP_K}.json'
    # --- Answer Correctness evaluieren oder aus Cache laden ---
    if LOAD_FROM_CACHE and AC_CACHE.exists():
        with open(AC_CACHE, 'r') as _f:
            ac_scores = json.load(_f)
        print(f'Answer Correctness aus Cache geladen: {AC_CACHE.name}')
    else:
        ac_inputs = [{'user_input': questions[_i], 'response': responses[_i], 'reference': references[_i]} for _i in range(len(questions))]
        print(f'Evaluiere Answer Correctness für {len(questions)} Fragen (parallel)...\n')
        ac_results = await answer_correctness.abatch_score(ac_inputs)  # Eingaben aufbauen — Answer Correctness braucht: user_input, response, reference
        ac_scores = [float(r.value) for r in ac_results]
        with open(AC_CACHE, 'w') as _f:
            json.dump(ac_scores, _f)
        print(f'Ergebnisse gespeichert: {AC_CACHE.name}')
    print(f'Answer Correctness (Durchschnitt): {sum(ac_scores) / len(ac_scores):.2f}')
    ac_df = pd.DataFrame({'user_input': questions, 'reference': references, 'response': responses, 'answer_correctness': ac_scores})
    # Per-Frage Scores anzeigen
    ac_df[['user_input', 'answer_correctness']].head(5)  # Parallele Evaluation  # Ergebnisse speichern
    return ac_df, ac_scores


@app.cell(hide_code=True)
def _(
    CHUNKING_MODE,
    DATASET,
    EMBED_SHORT,
    EVALUATOR_SHORT,
    FIGURES_DIR,
    TOP_K,
    ac_df,
    plt,
):
    # Histogramm der Answer Correctness Scores
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _ax.hist(ac_df['answer_correctness'], bins=20, edgecolor='black', range=(0, 1))
    _ax.set_xlabel('Score')
    _ax.set_ylabel('Count')
    _mean_score = ac_df['answer_correctness'].mean()
    _ax.set_title(f'Answer Correctness  (mean: {_mean_score:.2})')
    _ax.axvline(_mean_score, color='red', linestyle='--', label=f'Mean: {_mean_score:.2f}')
    _ax.legend()
    _fig.suptitle(f'Answer Correctness — {EMBED_SHORT} | {CHUNKING_MODE} | {DATASET} | top_k={TOP_K} | eval={EVALUATOR_SHORT}', fontsize=13, y=1.03)
    plt.tight_layout()
    _fig_dir = FIGURES_DIR / '04_generation_evaluation'
    _fig_dir.mkdir(parents=True, exist_ok=True)
    _fig_path = _fig_dir / f'answer_correctness_hist__{EMBED_SHORT}__{CHUNKING_MODE}__{DATASET}__top{TOP_K}__{EVALUATOR_SHORT}.png'
    _fig.savefig(_fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {_fig_path}')
    # Figur speichern
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(Markdown, ac_df, display):
    # Beste und schlechteste Frage (Answer Correctness) vergleichen
    for _label, _row in [('Schlechteste', ac_df.nsmallest(1, 'answer_correctness').iloc[0]), ('Beste', ac_df.nlargest(1, 'answer_correctness').iloc[0])]:
        print(f"=== {_label} Frage (Score: {_row['answer_correctness']:.3f}) ===")
        print(f"Frage: {_row['user_input']}")
        display(Markdown(f"**Referenzantwort:**\n\n{_row['reference']}"))
        display(Markdown(f"**Generierte Antwort:**\n\n{_row['response']}"))
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Faithfulness

    **Faithfulness** misst, ob die generierte Antwort **durch die abgerufenen Kontexte belegt** ist
    (keine Halluzination).

    **So funktioniert die Metrik:**
    1. Die generierte Antwort wird in einzelne **Aussagen (Claims)** zerlegt
    2. Für jede Aussage prüft das Evaluator-LLM: Ist diese Aussage im Kontext belegt? (Verdict: 1=Ja, 0=Nein)
    3. Score = Anzahl belegter Aussagen / Gesamtanzahl Aussagen

    $$ \text{Faithfulness} = \frac{\text{Anzahl belegter Claims}}{\text{Gesamtanzahl Claims}} $$

    ### Beispiel

    **Frage:** *„Welche Sicherheitsanforderungen gelten für Datei-Uploads und -Downloads über einen Webserver?"*

    Das Evaluator-LLM zerlegt die generierte Antwort in **11 Claims** und prüft jeden einzeln
    gegen die abgerufenen Kontexte:

    | Claim | Verdict | Begründung des Evaluators |
    |---|---|---|
    | Alle Dateien müssen vor Bereitstellung auf Schadsoftware geprüft werden — **einschließlich hochgeladener Dateien** | ✗ | Kontext nennt nur *veröffentlichte* Dateien, nicht explizit *hochgeladene* |
    | Maximale Dateigröße für **Uploads und Downloads** definieren | ✗ | Kontext fordert nur Maximalgröße für *Uploads*, nicht für Downloads |
    | Ausreichend Speicherplatz für zulässige Maximalgröße bereitstellen | ✓ | Kontext verlangt ausreichend Speicherplatz für Uploads |
    | Upload-Funktion einschränken (Dateigröße, Dateitypen, Speicherorte, Clients) | ✓ | Kontext fordert Einschränkung und nennt alle vier Parameter |
    | Restriktive Zugriffs- und Ausführungsrechte für Upload-Verzeichnisse | ✓ | Kontext verlangt restriktive Rechte |
    | Benutzer dürfen nur in vorgegebenen Verzeichnissen speichern (keine Pfad-Manipulation) | ✓ | Kontext legt fest, dass Ablageort nicht beeinflusst werden darf |
    | Upload-Parameter müssen während des Betriebs konfigurierbar sein | ✓ | Kontext verlangt Konfigurationsfunktionen für Uploads |
    | Nur Dateien aus dem WWW-Wurzelverzeichnis ausliefern | ✓ | Kontext schreibt Beschränkung auf WWW-Wurzelverzeichnis vor |
    | Verzeichnis-Listing deaktivieren | ✓ | Kontext fordert Deaktivierung nicht benötigter Listing-Funktionen |
    | Vertrauliche Dateien nicht in öffentlichen Verzeichnissen + regelmäßige Kontrollen | ✓ | Kontext verlangt Schutz und regelmäßige Überprüfungen |
    | **Kombination** der Maßnahmen garantiert Sicherheit | ✗ | Kontext macht keine Aussage über Garantien durch Kombination |

    → **8 von 11 Claims belegt** → Faithfulness = 8/11 = **0.73**

    **Warum nicht 1.0?** Die Antwort enthält drei Claims, die leicht über den Kontext hinausgehen:
    zwei erweitern korrekte Aussagen um nicht belegte Details (Downloads, hochgeladene Dateien)
    und eine ist eine Schlussfolgerung, die der Kontext nicht explizit nennt.
    Das sind keine groben Halluzinationen, aber streng genommen nicht durch den Kontext belegt.

    **Kernfrage:** Erfindet das LLM Informationen, die nicht im Kontext stehen?

    **Benötigte Felder:** `user_input`, `response`, `retrieved_contexts`
    """)
    return


@app.cell(hide_code=True)
async def _(
    Faithfulness,
    demo_question,
    demo_response,
    evaluator_llm,
    question_number,
    retrieved_contexts,
):
    # Faithfulness initialisieren (braucht nur LLM, keine Embeddings)
    faithfulness = Faithfulness(llm=evaluator_llm)

    # ---------------------------------------------------------------------------
    # Blick hinter die Kulissen: Wie bewertet Faithfulness einzelne Claims?
    # ---------------------------------------------------------------------------
    # Faithfulness zerlegt die generierte Antwort in Claims und prüft für jeden:
    # "Ist diese Aussage durch die abgerufenen Kontexte belegt?" (Verdict: 1=Ja, 0=Nein)
    #
    # Score = Anzahl belegter Claims / Gesamtanzahl Claims
    #
    # WICHTIG: Wie bei Answer Correctness sind dies NICHT exakt die Claims aus
    # der Bewertung oben, sondern ein separater LLM-Aufruf mit dem gleichen Prompt.
    # ---------------------------------------------------------------------------

    # --- Schritt 1: Antwort in atomare Claims zerlegen ---
    print(f'Frage: {demo_question}\n')
    print('Claims aus der generierten Antwort:')
    faith_statements = await faithfulness._create_statements(demo_question, demo_response)

    # --- Schritt 2: Jeden Claim gegen die Kontexte prüfen (NLI) ---
    context_str = '\n'.join(retrieved_contexts[question_number])
    verdicts = await faithfulness._create_verdicts(faith_statements, context_str)

    print(f'\n→ {len(faith_statements)} Claims, davon {sum(1 for v in verdicts.statements if v.verdict)} durch Kontexte belegt\n')

    # Ergebnisse im Detail anzeigen
    for v in verdicts.statements:
        icon = '✓ belegt' if v.verdict else '✗ NICHT belegt'
        print(f'  [{icon}] {v.statement}')
        print(f'           Grund: {v.reason}')

    # --- Score berechnen ---
    score = faithfulness._compute_score(verdicts)
    print(f'\nFaithfulness = {sum(1 for v in verdicts.statements if v.verdict)}/{len(verdicts.statements)} = {score:.3f}')
    return (faithfulness,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we calculate fiathfulness for all questions in the dataset.
    """)
    return


@app.cell(hide_code=True)
async def _(
    CACHE_DIR,
    LOAD_FROM_CACHE,
    TOP_K,
    display,
    faithfulness,
    json,
    pd,
    questions,
    references,
    responses,
    retrieved_contexts,
):
    # --- Faithfulness evaluieren oder aus Cache laden ---
    FAITH_CACHE = CACHE_DIR / f'faithfulness_top{TOP_K}.json'
    if LOAD_FROM_CACHE and FAITH_CACHE.exists():
        with open(FAITH_CACHE, 'r') as _f:
            faith_scores = json.load(_f)
        print(f'Faithfulness aus Cache geladen: {FAITH_CACHE.name}')
    else:
        faith_inputs = [{'user_input': questions[_i], 'response': responses[_i], 'retrieved_contexts': retrieved_contexts[_i]} for _i in range(len(questions))]
        print(f'Evaluiere Faithfulness für {len(questions)} Fragen (parallel)...\n')  # Eingaben aufbauen — Faithfulness braucht: user_input, response, retrieved_contexts
        faith_results = await faithfulness.abatch_score(faith_inputs)
        faith_scores = [float(r.value) for r in faith_results]
        with open(FAITH_CACHE, 'w') as _f:
            json.dump(faith_scores, _f)
        print(f'Ergebnisse gespeichert: {FAITH_CACHE.name}')
    avg_faith = sum(faith_scores) / len(faith_scores)
    print(f'Faithfulness (Durchschnitt): {avg_faith:.4f}')
    faith_df = pd.DataFrame({'user_input': questions, 'retrieved_contexts': retrieved_contexts, 'response': responses, 'reference': references, 'faithfulness': faith_scores})
    display(faith_df[['user_input', 'faithfulness']].head(5))
    # Per-Frage Scores anzeigen
    print('...')  # Parallele Evaluation  # Ergebnisse speichern
    return faith_df, faith_scores


@app.cell(hide_code=True)
def _(
    CHUNKING_MODE,
    DATASET,
    EMBED_SHORT,
    EVALUATOR_SHORT,
    FIGURES_DIR,
    TOP_K,
    faith_df,
    plt,
):
    # Histogramm der Faithfulness score
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _ax.hist(faith_df['faithfulness'], bins=20, edgecolor='black', range=(0, 1))
    _ax.set_xlabel('Score')
    _ax.set_ylabel('Count')
    _mean_score = faith_df['faithfulness'].mean()
    _ax.set_title(f'Faithfulness  (mean: {_mean_score:.2})')
    _ax.axvline(_mean_score, color='red', linestyle='--', label=f'Mean: {_mean_score:.2f}')
    _ax.legend()
    _fig.suptitle(f'Faithfulness — {EMBED_SHORT} | {CHUNKING_MODE} | {DATASET} | top_k={TOP_K} | eval={EVALUATOR_SHORT}', fontsize=13, y=1.03)
    plt.tight_layout()
    _fig_dir = FIGURES_DIR / '04_generation_evaluation'
    _fig_dir.mkdir(parents=True, exist_ok=True)
    _fig_path = _fig_dir / f'faithfulness_hist__{EMBED_SHORT}__{CHUNKING_MODE}__{DATASET}__top{TOP_K}__{EVALUATOR_SHORT}.png'
    _fig.savefig(_fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {_fig_path}')
    # Figur speichern
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(Markdown, display, faith_df):
    # Beste und schlechteste Frage (Faithfulness) vergleichen
    for _label, _row in [('Schlechteste', faith_df.nsmallest(1, 'faithfulness').iloc[0]), ('Beste', faith_df.nlargest(1, 'faithfulness').iloc[0])]:
        print(f"=== {_label} Frage (Score: {_row['faithfulness']:.3f}) ===")
        print(f"Frage: {_row['user_input']}\n")
        display(Markdown(f"**Referenzantwort:**\n\n{_row['reference']}"))
        display(Markdown(f"**Generierte Antwort:**\n\n{_row['response']}"))
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Experiment: System-Prompt verändern

    Was passiert mit der Faithfulness, wenn wir dem LLM **erlauben**, über den Kontext hinaus zu antworten?

    Wir ersetzen den restriktiven System-Prompt durch einen, der das Modell ermutigt,
    zusätzliches Wissen einzubringen — und messen, wie stark die Faithfulness sinkt.

    > **Erwartet:** Der Faithfulness-Score sollte deutlich fallen, da das LLM
    > nun Aussagen generiert, die nicht im Kontext belegt sind.
    """)
    return


@app.cell(hide_code=True)
async def _(
    ac_scores,
    answer_correctness,
    asyncio,
    faith_scores,
    faithfulness,
    generate_rag_answer,
    questions,
    references,
    retrieved_contexts,
):
    # Alternativer System-Prompt: ermutigt das LLM, Antworten möglichst kurz zu beantworten, ohne relevante Informationen zu verlieren.
    system_prompt_short = 'Du bist ein RAG-Assistent für IT-Grundschutz. Beantworte die Frage nur auf Basis des bereitgestellten Kontexts. Wenn die Information im Kontext nicht enthalten ist, sage das klar.Beantworte die Frage in kurzen Sätzen. Wenn möglich, beantworte die Frage in maximal 1 bis 5 Sätzen. Je kürzer und klarer deine Antwort ist, desto besser. Gib nur Informationen an, die relevant für die Frage sind. '
    system_prompt_hallucination = 'Du bist ein Experte für IT-Sicherheit und IT-Grundschutz. Beantworte die Frage ausführlich und umfassend. Nutze den bereitgestellten Kontext als Ausgangspunkt, aber ergänze dein Expertenwissen, um eine möglichst vollständige Antwort zu geben. Nenne auch Best Practices und Empfehlungen, die über den Kontext hinausgehen.'
    YOUR_PROMPT = '...'
    SYSTEM_PROMPT_MODIFIED = system_prompt_short
    N_EXPERIMENT = 1
    print(f'Generiere Antworten mit alternativem Prompt ({N_EXPERIMENT} Fragen, parallel)...')
    experiment_responses = list(await asyncio.gather(*[generate_rag_answer(q, ctxs, system_prompt=SYSTEM_PROMPT_MODIFIED) for q, ctxs in zip(questions[:N_EXPERIMENT], retrieved_contexts[:N_EXPERIMENT])]))
    print('Fertig.\n')
    print(f'Evaluiere Answer Correctness ({N_EXPERIMENT} Fragen)...')
    ac_experiment_inputs = [{'user_input': questions[_i], 'response': experiment_responses[_i], 'reference': references[_i]} for _i in range(N_EXPERIMENT)]
    # Alternativer System-Prompt: ermutigt das LLM, über den Kontext hinauszugehen
    ac_experiment_results = await answer_correctness.abatch_score(ac_experiment_inputs)
    ac_experiment_scores = [float(r.value) for r in ac_experiment_results]
    ac_subset = ac_scores[:N_EXPERIMENT]
    ac_subset_avg = sum(ac_subset) / len(ac_subset)
    ac_experiment_avg = sum(ac_experiment_scores) / len(ac_experiment_scores)
    print(f'  Answer Correctness (Original-Prompt, erste {N_EXPERIMENT}):  {ac_subset_avg:.2f}')
    print(f'  Answer Correctness (Modifizierter Prompt, erste {N_EXPERIMENT}): {ac_experiment_avg:.2f}')
    print(f'  Differenz:                                       {ac_experiment_avg - ac_subset_avg:+.2f}')
    # ------------------- YOUR TASK BELOW ---------------------
    # Versuchen Sie den system prompt zu verändern. Schaffen Sie es, bessere Scores zu erreichen? 
    # Welcher prompt hilft, welcher verschlechtert die Werte?
    # Warum?
    print()
    # Your custom prompt
    print(f'Evaluiere Faithfulness ({N_EXPERIMENT} Fragen)...')
    faith_experiment_inputs = [{'user_input': questions[_i], 'response': experiment_responses[_i], 'retrieved_contexts': retrieved_contexts[_i]} for _i in range(N_EXPERIMENT)]
    # Choose the modified system prompt (system_prompt_short, system_prompt_hallucination or your custom prompt)
    faith_experiment_results = await faithfulness.abatch_score(faith_experiment_inputs)
    # ------------------- YOUR TASK ABOVE ---------------------
    faith_experiment_scores = [float(r.value) for r in faith_experiment_results]
    # Nur die ersten N_EXPERIMENT Fragen für das Experiment, da es sonst zu lange dauert
    faith_subset = faith_scores[:N_EXPERIMENT]
    faith_subset_avg = sum(faith_subset) / len(faith_subset)
    # --- Schritt 1: Antworten mit alternativem Prompt parallel generieren ---
    faith_experiment_avg = sum(faith_experiment_scores) / len(faith_experiment_scores)
    print(f'  Faithfulness (Original-Prompt, erste {N_EXPERIMENT}):  {faith_subset_avg:.2f}')
    print(f'  Faithfulness (Modifizierter Prompt, erste {N_EXPERIMENT}): {faith_experiment_avg:.2f}')
    print(f'  Differenz:                                  {faith_experiment_avg - faith_subset_avg:+.2f}')
    print()
    print('=' * 80)
    print(f"{'Metrik':<25} {'Original':>10} {'Modifiziert':>12} {'Differenz':>10}")
    # --- Schritt 2: Answer Correctness evaluieren ---
    print('-' * 80)
    print(f"{'Answer Correctness':<25} {ac_subset_avg:>10.2f} {ac_experiment_avg:>12.2f} {ac_experiment_avg - ac_subset_avg:>+10.2f}")
    print(f"{'Faithfulness':<25} {faith_subset_avg:>10.2f} {faith_experiment_avg:>12.2f} {faith_experiment_avg - faith_subset_avg:>+10.2f}")
    # answer correctness scores vom ursprünglichen system_prompt
    # --- Schritt 3: Faithfulness evaluieren ---
    # faithfullness scores vom ursprünglichen system_prompt
    # --- Zusammenfassung ---
    print('=' * 80)
    return ac_experiment_scores, experiment_responses, faith_experiment_scores


@app.cell(hide_code=True)
def _(
    Markdown,
    ac_experiment_scores,
    ac_scores,
    display,
    experiment_responses,
    faith_experiment_scores,
    faith_scores,
    questions,
    references,
    responses,
):
    # Erste Frage des Experiments im Detail anzeigen
    _i = 0
    print(f'Frage: {questions[_i]}\n')
    print(f'Answer Correctness: Original={ac_scores[_i]:.2f}  |  Experiment={ac_experiment_scores[_i]:.2f}')
    print(f'Faithfulness:       Original={faith_scores[_i]:.2f}  |  Experiment={faith_experiment_scores[_i]:.2f}')
    print()
    display(Markdown(f'**Referenzantwort:**\n\n{references[_i]}'))
    print('—' * 80)
    display(Markdown(f'**Antwort (Original-Prompt):**\n\n{responses[_i]}'))
    print('—' * 80)
    display(Markdown(f'**Antwort (Modifizierter Prompt):**\n\n{experiment_responses[_i]}'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diskussion

    - Wie beeinflusst die Länge einer Antwort Answer Correctness und Faithfulness?
    - Wie unterscheiden sich Answer Correctness und Faithfulness? Bei welchen Fragen divergieren sie?
    - Ist mein RAG System gezwungenermaßen "schlecht" wenn mein Answer Correctness Score niedrig ist?
    - Bei welchen Antworten ist die Faithfulness besonders niedrig?
    - Ist der Faithfulness-Score mit dem offenen Prompt gefallen? Ist das Ergebnis intuitiv?
    - Kann ein RAG-System trotz guter Kontexte (hohe Context Precision) schlecht antworten?
    """)
    return


if __name__ == "__main__":
    app.run()
