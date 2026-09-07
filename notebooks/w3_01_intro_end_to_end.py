# ruff: noqa: PLR1711,RUF059  marimo returns; remaining codes are legacy notebook code kept as is
import marimo

__generated_with = "0.24.0"
app = marimo.App(css_file="ragkit/theme.css", html_head_file="ragkit/head.html")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _():
    from typing import List  # noqa: UP035 - the original annotations use these names
    return List,



@app.cell(hide_code=True)
def _(mo, theme):
    mo.Html(theme.header_html('RAG III'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 — Einführung: RAGAS End-to-End

    Dieses Notebook zeigt den vollständigen RAGAS-Evaluations-Workflow an einem einfachen Beispiel:

    1. Webseite des KI-Servicezentrums Berlin-Brandenburg crawlen und in Chunks aufteilen
    2. Minimale RAG-Pipeline (Retrieval + Generation) — ohne externe Datenbank
    3. Manuelles Testdateset erstellen
    4. Automatische Testdaten-Generierung mit dem RAGAS `TestsetGenerator`
    5. Evaluation mit **Context Precision** und **Answer Correctness**

    > Dieses Notebook ist unabhängig vom Hauptdatensatz (IT-Grundschutz) und dient als konzeptionelle Einführung.

    **Voraussetzung:** `notebooks/.env` ist angelegt (`cp .env.example .env`, siehe README).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Konfiguration
    """)
    return


@app.cell(hide_code=True)
def _():
    # --- Imports ---
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from docling.document_converter import DocumentConverter
    from IPython.display import Markdown, display
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from litellm import completion
    from openai import AsyncOpenAI
    from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerCorrectness, ContextPrecision
    from ragas.testset import TestsetGenerator

    # --- Repo-Root finden ---
    from ragkit import theme
    from ragkit.config import WORKSHOP_DIR, setup

    env = setup(require_qdrant=True, strict=False)
    theme.apply_mpl()

    # --- Modellkonfiguration ---
    # Drei Modell-Rollen im Workshop:
    #   1. Embedding-Modell:  Wandelt Text in Vektoren um (für Qdrant)
    #   2. RAG-Generierung:   Beantwortet Fragen basierend auf Kontexten
    #   3. RAGAS-Evaluator:   Bewertet die Qualität des RAG-Systems
    #
    # Warum unterschiedliche Formate?
    #   - Embedding + RAG werden über LiteLLM aufgerufen. LiteLLM unterstützt viele
    #     Provider (OpenAI, Cohere, HuggingFace, ...) und braucht das 'openai/' Präfix,
    #     um zu wissen, welches Provider-Protokoll verwendet werden soll.
    #   - Der Evaluator wird über RAGAS llm_factory aufgerufen, die intern einen
    #     OpenAI-Client (AsyncOpenAI) mit gesetzter base_url verwendet. Da der Client
    #     direkt mit der API spricht, ist kein Provider-Präfix nötig.
    EMBED_MODEL_NAME = 'openai/octen-embedding-8b'       # LiteLLM-Format (mit Provider-Präfix)
    RAG_MODEL_NAME = 'openai/gpt-oss-120b'               # LiteLLM-Format (Reasoning-Modell)
    EVALUATOR_MODEL_NAME = 'llama-3-3-70b'                # OpenAI-Client direkt (ohne Präfix)
    API_BASE_URL = os.getenv('OPENAI_API_BASE', 'https://api.aisc.hpi.de/')
    TOP_K = 3                                             # Anzahl abzurufender Chunks pro Frage

    return (
        env,
        API_BASE_URL,
        AnswerCorrectness,
        AsyncOpenAI,
        ChatOpenAI,
        ContextPrecision,
        Document,
        DocumentConverter,
        EMBED_MODEL_NAME,
        EVALUATOR_MODEL_NAME,
        list,
        LiteLLMEmbeddings,
        Markdown,
        OpenAIEmbeddings,
        RAG_MODEL_NAME,
        TOP_K,
        TestsetGenerator,
        WORKSHOP_DIR,
        completion,
        display,
        llm_factory,
        np,
        os,
        pd,
        plt,
    )



@app.cell(hide_code=True)
def _(env, mo):
    mo.md(env.summary_md())
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Dokumente vorbereiten

    Wir crawlen die Webseite des KI-Servicezentrums Berlin-Brandenburg mit Docling und verwenden
    die einzelnen Zeilen als Chunks für unsere Mini-RAG-Pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(DocumentConverter):
    # Webseite mit Docling crawlen und als Markdown extrahieren
    url = 'https://hpi.de/ki-servicezentrum/'
    converter = DocumentConverter()
    result = converter.convert(url)
    markdown_text = result.document.export_to_markdown()  # Strukturierter Text mit Überschriften
    chunks = [line.strip() for line in markdown_text.split('\n') if line.strip()]
    # Jede nicht-leere Zeile wird ein Chunk (einfachste Chunking-Strategie)
    print(f'{len(chunks)} Chunks aus {url}')
    print('\nBeispiele:')
    for _i, chunk in enumerate(chunks):
        print(f'  [{_i}] {chunk}')
    return chunks, markdown_text, url


@app.cell(hide_code=True)
def _(chunks):
    from ragkit.embed import embed

    # Alle Chunks einbetten
    doc_vectors = embed(chunks)
    print(f'{len(doc_vectors)} Chunks eingebettet (Dimension: {len(doc_vectors[0])})')
    return doc_vectors, embed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Minimale RAG-Pipeline

    Unsere Pipeline besteht aus zwei Schritten:
    1. **Retrieval** — Per Cosine Similarity die relevantesten Chunks finden
    2. **Generation** — Mit dem LLM eine Antwort aus den gefundenen Chunks erzeugen

    > In den Hauptnotebooks verwenden wir Qdrant als Vektordatenbank.
    > Hier reicht einfache In-Memory-Suche, da wir nur wenige Chunks haben.
    """)
    return


@app.cell(hide_code=True)
def _(
    API_BASE_URL,
    List,
    RAG_MODEL_NAME,
    TOP_K,
    chunks,
    completion,
    doc_vectors,
    embed,
    np,
    os,
):
    def retrieve(question: str, top_k: int=TOP_K) -> List[str]:
        """Retrieve the top-k most relevant chunks by cosine similarity.

        Args:
            question: The user's question.
            top_k: Number of chunks to return.

        Returns:
            List of the top-k most relevant chunk texts.
        """
        q_vec = np.array(embed([question])[0])  # Frage in denselben Vektorraum einbetten
        scores = []
        for dv in doc_vectors:
            dv_arr = np.array(dv)  # Cosine Similarity zu jedem Chunk berechnen
            sim = np.dot(q_vec, dv_arr) / (np.linalg.norm(q_vec) * np.linalg.norm(dv_arr))
            scores.append(float(sim))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [chunks[_i] for _i in ranked_idx[:top_k]]  # Kosinus-Ähnlichkeit

    def generate(question: str, contexts: List[str]) -> str:
        """Generate an answer from retrieved contexts via LLM.  # Nach Score absteigend sortieren, Top-K zurückgeben
      # Indizes nach Score sortiert
        Args:
            question: The user's question.
            contexts: Retrieved chunk texts as context.

        Returns:
            Generated answer string.
        """
        context_block = '\n\n'.join((f'[Quelle {_i + 1}]\n{c}' for _i, c in enumerate(contexts)))
        resp = completion(model=RAG_MODEL_NAME, messages=[{'role': 'system', 'content': 'Beantworte die Frage nur auf Basis des Kontexts. Antworte kurz und präzise. Wenn die Informationen im Kontext nicht ausreichen, um die Frage zu beantworten, gib das klar an. Verwende keine Informationen, die nicht im Kontext stehen.'}, {'role': 'user', 'content': f'Frage: {question}\n\nKontext:\n{context_block}'}], api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'), temperature=0.2, max_tokens=2000)
        return resp.choices[0].message.content
    print('RAG-Pipeline definiert: retrieve() + generate()')  # Kontexte nummeriert zusammenfügen
    return generate, retrieve


@app.cell(hide_code=True)
def _(generate, retrieve):
    # Ein End-to-End-Beispiel: Frage → Retrieval → Generation
    question = 'Welche Services bietet das KI-Servicezentrum an?'
    context = retrieve(question)
    answer = generate(question, context)
    print(f'Frage: {question}\n')
    for _i, ctx in enumerate(context):
        print(f"Kontext [{_i + 1}]: {ctx[:120]}{('...' if len(ctx) > 120 else '')}\n")
    print(f'Generierte Antwort:\n{answer}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Testdaten erstellen

    Bevor wir unser RAG-System evaluieren können, brauchen wir **Testdaten**:
    - **Fragen** (`user_input`) — Was wird an das System gestellt?
    - **Referenzantworten** (`reference`) — Was ist die korrekte Antwort?

    Es gibt zwei Wege:

    | Methode | Vorteil | Nachteil |
    |---|---|---|
    | **Manuell** (Fachexperten) | Hohe Qualität, realistische Fragen | Zeitaufwendig |
    | **Automatisch** (RAGAS TestsetGenerator) | Schnell, skalierbar | Qualität variiert |

    <br>

    > In den kommenden notebooks arbeiten wir mit einem **kuratierten Datensatz** (CSV mit 43 Fragen),
    > den Fachexperten erstellt haben. Hier probieren wir beide Wege aus.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4a) Manuelles Testdateset
    """)
    return


@app.cell(hide_code=True)
def _():
    # Vorgefertigte Fragen und Referenzantworten zur KISZ-BB Webseite.
    # Teilnehmer: Gerne weitere Fragen am Ende der Liste ergänzen!
    manual_questions = ['Welche Angebote bietet das KI_Servicezentrum an?', 'Sind alle Services kostenlos?', 'Welche Beratungsangebote bietet das KI-Servicezentrum?', 'Was passiert in einem Pilotprojekt?', 'Muss man sich auf ein Pilotprojekt bewerben, um die Recheninfrastruktur nutzen zu können?']
    manual_references = ['Das KI_Servicezentrum bietet kostenlose Beratung, Weiterbildung und Rechenleistung (Infrastruktur) an.', 'Ja, alle Services sind kostenlos, da das KI-Servicezentrum vom Bundesministerium für Forschung, Technologie und Raumfahrt gefördert wird.', 'Das KI-Servicezentrum bietet Gruppensprechstunden und individuelle Beratung zum Einsatz von KI. Zudem können drei-monatige Pilotprojekt gemeinsam durchgeführt werden.', 'Unternehmen können sich für Pilotprojekte bewerben und erhalten 2–3 Tage pro Woche aktive Projektbegleitung um gemeinsam einen Prototypen zu entwickeln. Dabei können sie auch die Recheninfrastruktur nutzen.', 'Nein. Wenn Sie diese nutzen möchten, können Sie eine Bewerbung über das Portal des KI-Servicezentrum einreichen. Bitte beachten Sie, dass kein Produktivbetrieb angeboten wird.']
    assert len(manual_questions) == len(manual_references), 'Anzahl Fragen und Referenzen muss übereinstimmen!'
    print(f'{len(manual_questions)} manuelle Testfragen:\n')
    for _i, (q, r) in enumerate(zip(manual_questions, manual_references)):
        print(f'  {_i + 1}. {q}')
        print(f'     → {r}\n')  # --- Eigene Frage hier einfügen: ---  # 'Ihre Frage?', bspw. 'Wie viele KI-Servicezentren gibt es?'  # --- Referenzantwort für Ihre Frage: ---  # 'Ihre Referenzantwort.', bspw. 'Das KI-Servicezentrum Berlin-Brandenburg ist eines von vier KI-Servicezentren Deutschlands.'
    return manual_questions, manual_references


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4b) Synthetische Testdaten mit RAGAS `TestsetGenerator`

    RAGAS kann automatisch Testfragen aus Dokumenten generieren. Intern wird ein *Knowledge Graph*
    aus den Dokumenten aufgebaut und daraus verschiedene Fragetypen erzeugt.

    > **Laufzeit:** ~1–3 Minuten je nach API-Geschwindigkeit.
    """)
    return


@app.cell(hide_code=True)
def _(
    API_BASE_URL,
    ChatOpenAI,
    Document,
    OpenAIEmbeddings,
    TestsetGenerator,
    markdown_text,
    os,
    url,
):
    # gpt-oss-120b als Generator-LLM (größtes verfügbares Modell, bessere German/JSON-Fähigkeiten).
    # Achtung: Reasoning-Modell — kann etwas länger dauern.
    generator_llm = ChatOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=API_BASE_URL,
        model='gpt-oss-120b',
    )
    generator_embeddings = OpenAIEmbeddings(
        model='octen-embedding-8b',
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=API_BASE_URL,
    )

    # TestsetGenerator bekommt das vollständige Dokument (nicht die Chunks!).
    # Er baut intern einen Knowledge Graph auf und generiert daraus Testfragen.
    lc_docs = [Document(page_content=markdown_text, metadata={'source': url})]

    generator = TestsetGenerator.from_langchain(generator_llm, generator_embeddings)  # RAGAS-Generator mit LangChain-LLM
    testset = generator.generate_with_langchain_docs(documents=lc_docs, testset_size=5)  # 5 synthetische Fragen erzeugen

    print(f'{len(testset.samples)} synthetische Testfragen generiert.')
    return (testset,)


@app.cell(hide_code=True)
def _(Markdown, WORKSHOP_DIR, display, pd, testset):
    # --- Toggle: auf True setzen, um gespeicherte Testdaten zu laden (for Reproducibility) ---
    LOAD_FROM_CACHE = True
    CACHE_PATH = WORKSHOP_DIR / 'data' / 'synthetic_testset_intro.csv'
    if LOAD_FROM_CACHE and CACHE_PATH.exists():
        synthetic_df = pd.read_csv(CACHE_PATH)
        print(f'{len(synthetic_df)} synthetische Testfragen aus Cache geladen: {CACHE_PATH.name}')  # Gespeicherte Testdaten laden (spart ~10 Min Generierungszeit)
    else:
        synthetic_df = testset.to_pandas()
        synthetic_df.to_csv(CACHE_PATH, index=False)
        print(f'{len(synthetic_df)} synthetische Testfragen generiert und gespeichert: {CACHE_PATH.name}')  # Frisch generierte Testdaten aus dem TestsetGenerator übernehmen
    print(f'\n{len(synthetic_df)} synthetische Testfragen:\n')
    for _i, row in synthetic_df.iterrows():  # Ergebnis als CSV speichern, damit es beim nächsten Mal schnell geladen werden kann
        print(f'------- Question {_i + 1} -------')
        display(Markdown(f"{row['user_input']} \n\n {row['reference']!s}\n"))
    return (synthetic_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Diskussion:**
    - Sind die generierten Fragen realistisch?
    - Welche Arten von Fragen hat RAGAS erzeugt?
    - In welcher Sprache sind die Fragen/Referenzantworten?
    - Gibt es Rechtschreibfehler?
    - Würden Sie persönlich die Fragen so ausführlich beantworten?

    > **Hinweis zur Sprache:** Die generierten Fragen sind überwiegend auf Englisch, obwohl das
    > Quelldokument deutsch ist. Das liegt daran, dass die internen Prompt-Templates von RAGAS
    > auf Englisch sind. RAGAS bietet eine [Sprachadaption](https://docs.ragas.io/en/stable/howtos/customizations/testgenerator/_language_adaptation/)
    > (`adapt_prompts("german", ...)`), die diese Templates übersetzt — dafür wird jedoch ein
    > Modell benötigt, das Sprachanweisungen zuverlässig befolgt (z.B. GPT-4o). Mit dem hier verwendeten gpt-oss-120b Modell scheint es nicht so gut zu funktionieren.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) RAGAS Evaluation

    Jetzt evaluieren wir unser Mini-RAG-System mit zwei RAGAS-Metriken:

    | Metrik | Was wird gemessen? | Kernfrage |
    |---|---|---|
    | **Context Precision** | Retrieval-Qualität | Stehen relevante Chunks oben im Ranking? |
    | **Answer Correctness** | Generations-Qualität | Stimmt die generierte Antwort mit der Referenz überein? |

    Für jeden Testdatensatz (manuell & synthetisch) durchlaufen wir die komplette Pipeline:

    **Fragen → Retrieval → Generation → RAGAS Evaluation**
    """)
    return


@app.cell(hide_code=True)
def _(
    API_BASE_URL,
    AnswerCorrectness,
    AsyncOpenAI,
    ContextPrecision,
    EMBED_MODEL_NAME,
    EVALUATOR_MODEL_NAME,
    List,
    LiteLLMEmbeddings,
    TOP_K,
    generate,
    llm_factory,
    os,
    pd,
    retrieve,
):
    # RAGAS-Evaluator einrichten — neue Collections-API mit llm_factory
    evaluator_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=API_BASE_URL)
    evaluator_llm = llm_factory(EVALUATOR_MODEL_NAME, client=evaluator_client, max_tokens=8192)
    evaluator_embeddings = LiteLLMEmbeddings(model=EMBED_MODEL_NAME, api_key=os.getenv('OPENAI_API_KEY'), api_base=API_BASE_URL, encoding_format='float')
    context_precision = ContextPrecision(llm=evaluator_llm)
    answer_correctness = AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings)

    async def run_evaluation(questions: List[str], references: List[str], label: str='') -> pd.DataFrame:
        """Run the full RAG pipeline (retrieve + generate) and evaluate with RAGAS.  # default = 1024. Erhöht, da lange Referenzantworten viele Claims erzeugen

        Args:
    # Embeddings für die Semantic-Similarity-Komponente von Answer Correctness
            questions: List of test questions.
            references: Corresponding gold-standard reference answers.
            label: Label for progress output.

        Returns:
            DataFrame with per-question RAGAS scores.
        """
    # Metriken mit Evaluator-LLM initialisieren
        print(f'--- {label} ({len(questions)} Fragen) ---')
        print(f'  Retrieval (top_k={TOP_K})...')
        all_contexts = [retrieve(q) for q in questions]
        print('  Generation...')
        all_responses = [generate(q, ctxs) for q, ctxs in zip(questions, all_contexts)]
        print('  RAGAS Evaluation (parallel)...')
        cp_inputs = [{'user_input': q, 'reference': ref, 'retrieved_contexts': ctxs} for q, ref, ctxs in zip(questions, references, all_contexts)]
        cp_results = await context_precision.abatch_score(cp_inputs)
        ac_inputs = [{'user_input': q, 'response': resp, 'reference': ref} for q, resp, ref in zip(questions, all_responses, references)]
        ac_results = await answer_correctness.abatch_score(ac_inputs)
        rows = [{'user_input': q, 'retrieved_contexts': ctxs, 'response': resp, 'reference': ref, 'context_precision': float(cp_results[_i].value), 'answer_correctness': float(ac_results[_i].value)} for _i, (q, ctxs, resp, ref) in enumerate(zip(questions, all_contexts, all_responses, references))]
        df = pd.DataFrame(rows)
        avg_cp = df['context_precision'].mean()
        avg_ac = df['answer_correctness'].mean()
        print(f'  Ergebnis: {{context_precision: {avg_cp:.4f}, answer_correctness: {avg_ac:.4f}}}\n')
        return df
    print('Evaluation-Pipeline bereit.')  # Schritt 1: Retrieval — relevante Dokumente finden  # Schritt 2: Generation — Antworten erzeugen  # Schritt 3: RAGAS Evaluation via abatch_score() (parallele Bewertung)  # Context Precision — braucht: user_input, reference, retrieved_contexts  # Answer Correctness — braucht: user_input, response, reference  # Ergebnisse zusammenführen — MetricResult.value in float umwandeln
    return (run_evaluation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Manuelles Dataset evaluieren

    > **Laufzeit:** ~1–2 Minuten (Embedding + Generation + RAGAS-Bewertung für 4 Fragen)
    """)
    return


@app.cell(hide_code=True)
async def _(manual_questions, manual_references, run_evaluation):
    manual_result_df = await run_evaluation(manual_questions, manual_references, label='Manuelles Dataset')

    # Ergebnisse anzeigen (nur Fragen und Scores)
    metric_cols = [c for c in manual_result_df.columns
                   if c not in ['user_input', 'retrieved_contexts', 'response', 'reference']]
    manual_result_df[['user_input'] + metric_cols]
    return manual_result_df, metric_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Synthetisches Dataset evaluieren
    """)
    return


@app.cell(hide_code=True)
async def _(metric_cols, run_evaluation, synthetic_df):
    # Fragen und Referenzantworten aus dem TestsetGenerator extrahieren
    synthetic_questions = synthetic_df['user_input'].dropna().tolist()
    synthetic_references = synthetic_df['reference'].dropna().tolist()

    # Sicherstellen, dass Fragen und Referenzen gleich lang sind
    min_len = min(len(synthetic_questions), len(synthetic_references))
    synthetic_questions = synthetic_questions[:min_len]
    synthetic_references = synthetic_references[:min_len]

    synthetic_result_df = await run_evaluation(
        synthetic_questions, synthetic_references, label='Synthetisches Dataset',
    )
    synthetic_result_df[['user_input'] + metric_cols]
    return (synthetic_result_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) Vergleich der Ergebnisse
    """)
    return


@app.cell(hide_code=True)
def _(manual_result_df, plt, synthetic_result_df):
    # --- Plot 1: Grouped bar comparison — Manual vs Synthetic (Automated) dataset ---
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ['context_precision', 'answer_correctness']
    manual_means = [manual_result_df[m].mean() for m in metrics]
    synthetic_means = [synthetic_result_df[m].mean() for m in metrics]
    x = range(len(metrics))
    width = 0.3
    bars1 = ax.bar([_i - width / 2 for _i in x], manual_means, width, label='Manual', edgecolor='black')
    bars2 = ax.bar([_i + width / 2 for _i in x], synthetic_means, width, label='Automated', edgecolor='black')
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{bar.get_height():.2f}', ha='center', fontsize=10)
    # Show values above bars
    ax.set_ylabel('Mean Score')
    ax.set_title('Manual vs. Automated Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diskussion

    - Welcher Datensatz hat schwächere Ergebnisse produziert? Warum?
    - Was fällt bei den synthetisch generierten Fragen auf?

    ## Zusammenfassung

    In diesem Notebook haben wir den kompletten RAGAS-Workflow kennengelernt:

    1. **Wissensbasis** — Dokumente als Grundlage für Retrieval
    2. **RAG-Pipeline** — Retrieval (Cosine Similarity) + Generation (LLM)
    3. **Testdaten** — Automatisch (TestsetGenerator) oder manuell erstellt
    4. **Evaluation** — RAGAS-Metriken bewerten Retrieval- und Generierungsqualität getrennt

    **Wichtige Erkenntnis:** Die Qualität der Testdaten bestimmt maßgeblich, wie aussagekräftig
    die Evaluation ist. Deshalb arbeiten wir in den folgenden Notebooks mit einem von Fachexperten
    kuratierten Datensatz.

    In den folgenden notebooks werden wir im Detail erklären wie die Evaluation mit RAGAS funktioniert.

    **Weiter geht es mit:**
    - `w3_02_ingestion.ipynb` — PDF-Dokument → Chunks → Vektordatenbank
    """)
    return


if __name__ == "__main__":
    app.run()
