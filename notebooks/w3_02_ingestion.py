# ruff: noqa: PLR1711  marimo returns; remaining codes are legacy notebook code kept as is
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
    # 02 — Document Ingestion

    Dieses Notebook führt die komplette Ingestion-Pipeline für Workshop 3 durch:

    1. PDF-Dokument mit Docling in Markdown konvertieren
    2. Markdown-Text normalisieren und in Chunks aufteilen
    3. Chunks mit dem Embedding-Modell vektorisieren
    4. Vektoren in die Qdrant-Datenbank hochladen

    Es führt also im Wesentlichen die Schritte aus, die in Workshop 2 besprochen wurden.

    > **Einmalig ausführen** vor den Notebooks 03–04. Die Qdrant-Collection bleibt
    > zwischen Notebook-Sessions erhalten (persistenter Docker-Storage).

    **Voraussetzung:** `notebooks/.env` ist angelegt (`cp .env.example .env`, siehe README) und Qdrant läuft.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Gemeinsame Konfiguration laden (Pfade, Modellnamen, Qdrant-Einstellungen)
    from ragkit import theme
    from ragkit.config import (
        DATA_DIR,
        EMBED_SHORT,
        FIGURES_DIR,
        MAX_CHUNK,
        OVERLAP,
        PDF_PATH,
        QDRANT_HOST,
        QDRANT_PORT,
        setup,
    )
    env = setup(strict=False)
    theme.apply_mpl()

    # --- Imports ---
    import json

    import matplotlib.pyplot as plt
    from docling.document_converter import DocumentConverter
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    return (
        DATA_DIR,
        Distance,
        DocumentConverter,
        EMBED_SHORT,
        FIGURES_DIR,
        MAX_CHUNK,
        OVERLAP,
        PDF_PATH,
        PointStruct,
        QDRANT_HOST,
        QDRANT_PORT,
        QdrantClient,
        VectorParams,
        env,
        json,
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
    ## 1) Docling-Konvertierung: PDF → Markdown + JSON

    Wir verwenden [Docling](https://github.com/DS4SD/docling) zur strukturierten Textextraktion
    aus dem IT-Grundschutz-Kompendium (PDF). Docling liefert zwei Exportformate:

    - **Markdown** — Text mit erhaltenen Überschriften (`##`), gut für überschriftenbasiertes Chunking
    - **JSON** — Strukturierte Elemente mit Labels (`section_header`, `body_text`, ...) und Seitenzahlen

    > **Laufzeit:** Die Konvertierung dauert ca. 4 Minuten. Mit `LOAD_FROM_CACHE = True`
    > werden gespeicherte Ergebnisse geladen (< 1 Sekunde).
    """)
    return


@app.cell(hide_code=True)
def _(DATA_DIR, DocumentConverter, PDF_PATH, json):
    # --- Toggle: auf True setzen, um gespeicherte Docling-Ergebnisse zu laden ---
    LOAD_FROM_CACHE = True
    MARKDOWN_CACHE = DATA_DIR / f'{PDF_PATH.stem}.md'
    JSON_CACHE = DATA_DIR / f'{PDF_PATH.stem}.json'

    if LOAD_FROM_CACHE and MARKDOWN_CACHE.exists() and JSON_CACHE.exists():
        # Gespeicherte Ergebnisse laden (spart ~4 Min Konvertierungszeit)
        markdown_text = MARKDOWN_CACHE.read_text(encoding='utf-8')
        with open(JSON_CACHE, 'r', encoding='utf-8') as f:
            doc_json = json.load(f)
        print(f'Aus Cache geladen: {MARKDOWN_CACHE.name} + {JSON_CACHE.name}')
    else:
        # Docling-Konvertierung durchführen
        converter = DocumentConverter()
        result = converter.convert(str(PDF_PATH))
        doc = result.document

        markdown_text = doc.export_to_markdown()       # Markdown-Export mit Überschriften
        doc_json = doc.export_to_dict()                # JSON-Export mit Struktur + Seitenzahlen

        # Ergebnisse für zukünftige Läufe speichern
        MARKDOWN_CACHE.write_text(markdown_text, encoding='utf-8')
        with open(JSON_CACHE, 'w', encoding='utf-8') as f:
            json.dump(doc_json, f, ensure_ascii=False, indent=2)
        print('Docling-Konvertierung abgeschlossen und gespeichert.')

    print(f'Markdown: {len(markdown_text):,} Zeichen')
    print(f'JSON: {len(doc_json.get("texts", []))} Textelemente')
    return LOAD_FROM_CACHE, doc_json, markdown_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Text-Normalisierung und Chunking

    Docling kann OCR-Artefakte bei deutschen Umlauten erzeugen (z.B. `/C231` statt `ü`).
    Die Funktion `normalize_text()` bereinigt diese im Markdown, `normalize_json()` im JSON.

    **Zwei Chunking-Methoden:**

    1. **`markdown_headers`** — Markdown-Export entlang von Überschriften splitten, mit `MAX_CHUNK` und `OVERLAP`
    2. **`json_structured_sections`** — JSON-Export entlang von Heading-Labels splitten, ohne Größenlimit

    Beide Methoden geben Records im gleichen Format zurück: `{chunk_id, text, metadata}`
    """)
    return


@app.cell(hide_code=True)
def _():
    # Normalisierung und Chunking kommen jetzt aus dem ragkit-Paket:
    # dieselben Funktionen werden von den Notebooks w2_03 und w3_02 benutzt.
    from ragkit.chunk import (
        normalize_json,
        records_from_docling_json_structured_sections,
        records_from_markdown_header_chunks,
    )

    print('Chunking-Funktionen aus ragkit.chunk geladen:')
    print('  markdown_headers          -> records_from_markdown_header_chunks()')
    print('  json_structured_sections  -> records_from_docling_json_structured_sections()')
    return (
        normalize_json,
        records_from_docling_json_structured_sections,
        records_from_markdown_header_chunks,
    )


@app.cell(hide_code=True)
def _(
    MAX_CHUNK,
    OVERLAP,
    PDF_PATH,
    doc_json,
    markdown_text,
    normalize_json,
    records_from_docling_json_structured_sections,
    records_from_markdown_header_chunks,
):
    # Alle Chunking-Methoden ausführen.
    # Neue Methode hinzufügen = eine Zeile in diesem Dict ergänzen.
    doc_json_clean = normalize_json(doc_json)  # Umlaute im JSON normalisieren
    all_records = {'markdown_headers': records_from_markdown_header_chunks(markdown_text, PDF_PATH, MAX_CHUNK, OVERLAP), 'json_structured_sections': records_from_docling_json_structured_sections(doc_json_clean, PDF_PATH, lang='de')}
    for _method_name, _records in all_records.items():
        _lengths = [len(r['text']) for r in _records]
        print(f'--- {_method_name} ---')
        print(f'  Chunks:       {len(_records)}')
        print(f'  Kürzester:    {min(_lengths)} Zeichen')
    # Statistiken pro Methode ausgeben
        print(f'  Längster:     {max(_lengths)} Zeichen')
        print(f'  Durchschnitt: {sum(_lengths) / len(_lengths):.0f} Zeichen')
        print()
    return (all_records,)


@app.cell(hide_code=True)
def _(EMBED_SHORT, FIGURES_DIR, all_records, plt):
    # Histogramme der Chunk-Längen für alle Methoden nebeneinander
    fig, axes = plt.subplots(1, len(all_records), figsize=(7 * len(all_records), 4))
    if len(all_records) == 1:
        axes = [axes]  # Sicherstellen, dass axes immer eine Liste ist
    for ax, (_method_name, _records) in zip(axes, all_records.items()):
        _lengths = [len(r['text']) for r in _records]
        ax.hist(_lengths, bins=40, edgecolor='black')
        ax.set_xlabel('Chunk Length (chars)')
        ax.set_ylabel('Count')
        ax.set_title(f'{_method_name}\n(n={len(_records)})')
    fig.suptitle(f'Chunk Length Distribution — Embedding: {EMBED_SHORT}', fontsize=13, y=1.03)
    plt.tight_layout()
    fig_dir = FIGURES_DIR / '02_ingestion'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f'chunk_length_hist__{EMBED_SHORT}.png'
    # Figur speichern
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Gespeichert: {fig_path}')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Embeddings erzeugen und in Qdrant hochladen

    Für jede Chunking-Methode wird eine eigene Qdrant-Collection erstellt.
    In `ragkit/config.py` kann per `CHUNKING_MODE` umgeschaltet werden, welche Collection
    die Evaluation-Notebooks (03–05) verwenden.

    > **Laufzeit:** Embedding dauert ca. 4–6 Minuten pro Methode (~5000 Chunks).
    """)
    return


@app.cell(hide_code=True)
def _():
    # embed() batcht, kürzt und cached automatisch — Details in ragkit/embed.py
    from ragkit.embed import embed

    return (embed,)


@app.cell(hide_code=True)
def _(
    Distance,
    EMBED_SHORT,
    LOAD_FROM_CACHE,
    PointStruct,
    QDRANT_HOST,
    QDRANT_PORT,
    QdrantClient,
    VectorParams,
    all_records,
    embed,
):
    # WARNING: Running this cell can take a couple of minutes (12) 
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # Verbindung zu Qdrant herstellen
    UPSERT_BATCH_SIZE = 64
    for _method_name, _records in all_records.items():
        _collection_name = f'grundschutz_chunks_{_method_name}__{EMBED_SHORT}'
    # Für jede Chunking-Methode: Embeddings erzeugen, Collection erstellen, hochladen. 
    # Bereits existierende Collections werden übersprungen (persistenter Docker-Storage).
        if LOAD_FROM_CACHE and qdrant.collection_exists(_collection_name):
            _info = qdrant.get_collection(_collection_name)  # Collection-Name aus Methode + Embedding ableiten
            print(f'=== {_method_name}: Collection existiert bereits ({_info.points_count} Vektoren), übersprungen ===\n')
            continue  # Prüfen, ob Collection bereits existiert — wenn ja, überspringen
        texts = [r['text'] for r in _records]
        print(f'=== {_method_name} ({len(texts)} Chunks) ===')
        print('Embedding...')
        vectors = embed(texts)
        vector_size = len(vectors[0])
        qdrant.create_collection(collection_name=_collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))  # Nur die Texte für das Embedding
        points = [PointStruct(id=r['chunk_id'], vector=vectors[i], payload={'text': r['text'], 'chunk_id': r['chunk_id'], **r['metadata']}) for i, r in enumerate(_records)]
        for batch_start in range(0, len(points), UPSERT_BATCH_SIZE):  # Schritt 1: Embeddings erzeugen
            batch = points[batch_start:batch_start + UPSERT_BATCH_SIZE]
            qdrant.upsert(collection_name=_collection_name, points=batch)
            print(f'  Uploaded {min(batch_start + UPSERT_BATCH_SIZE, len(points))}/{len(points)}', end='\r')
        _info = qdrant.get_collection(_collection_name)
        print(f'\nCollection "{_collection_name}" ready: {_info.points_count} Vektoren\n')  # Schritt 2: Collection erstellen  # Schritt 3: Punkte erstellen — Chunk-Text und Metadaten im Payload  # Metadaten (page_numbers, citation_hint, etc.) einbetten  # Schritt 4: Batch-Upload
    return (qdrant,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Verifikation

    Test-Suche auf allen Collections — gleiche Frage, verschiedene Chunking-Methoden.
    """)
    return


@app.cell(hide_code=True)
def _(EMBED_SHORT, all_records, embed, qdrant):
    # Test-Suche auf allen Collections mit der gleichen Frage
    test_frage = 'Was ist der Unterschied zwischen Prozess- und Systembausteinen?'
    test_vec = embed([test_frage])[0]  # Frage in Vektor umwandeln
    for _method_name in all_records:
        _collection_name = f'grundschutz_chunks_{_method_name}__{EMBED_SHORT}'
        _info = qdrant.get_collection(_collection_name)
        response = qdrant.query_points(collection_name=_collection_name, query=test_vec, limit=3, with_payload=True)
        print(f'=== {_method_name} ({_info.points_count} Vektoren) ===')
        print(f'Frage: "{test_frage}"\n')
        for i, hit in enumerate(response.points):
            pages = hit.payload.get('page_numbers', [])
            page_info = f' | Seiten: {pages}' if pages else ''
            print(f'  [{i + 1}] Score: {hit.score:.4f}{page_info}')
            print(f"      {hit.payload['text'][:200]}...\n")
        print()  # Seitenzahlen anzeigen, falls vorhanden (nur bei json_structured_sections)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fertig

    Für jede Chunking-Methode wurde eine eigene Qdrant-Collection erstellt.
    Der Collection-Name enthält sowohl die Chunking-Methode als auch das Embedding-Modell:

    `grundschutz_chunks_{chunking_mode}__{embedding_model}`

    In `ragkit/config.py` kann per `CHUNKING_MODE` und `EMBED_MODEL_NAME` umgeschaltet werden,
    welche Collection die Notebooks 03–05 verwenden.

    **Weiter geht es mit:**
    - `w3_03_retrieval_evaluation.ipynb` — Retrieval-Metriken
    """)
    return


if __name__ == "__main__":
    app.run()
