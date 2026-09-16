# ruff: noqa: PLR1711,F841,I001,RUF100  marimo returns; remaining codes are legacy notebook code kept as is

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    css_file="ragkit/theme.css",
    html_head_file="ragkit/head.html",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    import html as _html

    def panel(title: str, body: str):
        """Render text as a cell OUTPUT rather than printing it.

        print() goes to marimo's console channel: the editor tucks it into a small grey
        box and `marimo run` drops it altogether. A returned object is a real cell
        output, so it shows in both modes and in the exported HTML. pre-wrap also means
        long lines wrap instead of being clipped at the right edge.
        """
        return mo.Html(
            '<div style="border:1px solid #8b949e55;border-radius:8px;overflow:hidden;'
            'font-family:system-ui,sans-serif;margin:8px 0;">'
            '<div style="padding:6px 12px;font-size:12px;font-weight:600;opacity:0.75;'
            f'border-bottom:1px solid #8b949e55;">{_html.escape(title)}</div>'
            '<pre style="margin:0;padding:10px 12px;font-size:13px;line-height:1.55;'
            f'white-space:pre-wrap;overflow-x:auto;">{_html.escape(body)}</pre></div>'
        )

    return (panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 - Real World: IT-Grundschutz PDF -> JSON -> Qdrant -> RAG

    One real document, all the way through: `notebooks/raw_data/standard_200_1.pdf` (BSI IT-Grundschutz Standard 200-1).

    RAG is an open-book exam for a language model. It looks things up in our documents instead of answering from memory. The work is in preparing the document so a computer can search it precisely.

    <img src="/public/img/w2_03_pipeline.svg" alt="The RAG pipeline: PDF, Docling, chunking, embeddings, Qdrant, cited answer" style="max-width:100%; width:860px">

    The five steps:
    - **Docling** pulls structure out of the PDF
    - **Normalization** repairs encoding artifacts (`/C231` instead of `ü`)
    - **Chunking** cuts it into searchable pieces; we build five strategies
    - **Metadata** keeps every hit traceable (source, strategy, parameters)
    - **Page numbers** from the JSON (`prov.page_no`) make citations checkable

    By the end you can:
    1. Turn a real PDF into clean, searchable text
    2. Explain five chunking strategies and pick one with reasons
    3. Run RAG queries that cite page numbers
    4. Compare strategies on evidence
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Configuration (paths, collection, chunk parameters)

    Every tuning knob in one place, so an experiment changes one value here instead of hunting through the code.

    - `MAX_CHUNK`: characters per chunk (context window, embedding quality)
    - `OVERLAP`: overlap between chunks, so context is not lost at the boundary
    - `COLLECTION_NAME`: the Qdrant collection these chunks land in
    - `OPENAI_API_KEY`: needed for LiteLLM embeddings, read from `notebooks/.env`
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import os
    import re
    from pathlib import Path

    from ragkit import theme
    from ragkit.config import WORKSHOP_DIR, setup

    RAW_DIR = WORKSHOP_DIR / "raw_data"
    OUT_DIR = WORKSHOP_DIR / "processed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH = RAW_DIR / "standard_200_1.pdf"
    env = setup(required_files=(PDF_PATH,), strict=False)
    theme.apply_mpl()

    COLLECTION_NAME = "it_grundschutz_docling"
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333

    # LiteLLM/OpenAI embedding model
    EMBED_MODEL_NAME = "openai/octen-embedding-8b"
    LLM_MODEL_NAME = "openai/gpt-oss-120b"
    API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.aisc.hpi.de/")

    MAX_CHUNK = 1200
    OVERLAP = 200
    return (
        COLLECTION_NAME,
        EMBED_MODEL_NAME,
        LLM_MODEL_NAME,
        MAX_CHUNK,
        OUT_DIR,
        OVERLAP,
        PDF_PATH,
        Path,
        QDRANT_HOST,
        QDRANT_PORT,
        env,
        json,
        re,
    )


@app.cell(hide_code=True)
def _(env, mo):
    mo.md(env.summary_md())
    return


@app.cell(hide_code=True)
def _(
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    LLM_MODEL_NAME,
    MAX_CHUNK,
    OUT_DIR,
    OVERLAP,
    PDF_PATH,
    panel,
):
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
    # Everything the pipeline was configured with, on screen. The API base, the key and
    # Qdrant are in the status table above, so they are not repeated here.
    panel(
        "Configuration",
        "\n".join(
            [
                f"PDF              {PDF_PATH}",
                f"PDF exists       {PDF_PATH.exists()}",
                f"OUT_DIR          {OUT_DIR}",
                f"Collection       {COLLECTION_NAME}",
                f"Embedding model  {EMBED_MODEL_NAME}",
                f"LLM model        {LLM_MODEL_NAME}",
                f"MAX_CHUNK        {MAX_CHUNK}",
                f"OVERLAP          {OVERLAP}",
            ]
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Text normalization and chunking building blocks

    PDF text arrives with systematic typos: `ü` comes out as `/C231`. Ugly for us, fatal for search. A chunk holding `Schl/C231ssel` is never found by a question about "Schlüssel".

    <img src="/public/img/w2_03_normalization.svg" alt="A broken chunk with /C231 artifacts is not found by a search; after normalization retrieval works" style="max-width:100%; width:860px">

    What we build here:
    - repair umlaut and encoding artifacts (`/C196`, `/C231`, ...)
    - normalize only the content fields of the JSON (`text`, `orig`)
    - chunk along paragraph blocks
    """)
    return


@app.cell(hide_code=True)
def _():
    from ragkit.chunk import (
        count_umlaut_placeholders,
        format_citation,
        normalize_json,
        normalize_text,
    )
    from ragkit.chunk import _fix_german_umlauts as fix_german_umlauts  # noqa: F401 - used two cells below

    # --- Chunking building blocks (used by the strategies in section 4) ---
    return (
        count_umlaut_placeholders,
        fix_german_umlauts,
        format_citation,
        normalize_json,
        normalize_text,
    )


@app.cell(hide_code=True)
def _(panel):
    import numpy as np

    import ragkit.embed as rk_embed
    from ragkit.embed import cached_embed, embed

    # embed() batches, truncates and retries; cached_embed() persists the vectors as
    # .npy so a re-run costs nothing. Set RECREATE to True to force re-embedding.
    rk_embed.RECREATE = False

    panel("Embedding cache", f"{rk_embed.CACHE_DIR}/\nRECREATE = {rk_embed.RECREATE}")
    return cached_embed, embed, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What happens if we skip this step?

    The same chunk twice: once with the raw `/C231` artifacts, once normalized. Identical content, different similarity to the question. The artifacts push the chunk away from the question it should answer.
    """)
    return


@app.cell(hide_code=True)
def _(EMBED_MODEL_NAME, embed, fix_german_umlauts, np, panel):
    # One question, two versions of the same chunk — only the encoding differs
    demo_query = "Wie werden Schlüssel verwaltet?"  # German query on purpose — the corpus is German
    broken_chunk = "Die Verwaltung der Schl/C231ssel und die Verschl/C231sselung sind zentrale Aufgaben des ISMS."
    clean_chunk = fix_german_umlauts(broken_chunk)

    q_vec, broken_vec, clean_vec = np.asarray(
        embed([demo_query, broken_chunk, clean_chunk], model=EMBED_MODEL_NAME),
        dtype=np.float32,
    )

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    panel(
        "The same chunk, two encodings",
        "\n".join(
            [
                f"broken      {broken_chunk}",
                f"normalized  {clean_chunk}",
                "",
                f"similarity(question, broken chunk)      {cos(q_vec, broken_vec):.4f}",
                f"similarity(question, normalized chunk)  {cos(q_vec, clean_vec):.4f}",
            ]
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Docling conversion: PDF -> Markdown + JSON

    A PDF stores where each letter sits on the page, not what is a heading, a paragraph or a table. Docling reconstructs that lost structure.

    Two outputs:
    - **Markdown**: readable, good for heading and paragraph chunking
    - **JSON**: provenance, page references, bounding boxes, for the structure-driven strategies
    """)
    return


@app.cell(hide_code=True)
def _(
    OUT_DIR,
    PDF_PATH,
    Path,
    count_umlaut_placeholders,
    json,
    normalize_json,
    normalize_text,
    panel,
):
    from docling.document_converter import DocumentConverter

    def docling_pdf_to_markdown_and_json(pdf_path: Path):
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        # Optimize the Markdown for chunking/retrieval
        markdown = normalize_text(doc.export_to_markdown())

        # Load the structured JSON representation and normalize its text fields
        doc_json = doc.export_to_dict()
        before = count_umlaut_placeholders(doc_json)
        doc_json_clean = normalize_json(doc_json)
        after = count_umlaut_placeholders(doc_json_clean)

        return markdown, doc_json_clean, {"replaced": before - after, "remaining": after}

    markdown_text, docling_json, _conv = docling_pdf_to_markdown_and_json(PDF_PATH)

    markdown_out = OUT_DIR / f"{PDF_PATH.stem}.md"
    json_out = OUT_DIR / f"{PDF_PATH.stem}.docling.json"
    markdown_out.write_text(markdown_text, encoding="utf-8")
    json_out.write_text(json.dumps(docling_json, ensure_ascii=False, indent=2), encoding="utf-8")

    panel(
        "Docling conversion",
        "\n".join(
            [
                f"Umlaut placeholders replaced   {_conv['replaced']}",
                f"Remaining placeholders         {_conv['remaining']}",
                f"Markdown characters            {len(markdown_text):,}",
                f"Saved markdown                 {markdown_out.name}",
                f"Saved JSON                     {json_out.name}",
            ]
        ),
    )
    return docling_json, markdown_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Chunking strategies

    Chunking cuts the document into searchable pieces. Too small and they lose their context; too big and one embedding blurs across several topics. No cut is universally right, so we build five and compare them in section 8.

    <img src="/public/img/w2_03_chunking_strategies.svg" alt="The same document cut five different ways by the five chunking strategies" style="max-width:100%; width:860px">

    Five strategies, one section each:
    - **4a** `markdown_headers`: split at Markdown headings, fine-split where needed
    - **4b** `json_text_no_chunk`: one record per JSON `text` field, no artificial boundaries
    - **4c** `json_structured_sections`: structure-driven from the Docling JSON (`content_layer`, `label`, `prov.page_no`)
    - **4d** `hybrid_docling`: Docling's HybridChunker, structure-faithful and token-aware
    - **4e** `semantic`: boundaries wherever the embedding model sees a topic shift

    Then **4f** loads one of them into Qdrant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4a) markdown_headers: split at the Markdown headings

    The simplest strategy. Cut the Markdown wherever a heading starts, then split any section that overruns `MAX_CHUNK` further along paragraph boundaries, with `OVERLAP` characters carried over.

    Cheap and robust. The cost: plain Markdown carries no page numbers, so these chunks cannot cite a page.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import inspect as _inspect_md

    from ragkit.chunk import chunk_markdown_by_headers

    # The heading split itself, straight from ragkit/chunk.py. A section longer than
    # max_chunk falls through to parser_aware_split(), which cuts on paragraph breaks.
    mo.md(f"```python\n{_inspect_md.getsource(chunk_markdown_by_headers)}```")
    return


@app.cell(hide_code=True)
def _():
    from typing import Any

    from ragkit.chunk import records_from_markdown_header_chunks

    return Any, records_from_markdown_header_chunks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4b) json_text_no_chunk: one record per JSON text field

    No artificial boundaries at all. Every `text` field Docling extracted becomes one record, and its `prov.page_no` is kept as metadata, so these chunks can cite a page.

    The chunks come out as uneven as the document: a one-line heading and a full paragraph are both a record. Useful as a baseline, and the closest thing to the raw extraction.
    """)
    return


@app.cell(hide_code=True)
def _(Any, PDF_PATH, normalize_text):
    def records_from_docling_json_text_fields(doc_json: dict) -> list[dict[str, Any]]:
        """One record per JSON text field, incl. page numbers from prov.page_no."""
        items = []

        def extract_page_numbers(node: dict):
            pages = []
            prov = node.get("prov", [])
            if isinstance(prov, list):
                for p in prov:
                    if isinstance(p, dict) and isinstance(p.get("page_no"), int):
                        pages.append(p["page_no"])
            return sorted(set(pages))

        def walk(obj):
            if isinstance(obj, dict):
                text_value = obj.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    pages = extract_page_numbers(obj)
                    citation_hint = f"p. {', '.join(map(str, pages))}" if pages else None
                    items.append(
                        {
                            "text": normalize_text(text_value),
                            "page_numbers": pages,
                            "citation_hint": citation_hint,
                        }
                    )
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for x in obj:
                    walk(x)

        walk(doc_json)

        # No additional chunking/overlap: 1 record = 1 extracted text segment
        return [
            {
                "chunk_id": i,
                "text": item["text"],
                "metadata": {
                    "source_file": PDF_PATH.name,
                    "source_path": str(PDF_PATH),
                    "doc_type": "pdf",
                    "converter": "docling",
                    "chunking_mode": "json_text_no_chunk",
                    "max_chunk": None,
                    "overlap": 0,
                    "total_chunks": len(items),
                    "page_numbers": item["page_numbers"],
                    "citation_hint": item["citation_hint"],
                },
            }
            for i, item in enumerate(items)
        ]

    return (records_from_docling_json_text_fields,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4c) Structure-driven chunking from Docling JSON

    Uses Docling's own signals:
    - `content_layer: furniture` (headers, footers, page numbers) is discarded first
    - labels like `section_heading` and `page_header` are the boundaries
    - `prov.page_no` is stored per chunk, which is what makes citations possible

    No length-based splitting at all: no `MAX_CHUNK`, no `OVERLAP`. One chunk is exactly one section, heading to heading.
    """)
    return


@app.cell(hide_code=True)
def _():
    from ragkit.chunk import records_from_docling_json_structured_sections

    return (records_from_docling_json_structured_sections,)


@app.cell(hide_code=True)
def _(
    PDF_PATH,
    docling_json,
    mo,
    normalize_text,
    panel,
    records_from_docling_json_structured_sections,
):
    import inspect

    def structured_sections(texts):
        """One chunk per section, heading to heading. No length limit, no overlap."""
        HEADINGS = {"section_header", "section_heading", "heading",
                    "title", "page_header"}
        sections, heading, parts, pages = [], "", [], []

        def flush():
            if parts:
                body = normalize_text(" ".join(parts))
                if body:
                    sections.append(
                        {
                            "text": f"{heading}\n\n{body}".strip() if heading else body,
                            "page_numbers": sorted(set(pages)),
                        }
                    )

        for el in texts:
            if el.get("content_layer") == "furniture":  # drop headers/footers
                continue
            text = normalize_text(el.get("text") or el.get("orig") or "")
            if not text:
                continue
            el_pages = [p["page_no"] for p in el.get("prov", [])]  # for citations
            if el.get("label") in HEADINGS:  # a heading closes the section
                flush()
                heading, parts, pages = text, [], list(el_pages)
            else:
                parts.append(text)
                pages += el_pages
        flush()
        return sections

    # ragkit/chunk.py holds the production version: same logic, plus defensive checks
    # and the full metadata payload. Confirm the two agree before trusting the short one.
    _teaching = structured_sections(docling_json.get("texts", []))
    _production = records_from_docling_json_structured_sections(docling_json, PDF_PATH)
    _same_text = sum(1 for a, b in zip(_teaching, _production) if a["text"] == b["text"])
    _same_pages = sum(
        1
        for a, b in zip(_teaching, _production)
        if a["page_numbers"] == b["metadata"]["page_numbers"]
    )

    mo.vstack(
        [
            mo.md(f"```python\n{inspect.getsource(structured_sections)}```"),
            panel(
                "Checked against the production version in ragkit/chunk.py",
                "\n".join(
                    [
                        f"sections, short version   {len(_teaching)}",
                        f"sections, production      {len(_production)}",
                        f"identical section text    {_same_text}/{len(_production)}",
                        f"identical page numbers    {_same_pages}/{len(_production)}",
                    ]
                ),
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4d) Docling HybridChunker: structure-faithful + token-aware

    Models read **tokens**, not characters. A token averages about three quarters of a word. Every embedding model has a fixed token budget per chunk, and anything past it is cut off silently, with no error message. The strategies so far count characters, so they can only estimate that budget.

    <img src="/public/img/w2_03_tokens.svg" alt="A sentence measured in characters seems to fit, but measured in tokens it overflows the 512-token budget and the tail is silently cut off" style="max-width:100%; width:860px">

    `HybridChunker` cuts along the document structure first, then splits or merges sections so every chunk fits `max_tokens`. Page numbers and headings survive as metadata.

    Two details:
    - **The tokenizer is an approximation.** Our embedding model sits behind a gateway, so its real tokenizer is not available locally. `tiktoken` (`cl100k_base`) stands in: not exact, but conservative enough.
    - **`contextualize()`** prepends the chunk's headings before embedding, like writing the chapter at the top of every index card. That enriched text is stored as `embed_text`, the plain text as `text`. The enriched version gets embedded, the plain one gets displayed.
    """)
    return


@app.cell(hide_code=True)
def _(Any, PDF_PATH, docling_json, format_citation, panel):
    import tiktoken
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    from docling_core.types.doc.document import DoclingDocument

    HYBRID_MAX_TOKENS = 512
    _hybrid_tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens=HYBRID_MAX_TOKENS
    )  # token budget per chunk; halve it to 256 and the chunk count roughly doubles
    hybrid_chunker = HybridChunker(tokenizer=_hybrid_tokenizer, merge_peers=True)
    doc_norm = DoclingDocument.model_validate(docling_json)

    def records_from_hybrid_chunker(dl_doc) -> list[dict[str, Any]]:
        chunks = list(hybrid_chunker.chunk(dl_doc=dl_doc))
        records = []
        for i, chunk in enumerate(chunks):
            # Round trip: our normalized JSON back into a DoclingDocument
            pages = sorted({prov.page_no for item in chunk.meta.doc_items for prov in item.prov})
            headings = list(chunk.meta.headings or [])
            records.append(
                {
                    "chunk_id": i,
                    "text": chunk.text,
                    "embed_text": hybrid_chunker.contextualize(chunk=chunk),
                    "metadata": {
                        "source_file": PDF_PATH.name,
                        "source_path": str(PDF_PATH),
                        "doc_type": "pdf",
                        "converter": "docling",
                        "chunking_mode": "hybrid_docling",
                        "max_chunk": None,
                        "max_tokens": HYBRID_MAX_TOKENS,
                        "overlap": 0,
                        "total_chunks": len(chunks),
                        "page_numbers": pages,
                        "citation_hint": format_citation(pages),
                        "headings": headings,
                    },
                }
            )
        return records

    hybrid_records = records_from_hybrid_chunker(doc_norm)
    _demo = next((r for r in hybrid_records if r["metadata"]["headings"]), hybrid_records[0])
    # One example: text vs. embed_text (contextualize prepends the headings)
    panel(
        f"HybridChunker: {len(hybrid_records)} chunks at max_tokens={HYBRID_MAX_TOKENS}",
        "\n".join(
            [
                f"Headings   {_demo['metadata']['headings']}",
                f"Pages      {_demo['metadata']['page_numbers']}  |  {_demo['metadata']['citation_hint']}",
                "",
                "text (what gets displayed)",
                _demo["text"][:300],
                "",
                "embed_text (what gets embedded, headings prepended)",
                _demo["embed_text"][:300],
            ]
        ),
    )
    return doc_norm, hybrid_records, records_from_hybrid_chunker


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4e) Semantic chunking: boundaries at topic shifts

    Every strategy so far cuts by form: paragraphs, headings, length. This one cuts by meaning. It embeds each sentence and measures how similar neighbours are. Where the similarity drops, the topic has probably moved, and that is where the boundary goes.

    <img src="/public/img/w2_03_semantic_curve.svg" alt="Cosine similarity between neighboring sentences with a threshold; dips below it become chunk boundaries" style="max-width:100%; width:860px">

    Four steps:
    1. Split the text into sentences (a simple regex, deliberately basic)
    2. Embed all sentences in batches through the LiteLLM gateway
    3. Cosine similarity between sentence *i* and sentence *i+1*
    4. Every transition below the percentile threshold becomes a breakpoint

    **Cost:** this is the only strategy that needs embeddings already at chunking time, about 700 sentence embeddings here. A few cents, under a minute.

    **Two limits worth knowing:**
    - `split_sentences_de` is a regex tuned for German (it looks for a following `[A-ZÄÖÜ]`). Abbreviations like "z. B." or "Dr." create false boundaries, which we patch by gluing very short fragments onto the previous sentence. Production splitters such as spaCy do this properly. It also runs on the Markdown text, so page numbers are lost.
    - A percentile threshold **always** declares the lowest X&#8202;% of transitions to be boundaries, even in a passage that never changes topic. The next cell shows both the clearest and the weakest cut, so you can judge.
    """)
    return


@app.cell(hide_code=True)
def _(
    Any,
    EMBED_MODEL_NAME,
    PDF_PATH,
    cached_embed,
    embed,
    normalize_text,
    np,
    re,
):
    SEMANTIC_BREAKPOINT_PERCENTILE = 20

    def split_sentences_de(
        text: str, min_len: int = 25
    ) -> list[str]:  # the lowest 20% of neighbor similarities become breakpoints
        """Simple German-aware sentence splitter: breaks after . ! ? when an uppercase letter/digit follows.
        Very short fragments (often abbreviation artifacts like 'z. B.') are appended to the previous sentence."""
        text = re.sub("(?m)^#{1,6}\\s*", "", normalize_text(text))
        text = re.sub("\\s+", " ", text).strip()
        parts = re.split("(?<=[.!?])\\s+(?=[A-ZÄÖÜ0-9])", text)  # strip Markdown heading markers
        sentences: list[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if sentences and len(p) < min_len:
                sentences[-1] = sentences[-1] + " " + p
            else:
                sentences.append(p)
        return sentences

    def semantic_chunk_text(
        text: str, percentile: float = SEMANTIC_BREAKPOINT_PERCENTILE, model: str = EMBED_MODEL_NAME
    ) -> dict[str, Any]:
        sentences = split_sentences_de(text)
        vecs = cached_embed(
            f"w203_sentences_{len(sentences)}", lambda: embed(sentences, model=model)
        ).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
        threshold = float(np.percentile(sims, percentile))
        breakpoints = [
            i for i, s in enumerate(sims) if s < threshold
        ]  # cached: re-running with a different percentile does NOT re-embed the sentences
        chunks, start = ([], 0)
        for bp in breakpoints:
            chunks.append(" ".join(sentences[start : bp + 1]))
            start = bp + 1  # cosine similarity sentence i <-> i+1
        chunks.append(" ".join(sentences[start:]))
        return {
            "chunks": chunks,
            "sentences": sentences,
            "similarities": sims,
            "threshold": threshold,
            "breakpoints": breakpoints,
            "percentile": percentile,
        }

    def records_from_semantic_chunks(result: dict[str, Any]) -> list[dict[str, Any]]:
        chunks = result["chunks"]
        return [
            {
                "chunk_id": i,
                "text": chunk_text,
                "metadata": {
                    "source_file": PDF_PATH.name,
                    "source_path": str(PDF_PATH),
                    "doc_type": "pdf",
                    "converter": "docling",
                    "chunking_mode": "semantic",
                    "max_chunk": None,
                    "overlap": 0,
                    "breakpoint_percentile": result["percentile"],
                    "total_chunks": len(chunks),
                    "page_numbers": [],
                    "citation_hint": None,
                },
            }
            for i, chunk_text in enumerate(chunks)
        ]  # Markdown-based: no reliable page mapping

    return (
        records_from_semantic_chunks,
        semantic_chunk_text,
        split_sentences_de,
    )


@app.cell(hide_code=True)
def _(panel, split_sentences_de):
    # How the sentence splitter fails, and how the heuristic patches it.
    # (A workshop approximation, not a production splitter: spaCy & Co. handle this properly.)
    _demo = "Die Maßnahmen sind z. B. hier beschrieben. Weitere Details folgen im Anhang."
    _lines = ["Naive split (min_len=1):"]
    _lines += [f"  | {s}" for s in split_sentences_de(_demo, min_len=1)]
    _lines += ["", "With the min_len heuristic (default, tiny fragments merged back):"]
    _lines += [f"  | {s}" for s in split_sentences_de(_demo)]
    _lines += [
        "",
        "The heuristic only catches SHORT fragments. A longer clause after an",
        "abbreviation would still be split wrongly. Good enough for a workshop,",
        "not for production.",
    ]
    panel("Where the German sentence splitter breaks", "\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(markdown_text, mo, panel, semantic_chunk_text):
    import matplotlib.pyplot as plt

    semantic_result = semantic_chunk_text(markdown_text)
    # Note: this call embeds all sentences (once — the result is cached
    # in 'semantic_result' and reused later)
    from IPython.display import display as _display

    _display(
        panel(
            "Semantic chunking result",
            "\n".join(
                [
                    f"Sentences     {len(semantic_result['sentences'])}",
                    f"Breakpoints   {len(semantic_result['breakpoints'])}",
                    f"Chunks        {len(semantic_result['chunks'])}",
                ]
            ),
        )
    )
    sims = semantic_result["similarities"]
    _fig, _ax = plt.subplots(figsize=(14, 4))
    _ax.plot(sims, linewidth=1, label="Cosine similarity of neighboring sentences")
    _ax.axhline(
        semantic_result["threshold"],
        color="red",
        linestyle="--",
        label=f"Threshold P{semantic_result['percentile']:.0f} = {semantic_result['threshold']:.3f}",
    )
    for bp in semantic_result["breakpoints"]:
        _ax.axvline(bp, color="orange", alpha=0.2, linewidth=0.8)
    _ax.set_xlabel("Sentence index")
    _ax.set_ylabel("Cosine similarity")
    _ax.set_title("Semantic chunking: similarity curve and breakpoints (orange)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # Render the figure explicitly. marimo only auto-displays a cell's LAST expression,
    # and this cell ends with the breakpoint cards below, so a bare plt.gcf() here
    # would be built and silently dropped (it did display under Jupyter's inline backend).
    _display(mo.as_html(plt.gcf()))
    import html

    from IPython.display import HTML, display

    def breakpoint_card(bp: int, label: str) -> str:
        s_before = html.escape(semantic_result["sentences"][bp][:300])
        # What actually happens at the breakpoints? Two instructive examples:
        # the CLEAREST cut (lowest similarity) and the WEAKEST one (barely under the threshold).
        s_after = html.escape(semantic_result["sentences"][bp + 1][:300])
        chip = "display:inline-block;padding:0 7px;border-radius:9px;font-size:11px;background:#fff1e5;color:#bc4c00;margin-left:6px;"
        return f'''<div style="border:1px solid #8b949e55;border-radius:8px;overflow:hidden;"><div style="padding:6px 10px;font-size:12px;font-weight:600;border-bottom:1px solid #8b949e55;">{label}<span style="{chip}">similarity {semantic_result["similarities"][bp]:.3f}</span></div><div style="padding:8px 10px;font-size:12.5px;line-height:1.5;">{s_before}…</div><div style="border-top:2px dashed #bc4c00;margin:6px 10px;position:relative;"><span style="position:absolute;top:-9px;left:8px;background:#fff1e5;color:#bc4c00;font-size:10px;padding:0 6px;border-radius:8px;">✂ chunk boundary</span></div><div style="padding:12px 10px 8px;font-size:12.5px;line-height:1.5;">{s_after}…</div></div>'''

    _bps_by_similarity = sorted(semantic_result["breakpoints"], key=lambda b: sims[b])
    display(
        HTML(
            '<div style="font-family:system-ui,sans-serif;margin:10px 0;"><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;">'
            + breakpoint_card(_bps_by_similarity[0], "Clearest cut — a convincing topic shift")
            + breakpoint_card(_bps_by_similarity[-1], "Weakest cut — barely under the threshold")
            + f"""</div><div style="font-size:12px;opacity:0.7;margin-top:8px;">Not every cut is a real topic change: the percentile threshold <b>always</b> turns the lowest {semantic_result["percentile"]:.0f}&#8202;% of transitions into boundaries — even between sentences that clearly belong together (the weakest cuts often split a statement from its example). A known weakness of simple semantic chunking, and a good reason to eyeball the breakpoints instead of trusting the plot alone.</div></div>"""
        )
    )
    return HTML, display, html, plt, semantic_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the plot shows.** Similarity between neighbouring sentences; every dip below the threshold becomes a chunk boundary. The two cards are its extremes.

    - The **clearest** cut usually lands on a real transition, often rediscovering a chapter boundary that the structure-based strategies get for free from the headings.
    - The **weakest** cut is a judgment call that can split related content. The RAG cost is real: one answer spread across two chunks, so retrieval returns half the picture. The usual repair is parent-child retrieval, which finds the small chunk and hands the model the surrounding section.

    **Two different "semantics" in this notebook.** Semantic *chunking* cuts by meaning at ingestion time (here). Semantic *search* finds by meaning at query time (section 6). Same embedding model, independent steps. Section 8 pairs semantic search with all five chunking strategies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Cross-check with the HybridChunker.** The next cell looks up the two example sentences inside the `hybrid_docling` chunks from 4d and asks whether the structure-aware strategy cuts in the same place.

    Typically the clearest semantic cut sits on a heading boundary, so meaning and structure agree. The weakest one sits inside a section, which the HybridChunker keeps together. Where the two disagree, you are looking straight at the trade-off.
    """)
    return


@app.cell(hide_code=True)
def _(HTML, display, hybrid_records, semantic_result):
    # Does the structure-aware HybridChunker cut at the same spots as the semantic strategy?
    from difflib import SequenceMatcher

    def _find_hybrid_chunk(sentence: str):
        """Locate the hybrid chunk containing the sentence (fuzzy — the two texts
        come from different export paths and may differ in whitespace/details)."""
        probe = " ".join(sentence.split()).lower()[:80]
        best = (0, -1)
        for i, r in enumerate(hybrid_records):
            hay = " ".join(r["text"].split()).lower()
            m = SequenceMatcher(None, probe, hay, autojunk=False).find_longest_match(
                0, len(probe), 0, len(hay)
            )
            if m.size > best[1]:
                best = (i, m.size)
        return hybrid_records[best[0]]

    def compare_cut_with_hybrid(bp: int, label: str) -> str:
        chip_o = "display:inline-block;padding:0 7px;border-radius:9px;font-size:11px;background:#fff1e5;color:#bc4c00;margin-right:4px;"
        chip_g = chip_o.replace("#fff1e5", "#dafbe1").replace("#bc4c00", "#1a7f37")
        r_before = _find_hybrid_chunk(semantic_result["sentences"][bp])
        r_after = _find_hybrid_chunk(semantic_result["sentences"][bp + 1])
        row = "padding:6px 10px;font-size:12.5px;border-top:1px solid #8b949e40;"
        if r_before["chunk_id"] == r_after["chunk_id"]:
            verdict = f'''<span style="{chip_g}">keeps both sentences in ONE chunk</span> <span style="opacity:0.65;">#{r_before["chunk_id"]} · headings: {", ".join(r_before["metadata"]["headings"]) or "–"}</span><div style="font-size:11.5px;opacity:0.65;margin-top:2px;">→ the structure says this is one section — the semantic cut here splits related content.</div>'''
        else:
            verdict = f'''<span style="{chip_o}">also cuts here</span> <span style="opacity:0.65;">chunk #{r_before["chunk_id"]} (headings: {", ".join(r_before["metadata"]["headings"]) or "–"}) → chunk #{r_after["chunk_id"]} (headings: {", ".join(r_after["metadata"]["headings"]) or "–"})</span><div style="font-size:11.5px;opacity:0.65;margin-top:2px;">→ meaning and structure agree: a real boundary.</div>'''
        return f'''<div style="border:1px solid #8b949e55;border-radius:8px;overflow:hidden;"><div style="padding:6px 10px;font-size:12px;font-weight:600;">{label} <span style="{chip_o}">similarity {semantic_result["similarities"][bp]:.3f}</span></div><div style="{row}"><b style="font-size:11px;">semantic</b> — <span style="{chip_o}">cuts here (by definition)</span></div><div style="{row}"><b style="font-size:11px;">hybrid_docling</b> — {verdict}</div></div>'''

    _bps_by_similarity = sorted(
        semantic_result["breakpoints"], key=lambda b: semantic_result["similarities"][b]
    )
    display(
        HTML(
            '<div style="display:grid;gap:12px;margin:10px 0;font-family:system-ui,sans-serif;">'
            + compare_cut_with_hybrid(_bps_by_similarity[0], "Clearest semantic cut")
            + compare_cut_with_hybrid(_bps_by_similarity[-1], "Weakest semantic cut")
            + "</div>"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4f) Choose a strategy and create the `records`

    `CHUNKING_MODE` decides which of the five actually goes into Qdrant. The choice is written into every chunk's metadata as `chunking_mode`, so any hit can be traced back to how it was cut.

    `hybrid_docling` and `semantic` are both available here. A record may carry an `embed_text` field; where it does, that text is embedded while `text` is used for display and LLM context (see `contextualize()` in 4d).

    One strategy runs the full pipeline. Section 8 compares all five, so there is no need to re-run the notebook once per strategy.
    """)
    return


@app.cell(hide_code=True)
def _(
    MAX_CHUNK,
    OUT_DIR,
    OVERLAP,
    PDF_PATH,
    doc_norm,
    docling_json,
    json,
    markdown_text,
    panel,
    records_from_docling_json_structured_sections,
    records_from_docling_json_text_fields,
    records_from_hybrid_chunker,
    records_from_markdown_header_chunks,
    records_from_semantic_chunks,
    semantic_result,
):
    # Options: 'markdown_headers' | 'json_text_no_chunk' | 'json_structured_sections' | 'hybrid_docling' | 'semantic'
    CHUNKING_MODE = "json_structured_sections"

    if CHUNKING_MODE == "markdown_headers":
        records = records_from_markdown_header_chunks(markdown_text, PDF_PATH, MAX_CHUNK, OVERLAP)
    elif CHUNKING_MODE == "json_text_no_chunk":
        records = records_from_docling_json_text_fields(docling_json)
    elif CHUNKING_MODE == "json_structured_sections":
        records = records_from_docling_json_structured_sections(docling_json, PDF_PATH)
    elif CHUNKING_MODE == "hybrid_docling":
        records = records_from_hybrid_chunker(doc_norm)
    elif CHUNKING_MODE == "semantic":
        records = records_from_semantic_chunks(semantic_result)
    else:
        raise ValueError(f"Unknown CHUNKING_MODE: {CHUNKING_MODE}")

    chunks_out = OUT_DIR / f"{PDF_PATH.stem}.{CHUNKING_MODE}.chunks.json"
    chunks_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    _lines = [
        f"Chunking mode      {CHUNKING_MODE}",
        f"Records            {len(records)}",
        f"Saved chunks       {chunks_out.name}",
    ]
    if records:
        _lines += [
            "",
            "Record 0",
            records[0]["text"][:800],
            "",
            f"Pages              {records[0]['metadata'].get('page_numbers')}",
            f"Citation           {records[0]['metadata'].get('citation_hint')}",
        ]
    panel("Chunks going into Qdrant", "\n".join(_lines))
    return CHUNKING_MODE, records


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Embeddings with LiteLLM + upload to Qdrant

    An embedding turns text into a long row of numbers: a point on a map of meanings. Related things sit close together ("Hund" next to "Welpe"), unrelated things far apart ("Hund" nowhere near "Steuererklärung"). Qdrant does one job very fast: given a point, find its nearest neighbours.

    - LiteLLM gives one API for embeddings, so swapping model or provider is a small change
    - Where a record has `embed_text` (as with `hybrid_docling`), that is what gets embedded; `text` stays the display and LLM text
    - The upload runs in batches to stay under Qdrant's request-size limit
    """)
    return


@app.cell(hide_code=True)
def _(
    CHUNKING_MODE,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    cached_embed,
    embed,
    panel,
    records,
):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from ragkit.embed import batched

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 'embed_text' (if present, e.g. with hybrid_docling) gets embedded; 'text' remains the display/LLM text
    texts = [r.get("embed_text", r["text"]) for r in records]
    vectors = cached_embed(
        f"w203_main_{CHUNKING_MODE}_{len(texts)}", lambda: embed(texts, model=EMBED_MODEL_NAME)
    ).tolist()
    vector_size = len(vectors[0])

    # New API instead of the deprecated recreate_collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = []
    for r, vec in zip(records, vectors):
        payload = {
            "text": r["text"],
            **r["metadata"],
            "chunk_id": r["chunk_id"],
        }
        points.append(PointStruct(id=r["chunk_id"], vector=vec, payload=payload))

    # Batch upload to stay under the request-size limit (32MB)
    UPSERT_BATCH_SIZE = 64
    _batches = []
    for batch_idx, batch_points in enumerate(batched(points, UPSERT_BATCH_SIZE), start=1):
        client.upsert(collection_name=COLLECTION_NAME, points=batch_points)
        _batches.append(f"  batch {batch_idx}: {len(batch_points)} points")

    info = client.get_collection(COLLECTION_NAME)
    panel(
        "Uploaded to Qdrant",
        "\n".join(
            [
                f"Collection         {COLLECTION_NAME}",
                f"Vectors count      {info.points_count}",
                f"Vector size        {vector_size}",
                f"Points uploaded    {len(points)}",
                "",
                *_batches,
            ]
        ),
    )
    return Distance, PointStruct, VectorParams, batched, client


@app.cell(hide_code=True)
def _(COLLECTION_NAME, client, panel):
    # Sanity check: pull one chunk back out of Qdrant. Did the upload preserve the payload?
    sample = client.retrieve(collection_name=COLLECTION_NAME, ids=[0], with_payload=True)[0]
    panel(
        "Read one chunk back out of Qdrant",
        "\n".join(
            [
                f"chunk_id   {sample.payload.get('chunk_id')}",
                f"pages      {sample.payload.get('page_numbers')}  |  {sample.payload.get('citation_hint')}",
                f"mode       {sample.payload.get('chunking_mode')}",
                f"text       {(sample.payload.get('text') or '')[:300].replace(chr(10), ' ')}",
            ]
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) RAG query demo: semantic search vs keyword search

    Semantic search compares meaning, not letters. A question about "Verantwortung der Geschäftsführung" also finds chunks that only say "Pflichten der Leitungsebene", because both sit close together on the map. The question becomes a point using the same embedding model, and Qdrant returns the nearest chunks.

    The **score** (0 to 1) is how close a chunk sits to the question.
    """)
    return


@app.cell(hide_code=True)
def _(mo, rag_search):
    import inspect as _inspect

    # The search itself lives in ragkit/search.py. Twenty-four lines, so here it is:
    # embed the question, ask Qdrant for the nearest points, flatten the payloads.
    mo.md(f"```python\n{_inspect.getsource(rag_search)}```")
    return


@app.cell(hide_code=True)
def _(COLLECTION_NAME, HTML, display, html):
    from ragkit.search import rag_search

    query = (
        "Welche Anforderungen stellt der Standard an Informationssicherheit und Risikomanagement?"
    )
    # German query on purpose — the corpus is German
    hits = rag_search(query, top_k=5, collection=COLLECTION_NAME)

    def show_hits(query: str, hits) -> None:
        """Render retrieval hits as cards: score/pages chips up top, chunk text below."""
        chip = "display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;background:#ddf4ff;color:#0969da;margin-right:6px;"
        cards = []
        for i, h in enumerate(hits, start=1):
            pages = ", ".join(map(str, h["page_numbers"])) if h["page_numbers"] else "–"
            cite = h["citation_hint"] or "–"
            body = html.escape(h["text"])
            cards.append(
                f'''<div style="border:1px solid #8b949e55;border-radius:8px;overflow:hidden;"><div style="padding:6px 12px;border-bottom:1px solid #8b949e55;font-size:12px;"><b>#{i}</b>&nbsp; <span style="{chip}">score {h["score"]:.3f}</span><span style="{chip}">pages {pages}</span><span style="{chip}">{cite}</span><span style="opacity:0.6;">chunk {h["chunk_id"]} · {h["source_file"]}</span></div><div style="padding:10px 12px;max-height:180px;overflow:auto;font-size:13px;line-height:1.5;white-space:pre-wrap;">{body}</div></div>'''
            )
        display(
            HTML(
                f"""<div style="font-family:system-ui,sans-serif;"><div style="font-size:13px;margin:4px 0 10px;"><b>Query:</b> <code>{html.escape(query)}</code></div><div style="display:grid;gap:10px;">{"".join(cards)}</div></div>"""
            )
        )

    show_hits(query, hits)
    return hits, query, rag_search


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7) RAG + LLM answer with LiteLLM

    So far we have only retrieved. Now the model answers, open-book: it gets the retrieved passages as context, answers only from them, and lists the page numbers at the end. That is what makes an answer checkable, and what separates RAG from a model answering freely from memory.

    Default model: `openai/gpt-oss-120b` through your LiteLLM gateway. A different deployment name goes in `LLM_MODEL_NAME` in `.env`.
    """)
    return


@app.cell(hide_code=True)
def _(display, hits, query):
    from IPython.display import Markdown

    from ragkit.search import answer_with_llm

    llm_answer = answer_with_llm(query, hits)
    display(Markdown(f"**LLM answer**\n\n{llm_answer}"))
    return (answer_with_llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Every line of the system prompt earns its place:**

    - *"Answer only based on the provided context"*: grounding. Without it the model mixes in half-remembered training knowledge, hallucinations included.
    - *"If the information is missing, say so clearly"*: permission to admit ignorance. Otherwise it would rather guess than disappoint you.
    - *"List the sources with page numbers"*: verifiability. Anyone in doubt opens the cited page.
    - *"Answer in the language of the question"*: the corpus is German, your question may not be.

    **Try it:** delete the first instruction from `system_prompt` and compare the answers. Does the model still stick to the document?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8) Comparing the strategies

    Five filing systems, the same questions: which one surfaces the best index cards? First how differently they cut the document, then how they answer.

    1. **Chunk statistics**, free and offline: count and length distribution per strategy
    2. **Retrieval comparison**: one Qdrant collection per strategy (`it_grundschutz_cmp_<strategy>`), then the same questions to all five

    **A higher score does not mean a better strategy.** Each strategy produces different chunk lengths and embeds different text (`hybrid_docling` embeds heading-enriched text), so the cosine scores live on different scales. 0.70 for a one-line chunk is not better than 0.63 for a full section. Compare within a strategy. Across strategies, judge the content: does the top hit answer the question, does it carry enough context, do the page numbers agree?

    **Cost:** the first run embeds every chunk of all five strategies, about 1,500 texts, roughly 1 to 2 minutes through the gateway. Re-runs are nearly free from `embedding_cache/` (set `RECREATE = True` in section 2 to force re-embedding).
    """)
    return


@app.cell(hide_code=True)
def _(
    MAX_CHUNK,
    OVERLAP,
    PDF_PATH,
    display,
    doc_norm,
    docling_json,
    markdown_text,
    mo,
    plt,
    records_from_docling_json_structured_sections,
    records_from_docling_json_text_fields,
    records_from_hybrid_chunker,
    records_from_markdown_header_chunks,
    records_from_semantic_chunks,
    semantic_result,
):
    from statistics import mean, median

    import pandas as pd

    strategy_records = {
        "markdown_headers": records_from_markdown_header_chunks(
            markdown_text, PDF_PATH, MAX_CHUNK, OVERLAP
        ),
        "json_text_no_chunk": records_from_docling_json_text_fields(docling_json),
        "json_structured_sections": records_from_docling_json_structured_sections(
            docling_json, PDF_PATH
        ),
        "hybrid_docling": records_from_hybrid_chunker(doc_norm),
        "semantic": records_from_semantic_chunks(semantic_result),
    }
    rows = []
    for _name, _recs in strategy_records.items():
        lengths = [len(r["text"]) for r in _recs]
        rows.append(
            {
                "strategy": _name,
                "chunks": len(_recs),
                "chars_mean": round(mean(lengths)),
                "chars_median": round(median(lengths)),
                "chars_min": min(lengths),
                "chars_max": max(lengths),
            }
        )
    df_stats = pd.DataFrame(rows).set_index("strategy")
    display(df_stats)  # reuses the sentence embeddings computed earlier
    _fig, _ax = plt.subplots(figsize=(12, 4))
    _ax.boxplot(
        [[len(r["text"]) for r in _recs] for _recs in strategy_records.values()],
        tick_labels=list(strategy_records.keys()),
        showfliers=False,
    )
    _ax.set_ylabel("Chunk length (characters)")
    _ax.set_title("Distribution of chunk lengths per strategy")
    _ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    plt.tight_layout()
    # display() output is discarded if the cell's last statement is an expression with a
    # value, so render the figure through display() too. A bare plt.gcf() here would take
    # over as the cell output and drop the statistics table above it.
    display(mo.as_html(plt.gcf()))
    return (strategy_records,)


@app.cell(hide_code=True)
def _(
    batched,
    cached_embed,
    client,
    display,
    Distance,
    embed,
    EMBED_MODEL_NAME,
    HTML,
    html,
    panel,
    PointStruct,
    strategy_records,
    VectorParams,
):
    COMPARE_QUERIES = [
        "Welche Anforderungen stellt der Standard an Informationssicherheit und Risikomanagement?",
        "Welche Aufgaben und Verantwortung hat die Leitungsebene im Sicherheitsprozess?",
    ]
    TOP_K_COMPARE = 3

    def ingest_records_to_collection(recs, collection_name):
        texts = [r.get("embed_text", r["text"]) for r in _recs]
        vectors = cached_embed(
            f"w203_cmp_{collection_name}_{len(texts)}", lambda: embed(texts, model=EMBED_MODEL_NAME)
        ).tolist()
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=r["chunk_id"],
                vector=v,
                payload={"text": r["text"], **r["metadata"], "chunk_id": r["chunk_id"]},
            )
            for r, v in zip(_recs, vectors)
        ]
        for batch_points in batched(points, 64):
            client.upsert(collection_name=collection_name, points=batch_points)

    _ingested = []
    for _name, _recs in strategy_records.items():
        col = f"it_grundschutz_cmp_{_name}"
        ingest_records_to_collection(_recs, col)
        _ingested.append(f"{_name:26} {len(_recs):5} chunks  ->  {col}")
    # panel() returns an object rather than displaying it, so it needs display() here:
    # a bare call mid-cell would just throw the value away.
    display(panel("One Qdrant collection per strategy", "\n".join(_ingested)))

    def compare_strategies(query: str, top_k: int = TOP_K_COMPARE) -> None:
        """One card per strategy with its top hits.
        Careful: scores are comparable only WITHIN a card, never across strategies —
        chunk length and embedded text differ, so the score scales differ too."""
        chip = "display:inline-block;padding:0 7px;border-radius:9px;font-size:11px;background:#ddf4ff;color:#0969da;margin-right:4px;"
        q_vec = embed([query], model=EMBED_MODEL_NAME)[0]
        columns = []
        for _name in strategy_records:
            resp = client.query_points(
                collection_name=f"it_grundschutz_cmp_{_name}",
                query=q_vec,
                limit=top_k,
                with_payload=True,
            )
            hits_html = []
            for rank, p in enumerate(resp.points, start=1):
                payload = p.payload or {}
                text = payload.get("text") or ""
                pages = ", ".join(map(str, payload.get("page_numbers") or [])) or "–"
                border = "border-top:1px solid #8b949e40;" if rank > 1 else ""
                hits_html.append(
                    f'<div style="padding:7px 10px;{border}"><div style="margin-bottom:3px;"><b style="font-size:11px;">#{rank}</b> <span style="{chip}">score {p.score:.3f}</span><span style="{chip}">p. {pages}</span><span style="{chip}">{len(text):,} chars</span></div><div style="font-size:12px;line-height:1.45;opacity:0.9;">{html.escape(text[:280])}…</div></div>'
                )
            columns.append(
                f"""<div style="border:1px solid #8b949e55;border-radius:8px;overflow:hidden;"><div style="padding:6px 10px;font-size:12px;font-weight:600;border-bottom:1px solid #8b949e55;">{_name} <span style="opacity:0.55;font-weight:400;">· {len(strategy_records[_name])} chunks</span></div>{"".join(hits_html)}</div>"""
            )
        display(
            HTML(
                f"""<div style="font-family:system-ui,sans-serif;margin:10px 0 26px;"><div style="font-size:13px;margin-bottom:4px;"><b>Query:</b> <code>{html.escape(query)}</code></div><div style="font-size:12px;opacity:0.7;margin-bottom:8px;">Scores are comparable <b>within</b> a card, not across strategies — chunk length and embedded text differ, so the score scales differ. Judge by content: does the top hit actually answer the question, and with enough context?</div><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px;">{"".join(columns)}</div></div>"""
            )
        )

    for q in COMPARE_QUERIES:
        compare_strategies(q)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9) Optional: mini chat

    Ask the document your own questions and watch the sources that come back. Can you find a query that exposes the weakness of the strategy you picked in 4f? Detail questions and broad overview questions favour different chunkings, and the comparison above tells you which.

    Type a question and press Ask. Every answer comes with its retrieved sources.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    chat_question = mo.ui.text(
        value="", placeholder="Ask the document a question", label="Question", full_width=True
    )
    chat_ask = mo.ui.run_button(label="Ask")
    mo.vstack([chat_question, chat_ask])
    return chat_ask, chat_question


@app.cell(hide_code=True)
def _(
    COLLECTION_NAME,
    LLM_MODEL_NAME,
    answer_with_llm,
    chat_ask,
    chat_question,
    mo,
    rag_search,
):
    mo.stop(
        not chat_ask.value or not chat_question.value.strip(),
        mo.md("*Type a question and press Ask.*"),
    )
    _hits = rag_search(chat_question.value, top_k=5, collection=COLLECTION_NAME)
    _answer = answer_with_llm(chat_question.value, _hits, model=LLM_MODEL_NAME)
    _sources = "\n".join(
        f"[{i}] score={h['score']:.4f} | pages={h.get('page_numbers') or '-'} | cite={h.get('citation_hint') or '-'}"
        for i, h in enumerate(_hits, start=1)
    )
    mo.vstack([mo.md(f"### Answer\n\n{_answer}"), mo.md("**Sources**"), mo.plain_text(_sources)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10) Recap and takeaways

    What you built:
    1. Extracted a PDF with Docling
    2. Cleaned up the encoding artifacts
    3. Chose a chunking strategy
    4. Stored embeddings and metadata in Qdrant
    5. Ran a RAG query
    6. Generated an answer from the retrieved sources
    7. Compared five chunking strategies head to head

    Where each one fits:
    - `markdown_headers`: quick, robust baseline on Markdown
    - `json_text_no_chunk`: closest to the raw extraction, with page references per text field
    - `json_structured_sections`: clean heading-to-heading sections, no artificial splitting
    - `hybrid_docling`: best balance of structure and token budget, page and heading metadata included. A good default in practice
    - `semantic`: topically coherent regardless of layout, but pays for embeddings at chunking time and carries no page metadata here

    **Structure first when you have it, semantic when you don't.** This notebook's own data showed it. At the real chapter boundary the semantic cut only confirmed what the headings already knew, and at its weakest cut it split a section the HybridChunker kept whole, spreading one answer across two chunks. On a well-structured document like this standard, `hybrid_docling` wins, and brings pages, headings and the token budget along for free. Semantic chunking is the specialist for text without usable structure: transcripts, chat logs, emails, flat OCR output.

    **No strategy wins everywhere.** It depends on the document (highly structured vs running text) and on the question (detail vs overview). Which is exactly why you compare instead of grabbing the first one that works.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
