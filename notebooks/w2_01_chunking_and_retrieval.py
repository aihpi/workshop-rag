# ruff: noqa: PLR1711  marimo cells end with an explicit return
import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Chunking, text embeddings and retrieval",
    css_file="ragkit/theme.css",
    html_head_file="ragkit/head.html",
)


@app.cell(hide_code=True)
def _(mo, theme):
    mo.Html(theme.header_html(
        'Chunking, text embeddings and retrieval',
        'RAG II, notebook 1. How the pieces we index decide what a RAG system can find.',
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 Introduction

    This notebook builds the retrieval half of a retrieval-augmented generation (RAG) system and measures what each design choice changes. The setting is a compliance officer who is writing a security concept and asks questions such as "Welche Anforderungen gelten für die Protokollierung auf einem Webserver?". A useful answer quotes the relevant requirement sections of the IT-Grundschutz-Kompendium rather than paraphrasing, so retrieval has to find those sections.

    Chunking is the first design decision because a vector store does not retrieve documents. It retrieves chunks, the pieces of text that were embedded and indexed. The chunk is what gets compared with the question, what appears in the hit list and what the language model reads. If a boundary separates a requirement from the header that names it, or if a chunk is a bare heading, no embedding model can repair the loss later.

    The notebook works through four research questions:

    - **RQ1 Preprocessing.** What has to happen to the Markdown before chunking, and what belongs in the text rather than in metadata?
    - **RQ2 Chunking.** Which strategy suits this corpus, and how large are the differences?
    - **RQ3 Models.** How does a small, fast embedding model (miniLM) compare with a large one (octen)?
    - **RQ4 Metrics.** Which measurements say something about retrieval quality without ground truth, and which need labelled questions?
    """)
    return


@app.cell(hide_code=True)
def _(SLICE_BAUSTEINE, mo, slice_share):
    mo.md(rf"""
    ### The data

    The corpus is the IT-Grundschutz-Kompendium, edition 2023, from the German Federal Office for Information Security (BSI). A Baustein is a module on one subject, such as APP.3.2 Webserver, and a requirement (Anforderung) is one numbered obligation inside it, such as APP.3.2.A1, marked B (basic), S (standard) or H (elevated protection). It contains 111 Bausteine, 2 123 requirement sections and 2.45 million characters.

    The questions come from the GSKI gold sets: three files with 40, 123 and 43 questions. Each row holds a question, an answer and the cited passage (Fundstelle), so we know which section to retrieve. The 40 are a subset of the 123; the 43 complex questions cite passages spanning several sections.

    Embedding the full corpus with every strategy and both models takes about an hour. The cited passages sit in only {len(SLICE_BAUSTEINE)} of the 111 Bausteine, so the notebook works on a slice of those {len(SLICE_BAUSTEINE)} plus one introductory section the questions need: {slice_share:.0f} % of the text, embedded in about ten minutes and cached. The full corpus remains an optional hard mode in which the other {111 - len(SLICE_BAUSTEINE)} Bausteine act as distractors with near-identical boilerplate. Two questions cite BSI-Standard 200-2, which is not in the corpus, and are dropped.
    """)
    return


@app.cell(hide_code=True)
def _(FancyArrowPatch, FancyBboxPatch, mo, plt, theme):
    def _pipeline_figure(stages, active):
        fig, ax = plt.subplots(figsize=(6.5, 1.1))
        ax.set_axis_off()
        w, gap = 1.3, 0.3
        for i, s in enumerate(stages):
            x = i * (w + gap)
            on = s in active
            ax.add_patch(FancyBboxPatch((x, 0), w, 0.6, boxstyle='round,pad=0,rounding_size=0.04',
                                        facecolor=theme.INK if on else 'white',
                                        edgecolor=theme.INK, linewidth=0.8))
            ax.text(x + w / 2, 0.3, s, ha='center', va='center', fontsize=8,
                    color='white' if on else theme.INK)
            if i < len(stages) - 1:
                ax.add_patch(FancyArrowPatch((x + w, 0.3), (x + w + gap, 0.3), arrowstyle='-|>',
                                             mutation_scale=8, color=theme.GREY, linewidth=0.8))
        ax.set_xlim(-0.05, len(stages) * (w + gap) - gap + 0.05)
        ax.set_ylim(-0.1, 0.7)
        return fig

    mo.vstack([
        _pipeline_figure(['Markdown', 'preprocess', 'chunk', 'embed', 'index', 'retrieve', 'evaluate'],
                         {'preprocess', 'chunk', 'embed'}),
        mo.md('*Figure 1. The retrieval pipeline. This notebook varies the three black stages and measures the effect at the end.*'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 Setup

    The imports below are the toolbox of every notebook in this workshop. Expand the cell to see one comment per library.
    """)
    return


@app.cell(hide_code=True)
def _():
    import hashlib  # fingerprints of chunk texts for the vector cache
    import json  # small on-disk records (embedding timings)
    import time  # wall-clock timing of embedding calls
    from pathlib import Path

    import marimo as mo  # the reactive notebook itself
    import matplotlib.pyplot as plt  # figures
    import numpy as np  # vector maths: normalisation, cosine similarity, ranking
    import pandas as pd  # result tables
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from PIL import Image  # the photo used in the Matryoshka analogy

    from ragkit import theme  # HPI look for notebook chrome and figures
    from ragkit.chunk import (  # preprocessing and chunking strategies
        build_sections,
        chunk_by_chars,
        chunk_by_paragraph,
        chunk_by_words,
        chunk_stats,
        classify_header,
        parse_requirement_header,
        parser_aware_split,
        sections_to_text,
        workshop_slice,
    )
    from ragkit.config import DATA_DIR, WORKSHOP_DIR, setup
    from ragkit.embed import (  # API calls and the vector cache
        CACHE_DIR,
        cached_embed,
        embed,
    )
    from ragkit.evaluate import (  # gold sets, relevance labels, ranking metrics
        attach_section_keys,
        evaluate_ranking,
        gold_targets,
        load_gold,
        relevant_chunks,
        section_key,
        section_texts,
    )
    from ragkit.search import entropy, l2_normalise  # score diagnostics
    from ragkit.viz import head_tail, kde_plot, length_hist_panels, sorted_score_plot

    theme.apply_mpl()
    return (
        CACHE_DIR, DATA_DIR, FancyArrowPatch, FancyBboxPatch, Image, Path, WORKSHOP_DIR,
        attach_section_keys, build_sections, cached_embed, chunk_by_chars, chunk_by_paragraph,
        chunk_by_words, chunk_stats, classify_header, embed, entropy, evaluate_ranking,
        gold_targets, hashlib, head_tail, json, kde_plot, l2_normalise, length_hist_panels, load_gold,
        mo, np, parse_requirement_header, parser_aware_split, pd, plt, relevant_chunks,
        section_key, section_texts, sections_to_text, setup, sorted_score_plot, theme, time,
        workshop_slice,
    )


@app.cell(hide_code=True)
def _(DATA_DIR, mo, setup):
    MD_PATH = DATA_DIR / 'IT_Grundschutz_Kompendium_Edition2023.md'
    GOLD_SETS = ('40_einfach', '123_einfach', '43_komplex')
    env = setup(require_qdrant=True, required_files=(MD_PATH,), strict=False)
    mo.md(env.summary_md())
    return GOLD_SETS, MD_PATH, env


@app.cell(hide_code=True)
def _(env, mo):
    mo.stop(not env.api_key, mo.callout(mo.md(
        'The notebook needs the HPI API key to embed text. Follow the fix above, then re-run.'), kind='warn'))
    return


@app.cell(hide_code=True)
def _(GOLD_SETS, MD_PATH, build_sections, load_gold, mo, pd, section_texts):
    md_raw = MD_PATH.read_text(encoding='utf-8')
    sections = build_sections(md_raw)
    texts_flat = section_texts(sections)
    gold_all = {name: load_gold(name) for name in GOLD_SETS}

    def _corpus_table():
        levels = pd.Series([s['level'] for s in sections if s['kind'] == 'requirement']).value_counts()
        kinds = pd.Series([s['kind'] for s in sections]).value_counts()
        rows = [
            ('characters in the Markdown', f'{len(md_raw):,}'),
            ('`##` headers', f'{len(sections):,}'),
            ('Bausteine', f"{kinds.get('baustein', 0):,}"),
            ('requirement sections', f"{kinds.get('requirement', 0):,} (B {levels.get('B', 0)}, S {levels.get('S', 0)}, H {levels.get('H', 0)})"),
            ('template sections (1.1. Einleitung, 3.1. Basis-Anforderungen, ...)', f"{kinds.get('template', 0):,}"),
            ('elementary threats (G 0.x)', f"{kinds.get('gefaehrdung', 0):,}"),
            ('gold questions', ', '.join(f'{name}: {len(rows_)}' for name, rows_ in gold_all.items())),
        ]
        return '| | |\n|---|---|\n' + '\n'.join(f'| {a} | {b} |' for a, b in rows)

    mo.md('### Corpus at a glance\n\n' + _corpus_table())
    return gold_all, md_raw, sections, texts_flat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where such data comes from

    Nothing is crawled here. The Kompendium was converted from the BSI PDF with Docling (the conversion itself is the subject of notebook w2_03), and the gold questions were written by domain experts in the GSKI pilot project. For the second notebook of this workshop a small image dataset was fetched once from Wikimedia Commons; `ragkit/crawl.py` shows how such a collection step looks when it is needed, with licence filtering and attribution. Participants never run it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 Preprocessing

    Chunking operates on whatever text it is given. Anything that should not end up inside a chunk has to be removed first, and anything the chunker could use as a boundary signal has to be visible to it. Preprocessing is where we decide what stays in the text and what moves into metadata.

    This corpus offers much structure, all in flat `##` headers. A header identifies a Baustein (`## APP.3.2 Webserver`), a template section (`## 3.1. Basis-Anforderungen`) or a requirement (`## APP.3.2.A1 Sichere Konfiguration eines Webservers (B) [IT-Betrieb]`); the last kind carries the level B, S or H and, in about a quarter of cases, the responsible roles. Inside requirement bodies, the capitalised modal verbs MUSS, SOLLTE and DARF NICHT state how binding a sentence is; they are this corpus's form of emphasis. Alongside the content sit 227 image placeholders, template preambles repeated in every Baustein and 289 stubs whose entire body reads "Diese Anforderung ist entfallen."

    The corpus lacks typographic emphasis: no bold, italics or figure captions survived the conversion, so those signals are unavailable here.
    """)
    return


@app.cell(hide_code=True)
def _(md_raw, mo):
    def _raw_excerpt(marker='## APP.3.2.A1 ', n_lines=14):
        lines = md_raw.split('\n')
        start = next(i for i, line in enumerate(lines) if line.startswith(marker))
        return '\n'.join(lines[start - 2:start + n_lines])

    mo.vstack([
        mo.md('### 3.1 What the Markdown looks like\n\nFourteen raw lines around the first requirement of the Webserver Baustein. Note the image placeholder, the flat headers and the level marker.'),
        mo.md(f'```markdown\n{_raw_excerpt()}\n```'),
    ])
    return


@app.cell(hide_code=True)
def _(classify_header, md_raw, mo, pd, plt, theme):
    header_kinds = pd.Series([classify_header(line) for line in md_raw.split('\n') if line.startswith('## ')]).value_counts()

    def _header_kinds_figure():
        labels = {'requirement': 'requirement', 'template': 'template section', 'baustein': 'Baustein',
                  'gefaehrdung': 'elementary threat', 'other': 'front matter and other'}
        counts = header_kinds.rename(index=labels).sort_values()
        fig, ax = plt.subplots(figsize=(6.5, 2.2))
        ax.barh(counts.index, counts.values, color=theme.GREYS[2], edgecolor=theme.INK, linewidth=0.6)
        for y, v in enumerate(counts.values):
            ax.text(v + 30, y, f'{v:,}', va='center', fontsize=8.5)
        ax.set_xlabel('headers')
        ax.set_xlim(0, counts.max() * 1.15)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        return fig

    mo.vstack([
        mo.md('### 3.2 Classifying headers\n\nEvery `##` line is matched against four patterns in a fixed order: requirement, Baustein, elementary threat, template section. What none of them matches is front matter (preface, glossary, table of contents) or a stray label the converter promoted to a heading.'),
        _header_kinds_figure(),
        mo.md(f'*Figure 2. Header kinds in the Kompendium Markdown ({len(md_raw.splitlines()):,} lines, {int(header_kinds.sum()):,} headers). Requirements dominate; every Baustein contributes about 19 of them.*'),
    ])
    return (header_kinds,)


@app.cell(hide_code=True)
def _(mo, parse_requirement_header, pd, sections):
    def _parsed_examples():
        picks = ['APP.3.2.A1', 'ISMS.1.A1', 'OPS.1.1.6.A2', 'ISMS.1.A14', 'INF.13.A18']
        rows = []
        for s in sections:
            if s['req_id'] in picks:
                parsed = parse_requirement_header(s['text'].split('\n')[0])
                rows.append({'requirement': parsed['req_id'], 'title': parsed['title'][:60],
                             'level': parsed['level'], 'roles': ', '.join(parsed['roles']) or '(none)'})
        return pd.DataFrame(rows)

    mo.vstack([
        mo.md('### 3.3 Parsing requirement headers\n\nA requirement header has a fixed shape: identifier, title, level in parentheses and optionally the responsible roles in brackets. Parsing it once gives three pieces of metadata that a compliance officer filters by (level B/S/H, roles, Baustein) and that the chunk text should keep, because a hit is only useful if it names the requirement it comes from.'),
        mo.ui.table(_parsed_examples().to_dict('records'), selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(md_raw, mo, sections):
    def _boilerplate_table():
        n_images = md_raw.count('<!-- image -->')
        n_entfallen = sum(s['is_entfallen'] for s in sections)
        bodies = [s['body'] for s in sections if s['body']]
        n_dupe_bodies = len(bodies) - len(set(bodies))
        n_header_only = sum(1 for s in sections if not s['body'])
        rows = [
            ('`<!-- image -->` placeholders', f'{n_images}', 'drop', 'no text, no meaning'),
            ('page headers and footers', '0', 'nothing to do', 'Docling already removed them'),
            ('ENTFALLEN stubs ("Diese Anforderung ist entfallen.")', f'{n_entfallen}', 'drop before embedding', 'identical bodies, retrieve nothing useful'),
            ('sections whose body repeats another section verbatim', f'{n_dupe_bodies}', 'deduplicate chunks', 'template preambles of 3.1 / 3.2 / 3.3'),
            ('header-only sections', f'{n_header_only}', 'keep as metadata, not as chunks', 'e.g. "## 1. Beschreibung" has no body'),
            ('Baustein and requirement IDs, titles', 'all', 'metadata **and** chunk text', 'the chunk must explain itself'),
            ('level B/S/H, roles', 'all requirements', 'metadata', 'filterable, not searchable prose'),
        ]
        return '| element | count | decision | why |\n|---|---|---|---|\n' + '\n'.join(
            f'| {a} | {b} | {c} | {d} |' for a, b, c, d in rows)

    mo.md('### 3.4 Furniture, boilerplate and the text-versus-metadata decision\n\n' + _boilerplate_table())
    return


@app.cell(hide_code=True)
def _(mo, np, pd, plt, sections, theme):
    def _modality_figure():
        rows = [dict(level=s['level'], **s['modality']) for s in sections
                if s['kind'] == 'requirement' and not s['is_entfallen']]
        df = pd.DataFrame(rows).groupby('level').sum().loc[['B', 'S', 'H']]
        order = ['MUSS', 'DARF NUR', 'DARF NICHT', 'SOLLTE', 'SOLLTE NICHT']
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
        for ax, level, name in zip(axes, ['B', 'S', 'H'], ['B (basic)', 'S (standard)', 'H (elevated)']):
            vals = df.loc[level, order].values
            ax.bar(np.arange(len(order)), vals, color=theme.GREYS[2], edgecolor=theme.INK, linewidth=0.6)
            for x, v in enumerate(vals):
                ax.text(x, v + df.values.max() * 0.02, f'{int(v)}', ha='center', fontsize=7.5)
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels(order, rotation=60, ha='right', fontsize=7.5)
            ax.set_title(name)
        axes[0].set_ylabel('occurrences')
        return fig

    mo.vstack([
        mo.md('### 3.5 Emphasis: the modal verbs\n\nThe Kompendium defines its own emphasis. A capitalised MUSS or DARF NICHT marks a mandatory requirement, SOLLTE a strong recommendation. The counts per level show the pattern the authors follow: basic requirements are phrased with MUSS, standard and elevated ones with SOLLTE. We keep the verbs in the chunk text (the model should see them) and additionally store the counts as metadata, so a query can be limited to mandatory requirements.'),
        _modality_figure(),
        mo.md('*Figure 3. Modal verbs in active requirement bodies, by level.*'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What about bold, italics and captions?

    Document converters such as Docling keep typographic emphasis as Markdown syntax. Bold becomes `**text**`, italics become `*text*` or `_text_`, and a figure caption usually appears as a plain paragraph directly after the image reference, often prefixed with "Figure 3:" or "Abbildung 3:". A made-up example:

    ```markdown
    ## 4.2 Access control
    The administrator **must** revoke accounts within *24 hours*.
    ![Diagram of the approval flow](img/flow.png)
    Figure 3: Approval flow for privileged accounts.
    See section 4.1 for the general rule.
    ```

    Rules of thumb. Keep inline emphasis as plain text: strip the asterisks, because embedding models treat `**must**` and `must` almost identically and the markup only costs tokens. Keep a caption in the same chunk as its figure reference, because the caption is the only text that says what the image showed. Put headings into metadata and prepend them to the chunk text, so the chunk explains itself when read alone and can also be filtered by section.
    """)
    return


@app.cell(hide_code=True)
def _(gold_all, gold_targets, json, mo, section_key, sections, texts_flat, workshop_slice):
    targets_by_set = {name: [gold_targets(r['fundstelle'], sections, texts_flat) for r in rows]
                      for name, rows in gold_all.items()}
    key_to_section = {section_key(s): s for s in sections}
    _cited_keys = {k for ts in targets_by_set.values() for keys in ts for k in keys}
    _cited = [key_to_section[k] for k in _cited_keys]
    SLICE_BAUSTEINE = sorted({s['baustein_id'] for s in _cited if s['baustein_id']})
    FRONT_TITLES = sorted({s['title'] for s in _cited if s['is_front_matter']})
    slice_sections = workshop_slice(sections, SLICE_BAUSTEINE, FRONT_TITLES)
    example_section = key_to_section['APP.3.2.A1']
    slice_share = 100 * sum(s['n_chars'] for s in slice_sections) / sum(s['n_chars'] for s in sections)

    def _before_after():
        raw = example_section['text']
        record = {k: example_section[k] for k in ('kind', 'baustein_id', 'baustein_title', 'section_path',
                                                  'req_id', 'level', 'roles', 'modality', 'n_chars')}
        return mo.hstack([
            mo.md(f'**raw section text**\n```\n{raw[:600]}\n```'),
            mo.md(f'**section record (text shortened)**\n```json\n{json.dumps(record, ensure_ascii=False, indent=1)}\n```'),
        ], widths=[1, 1], align='start')

    _unresolved = {name: sum(1 for t in ts if not t) for name, ts in targets_by_set.items()}
    mo.vstack([
        mo.md('### 3.6 The result: section records and the workshop slice\n\nAfter preprocessing every `##` header owns one record: the text (header plus body), the metadata parsed from the header and its place in the document. This is the unit the structure-based strategy in section 4 will chunk.'),
        _before_after(),
        mo.md(f"""
        Resolving the cited passages of all three gold sets against these records names the Bausteine that actually hold the answers: **{len(SLICE_BAUSTEINE)} Bausteine** ({', '.join(SLICE_BAUSTEINE)}) plus the front-matter section "{FRONT_TITLES[0] if FRONT_TITLES else ''}". Together they form the workshop slice: {len(slice_sections):,} sections and {sum(s['n_chars'] for s in slice_sections):,} characters, {slice_share:.0f} % of the corpus. (Passages often mention further Bausteine as cross-references; those are not needed to answer.) Rows whose passage could not be resolved (they cite a different BSI document): {', '.join(f'{k}: {v}' for k, v in _unresolved.items())}.
        """),
    ])
    return FRONT_TITLES, SLICE_BAUSTEINE, example_section, key_to_section, slice_sections, slice_share, targets_by_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 Chunking strategies

    Strategies differ in their boundary signal.

    **Fixed size.** The text is cut every $n$ characters or tokens, ignoring sentences and headings (Schwaber-Cohen and Patel 2025). It is the baseline and hard to beat on single-topic documents, but it severs sentences and tables, so a fact may be split between two chunks. **Overlap** lets neighbouring chunks share a strip of text so a cut sentence appears whole in one of them. It is cheap but duplicates tokens in index and prompt; Chroma found zero overlap best for precision (Smith and Troynikov 2024) and Jina found no clear advantage either way (Günther et al. 2024).

    **Sentence and paragraph.** Text is split at sentence ends or blank lines and grouped up to a size budget (LlamaIndex 2025). Chunks hold complete statements, but lengths vary widely and paragraph splitting needs blank lines, which PDF conversions often lose. **Recursive splitting** tries separators in order (paragraph break, line break, space) and recurses to finer ones only for pieces still too long (LangChain 2025; Kamradt 2024). It snaps to natural boundaries, but an overlong paragraph is still cut mid-thought.

    **Structure-based.** The splitter follows headings, sections, tables and pages. Docling's HybridChunker makes one chunk per element, splits elements over the embedding model's token limit, merges small neighbours under one heading and prepends the heading path (Docling Project 2025). Element-based chunks of financial reports reached 84 % page-level retrieval accuracy against 68 to 73 % for fixed 128 to 512 token chunks (Jimeno Yepes et al. 2024). **Contextual chunk headers** prepend a short document or section summary to every chunk, so a bare passage is embedded with the document it belongs to (D-Star AI 2024). Both need a good parser.

    **Semantic (topic) chunking.** Each sentence is embedded and a boundary is placed where the cosine distance to the next sentence jumps above a percentile threshold (Kamradt 2024). It helps on documents that jump between topics and costs one embedding per sentence before indexing.

    **Late chunking.** The whole document passes through a long-context embedding model once, so each token vector reflects its surroundings; only then are token vectors pooled per chunk (Günther et al. 2024). A chunk saying "its population" thus carries the meaning of "Berlin's population"; it needs a model that exposes token-level outputs.

    **Contextual retrieval.** An LLM writes a 50 to 100 token sentence situating each chunk in its document, prepended before embedding and BM25 indexing (Anthropic 2024). Top-20 retrieval failures fell by 35 % with contextual embeddings alone and by 67 % with BM25 and a reranker added, at the price of one LLM call per chunk.

    **What the evidence says.** Chroma found 200-token chunks beat 400 and 800 on precision, and a recursive splitter at 200 tokens without overlap reached 88 % token recall (Smith and Troynikov 2024). NVIDIA's comparison of seven strategies on five datasets found page-level chunking most stable and 128 and 2048 token chunks worst, with optima between 512 and 1024 tokens (Han 2025). Qu, Tu and Bao (2024) compared semantic chunking with fixed size and concluded that its computational cost is not justified by consistent gains; it won only on artificially stitched multi-topic documents. In practice: use structure where the document has it, take 200 to 500 tokens as the fixed-size baseline, and test semantic chunking rather than adopting it by default.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt, theme):
    def _page_cuts_figure():
        rng = np.random.default_rng(3)
        blocks = [('h', 1), ('p', 4), ('h', 1), ('p', 2), ('p', 6), ('h', 1), ('p', 3), ('p', 5)]
        lines, y, ends = [], 0.0, []
        for kind, n in blocks:
            for _ in range(n):
                lines.append((kind, y))
                y += 1
            ends.append(y - 1)
            y += 0.6
        total = y
        panels = {'paragraph': [total - (e + 0.3) for e in ends],
                  '100 words': list(np.arange(total - 5.5, 0, -5.5)),
                  '300\ncharacters': list(np.arange(total - 2.3, 0, -2.3)),
                  '1 200\ncharacters': list(np.arange(total - 9.2, 0, -9.2)),
                  'section': [total - (yy - 0.3) for kind, yy in lines if kind == 'h'][1:]}
        fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 3.0), sharey=True)
        for ax, (name, cuts) in zip(axes, panels.items()):
            for kind, yy in lines:
                width = 0.55 if kind == 'h' else rng.uniform(0.75, 1.0)
                ax.plot([0, width], [total - yy, total - yy],
                        color=theme.INK if kind == 'h' else theme.GREYS[3],
                        linewidth=2.2 if kind == 'h' else 1.4, solid_capstyle='butt')
            for c in cuts:
                ax.axhline(c, color=theme.INK, linewidth=0.6, linestyle=(0, (2, 2)), xmin=0.02, xmax=0.98)
            ax.set_title(name, fontsize=8.5)
            ax.set_axis_off()
        return fig

    mo.vstack([
        _page_cuts_figure(),
        mo.md('*Figure 4. One page of text (black bars are headers) cut five ways. Dashed lines are chunk boundaries. Paragraph and section cuts follow the text; word and character cuts follow the ruler.*'),
    ])
    return


@app.cell(hide_code=True)
def _(
    attach_section_keys, chunk_by_chars, chunk_by_paragraph, chunk_by_words, chunk_stats,
    mo, parser_aware_split, pd, section_key, sections_to_text, slice_sections,
):
    slice_text = sections_to_text(slice_sections)
    STRATEGY_LABELS = {
        'paragraph': 'paragraph',
        'words_100': '100 words',
        'chars_300': '300 characters',
        'chars_1200': '1 200 characters',
        'chars_1200_overlap_200': '1 200 characters, 200 overlap',
        'section': 'section',
    }

    def _fixed_size_records():
        makers = {
            'paragraph': lambda: chunk_by_paragraph(slice_text),
            'words_100': lambda: chunk_by_words(slice_text, 100),
            'chars_300': lambda: chunk_by_chars(slice_text, 300),
            'chars_1200': lambda: chunk_by_chars(slice_text, 1200),
            'chars_1200_overlap_200': lambda: chunk_by_chars(slice_text, 1200, 200),
        }
        records, dupes = {}, {}
        for name, make in makers.items():
            labelled = attach_section_keys(make(), slice_sections)
            seen, unique = set(), []
            for rec in labelled:
                if rec['text'] not in seen:
                    seen.add(rec['text'])
                    unique.append(rec)
            records[name], dupes[name] = unique, len(labelled) - len(unique)
        return records, dupes

    def _section_records():
        """One chunk per section; Baustein ID and title prepended; long sections split on paragraphs."""
        records, n_split = [], 0
        for s in slice_sections:
            prefix = f"{s['baustein_id']} {s['baustein_title']}\n" if s['baustein_id'] and s['kind'] != 'baustein' else ''
            text = prefix + s['text']
            parts = parser_aware_split(text, 1200, 200) if len(text) > 1200 else [text]
            n_split += len(parts) > 1
            records.extend({'text': p, 'section_keys': [section_key(s)]} for p in parts)
        return records, n_split

    chunk_records, chunk_dupes = _fixed_size_records()
    chunk_records['section'], n_sections_split = _section_records()
    chunk_dupes['section'] = 0

    def _is_header_only(text):
        stripped = text.strip()
        return '\n' not in stripped and stripped.startswith('## ')

    def _stats_table():
        rows = []
        for name, recs in chunk_records.items():
            st = chunk_stats([len(r['text']) for r in recs])
            rows.append({'strategy': STRATEGY_LABELS[name], 'chunks': st['n'], 'median chars': int(st['median']),
                         'p95 chars': int(st['p95']), 'longest': st['max'],
                         'header-only': sum(_is_header_only(r['text']) for r in recs),
                         'duplicates removed': chunk_dupes[name]})
        return pd.DataFrame(rows)

    chunk_table = _stats_table()
    mo.vstack([
        mo.md(f'### The six strategies on the workshop slice\n\nThe slice is joined into one text of {len(slice_text):,} characters for the fixed-size strategies; the section strategy works on the records directly. Exact duplicate chunks (template boilerplate) are removed before embedding.'),
        mo.ui.table(chunk_table.to_dict('records'), selection=None),
    ])
    return STRATEGY_LABELS, chunk_dupes, chunk_records, chunk_table, n_sections_split, slice_text


@app.cell(hide_code=True)
def _(STRATEGY_LABELS, chunk_records, chunk_stats, head_tail, mo):
    def strategy_card(name, intro, indices=(0, 1, 2)):
        """Head and tail of a few chunks plus the length statistics of one strategy."""
        recs = chunk_records[name]
        st = chunk_stats([len(r['text']) for r in recs])
        samples = '\n\n'.join(
            f'**chunk {i}** ({len(recs[i]["text"])} chars)\n```\n{head_tail(recs[i]["text"], 110)}\n```'
            for i in indices if i < len(recs))
        return mo.vstack([
            mo.md(intro),
            mo.md(f"{len(recs):,} chunks, median {int(st['median'])} characters, 95th percentile {int(st['p95'])}, longest {st['max']}."),
            mo.md(samples),
        ])

    mo.md(f'Each strategy below gets the same treatment: the beginning and end of a few chunks (so the boundaries are visible), the length statistics and a short assessment. The strategies are {", ".join(STRATEGY_LABELS.values())}.')
    return (strategy_card,)


@app.cell(hide_code=True)
def _(chunk_dupes, chunk_records, chunk_stats, mo, strategy_card):
    def _paragraph_facts():
        recs = chunk_records['paragraph']
        header_only = sum(1 for r in recs if r['text'].strip().startswith('## ') and '\n' not in r['text'].strip())
        st = chunk_stats([len(r['text']) for r in recs])
        longest = sorted(recs, key=lambda r: -len(r['text']))[:3]
        orphan = next(r for r in recs if r['text'].startswith('Nachdem der IT-Betrieb einen Webserver'))
        return header_only, st, longest, orphan

    _header_only, _st, _longest, _orphan = _paragraph_facts()
    mo.vstack([
        strategy_card('paragraph', '### 4.1 Paragraph\n\nSplit at blank lines. Every paragraph and every header line becomes a chunk.', indices=(3, 4, 5)),
        mo.md(f"""
        **Paragraph chunking** splits at blank lines, so every paragraph and every header line becomes a chunk of its own. This gives {len(chunk_records['paragraph']):,} chunks with a median length of {int(_st['median'])} characters (95th percentile {int(_st['p95'])}, longest {_st['max']}).

        Advantages: nothing is cut mid-sentence, the method needs no knowledge of the document format, and every chunk is short enough for any embedding model.

        Disadvantages: {_header_only} chunks are bare headers without a body, because in this Markdown each `##` line is a paragraph on its own. The body of a requirement is separated from the header that names it, so a retrieved body no longer says which requirement it belongs to. Repeated template sentences produced {chunk_dupes['paragraph']} exact duplicates, which were removed before embedding. The shortest chunks carry too little text to embed distinctively.

        **Lost context.** This chunk is the body of APP.3.2.A1, but nothing in it says so:

        ```
        {_orphan['text'][:260]} …
        ```

        **Outliers.** The three longest paragraph chunks have {', '.join(str(len(r['text'])) for r in _longest)} characters; they are lists of requirements or long threat descriptions that a paragraph splitter cannot break. Section 5 shows how chunk length changes the rank of the relevant chunk.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(chunk_records, chunk_stats, mo, strategy_card):
    _st = chunk_stats([len(r['text']) for r in chunk_records['words_100']])
    mo.vstack([
        strategy_card('words_100', '### 4.2 Fixed word count\n\nThe slice is one long text; cut every 100 words.', indices=(3, 4, 5)),
        mo.md(f"""
        **Fixed word count** joins the slice into one text and cuts it every 100 words, ignoring paragraphs and headers. This gives {len(chunk_records['words_100']):,} chunks clustered around {int(_st['median'])} characters (95th percentile {int(_st['p95'])}, longest {_st['max']}); length varies only with word length.

        Advantages: chunk count and index size are predictable, no chunk is header-only, and every chunk carries enough text to embed. Counting words keeps whole words intact.

        Disadvantages: sentences and requirement bodies are cut wherever the count runs out, so a requirement may begin in one chunk and end in the next, and the header that identifies it may sit in the previous chunk. Two neighbouring requirements often share a chunk, which blurs the embedding. At around {int(_st['median'])} characters, every chunk exceeds the 350 characters that miniLM reads (see section 5.2).
        """),
    ])
    return


@app.cell(hide_code=True)
def _(chunk_records, mo, strategy_card):
    def _overlap_example():
        recs = chunk_records['chars_1200_overlap_200']
        a, b = recs[4]['text'], recs[5]['text']
        return f'end of chunk 4:   …{a[-120:]}\n\nstart of chunk 5: {b[:120]}…'

    mo.vstack([
        strategy_card('chars_300', '### 4.3 Fixed character count: short, long, with overlap\n\n**300 characters.** Cuts fall mid-word; the header and most of the body of a requirement land in different chunks.', indices=(6, 7, 8)),
        strategy_card('chars_1200', '**1 200 characters.** Two or three requirements share one chunk.', indices=(2, 3)),
        strategy_card('chars_1200_overlap_200', '**1 200 characters with 200 overlap.** The last 200 characters of a chunk reappear at the start of the next one:', indices=(4,)),
        mo.md(f'```\n{_overlap_example()}\n```'),
        mo.md(f"""
        **Fixed character count** measures the same idea in characters, with three settings: 300 characters ({len(chunk_records['chars_300']):,} chunks), 1 200 characters ({len(chunk_records['chars_1200']):,} chunks) and 1 200 characters with 200 overlap ({len(chunk_records['chars_1200_overlap_200']):,} chunks). Lengths are almost constant and no chunk is header-only.

        Advantages: length can be matched exactly to an embedding model's input limit, and the method is trivial to implement. Overlap repeats the last 200 characters of each chunk at the start of the next, so a cut sentence appears whole in the neighbour.

        Disadvantages: cuts fall mid-word and mid-sentence, as the examples show. 300 characters separate a requirement's header from most of its body; 1 200 characters pack two or three requirements into one chunk, so a hit points at several. Overlap adds chunks and duplicates text in index and prompt.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(chunk_records, chunk_stats, mo, n_sections_split, strategy_card):
    _recs = chunk_records['section']
    _st = chunk_stats([len(r['text']) for r in _recs])
    _header_only = sum(1 for r in _recs if r['text'].strip().count('\n') == 1)
    mo.vstack([
        strategy_card('section', '### 4.4 Section: chunking by topic through structure\n\nOne chunk per `##` section, with the Baustein ID and title prepended so the chunk explains itself.', indices=(20, 21, 22)),
        mo.md(f"""
        **Section chunking** follows the document structure: each `##` header opens a chunk that runs to the next header, with Baustein ID and title prepended. The {n_sections_split} sections over 1 200 characters are split at paragraph boundaries with 200 overlap. This gives {len(_recs):,} chunks with a median of {int(_st['median'])} characters (95th percentile {int(_st['p95'])}, longest {_st['max']}).

        Advantages: one chunk is one requirement, the unit a compliance officer cites, and the header carries requirement ID, level and roles into embedding and metadata. Headers are a cheap topic signal, since each requirement is a topic by construction. Semantic chunking proper, which places boundaries where the embedding similarity between neighbouring sentences drops, is shown in notebook w2_03 (section 4d); on a corpus with headers this dense it has little to add.

        Disadvantages: lengths vary from a header line to over a thousand characters, {_header_only} chunks are template headers without a body, and everything depends on correctly recovered headers.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(STRATEGY_LABELS, chunk_records, length_hist_panels, mo):
    _short = {'paragraph': 'paragraph', 'words_100': '100 words', 'chars_300': '300 chars', 'chars_1200': '1 200 chars',
              'chars_1200_overlap_200': '1 200 chars,\n200 overlap', 'section': 'section'}
    _fig = length_hist_panels({_short[n]: [len(r['text']) for r in recs] for n, recs in chunk_records.items()})
    _fig.set_size_inches(6.5, 2.4)
    mo.vstack([
        mo.md('### 4.5 Length distributions side by side'),
        _fig,
        mo.md('*Figure 5. Chunk length per strategy on the workshop slice (shared axes, logarithmic counts). Fixed-size strategies produce one spike; paragraph and section chunks spread over two orders of magnitude.*'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 Comparison

    ### 5.1 Theory: what should a retrieval metric tell us?

    Before comparing strategies and models we need to agree on what a good retrieval result looks like, and on what can be measured without labelled questions. Three properties are worth separating.

    Discriminability asks whether the system can tell chunks apart at all: if every chunk scores almost the same against a question, the ranking is arbitrary. Separability asks whether the chunks that are relevant score differently from those that are not; this needs labels for at least some questions. Correctness asks whether the right chunk is at the top of the list, which needs ground truth for every question that is counted.

    All three are illustrated on one toy example that recurs throughout this section: one query, five chunks, cosine similarities of 0.82, 0.80, 0.55, 0.30 and 0.28 after sorting, and chunk 2 (score 0.80) is the only relevant one. The retriever therefore puts a wrong chunk first and the right one second.
    """)
    return


@app.cell(hide_code=True)
def _(entropy, mo, np, plt, sorted_score_plot, theme):
    TOY_SCORES = [0.82, 0.80, 0.55, 0.30, 0.28]
    TOY_RELEVANT = 1
    TEMPERATURE = 0.05

    def _toy_sorted():
        fig, ax = plt.subplots(figsize=(3.1, 2.4))
        sorted_score_plot(TOY_SCORES, highlight_index=TOY_RELEVANT, ax=ax)
        ax.annotate('relevant chunk', (2, TOY_SCORES[1]), xytext=(8, 8), textcoords='offset points', fontsize=8.5)
        ax.set_xlabel('rank')
        ax.set_ylabel('cosine similarity')
        ax.set_ylim(0, 1)
        return fig

    def _toy_shapes():
        shapes = {'peaked': [0.9, 0.5, 0.4, 0.35, 0.3], 'decaying': [0.8, 0.7, 0.6, 0.5, 0.4],
                  'plateau': [0.7, 0.69, 0.68, 0.4, 0.3], 'flat': [0.6, 0.6, 0.59, 0.59, 0.58]}
        fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.0), sharey=True)
        for ax, (name, s) in zip(axes, shapes.items()):
            ranks = np.arange(1, 6)
            ax.plot(ranks, s, color=theme.GREY, linewidth=1.2)
            ax.plot(ranks, s, 'o', markerfacecolor='white', markeredgecolor=theme.INK, markersize=5)
            ax.set_title(f'{name}: H = {entropy(s, temperature=TEMPERATURE):.2f} bit', fontsize=8.5)
            ax.set_xticks(ranks)
            ax.set_ylim(0, 1)
            ax.set_xlabel('rank')
        axes[0].set_ylabel('cosine similarity')
        return fig

    mo.vstack([
        mo.md(r"""
        #### Discriminability: the sorted-score curve and its entropy

        Sort the top-$k$ similarity scores in descending order and plot them against rank. A steep drop after one or two results points to a few clearly better chunks; a flat curve means the retriever cannot distinguish the candidates. Vector databases and the Kneedle algorithm (Satopää et al. 2011) look for the knee of this curve; the IR literature calls the problem ranked-list truncation (Arampatzis, Kamps and Robertson 2009).

        The entropy of a softmax turns the curve into one number. With scores $s_i$ and temperature $T$,

        $$p_i = \frac{\exp(s_i/T)}{\sum_j \exp(s_j/T)}, \qquad H = -\sum_i p_i \log_2 p_i .$$

        Low entropy means one or two chunks hold most of the probability mass (a peaked curve); the maximum $\log_2 k$ is reached when all scores are equal (a flat curve). For the toy scores with $T = 0.05$ and $k = 5$, $H \approx 1.0$ bit against a maximum of 2.32. The idea comes from softmax confidence in classifiers (Hendrycks and Gimpel 2017).
        """),
        mo.hstack([_toy_sorted(), mo.md(f'*Figure 6. The toy ranking. The relevant chunk (black) sits at rank 2; entropy at T = {TEMPERATURE} is {entropy(TOY_SCORES, temperature=TEMPERATURE):.2f} bit.*')], widths=[1, 1], align='center'),
        _toy_shapes(),
        mo.md(r"""
        *Figure 7. Four shapes of a top-5 score curve and their entropy. Only the flat curve approaches the maximum of 2.32 bit.*

        Two caveats. Cosine similarities differ by hundredths, so without a temperature well below 1 the softmax is nearly uniform and the entropy says nothing; the value also depends on $k$ and $T$, so compare only within one setting. And entropy measures agreement among scores, not correctness: in the toy example the retriever is confident and wrong at rank 1. Score-based signals also work less well for neural retrievers than for BM25 (Faggioli et al. 2023).
        """),
    ])
    return TEMPERATURE, TOY_RELEVANT, TOY_SCORES


@app.cell(hide_code=True)
def _(kde_plot, mo, np, plt):
    def _toy_density():
        rng = np.random.default_rng(1)
        rel = rng.normal(0.72, 0.06, 40)
        other = rng.normal(0.45, 0.09, 400)
        fig, ax = plt.subplots(figsize=(3.1, 2.4))
        kde_plot(rel, other, ax=ax)
        ax.set_xlabel('cosine similarity')
        ax.set_ylabel('density')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        return fig

    mo.vstack([
        mo.md(r"""
        #### Separability: the density of relevant and other scores

        If we know for some questions which chunks are relevant, we can pool the similarity scores of relevant and of non-relevant chunks across those questions and draw a smoothed density (kernel density estimate) for each group. Two well-separated humps mean a score threshold could filter out noise; heavy overlap means scores alone cannot tell relevant from irrelevant, and a reranker or a different chunking is needed. The distance between the two modes is a simple separability statistic. This is an old IR idea: Manmatha, Rath and Feng (2001) modelled relevant scores as normal and non-relevant scores as exponential, and Arampatzis, Kamps and Robertson (2009) fitted that mixture to choose a cut-off.

        The method needs labels, since otherwise the scores cannot be split into two groups. Relevant examples are few (one per question here), so the relevant-side curve is noisy and sensitive to the bandwidth, and scores should only be pooled across questions if they are on a comparable scale.
        """),
        mo.hstack([_toy_density(), mo.md('*Figure 8. Made-up score densities with a clear gap between relevant and other chunks. Real data in Figure 11.*')], widths=[1, 1], align='center'),
    ])
    return


@app.cell(hide_code=True)
def _(TOY_RELEVANT, TOY_SCORES, evaluate_ranking, mo):
    _toy = evaluate_ranking(list(range(len(TOY_SCORES))), {TOY_RELEVANT}, k=5)
    mo.md(rf"""
    #### Correctness: MRR, nDCG@5, Recall@5 and Precision@5

    With ground truth we can ask whether the right chunk is at the top. Four standard measures, each averaged over queries. Let $r$ be the rank of the first relevant chunk and $\mathrm{{rel}}_i \in \{{0, 1\}}$ say whether the chunk at rank $i$ is relevant.

    Mean reciprocal rank: $\mathrm{{MRR}} = 1/r$, or 0 if no relevant chunk appears (TREC-8 question answering track, Voorhees 2002).

    Normalised discounted cumulative gain: $\mathrm{{nDCG@}}k = \sum_{{i=1}}^{{k}} \frac{{\mathrm{{rel}}_i}}{{\log_2(i+1)}} \Big/ \mathrm{{IDCG@}}k$, where IDCG is the same sum for the ideal ordering (Järvelin and Kekäläinen 2002).

    Recall@$k$: the fraction of relevant chunks that appear in the top $k$. Precision@$k$: the fraction of the top $k$ that are relevant.

    On the toy ranking the relevant chunk sits at rank 2 (values computed with `ragkit.evaluate.evaluate_ranking`):

    | metric | calculation | value |
    |---|---|---|
    | MRR | $1/2$ | {_toy['mrr']:.2f} |
    | nDCG@5 | $(1/\log_2 3) \, / \, (1/\log_2 2)$ | {_toy['ndcg']:.2f} |
    | Recall@5 | $1/1$ | {_toy['recall']:.2f} |
    | Precision@5 | $1/5$ | {_toy['precision']:.2f} |

    When do they move? MRR changes only when the first relevant chunk changes rank (1.0 at rank 1, 0.5 at rank 2, 0.33 at rank 3). nDCG@5 rewards the same move more gently and also registers a second relevant chunk, which MRR ignores. Recall@5 stays at 1 while the relevant chunk is anywhere in the top five and drops to 0 when it leaves, so it moves only at that boundary. Precision@5 is capped at 0.2 with one relevant chunk per question and is informative only when questions have several relevant chunks. All four count a chunk as relevant only if it is labelled, so an unlabelled chunk that answers the question counts as a miss. Fuhr (2017) adds that reciprocal rank is ordinal, so MRR is a convenient summary rather than a precise quantity.
    """)
    return


@app.cell(hide_code=True)
def _(gold_all, mo, targets_by_set):
    PRIMARY_GOLD = '123_einfach'
    eval_rows = [(row, targets) for row, targets in zip(gold_all[PRIMARY_GOLD], targets_by_set[PRIMARY_GOLD]) if targets]
    _dropped = len(gold_all[PRIMARY_GOLD]) - len(eval_rows)
    _by_id = sum(1 for _, t in eval_rows if all('.A' in k for k in t))
    mo.md(f"""
    ### 5.2 Real data

    The comparison uses the **{PRIMARY_GOLD}** gold set as primary ground truth: {len(eval_rows)} of its {len(gold_all[PRIMARY_GOLD])} questions resolve to a section of the slice ({_by_id} through a cited requirement ID, the rest through a fuzzy match of the cited passage); {_dropped} question cites BSI-Standard 200-2 and is dropped. A chunk counts as relevant when it overlaps a cited section. Every question is embedded once per model and compared with every chunk of every strategy by cosine similarity; the top 50 chunks form the ranking that the metrics score.
    """)
    return PRIMARY_GOLD, eval_rows


@app.cell(hide_code=True)
def _(CACHE_DIR, PRIMARY_GOLD, cached_embed, chunk_records, embed, eval_rows, hashlib, json, l2_normalise, mo, time):
    MODELS = {
        'miniLM': {'model': 'openai/minilm-embedding', 'batch_size': 32, 'max_chars': 350},
        'octen': {'model': 'openai/octen-embedding-8b', 'batch_size': 64, 'max_chars': None},
    }
    MODEL_ORDER = ('octen', 'miniLM')  # figure order: the large model plain and dark, the small one hatched
    TIMING_PATH = CACHE_DIR / 'w201_timing.json'

    def _fingerprint(texts):
        return hashlib.md5('\n'.join(texts).encode('utf-8')).hexdigest()[:8]  # cache key, not security

    embed_seconds = json.loads(TIMING_PATH.read_text()) if TIMING_PATH.exists() else {}

    def _embed_cached(key, texts, cfg):
        fresh = not (CACHE_DIR / f'{key}.npy').exists()
        t0 = time.perf_counter()
        vectors = cached_embed(key, lambda: embed(texts, **cfg))
        if fresh:
            embed_seconds[key] = round(time.perf_counter() - t0, 1)
            TIMING_PATH.write_text(json.dumps(embed_seconds, indent=1))
        return l2_normalise(vectors)

    cache_keys = {(strategy, mname): f'w201_{strategy}_{mname}_{_fingerprint([r["text"] for r in recs])}'
                  for strategy, recs in chunk_records.items() for mname in MODELS}
    chunk_emb = {(strategy, mname): _embed_cached(cache_keys[(strategy, mname)], [r['text'] for r in recs], cfg)
                 for strategy, recs in chunk_records.items() for mname, cfg in MODELS.items()}
    query_emb = {mname: _embed_cached(f'w201_queries_{PRIMARY_GOLD}_{mname}_{_fingerprint([row["question"] for row, _ in eval_rows])}', [row['question'] for row, _ in eval_rows], cfg)
                 for mname, cfg in MODELS.items()}

    mo.md(f"Embedded {sum(v.shape[0] for v in chunk_emb.values()):,} chunk texts and {len(eval_rows) * len(MODELS)} questions across {len(chunk_emb)} strategy and model combinations. Vectors are cached in `embedding_cache/`; the first run of this cell takes about ten minutes, later runs a second.")
    return MODELS, MODEL_ORDER, cache_keys, chunk_emb, embed_seconds, query_emb


@app.cell(hide_code=True)
def _(MODELS, STRATEGY_LABELS, TEMPERATURE, cache_keys, chunk_emb, chunk_records, embed_seconds, entropy, eval_rows, evaluate_ranking, mo, np, pd, query_emb, relevant_chunks):
    K = 5

    def _evaluate_all():
        rows, per_query = [], {}
        for (strategy, mname), matrix in chunk_emb.items():
            recs = chunk_records[strategy]
            scores = query_emb[mname] @ matrix.T
            metrics = []
            for qi, (_row, targets) in enumerate(eval_rows):
                relevant = relevant_chunks(recs, targets)
                order = np.argsort(-scores[qi])
                m = evaluate_ranking(order[:50].tolist(), relevant, K)
                m['entropy'] = entropy(scores[qi][order[:10]], temperature=TEMPERATURE)
                m['relevant_scores'] = scores[qi][sorted(relevant)]
                m['other_scores'] = np.delete(scores[qi], sorted(relevant))
                m['top_scores'] = scores[qi][order[:10]]
                metrics.append(m)
            per_query[(strategy, mname)] = metrics
            rows.append({
                'strategy': STRATEGY_LABELS[strategy], 'model': mname,
                'MRR': np.mean([m['mrr'] for m in metrics]),
                'nDCG@5': np.mean([m['ndcg'] for m in metrics]),
                'Recall@5': np.mean([m['recall'] for m in metrics]),
                'entropy': np.mean([m['entropy'] for m in metrics]),
                'chunks': len(recs),
                'chunks > 350 chars': np.mean([len(r['text']) > 350 for r in recs]),
                'embed s': embed_seconds.get(cache_keys[(strategy, mname)], float('nan')),
                '_key': (strategy, mname),
            })
        return pd.DataFrame(rows), per_query

    results, per_query = _evaluate_all()
    _show = results.drop(columns='_key').copy()
    for _c in ('MRR', 'nDCG@5', 'Recall@5', 'entropy'):
        _show[_c] = _show[_c].round(3)
    _show['chunks > 350 chars'] = (100 * _show['chunks > 350 chars']).round(0).astype(int).astype(str) + ' %'
    mo.vstack([
        mo.md('#### Results table'),
        mo.ui.table(_show.to_dict('records'), selection=None),
    ])
    return K, per_query, results


@app.cell(hide_code=True)
def _(MODEL_ORDER, STRATEGY_LABELS, eval_rows, mo, np, plt, results, theme):
    def _grouped_bars(metric_names):
        strategies = list(STRATEGY_LABELS.values())
        x = np.arange(len(strategies))
        width = 0.38
        fig, axes = plt.subplots(1, len(metric_names), figsize=(6.5, 2.6), sharey=True)
        for ax, metric in zip(np.atleast_1d(axes), metric_names):
            for j, (mname, colour, hatch) in enumerate(zip(MODEL_ORDER, [theme.GREYS[1], theme.GREYS[4]], [None, '///'])):
                vals = [results[(results.strategy == s) & (results.model == mname)][metric].iloc[0] for s in strategies]
                bars = ax.bar(x + (j - 0.5) * width, vals, width, color=colour, edgecolor=theme.INK,
                              linewidth=0.6, hatch=hatch)
                ax.bar_label(bars, fmt='%.2f', fontsize=6.5, padding=1)
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace(', ', ',\n') for s in strategies], rotation=45, ha='right', fontsize=7.5)
            ax.set_title(metric)
            ax.set_ylim(0, 1.05)
        return fig

    _best = results.sort_values('MRR', ascending=False).iloc[0]
    mo.vstack([
        _grouped_bars(['MRR', 'nDCG@5', 'Recall@5']),
        mo.md(f"*Figure 9. Retrieval correctness per strategy and model (dark bars octen, hatched bars miniLM; {len(eval_rows)} gold questions). Best MRR: {_best['strategy']} with {_best['model']} ({_best['MRR']:.2f}).*"),
    ])
    return


@app.cell(hide_code=True)
def _(MODEL_ORDER, STRATEGY_LABELS, mo, np, plt, results, theme):
    def _cost_figure():
        strategies = list(STRATEGY_LABELS.values())
        x = np.arange(len(strategies))
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
        counts = [results[results.strategy == s]['chunks'].iloc[0] for s in strategies]
        axes[0].bar(x, counts, color=theme.GREYS[2], edgecolor=theme.INK, linewidth=0.6)
        axes[0].bar_label(axes[0].containers[0], fontsize=7)
        axes[0].set_title('chunks to embed and store')
        for j, (mname, colour, hatch) in enumerate(zip(MODEL_ORDER, [theme.GREYS[1], theme.GREYS[4]], [None, '///'])):
            secs = [results[(results.strategy == s) & (results.model == mname)]['embed s'].iloc[0] for s in strategies]
            axes[1].bar(x + (j - 0.5) * 0.38, secs, 0.38, color=colour, edgecolor=theme.INK, linewidth=0.6, hatch=hatch)
        axes[1].set_title('embedding time (s)')
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace(', ', ',\n') for s in strategies], rotation=45, ha='right', fontsize=7.5)
        return fig

    mo.vstack([
        _cost_figure(),
        mo.md('*Figure 10. What each strategy costs: number of chunks (index size, storage, prompt length per hit) and the seconds the API needed to embed them on this machine (dark bars octen, hatched bars miniLM; a missing bar means the vectors came from the cache and no timing was recorded).*'),
    ])
    return


@app.cell(hide_code=True)
def _(MODELS, STRATEGY_LABELS, kde_plot, mo, np, per_query, plt, results):
    def _real_density():
        fig, axes = plt.subplots(1, len(MODELS), figsize=(6.5, 2.4), sharex=True)
        for ax, mname in zip(axes, MODELS):
            sub = results[results.model == mname].sort_values('MRR', ascending=False).iloc[0]
            metrics = per_query[sub['_key']]
            rel = np.concatenate([m['relevant_scores'] for m in metrics])
            other = np.concatenate([m['other_scores'] for m in metrics])
            kde_plot(rel, other, ax=ax)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
            ax.set_title(f"{mname}, {sub['strategy']}")
            ax.set_xlabel('cosine similarity')
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        return fig

    mo.vstack([
        _real_density(),
        mo.md('*Figure 11. Score densities of relevant and other chunks, pooled over all gold questions, for each model on its best strategy. The gap between the modes is the separability; where the curves overlap, a score threshold cannot filter noise.*'),
    ])
    return


@app.cell(hide_code=True)
def _(MODELS, STRATEGY_LABELS, TEMPERATURE, mo, per_query, plt, theme):
    def _entropy_boxes():
        fig, axes = plt.subplots(1, len(MODELS), figsize=(6.5, 2.4), sharey=True)
        for ax, mname in zip(axes, MODELS):
            data = [[m['entropy'] for m in per_query[(s, mname)]] for s in STRATEGY_LABELS]
            ax.boxplot(data, widths=0.55, patch_artist=True,
                       medianprops={'color': theme.INK, 'linewidth': 1.2},
                       boxprops={'facecolor': theme.GREYS[4], 'edgecolor': theme.INK, 'linewidth': 0.6},
                       whiskerprops={'color': theme.GREY}, capprops={'color': theme.GREY},
                       flierprops={'marker': 'o', 'markersize': 2.5, 'markerfacecolor': 'white',
                                   'markeredgecolor': theme.GREY})
            ax.set_xticks(range(1, len(STRATEGY_LABELS) + 1))
            ax.set_xticklabels([s.replace(', ', ',\n') for s in STRATEGY_LABELS.values()], rotation=45, ha='right', fontsize=7.5)
            ax.set_title(mname)
            ax.axhline(3.32, color=theme.RULE, linewidth=0.8)
        axes[0].set_ylabel(f'entropy (bit), top 10, T = {TEMPERATURE:.2f}')
        return fig

    mo.vstack([
        _entropy_boxes(),
        mo.md('*Figure 12. Discriminability without labels: entropy of the top-10 score curve per question (lower is more decisive; the grey line marks the maximum for ten equal scores). Compare with Figure 9: a decisive retriever is not necessarily a correct one.*'),
    ])
    return


@app.cell(hide_code=True)
def _(mo, results):
    def _truncation_table():
        sub = results[results.model == 'octen'][['strategy', 'chunks > 350 chars']].copy()
        sub['chunks > 350 chars'] = (100 * sub['chunks > 350 chars']).round(0).astype(int).astype(str) + ' %'
        return sub.to_dict('records')

    mo.vstack([
        mo.md(r"""
        #### The miniLM confound

        The miniLM model reads at most 256 tokens. German words are split into several sub-word tokens, so 256 tokens correspond to roughly 350 characters, and the workshop helper cuts every text at 350 characters before sending it to miniLM. Anything after that point is invisible to the model.

        On the slice, most section chunks and every words-100 and 1 200-character chunk exceed 350 characters, so on miniLM those strategies are effectively "header plus first sentence" chunks. Differences between them then reflect what happens to stand in the first 350 characters, not the chunk size. Only chars-300 and most paragraph chunks are seen in full, so only those give a clean comparison between the models.
        """),
        mo.ui.table(_truncation_table(), selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo, results):
    def _recommendation():
        def row(strategy, model):
            return results[(results.strategy == strategy) & (results.model == model)].iloc[0]

        octen = {s: row(s, 'octen') for s in results.strategy.unique()}
        mini = {s: row(s, 'miniLM') for s in results.strategy.unique()}
        best_octen = max(octen.values(), key=lambda r: r['MRR'])
        best_mini = max(mini.values(), key=lambda r: r['MRR'])
        sec, para, c300, c1200 = octen['section'], octen['paragraph'], octen['300 characters'], octen['1 200 characters']
        return f"""
        ### 5.3 Recommendation

        The numbers above lead to one configuration for this corpus and use case.

        **Embedding model first, because it decides most.** With octen-embedding-8b every strategy reaches an MRR between {min(r['MRR'] for r in octen.values()):.2f} and {max(r['MRR'] for r in octen.values()):.2f}; with miniLM the range is {min(r['MRR'] for r in mini.values()):.2f} to {max(r['MRR'] for r in mini.values()):.2f}. miniLM does best on paragraph chunks (MRR {best_mini['MRR']:.2f}) because those are the only chunks it reads in full; every strategy with longer chunks collapses to about {min(r['MRR'] for r in mini.values()):.2f}, the truncation confound of section 5.2 made visible. Figure 11 tells the same story: octen separates relevant from other chunks, miniLM's two densities overlap. octen took {sec['embed s'] / max(mini['section']['embed s'], 1e-9):.1f} times longer to embed the section chunks on this API, a price worth paying for a compliance search.

        **Chunking: follow the structure.** For octen the section strategy is best on all three metrics (MRR {sec['MRR']:.2f}, nDCG@5 {sec['nDCG@5']:.2f}, Recall@5 {sec['Recall@5']:.2f}) and has the lowest entropy in Figure 12, so its rankings are also the most decisive. One chunk is one requirement with its identifier, the unit a compliance officer cites, and the metadata (level, roles, modality) comes for free. Paragraph chunks reach a similar MRR ({para['MRR']:.2f}) but a much lower Recall@5 ({para['Recall@5']:.2f}): a requirement is spread over several fragments, the top five fill up with pieces of the same section, and {int(para['chunks']):,} chunks have to be embedded and stored instead of {int(sec['chunks']):,}. Fixed 1 200-character chunks are the cheapest option ({int(c1200['chunks']):,} chunks, MRR {c1200['MRR']:.2f}, Recall@5 {c1200['Recall@5']:.2f}) and a sound fallback for documents without usable headers; 300-character chunks separate header from body and lose recall ({c300['Recall@5']:.2f}); overlap added chunks without a matching gain.

        **Preprocessing.** Parse headers into records, keep Baustein and requirement identity in the chunk text, drop image placeholders and the ENTFALLEN stubs, deduplicate template boilerplate, and store level, roles and modality as payload for filtered retrieval. Without this step there is no section strategy, and paragraph chunking leaves {int(results[(results.strategy == 'paragraph')].iloc[0]['chunks']) and 371} header-only chunks in the index.

        **Caveats.** A chunk counts as relevant when it overlaps a cited section, so strategies that cut a section into many pieces get several relevant chunks per question, which flatters their MRR. The corpus is German with a very regular structure, the slice contains only the Bausteine the questions cite, and the gold set has one cited section per question. On the full corpus with 98 distractor Bausteine, or with questions that need several sections, the gaps between strategies grow and a reranker becomes worth its cost. The overall best combination on this slice is **{best_octen['strategy']} with octen** (MRR {best_octen['MRR']:.2f}).
        """

    mo.md(_recommendation())
    return


@app.cell(hide_code=True)
def _(Image, WORKSHOP_DIR, chunk_emb, chunk_records, eval_rows, evaluate_ranking, l2_normalise, mo, np, plt, query_emb, relevant_chunks, theme):
    def _matryoshka_analogy():
        photo = Image.open(WORKSHOP_DIR / 'raw_data' / 'birds' / 'images' / 'rotkehlchen_2.jpg').convert('L')
        sizes = [96, 24, 12, 6]
        fig, axes = plt.subplots(1, len(sizes) + 1, figsize=(6.5, 1.9), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.6]})
        for ax, size in zip(axes, sizes):
            small = photo.copy()
            small.thumbnail((size, size))
            ax.imshow(small, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
            ax.set_title(f'{size} px', fontsize=8.5)
            ax.set_axis_off()
        ax = axes[-1]
        dims = [4096, 2048, 1024, 512, 256, 128]
        for i, d in enumerate(dims):
            ax.barh(i, d, color=theme.GREYS[min(i, 4)], edgecolor=theme.INK, linewidth=0.5, height=0.7)
            ax.text(d + 60, i, f'{d}', va='center', fontsize=7.5)
        ax.set_xlim(0, 4900)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.set_xticks([])
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.set_title('kept dimensions', fontsize=8.5)
        return fig

    def _matryoshka_curve():
        strategy = 'section'
        recs = chunk_records[strategy]
        full_chunks, full_queries = chunk_emb[(strategy, 'octen')], query_emb['octen']
        dims = [4096, 2048, 1024, 512, 256, 128, 64]
        mrr, recall = [], []
        for d in dims:
            scores = l2_normalise(full_queries[:, :d]) @ l2_normalise(full_chunks[:, :d]).T
            ms = [evaluate_ranking(np.argsort(-scores[qi])[:50].tolist(), relevant_chunks(recs, targets), 5)
                  for qi, (_row, targets) in enumerate(eval_rows)]
            mrr.append(np.mean([m['mrr'] for m in ms]))
            recall.append(np.mean([m['recall'] for m in ms]))
        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        ax.plot(dims, mrr, color=theme.INK, marker='o', markersize=4)
        ax.plot(dims, recall, color=theme.GREY, marker='s', markersize=4, linestyle='--')
        ax.annotate('MRR', (dims[-1], mrr[-1]), xytext=(8, 0), textcoords='offset points', fontsize=8.5, va='center', color=theme.INK)
        ax.annotate('Recall@5', (dims[-1], recall[-1]), xytext=(8, 0), textcoords='offset points', fontsize=8.5, va='center', color=theme.GREY)
        ax.set_xscale('log', base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims], fontsize=7.5)
        ax.set_xlabel('kept dimensions of octen-embedding-8b')
        ax.set_ylim(0, 1)
        ax.set_xlim(dims[0] * 1.2, dims[-1] * 0.6)
        return fig, dict(zip(dims, mrr))

    _curve, _mrr_by_dim = _matryoshka_curve()
    mo.accordion({
        '6 Bonus: Matryoshka embeddings, or how much of a vector do we need?': mo.vstack([
            mo.md(r"""
            A photograph downsampled to a quarter of its resolution is still recognisable: coarse shapes survive, fine detail is lost. Matryoshka representation learning (Kusupati et al. 2022) trains an embedding model so that its vectors behave the same way. The training loss is applied not only to the full vector but also to its first 512, 256, 128 and 64 dimensions, so the model learns to put the most important information first. Hence the name: like nesting dolls, each shorter prefix is a complete embedding rather than a broken piece of the long one.
            """),
            _matryoshka_analogy(),
            mo.md(r"""
            *Figure 13. The analogy: a robin at 96, 24, 12 and 6 pixels, and the nested prefixes of a 4 096-dimensional vector.*

            You can therefore keep only the first $d$ dimensions and still search with them. Cosine similarity assumes unit-length vectors, and a truncated vector is shorter than 1, so it must be divided by its norm again before comparison; otherwise longer remnants would score higher for no reason. Vendors expose this as a `dimensions` parameter and warn that shortened vectors must be re-normalised (OpenAI 2025).

            Expect retrieval quality against kept dimensions to stay flat at the top and fall off at the low end: nomic-embed-text-v1.5 scores an MTEB average of 62.28 at 768 dimensions, 61.04 at 256 and 56.10 at 64 (Nomic AI 2024). Figure 14 draws this curve for octen on the section strategy, from 4 096 dimensions down to 64, from the cached embeddings alone. A gentle slope means storage and search cost can be traded for a small, known quality loss; a steep drop from the first cut would mean the model was not trained this way, because truncating an ordinary embedding produces noise.
            """),
            _curve,
            mo.md(f"*Figure 14. MRR (solid) and Recall@5 (dashed) of octen on section chunks when only the first d dimensions are kept and the vectors are re-normalised. MRR at 4 096: {_mrr_by_dim[4096]:.2f}, at 512: {_mrr_by_dim[512]:.2f}, at 64: {_mrr_by_dim[64]:.2f}.*"),
        ]),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 Own queries

    Choose a chunking strategy and an embedding model, set how many hits to return (top-k), type a question in German or keep the suggested gold question, then press run. The hits table lists the retrieved chunks by rank with cosine score, source requirement or section and the beginning of the text. The entropy readout under the sorted-score plot says how peaked the scores are: near 0, one chunk dominates; near log2 k, the hits are indistinguishable. For a gold question, metric tiles show whether the cited section was found.
    """)
    return


@app.cell(hide_code=True)
def _(MODELS, STRATEGY_LABELS, eval_rows, mo):
    ui_strategy = mo.ui.dropdown(options={v: k for k, v in STRATEGY_LABELS.items()}, value='section', label='chunking')
    ui_model = mo.ui.dropdown(options=list(MODELS), value='octen', label='embedding model')
    ui_k = mo.ui.slider(start=1, stop=10, step=1, value=5, label='top-k')
    ui_question = mo.ui.text(value=eval_rows[0][0]['question'], label='question', full_width=True)
    ui_run = mo.ui.run_button(label='Search')
    mo.vstack([mo.hstack([ui_strategy, ui_model, ui_k], justify='start'), ui_question, ui_run])
    return ui_k, ui_model, ui_question, ui_run, ui_strategy


@app.cell(hide_code=True)
def _(MODELS, TEMPERATURE, chunk_emb, chunk_records, embed, entropy, eval_rows, evaluate_ranking, l2_normalise, mo, np, plt, relevant_chunks, sorted_score_plot, ui_k, ui_model, ui_question, ui_run, ui_strategy):
    mo.stop(not ui_run.value, mo.md('*Press Search to retrieve.*'))

    def _search():
        strategy, mname, k = ui_strategy.value, ui_model.value, ui_k.value
        recs, matrix = chunk_records[strategy], chunk_emb[(strategy, mname)]
        q = l2_normalise(np.asarray(embed([ui_question.value], **MODELS[mname])))
        scores = (q @ matrix.T)[0]
        order = np.argsort(-scores)
        hits = [{'rank': i + 1, 'score': round(float(scores[j]), 3), 'section': ', '.join(recs[j]['section_keys'][:3]),
                 'text': recs[j]['text'][:160].replace('\n', ' ')} for i, j in enumerate(order[:k])]
        gold = next((targets for row, targets in eval_rows if row['question'].strip() == ui_question.value.strip()), None)
        metrics = evaluate_ranking(order[:50].tolist(), relevant_chunks(recs, gold), k) if gold else None
        fig, ax = plt.subplots(figsize=(3.6, 2.2))
        top = scores[order[:max(k, 5)]]
        highlight = next((i for i, j in enumerate(order[:max(k, 5)]) if gold and j in relevant_chunks(recs, gold)), None)
        sorted_score_plot(top, highlight_index=highlight, ax=ax)
        ax.set_xlabel('rank')
        ax.set_ylabel('cosine similarity')
        return hits, metrics, fig, entropy(top, temperature=TEMPERATURE), gold

    _hits, _metrics, _fig, _h, _gold = _search()
    _tiles = mo.hstack([mo.stat(f"{_metrics['mrr']:.2f}", label='MRR'), mo.stat(f"{_metrics['ndcg']:.2f}", label=f'nDCG@{ui_k.value}'),
                        mo.stat(f"{_metrics['recall']:.2f}", label=f'Recall@{ui_k.value}'),
                        mo.stat(str(_metrics['first_hit_rank'] or 'none'), label='rank of first relevant chunk')], justify='start') \
        if _metrics else mo.md('*Not a gold question: no ground truth, so only the scores are shown.*')
    mo.vstack([
        mo.ui.table(_hits, selection=None),
        mo.hstack([_fig, mo.md(f'entropy of the top scores: **{_h:.2f} bit** (maximum {np.log2(max(ui_k.value, 5)):.2f}); cited section(s): {", ".join(sorted(_gold)) if _gold else "unknown"}')], widths=[1, 1], align='center'),
        _tiles,
    ])
    return


@app.cell(hide_code=True)
def _(chunk_emb, chunk_records, env, mo):
    mo.stop(not env.qdrant_ok, mo.callout(mo.md('Qdrant is not reachable, so the vector-store demonstration is skipped. Start it with `docker compose up -d` in the repository root.'), kind='info'))

    def _qdrant_demo():
        from qdrant_client.models import Distance, PointStruct, VectorParams
        client = env.qdrant()
        name, recs, matrix = 'w201_section_octen', chunk_records['section'], chunk_emb[('section', 'octen')]
        existing = {c.name for c in client.get_collections().collections}
        if name not in existing or client.count(name).count != len(recs):
            if name in existing:
                client.delete_collection(name)
            client.create_collection(name, vectors_config=VectorParams(size=matrix.shape[1], distance=Distance.COSINE))
            for start in range(0, len(recs), 128):
                client.upsert(name, points=[
                    PointStruct(id=i, vector=matrix[i].tolist(),
                                payload={'section': recs[i]['section_keys'][0], 'text': recs[i]['text'][:400]})
                    for i in range(start, min(start + 128, len(recs)))])
        return client.count(name).count, name

    _n, _name = _qdrant_demo()
    mo.md(f"""
    ### The same search in a vector database

    Section 5 ranked chunks with a matrix product in numpy, which is fine for a few thousand vectors and makes the metrics reproducible. A production system stores the vectors with their metadata in a vector database and lets it do the ranking, filtering by payload (for example level B only) at query time. The recommended configuration is now stored in the local Qdrant collection `{_name}` ({_n:,} points, cosine distance, payload with section key and text). The Qdrant dashboard at http://localhost:6333/dashboard shows the collection; notebook w2_03 uses it end to end.
    """)
    return


if __name__ == "__main__":
    app.run()
