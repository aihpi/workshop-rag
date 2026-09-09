# ruff: noqa: PLR1711, B018  marimo cells end with an explicit return and render a bare expression

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

    The setting is a compliance officer writing a security concept, asking things like *"Welche Anforderungen gelten für die Protokollierung auf einem Webserver?"*. A useful answer quotes the requirement rather than paraphrasing it, so retrieval has to find the right section.

    By the end you will be able to:

    - **explain** why the chunk, not the document, decides what a RAG system can find;
    - **choose** a chunking configuration for a corpus you have not seen, and name the evidence you chose it on;
    - **separate** what can be measured without labelled questions from what cannot.

    Then you go back to the playground and find out whether that changed your score.
    """)
    return


@app.cell(hide_code=True)
def _(grid_questions, mo):
    _easy = int((grid_questions.difficulty == 'easy').sum()) if grid_questions is not None else 122
    _complex = int((grid_questions.difficulty == 'complex').sum()) if grid_questions is not None else 42
    mo.md(rf"""
    ### 1.1 The data

    **Corpus.** The [IT-Grundschutz-Kompendium, edition 2023](https://www.bsi.bund.de/dok/it-grundschutz-kompendium) of the German Federal Office for Information Security (BSI).

    **Questions.** The GSKI gold sets, {_easy} easy and {_complex} complex, so {_easy + _complex} in total. Every row carries a question, an answer and the cited passage (*Fundstelle*), which is what makes a hit checkable.
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

    Nothing is embedded while you read. Every number below was measured once, offline, over all 111 Bausteine, so the notebook opens instantly and no two figures can disagree. Expand any cell to see the code that drew it.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json  # the measured extras written by tools/run_extras.py

    import marimo as mo  # the reactive notebook itself
    import matplotlib.pyplot as plt  # figures
    import numpy as np  # ranking arithmetic in the interactive parts
    import pandas as pd  # the measured result tables
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
    from PIL import Image  # the photo used in the Matryoshka analogy

    from ragkit import theme  # HPI look for notebook chrome and figures
    from ragkit.chunk import (  # what section 3 needs; no chunking happens at runtime
        build_sections,
        classify_header,
        parse_requirement_header,
    )
    from ragkit.config import DATA_DIR, WORKSHOP_DIR
    from ragkit.evaluate import (  # gold sets, relevance labels, ranking metrics
        evaluate_ranking,
        gold_targets,
        load_gold,
        section_key,
        section_texts,
    )
    from ragkit.search import ndcg_graded  # graded nDCG for the correctness dashboard
    from ragkit.viz import (
        kde_plot,
        length_hist_panels_from_bins,
        md_table,
        sorted_score_plot,
    )

    theme.apply_mpl()
    return (
        DATA_DIR,
        FancyArrowPatch,
        FancyBboxPatch,
        Image,
        Patch,
        WORKSHOP_DIR,
        build_sections,
        classify_header,
        evaluate_ranking,
        gold_targets,
        json,
        kde_plot,
        length_hist_panels_from_bins,
        load_gold,
        md_table,
        mo,
        ndcg_graded,
        np,
        parse_requirement_header,
        pd,
        plt,
        section_key,
        section_texts,
        sorted_score_plot,
        theme,
    )


@app.cell(hide_code=True)
def _(DATA_DIR, json, mo, pd):
    MD_PATH = DATA_DIR / 'IT_Grundschutz_Kompendium_Edition2023.md'
    GOLD_SETS = ('123_einfach', '43_komplex')
    GRID_DIR = DATA_DIR / 'grid'
    METRIC_ANIMATIONS = [  # rendered by tools/make_metric_animations.py, served from public/
        ('/public/img/metric_rank_sinks.gif', 'One relevant chunk sinks down the list.'),
        ('/public/img/metric_cutoff_moves.gif', 'The same hit list, a moving cut-off.'),
        ('/public/img/metric_more_relevant.gif', 'More chunks become relevant.'),
    ]

    # Everything measured was measured offline by `tools/run_grid.py`; this notebook only reads it,
    # so it needs no API key, no Qdrant and no network.
    _grid = GRID_DIR / 'grid_scores.parquet'
    grid_scores = pd.read_parquet(_grid) if _grid.exists() else None
    _hits = GRID_DIR / 'grid_hits.parquet'
    grid_hits = pd.read_parquet(_hits) if _hits.exists() else None
    _questions = GRID_DIR / 'grid_questions.parquet'
    grid_questions = pd.read_parquet(_questions) if _questions.exists() else None
    _extras = GRID_DIR / 'extras.json'
    EXTRAS = json.loads(_extras.read_text(encoding='utf-8')) if _extras.exists() else None

    # Three figures need the embedding API, which this notebook deliberately never calls. They are
    # rendered once into public/img/ and embedded as files; this says how to redraw them.
    REGENERATE = mo.md(
        'This figure needs the embedding API, so it is not drawn while you read. It was rendered once '
        'into `public/img/` and is committed with the repository. To redraw it after the corpus or the '
        'question set changes, run from `notebooks/`:\n\n'
        '```bash\nuv run python -m tools.run_grid --stage all\nuv run python -m tools.run_extras\n```')

    _view = mo.md('Measurements read from `data/grid/`.')
    if grid_scores is None or EXTRAS is None:
        _view = mo.callout(mo.md(
            'The measured results in `data/grid/` are missing or incomplete. They are committed with '
            'the repository; rebuild them from `notebooks/` with `uv run python -m tools.run_grid '
            '--stage all` followed by `uv run python -m tools.run_extras`.'), kind='warn')
    _view
    return (
        EXTRAS,
        GOLD_SETS,
        GRID_DIR,
        MD_PATH,
        METRIC_ANIMATIONS,
        REGENERATE,
        grid_hits,
        grid_questions,
        grid_scores,
    )


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
        ]
        return '| | |\n|---|---|\n' + '\n'.join(f'| {a} | {b} |' for a, b in rows)

    mo.md('### Corpus at a glance\n\n' + _corpus_table())
    return gold_all, md_raw, sections, texts_flat


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        'Where such data comes from': mo.md(r"""
    Nothing is crawled here. The Kompendium was converted from the [BSI PDF](https://www.bsi.bund.de/dok/it-grundschutz-kompendium) with [Docling](https://docling-project.github.io/docling/), which is the subject of notebook w2_03, and the gold questions were written by domain experts in the GSKI pilot project.
    """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 Preprocessing

    Chunking cuts whatever it is given, so what the text contains is decided here. **Removed**: anything uninformative, such as `<!-- image -->` placeholders and the template preamble "Diese Anforderung ist entfallen". **Moved into metadata**: everything the header line encodes. The corpus carries no bold, italics or figure captions, because none survived the conversion.
    """)
    return


@app.cell(hide_code=True)
def _(example_section, json, md_raw, mo):
    def _raw_section(marker='## APP.3.2.A1 '):
        """Exactly the section the record on the right describes: its header line to the next one."""
        start = md_raw.index(marker)
        return md_raw[start:md_raw.index('\n## ', start + len(marker))].strip()

    def _record():
        keys = ('kind', 'baustein_id', 'baustein_title', 'section_path', 'req_id', 'level', 'roles',
                'modality', 'n_chars')
        return json.dumps({k: example_section[k] for k in keys}, ensure_ascii=False, indent=1)

    mo.vstack([
        mo.md('### 3.1 Before and after'),
        mo.hstack([
            mo.md(f'**as the converter left it**\n```markdown\n{_raw_section()}\n```'),
            mo.md(f'**as the notebook indexes it**\n```json\n{_record()}\n```'),
        ], widths=[1, 1], align='start'),
        mo.md('*The same requirement on both sides. The header line became five fields, the modal verbs were counted, and the body travels on unchanged as the chunkable text. Elsewhere in the corpus preprocessing also deletes: 277 sections whose entire body is the stub "ENTFALLEN", and every `<!-- image -->` the converter left behind.*'),
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

    view_32 = mo.vstack([
        mo.md('### 3.2 Classifying headers\n\nEvery `##` line is matched against four patterns in a fixed order: requirement, Baustein, elementary threat, template section. What none of them matches is front matter (preface, glossary, table of contents) or a stray label the converter promoted to a heading.'),
        _header_kinds_figure(),
        mo.md(f'*Figure 2. Header kinds in the Kompendium Markdown ({len(md_raw.splitlines()):,} lines, {int(header_kinds.sum()):,} headers). Requirements dominate; every Baustein contributes about 19 of them.*'),
    ])
    return (view_32,)


@app.cell(hide_code=True)
def _(md_table, mo, parse_requirement_header, pd, sections):
    def _parsed_examples():
        picks = ['APP.3.2.A1', 'ISMS.1.A1', 'OPS.1.1.6.A2', 'ISMS.1.A14', 'INF.13.A18']
        rows = []
        for s in sections:
            if s['req_id'] in picks:
                parsed = parse_requirement_header(s['text'].split('\n')[0])
                rows.append({'requirement': parsed['req_id'], 'title': parsed['title'][:60],
                             'level': parsed['level'], 'roles': ', '.join(parsed['roles']) or '(none)'})
        return pd.DataFrame(rows)

    view_33 = mo.vstack([
        mo.md('### 3.3 Parsing requirement headers\n\nA requirement header has a fixed shape: identifier, title, level in parentheses and optionally the responsible roles in brackets. Parsing it once gives three pieces of metadata that a compliance officer filters by (level B/S/H, roles, Baustein) and that the chunk text should keep, because a hit is only useful if it names the requirement it comes from.'),
        mo.md(md_table(_parsed_examples().to_dict('records'))),
    ])
    return (view_33,)


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

    view_34 = mo.md('### 3.4 Furniture, boilerplate and the text-versus-metadata decision\n\n' + _boilerplate_table())
    return (view_34,)


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

    view_35 = mo.vstack([
        mo.md('### 3.5 Emphasis: the modal verbs\n\nThe Kompendium defines its own emphasis. A capitalised MUSS or DARF NICHT marks a mandatory requirement, SOLLTE a strong recommendation. The counts per level show the pattern the authors follow: basic requirements are phrased with MUSS, standard and elevated ones with SOLLTE. We keep the verbs in the chunk text (the model should see them) and additionally store the counts as metadata, so a query can be limited to mandatory requirements.'),
        _modality_figure(),
        mo.md('*Figure 3. Modal verbs in active requirement bodies, by level.*'),
    ])
    return (view_35,)


@app.cell(hide_code=True)
def _(mo):
    view_bold = mo.md(r"""
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
    return (view_bold,)


@app.cell(hide_code=True)
def _(gold_all, gold_targets, mo, section_key, sections, texts_flat):
    targets_by_set = {name: [gold_targets(r['fundstelle'], sections, texts_flat) for r in rows]
                      for name, rows in gold_all.items()}
    key_to_section = {section_key(s): s for s in sections}
    _cited_keys = {k for ts in targets_by_set.values() for keys in ts for k in keys}
    CITED_BAUSTEINE = sorted({key_to_section[k]['baustein_id'] for k in _cited_keys
                              if key_to_section[k]['baustein_id']})
    example_section = key_to_section['APP.3.2.A1']

    _unresolved = {name: sum(1 for t in ts if not t) for name, ts in targets_by_set.items()}
    view_36 = mo.md(f"""
    ### 3.6 The result: section records

    Every `##` header owns one record, as §3.1 showed: the text, the metadata parsed from the header and its place in the document. This is the unit the section strategy chunks.

    Resolving the cited passages against these records shows where the answers live: **{len(CITED_BAUSTEINE)} of the 111 Bausteine** ({', '.join(CITED_BAUSTEINE)}). The other {111 - len(CITED_BAUSTEINE)} are indexed anyway, as distractors. Rows whose passage resolves to no section at all, because they cite a different BSI document: {', '.join(f'{k}: {v}' for k, v in _unresolved.items())}.
    """)
    return example_section, view_36


@app.cell(hide_code=True)
def _(mo, view_32, view_33, view_34, view_35, view_36, view_bold):
    mo.vstack([
        mo.md(r"""
    The five steps that produced it are below, for whenever someone wants to discuss them. Notebook w2_03 does document conversion properly.
    """),
        mo.accordion({
            '3.2 Classifying headers': view_32,
            '3.3 Parsing requirement headers': view_33,
            '3.4 Furniture, boilerplate and the text-versus-metadata decision': view_34,
            '3.5 Emphasis: the modal verbs': view_35,
            'What about bold, italics and captions?': view_bold,
            '3.6 The result: section records': view_36,
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 Chunking strategies

    Strategies differ in one thing: **what they treat as a boundary signal.** Here is the same page of text cut five ways.
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
def _(mo):
    mo.md(r"""
    | strategy | cuts on | speaks for it | speaks against it |
    |---|---|---|---|
    | **Paragraph** | blank lines | never cuts mid-sentence<br>needs no parser<br>every chunk fits any model | a third of the chunks are bare headers<br>a body is separated from the header that names it |
    | **Fixed word count** | a word counter | predictable chunk count and index size<br>keeps whole words | ignores every boundary<br>long chunks silently exceed what the model reads |
    | **Fixed character count** | a character counter | length matched exactly to a model's limit<br>trivial to implement | cuts mid-sentence<br>one requirement runs across two chunks, and two share a third |
    | **Section** | the document's own headings | one requirement per chunk<br>the chunk carries its own identifier and metadata | needs a parser<br>needs a document that has structure at all |

    Overlap is a modifier on the two fixed-size rows: neighbouring chunks share a strip of text, so a cut sentence survives whole somewhere, at the price of duplicating it in the index and in the prompt.

    Three of the four fail in a way you can see. Below is each failure on this corpus; section 5 measures what it costs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        'Read more: the wider family of strategies, and what the literature measured':
        mo.md(r"""
    **Fixed size.** The text is cut every $n$ characters or tokens, ignoring sentences and headings ([Pinecone](https://www.pinecone.io/learn/chunking-strategies/)). It is the baseline and hard to beat on single-topic documents, but it severs sentences and tables, so a fact may be split between two chunks. **Overlap** lets neighbouring chunks share a strip of text so a cut sentence appears whole in one of them. It is cheap but duplicates tokens in index and prompt; [Chroma](https://research.trychroma.com/evaluating-chunking) found zero overlap best for precision and [Jina](https://arxiv.org/abs/2409.04701) found no clear advantage either way.

    **Sentence and paragraph.** Text is split at sentence ends or blank lines and grouped up to a size budget ([LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/)). Chunks hold complete statements, but lengths vary widely and paragraph splitting needs blank lines, which PDF conversions often lose. **Recursive splitting** tries separators in order (paragraph break, line break, space) and recurses to finer ones only for pieces still too long ([LangChain](https://python.langchain.com/docs/concepts/text_splitters/), [Kamradt](https://github.com/FullStackRetrieval-com/RetrievalTutorials)). It snaps to natural boundaries, but an overlong paragraph is still cut mid-thought.

    **Structure-based.** The splitter follows headings, sections, tables and pages. Docling's [HybridChunker](https://docling-project.github.io/docling/concepts/chunking/) makes one chunk per element, splits elements over the embedding model's token limit, merges small neighbours under one heading and prepends the heading path. Element-based chunks of financial reports reached 84 % page-level retrieval accuracy against 68 to 73 % for fixed 128 to 512 token chunks ([Jimeno Yepes et al. 2024](https://arxiv.org/abs/2402.05131)). **Contextual chunk headers** prepend a short document or section summary to every chunk ([dsRAG](https://github.com/D-Star-AI/dsRAG)). Both need a good parser.

    **Semantic (topic) chunking.** Each sentence is embedded and a boundary is placed where the cosine distance to the next sentence jumps above a percentile threshold ([Kamradt](https://github.com/FullStackRetrieval-com/RetrievalTutorials)). It helps on documents that jump between topics and costs one embedding per sentence before indexing.

    **Late chunking.** The whole document passes through a long-context embedding model once, so each token vector reflects its surroundings; only then are token vectors pooled per chunk ([Günther et al. 2024](https://arxiv.org/abs/2409.04701)). A chunk saying "its population" thus carries the meaning of "Berlin's population"; it needs a model that exposes token-level outputs.

    **Contextual retrieval.** An LLM writes a 50 to 100 token sentence situating each chunk in its document, prepended before embedding and BM25 indexing ([Anthropic 2024](https://www.anthropic.com/news/contextual-retrieval)). Top-20 retrieval failures fell by 35 % with contextual embeddings alone and by 67 % with BM25 and a reranker added, at the price of one LLM call per chunk.

    **What the evidence says.** [Chroma](https://research.trychroma.com/evaluating-chunking) found 200-token chunks beat 400 and 800 on precision, and a recursive splitter at 200 tokens without overlap reached 88 % token recall. [NVIDIA](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/) compared seven strategies on five datasets and found page-level chunking most stable, 128 and 2048 token chunks worst, and optima between 512 and 1024 tokens. [Qu, Tu and Bao (2024)](https://arxiv.org/abs/2410.13070) concluded that the computational cost of semantic chunking is not justified by consistent gains; it won only on artificially stitched multi-topic documents. In practice: use structure where the document has it, take 200 to 500 tokens as the fixed-size baseline, and test semantic chunking rather than adopting it by default.
    """),
    })
    return


@app.cell(hide_code=True)
def _(TEACHING, grid_scores, md_table, mo, pd):
    def _stats_table():
        rows = []
        for chunking_id, label in TEACHING:
            match = grid_scores[grid_scores.config_id == f'{chunking_id}__octen']
            if match.empty:
                continue
            r = match.iloc[0]
            rows.append({'strategy': label, 'chunks': f"{int(r['chunks']):,}",
                         'median chars': int(r['median_chars']), 'p95 chars': int(r['p95_chars']),
                         'over 350 chars': f"{100 * r['over_350_chars']:.0f} %"})
        return pd.DataFrame(rows)

    mo.accordion({
        'What each strategy produced on the full corpus': mo.vstack([
            mo.md('All 111 Bausteine, joined into one text of 2.45 million characters for the fixed-size strategies; the section strategy works on the records directly. Exact duplicates, the template boilerplate that repeats in every Baustein, are removed before embedding.'),
            mo.md(md_table(_stats_table().to_dict('records'))),
        ]),
    }) if grid_scores is not None else mo.md('')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Failure 1: paragraph chunking indexes bare headers

    Splitting on blank lines makes every `##` line a chunk of its own. **3 192 of the 9 292 paragraph chunks are a header and nothing else**, and they look like this:

    ```
    ## ORP.4.A5 Vergabe von Zutrittsberechtigungen (B)
    ## ORP.4.A6 Vergabe von Zugangsberechtigungen (B)
    ## ORP.4.A7 Vergabe von Zugriffsrechten (B)
    ```

    Whether that matters is easier to feel than to be told.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Exercise

    Five requirement titles from the same Baustein, and one requirement body. **Which title does the body belong to?** Decide before you open the solution.

    1. Regelung für die Einrichtung und Löschung von Benutzenden und Benutzendengruppen
    2. Vergabe von Zutrittsberechtigungen
    3. Vergabe von Zugangsberechtigungen
    4. Vergabe von Zugriffsrechten
    5. Identifikation und Authentisierung

    > Es MUSS festgelegt werden, welche Zugangsberechtigungen an welche Personen im Rahmen ihrer Funktion vergeben bzw. ihnen entzogen werden. Werden Zugangsmittel wie Chipkarten verwendet, so MUSS die Ausgabe bzw. der Entzug dokumentiert werden. Wenn Zugangsmittel kompromittiert wurden, MÜSSEN sie ausgewechselt werden. …
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        'Solution': mo.md(r"""
    **Title 3, Vergabe von Zugangsberechtigungen.** The body of title 2 begins *"Es MUSS festgelegt werden, welche Zutrittsberechtigungen an welche Personen im Rahmen ihrer Funktion vergeben bzw. ihnen entzogen werden. Die Ausgabe bzw. der Entzug von verwendeten Zutrittsmittel wie Chipkarten MUSS dokumentiert werden."* Title 4's body opens the same way again, with Zugriffsrechte.

    The three differ by one morpheme. *Zutritt* is getting into the building, *Zugang* is getting onto the system, *Zugriff* is reading the data. The Kompendium is precise about it; nothing else in the sentence is.
    """),
    })
    return


@app.cell(hide_code=True)
def _(EXTRAS, REGENERATE, mo):
    _cos = EXTRAS['header_chunk_projection']
    mo.vstack([
        mo.md('**Why that was hard.** The titles are near-interchangeable, and so are their embeddings. An embedding model has no more to go on than you did. A retriever cannot separate what the embedding does not separate, so a question about Zugang will retrieve Zutritt and Zugriff alongside it, and a bare header chunk carries nothing else to break the tie.'),
        mo.image(src='/public/img/w201_header_projection.png', width=620),
        mo.md(f"*Figure 5. Header-only chunks (filled) against three chunks that carry a body (hollow), projected to two dimensions. Mean pairwise cosine among the headers **{_cos['pairwise_cosine_headers']:.2f}**, among the three with text **{_cos['pairwise_cosine_distinct']:.2f}**.*"),
        mo.accordion({'How this figure was made': REGENERATE}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Failure 2: word chunking overruns what the model reads

    Cutting every 100 words gives chunks of about 830 characters. miniLM reads the first **350**. It does not complain; it embeds what it saw and discards the rest.

    ```
    Grundlage benutzt werden, um Sicherheitskonzepte zu erstellen. Zudem stehen Hilfsmittel und
    Musterlösungen zur Verfügung, die dabei unterstützen können, die Anforderungen geeignet zu
    erfüllen. Da der IT-Grundschutz auch international großen Anklang findet, werden das
    IT-Grundschutz-Kompendium und weitere Veröffentlichungen auch in englischer Sprach|e online
    zur Verfügung gestellt. ## Weiterentwicklung des IT-Grundschutz-Kompendiums …
    ```

    The `|` is where miniLM stops, mid-word. The remaining **524 characters** of that chunk are never embedded and can never be retrieved. Silent truncation is the more dangerous failure of the two on this page, because retrieval quality drops and nothing in the logs says why.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Failure 3: character chunking cuts mid-sentence

    A character counter respects word boundaries but nothing above them. At 300 characters a heading is torn in half:

    ```
    end of one chunk    …  - OPS.3.2 Anbieten von Outsourcing  DER: Detektion
    start of the next      und Reaktion  - DER.1 Detektion von sicherheitsrelevanten Ereignissen …
    ```

    "DER: Detektion und Reaktion" is now two chunks, neither of which is the heading. The same cut puts the end of one requirement and the start of the next into one chunk, so a hit points at two things at once.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 Comparison

    ### 5.1 Theory: what should a retrieval metric tell us?

    **Discriminability** asks whether the system can tell chunks apart at all: if every chunk scores almost the same against a question, the ranking is arbitrary.

    **Separability** asks whether the chunks that are relevant score differently from those that are not; this needs labels for at least some questions.

    **Correctness** asks whether the right chunk is at the top of the list, which needs ground truth for every question that is counted.

    All three are illustrated below on one toy example that recurs throughout this section: one query, five chunks.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt, sorted_score_plot, theme):
    TOY_SCORES = [0.82, 0.80, 0.55, 0.30, 0.28]
    TOY_RELEVANT = 1

    def gap(scores):
        """Discriminability: how far the best chunk stands above the runner-up."""
        top = sorted(scores, reverse=True)
        return top[0] - top[1] if len(top) > 1 else float('nan')

    def _toy_sorted():
        fig, ax = plt.subplots(figsize=(3.1, 2.4))
        sorted_score_plot(TOY_SCORES, highlight_index=TOY_RELEVANT, ax=ax)
        ax.annotate('relevant chunk', (2, TOY_SCORES[1]), xytext=(8, 8), textcoords='offset points', fontsize=8.5)
        ax.annotate('', xy=(1, TOY_SCORES[0]), xytext=(1, TOY_SCORES[1]),
                    arrowprops={'arrowstyle': '<->', 'color': theme.ACCENT, 'linewidth': 1.1})
        ax.annotate(f'gap {gap(TOY_SCORES):.2f}', (1, (TOY_SCORES[0] + TOY_SCORES[1]) / 2),
                    xytext=(10, -2), textcoords='offset points', fontsize=8.5, color=theme.ACCENT)
        ax.set_xlabel('rank')
        ax.set_ylabel('cosine similarity')
        ax.set_ylim(0, 1)
        return fig

    def _toy_shapes():
        shapes = {'decisive': [0.9, 0.5, 0.4, 0.35, 0.3], 'decaying': [0.8, 0.7, 0.6, 0.5, 0.4],
                  'plateau': [0.7, 0.69, 0.68, 0.4, 0.3], 'flat': [0.6, 0.6, 0.59, 0.59, 0.58]}
        fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.0), sharey=True)
        for ax, (name, sc) in zip(axes, shapes.items()):
            ranks = np.arange(1, 6)
            ax.plot(ranks, sc, color=theme.GREY, linewidth=1.2)
            ax.plot(ranks, sc, 'o', markerfacecolor='white', markeredgecolor=theme.INK, markersize=5)
            ax.set_title(f'{name}: gap = {gap(sc):.2f}', fontsize=8.5)
            ax.set_xticks(ranks)
            ax.set_ylim(0, 1)
            ax.set_xlabel('rank')
        axes[0].set_ylabel('cosine similarity')
        return fig

    mo.vstack([
        mo.md(r"""
        ## Discriminability

        **Can the system tell the candidates apart at all?**

        Sort the top-$k$ scores against rank. One number captures the shape: the **gap** between rank 1 and rank 2. A large gap is a decisive retrieval, a gap near zero is a coin toss. Nothing has to be labelled, which is what makes it the diagnostic you can actually run on your own corpus. Vector databases and the Kneedle algorithm ([Satopää et al. 2011](https://doi.org/10.1109/ICDCSW.2011.20)) read the same curve to decide how many hits to keep.
        """),
        mo.hstack([_toy_sorted(),
                   mo.md('*Figure 6. The toy ranking that runs through this section: one query, five '
                         f'chunks, and only the chunk at rank 2 is relevant. The gap is '
                         f'{gap(TOY_SCORES):.2f}.*')],
                  widths=[1, 1], align='center'),
        _toy_shapes(),
        mo.md(r"""
        *Figure 7. Four shapes of a top-5 curve. A plateau and a flat curve both mean the retriever is guessing.*

        **The catch.** A large gap means confident, not correct: had the irrelevant chunk above scored 0.95, the gap would be wide and the answer still wrong. The gap tells you whether the scores carry signal, never whether they are right.
        """),
    ])
    return TOY_RELEVANT, TOY_SCORES


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
        ## Separability

        **Do relevant chunks score differently from irrelevant ones?**

        Pool the scores of relevant and of non-relevant chunks across many questions and draw a density for each. Two separated humps mean a threshold could filter noise; heavy overlap means no cut-off will, and the fix is a reranker or different chunking. Unlike the gap, this needs labels, since otherwise the scores cannot be split into two groups.

        An old idea: [Manmatha, Rath and Feng (2001)](https://doi.org/10.1145/383952.384005) modelled relevant scores as normal and the rest as exponential, and [Arampatzis, Kamps and Robertson (2009)](https://doi.org/10.1145/1571941.1572031) fitted that mixture to pick a cut-off.
        """),
        mo.hstack([_toy_density(), mo.md('*Figure 8. Made-up score densities with a clear gap between relevant and other chunks. Real data in Figure 11.*')], widths=[1, 1], align='center'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correctness

    **Is the right chunk at the top?**

    This is the only one of the three that needs ground truth for every question counted, and it is the one people quote. There are four standard numbers and they disagree with each other, which is the part worth feeling before it is explained.

    Below is a hit list you control. Set how many chunks the retriever returned, how relevant each one is (0 not relevant, 1 partly, 2 fully) and where the cut-off $k$ falls. Watch which metric moves and which does not. **Play first. The definitions are in the collapsed box underneath, and they read very differently once you have seen the numbers behave.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ui_chunks = mo.ui.slider(2, 10, value=5, step=1, label='chunks returned')
    ui_cut = mo.ui.slider(1, 10, value=5, step=1, label='k (the cut-off)')
    mo.hstack([ui_chunks, ui_cut], justify='start')
    return ui_chunks, ui_cut


@app.cell(hide_code=True)
def _(mo, ui_chunks):
    GRADES = {'0 not relevant': 0, '1 partly relevant': 1, '2 fully relevant': 2}
    _default = ['0 not relevant', '2 fully relevant'] + ['0 not relevant'] * 8
    ui_grades = mo.ui.array([mo.ui.dropdown(GRADES, value=_default[i], label=f'rank {i + 1}')
                             for i in range(ui_chunks.value)])
    mo.vstack([mo.md('**Relevance of each rank** (rank 1 is the chunk the retriever liked best):'),
               mo.hstack(list(ui_grades), justify='start', wrap=True)])
    return (ui_grades,)


@app.cell(hide_code=True)
def _(mo, ndcg_graded, plt, theme, ui_cut, ui_grades):
    grades = [g for g in ui_grades.value]
    k = min(ui_cut.value, len(grades))
    positive = [i for i, g in enumerate(grades) if g > 0]

    mrr = 1.0 / (positive[0] + 1) if positive else 0.0
    recall = (len([i for i in positive if i < k]) / len(positive)) if positive else 0.0
    precision = len([i for i in positive if i < k]) / k
    ndcg = ndcg_graded(grades, k)

    def _hit_list_figure():
        fig, ax = plt.subplots(figsize=(6.4, 1.5))
        for i, g in enumerate(grades):
            shade = {0: 'white', 1: theme.GREYS[2], 2: theme.INK}[g]
            ax.add_patch(plt.Rectangle((i, 0), 0.86, 1, facecolor=shade,
                                       edgecolor=theme.INK, linewidth=0.8))
            ax.text(i + 0.43, -0.28, str(i + 1), ha='center', va='top', fontsize=8,
                    color=theme.INK)
        ax.axvline(k - 0.07, color=theme.ACCENT, linewidth=1.6)
        ax.text(k - 0.07, 1.15, f'k = {k}', color=theme.ACCENT, fontsize=8.5, ha='center')
        ax.set_xlim(-0.1, max(len(grades), k) + 0.1)
        ax.set_ylim(-0.6, 1.5)
        ax.set_axis_off()
        return fig

    mo.vstack([
        _hit_list_figure(),
        mo.hstack([
            mo.stat(f'{mrr:.2f}', label='MRR'),
            mo.stat(f'{ndcg:.2f}', label=f'nDCG@{k}'),
            mo.stat(f'{recall:.2f}', label=f'Recall@{k}'),
            mo.stat(f'{precision:.2f}', label=f'Precision@{k}'),
        ], justify='start'),
        mo.md('*Filled black is fully relevant, grey is partly relevant, white is not relevant. '
              'The red line is the cut-off.*'),
    ])
    return


@app.cell(hide_code=True)
def _(TOY_RELEVANT, TOY_SCORES, evaluate_ranking, mo):
    _toy = evaluate_ranking(list(range(len(TOY_SCORES))), {TOY_RELEVANT}, k=5)
    mo.accordion({
        'Now read this: what the four numbers actually measure': mo.md(rf"""
    Let $r$ be the rank of the first relevant chunk and $\mathrm{{rel}}_i$ the relevance of the chunk at rank $i$.

    | metric | definition | moves when |
    |---|---|---|
    | **MRR** | $1/r$, averaged over queries; 0 if nothing relevant appears ([Voorhees, TREC-8](https://trec.nist.gov/pubs/trec8/t8_proceedings.html)) | only the *first* relevant chunk changes rank: 1.0 at rank 1, 0.50 at rank 2, 0.33 at rank 3 |
    | **nDCG@$k$** | $\sum_{{i \le k}} \frac{{2^{{\mathrm{{rel}}_i}} - 1}}{{\log_2(i+1)}}$ divided by the same sum for the ideal order ([Järvelin and Kekäläinen 2002](https://doi.org/10.1145/582415.582418)) | any relevant chunk moves, and it is the only one that notices *grades* and a second relevant chunk |
    | **Recall@$k$** | share of all relevant chunks that are inside the top $k$ | a relevant chunk crosses the cut-off; nothing inside the top $k$ matters |
    | **Precision@$k$** | share of the top $k$ that is relevant | the same crossing, but it is capped at $1/k$ when only one chunk is relevant |

    That cap is why Precision@5 is nearly useless on this corpus: with one relevant chunk per question the best attainable value is 0.20, so it measures the question set more than the retriever.

    On the toy ranking of Figure 6, where the relevant chunk sits at rank 2:

    | metric | calculation | value |
    |---|---|---|
    | MRR | $1/2$ | {_toy['mrr']:.2f} |
    | nDCG@5 | $(1/\log_2 3) / (1/\log_2 2)$ | {_toy['ndcg']:.2f} |
    | Recall@5 | $1/1$ | {_toy['recall']:.2f} |
    | Precision@5 | $1/5$ | {_toy['precision']:.2f} |

    All four count a chunk as relevant only if a human labelled it, so an unlabelled chunk that answers the question perfectly counts as a miss. Fuhr (2017) adds that reciprocal rank is ordinal, which makes MRR a convenient summary rather than a precise quantity.

    **Which to report.** Recall@$k$ answers the question a RAG pipeline actually asks, namely whether the answer reached the context window at all. MRR and nDCG@$k$ answer whether it arrived near the top, which matters once the context window is small or a reranker follows.
    """),
    })
    return


@app.cell(hide_code=True)
def _(METRIC_ANIMATIONS, mo):
    mo.accordion({
        'Watch a metric move': mo.vstack([
            mo.md('Each loop changes one thing about the hit list and shows what the metric does.'),
            *[mo.vstack([mo.md(f'**{caption}**'), mo.image(src=path)])
              for path, caption in METRIC_ANIMATIONS],
        ]),
    }) if METRIC_ANIMATIONS else mo.md('')
    return


@app.cell(hide_code=True)
def _(grid_questions, grid_scores, mo):
    mo.stop(grid_scores is None, mo.md('### 5.2 Real data\n\n*The measured tables are missing.*'))

    TEACHING = [  # the six configurations chapter 4 walked through, in chapter order
        ('paragraph', 'paragraph'),
        ('words_100', '100 words'),
        ('chars_300', '300 characters'),
        ('chars_1200', '1 200 characters'),
        ('chars_1200_ov200', '1 200 chars, 200 overlap'),
        ('section_1200', 'section'),
    ]
    MODEL_ORDER = ('octen', 'miniLM')  # the large model solid, the small one hatched

    def score_row(chunking_id, model):
        match = grid_scores[grid_scores.config_id == f'{chunking_id}__{model}']
        return None if match.empty else match.iloc[0]

    _easy = int((grid_questions.difficulty == 'easy').sum())
    _complex = int((grid_questions.difficulty == 'complex').sum())
    mo.md(f"""
    ### 5.2 Real data

    Every configuration below was measured once, offline, over the whole corpus and all **{len(grid_questions)} evaluated questions** ({_easy} easy, {_complex} complex). A chunk counts as relevant when it overlaps a cited section. Each question is embedded once per model, compared with every chunk by cosine similarity, and the top 50 form the ranking the metrics score.

    Two of the 166 gold rows are not here: they cite BSI-Standard 200-2, which is not indexed, so no chunking could ever retrieve them. Counting them as failures would punish every strategy equally and tell us nothing.
    """)
    return MODEL_ORDER, TEACHING, score_row


@app.cell(hide_code=True)
def _(GRID_DIR, TEACHING, length_hist_panels_from_bins, mo, pd):
    _path = GRID_DIR / 'grid_lengths.parquet'
    mo.stop(not _path.exists(), mo.md('*`grid_lengths.parquet` is missing.*'))

    def _bins():
        lengths = pd.read_parquet(_path)
        out = {}
        for chunking_id, label in TEACHING:
            rows = lengths[lengths.config_id == f'{chunking_id}__octen'].sort_values('bin_left')
            if not rows.empty:
                out[label] = (list(rows.bin_left), list(rows.bin_right), list(rows['count']))
        return out

    _fig = length_hist_panels_from_bins(_bins())
    _fig.set_size_inches(6.5, 2.4)
    mo.vstack([
        mo.md('#### What each strategy produced\n\nBefore the scores, the shape of what was indexed. This is the variable everything else follows from.'),
        _fig,
        mo.md('*Figure 9. Chunk length per strategy over the full corpus (shared x axis, logarithmic counts). Fixed-size strategies produce one spike; paragraph and section chunks spread over two orders of magnitude.*'),
    ])
    return


@app.cell(hide_code=True)
def _(MODEL_ORDER, TEACHING, md_table, mo, pd, score_row):
    def _results_table():
        rows = []
        for chunking_id, label in TEACHING:
            for model in MODEL_ORDER:
                r = score_row(chunking_id, model)
                if r is None:
                    continue
                rows.append({
                    'strategy': label, 'model': model, 'chunks': f"{int(r['chunks']):,}",
                    'MRR': round(r['MRR'], 3), 'nDCG@5': round(r['nDCG@5'], 3),
                    'Recall@5': round(r['Recall@5'], 3),
                    'Recall@5 easy': round(r['Recall@5_easy'], 3),
                    'Recall@5 complex': round(r['Recall@5_complex'], 3),
                    'gap': round(r['gap'], 3),
                    '> 350 chars': f"{100 * r['over_350_chars']:.0f} %",
                })
        return pd.DataFrame(rows)

    results = _results_table()
    mo.vstack([
        mo.md('#### Results'),
        mo.md(md_table(results.to_dict('records'))),
    ])
    return


@app.cell(hide_code=True)
def _(MODEL_ORDER, Patch, TEACHING, mo, np, plt, score_row, theme):
    def _grouped_bars(metrics):
        labels = [label for _, label in TEACHING]
        fig, axes = plt.subplots(1, len(metrics), figsize=(6.6, 2.6), sharey=True)
        x = np.arange(len(labels))
        width = 0.38
        for ax, metric in zip(axes, metrics):
            for offset, model in zip((-width / 2, width / 2), MODEL_ORDER):
                values = [(score_row(c, model)[metric] if score_row(c, model) is not None else np.nan)
                          for c, _ in TEACHING]
                ax.bar(x + offset, values, width, color=theme.GREYS[1], edgecolor=theme.INK,
                       linewidth=0.7, hatch='' if model == 'octen' else '///')
            ax.set_title(metric, fontsize=9)
            ax.set_xticks(x, labels, rotation=45, ha='right', fontsize=7.5)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel('score')
        axes[-1].legend(handles=[Patch(facecolor=theme.GREYS[1], edgecolor=theme.INK, label='octen'),
                                 Patch(facecolor=theme.GREYS[1], edgecolor=theme.INK, hatch='///',
                                       label='miniLM')],
                        fontsize=8, frameon=False, loc='upper right')
        return fig

    mo.vstack([
        _grouped_bars(['MRR', 'nDCG@5', 'Recall@5']),
        mo.md('*Figure 10. Retrieval correctness per strategy and model.*'),
    ])
    return


@app.cell(hide_code=True)
def _(EXTRAS, REGENERATE, mo):
    mo.vstack([
        mo.image(src='/public/img/w201_score_density.png', width=560),
        mo.md(f"*Figure 11. Separability on real data: the score densities of relevant and of other chunks, pooled over all questions, for `{EXTRAS['score_distribution']['config_id']}`. Where the curves overlap, no score threshold can filter noise.*"),
        mo.accordion({'How this figure was made': REGENERATE}),
    ])
    return


@app.cell(hide_code=True)
def _(MODEL_ORDER, TEACHING, grid_hits, mo, np, plt, theme):
    def _gap_figure():
        fig, ax = plt.subplots(figsize=(5.4, 2.6))
        for model, style in zip(MODEL_ORDER, ('-', '--')):
            best = f'{TEACHING[-1][0]}__{model}'
            rows = grid_hits[grid_hits.config_id == best]
            if rows.empty:
                continue
            top2 = rows[rows['rank'] <= 2].pivot_table(index='qid', columns='rank', values='score')
            gaps = (top2[1] - top2[2]).dropna()
            counts, edges = np.histogram(gaps, bins=24, range=(0, max(0.2, float(gaps.max()))))
            ax.step(edges[:-1], counts, where='post', color=theme.INK, linestyle=style,
                    linewidth=1.2, label=f'{model} (median {gaps.median():.3f})')
        ax.set_xlabel('gap between rank 1 and rank 2')
        ax.set_ylabel('questions')
        ax.legend(frameon=False, fontsize=8)
        return fig

    _view = mo.md('') if grid_hits is None else mo.vstack([
        _gap_figure(),
        mo.md('*Figure 12. Discriminability without labels: how far the best chunk stands above the '
              'runner-up, per question, on section chunks. A distribution pressed against zero means '
              'the ranking is close to arbitrary. Compare with Figure 9: decisive is not correct.*'),
    ])
    _view
    return


@app.cell(hide_code=True)
def _(TEACHING, md_table, mo, score_row):
    def _pairs():
        rows = []
        for chunking_id, label in TEACHING:
            octen, mini = score_row(chunking_id, 'octen'), score_row(chunking_id, 'miniLM')
            if octen is None or mini is None:
                continue
            rows.append({'strategy': label,
                         'chunks over 350 chars': f"{100 * octen['over_350_chars']:.0f} %",
                         'octen': round(octen['Recall@5'], 3), 'miniLM': round(mini['Recall@5'], 3),
                         'what the small model costs': round(octen['Recall@5'] - mini['Recall@5'], 3)})
        return sorted(rows, key=lambda r: float(r['chunks over 350 chars'].rstrip(' %')))

    _rows = _pairs()
    mo.stop(len(_rows) < 2, mo.md('*The miniLM comparison needs the second stage of `tools.run_grid`.*'))
    _cheapest, _dearest = _rows[0], _rows[-1]
    mo.vstack([
        mo.md('#### What the small model costs, and where'),
        mo.md(md_table(_rows)),
        mo.md(f"""
    miniLM is worse everywhere, but not evenly. Sorted by how much of each strategy overruns its 350-character window, the price rises with it: **{_cheapest['what the small model costs']:.2f} Recall@5** on `{_cheapest['strategy']}`, where nothing is truncated, against **{_dearest['what the small model costs']:.2f}** on `{_dearest['strategy']}`, where nearly everything is. That is failure 2 turned into a number.

    The tempting conclusion is wrong, though. miniLM's own best result is still a long-chunk strategy, because chunk length helps recall more than truncation hurts it. So the lesson is not "use short chunks with a small model" but: **the penalty you pay for the small model grows with chunk length, and you should know how large it is before you accept it.**
    """),
    ])
    return


@app.cell(hide_code=True)
def _(TEACHING, grid_scores, mo, score_row):
    _measured = [label for c, label in TEACHING if score_row(c, 'octen') is not None]
    mo.stop(len(_measured) < len(TEACHING), mo.callout(mo.md(
        f'### 5.3 What to do\n\nOnly {len(_measured)} of the {len(TEACHING)} teaching configurations have been measured. Finish `uv run python -m tools.run_grid --stage all` and this section fills itself in.'), kind='warn'))

    def _by_id(config_id):
        match = grid_scores[grid_scores.config_id == config_id]
        return None if match.empty else match.iloc[0]

    def _recommendation():
        octen = {label: score_row(c, 'octen') for c, label in TEACHING}
        octen = {k: v for k, v in octen.items() if v is not None}
        mini = {label: score_row(c, 'miniLM') for c, label in TEACHING}
        mini = {k: v for k, v in mini.items() if v is not None}
        best = grid_scores.sort_values('Recall@5', ascending=False).iloc[0]
        smallest = min(octen.values(), key=lambda r: r['Recall@5'])
        largest = max(octen.values(), key=lambda r: r['Recall@5'])
        best_mrr = max(octen.values(), key=lambda r: r['MRR'])
        section = _by_id('section_1200__octen')
        words = _by_id('words_400__octen')

        mini_line = (
            f"With octen the teaching strategies span Recall@5 {min(r['Recall@5'] for r in octen.values()):.2f} to {max(r['Recall@5'] for r in octen.values()):.2f}; with miniLM the same six span {min(r['Recall@5'] for r in mini.values()):.2f} to {max(r['Recall@5'] for r in mini.values()):.2f}."
        ) if mini else 'The miniLM comparison needs the second stage of the grid.'

        structure_line = (
            f"`section_1200` is the best strategy in the room for **easy** questions ({section['Recall@5_easy']:.2f}) and among the worst for **complex** ones ({section['Recall@5_complex']:.2f}), while `words_400` is far more even ({words['Recall@5_easy']:.2f} against {words['Recall@5_complex']:.2f}). The reason is in the question set, not the chunker: a complex question cites a passage that spans several sections, and a chunk that stops at a section boundary can only ever cover part of it. Structure-aware chunking matches the document; it does not match the question."
        ) if section is not None and words is not None else ''

        return f"""
        ### 5.3 What the measurement says

        **The line to beat: `{best.config_id}`**, Recall@5 **{best['Recall@5']:.3f}** (MRR {best['MRR']:.3f}, nDCG@5 {best['nDCG@5']:.3f}, {int(best['chunks']):,} chunks, easy {best['Recall@5_easy']:.3f}, complex {best['Recall@5_complex']:.3f}), best of {len(grid_scores)} configurations measured.

        - **Bigger chunks won**, Recall@5 {smallest['Recall@5']:.2f} to {largest['Recall@5']:.2f} across the range. A larger chunk is a wider net; the cost is that a hit points at more text than the answer needs, and generation pays for every character of it.
        - **The metrics disagree.** MRR peaks at {best_mrr['MRR']:.2f} on `{best_mrr.chunking_id}`, one of the *weakest* by Recall@5 ({best_mrr['Recall@5']:.2f}): cutting a requirement into fragments makes several of them count as relevant, so one lands at rank 1 while most of the answer never reaches the top five. Optimising MRR in the playground would have pointed you at nearly the worst option.
        - **Structure helps the questions it was shaped for.** {structure_line}
        - **The model decides more than the chunker.** {mini_line} Figure 11 says it from the score side: overlapping densities, no threshold that helps.
        - **Overlap bought nothing** and cost chunks, matching Chroma and Jina. Prepending the section title bought a little, for free.
        """

    mo.md(_recommendation())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Doing this on your own corpus

    The procedure, not the answer. The answer above is true of the Kompendium and of nothing else.

    1. **Write twenty questions and mark the passage that answers each.** Without them, everything else is opinion.
    2. **Plot the chunk-length distribution before embedding anything.** Free, and it shows at once whether the strategy fits the model.
    3. **Check the model's input limit against that distribution.** Whatever is over it is being truncated silently.
    4. **Sweep chunk size first.** Largest effect, cheapest to change.
    5. **Report Recall@k next to a rank-sensitive metric.** Disagreement between them is a finding, not noise to average away.
    6. **Split by question type.** A single mean hid that the best strategy overall is second-worst on complex questions.
    """)
    return


@app.cell(hide_code=True)
def _(EXTRAS, Image, REGENERATE, WORKSHOP_DIR, mo, plt, theme):
    def _matryoshka_analogy():
        photo = Image.open(WORKSHOP_DIR / 'raw_data' / 'birds' / 'images' / 'rotkehlchen_2.jpg').convert('L')
        sizes = [96, 24, 12, 6]
        fig, axes = plt.subplots(1, len(sizes) + 1, figsize=(6.5, 1.9),
                                 gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.6]})
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
        ax.set_yticks([])
        ax.set_xticks([])
        for side in ('left', 'bottom', 'top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_title('kept dimensions', fontsize=8.5)
        return fig

    _matry = EXTRAS['matryoshka']
    _curve = mo.image(src='/public/img/w201_matryoshka.png', width=480)
    _caption = (
        f"*Figure 14. MRR (solid) and Recall@5 (dashed) for `{_matry['config_id']}` when only the first "
        f"d dimensions are kept and the vectors are re-normalised. MRR at {_matry['dims'][0]}: "
        f"{_matry['mrr'][0]:.2f}, at {_matry['dims'][-1]}: {_matry['mrr'][-1]:.2f}.*"
    )

    mo.accordion({
        '6 Bonus: Matryoshka embeddings, or how much of a vector do we need?': mo.vstack([
            mo.md(r"""
            A 4 096-dimensional vector per chunk is a lot of storage and a lot of arithmetic per query. **Matryoshka representation learning** ([Kusupati et al. 2022](https://arxiv.org/abs/2205.13147)) trains a model so that the first $d$ dimensions of a vector are themselves a usable embedding. Truncating then costs a little quality instead of destroying the vector, the way truncating an ordinary embedding would.

            The analogy: a photo shrunk to fewer and fewer pixels stays recognisable for a surprisingly long time, and then stops.
            """),
            _matryoshka_analogy(),
            mo.md(r"""
            *Figure 13. A robin at 96, 24, 12 and 6 pixels, and the nested prefixes of a 4 096-dimensional vector.*

            Expect the curve to stay flat at the top and fall away at the low end: [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) reports an MTEB average of 62.28 at 768 dimensions, 61.04 at 256 and 56.10 at 64. A gentle slope means storage and search cost can be traded for a small, known loss. A steep drop from the first cut would mean the model was never trained this way.
            """),
            _curve,
            mo.md(_caption),
            mo.accordion({'How Figure 14 was made': REGENERATE}),
        ]),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 Look at a single question

    Averages hide the interesting cases. Pick a question and a configuration and read the five chunks that came back, exactly as the measurement saw them.
    """)
    return


@app.cell(hide_code=True)
def _(grid_questions, grid_scores, mo):
    mo.stop(grid_scores is None or grid_questions is None, mo.md('*The measured tables are missing.*'))
    ui_config = mo.ui.dropdown(
        {c: c for c in sorted(grid_scores.config_id)},
        value=grid_scores.sort_values('Recall@5', ascending=False).config_id.iloc[0],
        label='configuration')
    ui_pick = mo.ui.dropdown(
        {f'{q.difficulty}: {q.question[:88]}': q.qid for q in grid_questions.itertuples()},
        value=f'{grid_questions.iloc[0].difficulty}: {grid_questions.iloc[0].question[:88]}',
        label='question', full_width=True)
    mo.vstack([ui_config, ui_pick])
    return ui_config, ui_pick


@app.cell(hide_code=True)
def _(grid_hits, grid_questions, mo, plt, theme, ui_config, ui_pick):
    mo.stop(grid_hits is None or ui_pick.value is None, mo.md(''))
    _rows = grid_hits[(grid_hits.config_id == ui_config.value)
                      & (grid_hits.qid == ui_pick.value)].sort_values('rank')
    mo.stop(_rows.empty, mo.md('*Nothing measured for this combination.*'))

    def _score_curve(rows):
        fig, ax = plt.subplots(figsize=(4.0, 2.2))
        for row in rows.itertuples():
            ax.bar(row.rank, row.score, width=0.7,
                   color=theme.INK if row.relevant else 'white',
                   edgecolor=theme.INK, linewidth=0.8)
        ax.set_xlabel('rank')
        ax.set_ylabel('cosine similarity')
        ax.set_xticks(list(rows['rank']))
        return fig

    _question = grid_questions[grid_questions.qid == ui_pick.value].iloc[0]
    _items = []
    for _hit in _rows.head(5).itertuples():
        _mark = '**relevant**' if _hit.relevant else 'not relevant'
        _items.append(f'**{_hit.rank}.** ({_hit.score:.3f}, {_mark})\n\n```\n'
                      f'{_hit.text.strip()[:420]}\n```')
    mo.vstack([
        mo.md(f'**{_question.question}**\n\n*Cited passage: {_question.fundstelle[:160]}…*'),
        _score_curve(_rows),
        mo.md('*The top ten scores for this question. Filled bars are chunks that overlap the cited '
              'section.*'),
        mo.md('\n\n'.join(_items)),
    ])
    return


if __name__ == "__main__":
    app.run()
