# ruff: noqa: PLR1711  marimo cells end with an explicit return

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Wie gut ist mein RAG-System?",
    css_file="ragkit/theme.css",
    html_head_file="ragkit/head.html",
)


@app.cell(hide_code=True)
def _(mo, t, theme):
    # Row 1: header; SECTIONS_MARKER makes every H2 section collapsible (script in ragkit/head.html)
    mo.Html(theme.header_html(
        t('Wie gut ist mein RAG-System?', 'How good is my RAG system?'),
        t('RAG III, Notebook 1. Was RAGAS ist, was es braucht und wie Sie Ihr eigenes RAG-System bewerten können.',
          'RAG III, notebook 1. What RAGAS is, what it needs, and how you can evaluate your own RAG system.'),
    ) + theme.SECTIONS_MARKER)
    return


@app.cell(hide_code=True)
def _():
    import re  # finds navigation-only lines

    import marimo as mo  # the reactive notebook itself
    import pandas as pd

    from ragkit import (  # switches, HPI look, gold data set of notebooks 2 and 3 (read only)
        i18n,
        theme,
        w3_eval,
    )
    from ragkit import intro_rag as rag  # the mini pipeline and its two metrics
    from ragkit.config import setup
    from ragkit.viz import md_table

    return i18n, md_table, mo, pd, rag, re, setup, theme, w3_eval


@app.cell(hide_code=True)
def _(i18n, mo):
    # Row 2: language and scope, preset from `-- --lang en --short` or `?lang=en&short=1`
    ui_lang, ui_scope = i18n.switches(mo)
    mo.hstack([ui_lang, ui_scope], justify='start', gap=3)
    return ui_lang, ui_scope


@app.cell(hide_code=True)
def _(mo, t, theme):
    # Legend of the three box types (meta plan "Farben der Kästen"), above section A and outside every section
    mo.Html(theme.box_legend(t))
    return


@app.cell(hide_code=True)
def _(i18n, mo, theme, ui_lang, ui_scope):
    t = i18n.T(ui_lang.value, ui_scope.value)

    def feedback(ok, explanation=''):
        """The shared feedback box (meta plan 10.4a): True right, False wrong, None no right answer; Markdown text."""
        return mo.Html(theme.feedback_box(ok, mo.md(explanation).text if explanation else '', t))

    def quiz_feedback(value, correct, explanation):
        """Immediate feedback under a quiz radio; nothing before a choice."""
        if value is None:
            return mo.md('')
        return feedback(value == correct, explanation)
    return feedback, quiz_feedback, t


@app.cell(hide_code=True)
def _(hint, key, mo, t):
    # Section A (slide): the guiding question
    mo.vstack([
        mo.md(t('## A. Die Frage des Tages', '## A. The question of the day')),
        key(t('Wie kann ich mein RAG-System evaluieren?', 'How can I evaluate my RAG system?')),
        hint(t('Darum geht es heute in den drei Notebooks des Workshops. Dieses Notebook gibt den Überblick.',
               'That is what the three notebooks of the workshop are about today. This notebook gives the overview.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, rag, setup, t):
    # Row 4: environment status, visible only when something is missing
    env = setup(require_qdrant=False, strict=False)
    cache_ok = all(p.exists() for p in (rag.CHUNKS_FILE, rag.VECTORS_FILE, rag.RESULTS_FILE))
    _parts = []
    if not env.ok:
        _parts.append(mo.md(env.summary_md()))
    if not cache_ok:
        _parts.append(mo.callout(mo.md(t(
            'Der Cache mit den Ergebnissen fehlt. Bauen Sie ihn einmal mit `uv run python -m tools.precompute_w3_01` im Ordner `notebooks`.',
            'The results cache is missing. Build it once with `uv run python -m tools.precompute_w3_01` in the `notebooks` folder.',
        )), kind='danger'))
    mo.vstack(_parts) if _parts else None
    return (cache_ok,)


@app.cell(hide_code=True)
def _(cache_ok, mo, rag, w3_eval):
    mo.stop(not cache_ok)
    chunks, vectors = rag.load_corpus()
    # The gold data set of notebooks 2 and 3 (section D): 43 rows, Fundstelle cleaned the way w3_eval does it
    gold = w3_eval.load_gold() if w3_eval.GOLD_CSV.exists() else []
    GTR = w3_eval.load_translations().get('en', {})
    results = rag.load_results()
    manual, synthetic = results['manual'], results['synthetic']
    TR = rag.load_translations()  # German corpus text -> English, shown only in the English version
    JUDGE_DOWN = ('Der Richter ist gerade nicht erreichbar. Probieren Sie es in einer Minute noch einmal.',
                  'The judge cannot be reached right now. Try again in a minute.')
    LIMIT = 3  # own live evaluations per person, rows 10 and 13b together
    return GTR, JUDGE_DOWN, LIMIT, TR, chunks, gold, manual, synthetic, vectors


@app.cell(hide_code=True)
def _(LIMIT, TR, chunks, mo, pd, re, t, theme):
    _grey = {'color': theme.GREY, 'font-size': '0.85rem'}

    def en_of(text):
        """English translation of a cached German text; only in the English version, only when it differs."""
        if t.lang != 'en':
            return None
        _en = TR.get(text)
        return _en if _en and _en.strip() != text.strip() else None

    def bi(text):
        """Corpus text as it is, with its English translation below in small grey type.

        The corpus text carries the class hpi-corpus, so the language check can tell it from our own text.
        """
        _orig = mo.Html(f'<div class="hpi-corpus">{mo.md(text).text}</div>')
        _en = en_of(text)
        return _orig if _en is None else mo.vstack([_orig, theme.hint(_en)], gap=0.25)

    def bi_cell(text):
        """`bi` for one Markdown table cell."""
        def _flat(s):
            return ' '.join(str(s).split()).replace('|', '/')
        _en = en_of(text)
        _orig = f'<span class="hpi-corpus">{_flat(text)}</span>'
        return _orig if _en is None else (
            f'{_orig}<br><span style="color:{theme.GREY};font-size:.85em">{_flat(_en)}</span>')

    def chunk_list(row):
        """The three retrieved chunks of a result row, labeled as the answer cites them."""
        return mo.vstack([
            mo.vstack([mo.md(t(f'*Chunk {i + 1} (in der Antwort: Quelle {i + 1}), Zeile {n}*',
                               f'*Chunk {i + 1} (in the answer: Quelle {i + 1}), line {n}*')), bi(c)])
            for i, (n, c) in enumerate(zip(row['chunk_ids'], row['contexts']))
        ], gap=1)

    _nav = re.compile(r'^(?:[-*]|\d+\.)?\s*\[[^\]]*\]\([^)]*\)\s*$')  # a line that is only a link

    def web_table():
        """Searchable table of the website lines, one row per chunk, without image and navigation lines."""
        _ids = [i for i, c in enumerate(chunks) if c != '<!-- image -->' and not _nav.match(c)]
        _cols = {t('Zeile', 'Line'): _ids, 'Text': [chunks[i] for i in _ids]}
        if t.lang == 'en':
            _cols['English'] = [en_of(chunks[i]) or '' for i in _ids]
        return mo.ui.table(pd.DataFrame(_cols), selection=None, page_size=10, show_download=False,
                           show_column_summaries=False, show_data_types=False,
                           wrapped_columns=[c for c in ('Text', 'English') if c in _cols],
                           label=t('Die Webseite, Zeile für Zeile', 'The website, line by line'))

    def eval_button(used):
        """The Evaluate button of rows 10 and 13b; spent after LIMIT evaluations."""
        if used >= LIMIT:
            return mo.ui.run_button(label=t('Alle drei Bewertungen genutzt', 'All three evaluations used'), disabled=True)
        return mo.ui.run_button(label=t('Bewerten', 'Evaluate'))

    # The four metrics in the order of the diagram in section C: retrieval group, answer group.
    GROUPS = [(('Suche', 'Retrieval'), ['context_precision', 'context_recall']),
              (('Antwort', 'Answer'), ['answer_correctness', 'faithfulness'])]
    NAMES = {'context_precision': 'Context Precision', 'context_recall': 'Context Recall',
             'answer_correctness': 'Answer Correctness', 'faithfulness': 'Faithfulness'}
    return GROUPS, NAMES, bi, bi_cell, chunk_list, en_of, eval_button, web_table


@app.cell(hide_code=True)
def _(frame, hint, key, left, md_table, mo, t):
    # Section B (slide, 3e): the intention of RAGAS against Recall@k from part II
    _rk, _rg = t('Recall@k (Teil II)', 'Recall@k (part II)'), 'RAGAS'
    _cmp = md_table([
        {'': t('bewertet', 'scores'), _rk: t('nur die Suche', 'retrieval only'), _rg: t('Suche und Antwort', 'retrieval and answer')},
        {'': t('braucht', 'needs'), _rk: t('Zuordnung Chunk zu Fundstelle, neu für jedes Chunking', 'chunk-to-passage mapping, new for every chunking'),
         _rg: t('Frage und Referenzantwort, für Faithfulness nicht einmal das', 'question and reference answer, for Faithfulness not even that')},
        {'': t('Konfigurationen vergleichen', 'compare configurations'), _rk: t('Labels neu zuordnen', 're-assign labels'),
         _rg: t('ohne neue Labels', 'without new labels')},
        {'': t('zählt als Treffer', 'counts as a hit'), _rk: t('nur der markierte Chunk', 'only the marked chunk'),
         _rg: t('jeder Chunk, der inhaltlich hilft', 'every chunk that helps by content')},
        {'': t('Preis', 'cost'), _rk: t('billig und stabil', 'cheap and stable'),
         _rg: t('LLM-Aufrufe, Streuung, Fehlurteile', 'LLM calls, variance, wrong verdicts')},
    ])
    mo.vstack([
        mo.md(t('## B. Was ist RAGAS, und warum nutzen wir es?', '## B. What is RAGAS, and why do we use it?')),
        key(t('[RAGAS](https://docs.ragas.io) bewertet das ganze RAG-System, auch die Antwort, und macht Einstellungen vergleichbar, ohne neu zu annotieren.',
              '[RAGAS](https://docs.ragas.io) scores the whole RAG system, including the answer, and makes settings comparable without annotating again.')),
        hint(t('RAGAS ist eine Open-Source-Bibliothek in Python. Ein Richter-LLM urteilt über Inhalte; hier nutzen wir die offenen Modelle der AISC-API.',
               'RAGAS is an open-source Python library. A judge LLM judges content; here we use the open models of the AISC API.')),
        t.full(mo.md(left(_cmp)), mo.accordion({t('Recall@k oder RAGAS?', 'Recall@k or RAGAS?'): mo.md(left(_cmp))})),
        frame(t('Fundament', 'Foundation'), t(
            'Ein Gold-Datensatz bleibt das Fundament: Fragen mit Referenzantworten von Fachleuten, gern mit Fundstellen. Je besser er ist, desto aussagekräftiger die Zahlen.',
            'A gold data set remains the foundation: questions with reference answers by experts, ideally with gold passages. The better it is, the more meaningful the numbers.',
        )),
        mo.md(t('RAGAS bietet [viele weitere Kennzahlen](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/), zum Beispiel für die Relevanz der Antwort oder die Robustheit gegen störende Chunks. Wir nutzen heute vier davon.',
                'RAGAS offers [many more metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/), for example for the relevance of the answer or the robustness against distracting chunks. Today we use four of them.')),
        mo.md(t('Wer Fundstellen hat, nutzt beides: Recall@k für die Suche, RAGAS für die Antwort und für Vergleiche.',
                'If you have gold passages, use both: Recall@k for retrieval, RAGAS for the answer and for comparisons.')),
        *t.full([mo.accordion({t('Und andere Werkzeuge?', 'And other tools?'): mo.md(t(
            'Es gibt weitere Bibliotheken mit ähnlichem Ansatz, zum Beispiel DeepEval oder TruLens, und die klassische Bewertung durch Fachleute. Die Ideen dieses Workshops (Testdaten, Richter, getrennte Kennzahlen, Streuung) gelten für alle.',
            'There are other libraries with a similar approach, for example DeepEval or TruLens, and the classic evaluation by experts. The ideas of this workshop (test data, judge, separate metrics, variance) apply to all of them.',
        ))})], []),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(GROUPS, NAMES, t, theme):
    def diagram_svg():
        """Figure 1: what RAGAS needs and what comes out, greyscale, labels in the active language.

        The figure sits on its own white panel, so lines and text keep their contrast in the
        dark theme as well.
        """
        ink, grey, mid, light = theme.INK, theme.GREY, theme.GREYS[2], theme.GREYS[4]
        font = 'font-family="Inter,Verdana,Arial,sans-serif"'

        def text(x, y, s, size=12, color=ink, weight=400, anchor='middle'):
            return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" fill="{color}" '
                    f'font-size="{size}" font-weight="{weight}" {font}>{s}</text>')

        def box(x, y, w, h, lines, fill='#ffffff', stroke=ink, sw=1.2, color=ink, weight=400, size=12):
            _top = y + h / 2 - (len(lines) - 1) * 7.5
            return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
                    + ''.join(text(x + w / 2, _top + i * 15, ln, size, color, weight) for i, ln in enumerate(lines)))

        def arrow(d):
            return f'<path d="{d}" fill="none" stroke="{ink}" stroke-width="1.3" marker-end="url(#hpi-arrow)"/>'

        def label(x, y, s):  # arrow label in small grey type
            return text(x, y, s, size=11, color=grey)

        _src1 = t(['eigene Gold-Daten', '(von Fachleuten geschrieben)'], ['your own gold data', '(written by experts)'])
        _src2 = [t('von RAGAS erzeugt', 'generated by RAGAS')]
        _test = t(['Testdaten:', 'Fragen und', 'Referenzantworten'], ['Test data:', 'questions and', 'reference answers'])
        _judge = t(['RAGAS:', 'Richter-LLM und', 'Embeddings'], ['RAGAS:', 'judge LLM and', 'embeddings'])
        _parts = [
            '<rect x="0" y="0" width="960" height="300" rx="6" fill="#ffffff"/>',
            box(14, 14, 210, 44, _src1, stroke=mid), box(14, 70, 210, 34, _src2, stroke=mid),
            arrow('M119 104 L119 136'),
            box(14, 140, 210, 60, _test, sw=1.6, weight=500),
            # Testdaten -> pipeline: the question
            arrow('M224 170 L296 170'), label(260, 160, t('Frage', 'question')),
            f'<rect x="300" y="118" width="180" height="104" rx="3" fill="{light}" stroke="{grey}" stroke-width="1.2"/>',
            text(390, 134, t('Ihre RAG-Pipeline', 'Your RAG pipeline'), weight=500),
            box(315, 148, 150, 28, [t('gefundene Chunks', 'retrieved chunks')]),
            box(315, 184, 150, 28, [t('Antwort', 'answer')]),
            # Pipeline -> judge: chunks and answer
            arrow('M465 162 L586 162'), label(526, 152, t('Chunks', 'chunks')),
            arrow('M465 198 L586 198'), label(526, 188, t('Antwort', 'answer')),
            box(590, 112, 170, 112, _judge + [''], fill=ink, stroke=ink, color='#ffffff', weight=500),
            text(675, 196, t('für die Ähnlichkeit bei', 'for the similarity in'), size=10, color='#ffffff'),
            text(675, 209, 'Answer Correctness', size=10, color='#ffffff'),
            # Testdaten -> judge directly: question and reference answer
            arrow('M119 200 L119 272 L675 272 L675 228'),
            label(400, 262, t('Frage und Referenzantwort', 'question and reference answer')),
        ]
        # Metrics in two groups, the same order as the four tiles in section E.
        for _g, ((_de, _en), _keys) in enumerate(GROUPS):
            _y = 54 + _g * 112
            _parts.append(f'<rect x="790" y="{_y}" width="160" height="96" rx="3" fill="none" stroke="{light}" stroke-width="1.2"/>')
            _parts.append(text(800, _y + 14, t(_de, _en), size=11, color=grey, weight=500, anchor='start'))
            for _k, _key in enumerate(_keys):
                _parts.append(box(800, _y + 26 + _k * 34, 140, 28, [NAMES[_key]]))
            _parts.append(arrow(f'M760 168 L775 168 L775 {_y + 48} L786 {_y + 48}'))
        return (
            '<svg viewBox="0 0 960 300" width="100%" style="max-width:960px" '
            'xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="{t("Wie RAGAS arbeitet", "How RAGAS works")}">'
            '<defs><marker id="hpi-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0 0 L10 5 L0 10 z" fill="{ink}"/></marker></defs>'
            + ''.join(_parts) + '</svg>'
        )
    return (diagram_svg,)


@app.cell(hide_code=True)
def _(card, cards, diagram_svg, hint, key, left, md_table, mo, t):
    # Section C (slide): the diagram, the four metrics as cards, the needs table in an accordion
    _q, _ref, _ch, _ans = t('Frage', 'question'), t('Referenzantwort', 'reference answer'), t('Chunks', 'chunks'), t('Antwort', 'answer')
    _rows = [
        ('Context Precision', t('Stehen die nützlichen Chunks oben?', 'Are the useful chunks at the top?'), f'{_q}, {_ref}, {_ch}', '2'),
        ('Context Recall', t('Wurde alles gefunden, was die Referenzantwort braucht?', 'Was everything found that the reference answer needs?'), f'{_q}, {_ref}, {_ch}', '2'),
        ('Answer Correctness', t('Stimmt die Antwort mit der Referenzantwort überein?', 'Does the answer agree with the reference answer?'), f'{_q}, {_ref}, {_ans}', '3'),
        ('Faithfulness', t('Steht, was die Antwort sagt, in den Chunks?', 'Is what the answer says in the chunks?'),
         f"{_q}, {_ch}, {_ans} (**{t('keine Referenzantwort', 'no reference answer')}**)", '3'),
    ]
    _table = md_table([{t('Kennzahl', 'Metric'): n, t('braucht', 'needs'): need} for n, _k, need, _nb in _rows])
    mo.vstack([
        mo.md(t('## C. Wie RAGAS arbeitet', '## C. How RAGAS works')),
        key(t('Testdaten und die Ausgaben Ihrer Pipeline gehen an einen Richter; heraus kommen vier Kennzahlen.',
              'Test data and the outputs of your pipeline go to a judge; four metrics come out.')),
        mo.Html(diagram_svg()),
        hint(t('Abbildung 1. Was RAGAS braucht (links), was Ihre Pipeline liefert (Mitte) und was herauskommt (rechts).',
               'Figure 1. What RAGAS needs (left), what your pipeline delivers (middle) and what comes out (right).')),
        cards([card(n, k, t(f'Notebook {nb}', f'notebook {nb}')) for n, k, _need, nb in _rows]),
        hint(t('Die vier Kennzahlen lernen Sie in Notebook 2 (Suche) und Notebook 3 (Antwort) genau kennen.',
               'You get to know the four metrics in detail in notebook 2 (retrieval) and notebook 3 (answer).')),
        mo.accordion({t('Welche Kennzahl braucht was?', 'Which metric needs what?'): mo.vstack([
            mo.md(left(_table)),
            mo.md(t('Faithfulness braucht als einzige Kennzahl keine Referenzantwort.',
                    'Faithfulness is the only metric that needs no reference answer.')),
        ])}),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(t, theme):
    # Slide building blocks (3e): key sentence on the note ground, hairline cards, grey hint lines,
    # information and task boxes; shared by all w3 notebooks in ragkit/theme.py.
    key, card, cards, hint, frame, info, left = (theme.key_sentence, theme.card, theme.cards, theme.hint,
                                                 theme.frame, theme.info, theme.left)

    _EN = (' the ', ' and ', ' of ', ' to ', ' for ', ' how ', ' what ', ' can ', ' is ')
    _POOR_GERMAN = ('Wenr und wiie',)  # RAGAS question in German with many errors

    def ragas_note(text, kind='question'):
        """One grey line under RAGAS output that names its weakness; None when there is nothing to say."""
        _english = sum(w in f' {text.lower()} ' for w in _EN) >= 3
        if kind == 'reference':
            return hint(t('Auch die Referenzantwort hat RAGAS auf Englisch erzeugt.',
                          'RAGAS generated the reference answer in English as well.')) if _english else None
        if _english:
            return hint(t('RAGAS hat diese Frage auf Englisch erzeugt, obwohl die Webseite deutsch ist: eine Schwäche der automatischen Testdaten.',
                          'RAGAS generated this question in English although the website is German: a weakness of automatic test data.'))
        if text.startswith(_POOR_GERMAN):
            return hint(t('RAGAS hat diese Frage in fehlerhaftem Deutsch erzeugt: eine Schwäche der automatischen Testdaten.',
                          'RAGAS generated this question in poor German: a weakness of automatic test data.'))
        return None

    # Every place where participants act gets the same task box (theme.task_box); one list numbers them,
    # so the short version (no writing, no comparing) counts 1 of 2, 2 of 2.
    TASK_ORDER = ['judge', 'quiz'] + t.full(['write', 'assess'], [])

    def task(name, title, steps, time):
        """The task box of a numbered task, or of the extra task (name='extra')."""
        return theme.numbered_task(t, TASK_ORDER, name, title, steps, time)
    return card, cards, frame, hint, info, key, left, ragas_note, task


@app.cell(hide_code=True)
def _(mo, theme):
    def chain_html(steps, judge_index):
        """The steps of the pipeline as a chain of boxes, styled like Figure 1 (white panel, greys)."""
        _box = ('display:inline-block;padding:.35rem .7rem;border:1.2px solid {b};border-radius:3px;'
                'background:{bg};color:{c};font-size:.85rem;white-space:nowrap')
        _parts = []
        for _i, _s in enumerate(steps):
            _judge = _i == judge_index
            _parts.append(f'<span style="{_box.format(b=theme.INK, bg=theme.INK if _judge else "#ffffff", c="#ffffff" if _judge else theme.INK)}">{_s}</span>')
        _arrow = f'<span style="color:{theme.INK};padding:0 .35rem">&#8594;</span>'
        return mo.Html(f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:.25rem 0;background:#ffffff;'
                       f'padding:.6rem .8rem;border-radius:6px;border:1px solid {theme.RULE}">{_arrow.join(_parts)}</div>')
    return (chain_html,)


@app.cell(hide_code=True)
def _(web_table):
    # One table of the website, shown in section D and reused in the extra task (parts a and b)
    site_table = web_table()
    return (site_table,)


@app.cell(hide_code=True)
def _(GTR, gold, mo, pd, t):
    # The searchable gold table (normal only); reference and Fundstelle shortened so rows stay low,
    # selecting a row shows it in full below section D's accordions
    def _short(s, n=160):
        s = ' '.join(str(s).split())
        return s if len(s) <= n else s[:n].rsplit(' ', 1)[0] + ' ...'

    _cols = {'#': [r['index'] + 1 for r in gold],
             t('Frage', 'Question'): [r['question'] for r in gold],
             t('Referenzantwort (gekürzt)', 'Reference answer (shortened)'): [_short(r['reference']) for r in gold],
             t('Fundstelle (gekürzt)', 'Gold passage (shortened)'): [_short(r['fundstelle']) for r in gold]}
    if t.lang == 'en':
        _cols['English question'] = [GTR.get(r['question'], '') for r in gold]
    gold_table = mo.ui.table(pd.DataFrame(_cols), selection='single', page_size=8, show_download=False,
                             show_column_summaries=False, show_data_types=False,
                             wrapped_columns=[c for c in _cols if c != '#'],
                             label=t('Wählen Sie eine Zeile, um sie ganz zu sehen.', 'Select a row to see it in full.'))
    return (gold_table,)


@app.cell(hide_code=True)
def _(GTR, card, cards, chunks, gold, gold_table, hint, key, manual, mo, site_table, synthetic, t, theme):
    # Section D (3i): our two data sources, two cards; normal adds one gold row and two tables to browse
    _grey = f'<span style="color:{theme.GREY};font-size:.85rem">'

    def _corpus(de):
        """German corpus text marked for the language check, with the English translation in the English version."""
        _en = GTR.get(de) if t.lang == 'en' else None
        return f'<span class="hpi-corpus">{de}</span>' + (f'\n\n{_grey}{_en}</span>' if _en and _en != de else '')

    _site = card(t('KISZ-Webseite', 'AI Service Center website'), t(
        f'- Webseite des KI-Servicezentrums, {len(chunks)} Zeilen, jede Zeile ein Chunk\n- {len(manual)} Fragen mit Referenzantwort von Hand, {len(synthetic)} von RAGAS erzeugt\n- kurze Texte, die Sie selbst beurteilen können, zum Nachvollziehen\n- in Notebook 1 und in den Aufgaben zum Selbst-Beurteilen in Notebook 2 und 3',
        f'- website of the AI Service Center, {len(chunks)} lines, each line a chunk\n- {len(manual)} questions with reference answers by hand, {len(synthetic)} generated by RAGAS\n- short texts you can judge yourself, to follow along\n- in notebook 1 and in the self-assessment tasks of notebooks 2 and 3'))
    _gold = card(t('Gold-Datensatz', 'Gold data set'), t(
        f'- IT-Grundschutz-Kompendium, {len(gold)} Fragen von Fachleuten\n- zu jeder Frage eine Referenzantwort und eine Fundstelle\n- viele Fragen für belastbare Mittelwerte, zum Messen\n- für die Kurven und die Gesamtansicht in Notebook 2 und 3',
        f'- IT-Grundschutz-Kompendium, {len(gold)} questions by experts\n- a reference answer and a gold passage for each question\n- many questions for reliable means, to measure\n- for the curves and the overall view in notebooks 2 and 3'))
    _parts = [
        mo.md(t('## D. Unsere Daten', '## D. Our data')),
        key(t('Im Workshop arbeiten wir mit zwei Datenquellen: kurzen Texten zum Nachvollziehen und einem Gold-Datensatz zum Messen.',
              'In the workshop we work with two data sources: short texts to follow along and a gold data set to measure.')),
        cards([_site, _gold]),
    ]
    if not t.short and gold:
        _ex = next((r for r in gold if r['question'].startswith('Weshalb sind Passwortregeln')), gold[0])
        _fund = _ex['fundstelle']
        _short_fund = _fund if len(_fund) <= 380 else _fund[:380].rsplit(' ', 1)[0] + ' ...'
        _arrow = mo.md('&#8594;').style({'font-size': '1.4rem', 'color': theme.GREY, 'padding-top': '2.2rem'})
        _parts += [
            mo.md(t('### Aufbau des Gold-Datensatzes', '### Structure of the gold data set')),
            mo.hstack([card(t('Frage', 'Question'), _corpus(_ex['question'])), _arrow,
                       card(t('Referenzantwort', 'Reference answer'), _corpus(_ex['reference'])), _arrow,
                       card(t('Fundstelle', 'Gold passage'), _corpus(' '.join(_short_fund.split())))],
                      widths=[3, 0.25, 4, 0.25, 5], gap=0.5, align='stretch'),
            hint(t('Eine von 43 Zeilen. Die Fundstelle ist der Abschnitt im Kompendium, aus dem die Referenzantwort stammt.',
                   'One of 43 rows. The gold passage is the section of the Kompendium that the reference answer comes from.')),
            mo.accordion({
                t('Die Webseite Zeile für Zeile', 'The website line by line'): site_table,
                t('Der Gold-Datensatz: 43 Fragen mit Referenz und Fundstelle', 'The gold data set: 43 questions with reference and gold passage'): gold_table,
            }),
        ]
    mo.vstack(_parts, gap=1)
    return


@app.cell(hide_code=True)
def _(GTR, bi, gold, gold_table, info, mo, t, theme):
    # The selected gold row in full (normal only)
    mo.stop(t.short or gold_table.value is None or len(gold_table.value) == 0)
    _r = gold[int(gold_table.value.iloc[0]['#']) - 1]

    def _de_en(text):
        _en = GTR.get(text) if t.lang == 'en' else None
        _orig = mo.Html(f'<div class="hpi-corpus">{mo.md(text).text}</div>')
        return _orig if not _en or _en == text else mo.vstack([_orig, mo.Html(f'<div class="hpi-hint">{mo.md(_en).text}</div>')], gap=0.25)

    info(t(f'Zeile {_r["index"] + 1} des Gold-Datensatzes', f'Row {_r["index"] + 1} of the gold data set'), mo.vstack([
        mo.md(f"**{t('Frage', 'Question')}**"), _de_en(_r['question']),
        mo.md(f"**{t('Referenzantwort', 'Reference answer')}**"), _de_en(_r['reference']),
        mo.md(f"**{t('Fundstelle', 'Gold passage')}**"), _de_en(_r['fundstelle']),
    ], gap=0.4))
    return


@app.cell(hide_code=True)
def _(chain_html, hint, key, manual, mo, t):
    # Section D (slide): one end-to-end example; the chain names the steps in the order of Figure 1
    _options = {r['question']: i for i, r in enumerate(manual)}
    ui_q = mo.ui.dropdown(_options, value='Welche Angebote bietet das KI-Servicezentrum an?',
                          label=t('Wählen Sie eine Frage', 'Choose a question'), full_width=True)
    mo.vstack([
        mo.md(t('## E. Einmal durch die ganze Pipeline: ein Beispiel', '## E. Once through the whole pipeline: an example')),
        key(t('Eine Frage läuft durch die Pipeline; Sie urteilen zuerst selbst, dann der Richter.',
              'One question runs through the pipeline; you judge first, then the judge does.')),
        chain_html(t(['Frage', 'gefundene Chunks', 'Antwort', 'Richter', 'vier Kennzahlen'],
                     ['question', 'retrieved chunks', 'answer', 'judge', 'four metrics']), judge_index=3),
        hint(t('Die Pipeline kennt nur die Webseite des KI-Servicezentrums. Jede Zeile ist ein Chunk; die drei ähnlichsten gehen an das Sprachmodell.',
               'The pipeline knows only the website of the AI Service Center. Every line is a chunk; the three most similar go to the language model.')),
        *([mo.md(t('Bleiben Sie bei der voreingestellten Frage.', 'Stay with the preset question.'))] if t.short else []),
        ui_q,
    ], gap=1)
    return (ui_q,)


@app.cell(hide_code=True)
def _(bi, chunk_list, en_of, hint, info, manual, mo, t, task, ui_q):
    # Rows 5 and 6 as two columns: the pipeline's answer left, the source and the own judgment right
    _r = manual[ui_q.value]
    _q_en = en_of(_r['question'])
    ui_judge = mo.ui.radio(
        {t('gut', 'good'): 'good', t('teilweise richtig', 'partly correct'): 'partly',
         t('schlecht', 'poor'): 'poor'}, inline=True,
        label=t('Wie gut ist die Antwort?', 'How good is the answer?'))
    mo.vstack([
        task('judge', t('Selbst urteilen', 'Judge for yourself'), t(
            ['Lesen Sie die Antwort der Pipeline links.', 'Vergleichen Sie sie mit der Stelle auf der Webseite rechts.',
             'Wählen Sie unten, wie gut die Antwort ist, bevor Sie weiterlesen.'],
            ['Read the answer of the pipeline on the left.', 'Compare it with the passage of the website on the right.',
             'Choose below how good the answer is, before you read on.']), t('etwa 2 Minuten', 'about 2 minutes')),
        *([hint(_q_en)] if _q_en else []),
        mo.hstack([
            mo.vstack([
                mo.md(f"#### {t('Antwort der Pipeline', 'Answer of the pipeline')}"),
                bi(_r['answer']),
                mo.accordion({t('Die drei gefundenen Chunks', 'The three retrieved chunks'): chunk_list(_r)}),
            ]),
            info(t('Das steht auf der Webseite', 'This is what the website says'),
                 mo.vstack([mo.vstack([mo.md(f"*{t('Zeile', 'Line')} {n}*"), bi(line)])
                            for n, line in zip(_r['source_ids'], _r['source_lines'])], gap=1)),
        ], widths='equal', gap=2, align='start'),
        ui_judge,
    ], gap=1)
    return (ui_judge,)


@app.cell(hide_code=True)
def _(GROUPS, NAMES, bi, hint, info, manual, mo, t, theme, ui_judge, ui_q):
    # Row 7 (slide part): the reference left, the four tiles of the judge right
    mo.stop(ui_judge.value is None)
    _r = manual[ui_q.value]
    # Variant c of the review: this reference knows more than the website, shown as a learning moment.
    _beyond = info(t('Hinweis', 'Note'), mo.md(t(
        'Die Referenzantwort sagt, dass Pilotprojekte drei Monate dauern; die Webseite sagt nur, dass man sich alle drei Monate bewerben kann. Der Richter bewertet dann gegen etwas, das die Pipeline gar nicht finden konnte.',
        'The reference answer says that pilot projects last three months; the website only says that you can apply every three months. The judge then scores against something the pipeline could not find at all.',
    ))) if _r['question'].startswith('Welche Beratungsangebote') else None
    _tiles = mo.vstack([
        mo.vstack([
            mo.md(f"**{t(_de, _en)}**").style({'color': theme.GREY}),
            mo.hstack([
                mo.stat(t.num(_r[k]), label=NAMES[k], bordered=True,
                        caption='Median' if f'{k}_runs' in _r else None).style(
                    {'outline': f'2px solid {theme.INK}', 'border-radius': '8px'} if k == 'answer_correctness' else {})
                for k in _keys
            ], justify='start', gap=1),
        ], gap=0.5)
        for (_de, _en), _keys in GROUPS
    ], gap=1)
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md(f"#### {t('Referenzantwort', 'Reference answer')}"),
                       hint(t('Das hätte eine Fachperson geantwortet.', 'This is what an expert would have answered.')),
                       bi(_r['reference']), *([_beyond] if _beyond is not None else [])]),
            mo.vstack([mo.md(f"#### {t('Richter: vier Kennzahlen', 'Judge: four metrics')}"), _tiles]),
        ], widths='equal', gap=2, align='start'),
        mo.md(t('**Ihr Urteil vergleichen Sie mit Answer Correctness.**', '**You compare your judgment with Answer Correctness.**')),
        hint(t('Median aus drei Bewertungen; warum der Richter schwankt, sehen Sie in Notebook 2.',
               'Median of three evaluations; why the judge varies, you will see in notebook 2.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(hint, left, md_table, mo, t):
    # Row 8 (3h): the table about single metrics leaves the main path; a pointer and two hint lines stay
    _who = md_table([
        {t('Kennzahl', 'Metric'): 'Context Precision', t('prüft', 'checks'): t('jeden gefundenen Chunk', 'each retrieved chunk'),
         t('wogegen', 'against'): t('Frage und Referenzantwort', 'question and reference answer')},
        {t('Kennzahl', 'Metric'): 'Context Recall', t('prüft', 'checks'): t('jede Aussage der Referenzantwort', 'each statement of the reference answer'),
         t('wogegen', 'against'): t('gefundene Chunks', 'retrieved chunks')},
        {t('Kennzahl', 'Metric'): 'Answer Correctness', t('prüft', 'checks'): t('Aussagen der Antwort; ein Viertel des Werts kommt aus der Ähnlichkeit der Embeddings', 'statements of the answer; a quarter of the value comes from the similarity of the embeddings'),
         t('wogegen', 'against'): t('Referenzantwort', 'reference answer')},
        {t('Kennzahl', 'Metric'): 'Faithfulness', t('prüft', 'checks'): t('jede Aussage der Antwort', 'each statement of the answer'),
         t('wogegen', 'against'): t('gefundene Chunks', 'retrieved chunks')},
    ])
    mo.vstack([
        mo.md(t('Was die vier Zahlen genau messen, sehen Sie in Notebook 2 und 3.', 'What the four numbers measure exactly, you will see in notebooks 2 and 3.')),
        mo.accordion({t('Wer urteilt wogegen?', 'Who judges against what?'): mo.md(left(_who))}),
        hint(t('Es gibt keinen festen Wert, ab dem eine Antwort gut ist: Die Zahlen sind zum Vergleichen da.', 'There is no fixed value above which an answer is good: the numbers are for comparing.')),
        hint(t('Richter = Evaluator-LLM = LLM-as-a-Judge.', 'Judge = evaluator LLM = LLM-as-a-judge.')),
    ], gap=0.75)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Row 9 (3h): one quiz question, about the data set
    ui_quiz2 = mo.ui.radio(
        {t('die markierte Fundstelle', 'the marked gold passage'): 'passage',
         t('eine Referenzantwort', 'a reference answer'): 'reference',
         t('nichts, er kennt die Antwort selbst', 'nothing, it knows the answer itself'): 'nothing'},
        label=t('Was braucht der Richter zu jeder Testfrage?', 'What does the judge need for each test question?'))
    return (ui_quiz2,)


@app.cell(hide_code=True)
def _(hint, key, mo, t):
    # Section E (3h): test data by experts or by RAGAS
    mo.vstack([
        mo.md(t('## F. Testdaten: von Fachleuten oder von RAGAS?', '## F. Test data: by experts or by RAGAS?')),
        key(t('Testdaten stammen aus eigenen Gold-Daten oder von RAGAS, links im Schaubild.',
              'Test data come from your own gold data or from RAGAS, on the left of the diagram.')),
        hint(t.full(
            t('Hier schreiben Sie selbst eine Referenzantwort zu einer Frage von RAGAS und vergleichen sie mit der von RAGAS.',
              'Here you write a reference answer yourself for a question by RAGAS and compare it with the one by RAGAS.'),
            t('Ab Notebook 2 arbeiten wir mit Gold-Daten von Fachleuten.', 'From notebook 2 on we work with gold data by experts.'))),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, quiz_feedback, t, task, ui_quiz2):
    # Before step 1: the quiz as its own step, one question about the data set
    mo.vstack([
        mo.md(t('### Kurz geprüft', '### Quick check')),
        task('quiz', '', t(['Beantworten Sie die Frage.', 'Lesen Sie die Rückmeldung darunter.'],
                           ['Answer the question.', 'Read the feedback below it.']), t('etwa 1 Minute', 'about 1 minute')),
        ui_quiz2,
        quiz_feedback(ui_quiz2.value, 'reference', t(
            'Ohne Referenzantwort hat der Richter nichts, wogegen er urteilen kann. Die Fundstelle braucht er nicht, weil er Inhalte vergleicht.',
            'Without a reference answer the judge has nothing to judge against. It does not need the gold passage, because it compares content.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    # Session state, never cached: the accepted reference of E, the live rows of extra task a, the crawl
    get_accepted, set_accepted = mo.state({'text': None, 'error': None})
    get_own, set_own = mo.state({'rows': [], 'error': None})
    get_crawl, set_crawl = mo.state(None)
    get_used, set_used = mo.state(0)  # successful live evaluations (extra task a)
    return get_accepted, get_crawl, get_own, get_used, set_accepted, set_crawl, set_own, set_used


@app.cell(hide_code=True)
def _(bi, card, hint, info, mo, ragas_note, synthetic, t):
    # 3g step 1 (normal only): the RAGAS question and its source in an information box, the pipeline answer visible below
    WRITE_I = next(i for i, r in enumerate(synthetic) if 'source_ids' in r)  # the Newsletter question, lines 13 and 36
    _r = synthetic[WRITE_I]
    _note = ragas_note(_r['question'], 'question')
    t.full(mo.vstack([
        mo.md(t('### Schritt 1: Frage und Antwort des Systems', '### Step 1: question and answer of the system')),
        info(t('Frage von RAGAS und Quelle', 'Question by RAGAS and source'), mo.hstack([
            mo.vstack([mo.md(f"**{t('Frage von RAGAS', 'Question by RAGAS')}**"), bi(_r['question']), *([_note] if _note else [])]),
            mo.vstack([mo.md(f"**{t('Quelle auf der Webseite', 'Source on the website')}**"),
                       *[mo.vstack([mo.md(f"*{t('Zeile', 'Line')} {n}*"), bi(line)])
                         for n, line in zip(_r['source_ids'], _r['source_lines'])]], gap=0.5),
        ], widths='equal', gap=2, align='start')),
        card(t('Antwort der Pipeline', 'Answer of the pipeline'), _r['answer'], corpus=True),
        hint(t('Diese Antwort soll bewertet werden. Dafür braucht der Richter eine Referenzantwort.',
               'This answer is to be scored. For that, the judge needs a reference answer.')),
    ], gap=1))
    return (WRITE_I,)


@app.cell(hide_code=True)
def _(mo, t, task, theme):
    # Step 2 (normal only): write the reference answer, then confirm it with the green button
    ui_myref = mo.ui.text_area(label=t('Ihre Referenzantwort', 'Your reference answer'), rows=4, full_width=True)
    ui_accept = mo.ui.run_button(label=theme.cta_label(t('Referenzantwort übernehmen', 'Use this reference answer')))
    t.full(mo.vstack([
        mo.md(t('### Schritt 2: Ihre Referenzantwort', '### Step 2: your reference answer')),
        task('write', '', t(['Schreiben Sie die richtige Antwort auf die Frage, nur mit dem, was in der Quelle steht.',
                             'Drücken Sie auf "Referenzantwort übernehmen".'],
                            ['Write the correct answer to the question, using only what the source says.',
                             'Press "Use this reference answer".']),
             t('etwa 3 Minuten', 'about 3 minutes')),
        ui_myref,
        mo.Html(f'<div class="hpi-cta">{ui_accept}</div>'),
    ], gap=1))
    return ui_accept, ui_myref



@app.cell(hide_code=True)
def _(mo, set_accepted, ui_accept, ui_myref):
    # Takes over the written reference; step 3 shows this text, even if the field changes later
    mo.stop(not ui_accept.value)
    _text = ui_myref.value.strip()
    set_accepted({'text': _text, 'error': None} if _text else {'text': None, 'error': 'empty'})
    return



@app.cell(hide_code=True)
def _(mo, t):
    # The four criteria of step 3, defined apart so a second "übernehmen" does not reset them
    _yn = {t('ja', 'yes'): 'yes', t('nein', 'no'): 'no'}
    ui_crit = mo.ui.array([
        mo.ui.radio({t('ja', 'yes'): 'yes', t('teilweise', 'partly'): 'partly', t('nein', 'no'): 'no'}, inline=True,
                    label=t('Enthält sie alles Wichtige aus der Quelle?', 'Does it contain everything important from the source?')),
        mo.ui.radio(_yn, inline=True, label=t('Steht etwas darin, das nicht in der Quelle steht?', 'Does it say anything that is not in the source?')),
        mo.ui.radio(_yn, inline=True, label=t('Ist sie für eine Referenzantwort zu lang?', 'Is it too long for a reference answer?')),
        mo.ui.radio(_yn, inline=True, label=t('Ist sie in der Sprache der Quelle?', 'Is it in the language of the source?')),
    ])
    return (ui_crit,)


@app.cell(hide_code=True)
def _(WRITE_I, card, cards, get_accepted, hint, mo, ragas_note, synthetic, t, task, ui_crit):
    # Step 3 (normal only), after "übernehmen": both references side by side, then the own assessment of the RAGAS one
    mo.stop(t.short)
    _s = get_accepted()
    mo.stop(_s['error'] == 'empty', mo.callout(mo.md(t('Schreiben Sie zuerst eine Referenzantwort.', 'Write a reference answer first.')), kind='warn'))
    mo.stop(_s['text'] is None)
    _mine, _ragas = _s['text'], synthetic[WRITE_I]['reference']

    def _words(s):
        return t(f'{len(s.split())} Wörter', f'{len(s.split())} words')

    _note = ragas_note(_ragas, 'reference')
    _context = [t('Vergleichen Sie mit den Zeilen 13 und 36 der Quelle.', 'Compare with lines 13 and 36 of the source.'),
                t('Achten Sie auf Aussagen, die über die Quelle hinausgehen.', 'Look for statements that go beyond the source.'),
                t('Eine Referenz soll die Frage beantworten, mehr nicht.', 'A reference should answer the question, nothing more.'),
                t('Die Webseite ist deutsch.', 'The website is German.')]
    mo.vstack([
        mo.md(t('### Schritt 3: Ihre Referenz und die von RAGAS', '### Step 3: your reference and the one by RAGAS')),
        cards([
            card(t('Ihre Referenzantwort', 'Your reference answer'), _mine, _words(_mine), corpus=True),
            mo.vstack([card(t('Referenzantwort von RAGAS', 'Reference answer by RAGAS'), _ragas, _words(_ragas), corpus=True),
                       *([_note] if _note else [])], gap=0.25),
        ]),
        task('assess', t('Wie gut ist die Referenz von RAGAS?', 'How good is the reference by RAGAS?'), t(
            ['Beantworten Sie die vier Fragen zur Referenzantwort von RAGAS.', 'Lesen Sie danach die Auflösung.'],
            ['Answer the four questions about the reference answer by RAGAS.', 'Then read the solution.']),
            t('etwa 2 Minuten', 'about 2 minutes')),
        *[mo.vstack([ui_crit[i], hint(_context[i])], gap=0.1) for i in range(4)],
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(WRITE_I, feedback, get_accepted, left, md_table, mo, synthetic, t, ui_crit):
    # Solution of step 3 from the real texts, once all four criteria are answered; in the feedback box (meta plan 10.4a),
    # right when all four choices match the solution, each choice marked with a check mark or a cross
    mo.stop(t.short or get_accepted()['text'] is None or any(v is None for v in ui_crit.value))
    _n_ragas, _n_pipe = len(synthetic[WRITE_I]['reference'].split()), len(synthetic[WRITE_I]['answer'].split())
    _label = {'yes': t('ja', 'yes'), 'no': t('nein', 'no'), 'partly': t('teilweise', 'partly')}
    _rows = [
        (t('alles Wichtige aus der Quelle', 'everything important from the source'), 'partly',
         t('Teilweise. Sie nennt den Newsletter und den Starttermin, aber nicht, dass Bewerbungen alle drei Monate möglich sind, und nicht das Bewerbungsformular.',
           'Partly. It names the newsletter and the start date, but not that applications are possible every three months, and not the application form.')),
        (t('etwas, das nicht in der Quelle steht', 'something not in the source'), 'yes',
         t('Ja, ein wenig. Der Zusatz "<span class=\"hpi-corpus\">allowing you to receive timely updates</span>" steht so nicht in der Quelle.',
           'Yes, a little. The addition "<span class=\"hpi-corpus\">allowing you to receive timely updates</span>" is not in the source like that.')),
        (t('zu lang', 'too long'), 'no',
         t(f'Eher nein. Mit {_n_ragas} Wörtern ist sie länger als nötig (die Antwort der Pipeline braucht {_n_pipe}); andere Referenzen von RAGAS in diesem Datensatz haben über 100 Wörter.',
           f'Rather not. At {_n_ragas} words it is longer than needed (the pipeline answer takes {_n_pipe}); other RAGAS references in this data set have over 100 words.')),
        (t('in der Sprache der Quelle', 'in the language of the source'), 'no',
         t('Nein. Die Webseite ist deutsch, die Referenz englisch.', 'No. The website is German, the reference is English.')),
    ]
    _table = md_table([{t('Frage', 'Question'): q,
                        t('Ihre Wahl', 'Your choice'): f"<span style=\"white-space:nowrap\">{_label[v]}&nbsp;{'&#10003;' if v == right else '&#10007;'}</span>",
                        t('Auflösung', 'Solution'): r}
                       for (q, right, r), v in zip(_rows, ui_crit.value)])
    feedback(all(v == right for (_q, right, _r), v in zip(_rows, ui_crit.value)), left(_table))
    return



@app.cell(hide_code=True)
def _(card, cards, get_accepted, hint, mo, t):
    # 3h, for everyone: experts or RAGAS, pros and cons as two cards
    _experts = t(
        'Vorteile\n\n- realistische Fragen\n- präzise Referenzantworten\n- Fundstellen\n\nNachteile\n\n- aufwendig und teuer\n- wenige Fragen',
        'Pros\n\n- realistic questions\n- precise reference answers\n- gold passages\n\nCons\n\n- laborious and expensive\n- few questions')
    _ragas = t(
        'Vorteile\n\n- schnell\n- viele Fragen\n- deckt das Dokument breit ab\n\nNachteile\n\n- Sprache und Tippfehler\n- unrealistische oder zu lange Fragen und Referenzen\n- Fehler möglich',
        'Pros\n\n- fast\n- many questions\n- covers the document broadly\n\nCons\n\n- language and typos\n- unrealistic or overly long questions and references\n- errors possible')
    mo.vstack([
        mo.md(t('### Fachleute oder RAGAS?', '### Experts or RAGAS?')),
        cards([card(t('Datensatz von Fachleuten', 'Data set by experts'), _experts),
               card(t('Datensatz von RAGAS', 'Data set by RAGAS'), _ragas)]),
        mo.md(t('**In der Praxis: RAGAS für Entwürfe, Fachleute prüfen und korrigieren.**',
                '**In practice: RAGAS for drafts, experts check and correct.**')),
        # The look back only once step 3 exists.
        *([] if t.short or get_accepted()['text'] is None else [hint(t('Genau das haben Sie in Schritt 3 gesehen: Die Referenz von RAGAS ist englisch, lässt Wichtiges weg und schmückt aus.',
                        'That is what you saw in step 3: the RAGAS reference is in English, leaves out important points and embellishes.'))]),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(bi, bi_cell, hint, manual, md_table, mo, site_table, synthetic, t, task):
    # Extra task (normal only, 3h): part a, an own test question (live), and part b, the question types.
    # The button and the feedback sit in the cells below, so the accordion is never redrawn while someone works in it.
    ui_own_q = mo.ui.text_area(label=t('Frage', 'Question'), rows=2, full_width=True)
    ui_own_ref = mo.ui.text_area(label=t('Referenzantwort', 'Reference answer'), rows=3, full_width=True)
    ui_hops = mo.ui.array([mo.ui.dropdown({'single hop': 'single', 'multi hop': 'multi'}) for _ in synthetic])
    _manual_rows = [{'#': str(i + 1), t('Frage', 'Question'): bi_cell(r['question']),
                     t('Referenzantwort', 'Reference answer'): bi_cell(r['reference'])}
                    for i, r in enumerate(manual)]
    _hop_rows = [mo.hstack([ui_hops[i], bi(r['question'])], widths=[1, 5], align='center')
                 for i, r in enumerate(synthetic)]
    t.full(mo.vstack([
        task('extra', '', t(
            ['Öffnen Sie den Kasten "Zusatzaufgabe" darunter.', 'Wählen Sie Teil a oder Teil b und bearbeiten Sie ihn.',
             'Rückmeldungen und den Knopf zum Bewerten finden Sie unter dem Kasten.', 'Alle anderen machen mit Abschnitt G weiter.'],
            ['Open the box "Extra task" below.', 'Pick part a or part b and work on it.',
             'You find feedback and the Evaluate button below the box.', 'Everyone else continues with section G.']),
            t('so lange, bis die anderen fertig sind', 'until the others are done')),
        mo.accordion({t('Zusatzaufgabe', 'Extra task'): mo.vstack([
            mo.md(t(
                '#### Teil a: eine eigene Testfrage\n\nSo entstehen Testfragen in der Praxis: Jemand sucht eine Stelle im Dokument, schreibt eine Frage dazu und die Antwort, die dort steht. Machen Sie es genauso: Suchen Sie in der Tabelle rechts eine Zeile, schreiben Sie eine Frage und die Antwort aus dieser Zeile, und drücken Sie dann auf Bewerten.',
                '#### Part a: your own test question\n\nThis is how test questions come about in practice: someone finds a passage in the document, writes a question about it and the answer that is there. Do the same: find a line in the table on the right, write a question and the answer from that line, then press Evaluate.',
            )),
            mo.hstack([mo.vstack([ui_own_q, ui_own_ref]), site_table], widths=[2, 3], gap=2, align='start'),
            mo.accordion({t('Die fünf von Hand geschriebenen Fragen aus Abschnitt E', 'The five hand-written questions from section E'): mo.md(md_table(_manual_rows))}),
            mo.md(t(
                '#### Teil b: Fragetypen\n\nRAGAS erzeugt zwei Arten von Fragen. Bei single hop steht die Antwort an einer Stelle des Dokuments, bei multi hop muss man mehrere Stellen verbinden. Ordnen Sie alle fünf synthetischen Fragen zu.\n\nDie Einteilung beschreibt, wie RAGAS die Frage erzeugt hat (aus einem oder aus mehreren Abschnitten), nicht wie lang die Frage ist.',
                '#### Part b: question types\n\nRAGAS generates two kinds of questions. With single hop the answer is in one place in the document; with multi hop you have to combine several places. Assign all five synthetic questions.\n\nThe split describes how RAGAS generated the question (from one section or from several), not how long the question is.',
            )),
            mo.hstack([mo.vstack(_hop_rows, gap=1), site_table], widths=[2, 3], gap=2, align='start'),
            hint(t('Vier der fünf Fragen hat RAGAS auf Englisch erzeugt, obwohl die Webseite deutsch ist, die fünfte in fehlerhaftem Deutsch: eine Schwäche der automatischen Testdaten.',
                   'RAGAS generated four of the five questions in English although the website is German, the fifth in poor German: a weakness of automatic test data.')),
        ])}),
    ], gap=1))
    return ui_hops, ui_own_q, ui_own_ref


@app.cell(hide_code=True)
def _(eval_button, get_used, hint, mo, t, ui_own_q):
    # Extra task, part a: the button, directly under the accordion once a question has been typed
    ui_own_run = eval_button(get_used())
    mo.stop(t.short or not ui_own_q.value.strip())
    mo.vstack([
        mo.hstack([mo.md(t('**Teil a**', '**Part a**')), ui_own_run], justify='start', align='center', gap=1),
        hint(t('Sie können bis zu dreimal bewerten lassen.', 'You can have it scored up to three times.')),
    ], gap=0.25)
    return (ui_own_run,)


@app.cell(hide_code=True)
async def _(LIMIT, chunks, get_used, mo, rag, set_own, set_used, t, ui_own_q, ui_own_ref, ui_own_run, vectors):
    mo.stop(not ui_own_run.value or get_used() >= LIMIT)
    _q, _ref = ui_own_q.value.strip(), ui_own_ref.value.strip()
    if not (_q and _ref):
        set_own(lambda s: {**s, 'error': 'empty'})
    else:
        try:
            with mo.status.spinner(title=t('Retrieval, Antwort und Richter laufen ...', 'Retrieval, answer and judge are running ...')):
                _row = await rag.aevaluate_one(_q, _ref, chunks, vectors)
        except Exception:  # noqa: BLE001 - any API failure gets the same sentence
            set_own(lambda s: {**s, 'error': 'api'})
        else:
            set_own(lambda s: {'rows': s['rows'] + [_row], 'error': None})
            set_used(lambda n: n + 1)
    return


@app.cell(hide_code=True)
def _(JUDGE_DOWN, bi, chunk_list, get_own, hint, md_table, mo, t):
    # Extra task, part a: latest result under the button, all own rows as a small table
    mo.stop(t.short)
    _err = get_own()['error']
    _msg = {
        'api': mo.callout(mo.md(t(*JUDGE_DOWN)), kind='warn'),
        'empty': mo.callout(mo.md(t('Tragen Sie zuerst Frage und Referenzantwort ein.',
                                    'Enter a question and a reference answer first.')), kind='warn'),
    }.get(_err)
    _rows = get_own()['rows']
    _last = _rows[-1] if _rows and _err is None else None

    def _clean(s):
        return ' '.join(str(s).split()).replace('|', '/')

    _result = None if _last is None else mo.vstack([
        mo.hstack([
            mo.stat(t.num(_last['answer_correctness']), label='Answer Correctness', bordered=True),
            mo.stat(t.num(_last['context_precision']), label='Context Precision', bordered=True),
        ], justify='start', gap=2),
        hint(t('Hier bewerten wir live nur zwei der vier Kennzahlen, damit es schnell geht. Schauen Sie zuerst auf Answer Correctness.',
               'Here we score only two of the four metrics live, so that it is quick. Look at Answer Correctness first.')),
        mo.accordion({t('Antwort und Chunks', 'Answer and chunks'): mo.vstack([
            mo.md(f"#### {t('Antwort', 'Answer')}"), bi(_last['answer']), chunk_list(_last)])}),
        *([mo.md(md_table([{t('Ihre Frage', 'Your question'): _clean(r['question']),
                            t('Referenzantwort', 'Reference answer'): _clean(r['reference']),
                            'Answer Correctness': t.num(r['answer_correctness']),
                            'Context Precision': t.num(r['context_precision'])} for r in _rows]))]
          if len(_rows) > 1 else []),
        mo.md(t(
            'Wollen Sie sehen, wie stark der Richter schwankt? Einigen Sie sich mit Ihrem Nachbarn auf dieselbe Frage und dieselbe Referenzantwort, und bewerten Sie beide. Die Zahlen werden sich trotzdem oft unterscheiden. Warum, sehen Sie in Notebook 2.',
            'Want to see how much the judge varies? Agree with your neighbor on the same question and the same reference answer, and both evaluate it. The numbers will still often differ. Notebook 2 shows why.',
        )),
    ])
    _parts = [v for v in (_msg, _result) if v is not None]
    mo.vstack(_parts) if _parts else None
    return


@app.cell(hide_code=True)
def _(bi_cell, feedback, md_table, mo, synthetic, t, ui_hops):
    # Extra task, part b: the solution under the box once all five questions are assigned; in the feedback box
    # (meta plan 10.4a), right when all five match RAGAS, each choice marked with a check mark or a cross
    mo.stop(t.short or not all(v is not None for v in ui_hops.value))
    _label = {'single': 'single hop', 'multi': 'multi hop'}
    _truth = ['multi' if r['synthesizer_name'].startswith('multi') else 'single' for r in synthetic]
    feedback(list(ui_hops.value) == _truth, md_table([
        {t('Frage', 'Question'): bi_cell(r['question']),
         t('Ihre Zuordnung', 'Your choice'): f"<span style=\"white-space:nowrap\">{_label[v]}&nbsp;{'&#10003;' if v == _truth[i] else '&#10007;'}</span>",
         'RAGAS': _label[_truth[i]]}
        for i, (r, v) in enumerate(zip(synthetic, ui_hops.value))]))
    return


@app.cell(hide_code=True)
def _(card, cards, frame, hint, key, left, md_table, mo, t):
    # Section F (3h): summary with the focus on data set and pipeline, outlook, next step
    _outlook = md_table([
        {'Notebook': '2', t('bewertet', 'scores'): t('Suche', 'Retrieval'), t('Kennzahlen', 'Metrics'): 'Context Precision, Context Recall'},
        {'Notebook': '3', t('bewertet', 'scores'): t('Antwort', 'Answer'), t('Kennzahlen', 'Metrics'): 'Answer Correctness, Faithfulness'},
    ])
    mo.vstack([
        mo.md(t('## G. Zusammenfassung', '## G. Summary')),
        key(t('So evaluieren Sie Ihr RAG-System: Testfragen mit Referenzantworten sammeln, die Pipeline darauf laufen lassen, die vier Kennzahlen vom Richter messen lassen und Stichproben selbst prüfen.',
              'This is how you evaluate your RAG system: collect test questions with reference answers, run the pipeline on them, let the judge measure the four metrics and check samples yourself.')),
        cards([
            card(t('Pipeline', 'Pipeline'), t('Frage, gefundene Chunks, Antwort; ein Richter-LLM bewertet Suche und Antwort.',
                                              'Question, retrieved chunks, answer; a judge LLM scores retrieval and answer.')),
            card(t('Gold-Daten', 'Gold data'), t('Fragen mit Referenzantworten von Fachleuten sind das Fundament.',
                                                 'Questions with reference answers by experts are the foundation.')),
            card('RAGAS', t('Liefert schnell viele Entwürfe; Fachleute prüfen und korrigieren sie.',
                            'Delivers many drafts quickly; experts check and correct them.')),
        ]),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'RAGAS-Zahlen sind Urteile eines Sprachmodells gegen Ihre Referenzantworten. Gute Testdaten sind die Voraussetzung für aussagekräftige Zahlen.',
            'RAGAS numbers are judgments of a language model against your reference answers. Good test data are the precondition for meaningful numbers.')),
        mo.hstack([
            mo.vstack([mo.md(t('#### Ausblick', '#### Outlook')), mo.md(left(_outlook)),
                       # the short version of notebook 3 drops this part
                       *t.full([hint(t('Notebook 3 zeigt auch, wie eine andere Referenzantwort den Score verändert.',
                                       'Notebook 3 also shows how a different reference answer changes the score.'))], [])]),
            mo.vstack([
                mo.md(t('#### Nächster Schritt', '#### Next step')),
                mo.md(t('Weiter mit Notebook 2.', 'Continue with notebook 2.')),
                mo.md('`uv run marimo run w3_02_retrieval_evaluation.py`'),
                hint(t('Im Ordner notebooks.', 'In the notebooks folder.')),
                hint(t('Ab Notebook 2 arbeiten wir mit Gold-Daten von Fachleuten (links oben im Schaubild): 43 Fragen, die Fachleute zum IT-Grundschutz-Kompendium geschrieben haben.',
                       'From notebook 2 on we work with gold data by experts (top left in the diagram): 43 questions that experts wrote about the IT-Grundschutz-Kompendium.')),
            ]),
        ], widths='equal', gap=2, align='start'),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # FAQ F1 to F7; the crawl button of F7 renders inside, its result below.
    ui_crawl = mo.ui.run_button(label=t('Neu crawlen', 'Crawl again'))
    _f1 = {
        t('Warum ist jede Zeile ein Chunk?', 'Why is every line a chunk?'): mo.md(t(
            'Die Webseite ist kurz, und jede Zeile ist meist ein abgeschlossener Satz. Für eine Einführung reicht das. In Teil II haben Sie gesehen, dass die Chunk-Größe die Ergebnisse stark beeinflusst; in Notebook 2 arbeiten wir mit Abschnitten des Kompendiums.',
            'The website is short, and each line is usually a complete sentence. For an introduction that is enough. In part II you saw that chunk size strongly affects the results; in notebook 2 we work with sections of the Kompendium.')),
    }
    # F2 to F5 only in the normal version: the short one shows neither synthetic questions nor the neighbor comparison.
    _f2_f5 = {
        t('Was ist ein Knowledge Graph im TestsetGenerator?', 'What is a knowledge graph in the TestsetGenerator?'): mo.md(t(
            'RAGAS zerlegt das Dokument in Abschnitte, verbindet Abschnitte mit gemeinsamen Begriffen und erzeugt daraus Fragen, die eine Stelle (single hop) oder mehrere verbundene Stellen (multi hop) brauchen.',
            'RAGAS splits the document into sections, connects sections that share terms and generates questions from them that need one passage (single hop) or several connected passages (multi hop).')),
        t('Wofür nutzt man den TestsetGenerator?', 'What do you use the TestsetGenerator for?'): mo.md(t(
            'Als Startpunkt: Er liefert schnell viele Entwürfe, die Fachleute prüfen und korrigieren. Als alleinige Grundlage für eine Bewertung taugt er nicht.',
            'As a starting point: it quickly produces many drafts that experts check and correct. It is not fit to be the sole basis of an evaluation.')),
        t('Warum sind die synthetischen Fragen englisch?', 'Why are the synthetic questions in English?'): mo.md(t(
            'Die Prompts, mit denen RAGAS Fragen erzeugt, sind englisch. RAGAS kann sie übersetzen (`adapt_prompts`), dafür braucht es ein Modell, das Sprachvorgaben zuverlässig befolgt. Mit dem hier genutzten Modell gelang das nicht zuverlässig.',
            'The prompts RAGAS uses to generate questions are in English. RAGAS can translate them (`adapt_prompts`), which needs a model that follows language instructions reliably. With the model used here, that did not work reliably.')),
        t('Warum ist meine Zahl anders als die meines Nachbarn?', "Why is my number different from my neighbor's?"): mo.md(t(
            'Bei einer eigenen Frage laufen Antwort und Richter live, und Sprachmodelle antworten nicht jedes Mal gleich. Haben Sie unterschiedliche Referenzantworten geschrieben, unterscheiden sich die Zahlen schon deshalb. Die vorbereiteten Fragen kommen aus einem Cache und sind für alle gleich.',
            'For your own question, answer and judge run live, and language models do not answer the same way every time. If you wrote different reference answers, the numbers differ for that reason alone. The prepared questions come from a cache and are the same for everyone.')),
    }
    _f6_f7 = {
        t('Wo finde ich die Dokumentation von RAGAS?', 'Where do I find the RAGAS documentation?'): mo.md(t(
            'Unter [docs.ragas.io](https://docs.ragas.io). Dort sind die Kennzahlen und der TestsetGenerator beschrieben.',
            'At [docs.ragas.io](https://docs.ragas.io). It describes the metrics and the TestsetGenerator.')),
        t('Welche Modelle laufen hier?', 'Which models run here?'): mo.md(t(
            'Embedding: octen-embedding-8b. Antwort und Richter: gpt-oss-120b, dasselbe Modell in zwei Rollen. Alle über die AISC-API. Das Präfix `openai/` in den Modellnamen sagt LiteLLM nur, welches Protokoll es sprechen soll.',
            'Embedding: octen-embedding-8b. Answer and judge: gpt-oss-120b, the same model in two roles. All via the AISC API. The prefix `openai/` in the model names only tells LiteLLM which protocol to speak.')),
        t('Wie führe ich den Crawl selbst aus?', 'How do I run the crawl myself?'): mo.vstack([mo.md(t(
            'Die Chunks stammen aus einem gespeicherten Crawl der Webseite. Mit dem Button unten crawlt Docling die Seite neu; das dauert etwa eine Minute und braucht Internet. Die Ergebnisse oben ändern sich dadurch nicht.',
            'The chunks come from a saved crawl of the website. With the button below, Docling crawls the page again; this takes about a minute and needs internet. The results above do not change.')),
            ui_crawl]),
    }
    _faq = {**_f1, **t.full(_f2_f5, {}), **_f6_f7}
    mo.vstack([mo.md(t('## Häufige Fragen', '## Frequently asked questions')), mo.accordion(_faq)])
    return (ui_crawl,)


@app.cell(hide_code=True)
def _(mo, rag, set_crawl, t, ui_crawl):
    mo.stop(not ui_crawl.value)
    try:
        with mo.status.spinner(title=t('Docling crawlt die Webseite ...', 'Docling is crawling the website ...')):
            _n = len(rag.crawl())
    except Exception:  # noqa: BLE001 - no network, site down, converter error
        set_crawl(-1)
    else:
        set_crawl(_n)
    return


@app.cell(hide_code=True)
def _(get_crawl, mo, t):
    _n = get_crawl()
    mo.stop(_n is None)
    (mo.callout(mo.md(t('Der Crawl ist fehlgeschlagen. Prüfen Sie die Internetverbindung.',
                        'The crawl failed. Check the internet connection.')), kind='warn')
     if _n < 0 else mo.md(t(f'Der neue Crawl hat {_n} Chunks.', f'The new crawl has {_n} chunks.')))
    return


if __name__ == "__main__":
    app.run()
