# ruff: noqa: PLR1711, B018  marimo cells end with an explicit return and render a bare expression

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Findet mein System die richtigen Stellen?",
    css_file="ragkit/theme.css",
    html_head_file="ragkit/head.html",
)


@app.cell(hide_code=True)
def _(mo, t, theme):
    # Row 1: header; SECTIONS_MARKER makes every H2 section collapsible (script in ragkit/head.html)
    mo.Html(theme.header_html(
        t('Findet mein System die richtigen Stellen?', 'Does my system find the right passages?'),
        t('RAG III, Notebook 2. Sie bewerten die Suche, noch ohne Antwort, mit Context Precision und Context Recall.',
          'RAG III, notebook 2. You evaluate the retrieval, still without an answer, with Context Precision and Context Recall.'),
    ) + theme.SECTIONS_MARKER)
    return


@app.cell(hide_code=True)
def _():
    import json  # the cached diagnosis of the extra task, part b
    import os  # the API key check before a live judge run

    import marimo as mo  # the reactive notebook itself
    import matplotlib.pyplot as plt  # figures of sections C, E, F and the extra task
    import pandas as pd

    from ragkit import i18n, theme  # language and scope switches, HPI look
    from ragkit import intro_rag as rag  # the website lines of w3_01 (module A, C27)
    from ragkit import w3_eval as ev  # gold data, caches and the judge
    from ragkit.viz import md_table

    theme.apply_mpl()
    return ev, i18n, json, md_table, mo, os, pd, plt, rag, theme


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
def _(mo, t, theme):
    # Slide building blocks as in w3_01, shared in ragkit/theme.py: key sentence, cards, grey hint lines,
    # information and task boxes.
    key, card, cards, hint, frame, info, left = (theme.key_sentence, theme.card, theme.cards, theme.hint,
                                                 theme.frame, theme.info, theme.left)

    # Meta plan 10.4: free-text and number fields are confirmed with a green button (CTA style); choices reveal directly.
    def confirm(label, source):
        """A green confirm button. Its value is None before the first click, then {'v': ...} with the value of
        `source` (a UI element, or a function for several elements) at the moment of the click. Define it in the
        same cell as `source`, so typing never rebuilds the button."""
        _get = source if callable(source) else (lambda: source.value)
        return mo.ui.button(value=None, on_click=lambda _: {'v': _get()}, label=theme.cta_label(label))

    def cta(button):
        """A button in the CTA frame of w3_01."""
        return mo.Html(f'<div class="hpi-cta">{button}</div>')

    def confirmed(button):
        """The value confirmed with `button`: None before the first click."""
        return None if button.value is None else button.value['v']

    # Every place where participants act gets a numbered task box; the short version has no TOP_K choice.
    # Each quiz is its own counted step (CR-19, as in w3_01): "Kurz geprüft" before the task it prepares.
    TASK_ORDER = (['judge', 'quiz_cp', 'cp', 'quiz_cr', 'cr', 'check', 'spread', 'topk_predict']
                  + t.full(['quiz_topk', 'topk_choose'], []))

    def task(name, title, steps, time):
        """The task box of a numbered task, or of the extra task (name='extra')."""
        return theme.numbered_task(t, TASK_ORDER, name, title, steps, time)

    def quiz_task(name):
        """H3 "Kurz geprüft" and the counted task box of a quiz, as in w3_01."""
        return mo.vstack([
            mo.md(t('### Kurz geprüft', '### Quick check')),
            task(name, '', t(['Beantworten Sie die Frage.', 'Lesen Sie die Rückmeldung darunter.'],
                             ['Answer the question.', 'Read the feedback below it.']), t('etwa 1 Minute', 'about 1 minute')),
        ], gap=1)

    def chain_html(steps, judge_index):
        """Steps as a chain of boxes, the judge dark; the same look as the chain in w3_01, section D."""
        _box = ('display:inline-block;padding:.35rem .7rem;border:1.2px solid {b};border-radius:3px;'
                'background:{bg};color:{c};font-size:.85rem')
        _parts = [f'<span style="{_box.format(b=theme.INK, bg=theme.INK if i == judge_index else "#ffffff", c="#ffffff" if i == judge_index else theme.INK)}">{s}</span>'
                  for i, s in enumerate(steps)]
        _arrow = f'<span style="color:{theme.INK};padding:0 .35rem">&#8594;</span>'
        return mo.Html(f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:.25rem 0;background:#ffffff;'
                       f'padding:.6rem .8rem;border-radius:6px;border:1px solid {theme.RULE}">{_arrow.join(_parts)}</div>')
    return card, cards, chain_html, confirm, confirmed, cta, frame, hint, info, key, left, quiz_task, task


@app.cell(hide_code=True)
def _(card, cards, hint, key, mo, t):
    # Section A (agenda slide, section 3g): what this is about, the route as four cards, the data in one grey line
    mo.vstack([
        mo.md(t('## A. Worum es geht', '## A. What this is about')),
        key(t('Heute bewerten Sie nur die Suche: Findet sie die Stellen, die eine Fachperson zur Antwort braucht?',
              'Today you evaluate only the search: does it find the passages that an expert needs for the answer?')),
        mo.md(t('### Ihr Weg in diesem Notebook', '### Your route through this notebook')),
        cards([
            card(t('1. Urteilen', '1. Judge'), t('Sie entscheiden selbst, welche gefundenen Chunks nützlich sind.',
                                                 'You decide yourself which retrieved chunks are useful.')),
            card(t('2. Rechnen', '2. Compute'), t('Sie rechnen Context Precision und Context Recall von Hand aus.',
                                                  'You compute Context Precision and Context Recall by hand.')),
            card(t('3. Prüfen', '3. Check'), t('Sie prüfen, wie verlässlich das Urteil des Richters ist.',
                                               "You check how reliable the judge's verdict is.")),
            card(t('4. Entscheiden', '4. Decide'), t.full(
                t('Sie entscheiden, wie viele Chunks Ihr System abrufen soll.', 'You decide how many chunks your system should retrieve.'),
                t('Sie sehen, wie die Zahl der abgerufenen Chunks beide Kennzahlen verschiebt.',
                  'You see how the number of retrieved chunks shifts both metrics.'))),
        ]),
        hint(t('Die Daten kennen Sie aus Notebook 1: Zum Nachvollziehen nehmen wir die KISZ-Webseite, zum Messen den Gold-Datensatz.',
               'You know the data from notebook 1: to follow the steps we use the AI Service Center website, to measure the gold data set.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(ev, mo, rag, t):
    # Row 4 (C30): no note about the collection any more; a red callout only when the cache is missing
    cache_ok = ev.cache_complete() and rag.CHUNKS_FILE.exists()
    (None if cache_ok else mo.callout(mo.md(t(
        'Der Cache mit den Ergebnissen fehlt. Bauen Sie ihn einmal mit `uv run python -m tools.precompute_w3_02` im Ordner `notebooks`.',
        'The results cache is missing. Build it once with `uv run python -m tools.precompute_w3_02` in the `notebooks` folder.',
    )), kind='danger'))
    return (cache_ok,)


@app.cell(hide_code=True)
def _(cache_ok, ev, mo, rag):
    mo.stop(not cache_ok)
    # Module A (C27): one question about the website of the KI-Servicezentrum, its top-5 lines and five judge runs
    _lines = rag.load_corpus()[0]
    _kisz = ev.load_kisz()
    _q = _kisz['questions'][_kisz['demos'][0]]
    demo = {**_q, 'chunks': [_lines[i] for i in _q['line_ids']], 'run': _q['runs'][0],
            'sources': [(i, _lines[i]) for i in _q['source_ids']], 'second_judge': _kisz['second_judge']}
    # Module B: the gold data set (43 questions about the Kompendium) and the four cached configurations
    gold = ev.load_gold()
    retrieval = ev.load_retrieval()
    TR = ev.load_translations()  # {'en': {German: English}, 'de': {English: German}}
    KTR = ev.load_kisz_translations()
    CFG = {name: ev.load_config(name) for name in ev.CONFIGS}
    JUDGE_DOWN = ('Der Richter ist gerade nicht erreichbar. Probieren Sie es in einer Minute noch einmal.',
                  'The judge cannot be reached right now. Try again in a minute.')
    LIMIT = 3  # live judge runs per person (section E, part c)
    return CFG, JUDGE_DOWN, KTR, LIMIT, TR, demo, gold, retrieval


@app.cell(hide_code=True)
def _(KTR, TR, feedback, mo, t, theme):
    # Corpus and RAGAS output: shown unchanged and marked with the class hpi-corpus (the language check skips it);
    # in the English version a grey translation below.
    def en_of(text):
        """English translation of a German text from the cache; only in the English version, only when it differs."""
        if t.lang != 'en':
            return None
        _en = KTR['en'].get(text) or TR['en'].get(text)
        return _en if _en and _en.strip() != text.strip() else None

    def _corpus(text):
        return mo.Html(f'<div class="hpi-corpus">{mo.md(text).text}</div>')

    def bi(text):
        """Corpus text as it is, with its English translation below in small grey type."""
        _en = en_of(text)
        return _corpus(text) if _en is None else mo.vstack([_corpus(text), theme.hint(_en)], gap=0.25)

    def flat(s):
        return ' '.join(str(s).split()).replace('|', '/')

    def short(text, n=22):
        _w = flat(text).split()
        return ' '.join(_w[:n]) + (' ...' if len(_w) > n else '')

    def bi_cell(text, n=None):
        """`bi` for one Markdown table cell, optionally shortened to n words."""
        _cut = (lambda s: short(s, n)) if n else flat
        _en = en_of(text)
        _orig = f'<span class="hpi-corpus">{_cut(text)}</span>'
        return _orig if _en is None else f'{_orig}<br><span style="color:{theme.GREY};font-size:.85em">{_cut(_en)}</span>'

    _EN = (' the ', ' and ', ' of ', ' to ', ' is ', ' does ', ' not ', ' context ')

    def is_english(text):
        return sum(w in f' {text.lower()} ' for w in _EN) >= 3

    def nw(s):
        """A short table value that must not break inside the word."""
        return f'<span style="white-space:nowrap">{s}</span>'

    def yes_no(v):
        return t('ja', 'yes') if v else t('nein', 'no')

    def strip_html(labels_values):
        """A row of small boxes, dark for 'yes', light for 'no' (the judge's verdicts above an input field)."""
        _boxes = ''.join(
            f'<span style="display:inline-block;margin:0 .35rem .35rem 0;padding:.2rem .55rem;border:1px solid {theme.INK};'
            f'background:{theme.INK if v else "white"};color:{"white" if v else theme.INK};font-size:.85rem">{lab}: {yes_no(v)}</span>'
            for lab, v in labels_values)
        return mo.Html(f'<div>{_boxes}</div>')

    def parse_num(s):
        """A number typed with a decimal comma or point; None when it is not a number."""
        try:
            return float(str(s).strip().replace(',', '.'))
        except ValueError:
            return None

    def ap_steps(verdicts):
        """The computation of Context Precision, step by step, for the ranks of one question."""
        _hits, _terms, _steps = 0, [], []
        for _i, _v in enumerate(verdicts):
            if _v:
                _hits += 1
                _terms.append(_hits / (_i + 1))
                _steps.append(t(f'Rang {_i + 1}: Precision@{_i + 1} = {_hits} von {_i + 1} = {t.num(_terms[-1])}.',
                                f'Rank {_i + 1}: Precision@{_i + 1} = {_hits} of {_i + 1} = {t.num(_terms[-1])}.'))
        _ranks = [str(i + 1) for i, v in enumerate(verdicts) if v]
        _lead = t(f"Nützlich sind die Ränge {', '.join(_ranks[:-1])} und {_ranks[-1]}.",
                  f"Useful are ranks {', '.join(_ranks[:-1])} and {_ranks[-1]}.") if len(_ranks) > 1 else ''
        _sum = ' + '.join(t.num(x) for x in _terms)
        _cp = sum(_terms) / len(_terms) if _terms else 0.0
        return ' '.join([_lead, *_steps, f'Context Precision = ({_sum}) / {len(_terms)} = {t.num(_cp)}.'])

    def check_answer(value, target, wrong_text):
        """Feedback for a value confirmed with "Prüfen": None before the click, a hint for no number, right within 0.01,
        else the way."""
        if value is None:
            return None
        _x = parse_num(value)
        if _x is None:
            return mo.callout(mo.md(t('Geben Sie eine Zahl ein, zum Beispiel 0,50.', 'Enter a number, for example 0.50.')), kind='warn')
        return feedback(True) if abs(_x - target) <= 0.01 + 1e-9 else feedback(False, wrong_text)
    return ap_steps, bi, bi_cell, check_answer, en_of, is_english, nw, strip_html, yes_no


@app.cell(hide_code=True)
def _(key, mo, t, task):
    # Section B (rows 5 and 6, section 3f): the KISZ question, your own verdict first
    mo.vstack([
        mo.md(t('## B. Welche Chunks helfen?', '## B. Which chunks help?')),
        key(t('Sie urteilen zuerst selbst, mit der Quelle daneben; danach vergleichen Sie mit dem Richter.',
              'You judge first, with the source next to you; then you compare with the judge.')),
        task('judge', t('Selbst urteilen', 'Judge for yourself'), t(
            ['Lesen Sie links Frage, Referenzantwort und die Stelle auf der Webseite.',
             'Markieren Sie rechts jede der fünf gefundenen Zeilen als nützlich oder nicht nützlich: Hilft sie, die Frage so zu beantworten wie die Referenzantwort?',
             'Vergleichen Sie darunter Ihr Urteil mit dem des Richters.'],
            ['Read the question, the reference answer and the passage of the website on the left.',
             'Mark each of the five retrieved lines on the right as useful or not useful: does it help to answer the question the way the reference answer does?',
             "Compare your verdicts with the judge's below."]), t('etwa 4 Minuten', 'about 4 minutes')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(bi, demo, hint, info, mo, t):
    # Rows 5 and 6 as two columns: question, reference and source lines left; the five lines with a verdict each right
    _states = {t('nützlich', 'useful'): 'yes', t('nicht nützlich', 'not useful'): 'no',
               t('noch offen', 'not decided yet'): None}
    ui_my = mo.ui.array([mo.ui.radio(_states, value=t('noch offen', 'not decided yet'), inline=True) for _ in demo['chunks']])
    _left = mo.vstack([
        mo.md(f"#### {t('Frage', 'Question')}"), bi(demo['question']),
        mo.md(f"#### {t('Referenzantwort', 'Reference answer')}"),
        hint(t('Das hätte eine Fachperson geantwortet.', 'This is what an expert would have answered.')),
        bi(demo['reference']),
        info(t('Das steht auf der Webseite', 'This is what the website says'),
             mo.vstack([mo.vstack([mo.md(f"*{t('Zeile', 'Line')} {n}*"), bi(line)]) for n, line in demo['sources']], gap=1)),
    ])
    _right = mo.vstack([
        mo.md(f"#### {t('Die fünf gefundenen Zeilen', 'The five retrieved lines')}"),
        *[mo.vstack([mo.md(f"**{t('Rang', 'Rank')} {i + 1}** &nbsp; *{t('Zeile', 'line')} {n}*"), bi(c), ui_my[i]], gap=0.4)
          for i, (n, c) in enumerate(zip(demo['line_ids'], demo['chunks']))],
    ], gap=1.25)
    mo.hstack([_left, _right], widths='equal', gap=2, align='start')
    return (ui_my,)


@app.cell(hide_code=True)
def _(bi_cell, demo, ev, frame, hint, is_english, left, md_table, mo, t, ui_my, yes_no):
    # Row 6: the judge against your own verdicts; a hint after the first verdict, the comparison as soon as all five
    # are set (choices reveal directly, meta plan section 10.4)
    _mine = ui_my.value
    _open = sum(v is None for v in _mine)
    mo.stop(_open == len(_mine))
    _run = demo['run']
    if _open:
        _view = hint(t(f'Noch {_open} von 5 Zeilen offen. Der Vergleich mit dem Richter erscheint, sobald alle fünf beurteilt sind.',
                       f'{_open} of 5 lines still open. The comparison with the judge appears once all five are judged.'))
    else:
        _agree = sum((m == 'yes') == bool(c['verdict']) for m, c in zip(_mine, _run['chunks']))

        def _nw(s):  # narrow columns: keep the header word whole
            return f'<span style="white-space:nowrap">{s}</span>'

        _rows = [{_nw(t('Rang', 'Rank')): str(i + 1), _nw(t('Zeile', 'Line')): str(n), _nw(t('Sie', 'You')): _nw(yes_no(m == 'yes')),
                  _nw(t('Richter', 'Judge')): _nw(yes_no(c['verdict'])),
                  t('Begründung des Richters (RAGAS)', "Judge's reason (RAGAS)"): bi_cell(c['reason'], 22)}
                 for i, (n, m, c) in enumerate(zip(demo['line_ids'], _mine, _run['chunks']))]
        _mixed = any(is_english(c['reason']) for c in _run['chunks']) and t.lang == 'de'
        _cp_prompt, _ = ev.judge_prompt_text()
        _view = mo.vstack([
            mo.hstack([mo.stat(f'{_agree} / 5', label=t('Übereinstimmung mit dem Richter', 'Agreement with the judge'), bordered=True)],
                      justify='start'),
            mo.md(left(md_table(_rows))),
            *([hint(t('Der Richter begründet auf Englisch, weil die Prompts von RAGAS englisch sind: eine Schwäche von RAGAS bei deutschen Texten.',
                      'The judge gives its reasons in English because the RAGAS prompts are in English: a weakness of RAGAS with German texts.'))]
              if _mixed else []),
            frame(t('Hinweis', 'Note'), t(
                f'Der Richter (hier {ev.JUDGE_MODEL}) sieht die Quelle nicht. Er urteilt nur über den Inhalt von Chunk und Referenzantwort; welche Stelle auf der Webseite gemeint ist, weiß er nicht.',
                f'The judge (here {ev.JUDGE_MODEL}) does not see the source. It judges only the content of chunk and reference answer; it does not know which passage of the website is meant.')),
            mo.accordion({
                t('Begründungen des Richters ganz', "The judge's reasons in full"): mo.md('\n\n'.join(
                    f"**{t('Rang', 'Rank')} {i + 1}** ({yes_no(c['verdict'])}). {bi_cell(c['reason'])}"
                    for i, c in enumerate(_run['chunks']))),
                t('Was der Richter genau gefragt wird', 'What exactly the judge is asked'): mo.md(t(
                    f'RAGAS schickt für jeden Chunk einen eigenen Aufruf mit Frage, Referenzantwort und Chunk. Die Anweisung lautet:\n\n> <span class="hpi-corpus">{_cp_prompt}</span>',
                    f'RAGAS sends a separate call for every chunk with question, reference answer and chunk. The instruction reads:\n\n> {_cp_prompt}',
                )),
            }),
        ], gap=1)
    _view
    return


@app.cell(hide_code=True)
def _(plt, t, theme):
    # Figure of section C: the example of five ranks, useful ones dark
    def example_fig():
        _useful = [1, 0, 1, 0, 0]
        _fig, _ax = plt.subplots(figsize=(6.5, 1.25))
        for _i, _v in enumerate(_useful):
            _ax.add_patch(plt.Rectangle((_i * 1.2, 0), 1, 1, facecolor=theme.INK if _v else 'white',
                                        edgecolor=theme.INK, linewidth=0.8))
            _ax.text(_i * 1.2 + 0.5, 0.5, t('nützlich', 'useful') if _v else t('nicht\nnützlich', 'not\nuseful'),
                     ha='center', va='center', fontsize=8.5, color='white' if _v else theme.INK)
            _ax.text(_i * 1.2 + 0.5, -0.22, f"{t('Rang', 'Rank')} {_i + 1}", ha='center', va='top', fontsize=8.5, color=theme.GREY)
        _ax.set_xlim(-0.1, 5.9)
        _ax.set_ylim(-0.6, 1.05)
        _ax.set_aspect('equal')
        _ax.axis('off')
        return _fig
    return (example_fig,)


@app.cell(hide_code=True)
def _(example_fig, hint, key, mo, t):
    # Section C (row 7, section 3f): Context Precision with the formula, the example and its computation
    _formula = t(
        r'$$\text{Context Precision} = \frac{\text{Summe von Precision@}k\text{ über alle Ränge }k\text{ mit nützlichem Chunk}}{\text{Anzahl nützlicher Chunks}}$$'
        '\n\n'
        r'$$\text{Precision@}k = \frac{\text{nützliche Chunks bis Rang }k}{k}$$',
        r'$$\text{Context Precision} = \frac{\text{sum of Precision@}k\text{ over all ranks }k\text{ with a useful chunk}}{\text{number of useful chunks}}$$'
        '\n\n'
        r'$$\text{Precision@}k = \frac{\text{useful chunks up to rank }k}{k}$$')
    mo.vstack([
        mo.md(t('## C. Context Precision: Stehen die nützlichen Chunks oben?', '## C. Context Precision: are the useful chunks at the top?')),
        key(t('Context Precision ist hoch, wenn die nützlichen Chunks weit oben stehen.',
              'Context Precision is high when the useful chunks stand near the top.')),
        hint(t('An jeder Stelle mit einem nützlichen Chunk zählt der Anteil nützlicher Chunks bis dorthin. Der Durchschnitt dieser Anteile ist Context Precision.',
               'At every position with a useful chunk, the share of useful chunks up to there counts. The average of these shares is Context Precision.')),
        mo.md(_formula),
        example_fig(),
        mo.md(t('**Rang 1:** Precision@1 = 1 von 1 = 1,00. **Rang 3:** Precision@3 = 2 von 3 = 0,67. **Context Precision** = (1,00 + 0,67) / 2 = **0,83**.',
                '**Rank 1:** Precision@1 = 1 of 1 = 1.00. **Rank 3:** Precision@3 = 2 of 3 = 0.67. **Context Precision** = (1.00 + 0.67) / 2 = **0.83**.')),
        hint(t('Nicht nützliche Chunks unten in der Liste ändern den Wert nicht.',
               'Chunks that are not useful at the bottom of the list do not change the value.')),
        mo.accordion({t('Average Precision und MAP', 'Average Precision and MAP'): mo.md(t(
            'Der Wert für eine Frage heißt in der Fachliteratur Average Precision. Der Mittelwert über alle Fragen heißt Mean Average Precision. RAGAS nennt beides Context Precision.',
            'In the literature, the value for one question is called Average Precision. The mean over all questions is called Mean Average Precision. RAGAS calls both Context Precision.',
        ))}),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(confirm, mo, t):
    # Section C, own computation: quiz, input field and the green "Prüfen" button (3e). The quiz keeps its immediate feedback.
    ui_quiz8 = mo.ui.radio({t('0,20', '0.20'): 'a', t('0,50', '0.50'): 'b', t('1,00', '1.00'): 'c'}, label=t(
        'Rang 1 nicht nützlich, Rang 2 nützlich, Rang 3 bis 5 nicht nützlich. Wie hoch ist Context Precision?',
        'Rank 1 not useful, rank 2 useful, ranks 3 to 5 not useful. What is Context Precision?'))
    ui_cp_in = mo.ui.text(placeholder=t('0,00', '0.00'), label='Context Precision')
    ui_cp_go = confirm(t('Prüfen', 'Check'), ui_cp_in)
    return ui_cp_go, ui_cp_in, ui_quiz8


@app.cell(hide_code=True)
def _(mo, quiz_feedback, quiz_task, t, ui_quiz8):
    # Row 8: the quiz as its own counted step (CR-19)
    mo.vstack([
        quiz_task('quiz_cp'),
        ui_quiz8,
        quiz_feedback(ui_quiz8.value, 'b', t(
            'Nur Rang 2 ist nützlich, dort sind 1 von 2 Chunks nützlich: 0,50. Die Ränge 3 bis 5 zählen nicht mit.',
            'Only rank 2 is useful, and there 1 of 2 chunks are useful: 0.50. Ranks 3 to 5 do not count.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(ap_steps, check_answer, confirmed, cta, demo, ev, mo, strip_html, t, task, ui_cp_go, ui_cp_in):
    # Row 9: Context Precision by hand (G3); the judge's verdicts as a strip right above the field
    _verdicts = [c['verdict'] for c in demo['run']['chunks']]
    _target = round(ev.average_precision(_verdicts), 2)
    mo.vstack([v for v in (
        mo.md(t('### Ihre Rechnung', '### Your computation')),
        task('cp', '', t(
            ['Rechnen Sie Context Precision für die Frage aus Abschnitt B aus. Die Urteile des Richters stehen in der Leiste.',
             'Tragen Sie den Wert ein und drücken Sie auf "Prüfen".'],
            ["Compute Context Precision for the question from section B. The judge's verdicts are in the strip.",
             'Enter the value and press "Check".']), t('etwa 3 Minuten', 'about 3 minutes')),
        strip_html([(f"{t('Rang', 'Rank')} {i + 1}", v) for i, v in enumerate(_verdicts)]),
        mo.hstack([ui_cp_in, cta(ui_cp_go)], justify='start', align='center', gap=1),
        check_answer(confirmed(ui_cp_go), _target, ap_steps(_verdicts))) if v is not None], gap=0.75)
    return


@app.cell(hide_code=True)
def _(bi_cell, chain_html, demo, hint, is_english, key, left, md_table, mo, t, yes_no):
    # Section D (row 10, section 3f): the chain first, then the claims of the KISZ reference
    _claims = demo['run']['claims']
    _nw = '<span style="white-space:nowrap">{}</span>'.format
    # Section 3g: the judge's reasons in full; long ones wrap (theme.css), nothing is cut off
    _rows = [{'#': str(_n + 1), t('Aussage (RAGAS)', 'Claim (RAGAS)'): bi_cell(c['statement']), _nw(t('belegt', 'supported')): _nw(yes_no(c['attributed'])),
              t('Begründung des Richters', "Judge's reason"): bi_cell(c['reason'])} for _n, c in enumerate(_claims)]
    _formula = t(r'$$\text{Context Recall} = \frac{\text{belegte Aussagen der Referenzantwort}}{\text{alle Aussagen der Referenzantwort}}$$',
                 r'$$\text{Context Recall} = \frac{\text{supported claims of the reference answer}}{\text{all claims of the reference answer}}$$')
    _mixed = any(is_english(c['reason']) for c in _claims) and t.lang == 'de'
    mo.vstack([
        mo.md(t('## D. Context Recall: Fehlt etwas?', '## D. Context Recall: is anything missing?')),
        key(t('Context Recall ist der Anteil der Aussagen aus der Referenzantwort, die in den gefundenen Chunks stehen.',
              'Context Recall is the share of claims from the reference answer that are in the retrieved chunks.')),
        chain_html(t(['Referenzantwort', 'Richter zerlegt sie in Aussagen', 'je Aussage: steht sie in den Chunks?',
                      'Anteil = Context Recall'],
                     ['reference answer', 'judge splits it into claims', 'per claim: is it in the chunks?',
                      'share = Context Recall']), judge_index=1),
        hint(t('Gemeint ist jeweils eine einzelne, für sich prüfbare Aussage; RAGAS nennt sie Claim.',
               'Each claim is a single statement that can be checked on its own; RAGAS uses the same word.')),
        mo.md(_formula),
        mo.md(t(f'**Die {len(_claims)} Aussagen der Referenzantwort aus Abschnitt B**',
                f'**The {len(_claims)} claims of the reference answer from section B**')),
        mo.md(left(md_table(_rows))),
        *([hint(t('Die Begründungen sind englisch, weil die Prompts von RAGAS englisch sind: eine Schwäche von RAGAS bei deutschen Texten.',
                  'The reasons are in English because the RAGAS prompts are in English: a weakness of RAGAS with German texts.'))]
          if _mixed else []),
        mo.accordion({t('Precision und Recall zusammen', 'Precision and recall together'): mo.md(t(
            '| | Precision hoch | Precision niedrig |\n|---|---|---|\n'
            '| **Recall hoch** | Ideal: Die nützlichen Chunks stehen oben, und alles ist abgedeckt. | Alles ist abgedeckt, aber dafür sind viele Chunks nötig. |\n'
            '| **Recall niedrig** | Wenige, gute Chunks, aber Information fehlt. | Die Suche ist insgesamt schlecht. |',
            '| | Precision high | Precision low |\n|---|---|---|\n'
            '| **Recall high** | Ideal: the useful chunks are at the top, and everything is covered. | Everything is covered, but it takes many chunks. |\n'
            '| **Recall low** | Few, good chunks, but information is missing. | The search is poor overall. |',
        ))}),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(confirm, mo, t):
    # Section D, own computation: quiz, input field and the green "Prüfen" button (both versions, sections 3a, 3e)
    ui_quiz11 = mo.ui.radio({t('0,25', '0.25'): 'a', t('0,75', '0.75'): 'b', '3': 'c'}, label=t(
        'Die Referenzantwort hat 4 Aussagen, 3 davon stehen in den Chunks. Wie hoch ist Context Recall?',
        'The reference answer has 4 claims, and 3 of them are in the chunks. What is Context Recall?'))
    ui_cr_in = mo.ui.text(placeholder=t('0,00', '0.00'), label='Context Recall')
    ui_cr_go = confirm(t('Prüfen', 'Check'), ui_cr_in)
    return ui_cr_go, ui_cr_in, ui_quiz11


@app.cell(hide_code=True)
def _(mo, quiz_feedback, quiz_task, t, ui_quiz11):
    # Row 11, quiz as its own counted step (CR-19)
    mo.vstack([
        quiz_task('quiz_cr'),
        ui_quiz11,
        quiz_feedback(ui_quiz11.value, 'b', t('Belegte Aussagen geteilt durch alle Aussagen: 3 / 4 = 0,75.',
                                              'Supported claims divided by all claims: 3 / 4 = 0.75.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(check_answer, confirmed, cta, demo, mo, strip_html, t, task, ui_cr_go, ui_cr_in):
    # Row 11: Context Recall by hand (G3)
    _claims = demo['run']['claims']
    _n, _k = len(_claims), sum(c['attributed'] for c in _claims)
    _target = round(_k / _n, 2)
    mo.vstack([v for v in (
        mo.md(t('### Jetzt rechnen Sie selbst', '### Now you compute')),
        task('cr', '', t(
            ['Rechnen Sie Context Recall für die Frage aus Abschnitt B aus. Die Urteile zu den Aussagen stehen in der Leiste und in der Tabelle oben.',
             'Tragen Sie den Wert ein und drücken Sie auf "Prüfen".'],
            ['Compute Context Recall for the question from section B. The verdicts on the claims are in the strip and in the table above.',
             'Enter the value and press "Check".']), t('etwa 2 Minuten', 'about 2 minutes')),
        strip_html([(f"{t('Aussage', 'Claim')} {i + 1}", c['attributed']) for i, c in enumerate(_claims)]),
        mo.hstack([ui_cr_in, cta(ui_cr_go)], justify='start', align='center', gap=1),
        check_answer(confirmed(ui_cr_go), _target, t(
            f'Der Richter hat {_n} Aussagen gebildet, {_k} davon sind belegt. Belegte Aussagen geteilt durch alle Aussagen: {_k} / {_n} = {t.num(_target)}.',
            f'The judge formed {_n} claims, {_k} of them are supported. Supported claims divided by all claims: {_k} / {_n} = {t.num(_target)}.'))) if v is not None],
        gap=0.75)
    return


@app.cell(hide_code=True)
def _(demo, mo, t):
    # Section E, part a (row 11b, sections 3c and 3f): both claims of run 1 checked against the five lines.
    # The resolution below is written for the cached question "Was passiert in einem Pilotprojekt?" (lines 31, 37, 36,
    # 72, 30; claim 1 not supported, claim 2 supported); FITS guards it against a rebuilt cache.
    FITS = (demo['line_ids'] == [31, 37, 36, 72, 30] and [c['attributed'] for c in demo['run']['claims']] == [0, 1])
    ui_check = mo.ui.array([
        mo.ui.radio({t('steht drin', 'is in there'): 'in', t('steht nur teilweise drin', 'is only partly in there'): 'part',
                     t('steht nicht drin', 'is not in there'): 'out'}, inline=True)
        for _ in demo['run']['claims']])
    return FITS, ui_check


@app.cell(hide_code=True)
def _(bi, demo, key, mo, t, task, ui_check):
    # Section E, part a: the two claims left, the five lines right (visible context, section 10.3 of the meta plan)
    _claims = demo['run']['claims']
    mo.vstack([
        mo.md(t('## E. Wie verlässlich ist das Urteil?', '## E. How reliable is the verdict?')),
        key(t('Ein Urteil des Richters kann falsch sein, hängt vom Richter ab und schwankt von Lauf zu Lauf.',
              'A verdict of the judge can be wrong, depends on the judge and varies from run to run.')),
        mo.md(t('### a. Stimmt das Urteil?', '### a. Is the verdict right?')),
        task('check', t('Zwei Urteile nachprüfen', 'Check two verdicts'), t(
            ['Lesen Sie links die beiden Aussagen und das Urteil des Richters.',
             'Suchen Sie sie rechts in den fünf gefundenen Zeilen.',
             'Wählen Sie bei jeder Aussage, ob sie dort steht.'],
            ["Read the two claims and the judge's verdict on the left.",
             'Look for them in the five retrieved lines on the right.',
             'Choose for each claim whether it is in there.']), t('etwa 3 Minuten', 'about 3 minutes')),
        mo.hstack([
            mo.vstack([
                mo.vstack([
                    mo.md(t(f'#### Aussage {i + 1}', f'#### Claim {i + 1}')),
                    mo.md(t(f"*vom Richter als {'belegt' if c['attributed'] else 'nicht belegt'} markiert*",
                            f"*marked by the judge as {'supported' if c['attributed'] else 'not supported'}*")),
                    bi(c['statement']), ui_check[i]], gap=0.4)
                for i, c in enumerate(_claims)], gap=1.5),
            mo.vstack([mo.vstack([mo.md(f"**{t('Rang', 'Rank')} {i + 1}** &nbsp; *{t('Zeile', 'line')} {n}*"), bi(c)], gap=0.25)
                       for i, (n, c) in enumerate(zip(demo['line_ids'], demo['chunks']))], gap=1),
        ], widths=[2, 3], gap=2, align='start'),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(FITS, feedback, frame, left, md_table, mo, nw, t, ui_check):
    # Section E, part a: resolution as soon as both radios are answered; facts checked against the cache on 2026-10-19.
    # Feedback box (3g): right when claim 1 is "partly" and claim 2 "not in there", as the solution says.
    mo.stop(not FITS or any(v is None for v in ui_check.value))
    _right = ['part', 'out']
    _label = {'in': t('steht drin', 'is in there'), 'part': t('teilweise', 'partly'),
              'out': t('steht nicht drin', 'is not in there')}

    def _mine(i):
        _v = ui_check.value[i]
        return nw(f"{_label[_v]} {'&#10003;' if _v == _right[i] else '&#10007;'}")

    _rows = [
        {'#': '1', t('Richter', 'Judge'): nw(t('nicht belegt', 'not supported')), nw(t('Ihre Wahl', 'Your choice')): _mine(0),
         t('Auflösung', 'Solution'): t(
             'Teilweise. Bewerben (Zeile 36) und Prototyp (Zeile 31) stehen drin, die 2 bis 3 Tage Begleitung pro Woche nicht. Weil ein Teil fehlt, zählt die ganze Aussage als nicht belegt.',
             'Partly. Applying (line 36) and prototype (line 31) are in there, the 2 to 3 days of support per week are not. Because one part is missing, the whole claim counts as not supported.')},
        {'#': '2', t('Richter', 'Judge'): nw(t('belegt', 'supported')), nw(t('Ihre Wahl', 'Your choice')): _mine(1),
         t('Auflösung', 'Solution'): t(
             'Eher nicht. Zeile 72 gehört zum Angebot Infrastruktur, nicht zu den Pilotprojekten. Die passende Zeile 35 hat die Suche nicht gefunden.',
             'Rather not. Line 72 belongs to the infrastructure offer, not to the pilot projects. The search did not find the matching line 35.')},
    ]
    mo.vstack([
        feedback(list(ui_check.value) == _right, left(md_table(_rows))),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'Ein Richter kann zu großzügig sein. Stichproben gegen die Quelle, so wie Sie es eben gemacht haben, gehören deshalb zu jeder Evaluation.',
            'A judge can be too generous. That is why spot checks against the source, as you just did, belong in every evaluation.')),
    ], gap=0.75)
    return


@app.cell(hide_code=True)
def _(bi_cell, demo, ev, frame, hint, left, md_table, mo, nw, t, yes_no):
    # Section E, part b (section 3f): the same reference split into claims by two judges, side by side
    def _column(name, claims):
        _cr = ev.recall(claims)
        _rows = [{'#': str(i + 1), t('Aussage (RAGAS)', 'Claim (RAGAS)'): bi_cell(c['statement']),
                  nw(t('belegt', 'supported')): nw(yes_no(c['attributed']))} for i, c in enumerate(claims)]
        return mo.vstack([
            mo.md(f'#### {name}'),
            mo.hstack([mo.stat(str(len(claims)), label=t('Aussagen', 'claims'), bordered=True),
                       mo.stat(t.num(_cr), label='Context Recall', bordered=True)], justify='start', gap=1),
            mo.md(left(md_table(_rows))),
        ], gap=0.5)

    mo.vstack([
        mo.md(t('### b. Was zählt als Aussage?', '### b. What counts as a claim?')),
        hint(t('Dieselbe Referenzantwort, dieselben fünf Zeilen; zwei Richter zerlegen die Referenzantwort in Aussagen.',
               'The same reference answer, the same five lines; two judges split the reference answer into claims.')),
        mo.hstack([_column(ev.JUDGE_MODEL, demo['run']['claims']), _column(demo['second_judge'], demo['second'][0]['claims'])],
                  widths='equal', gap=2, align='start'),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'Schon die Zerlegung entscheidet mit: Wer mehr oder feinere Aussagen bildet, ändert den Nenner und damit den Score.',
            'The split alone already has a say: whoever forms more or finer claims changes the denominator and with it the score.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t, task):
    # Section E, part c (row 12, part 1): heading and prediction; the diagram appears as soon as a prediction is chosen
    ui_pred12 = mo.ui.radio(
        {t('immer dieselben Zahlen', 'always the same numbers'): 'same',
         t('leicht unterschiedliche Zahlen', 'slightly different numbers'): 'slight',
         t('völlig verschiedene Zahlen', 'completely different numbers'): 'wild'},
        label=t('Was erwarten Sie?', 'What do you expect?'))
    mo.vstack([
        mo.md(t('### c. Derselbe Richter, fünf Läufe', '### c. The same judge, five runs')),
        mo.md(t('Der Richter bewertet die Frage aus Abschnitt B mit denselben fünf Zeilen fünfmal.',
                'The judge scores the question from section B with the same five lines five times.')),
        task('spread', t('Vorhersage', 'Prediction'), t(
            ['Wählen Sie, was Sie erwarten, bevor Sie das Diagramm sehen.', 'Vergleichen Sie danach mit dem Diagramm.'],
            ['Choose what you expect before you see the diagram.', 'Then compare with the diagram.']),
            t('etwa 2 Minuten', 'about 2 minutes')),
        ui_pred12,
    ], gap=1)
    return (ui_pred12,)


@app.cell(hide_code=True)
def _(mo):
    # Live runs of part c: written by the work cell, rendered by the figure cell.
    get_live, set_live = mo.state({'runs': [], 'error': None})  # error: 'api' or 'key'
    get_used, set_used = mo.state(0)
    return get_live, get_used, set_live, set_used


@app.cell(hide_code=True)
def _(demo, feedback, frame, get_live, hint, mo, plt, t, theme, ui_pred12):
    # Row 12, part 2: the five cached runs (and your own ones), the resolution and the key takeaway
    mo.stop(ui_pred12.value is None)
    _runs = demo['runs']
    _own = get_live()['runs']
    _all = _runs + _own
    _x = list(range(len(_all)))
    _fig, _ax = plt.subplots(figsize=(6.5, 2.9))
    # Run 1 is the one the participant recomputed above; a light band marks it.
    _ax.axvspan(-0.4, 0.4, color=theme.RULE, alpha=0.6, zorder=0)
    _ax.annotate(t('Ihre Rechnung', 'Your computation'), (0, 1.0), xycoords=('data', 'axes fraction'), xytext=(0, -2),
                 textcoords='offset points', ha='center', va='top', fontsize=8, color=theme.GREY)
    for _i, _r in enumerate(_all):
        _new = _i >= len(_runs)
        _ax.plot(_i, _r['context_precision'], 'o', color='black' if _new else theme.GREYS[1],
                 markersize=8 if _new else 6, zorder=3)
        _ax.plot(_i, _r['context_recall'], 's', markerfacecolor='white', markeredgecolor='black' if _new else theme.GREYS[1],
                 markeredgewidth=1.6 if _new else 1.0, markersize=8 if _new else 6, zorder=3)
    # Direct labels left of run 1, so they cannot be mistaken for a new point.
    _ax.annotate('Context Precision', (0, _all[0]['context_precision']), xytext=(-10, 0),
                 textcoords='offset points', ha='right', va='center', fontsize=8.5, color=theme.INK)
    _ax.annotate('Context Recall', (0, _all[0]['context_recall']), xytext=(-10, 0),
                 textcoords='offset points', ha='right', va='center', fontsize=8.5, color=theme.INK)
    _labels = [t(f'Lauf {i + 1}', f'Run {i + 1}') for i in range(len(_runs))]
    _labels += [t(f'Neu {i + 1}', f'New {i + 1}') for i in range(len(_own))]
    _ax.set_xticks(_x, [f"{lab}\n{r['n_claims']}" for lab, r in zip(_labels, _all)], fontsize=8)
    # The second tick line is the number of claims of each run; it is named once, left of the ticks.
    _ax.annotate(t('Aussagen', 'Claims'), (0, 0), xycoords=('data', 'axes fraction'), xytext=(-24, -22),
                 textcoords='offset points', ha='right', va='center', fontsize=8, color=theme.GREY)
    for _tick in _ax.get_xticklabels()[len(_runs):]:
        _tick.set_fontweight('bold')
    _ax.set_xlim(-2.2, max(len(_all), 5) - 0.4)
    _ax.set_ylim(-0.05, 1.05)
    _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: t.num(v, 1)))
    _ax.set_ylabel(t('Wert', 'Value'))
    _fig.tight_layout()

    def _span(metric, name, inverted=False):
        """The range of one metric over the five runs; inverted puts the German verb first (after 'Bei dieser Frage')."""
        _v = [r[metric] for r in _runs]
        _same = round(min(_v), 2) == round(max(_v), 2)
        if t.lang == 'en':
            return (f'{name} stays at {t.num(_v[0])} in all five runs' if _same
                    else f'{name} lies between {t.num(min(_v))} and {t.num(max(_v))}')
        _verb, _rest = (('bleibt', f'in allen fünf Läufen bei {t.num(_v[0])}') if _same
                        else ('liegt', f'zwischen {t.num(min(_v))} und {t.num(max(_v))}'))
        return f'{_verb} {name} {_rest}' if inverted else f'{name} {_verb} {_rest}'

    _width = max(round(max(r[m] for r in _runs) - min(r[m] for r in _runs), 2)
                 for m in ('context_precision', 'context_recall'))
    _varies = _width > 0
    # Which prediction the five cached runs confirm (3g): no change, a change of at most half the scale, or more
    _actual = 'same' if not _varies else ('slight' if _width <= 0.5 else 'wild')
    _resolve = t(
        f"Ihre Rechnung oben war Lauf 1. Bei dieser Frage {_span('context_precision', 'Context Precision', inverted=True)}, und {_span('context_recall', 'Context Recall')}."
        + (' Immer dieselben Zahlen kamen also nicht heraus.' if _varies and ui_pred12.value == 'same' else ''),
        f"Your computation above was run 1. For this question, {_span('context_precision', 'Context Precision')}, and {_span('context_recall', 'Context Recall')}."
        + (' So the numbers were not always the same.' if _varies and ui_pred12.value == 'same' else ''))
    # What varies over the five runs: the verdicts on single ranks, the claims, or both.
    _ranks = [i for i in range(len(demo['chunks'])) if len({r['chunks'][i]['verdict'] for r in _runs}) > 1]
    _claims_same = len({tuple((c['statement'], c['attributed']) for c in r['claims']) for r in _runs}) == 1
    if _claims_same and len(_ranks) == 1:
        _r, _n = _ranks[0] + 1, demo['line_ids'][_ranks[0]]
        _why = t(f'Die Aussagen und ihre Urteile bleiben hier gleich. Es schwankt das Urteil zu Rang {_r} (Zeile {_n}): mal nützlich, mal nicht, weil ein Sprachmodell nicht jedes Mal gleich antwortet.',
                 f'The claims and their verdicts stay the same here. What varies is the verdict on rank {_r} (line {_n}): sometimes useful, sometimes not, because a language model does not answer the same way every time.')
    else:
        _why = t('Ein Sprachmodell antwortet nicht jedes Mal gleich: Es zerlegt die Referenzantwort nicht immer gleich, und es urteilt nicht immer gleich über Chunks und Aussagen.',
                 'A language model does not answer the same way every time: it does not always split the reference answer the same way, and it does not always judge chunks and claims the same way.')
    mo.vstack([
        _fig,
        hint(t('Punkte: Context Precision. Quadrate: Context Recall. Unter jedem Lauf die Zahl der Aussagen.',
               'Dots: Context Precision. Squares: Context Recall. Under each run the number of claims.')),
        feedback(ui_pred12.value == _actual, _resolve),
        hint(_why),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'Ein einzelner Wert ist eine Stichprobe. Belastbar sind Mittelwerte über viele Fragen, gemessen mit demselben Richter.',
            'A single value is a sample. Means over many questions, measured with the same judge, are what you can rely on.')),
    ], gap=0.75)
    return


@app.cell(hide_code=True)
def _(LIMIT, cta, get_used, hint, mo, t, theme, ui_pred12):
    # Row 12, part 3 (normal only): ask the judge again, live; prominent button, the limit as a hint (section 10.4)
    _spent = get_used() >= LIMIT
    ui_rerun = mo.ui.run_button(
        label=theme.cta_label(t('Alle drei Läufe genutzt', 'All three runs used') if _spent
                              else t('Richter noch einmal fragen', 'Ask the judge again')),
        disabled=_spent, kind='success')
    mo.stop(t.short or ui_pred12.value is None)
    mo.vstack([
        cta(ui_rerun),
        hint(t('Sie können den Richter bis zu dreimal neu urteilen lassen. Vergleichen Sie Ihre neuen Punkte mit denen Ihres Nachbarn.',
               "You can have the judge judge again up to three times. Compare your new points with your neighbor's.")),
    ], gap=0.25)
    return (ui_rerun,)


@app.cell(hide_code=True)
async def _(LIMIT, demo, ev, get_used, mo, os, set_live, set_used, t, ui_rerun):
    mo.stop(not ui_rerun.value or get_used() >= LIMIT)
    if not (os.getenv('OPENAI_API_KEY') or '').strip() or os.getenv('OPENAI_API_KEY', '').startswith('your_'):
        set_live(lambda s: {**s, 'error': 'key'})
        mo.stop(True)
    try:
        with mo.status.spinner(title=t('Der Richter urteilt ...', 'The judge is judging ...')):
            _run = await ev.judge_once(demo['question'], demo['reference'], demo['chunks'])
    except Exception:  # noqa: BLE001 - any API failure gets the same sentence
        set_live(lambda s: {**s, 'error': 'api'})
    else:
        set_live(lambda s: {'runs': s['runs'] + [_run], 'error': None})
        set_used(lambda n: n + 1)
    return


@app.cell(hide_code=True)
def _(JUDGE_DOWN, get_live, mo, t):
    _err = get_live()['error']
    mo.stop(t.short or _err is None)
    mo.callout(mo.md(t(*JUDGE_DOWN) if _err == 'api' else t(
        'Für diesen Knopf brauchen Sie einen API-Key in `notebooks/.env`.',
        'For this button you need an API key in `notebooks/.env`.')), kind='warn')
    return


@app.cell(hide_code=True)
def _(CFG, frame, hint, mo, t):
    # Section E, part d (row 15b, sections 3c and 3f): another judge, another number, on the gold data set
    _cr = {n: sum(CFG[n]['topk']['5']['context_recall']) / 43 for n in ('standard', 'llama')}
    mo.vstack([
        mo.md(t('### d. Anderer Richter, andere Zahl', '### d. Another judge, another number')),
        hint(t('Hier wechseln wir zum Gold-Datensatz: 43 Fragen von Fachleuten zum IT-Grundschutz-Kompendium.',
               'Here we switch to the gold data set: 43 questions by experts about the IT-Grundschutz-Kompendium.')),
        mo.hstack([
            mo.stat(t.num(_cr['standard']), label=t('Richter gpt-oss-120b', 'Judge gpt-oss-120b'),
                    caption=t('Context Recall, TOP_K 5, Mittelwert über 43 Fragen', 'Context Recall, TOP_K 5, mean over 43 questions'), bordered=True),
            mo.stat(t.num(_cr['llama']), label=t('Richter llama-3-3-70b', 'Judge llama-3-3-70b'),
                    caption=t('dieselben Chunks', 'the same chunks'), bordered=True),
        ], justify='start', gap=2),
        hint(t('Ein anderer Richter ändert die Messung, nicht das System. Vergleichen Sie Zahlen nur, wenn derselbe Richter gemessen hat.',
               'Another judge changes the measurement, not the system. Compare numbers only when the same judge measured them.')),
        frame(t('Kernaussage', 'Key takeaway'), t('Zu jeder Zahl gehört der Name des Richters, der sie gemessen hat.',
                                                  'Every number comes with the name of the judge that measured it.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(key, mo, t, task):
    # Section F (former G and H, section 3f): from one question to all 43, prediction first
    ui_pred14 = mo.ui.radio(
        {t('beide steigen', 'both rise'): 'both', t('Recall steigt, Precision fällt irgendwann', 'recall rises, precision falls at some point'): 'trade',
         t('beide fallen', 'both fall'): 'fall'},
        label=t('Was passiert, wenn TOP_K von 1 auf 10 steigt?', 'What happens when TOP_K rises from 1 to 10?'))
    ui_topk = mo.ui.radio({'1': 1, '3': 3, '5': 5, '10': 10}, value='5', inline=True, label='TOP_K')
    mo.vstack([
        mo.md(t('## F. Wie viele Chunks?', '## F. How many chunks?')),
        key(t('Bisher eine Frage, ab jetzt alle 43 des Gold-Datensatzes: Was ändert sich, wenn die Suche mehr Chunks liefert?',
              'So far one question, from now on all 43 of the gold data set: what changes when the search returns more chunks?')),
        task('topk_predict', t('Vorhersage', 'Prediction'), t(
            ['Wählen Sie Ihre Vorhersage.', 'Schalten Sie danach TOP_K durch und vergleichen Sie mit Ihrer Vorhersage.'],
            ['Choose your prediction.', 'Then switch through TOP_K and compare with your prediction.']),
            t('etwa 3 Minuten', 'about 3 minutes')),
        ui_pred14,
    ], gap=1)
    return ui_pred14, ui_topk


@app.cell(hide_code=True)
def _(CFG, ev, feedback, frame, hint, mo, plt, t, theme, ui_pred14, ui_topk):
    # Row 14: TOP_K choice, the two numbers and the curve of both means with the sweet spot, once a prediction is chosen
    mo.stop(ui_pred14.value is None)
    _ks = ev.TOP_K_VALUES
    _cp = [sum(CFG['standard']['topk'][str(k)]['context_precision']) / 43 for k in _ks]
    _cr = [sum(CFG['standard']['topk'][str(k)]['context_recall']) / 43 for k in _ks]
    _sel = _ks.index(ui_topk.value)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.0))
    _ax.axvspan(3, 5, color=theme.GREYS[4], alpha=0.55, zorder=0, linewidth=0)
    _ax.text(4, 0.98, 'Sweet Spot', ha='center', va='top', fontsize=8.5, color=theme.GREY)
    _ax.plot(_ks, _cp, '-o', color=theme.GREYS[0])
    _ax.plot(_ks, _cr, '--s', color=theme.GREYS[2], markerfacecolor='white')
    _ax.axvline(ui_topk.value, color=theme.GREYS[3], linewidth=1.2, linestyle=':', zorder=1)
    _ax.plot([ui_topk.value] * 2, [_cp[_sel], _cr[_sel]], 'o', color='black', markersize=7, zorder=4)
    _ax.annotate('Context Precision', (_ks[-1], _cp[-1]), xytext=(8, 0), textcoords='offset points', va='center', fontsize=8.5)
    _ax.annotate('Context Recall', (_ks[-1], _cr[-1]), xytext=(8, 0), textcoords='offset points', va='center', fontsize=8.5)
    _ax.set_xticks(_ks)
    _ax.set_xlim(0, 13.5)
    _ax.set_ylim(0, 1.0)
    _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: t.num(v, 1)))
    _ax.set_xlabel('TOP_K')
    _ax.set_ylabel(t('Mittelwert über 43 Fragen', 'Mean over 43 questions'))
    _fig.tight_layout()
    # Which prediction the curves confirm (3g), read from the cache: recall rises, precision peaks and falls after it
    _peak = _ks[_cp.index(max(_cp))]
    _actual = ('trade' if _cr[-1] > _cr[0] and _cp[-1] < max(_cp)
               else 'both' if _cr[-1] > _cr[0] else 'fall')
    _explain = t(
        f'Recall steigt von {t.num(_cr[0])} bei TOP_K 1 auf {t.num(_cr[-1])} bei TOP_K 10. Precision steigt bis TOP_K {_peak} und fällt danach auf {t.num(_cp[-1])}.',
        f'Recall rises from {t.num(_cr[0])} at TOP_K 1 to {t.num(_cr[-1])} at TOP_K 10. Precision rises up to TOP_K {_peak} and then falls to {t.num(_cp[-1])}.')
    mo.vstack([
        mo.hstack([ui_topk,
                   mo.stat(t.num(_cp[_sel]), label='Context Precision', bordered=True),
                   mo.stat(t.num(_cr[_sel]), label='Context Recall', bordered=True)], justify='start', align='center', gap=2),
        _fig,
        hint(t('Abbildung 1. Mittelwerte über 43 Fragen des Gold-Datensatzes, Standard-Konfiguration, Richter gpt-oss-120b. Grau hinterlegt: der Sweet Spot.',
               'Figure 1. Means over the 43 questions of the gold data set, standard configuration, judge gpt-oss-120b. Grey band: the sweet spot.')),
        feedback(ui_pred14.value == _actual, _explain),
        *([hint(t('Vor der Pause: Wer hat "Recall steigt, Precision fällt" getippt?',
                  'Before the break: who predicted "recall rises, precision falls"?'))] if t.short else []),
        frame('Sweet Spot', t(
            'Bis TOP_K 3 bis 5 bringt jeder Chunk deutlich mehr Recall; danach steigt Recall kaum noch, Precision fällt, und jeder Chunk kostet Platz im Prompt.',
            'Up to TOP_K 3 to 5 every chunk brings clearly more recall; after that recall hardly rises, precision falls, and every chunk costs space in the prompt.')),
    ], gap=0.75)
    return


@app.cell(hide_code=True)
def _(CFG, card, cards, mo, t, ui_pred14):
    # Section F, explanation (former H, row 15a, section 3c) as cards; normal only (C25: short keeps the key sentence).
    # The rise of precision from TOP_K 1 to 3 comes mostly from questions where rank 1 scored 0 at TOP_K 1 and the whole
    # list scored 1 at TOP_K 3. The lists of one question are prefixes of the same search (checked against Qdrant on
    # 2026-10-16), so the judge accepted the same rank-1 chunk the second time.
    mo.stop(t.short or ui_pred14.value is None)
    _k1 = CFG['standard']['topk']['1']['context_precision']
    _k3 = CFG['standard']['topk']['3']['context_precision']
    _rise = (sum(_k3) - sum(_k1)) / 43
    _flips = [i for i in range(43) if _k1[i] < 0.5 and _k3[i] > 0.99]
    _flip_part = sum(_k3[i] - _k1[i] for i in _flips) / 43
    _nums = t(' und ', ' and ').join(str(i + 1) for i in _flips)
    mo.vstack([
        mo.md(t('### Warum die Kurven so verlaufen', '### Why the curves run this way')),
        cards([
            card(t('Precision von 1 auf 3', 'Precision from 1 to 3'),
                 t(f'Steigt vor allem durch Streuung: Bei den Fragen {_nums} hat der Richter denselben Chunk auf Rang 1 einmal verworfen und einmal als nützlich gewertet.',
                   f'Rises mostly through variance: for questions {_nums} the judge once rejected and once accepted the same chunk at rank 1.'),
                 t(f'{t.num(_flip_part, 3)} des Anstiegs von {t.num(_rise, 3)}', f'{t.num(_flip_part, 3)} of the rise of {t.num(_rise, 3)}')),
            card(t('Precision ab 3', 'Precision from 3 on'),
                 t('Fällt, weil weiter unten noch nützliche Chunks auftauchen und dort der Anteil nützlicher Chunks kleiner ist. Die Suche wird dadurch nicht schlechter.',
                   'Falls because useful chunks still turn up further down, where the share of useful chunks is smaller. That does not make the search worse.')),
            card('Recall', t('Steigt, weil mehr Chunks mehr Stellen der Referenzantwort abdecken.',
                             'Rises because more chunks cover more parts of the reference answer.')),
            card(t('Kosten', 'Cost'), t('Jeder Chunk kostet später Platz im Prompt, Zeit und Geld; Text, der nicht zur Frage passt, kann das Sprachmodell ablenken.',
                                        'Every chunk later costs space in the prompt, time and money; text that does not fit the question can distract the language model.')),
        ]),
        mo.accordion({t('Warum kann Precision von TOP_K 1 auf 3 überhaupt steigen?', 'Why can precision rise from TOP_K 1 to 3 at all?'): mo.md(t(
            'Bei TOP_K 1 bekommt eine Frage nur 0 oder 1; ist Rang 1 nicht nützlich, holt ein nützlicher Chunk auf Rang 2 oder 3 sie auf etwa 0,5. Außerdem hat der Richter für jeden TOP_K-Wert neu geurteilt, und er urteilt nicht immer gleich.',
            'At TOP_K 1 a question gets only 0 or 1; if rank 1 is not useful, a useful chunk at rank 2 or 3 brings it to about 0.5. Besides, the judge judged anew for every TOP_K value, and it does not always judge the same way.'))}),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(confirm, mo, t):
    # Section G (rows 16 and 17, normal only): quiz, choice and reason, confirmed with the green button (3e)
    ui_quiz16 = mo.ui.radio(
        {t('weil Recall bei 10 sinkt', 'because recall drops at 10'): 'a',
         t('weil mehr Chunks den Prompt verlängern und die Antwort verschlechtern können',
           'because more chunks lengthen the prompt and can make the answer worse'): 'b',
         t('weil Qdrant nicht mehr als 5 liefert', 'because Qdrant returns no more than 5'): 'c'},
        label=t('Warum nimmt man nicht einfach immer TOP_K 10?', 'Why not simply always take TOP_K 10?'))
    ui_choice = mo.ui.radio({'1': 1, '3': 3, '5': 5, '10': 10}, inline=True, label=t('Ihr TOP_K', 'Your TOP_K'))
    ui_reason = mo.ui.text_area(label=t('Ihre Begründung', 'Your reason'), rows=2, full_width=True)
    ui_choice_go = confirm(t('Wahl bestätigen', 'Confirm choice'), lambda: (ui_choice.value, ui_reason.value))
    return ui_choice, ui_choice_go, ui_quiz16, ui_reason


@app.cell(hide_code=True)
def _(key, mo, quiz_feedback, quiz_task, t, ui_quiz16):
    # Section G, the quiz as its own counted step before the choice (CR-19)
    t.full(mo.vstack([
        mo.md(t('## G. Ihr TOP_K', '## G. Your TOP_K')),
        key(t('TOP_K ist eine Abwägung zwischen Precision und Recall.', 'TOP_K is a trade-off between precision and recall.')),
        quiz_task('quiz_topk'),
        ui_quiz16,
        quiz_feedback(ui_quiz16.value, 'b', t(
            'Recall steigt mit TOP_K. Der Preis kommt bei der Antwort: längere Prompts, mehr Kosten, und mehr Text, der nicht zur Frage passt.',
            'Recall rises with TOP_K. The price comes with the answer: longer prompts, more cost, and more text that does not fit the question.')),
    ], gap=1))
    return


@app.cell(hide_code=True)
def _(CFG, cta, ev, left, md_table, mo, t, task, ui_choice, ui_choice_go, ui_reason):
    # Section G: the task with both curves as a small table next to it (visible context)
    _means = [{'TOP_K': str(k),
               'Context Precision': t.num(sum(CFG['standard']['topk'][str(k)]['context_precision']) / 43),
               'Context Recall': t.num(sum(CFG['standard']['topk'][str(k)]['context_recall']) / 43)} for k in ev.TOP_K_VALUES]
    t.full(mo.vstack([
        mo.md(t('### Ihre Wahl', '### Your choice')),
        task('topk_choose', t('TOP_K wählen', 'Choose TOP_K'), t(
            ['Wählen Sie ein TOP_K für Ihr System.',
             'Begründen Sie Ihre Wahl in einem Satz mit beiden Kurven.', 'Drücken Sie auf "Wahl bestätigen".'],
            ['Choose a TOP_K for your system.',
             'Give your reason in one sentence, using both curves.', 'Press "Confirm choice".']), t('etwa 3 Minuten', 'about 3 minutes')),
        mo.hstack([mo.vstack([ui_choice, ui_reason, cta(ui_choice_go)]), mo.md(left(md_table(_means)))],
                  widths=[3, 2], gap=2, align='start'),
    ], gap=1))
    return


@app.cell(hide_code=True)
def _(CFG, confirmed, feedback, hint, mo, t, ui_choice_go):
    # Assessment of section G, after "Wahl bestätigen" with a TOP_K and a reason; a choice without a right answer (3g)
    _given = confirmed(ui_choice_go)
    mo.stop(t.short or ui_choice_go.value is None)
    mo.stop(_given[0] is None or not _given[1].strip(), mo.callout(mo.md(t(
        'Wählen Sie zuerst ein TOP_K und schreiben Sie eine Begründung.',
        'First choose a TOP_K and write a reason.')), kind='warn'))
    _m = {k: {m: sum(v) / len(v) for m, v in CFG['standard']['topk'][str(k)].items()} for k in (3, 5, 10)}
    mo.vstack([
        feedback(None, t(
            f"Viele wählen 3 oder 5. Bei 3 ist Precision am höchsten ({t.num(_m[3]['context_precision'])}), bei 5 ist Recall höher ({t.num(_m[5]['context_recall'])} statt {t.num(_m[3]['context_recall'])}), bei 10 noch höher ({t.num(_m[10]['context_recall'])}), dafür fällt Precision auf {t.num(_m[10]['context_precision'])}. Eine einzige richtige Antwort gibt es nicht; sie hängt davon ab, ob fehlende Information oder lange Prompts in Ihrem Fall teurer sind.",
            f"Many choose 3 or 5. At 3 precision is highest ({t.num(_m[3]['context_precision'])}), at 5 recall is higher ({t.num(_m[5]['context_recall'])} instead of {t.num(_m[3]['context_recall'])}), at 10 higher still ({t.num(_m[10]['context_recall'])}), but precision falls to {t.num(_m[10]['context_precision'])}. There is no single right answer; it depends on whether missing information or long prompts cost more in your case.",
        )),
        hint(t('Vor der Pause fragen wir im Raum, wer was gewählt hat (etwa 3 Minuten).',
               'Before the break we ask the room who chose what (about 3 minutes).')),
    ], gap=0.75)
    return


@app.cell(hide_code=True)
def _(ev, hint, mo, t, theme):
    # Row 18 (normal only, section 3g): the extra task as its own section, closed at the start like every section.
    # One orange box per part: the part's name, one sentence as the task, numbered steps; the input right below,
    # then the solution. Inputs and results sit in separate cells, so working in one never redraws a box.
    mo.stop(t.short)
    CFG_LABELS = {
        'standard': t('Standard', 'Standard'),
        'minilm': t('Embedding miniLM', 'Embedding miniLM'),
        'markdown': t('Chunking markdown_headers', 'Chunking markdown_headers'),
        'llama': t('Richter llama', 'Judge llama'),
    }
    ui_cfg = mo.ui.dropdown({CFG_LABELS[n]: n for n in ev.CONFIGS}, value=CFG_LABELS['minilm'],
                            label=t('Konfiguration', 'Configuration'))
    mo.vstack([
        mo.md(t('## Zusatzaufgabe, für alle, die schon fertig sind', '## Extra task, for everyone who is already done')),
        hint(t('Alle anderen machen mit Abschnitt H weiter.', 'Everyone else continues with section H.')),
        mo.md(t('### Teil a: System oder Messung?', '### Part a: system or measurement?')),
        mo.Html(theme.task_box(
            t('Zusatzaufgabe, Teil a', 'Extra task, part a'),
            t(['Wählen Sie direkt unter diesem Kasten eine Konfiguration und vergleichen Sie ihre Kurven mit der Standard-Konfiguration.',
               'Ordnen Sie darunter jede Änderung zu: Stellschraube am System oder an der Messung?',
               'Schreiben Sie in einem Satz, was Sie Ihrer Leitung sagen, wenn jemand mit dem llama-Richter bessere Zahlen meldet.',
               'Drücken Sie auf "Prüfen".'],
              ['Right below this box, choose a configuration and compare its curves with the standard configuration.',
               'Below that, assign each change: a lever on the system or on the measurement?',
               'Write in one sentence what you would tell your management if someone reports better numbers with the llama judge.',
               'Press "Check".']),
            t('so lange, bis die anderen fertig sind', 'until the others are done'),
            t('Welche Änderung macht das System besser, welche nur die Noten?',
              'Which change makes the system better, and which only the grades?'),
            kind='extra')),
        ui_cfg,
    ], gap=1)
    return CFG_LABELS, ui_cfg


@app.cell(hide_code=True)
def _(CFG, CFG_LABELS, ev, left, md_table, mo, plt, t, theme, ui_cfg):
    # Row 18a: the chosen configuration next to the standard one, as curves and as a table
    _ks = ev.TOP_K_VALUES

    def _means(name, metric):
        return [sum(CFG[name]['topk'][str(k)][metric]) / 43 for k in _ks]

    _sel = ui_cfg.value
    _fig, _axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    for _ax, _metric in zip(_axes, ('context_precision', 'context_recall')):
        _ax.plot(_ks, _means('standard', _metric), '--o', color=theme.GREYS[2], markerfacecolor='white')
        _ax.annotate(CFG_LABELS['standard'], (_ks[-1], _means('standard', _metric)[-1]), xytext=(5, 0),
                     textcoords='offset points', va='center', fontsize=8, color=theme.GREY)
        if _sel != 'standard':
            _ax.plot(_ks, _means(_sel, _metric), '-o', color='black')
            _ax.annotate(CFG_LABELS[_sel], (_ks[-1], _means(_sel, _metric)[-1]), xytext=(5, 0),
                         textcoords='offset points', va='center', fontsize=8)
        _ax.set_title('Context Precision' if _metric == 'context_precision' else 'Context Recall')
        _ax.set_xticks(_ks)
        _ax.set_xlim(0, 17)
        _ax.set_ylim(0, 1.0)
        _ax.set_xlabel('TOP_K')
        _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: t.num(v, 1)))
    _fig.tight_layout()
    _rows = [{'TOP_K': str(k),
              f"Precision ({CFG_LABELS['standard']})": t.num(_means('standard', 'context_precision')[i]),
              f'Precision ({CFG_LABELS[_sel]})': t.num(_means(_sel, 'context_precision')[i]),
              f"Recall ({CFG_LABELS['standard']})": t.num(_means('standard', 'context_recall')[i]),
              f'Recall ({CFG_LABELS[_sel]})': t.num(_means(_sel, 'context_recall')[i])}
             for i, k in enumerate(_ks)]
    mo.vstack([_fig, mo.md(left(md_table(_rows)))])
    return


@app.cell(hide_code=True)
def _(CFG_LABELS, confirm, cta, mo, t):
    # Row 18a: the assignment and the sentence for the management, confirmed with the green button (meta plan 10.4)
    _kind = {t('System', 'system'): 'system', t('Messung', 'measurement'): 'measurement'}
    ui_kind = mo.ui.dictionary({n: mo.ui.dropdown(_kind, label=CFG_LABELS[n]) for n in ('minilm', 'markdown', 'llama')})
    ui_boss = mo.ui.text_area(label=t('Was sagen Sie Ihrer Leitung?', 'What do you tell your management?'), rows=2, full_width=True)
    ui_kind_go = confirm(t('Prüfen', 'Check'), lambda: (dict(ui_kind.value), ui_boss.value))
    mo.vstack([ui_kind, ui_boss, cta(ui_kind_go)], gap=0.75)
    return (ui_kind_go,)


@app.cell(hide_code=True)
def _(CFG, confirmed, ev, feedback, mo, t, ui_kind_go):
    # Row 18a: the solution in the feedback box; right when all three changes are assigned as below
    _given = confirmed(ui_kind_go)
    mo.stop(t.short or _given is None)
    _kinds, _boss = _given
    mo.stop(any(v is None for v in _kinds.values()) or not _boss.strip(), mo.callout(mo.md(t(
        'Ordnen Sie zuerst alle drei Änderungen zu und schreiben Sie einen Satz für Ihre Leitung.',
        'First assign all three changes and write a sentence for your management.')), kind='warn'))
    _right = {'minilm': 'system', 'markdown': 'system', 'llama': 'measurement'}
    _cr = {n: sum(CFG[n]['topk']['5']['context_recall']) / 43 for n in ('standard', 'llama')}
    feedback(_kinds == _right, t(
        f'Embedding-Modell und Chunking ändern, was gefunden wird; das sind Stellschrauben am System. Der Richter ändert nur, wie streng bewertet wird; daher kommt der Sprung von {t.num(_cr["standard"])} auf {t.num(_cr["llama"])} aus Abschnitt E, bei denselben Chunks. Zahlen sind nur unter demselben Richter vergleichbar.',
        f'Embedding model and chunking change what is found; those are levers on the system. The judge changes only how strictly it scores; that is where the jump from {t.num(_cr["standard"])} to {t.num(_cr["llama"])} in section E comes from, with the same chunks. Numbers are comparable only under the same judge.'))
    return


@app.cell(hide_code=True)
def _(CFG, TR, bi, bi_cell, ev, gold, hint, json, left, md_table, mo, pd, plt, retrieval, t, theme, yes_no):
    # Row 18b (rebuilt 2026-10-19, option b): a short diagnosis of the question with the lowest Context Recall at
    # TOP_K 5. Distribution as context, all 43 questions only in an accordion, the orange box, then question,
    # reference, the headings of the five chunks and the judge's verdict per claim; the choice of a cause right below.
    mo.stop(t.short)
    _top5 = CFG['standard']['top5']
    _fig, _axes = plt.subplots(1, 2, figsize=(6.5, 2.3), sharey=True)
    for _ax, _metric in zip(_axes, ('context_precision', 'context_recall')):
        _v = _top5[_metric]
        _mean = sum(_v) / len(_v)
        _ax.hist(_v, bins=[i / 10 for i in range(11)], color=theme.GREYS[3], edgecolor=theme.INK)
        _ax.axvline(_mean, color='black', linestyle='--', linewidth=1)
        _ax.annotate(t(f'Mittelwert {t.num(_mean)}', f'mean {t.num(_mean)}'), (_mean, 1), xycoords=('data', 'axes fraction'),
                     xytext=(-4, -2), textcoords='offset points', ha='right', va='top', fontsize=8)
        _ax.set_title('Context Precision' if _metric == 'context_precision' else 'Context Recall')
        _ax.set_xlim(0, 1)
        _ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: t.num(v, 1)))
    _axes[0].set_ylabel(t('Fragen', 'Questions'))
    _fig.tight_layout()
    _n0 = sum(x == 0 for x in _top5['context_recall'])

    # All 43 questions for the curious, without a task
    _cols = {'#': [g['index'] + 1 for g in gold], t('Frage', 'Question'): [g['question'] for g in gold]}
    if t.lang == 'en':
        _cols['English'] = [TR['en'].get(g['question'], '') for g in gold]
    _cols['Context Precision'] = [round(x, 2) for x in _top5['context_precision']]
    _cols['Context Recall'] = [round(x, 2) for x in _top5['context_recall']]
    _all = mo.ui.table(pd.DataFrame(_cols), selection=None, page_size=10, show_download=False,
                       show_column_summaries=False, show_data_types=False,
                       wrapped_columns=[c for c in (t('Frage', 'Question'), 'English') if c in _cols])

    # The diagnosed question: tools/precompute_w3_02.py --stage diagnosis (lowest Context Recall, judge run 1)
    DIAG = json.loads(ev.DIAGNOSIS_FILE.read_text(encoding='utf-8'))
    _g = gold[DIAG['index']]
    _chunks = retrieval[DIAG['index']]['chunks']

    def _head(text):
        return text.split('\n')[0].strip()

    def _head_cell(text):
        _en = TR['en'].get(text) if t.lang == 'en' else None
        _de = f'<span class="hpi-corpus">{_head(text)}</span>'
        return _de if not _en else f'{_de}<br><span style="color:{theme.GREY};font-size:.85em">{_head(_en)}</span>'

    _nw = '<span style="white-space:nowrap">{}</span>'.format
    _chunk_rows = [{t('Rang', 'Rank'): str(c['rank']), t('Überschrift des Chunks', 'Heading of the chunk'): _head_cell(c['text'])}
                   for c in _chunks[:5]]
    _claim_rows = [{'#': str(n + 1), t('Aussage der Referenzantwort', 'Claim of the reference answer'): bi_cell(c['statement']),
                    _nw(t('in den Chunks gefunden', 'found in the chunks')): _nw(yes_no(c['attributed'])),
                    t('Begründung des Richters', "Judge's reason"): bi_cell(c['reason'])} for n, c in enumerate(DIAG['claims'])]
    _k, _n = sum(c['attributed'] for c in DIAG['claims']), len(DIAG['claims'])

    ui_why = mo.ui.radio({
        t('Frage und Kompendium verwenden andere Wörter.', 'The question and the Kompendium use different words.'): 'words',
        t('Die Information ist auf mehrere Stellen verteilt.', 'The information is spread over several places.'): 'spread',
        t('Die Referenzantwort enthält etwas, das nicht im Kompendium steht.',
          'The reference answer contains something that is not in the Kompendium.'): 'reference',
        t('Der Richter ist zu streng.', 'The judge is too strict.'): 'judge',
    }, label=t('Warum findet die Suche hier wenig?', 'Why does the search find little here?'))

    mo.vstack([
        mo.md(t('### Teil b: Warum findet die Suche hier wenig?', '### Part b: why does the search find little here?')),
        _fig,
        hint(t(f'TOP_K 5, Standard-Konfiguration, 43 Fragen. Bei {_n0} Fragen liegt Context Recall bei 0; eine davon sehen Sie unten.',
               f'TOP_K 5, standard configuration, 43 questions. For {_n0} questions Context Recall is 0; you see one of them below.')),
        mo.accordion({t('Alle 43 Fragen', 'All 43 questions'): _all}),
        mo.Html(theme.task_box(
            t('Zusatzaufgabe, Teil b', 'Extra task, part b'),
            t(['Lesen Sie direkt unter diesem Kasten die Frage und die Referenzantwort.',
               'Sehen Sie sich darunter die Überschriften der fünf gefundenen Chunks an.',
               'Lesen Sie in der Tabelle danach, welche Aussagen der Referenzantwort der Richter in diesen Chunks gefunden hat.',
               'Wählen Sie unter der Tabelle die Ursache, die Sie für die wahrscheinlichste halten.',
               'Lesen Sie die Rückmeldung darunter.'],
              ['Right below this box, read the question and the reference answer.',
               'Below that, look at the headings of the five retrieved chunks.',
               'In the table after that, read which claims of the reference answer the judge found in these chunks.',
               'Below the table, choose the cause you think is most likely.',
               'Read the feedback below it.']),
            t('etwa 4 Minuten', 'about 4 minutes'),
            t(f"Finden Sie heraus, warum die Suche bei Frage {DIAG['index'] + 1} keine Aussage der Referenzantwort findet.",
              f"Find out why the search finds no claim of the reference answer for question {DIAG['index'] + 1}."),
            kind='extra')),
        mo.md(f"#### {t('Frage', 'Question')} {DIAG['index'] + 1}"), bi(_g['question']),
        mo.md(f"#### {t('Referenzantwort', 'Reference answer')}"),
        hint(t('Eine Fachperson hat sie geschrieben. Sie gehört zum Gold-Datensatz und gilt als richtige Antwort.',
               'An expert wrote it. It belongs to the gold data set and counts as the right answer.')),
        bi(_g['reference']),
        mo.md(t('#### Die fünf gefundenen Chunks', '#### The five retrieved chunks')),
        hint(t('Alle fünf stammen aus dem Baustein APP.3.2 Webserver. Hier stehen nur die Überschriften.',
               'All five come from the module APP.3.2 Webserver. Only the headings are shown here.')),
        mo.md(left(md_table(_chunk_rows))),
        mo.accordion({t('Die fünf Chunks im Volltext', 'The five chunks in full'): mo.vstack(
            [mo.vstack([mo.md(f"**{t('Rang', 'Rank')} {c['rank']}**"), bi(c['text'])], gap=0.25) for c in _chunks[:5]], gap=1)}),
        mo.md(t('#### Urteil des Richters je Aussage', "#### The judge's verdict per claim")),
        mo.md(left(md_table(_claim_rows))),
        hint(t(f'Context Recall = {_k} von {_n} = {t.num(_k / _n)}. Die Begründungen sind englisch, weil die Prompts von RAGAS englisch sind.',
               f'Context Recall = {_k} of {_n} = {t.num(_k / _n)}.')),
        ui_why,
    ], gap=1)
    return DIAG, ui_why


@app.cell(hide_code=True)
def _(DIAG, bi, feedback, gold, info, mo, retrieval, t, ui_why):
    # Row 18b: feedback right after the choice (meta plan 10.4), then the gold passage as the solution.
    # The cause was read from the data of question 1 (2026-10-19): A15 and A18 carry only "(H)", not the words of the question.
    mo.stop(t.short or ui_why.value is None)
    _i = DIAG['index']
    _chunks = retrieval[_i]['chunks']

    def _rank(prefix):
        return next((c['rank'] for c in _chunks if c['text'].startswith(prefix)), None)

    _a15, _a18 = _rank('APP.3.2.A15'), _rank('APP.3.2.A18')
    mo.stop(_i != 0 or _a18 is None, mo.callout(mo.md(t(
        'Der Cache zeigt eine andere Frage oder andere Chunks als beim Schreiben dieser Auflösung. Die Auflösung passt nur zu Frage 1.',
        'The cache shows a different question or different chunks than when this solution was written. The solution fits question 1 only.')),
        kind='warn'))

    def _where(rank):
        return t(f'auf Rang {rank}', f'at rank {rank}') if rank else t('nicht unter den ersten 10', 'not among the first 10')

    _why = t(
        f'Die Antwort steht in zwei Anforderungen des Bausteins APP.3.2 Webserver, A15 und A18 (Kasten unten). Die Frage spricht von "erhöhtem Schutzbedarf". Diese Worte stehen im Kompendium nur über dem ganzen Abschnitt; die beiden Anforderungen tragen dafür nur das Kürzel "(H)". Die Suche findet deshalb den richtigen Baustein, aber allgemeine Teile wie Zielsetzung und Einleitung stehen weiter oben. A18 steht {_where(_a18)}, A15 {_where(_a15)}.',
        f'The answer is in two requirements of the module APP.3.2 Webserver, A15 and A18 (box below). The question speaks of "erhöhter Schutzbedarf" (increased protection requirements). In the Kompendium these words stand only above the whole section; the two requirements carry only the abbreviation "(H)" for it. The search therefore finds the right module, but general parts such as objective and introduction rank higher. A18 is {_where(_a18)}, A15 is {_where(_a15)}.')
    _first = {
        'words': '',
        'spread': t('Die Antwort steht zwar an zwei Stellen. Das erklärt aber nicht, warum keine der beiden unter den fünf Chunks ist.',
                    'The answer is indeed in two places. But that does not explain why neither of them is among the five chunks.'),
        'reference': t('Die Referenzantwort gibt die beiden Anforderungen fast wörtlich wieder; alles darin steht im Kompendium.',
                       'The reference answer restates the two requirements almost word for word; everything in it is in the Kompendium.'),
        'judge': t('Der Richter urteilt hier richtig: In den fünf Chunks steht nichts zu Redundanz und nichts zu DDoS-Angriffen.',
                   'The judge is right here: the five chunks say nothing about redundancy and nothing about DDoS attacks.'),
    }[ui_why.value]
    mo.vstack([
        feedback(ui_why.value == 'words', f'{_first} {_why}'.strip()),
        info(t('So steht die Antwort im Kompendium', 'This is how the Kompendium states the answer'), bi(gold[_i]['fundstelle'])),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(card, cards, frame, hint, key, mo, t):
    # Section H (row 19, section 10.7 of the meta plan): summary as a slide
    mo.vstack([
        mo.md(t('## H. Zusammenfassung', '## H. Summary')),
        key(t('Context Precision und Context Recall sind Urteile eines Richters über Chunks und Aussagen.',
              'Context Precision and Context Recall are verdicts of a judge on chunks and claims.')),
        cards([
            card('Context Precision', t('Stehen die nützlichen Chunks oben?', 'Are the useful chunks at the top?')),
            card('Context Recall', t('Wurde alles gefunden, was die Referenzantwort braucht?', 'Was everything found that the reference answer needs?')),
            card('TOP_K', t('Mehr Chunks heben Recall und senken Precision; der Sweet Spot liegt bei 3 bis 5.',
                            'More chunks raise recall and lower precision; the sweet spot lies at 3 to 5.')),
        ]),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'Ein einzelner Wert schwankt, ein Richter kann zu großzügig sein, und schon die Zerlegung in Aussagen verschiebt die Zahl. Belastbar sind Mittelwerte über viele Fragen, gemessen mit demselben Richter, und Stichproben, die Sie selbst prüfen.',
            'A single value varies, a judge can be too generous, and the split into claims alone shifts the number. Means over many questions, measured with the same judge, and samples you check yourself are what you can rely on.')),
        hint(t('Können Sie Context Precision und Context Recall in je einem Satz erklären?',
               'Can you explain Context Precision and Context Recall in one sentence each?')),
        mo.hstack([
            mo.vstack([mo.md(t('#### Ausblick', '#### Outlook')),
                       mo.md(t('Gute Chunks garantieren noch keine gute Antwort. In Notebook 3 bewerten Sie die Antwort selbst.',
                               'Good chunks do not guarantee a good answer yet. In notebook 3 you evaluate the answer itself.'))]),
            mo.vstack([mo.md(t('#### Nächster Schritt', '#### Next step')),
                       mo.md('`uv run marimo run w3_03_generation_evaluation.py`'),
                       hint(t('Im Ordner notebooks.', 'In the notebooks folder.'))]),
        ], widths='equal', gap=2, align='start'),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # FAQ F1 to F7; F3 and F4 only in the normal version
    _f1 = {t('Gegen was urteilt der Richter bei Context Precision?', 'What does the judge judge against for Context Precision?'): mo.md(t(
        'Er bekommt Frage, Referenzantwort und einen Chunk und entscheidet, ob der Chunk hilft, zu dieser Referenzantwort zu kommen. Deshalb braucht auch Context Precision eine Referenzantwort.',
        'It gets question, reference answer and one chunk and decides whether the chunk helps to arrive at this reference answer. That is why Context Precision, too, needs a reference answer.')),
        t('Warum nicht Recall@k gegen die Fundstellen?', 'Why not Recall@k against the gold passages?'): mo.md(t(
            'Die Fundstellen sind Volltext, keine Chunk-Nummern. Welcher Chunk sie enthält, hängt vom Chunking ab. Der Richter vergleicht Inhalte und funktioniert für jedes Chunking. Dafür kostet er API-Aufrufe und schwankt.',
            'The gold passages are full text, not chunk numbers. Which chunk contains them depends on the chunking. The judge compares content and works for any chunking. In return it costs API calls and varies.'))}
    _f3_f4 = {
        t('Geht RAGAS auch ohne Sprachmodell?', 'Does RAGAS also work without a language model?'): mo.md(t(
            'RAGAS hat auch Varianten von Context Precision und Context Recall ohne Sprachmodell. Sie vergleichen die gefundenen Chunks mit vorgegebenen Referenz-Chunks über Textähnlichkeit. Das ist billig und stabil, braucht aber genau die Markierungen, die wir hier nicht haben.',
            'RAGAS also has variants of Context Precision and Context Recall without a language model. They compare the retrieved chunks with given reference chunks by text similarity. That is cheap and stable, but needs exactly the labels we do not have here.')),
        t('Wie rechne ich andere Konfigurationen selbst?', 'How do I compute other configurations myself?'): mo.md(t(
            'In `ragkit/config.py` stehen Datensatz, Chunking, Embedding-Modell und Richter. Nach einer Änderung starten Sie das Notebook neu. Ohne passende Collection und Cache laufen Suche und Bewertung live, etwa zwei Minuten pro TOP_K-Wert. Dafür braucht es eine Qdrant-Collection; das ist Aufgabe der Kursleitung (siehe README).',
            'Dataset, chunking, embedding model and judge are set in `ragkit/config.py`. After a change, restart the notebook. Without a matching collection and cache, retrieval and scoring run live, about two minutes per TOP_K value. This needs a Qdrant collection; setting it up is the job of the instructors (see README).')),
    }
    _f5 = {t('Warum der Fragensatz mit 43 Fragen?', 'Why the set of 43 questions?'): mo.md(t(
        'Die 43 komplexen Fragen verteilen ihre Antwort oft auf mehrere Stellen. Dadurch zeigt die TOP_K-Kurve etwas. Bei den 40 einfachen Fragen findet das System fast immer alles, und Context Recall liegt fast überall bei 1.',
        'The 43 complex questions often spread their answer over several places. That is why the TOP_K curve shows something. With the 40 simple questions the system almost always finds everything, and Context Recall is close to 1 almost everywhere.'))}
    _f6_f7 = {
        t('Warum hat dieselbe Frage verschiedene Werte?', 'Why does the same question have different values?'): mo.md(t(
            'Tabelle und Kurve stammen aus verschiedenen Läufen des Richters. Jeder Lauf urteilt neu, deshalb weichen einzelne Werte ab. Belastbar sind Mittelwerte über viele Fragen.',
            'Table and curve come from different runs of the judge. Every run judges anew, so single values differ. Means over many questions are what you can rely on.')),
        t('Liegt der BSI-Standard 200-2 in der Collection?', 'Is BSI-Standard 200-2 in the collection?'): mo.md(t(
            'Nein, nur das Kompendium. Aussagen aus 200-2 kann die Suche nicht finden.',
            'No, only the Kompendium. The search cannot find claims from 200-2.')),
    }
    mo.vstack([mo.md(t('## Häufige Fragen', '## Frequently asked questions')),
               mo.accordion({**_f1, **t.full(_f3_f4, {}), **_f5, **_f6_f7})])
    return


if __name__ == "__main__":
    app.run()
