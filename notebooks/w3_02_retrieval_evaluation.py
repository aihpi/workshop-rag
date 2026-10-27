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
    from ragkit.corpus_text import (
        CorpusText,  # corpus text in the language of the notebook
    )
    from ragkit.viz import md_table

    theme.apply_mpl()
    return CorpusText, ev, i18n, json, md_table, mo, os, pd, plt, rag, theme


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
    TASK_ORDER = (['judge', 'quiz_cp', 'cp', 'quiz_cr', 'cr', 'check', 'topk_predict']
                  + t.full(['quiz_topk', 'topk_choose', 'topk_choose2'], []))

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
        mo.accordion({t('Die vier Konfigurationen in diesem Notebook', 'The four configurations in this notebook'): mo.md(t(
            'Eine Konfiguration ist eine Kombination aus Chunking, Embedding-Modell und Richter. Die Kurven in Abschnitt F zeigen die Standard-Konfiguration; die Zusatzaufgabe vergleicht sie mit drei Varianten, die je eine Stellschraube ändern.\n\n'
            '| Name | Chunking | Embedding-Modell | Richter | Was anders ist |\n|---|---|---|---|---|\n'
            '| Standard | json_structured_sections (ein Chunk je Abschnitt des Kompendiums) | octen-embedding-8b | gpt-oss-120b | |\n'
            '| Embedding miniLM | json_structured_sections | miniLM (ein kleines, älteres Modell) | gpt-oss-120b | das Embedding-Modell |\n'
            '| Chunking markdown_headers | markdown_headers (ein Chunk je Überschrift) | octen-embedding-8b | gpt-oss-120b | das Chunking |\n'
            '| Richter llama | json_structured_sections | octen-embedding-8b | llama-3-3-70b (eine andere Modellfamilie) | nur der Richter |',
            'A configuration is one combination of chunking, embedding model and judge. The curves in section F show the standard configuration; the extra task compares it with three variants that each change one lever.\n\n'
            '| Name | Chunking | Embedding model | Judge | What differs |\n|---|---|---|---|---|\n'
            '| Standard | json_structured_sections (one chunk per section of the Kompendium) | octen-embedding-8b | gpt-oss-120b | |\n'
            '| Embedding miniLM | json_structured_sections | miniLM (a small, older model) | gpt-oss-120b | the embedding model |\n'
            '| Chunking markdown_headers | markdown_headers (one chunk per heading) | octen-embedding-8b | gpt-oss-120b | the chunking |\n'
            '| Judge llama | json_structured_sections | octen-embedding-8b | llama-3-3-70b (another model family) | only the judge |',
        ))}),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(ev, mo, rag, t):
    # Row 4 (C30): no note about the collection any more; a red callout only when the cache is missing
    cache_ok = ev.cache_complete() and rag.CHUNKS_FILE.exists() and ev.LMM_FILE.exists()
    (None if cache_ok else mo.callout(mo.md(t(
        'Der Cache mit den Ergebnissen fehlt. Bauen Sie ihn einmal mit `uv run python -m tools.precompute_w3_02` im Ordner `notebooks`.',
        'The results cache is missing. Build it once with `uv run python -m tools.precompute_w3_02` in the `notebooks` folder.',
    )), kind='danger'))
    return (cache_ok,)


@app.cell(hide_code=True)
def _(cache_ok, ev, json, mo, rag):
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
    LMM = json.loads(ev.LMM_FILE.read_text(encoding='utf-8'))  # mixed models, tools/precompute_w3_02.py --stage lmm
    JUDGE_DOWN = ('Der Richter ist gerade nicht erreichbar. Probieren Sie es in einer Minute noch einmal.',
                  'The judge cannot be reached right now. Try again in a minute.')
    LIMIT = 3  # live judge runs per person (section E, part c)
    return CFG, JUDGE_DOWN, KTR, LIMIT, LMM, TR, demo, gold, retrieval


@app.cell(hide_code=True)
def _(CorpusText, KTR, TR, feedback, mo, t, theme):
    # Corpus and RAGAS output in the language of the notebook (ragkit/corpus_text.py): the English version shows
    # the cached translation and keeps the German original in an accordion; table cells show one language only.
    ct = CorpusText(mo, t, {**TR['en'], **KTR['en']})
    bi, bi_cell = ct.show, ct.cell

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
    return ap_steps, bi, bi_cell, check_answer, ct, is_english, nw, strip_html, yes_no


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
def _(card, cards, ev, frame, hint, key, left, md_table, mo, t):
    # Section D, rows D1 to D3: before the judge counts claims, the participants count (Mentimeter in the room),
    # four definitions, the working definition of this notebook with its rules, the three examples resolved, and
    # what RAGAS tells the judge (nothing). The texts are single sentences from gold answers 34, 28 and 30.
    _, _cr_prompt = ev.judge_prompt_text()
    EXAMPLES = [
        ('A', t('Die Anforderungen müssen dokumentiert werden, damit sie identifiziert und bei Planung, Betrieb und Beschaffung berücksichtigt werden können.',
                'The requirements must be documented so that they can be identified and taken into account during planning, operation, and procurement.'),
         [t(['Die Anforderungen müssen dokumentiert werden', 'damit sie identifiziert und bei Planung, Betrieb und Beschaffung berücksichtigt werden können'],
            ['The requirements must be documented', 'so that they can be identified and taken into account during planning, operation, and procurement']),
          t(['Die Anforderungen müssen dokumentiert werden.', 'Dokumentierte Anforderungen können identifiziert werden.', 'Dokumentierte Anforderungen können bei der Planung berücksichtigt werden.', '... beim Betrieb berücksichtigt werden.', '... bei der Beschaffung berücksichtigt werden.'],
            ['The requirements must be documented.', 'Documented requirements can be identified.', 'Documented requirements can be taken into account during planning.', '... during operation.', '... during procurement.']),
          t(['Anforderungen müssen dokumentiert werden.', 'Zweck: Anforderungen identifizieren.', 'Zweck: bei der Planung berücksichtigen.', 'Zweck: beim Betrieb berücksichtigen.', 'Zweck: bei der Beschaffung berücksichtigen.'],
            ['Requirements must be documented.', 'Purpose: identify requirements.', 'Purpose: take into account during planning.', 'Purpose: during operation.', 'Purpose: during procurement.'])]),
        ('B', t('Ohne verbindliche Passwortregeln besteht die Gefahr schwacher oder mehrfach genutzter Passwörter, die leicht kompromittiert werden können.',
                'Without binding password rules, there is a risk of weak or reused passwords that can be easily compromised.'),
         [t(['Ohne verbindliche Passwortregeln besteht die Gefahr schwacher oder mehrfach genutzter Passwörter', 'die leicht kompromittiert werden können'],
            ['Without binding password rules, there is a risk of weak or reused passwords', 'that can be easily compromised']),
          t(['Ohne verbindliche Passwortregeln besteht die Gefahr schwacher Passwörter.', 'Ohne verbindliche Passwortregeln besteht die Gefahr mehrfach genutzter Passwörter.', 'Schwache Passwörter können leicht kompromittiert werden.', 'Mehrfach genutzte Passwörter können leicht kompromittiert werden.'],
            ['Without binding password rules, there is a risk of weak passwords.', 'Without binding password rules, there is a risk of reused passwords.', 'Weak passwords can be easily compromised.', 'Reused passwords can be easily compromised.']),
          t(['Ohne Passwortregeln: Gefahr schwacher Passwörter.', 'Ohne Passwortregeln: Gefahr mehrfach genutzter Passwörter.', 'Schwache oder mehrfach genutzte Passwörter sind leicht zu kompromittieren.'],
            ['Without password rules: risk of weak passwords.', 'Without password rules: risk of reused passwords.', 'Weak or reused passwords are easy to compromise.'])]),
        ('C', t('Ausgabe und Entzug von Zutritts-, Zugangs- und Zugriffsmitteln sind zu dokumentieren und bei Kompromittierung sofort zu ersetzen.',
                'Issuance and revocation of entry, access, and usage media must be documented and replaced immediately if compromised.'),
         [t(['Ausgabe und Entzug von Zutritts-, Zugangs- und Zugriffsmitteln sind zu dokumentieren', 'und bei Kompromittierung sofort zu ersetzen'],
            ['Issuance and revocation of entry, access, and usage media must be documented', 'and replaced immediately if compromised']),
          t(['Die Ausgabe von Zutrittsmitteln ist zu dokumentieren.', '... von Zugangsmitteln ...', '... von Zugriffsmitteln ...', 'Der Entzug von Zutrittsmitteln ist zu dokumentieren.', '... von Zugangsmitteln ...', '... von Zugriffsmitteln ...', 'Zutrittsmittel sind bei Kompromittierung sofort zu ersetzen.', 'Zugangsmittel ...', 'Zugriffsmittel ...'],
            ['Issuance of entry media must be documented.', '... of access media ...', '... of usage media ...', 'Revocation of entry media must be documented.', '... of access media ...', '... of usage media ...', 'Entry media must be replaced immediately if compromised.', 'Access media ...', 'Usage media ...']),
          t(['Die Ausgabe von Zutritts-, Zugangs- und Zugriffsmitteln ist zu dokumentieren.', 'Der Entzug von Zutritts-, Zugangs- und Zugriffsmitteln ist zu dokumentieren.', 'Zutritts-, Zugangs- und Zugriffsmittel sind bei Kompromittierung sofort zu ersetzen.'],
            ['Issuance of entry, access and usage media must be documented.', 'Revocation of entry, access and usage media must be documented.', 'Entry, access and usage media must be replaced immediately if compromised.'])]),
    ]
    _rows_label = t(['Alltag, ganz', 'Alltag, feiner', 'Logik', 'Arbeitsdefinition'], ['Everyday, whole', 'Everyday, granular', 'Logic', 'Working definition'])

    def _breakdown(letter, text, splits):
        _lists = [[text], *splits]
        _rows = [{t('Definition', 'Definition'): lab, t('Aussagen', 'Claims'): ' '.join(f'{i + 1}) {c}' for i, c in enumerate(cl)),
                  t('Anzahl', 'Count'): str(len(cl))} for lab, cl in zip(_rows_label, _lists)]
        return mo.md(left(md_table(_rows)))

    claims_block = mo.vstack([
        mo.md(t('### Was ist eine Aussage?', '### What is a claim?')),
        key(t('Bevor der Richter zählt, zählen Sie: Wie viele Aussagen stecken in jedem dieser drei Sätze?',
              'Before the judge counts, you count: how many claims are in each of these three sentences?')),
        mo.vstack([mo.Html(f'<div class="hpi-corpus"><b>{letter}</b> &nbsp; {text}</div>') for letter, text, _ in EXAMPLES], gap=0.4),
        hint(t('Drei Sätze aus Referenzantworten des Gold-Datensatzes. Ihre Zahlen geben Sie im Raum über Mentimeter ab. Erst danach geht es hier weiter.',
               'Three sentences from reference answers of the gold data set. Give your numbers in the room via Mentimeter. Only then continue here.')),
        mo.md(t('#### Vier Antworten', '#### Four answers')),
        cards([
            card(t('Alltag, ganz', 'Everyday, whole'), t('Eine Aussage ist ein Satz, der etwas feststellt.',
                                                        'A claim is a sentence that states something.')),
            card(t('Alltag, feiner', 'Everyday, granular'), t('Jeder Teilsatz, der etwas feststellt, ist eine Aussage: Hauptsatz und Nebensatz zählen getrennt.',
                                                             'Every clause that states something is a claim: main clause and subordinate clause count separately.')),
            card(t('Logik', 'Logic'), t('Eine Aussage ist etwas, das für sich allein wahr oder falsch sein kann. Jedes Element einer Aufzählung zählt für jede Handlung, zu der es gehört.',
                                        'A claim is something that can be true or false on its own. Every element of a list counts for every action it belongs to.')),
            card(t('Arbeitsdefinition', 'Working definition'), t('Eine Aussage ist der kleinste Teil der Referenzantwort, der sich an einer Stelle im Kompendium prüfen lässt.',
                                                                'A claim is the smallest part of the reference answer that can be checked at one place in the Kompendium.')),
        ]),
        frame(t('Arbeitsdefinition für dieses Notebook', 'Working definition for this notebook'), t(
            'Eine Aussage ist der kleinste Teil der Referenzantwort, der sich an einer Stelle im Kompendium prüfen lässt. Jede Handlung (dokumentieren, ersetzen) ist eine eigene Aussage. Jede Begründung (weil, da) und jeder Zweck (damit, um zu) ist eine eigene Aussage. Eine Aufzählung, die das Kompendium als eine Liste nennt, bleibt eine Aussage. Wiederholungen und Füllwörter zählen nicht. Mit dieser Definition stimmen Sie jetzt noch einmal ab.',
            'A claim is the smallest part of the reference answer that can be checked at one place in the Kompendium. Every action (document, replace) is its own claim. Every reason (because) and every purpose (so that) is its own claim. A list that the Kompendium states as one list stays one claim. Repetitions and filler do not count. Vote again with this definition.')),
        mo.accordion({t(f'Auflösung: Satz {letter} nach jeder Definition', f'Solution: sentence {letter} by each definition'): _breakdown(letter, text, splits)
                      for letter, text, splits in EXAMPLES}),
        hint(t(f'RAGAS gibt dem Richter keine Definition. Die Anweisung lautet nur: "{_cr_prompt.splitlines()[0]}". Der Richter gpt-oss-120b zerlegt meist je Satz, llama-3-3-70b feiner (Abschnitt E b).',
               f'RAGAS gives the judge no definition. The instruction only says: "{_cr_prompt.splitlines()[0]}". The judge gpt-oss-120b mostly splits per sentence, llama-3-3-70b finer (section E b).')),
    ], gap=1)
    return (claims_block,)


@app.cell(hide_code=True)
def _(bi_cell, chain_html, claims_block, demo, hint, key, left, md_table, mo, t, yes_no):
    # Section D (row 10, section 3f): the chain first, the definition block (claims_block), then the claims of the
    # KISZ reference
    _claims = demo['run']['claims']
    _nw = '<span style="white-space:nowrap">{}</span>'.format
    # Section 3g: the judge's reasons in full; long ones wrap (theme.css), nothing is cut off
    _rows = [{'#': str(_n + 1), t('Aussage (RAGAS)', 'Claim (RAGAS)'): bi_cell(c['statement']), _nw(t('belegt', 'supported')): _nw(yes_no(c['attributed'])),
              t('Begründung des Richters', "Judge's reason"): bi_cell(c['reason'])} for _n, c in enumerate(_claims)]
    _formula = t(r'$$\text{Context Recall} = \frac{\text{belegte Aussagen der Referenzantwort}}{\text{alle Aussagen der Referenzantwort}}$$',
                 r'$$\text{Context Recall} = \frac{\text{supported claims of the reference answer}}{\text{all claims of the reference answer}}$$')
    mo.vstack([
        mo.md(t('## D. Context Recall: Fehlt etwas?', '## D. Context Recall: is anything missing?')),
        key(t('Context Recall ist der Anteil der Aussagen aus der Referenzantwort, die in den gefundenen Chunks stehen.',
              'Context Recall is the share of claims from the reference answer that are in the retrieved chunks.')),
        chain_html(t(['Referenzantwort', 'Richter zerlegt sie in Aussagen', 'je Aussage: steht sie in den Chunks?',
                      'Anteil = Context Recall'],
                     ['reference answer', 'judge splits it into claims', 'per claim: is it in the chunks?',
                      'share = Context Recall']), judge_index=1),
        mo.md(_formula),
        claims_block,
        mo.md(t(f'**Die {len(_claims)} Aussagen der Referenzantwort aus Abschnitt B**',
                f'**The {len(_claims)} claims of the reference answer from section B**')),
        mo.md(left(md_table(_rows))),
        hint(t('Der Richter gpt-oss-120b hat hier je Satz eine Aussage gebildet. Was eine Aussage ist, sagt ihm RAGAS nicht; siehe oben.',
               'The judge gpt-oss-120b formed one claim per sentence here. RAGAS does not tell it what a claim is; see above.')),
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
        hint(t(f'Dieselbe Referenzantwort, dieselben fünf Zeilen; zwei Richter zerlegen die Referenzantwort in Aussagen: {ev.JUDGE_MODEL} und {demo["second_judge"]}, ein zweites offenes Modell aus einer anderen Modellfamilie.',
               f'The same reference answer, the same five lines; two judges split the reference answer into claims: {ev.JUDGE_MODEL} and {demo["second_judge"]}, a second open model from another model family.')),
        mo.hstack([_column(ev.JUDGE_MODEL, demo['run']['claims']), _column(demo['second_judge'], demo['second'][0]['claims'])],
                  widths='equal', gap=2, align='start'),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'Schon die Zerlegung entscheidet mit: Wer mehr oder feinere Aussagen bildet, ändert den Nenner und damit den Score.',
            'The split alone already has a say: whoever forms more or finer claims changes the denominator and with it the score.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Section E, part c (row 12, part 1): heading only. The prediction task was dropped on 2026-10-27: whether CP or CR
    # varies depends on the question (rank of the flipped verdict, number of claims), so no answer holds in general.
    mo.vstack([
        mo.md(t('### c. Derselbe Richter, fünf Läufe', '### c. The same judge, five runs')),
        mo.md(t('Der Richter bewertet die Frage aus Abschnitt B mit denselben fünf Zeilen fünfmal.',
                'The judge scores the question from section B with the same five lines five times.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    # Live runs of part c: written by the work cell, rendered by the figure cell.
    get_live, set_live = mo.state({'runs': [], 'error': None})  # error: 'api' or 'key'
    get_used, set_used = mo.state(0)
    return get_live, get_used, set_live, set_used


@app.cell(hide_code=True)
def _(demo, frame, get_live, hint, mo, plt, t, theme):
    # Row 12, part 2: the five cached runs (and your own ones), the range per metric and the key takeaway
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

    _cp, _cr = ('context_precision', 'Context Precision'), ('context_recall', 'Context Recall')
    _range = hint(t(
        f'Ihre Rechnung oben war Lauf 1. Bei dieser Frage {_span(*_cp, inverted=True)}, und {_span(*_cr)}. Bei anderen Fragen kann es anders aussehen: Wie stark ein Wert schwankt, hängt davon ab, auf welchem Rang ein Urteil kippt und in wie viele Aussagen der Richter die Referenzantwort zerlegt.',
        f'Your computation above was run 1. For this question, {_span(*_cp)}, and {_span(*_cr)}. For other questions it can look different: how much a value varies depends on the rank at which a verdict flips and on how many claims the judge splits the reference answer into.'))
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
        _range,
        hint(_why),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'Ein einzelner Wert ist eine Stichprobe. Belastbar sind Mittelwerte über viele Fragen, gemessen mit demselben Richter.',
            'A single value is a sample. Means over many questions, measured with the same judge, are what you can rely on.')),
    ], gap=0.75)
    return


@app.cell(hide_code=True)
def _(LIMIT, cta, get_used, hint, mo, t, theme):
    # Row 12, part 3 (normal only): ask the judge again, live; prominent button, the limit as a hint (section 10.4)
    _spent = get_used() >= LIMIT
    ui_rerun = mo.ui.run_button(
        label=theme.cta_label(t('Alle drei Läufe genutzt', 'All three runs used') if _spent
                              else t('Richter noch einmal fragen', 'Ask the judge again')),
        disabled=_spent, kind='success')
    mo.stop(t.short)
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
        {t('Precision und Recall steigen', 'precision and recall rise'): 'both',
         t('Recall steigt, Precision fällt ab einem Punkt', 'recall rises, precision falls from some point on'): 'trade',
         t('Precision und Recall fallen', 'precision and recall fall'): 'fall'},
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
        hint(t('Abbildung 1. Mittelwerte über 43 Fragen des Gold-Datensatzes, Standard-Konfiguration, Richter gpt-oss-120b, ein Lauf des Richters je TOP_K. Die Streuung von Lauf zu Lauf aus Abschnitt E c steckt in jedem Punkt.',
               'Figure 1. Means over the 43 questions of the gold data set, standard configuration, judge gpt-oss-120b, one judge run per TOP_K. The run-to-run spread of section E c sits in every point.')),
        feedback(ui_pred14.value == _actual, _explain),
        *([hint(t('Vor der Pause: Wer hat "Recall steigt, Precision fällt" getippt?',
                  'Before the break: who predicted "recall rises, precision falls"?'))] if t.short else []),
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
    _flips = [i for i in range(43) if _k1[i] < 0.5 and _k3[i] > 0.99]
    _nums = t(' und ', ' and ').join(str(i + 1) for i in _flips)
    mo.vstack([
        mo.md(t('### Warum die Kurven so verlaufen', '### Why the curves run this way')),
        cards([
            card(t('Precision von 1 auf 3', 'Precision from 1 to 3'),
                 t(f'Steigt vor allem durch Streuung: Bei den Fragen {_nums} hat der Richter denselben Chunk auf Rang 1 einmal verworfen und einmal als nützlich gewertet.',
                   f'Rises mostly through variance: for questions {_nums} the judge once rejected and once accepted the same chunk at rank 1.'),
                 t(f'{len(_flips)} der 43 Fragen tragen fast den ganzen Anstieg', f'{len(_flips)} of the 43 questions carry almost the whole rise')),
            card(t('Precision ab 3', 'Precision from 3 on'),
                 t('Fällt, weil weiter unten noch nützliche Chunks auftauchen und dort der Anteil nützlicher Chunks kleiner ist. Die Suche wird dadurch nicht schlechter.',
                   'Falls because useful chunks still turn up further down, where the share of useful chunks is smaller. That does not make the search worse.')),
            card(t('Chunkgröße', 'Chunk size'),
                 t('Wie viele Chunks nützlich sein können, hängt von der Chunkgröße ab: Große Chunks tragen mehr der Referenzantwort, dann ist Recall schon bei kleinem TOP_K hoch und Precision fällt früher. Die Kurve gilt für diese Chunks (json_structured_sections).',
                   'How many chunks can be useful depends on chunk size: large chunks carry more of the reference answer, so recall is high at small TOP_K already and precision falls earlier. The curve holds for these chunks (json_structured_sections).')),
            card('Recall', t('Steigt, weil mehr Chunks mehr Stellen der Referenzantwort abdecken.',
                             'Rises because more chunks cover more parts of the reference answer.')),
            card(t('Kosten', 'Cost'), t('Jeder Chunk kostet später Platz im Prompt, Zeit und Geld; Text, der nicht zur Frage passt, kann das Sprachmodell ablenken.',
                                        'Every chunk later costs space in the prompt, time and money; text that does not fit the question can distract the language model.')),
        ]),
        mo.accordion({t('Warum kann Precision von TOP_K 1 auf 3 überhaupt steigen?', 'Why can precision rise from TOP_K 1 to 3 at all?'): mo.md(t(
            'Context Precision einer Frage kann beim Verlängern der Liste nur steigen, wenn bis dahin kein nützlicher Chunk dabei war (Wert 0): Ein nützlicher Chunk auf Rang 2 oder 3 hebt sie dann auf etwa 0,5. Ist schon einer dabei, senkt jeder weitere nützliche Chunk weiter unten den Durchschnitt oder lässt ihn gleich. Von 1 auf 3 stehen noch viele Fragen bei 0, von 10 auf 13 fast keine mehr; dort kann Precision also kaum noch steigen. Dazu kommt die Streuung des Richters: Er hat für jeden TOP_K-Wert neu geurteilt, und er urteilt nicht immer gleich.',
            'Context Precision of one question can only rise when the list grows if no useful chunk was in it so far (value 0): a useful chunk at rank 2 or 3 then lifts it to about 0.5. Once one is in, every further useful chunk lower down lowers the average or leaves it. From 1 to 3 many questions still sit at 0, from 10 to 13 almost none; so precision can hardly rise there. On top comes the judge\'s spread: it judged anew for every TOP_K value, and it does not always judge the same way.'))}),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(confirm, mo, t):
    # Section G (rows 16 and 17, normal only): quiz, choice and reason, confirmed with the green button (3e)
    ui_quiz16 = mo.ui.radio(
        {t('weil Recall bei 10 sinkt', 'because recall drops at 10'): 'a',
         t('weil mehr Chunks den Prompt verlängern und die Antwort verschlechtern können',
           'because more chunks lengthen the prompt and can make the answer worse'): 'b',
         t('weil die Suche nicht mehr als 5 liefert', 'because the search returns no more than 5'): 'c'},
        label=t('Warum nimmt man nicht einfach immer TOP_K 10?', 'Why not simply always take TOP_K 10?'))
    ui_choice = mo.ui.radio({'1': 1, '3': 3, '5': 5, '10': 10}, inline=True, label=t('Ihr TOP_K', 'Your TOP_K'))
    ui_reason = mo.ui.text_area(label=t('Ihre Begründung', 'Your reason'), rows=2, full_width=True)
    ui_choice_go = confirm(t('Wahl bestätigen', 'Confirm choice'), lambda: (ui_choice.value, ui_reason.value))
    # The second choice, after the intervals (its own cell would rebuild the first button; both live here)
    ui_choice2 = mo.ui.radio({'1': 1, '3': 3, '5': 5, '10': 10}, inline=True, label=t('Ihr TOP_K, zweite Wahl', 'Your TOP_K, second choice'))
    ui_reason2 = mo.ui.text_area(label=t('Ihre Begründung', 'Your reason'), rows=2, full_width=True)
    ui_choice2_go = confirm(t('Zweite Wahl bestätigen', 'Confirm second choice'), lambda: (ui_choice2.value, ui_reason2.value))
    return ui_choice, ui_choice2, ui_choice2_go, ui_choice_go, ui_quiz16, ui_reason, ui_reason2


@app.cell(hide_code=True)
def _(key, mo, quiz_feedback, quiz_task, t, ui_quiz16):
    # Section G, the quiz as its own counted step before the choice (CR-19)
    t.full(mo.vstack([
        mo.md(t('## G. Ihr TOP_K', '## G. Your TOP_K')),
        key(t('TOP_K ist eine Abwägung: Welche Kennzahl zählt, entscheidet der Einsatz; wo die Zahlen keinen Unterschied belegen, entscheiden die Kosten.',
              'TOP_K is a trade-off: the use case decides which metric counts; where the numbers show no difference, cost decides.')),
        quiz_task('quiz_topk'),
        ui_quiz16,
        quiz_feedback(ui_quiz16.value, 'b', t(
            'Recall steigt mit TOP_K. Der Preis kommt bei der Antwort: längere Prompts, mehr Kosten, und mehr Text, der nicht zur Frage passt.',
            'Recall rises with TOP_K. The price comes with the answer: longer prompts, more cost, and more text that does not fit the question.')),
    ], gap=1))
    return


@app.cell(hide_code=True)
def _(CFG, cta, ev, left, md_table, mo, t, task, ui_choice, ui_choice_go, ui_reason):
    # Section G: the first choice, with the means only (the intervals come after it, in the accordion)
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
def _(LMM, mo, nw, t):
    # The mixed models of the cache as tables and labels, shared by section G and the extra task
    METRICS = (('context_precision', 'Context Precision'), ('context_recall', 'Context Recall'))

    def signed(x, digits=2):
        return nw(t.num(x, digits).replace('-', '&minus;'))

    def p_text(p):
        return nw(t('< 0,001', '< 0.001') if p < 0.001 else t.num(p, 3))

    def level_label(model, level):
        """TOP_K as a number, a configuration by its name in the notebook."""
        return str(level) if model['factor'] == 'topk' else {'standard': t('Standard', 'Standard'), 'minilm': 'miniLM',
                                                             'markdown': 'markdown_headers', 'llama': 'llama'}[level]

    def coef_rows(model):
        """The fixed effects as a tidy table: term, estimate, standard error, z, p, interval."""
        _ref = level_label(model, model['reference'])
        _rows = []
        for c in model['coef']:
            _term = (t(f'Mittelwert bei {_ref} (Referenz)', f'Mean at {_ref} (reference)') if c['level'] is None
                     else t(f'{_ref} &#8594; {level_label(model, c["level"])}', f'{_ref} &#8594; {level_label(model, c["level"])}'))
            _rows.append({t('Term', 'Term'): _term, t('Schätzung', 'Estimate'): signed(c['estimate']),
                          nw(t('Std.-Fehler', 'Std. error')): nw(t.num(c['se'], 3)), 'z': signed(c['z'], 2), 'p': p_text(c['p']),
                          nw(t('95-%-Intervall', '95 % interval')): nw(f"[{t.num(c['lo'])}, {t.num(c['hi'])}]".replace('-', '&minus;'))})
        return _rows

    def variance_text(model):
        _gv, _rv = model['group_var'], model['resid_var']
        _share = _gv / (_gv + _rv)
        return t(f'Varianz zwischen den Fragen (zufälliger Achsenabschnitt): {t.num(_gv, 3)} (Standardabweichung {t.num(_gv ** 0.5)}). Restvarianz: {t.num(_rv, 3)} (Standardabweichung {t.num(_rv ** 0.5)}). {t.num(100 * _share, 0)} % der Streuung liegen zwischen den Fragen, nicht zwischen den TOP_K-Werten.',
                 f'Variance between questions (random intercept): {t.num(_gv, 3)} (standard deviation {t.num(_gv ** 0.5)}). Residual variance: {t.num(_rv, 3)} (standard deviation {t.num(_rv ** 0.5)}). {t.num(100 * _share, 0)} % of the spread lies between questions, not between TOP_K values.')

    def contrast_rows(model):
        return [{t('Vergleich', 'Comparison'): nw(f"{level_label(model, c['a'])} &#8594; {level_label(model, c['b'])}"),
                 t('Differenz', 'Difference'): signed(c['estimate']),
                 nw(t('95-%-Intervall', '95 % interval')): nw(f"[{t.num(c['lo'])}, {t.num(c['hi'])}]".replace('-', '&minus;')),
                 nw(t('Effektstärke', 'Effect size')): signed(c['estimate'] / model['resid_var'] ** 0.5, 2),
                 'p': p_text(c['p']), nw(t('p nach Holm', 'p after Holm')): p_text(c['p_holm']),
                 t('Belegt?', 'Shown?'): nw(t('ja', 'yes') if c['p_holm'] < 0.05 else t('nein', 'no'))} for c in model['contrasts']]

    def summary_accordion(models):
        return mo.accordion({t('Modellausgabe von statsmodels, unverändert', 'Model output of statsmodels, unchanged'): mo.vstack(
            [mo.md(f'```\n{m["summary"]}\n```') for m in models])})
    return METRICS, coef_rows, contrast_rows, level_label, signed, summary_accordion, variance_text


@app.cell(hide_code=True)
def _(LMM, METRICS, coef_rows, confirmed, contrast_rows, frame, hint, left, md_table, mo, plt, summary_accordion, t, theme, ui_choice_go, variance_text):
    # Section G, after the first choice: the neutral feedback, then the accordion with the mixed models
    _given = confirmed(ui_choice_go)
    mo.stop(t.short or ui_choice_go.value is None)
    mo.stop(_given[0] is None or not _given[1].strip(), mo.callout(mo.md(t(
        'Wählen Sie zuerst ein TOP_K und schreiben Sie eine Begründung.',
        'First choose a TOP_K and write a reason.')), kind='warn'))
    _models = [LMM['topk'][m] for m, _ in METRICS]

    # Effects plot: the estimated mean per TOP_K with its 95 % interval, one panel per metric
    _fig1, _axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True, constrained_layout=True)
    for _ax, _model, (_, _name) in zip(_axes, _models, METRICS):
        _lv = [m['level'] for m in _model['means']]
        _ax.plot(_lv, [m['estimate'] for m in _model['means']], '-', color=theme.GREYS[2], zorder=1)
        for _m in _model['means']:
            _ax.plot([_m['level']] * 2, [_m['lo'], _m['hi']], color=theme.INK, linewidth=1.2, zorder=2)
            _ax.plot(_m['level'], _m['estimate'], 'o', color=theme.INK, markersize=6, zorder=3)
        _ax.set_xticks(_lv)
        _ax.set_xlim(0, 11)
        _ax.set_ylim(0, 1)
        _ax.set_xlabel('TOP_K')
        _ax.set_title(_name, fontsize=9.5)
        _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: t.num(v, 1)))
    _axes[0].set_ylabel(t('Geschätzter Mittelwert', 'Estimated mean'))

    # Contrast plot: all six pairwise differences with intervals; filled when shown after the Holm correction
    _fig2, _axes2 = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True, constrained_layout=True)
    for _ax, _model, (_, _name) in zip(_axes2, _models, METRICS):
        _ax.axhline(0, color=theme.GREYS[3], linewidth=1, zorder=0)
        for _i, _c in enumerate(_model['contrasts']):
            _ax.plot([_i, _i], [_c['lo'], _c['hi']], color=theme.INK, linewidth=1.2, zorder=2)
            _ax.plot(_i, _c['estimate'], 'o', color=theme.INK, markerfacecolor=theme.INK if _c['p_holm'] < 0.05 else 'white', markersize=6, zorder=3)
        _ax.set_xticks(range(len(_model['contrasts'])), [t(f"{c['a']} auf {c['b']}", f"{c['a']} to {c['b']}") for c in _model['contrasts']],
                       fontsize=8, rotation=30, ha='right')
        _ax.set_title(_name, fontsize=9.5)
        _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: t.num(v, 1).replace('-', '\u2212')))
    _axes2[0].set_ylabel(t('Differenz der Mittelwerte', 'Difference of the means'))

    _accordion = mo.accordion({t('Ist der Unterschied echt? Signifikanz und Effektstärke', 'Is the difference real? Significance and effect size'): mo.vstack([
        mo.md(t(
            'Ein gemischtes Modell schätzt für jede Kennzahl den Mittelwert je TOP_K-Wert, mit der Frage als zufälligem Effekt: Jede Frage hat ihr eigenes Niveau, und nur die Änderung durch TOP_K innerhalb der Fragen zählt. Die Erklärung von Signifikanz und Effektstärke gibt es im Raum.',
            'A mixed-effects model estimates the mean per TOP_K value for each metric, with the question as a random effect: every question has its own level, and only the change through TOP_K within the questions counts. Significance and effect size are explained in the room.')),
        _fig1,
        hint(t('Abbildung 2. Geschätzter Mittelwert je TOP_K mit 95-%-Intervall, Standard-Konfiguration, 43 Fragen.',
               'Figure 2. Estimated mean per TOP_K with the 95 % interval, standard configuration, 43 questions.')),
        _fig2,
        hint(t('Abbildung 3. Differenz der Mittelwerte (zweiter minus erster TOP_K-Wert) mit 95-%-Intervall, alle sechs Paare. Gefüllt: nach der Holm-Korrektur belegt (p < 0,05).',
               'Figure 3. Difference of the means (second minus first TOP_K value) with the 95 % interval, all six pairs. Filled: shown after the Holm correction (p < 0.05).')),
        *[mo.vstack([mo.md(f'#### {name}'), mo.md(left(md_table(coef_rows(model)))), hint(variance_text(model)),
                     mo.md(t('**Alle Paare**', '**All pairs**')), mo.md(left(md_table(contrast_rows(model))))], gap=0.5)
          for model, (_, name) in zip(_models, METRICS)],
        hint(t('Effektstärke: Differenz geteilt durch die Reststandardabweichung des Modells. Holm: Korrektur für sechs Vergleiche je Kennzahl. Die Scores liegen zwischen 0 und 1; das lineare Modell ist eine Näherung. Jeder Wert ist ein eigener Lauf des Richters; die Streuung von Lauf zu Lauf (Abschnitt E c) steckt damit in der Restvarianz und so auch in den Intervallen.',
               'Effect size: difference divided by the residual standard deviation of the model. Holm: correction for six comparisons per metric. Scores lie between 0 and 1; the linear model is an approximation. Every value is a separate judge run; the run-to-run spread (section E c) is therefore part of the residual variance and so of the intervals.')),
        summary_accordion(_models),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'Ein Wert ist erst dann besser als ein anderer, wenn der Unterschied belegt und groß genug ist. Sonst entscheiden Kosten, Geschwindigkeit und Promptlänge. Belegen die Zahlen eine Abwägung, etwa mehr Recall gegen weniger Precision, entscheidet der Einsatz, welche Kennzahl zählt.',
            'One value is only better than another when the difference is shown and large enough. Otherwise cost, speed and prompt length decide. When the numbers show a trade-off, such as more recall against less precision, the use case decides which metric counts.')),
        hint(t('Wer so TOP_K oder eine Konfiguration gewählt hat, hat die 43 Fragen dafür verbraucht. Die endgültige Rechnung gehört auf einen zurückgehaltenen Fragensatz, den die Auswahl nie gesehen hat.',
               'Whoever chose TOP_K or a configuration this way has used up the 43 questions for it. The final computation belongs on a held-out question set that the selection never saw.')),
    ], gap=0.75)})
    mo.vstack([
        frame(t('Rückmeldung', 'Feedback'), t(
            'Viele wählen 3 oder 5. Für eine fundierte Entscheidung brauchen Sie aber auch Effektstärke und Signifikanz: Ist der Unterschied zwischen zwei Werten überhaupt belegt, und wie groß ist er? Öffnen Sie das Panel darunter.',
            'Many choose 3 or 5. For an informed decision, however, you also need effect size and significance: is the difference between two values shown at all, and how large is it? Open the panel below.')),
        _accordion,
    ], gap=0.75)
    return


@app.cell(hide_code=True)
def _(confirmed, cta, mo, t, task, ui_choice2, ui_choice2_go, ui_choice_go, ui_reason2):
    # Section G: the second choice, with the intervals in view
    _given = confirmed(ui_choice_go)
    mo.stop(t.short or _given is None or _given[0] is None or not _given[1].strip())
    mo.vstack([
        mo.md(t('### Ihre Wahl, mit Intervallen', '### Your choice, with intervals')),
        task('topk_choose2', t('TOP_K noch einmal wählen', 'Choose TOP_K again'), t(
            ['Wählen Sie noch einmal ein TOP_K, jetzt mit Abbildung 2 und der Tabelle aus dem Panel.',
             'Begründen Sie in einem Satz: Welcher Unterschied ist belegt, welcher nicht, und was entscheidet dann?',
             'Drücken Sie auf "Zweite Wahl bestätigen".'],
            ['Choose a TOP_K once more, now with figure 2 and the table from the panel.',
             'Give your reason in one sentence: which difference is shown, which is not, and what decides then?',
             'Press "Confirm second choice".']), t('etwa 3 Minuten', 'about 3 minutes')),
        mo.vstack([ui_choice2, ui_reason2, cta(ui_choice2_go)]),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(LMM, confirmed, feedback, hint, mo, signed, t, ui_choice2_go, ui_choice_go):
    # Section G: assessment of the second choice against the first, from the cached mixed models
    _first, _second = confirmed(ui_choice_go), confirmed(ui_choice2_go)
    mo.stop(t.short or _second is None)
    mo.stop(_second[0] is None or not _second[1].strip(), mo.callout(mo.md(t(
        'Wählen Sie zuerst ein TOP_K und schreiben Sie eine Begründung.',
        'First choose a TOP_K and write a reason.')), kind='warn'))
    _k1, _k2 = (_first[0] if _first else None), _second[0]

    def _c(metric, a, b):
        return next(c for c in LMM['topk'][metric]['contrasts'] if (c['a'], c['b']) == (a, b))

    _r13, _p310, _p35, _r35 = _c('context_recall', 1, 3), _c('context_precision', 3, 10), _c('context_precision', 3, 5), _c('context_recall', 3, 5)
    _r310 = _c('context_recall', 3, 10)
    _shown = [x['p_holm'] < 0.05 for x in (_r13, _p310, _p35, _r35)]
    _changed = t(f'Sie haben Ihre Wahl von {_k1} auf {_k2} geändert. ', f'You changed your choice from {_k1} to {_k2}. ') if _k1 != _k2 else t(
        f'Sie bleiben bei {_k2}. ', f'You stay with {_k2}. ')
    _three_five = t(
        f'Zwischen 3 und 5 ist {"keiner der beiden" if not any(_shown[2:]) else "nur einer der beiden"} Unterschiede belegt (Precision {signed(_p35["estimate"])}, p nach Holm {t.num(_p35["p_holm"], 2)}; Recall {signed(_r35["estimate"])}, p nach Holm {t.num(_r35["p_holm"], 2)}).',
        f'Between 3 and 5 {"neither" if not any(_shown[2:]) else "only one"} of the two differences is shown (precision {signed(_p35["estimate"])}, p after Holm {t.num(_p35["p_holm"], 2)}; recall {signed(_r35["estimate"])}, p after Holm {t.num(_r35["p_holm"], 2)}).')
    _text = t(
        f'{_changed}1 ist belegt schlechter im Recall (1 auf 3: {signed(_r13["estimate"])}, p nach Holm {t.num(_r13["p_holm"], 3)}). Von 3 auf 10 steigt Recall belegt ({signed(_r310["estimate"])}, p nach Holm {t.num(_r310["p_holm"], 3)}) und Precision fällt belegt ({signed(_p310["estimate"])}, p nach Holm {t.num(_p310["p_holm"], 3)}): eine Abwägung, keine Rangfolge. {_three_five} Welche Kennzahl zählt, entscheidet der Einsatz. Darf keine einschlägige Stelle fehlen, etwa bei Rechtsfragen, sind 5 oder 10 gut begründet, bezahlt mit Promptlänge. Sonst spart 3 Promptlänge, ohne belegt schlechter zu sein als 5. Nur 1 ist nicht vertretbar.',
        f'{_changed}1 is shown to be worse in recall (1 to 3: {signed(_r13["estimate"])}, p after Holm {t.num(_r13["p_holm"], 3)}). From 3 to 10 recall is shown to rise ({signed(_r310["estimate"])}, p after Holm {t.num(_r310["p_holm"], 3)}) and precision is shown to fall ({signed(_p310["estimate"])}, p after Holm {t.num(_p310["p_holm"], 3)}): a trade-off, not a ranking. {_three_five} The use case decides which metric counts. If no relevant passage may be missed, for example in legal questions, 5 or 10 are well founded, paid for with prompt length. Otherwise 3 saves prompt length without being shown to be worse than 5. Only 1 is not defensible.')
    mo.vstack([
        feedback(_k2 in (3, 5, 10), _text),
        hint(t('Vor der Pause fragen wir im Raum, wer was gewählt hat und wer gewechselt ist (etwa 3 Minuten).',
               'Before the break we ask the room who chose what and who switched (about 3 minutes).')),
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
            t(['Wählen Sie direkt unter diesem Kasten eine Konfiguration und vergleichen Sie ihre Kurven und die Differenz-Tabelle mit der Standard-Konfiguration.',
               'Ordnen Sie darunter jede Änderung zu: Stellschraube am System oder an der Messung?',
               'Schreiben Sie in einem Satz, was Sie Ihrer Leitung sagen, wenn jemand mit dem llama-Richter bessere Zahlen meldet.',
               'Drücken Sie auf "Prüfen".'],
              ['Right below this box, choose a configuration and compare its curves and the difference table with the standard configuration.',
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
def _(CFG, CFG_LABELS, LMM, METRICS, ev, left, md_table, mo, nw, plt, signed, summary_accordion, t, theme, ui_cfg):
    # Row 18a: the chosen configuration next to the standard one, as curves and as a table
    _ks = ev.TOP_K_VALUES

    def _means(name, metric):
        return [sum(CFG[name]['topk'][str(k)][metric]) / 43 for k in _ks]

    _sel = ui_cfg.value
    # Fixed axes and a legend inside the axes, so the figure does not move when the configuration changes
    _fig, _axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True, constrained_layout=True)
    for _ax, _metric in zip(_axes, ('context_precision', 'context_recall')):
        _ax.plot(_ks, _means('standard', _metric), '--o', color=theme.GREYS[2], markerfacecolor='white', label=CFG_LABELS['standard'])
        if _sel != 'standard':
            _ax.plot(_ks, _means(_sel, _metric), '-o', color='black', label=CFG_LABELS[_sel])
        _ax.set_title('Context Precision' if _metric == 'context_precision' else 'Context Recall')
        _ax.set_xticks(_ks)
        _ax.set_xlim(0, 11)
        _ax.set_ylim(0, 1.0)
        _ax.set_xlabel('TOP_K')
        _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: t.num(v, 1)))
        _ax.legend(loc='lower right', fontsize=8, frameon=False)
    _rows = [{'TOP_K': str(k),
              f"Precision ({CFG_LABELS['standard']})": t.num(_means('standard', 'context_precision')[i]),
              f'Precision ({CFG_LABELS[_sel]})': t.num(_means(_sel, 'context_precision')[i]),
              f"Recall ({CFG_LABELS['standard']})": t.num(_means('standard', 'context_recall')[i]),
              f'Recall ({CFG_LABELS[_sel]})': t.num(_means(_sel, 'context_recall')[i])}
             for i, k in enumerate(_ks)]
    # The mixed model configuration vs standard at TOP_K 5 (score ~ configuration + (1 | question)), as in section G
    _diff_rows = []
    if _sel != 'standard':
        for _metric, _name in METRICS:
            _m = LMM['configs'][_sel][_metric]
            _c = _m['contrasts'][0]
            _diff_rows.append({t('Kennzahl', 'Metric'): _name, t('Differenz zu Standard', 'Difference to standard'): signed(_c['estimate']),
                               nw(t('95-%-Intervall', '95 % interval')): nw(f"[{t.num(_c['lo'])}, {t.num(_c['hi'])}]".replace('-', '&minus;')),
                               nw(t('Effektstärke', 'Effect size')): signed(_c['estimate'] / _m['resid_var'] ** 0.5),
                               'p': nw(t('< 0,001', '< 0.001') if _c['p'] < 0.001 else t.num(_c['p'], 3)),
                               t('Belegt?', 'Shown?'): t('ja', 'yes') if _c['p'] < 0.05 else t('nein', 'no')})
    mo.vstack([_fig, mo.md(left(md_table(_rows))),
               *([mo.md(t('**Differenz zu Standard bei TOP_K 5, gemischtes Modell wie in Abschnitt G**', '**Difference to standard at TOP_K 5, mixed model as in section G**')),
                  mo.md(left(md_table(_diff_rows))), summary_accordion([LMM['configs'][_sel][m] for m, _ in METRICS])] if _sel != 'standard' else [])])
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
def _(CFG, LMM, confirmed, feedback, mo, t, ui_kind_go):
    # Row 18a: the solution in the feedback box; right when all three changes are assigned as below
    _given = confirmed(ui_kind_go)
    mo.stop(t.short or _given is None)
    _kinds, _boss = _given
    mo.stop(any(v is None for v in _kinds.values()) or not _boss.strip(), mo.callout(mo.md(t(
        'Ordnen Sie zuerst alle drei Änderungen zu und schreiben Sie einen Satz für Ihre Leitung.',
        'First assign all three changes and write a sentence for your management.')), kind='warn'))
    _right = {'minilm': 'system', 'markdown': 'system', 'llama': 'measurement'}
    _cr = {n: sum(CFG[n]['topk']['5']['context_recall']) / 43 for n in ('standard', 'llama')}
    _md_shown = LMM['configs']['markdown']['context_precision']['contrasts'][0]['p'] < 0.05
    feedback(_kinds == _right, t(
        f'Embedding-Modell und Chunking ändern, was gefunden wird; das sind Stellschrauben am System. Der Richter ändert nur, wie streng bewertet wird; daher kommt der Sprung von {t.num(_cr["standard"])} auf {t.num(_cr["llama"])} aus Abschnitt E, bei denselben Chunks. Zahlen sind nur unter demselben Richter vergleichbar.'
        + (' Dass markdown_headers in der Precision schlechter aussieht, belegt das Modell nicht: Das Intervall enthält die Null. Hier entscheiden Chunkgröße, Promptlänge und Aufwand, nicht der Score.' if not _md_shown else '')
        + ' Und: Jede Konfiguration wurde einmal bewertet; die Streuung aus Abschnitt E c steckt in jedem Wert.',
        f'Embedding model and chunking change what is found; those are levers on the system. The judge changes only how strictly it scores; that is where the jump from {t.num(_cr["standard"])} to {t.num(_cr["llama"])} in section E comes from, with the same chunks. Numbers are comparable only under the same judge.'
        + (' That markdown_headers looks worse in precision is not shown by the model: the interval contains zero. Here chunk size, prompt length and effort decide, not the score.' if not _md_shown else '')
        + ' And: each configuration was judged once; the spread of section E c sits in every value.'))
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
        """The German heading stays visible here: the solution argues from the German wording (see the hint below)."""
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
               'All five come from the module APP.3.2 Webserver. Only the headings are shown here, in German with the translation below, because the solution argues from the German wording.')),
        mo.md(left(md_table(_chunk_rows))),
        mo.accordion({t('Die fünf Chunks im Volltext', 'The five chunks in full'): mo.vstack(
            [mo.vstack([mo.md(f"**{t('Rang', 'Rank')} {c['rank']}**"), bi(c['text'])], gap=0.25) for c in _chunks[:5]], gap=1)}),
        mo.md(t('#### Urteil des Richters je Aussage', "#### The judge's verdict per claim")),
        mo.md(left(md_table(_claim_rows))),
        hint(t(f'Context Recall = {_k} von {_n} = {t.num(_k / _n)}.', f'Context Recall = {_k} of {_n} = {t.num(_k / _n)}.')),
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
            card('TOP_K', t('Mehr Chunks heben Recall und senken Precision. Zwischen 3 und 5 zeigen die Daten keinen klaren Sieger.',
                            'More chunks raise recall and lower precision. Between 3 and 5 the data show no clear winner.')),
            card(t('Signifikanz und Effektstärke', 'Significance and effect size'),
                 t('Ein Unterschied zählt erst, wenn er die Streuung über die Fragen übersteigt und groß genug ist.',
                   'A difference counts only when it exceeds the spread over the questions and is large enough.')),
        ]),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'Ein einzelner Wert schwankt, ein Richter kann zu großzügig sein, und schon die Zerlegung in Aussagen verschiebt die Zahl. Belastbar sind Mittelwerte über viele Fragen, gemessen mit demselben Richter, und Stichproben, die Sie selbst prüfen. Sind Unterschiede klein, entscheiden Kosten, Geschwindigkeit und Promptlänge.',
            'A single value varies, a judge can be too generous, and the split into claims alone shifts the number. Means over many questions, measured with the same judge, and samples you check yourself are what you can rely on. When differences are small, cost, speed and prompt length decide.')),
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
            'In `ragkit/config.py` stehen Datensatz, Chunking, Embedding-Modell und Richter. Nach einer Änderung starten Sie das Notebook neu. Ohne passende Collection und Cache laufen Suche und Bewertung live, etwa zwei Minuten pro TOP_K-Wert. Dafür braucht es eine Collection in Qdrant, der Vektordatenbank, in der die Chunks liegen; das ist Aufgabe der Kursleitung (siehe README).',
            'Dataset, chunking, embedding model and judge are set in `ragkit/config.py`. After a change, restart the notebook. Without a matching collection and cache, retrieval and scoring run live, about two minutes per TOP_K value. This needs a collection in Qdrant, the vector database that stores the chunks; setting it up is the job of the instructors (see README).')),
    }
    _f5 = {t('Warum der Fragensatz mit 43 Fragen?', 'Why the set of 43 questions?'): mo.md(t(
        'Die 43 komplexen Fragen verteilen ihre Antwort oft auf mehrere Stellen. Dadurch zeigt die TOP_K-Kurve etwas. Bei den 40 einfachen Fragen findet das System fast immer alles, und Context Recall liegt fast überall bei 1.',
        'The 43 complex questions often spread their answer over several places. That is why the TOP_K curve shows something. With the 40 simple questions the system almost always finds everything, and Context Recall is close to 1 almost everywhere.'))}
    _f6_f7 = {
        t('Warum nur ein Lauf des Richters je Konfiguration?', 'Why only one judge run per configuration?'): mo.md(t(
            'Jeder Lauf kostet etwa 1000 Aufrufe der API. Die Streuung von Lauf zu Lauf sehen Sie in Abschnitt E c an einer Frage; der Mittelwert über 43 Fragen dämpft sie, nimmt sie aber nicht weg. Mehrere Läufe je TOP_K sind als Erweiterung vorgemerkt (GitHub-Issue im Repository).',
            'Every run costs about 1000 API calls. You see the run-to-run spread in section E c for one question; the mean over 43 questions dampens it but does not remove it. Several runs per TOP_K are noted as an extension (GitHub issue in the repository).')),
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
