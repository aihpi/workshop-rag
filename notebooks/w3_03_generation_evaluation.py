# ruff: noqa: PLR1711, B018  marimo cells end with an explicit return and render a bare expression

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Ist die Antwort richtig, und ist sie belegt?",
    css_file="ragkit/theme.css",
    html_head_file="ragkit/head.html",
)


@app.cell(hide_code=True)
def _(mo, t, theme):
    # Row 1: header; SECTIONS_MARKER makes every H2 section collapsible (script in ragkit/head.html)
    mo.Html(theme.header_html(
        t('Ist die Antwort richtig, und ist sie belegt?', 'Is the answer correct, and is it supported?'),
        t('RAG III, Notebook 3. Sie bewerten Antworten mit Answer Correctness und Faithfulness und testen einen eigenen System-Prompt.',
          'RAG III, notebook 3. You score answers with Answer Correctness and Faithfulness and test a system prompt of your own.'),
    ) + theme.SECTIONS_MARKER)
    return


@app.cell(hide_code=True)
def _():
    import re  # word counts
    from collections import Counter  # questions that share a point in figure 1

    import marimo as mo  # the reactive notebook itself
    import matplotlib.pyplot as plt  # figures 1 and 2
    import numpy as np
    import pandas as pd

    # language and scope switches, HPI look; translations of the website texts of notebook 1;
    # cached decompositions, prompts and the live runs
    from ragkit import i18n, theme
    from ragkit import intro_rag as rag
    from ragkit import w3_03_gen as gen
    from ragkit.config import setup
    from ragkit.viz import md_table

    theme.apply_mpl()
    return Counter, gen, i18n, md_table, mo, np, pd, plt, rag, re, setup, theme


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

    def quiz_feedback(value, correct, explanation):
        """Immediate feedback under a quiz radio; nothing before a choice. correct=None: no right answer.

        Uses the shared neutral feedback box (meta plan 10.4a) once ragkit/theme.py has it; until then
        the information box, never the green of the task boxes.
        """
        if value is None:
            return mo.md('')
        ok = None if correct is None else value == correct
        return mo.Html(theme.feedback_box(ok, mo.md(explanation).text, t))
    return quiz_feedback, t


@app.cell(hide_code=True)
def _(mo, t, theme):
    # Slide building blocks as in w3_01, shared in ragkit/theme.py: key sentence on the note ground, hairline
    # cards, grey hint lines, information boxes, numbered task boxes. One red mark per view (the title rule).
    key, card, cards, hint, frame, info, left = (theme.key_sentence, theme.card, theme.cards, theme.hint,
                                                 theme.frame, theme.info, theme.left)

    def tiles(items):
        """A row of stat tiles; items are (value, label) or (value, label, caption)."""
        return mo.hstack([mo.stat(v, label=lab, caption=(rest[0] if rest else None), bordered=True)
                          for v, lab, *rest in items], justify='start', gap=1, wrap=True)

    def data_label(kind):
        """The small grey line under a section title that names its data (section 3e: read on KISZ, measure on gold)."""
        return hint(t('Daten: KISZ-Webseite', 'Data: website of the AI Service Center') if kind == 'kisz'
                    else t('Daten: Gold-Datensatz', 'Data: gold data set'))

    # Every place where participants act gets the same task box; one list numbers them.
    # The other-reference task (section C) and its quiz exist in the normal version only.
    TASK_ORDER = ['judge', 'talk', *t.full(['quizref', 'reference'], []), 'quiz1', 'matrix', 'quiz2', 'prompt', 'final']

    def task(name, title, steps, time):
        """The task box of a numbered task, or of the extra task (name='extra')."""
        return theme.numbered_task(t, TASK_ORDER, name, title, steps, time)
    return card, cards, data_label, frame, hint, info, key, left, task, tiles


@app.cell(hide_code=True)
def _(cards, card, hint, key, mo, t):
    # Section A (slide): the two questions this notebook asks of every answer, and the two data sources
    mo.vstack([
        mo.md(t('## A. Zwei Fragen an jede Antwort', '## A. Two questions for every answer')),
        key(t('Gute Chunks garantieren keine gute Antwort. Darum fragen wir: Stimmt die Antwort, und ist sie belegt?',
              'Good chunks do not guarantee a good answer. So we ask: is the answer correct, and is it supported?')),
        cards([
            card('Answer Correctness', t('Stimmt die Antwort mit der Referenzantwort überein?', 'Does the answer agree with the reference answer?'),
                 t('Abschnitt C', 'section C')),
            card('Faithfulness', t('Steht alles, was die Antwort behauptet, in den gefundenen Chunks?', 'Is everything the answer claims in the retrieved chunks?'),
                 t('Abschnitt D', 'section D')),
        ]),
        hint(t('In Notebook 1 haben Sie beide als fertige Zahlen gesehen; hier schauen wir hinein.',
               'In notebook 1 you saw both as finished numbers; here we look inside.')),
        mo.md(t('Zum Nachvollziehen nehmen wir kurze Texte von der Webseite des KI-Servicezentrums, zum Messen den Gold-Datensatz von Fachleuten.',
                'To follow the steps, we use short texts from the website of the AI Service Center; to measure, the gold data set by experts.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(gen, mo, setup, t):
    # Row 4: environment status, visible only when something is missing
    env = setup(require_qdrant=False, strict=False)
    cache_ok = gen.cache_ok()
    _parts = []
    if not env.ok:
        _parts.append(mo.md(env.summary_md()))
    if not cache_ok:
        _parts.append(mo.callout(mo.md(t(
            'Der Cache mit den Ergebnissen fehlt. Bauen Sie ihn einmal mit `uv run python -m tools.precompute_w3_03` im Ordner `notebooks`.',
            'The results cache is missing. Build it once with `uv run python -m tools.precompute_w3_03` in the `notebooks` folder.',
        )), kind='danger'))
    mo.vstack(_parts) if _parts else None
    return (cache_ok,)


@app.cell(hide_code=True)
def _(cache_ok, gen, mo, rag):
    mo.stop(not cache_ok)
    KZ = gen.load(gen.KISZ_EXAMPLES_FILE)   # KISZ website: examples for judging by hand (sections B to E)
    EX = gen.load(gen.EXAMPLES_FILE)        # gold set: the check of question 22 (FAQ)
    EXP = gen.load(gen.EXPERIMENT_FILE)     # prompt experiment on the gold set
    OV = gen.load(gen.OVERVIEW_FILE)        # four metrics for all 43 gold questions
    # German corpus text -> English, English version only: gold texts, website texts of notebook 1, new KISZ texts
    TR = {**gen.load_json_or_empty(gen.TRANSLATIONS_FILE), **rag.load_translations(),
          **gen.load_json_or_empty(gen.KISZ_TRANSLATIONS_FILE)}
    GOLD, SRC = gen.load_gold(), gen.load_source()
    LLM_DOWN = ('Das Sprachmodell ist gerade nicht erreichbar. Probieren Sie es in einer Minute noch einmal.',
                'The language model cannot be reached right now. Try again in a minute.')
    LIMIT = 3  # live runs per person, shared by the reference button (C) and the prompt button (I)
    return EX, EXP, GOLD, KZ, LIMIT, LLM_DOWN, OV, SRC, TR


@app.cell(hide_code=True)
def _(TR, info, mo, re, t, theme):
    def en_of(text):
        """English translation of a cached German text; only in the English version, only when it differs."""
        if t.lang != 'en':
            return None
        _en = TR.get(text)
        return _en if _en and _en.strip() != text.strip() else None

    _sep = re.compile(r'^\s*\|?(\s*:?-{3,}:?\s*\|)+\s*:?-*:?\s*$', re.M)  # the alignment row of a Markdown table

    def _left_tables(text):
        """Left-align the columns of tables in the answer model's Markdown; marimo right-aligns them by default."""
        return _sep.sub(lambda m: re.sub(r':?-{3,}:?', ':---', m.group(0)), text)

    def bi(text, literal=False):
        """Corpus text as it is (class hpi-corpus for the language check), English translation below in grey.

        literal=True shows a leading '#' of a website line as a character, not as a Markdown heading.
        """
        _shown = re.sub(r'^#+', lambda m: '\\#' * len(m.group(0)), text) if literal else text
        _orig = mo.Html(f'<div class="hpi-corpus">{mo.md(_left_tables(_shown)).text}</div>')
        _en = en_of(text)
        if _en is not None and literal:
            _en = re.sub(r'^#+\s*', '', _en)
        return _orig if _en is None else mo.vstack(
            [_orig, theme.hint(_left_tables(_en))], gap=0.25)

    def flat(s):
        return ' '.join(str(s).split()).replace('|', '/')

    def bi_cell(text):
        """`bi` for one Markdown table cell."""
        _en = en_of(text)
        _orig = f'<span class="hpi-corpus">{flat(text)}</span>'
        return _orig if _en is None else f'{_orig}<br><span style="color:{theme.GREY};font-size:.85em">{flat(_en)}</span>'

    def reason(text):
        """A judge reason exactly as RAGAS returned it (German or English, see the hint under each table)."""
        return f'<span class="hpi-corpus">{flat(text)}</span>'

    def fund_view(text):
        """The gold passage of a gold question in an information box."""
        return info(t('Das steht im Kompendium', 'This is what the Kompendium says'), bi(text))

    def web_view(e):
        """The website lines that answer a KISZ question, in an information box, as in notebook 1."""
        return info(t('Das steht auf der Webseite', 'This is what the website says'),
                    mo.vstack([mo.vstack([mo.md(f"*{t('Zeile', 'Line')} {n}*"), bi(line)], gap=0.25)
                               for n, line in zip(e['source_ids'], e['source_lines'])], gap=0.75))

    def chunk_lines(e):
        """The three retrieved website lines of a KISZ question; short enough to stay visible."""
        return mo.vstack([mo.vstack([mo.md(f"*Chunk {k + 1}, {t('Zeile', 'line')} {n}*"), bi(c, literal=True)], gap=0.25)
                          for k, (n, c) in enumerate(zip(e['chunk_ids'], e['contexts']))], gap=0.75)

    MATRIX = {t('richtig und belegt', 'correct and supported'): 'both',
              t('richtig, aber nicht belegt', 'correct, but not supported'): 'correct_only',
              t('belegt, aber falsch', 'supported, but wrong'): 'supported_only',
              t('weder noch', 'neither'): 'neither'}

    def judge_class(e):
        """Place an example the way the judge does; below 0.5 reads as low."""
        _ok, _sup = e['answer_correctness'] >= 0.5, e['faithfulness_score'] >= 0.5
        return {(True, True): 'both', (True, False): 'correct_only',
                (False, True): 'supported_only', (False, False): 'neither'}[(_ok, _sup)]

    def kisz_card(title, e, control):
        """A card for the matrix task: reference and the three chunks left, the answer right, the choice below."""
        _q_en = en_of(e['question'])
        return mo.Html(f'<div style="border:1px solid {theme.RULE};border-radius:2px;padding:.75rem 1rem;background:#fff">'
                       + mo.vstack([
                           mo.md(f"**{title}.** <span class=\"hpi-corpus\">{e['question']}</span>"),
                           *([mo.Html(f'<div class="hpi-hint">{_q_en}</div>')] if _q_en else []),
                           mo.hstack([
                               mo.vstack([mo.md(f"**{t('Referenzantwort', 'Reference answer')}**"), bi(e['reference']),
                                          mo.md(f"**{t('Die drei gefundenen Chunks', 'The three retrieved chunks')}**"),
                                          chunk_lines(e)], gap=0.5),
                               mo.vstack([mo.md(f"**{t('Antwort der Pipeline', 'Answer of the pipeline')}**"), bi(e['answer'])], gap=0.5),
                           ], widths='equal', gap=2, align='start'),
                           control,
                       ], gap=0.75).text + '</div>')

    def nwords(text):
        return len(re.findall(r'\S+', text))
    return MATRIX, bi, bi_cell, chunk_lines, en_of, flat, fund_view, judge_class, kisz_card, nwords, reason, web_view


@app.cell(hide_code=True)
def _(KZ, bi, data_label, key, mo, t, task, web_view):
    # Section B (row 5, 3e): judge one answer yourself; KISZ question, website lines beside it
    B = KZ['B']
    ui_judge = mo.ui.radio(
        {t('richtig und vollständig', 'correct and complete'): 'full',
         t('teilweise richtig', 'partly correct'): 'partly',
         t('falsch', 'wrong'): 'wrong'}, inline=True, label=t('Ist die Antwort fachlich richtig?', 'Is the answer correct on the subject matter?'))
    mo.vstack([
        mo.md(t('## B. Ihr Urteil zuerst', '## B. Your judgment first')),
        data_label('kisz'),
        key(t('Sie urteilen zuerst selbst, mit der Quelle auf der Webseite; danach sehen Sie den Score.',
              'You judge first yourself, with the source on the website; then you see the score.')),
        task('judge', t('Eine Antwort beurteilen', 'Judge an answer'), t(
            ['Lesen Sie links Frage, Referenzantwort und die Zeile der Webseite.', 'Lesen Sie rechts die Antwort der Pipeline.',
             'Wählen Sie unten, ob die Antwort fachlich richtig ist.'],
            ['Read the question, the reference answer and the line of the website on the left.', 'Read the answer of the pipeline on the right.',
             'Choose below whether the answer is correct on the subject matter.']), t('etwa 2 Minuten', 'about 2 minutes')),
        mo.hstack([
            mo.vstack([
                mo.md(f"#### {t('Frage', 'Question')}"), bi(B['question']),
                mo.md(f"#### {t('Referenzantwort', 'Reference answer')}"), bi(B['reference']),
                web_view(B),
            ]),
            mo.vstack([mo.md(f"#### {t('Antwort der Pipeline', 'Answer of the pipeline')}"), bi(B['answer'])]),
        ], widths='equal', gap=2, align='start'),
        ui_judge,
    ], gap=1)
    return B, ui_judge


@app.cell(hide_code=True)
def _(B, hint, mo, quiz_feedback, t, task, tiles, ui_judge):
    # Row 6: the score against your judgment (text depends on the choice, section 3b), then the talk with the neighbor
    mo.stop(ui_judge.value is None)
    _choice = {'full': t('richtig und vollständig', 'correct and complete'),
               'partly': t('teilweise richtig', 'partly correct'), 'wrong': t('falsch', 'wrong')}[ui_judge.value]
    _ac = t.num(B['answer_correctness'])
    _fit = (t(f'Ihr Urteil und der Score passen zusammen: {_ac} heißt hier nicht schlecht, sondern teilweise übereinstimmend.',
              f'Your judgment and the score fit together: {_ac} does not mean bad here, but partly in agreement.')
            if ui_judge.value == 'partly' else
            t('Ihr Urteil und der Score gehen auseinander.', 'Your judgment and the score diverge.'))
    mo.vstack([
        tiles([(_choice, t('Ihr Urteil', 'Your judgment')), (_ac, 'Answer Correctness'),
               (t.num(B['short']['answer_correctness']), t('kurze Antwort aus Notebook 1', 'short answer from notebook 1'), 'Answer Correctness'),
               (f"{B['words_answer']} / {B['words_reference']}", t('Wörter: Antwort / Referenz', 'words: answer / reference'))]),
        quiz_feedback(ui_judge.value, None, t(f'**{_fit}** Die Antwort nennt alle drei Angebote der Referenz, erklärt aber jedes davon.',
                                              f'**{_fit}** The answer names all three offers of the reference, but explains each of them.')),
        hint(t('Gleiche Frage, gleiche Chunks wie in Notebook 1; nur verlangt der System-Prompt hier keine Kürze (der Original-Prompt aus Abschnitt I).',
               'Same question, same chunks as in notebook 1; only the system prompt here does not ask for brevity (the original prompt from section I).')),
        hint(t('Warum die Erklärungen den Score senken, zeigt Abschnitt C. Ob sie auf der Webseite stehen, prüfen Sie in Abschnitt D.',
               'Section C shows why the explanations lower the score. Whether they are on the website, you check in section D.')),
        task('talk', t('Mit dem Nachbarn sprechen', 'Talk with your neighbor'), t(
            ['Sprechen Sie mit Ihrem Nachbarn: Wäre diese Antwort für Sie in der Praxis gut oder schlecht?',
             'Überlegen Sie gemeinsam: Was sagt der Score dann über Ihr System?'],
            ['Talk with your neighbor: would this answer be good or bad for you in practice?',
             'Consider together: what does the score then say about your system?']),
            t('etwa 1 Minute', 'about 1 minute') if t.short else t('etwa 2 Minuten', 'about 2 minutes')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(B, bi_cell, card, cards, data_label, hint, key, left, md_table, mo, reason, t, tiles):
    # Section C (row 7, 3e): key sentence, formula, the short claim table of example B, the calculation; the rest folded
    _ac = B['ac']
    _tp, _fp, _fn = len(_ac['TP']), len(_ac['FP']), len(_ac['FN'])
    _num = t.num
    _rows = [(kind, s) for kind in ('TP', 'FP', 'FN') for s in _ac[kind]]
    _claims = mo.md(left(md_table([{t('Einordnung', 'Class'): kind, t('Aussage der Antwort', 'Claim of the answer'): bi_cell(s['statement'])}
                                   for kind, s in _rows])))
    _reasons = mo.vstack([
        mo.md(left(md_table([{t('Einordnung', 'Class'): kind, t('Aussage', 'Claim'): bi_cell(s['statement']),
                              t('Begründung des Richters', "Judge's reason"): reason(s['reason'])} for kind, s in _rows]))),
        hint(t('So, wie RAGAS sie erzeugt hat; der Richter schreibt seine Begründungen mal deutsch, mal englisch.',
               'As RAGAS produced them; the judge writes its reasons sometimes in German, sometimes in English.')),
    ], gap=0.5)
    _f1_text = mo.md(t(
        'Recall kennen Sie aus Notebook 2: der Anteil der Aussagen, der sich wiederfindet. Precision ist hier der einfache Anteil, ohne Ränge: Wie viel von der Antwort steht in der Referenz?\n\nF1 verbindet beide. Die Ähnlichkeit ist die der Embeddings von Antwort und Referenz.',
        'You know recall from notebook 2: the share of claims that are found again. Precision is the plain share here, without ranks: how much of the answer is in the reference?\n\nF1 combines both. The similarity is that of the embeddings of answer and reference.'))
    mo.vstack([
        mo.md(t('## C. Answer Correctness: stimmt die Antwort mit der Referenz überein?',
                '## C. Answer Correctness: does the answer agree with the reference?')),
        data_label('kisz'),
        key(t('Answer Correctness misst Übereinstimmung mit der Referenz, nicht fachliche Qualität.',
              'Answer Correctness measures agreement with the reference, not subject-matter quality.')),
        hint(t('Der Richter zerlegt Antwort und Referenzantwort in Aussagen und ordnet jede Aussage zu:',
               'The judge splits answer and reference answer into claims and assigns each claim:')),
        cards([
            card(t('Treffer (TP)', 'Hit (TP)'), t('Die Aussage der Antwort steht auch in der Referenz.', 'The claim of the answer is also in the reference.')),
            card(t('Zu viel (FP)', 'Too many (FP)'), t('Die Aussage steht nur in der Antwort.', 'The claim is only in the answer.')),
            card(t('Fehlt (FN)', 'Missing (FN)'), t('Die Aussage steht nur in der Referenz.', 'The claim is only in the reference.')),
        ]),
        mo.md(t(r'$$\text{Answer Correctness} = 0{,}75 \cdot F1 + 0{,}25 \cdot \text{Ähnlichkeit} \qquad F1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$',
                r'$$\text{Answer Correctness} = 0.75 \cdot F1 + 0.25 \cdot \text{similarity} \qquad F1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$')),
        mo.accordion({t('Was heißen F1 und Ähnlichkeit?', 'What do F1 and similarity mean?'): _f1_text}),
        mo.md(t('### Die Antwort aus Abschnitt B', '### The answer from section B')),
        tiles([(str(_tp), 'TP'), (str(_fp), 'FP'), (str(_fn), 'FN'), (_num(_ac['f1']), 'F1'),
               (_num(_ac['similarity']), t('Ähnlichkeit', 'Similarity')), (_num(_ac['score']), 'Answer Correctness')]),
        mo.md(t(f"F1 = 2 · {_tp} / (2 · {_tp} + {_fp} + {_fn}) = {_num(_ac['f1'])}, Answer Correctness = 0,75 · {_num(_ac['f1'])} + 0,25 · {_num(_ac['similarity'])} = **{_num(_ac['score'])}**",
                f"F1 = 2 · {_tp} / (2 · {_tp} + {_fp} + {_fn}) = {_num(_ac['f1'])}, Answer Correctness = 0.75 · {_num(_ac['f1'])} + 0.25 · {_num(_ac['similarity'])} = **{_num(_ac['score'])}**")),
        _claims,
        mo.md(t(f'**Alle {_tp} Angebote der Referenz sind Treffer, nichts fehlt.** Die {_fp} Erklärungen dazu stehen nicht in der Referenz und zählen als zu viel.',
                f'**All {_tp} offers of the reference are hits, nothing is missing.** The {_fp} explanations do not appear in the reference and count as too many.')),
        mo.accordion({t('Die Begründungen des Richters', "The judge's reasons"): _reasons}),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    # Session state, never cached: the accepted own reference, its live score, and the live runs used,
    # shared by the reference button (section C) and the prompt button (section I).
    get_ref, set_ref = mo.state({'text': None, 'error': None})
    get_refscore, set_refscore = mo.state({'text': None, 'result': None, 'error': False})
    get_used, set_used = mo.state(0)
    return get_ref, get_refscore, get_used, set_ref, set_refscore, set_used


@app.cell(hide_code=True)
def _(mo, t):
    # Section 3d: the quiz before the other-reference task (normal only)
    ui_quiz_ref = mo.ui.radio(
        {t('als Treffer (TP)', 'as hits (TP)'): 'tp', t('als zu viel (FP)', 'as too many (FP)'): 'fp',
         t('gar nicht', 'not at all'): 'none'},
        label=t('Eine Antwort enthält zwei richtige Aussagen, die in der Referenzantwort nicht vorkommen. Wie zählt Answer Correctness sie?',
                'An answer contains two correct claims that do not appear in the reference answer. How does Answer Correctness count them?'))
    return (ui_quiz_ref,)


@app.cell(hide_code=True)
def _(KZ, bi, card, data_label, hint, info, mo, quiz_feedback, t, task, ui_quiz_ref):
    # Section 3d with the coordinator's refinement: the newsletter question of notebook 1, section E.
    # Step 1 shows question, website lines and the pipeline's answer; the RAGAS reference stays hidden until step 3.
    R = KZ['ref']
    t.full(mo.vstack([  # the short version drops this part entirely
        mo.md(t('### Gleiche Antwort, andere Referenz', '### Same answer, different reference')),
        data_label('kisz'),
        mo.md(t('**Erinnern Sie sich an Ihre Referenz aus Notebook 1? Jetzt sehen Sie, was sie am Score ändert.**',
                '**Do you remember your reference from notebook 1? Now you see what it changes in the score.**')),
        mo.md(t('### Kurz geprüft', '### Quick check')),
        task('quizref', '', t(['Beantworten Sie die Frage.', 'Lesen Sie die Rückmeldung darunter.'],
                              ['Answer the question.', 'Read the feedback below it.']), t('etwa 1 Minute', 'about 1 minute')),
        ui_quiz_ref,
        quiz_feedback(ui_quiz_ref.value, 'fp', t(
            'Answer Correctness vergleicht nur mit der Referenz; was dort fehlt, zählt als zu viel, auch wenn es stimmt. Darum hängt der Score davon ab, wer die Referenz geschrieben hat.',
            'Answer Correctness compares only with the reference; what is missing there counts as too many, even if it is correct. That is why the score depends on who wrote the reference.')),
        mo.md(t('#### Schritt 1: Frage, Quelle und Antwort der Pipeline', '#### Step 1: question, source and answer of the pipeline')),
        info(t('Frage von RAGAS und Quelle', 'Question by RAGAS and source'), mo.hstack([
            mo.vstack([mo.md(f"**{t('Frage von RAGAS', 'Question by RAGAS')}**"), bi(R['question']),
                       hint(t('RAGAS hat die Frage auf Englisch erzeugt, obwohl die Webseite deutsch ist.',
                              'RAGAS generated the question in English although the website is German.'))]),
            mo.vstack([mo.md(f"**{t('Quelle auf der Webseite', 'Source on the website')}**"),
                       *[mo.vstack([mo.md(f"*{t('Zeile', 'Line')} {n}*"), bi(line)])
                         for n, line in zip(R['source_ids'], R['source_lines'])]], gap=0.5),
        ], widths='equal', gap=2, align='start')),
        card(t('Antwort der Pipeline', 'Answer of the pipeline'), R['answer'], corpus=True),
    ], gap=1))
    return (R,)


@app.cell(hide_code=True)
def _(mo, t, task, theme):
    # Step 2 (normal only): write the reference answer, then confirm it with the green button
    ui_myref = mo.ui.text_area(label=t('Ihre Referenzantwort', 'Your reference answer'), rows=3, full_width=True)
    ui_accept = mo.ui.run_button(label=theme.cta_label(t('Referenzantwort übernehmen', 'Use this reference answer')))
    t.full(mo.vstack([
        mo.md(t('#### Schritt 2: Ihre Referenzantwort', '#### Step 2: your reference answer')),
        task('reference', t('Eine eigene Referenzantwort', 'A reference answer of your own'), t(
            ['Schreiben Sie die richtige Antwort, nur aus den Zeilen der Quelle; gern wieder Ihre aus Notebook 1.',
             'Drücken Sie auf "Referenzantwort übernehmen".',
             'Drücken Sie in Schritt 3 auf "Antwort gegen beide Referenzen bewerten".'],
            ['Write the correct answer, only from the lines of the source; your one from notebook 1 is welcome again.',
             'Press "Use this reference answer".',
             'In step 3, press "Score the answer against both references".']),
            t('etwa 4 Minuten', 'about 4 minutes')),
        ui_myref,
        mo.Html(f'<div class="hpi-cta">{ui_accept}</div>'),
    ], gap=1))
    return ui_accept, ui_myref


@app.cell(hide_code=True)
def _(mo, set_ref, ui_accept, ui_myref):
    # Takes over the written reference; step 3 shows this text, even if the field changes later
    mo.stop(not ui_accept.value)
    _text = ui_myref.value.strip()
    set_ref({'text': _text, 'error': None} if _text else {'text': None, 'error': 'empty'})
    return


@app.cell(hide_code=True)
def _(LIMIT, R, bi, get_ref, get_used, hint, mo, nwords, t, theme):
    # Step 3 (normal only), after "übernehmen": three columns with word counts, then the green scoring button
    mo.stop(t.short)
    _s = get_ref()
    mo.stop(_s['error'] == 'empty', mo.callout(mo.md(t('Schreiben Sie zuerst eine Referenzantwort.', 'Write a reference answer first.')), kind='warn'))
    mo.stop(_s['text'] is None)
    _spent = get_used() >= LIMIT
    ui_score = mo.ui.run_button(label=theme.cta_label(
        t('Alle drei Live-Läufe genutzt', 'All three live runs used') if _spent
        else t('Antwort gegen beide Referenzen bewerten', 'Score the answer against both references')), disabled=_spent)

    def _col(title, view, n):
        _foot = (f'<div class="hpi-hint" style="margin-top:.6rem;padding-top:.4rem;border-top:1px solid {theme.RULE}">'
                 f'{t(f"{n} Wörter", f"{n} words")}</div>')
        return mo.Html(f'<div style="border:1px solid {theme.RULE};border-radius:2px;padding:.75rem 1rem;background:#fff;height:100%">'
                       f'<p><strong>{title}</strong></p>{view.text}{_foot}</div>')

    _mine = mo.Html(f'<div class="hpi-corpus">{mo.md(_s["text"]).text}</div>')
    mo.vstack([
        mo.md(t('#### Schritt 3: dieselbe Antwort, zwei Referenzen', '#### Step 3: the same answer, two references')),
        mo.hstack([
            _col(t('Antwort der Pipeline', 'Answer of the pipeline'), bi(R['answer']), R['words_answer']),
            _col(t('Referenz von RAGAS', 'Reference by RAGAS'), bi(R['reference']), R['words_reference']),
            _col(t('Ihre Referenz', 'Your reference'), _mine, nwords(_s['text'])),
        ], widths='equal', gap=1, align='stretch'),
        mo.Html(f'<div class="hpi-cta">{ui_score}</div>'),
        hint(t('Sie können bis zu dreimal live bewerten lassen, gemeinsam mit dem Test eines eigenen Prompts in Abschnitt I. Eine Bewertung dauert etwa eine halbe Minute.',
               'You can have up to three live runs, shared with the test of a prompt of your own in section I. One scoring takes about half a minute.')),
    ], gap=1)
    return (ui_score,)


@app.cell(hide_code=True)
async def _(LIMIT, R, gen, get_ref, get_used, mo, set_refscore, set_used, t, ui_score):
    # Live: Answer Correctness of the same answer (same claims) against the own reference; nothing is cached
    mo.stop(t.short or not ui_score.value or get_used() >= LIMIT)
    _text = get_ref()['text']
    try:
        with mo.status.spinner(title=t('Der Richter bewertet ...', 'The judge is scoring ...')):
            _res = await gen.ascore_reference(R['question'], R['answer'], R['ac']['answer_claims'], _text)
    except Exception:  # noqa: BLE001 - any API failure gets the same sentence
        set_refscore({'text': _text, 'result': None, 'error': True})
    else:
        set_refscore({'text': _text, 'result': _res, 'error': False})
        set_used(lambda n: n + 1)
    return


@app.cell(hide_code=True)
def _(LLM_DOWN, R, get_ref, get_refscore, hint, mo, t, tiles):
    # Results under the columns: the RAGAS reference from the cache, the own one live; then the reasons from the real texts
    _exp = R['ac']
    _why_static = t(
        f"Die Referenz von RAGAS ist englisch und hat {R['words_reference']} Wörter, die Antwort ist deutsch und hat {R['words_answer']}. Gegen sie zählen alle {len(_exp['TP'])} Aussagen der Antwort als Treffer, nichts fehlt; die Ähnlichkeit der Embeddings liegt bei {t.num(_exp['similarity'])}.",
        f"The RAGAS reference is in English and has {R['words_reference']} words; the answer is in German and has {R['words_answer']}. Against it, all {len(_exp['TP'])} claims of the answer count as hits, nothing is missing; the similarity of the embeddings is {t.num(_exp['similarity'])}.")

    def _cap(tp, fp, fn):
        return f'TP {tp}, FP {fp}, FN {fn}'

    _exp_tile = (t.num(_exp['score']), t('Answer Correctness, Referenz von RAGAS', 'Answer Correctness, RAGAS reference'),
                 _cap(len(_exp['TP']), len(_exp['FP']), len(_exp['FN'])))

    mo.stop(t.short)  # the short version drops this part entirely
    _ref, _sc = get_ref(), get_refscore()
    mo.stop(_ref['text'] is None or _sc['text'] != _ref['text'])
    if _sc['error']:
        _view = mo.vstack([
            mo.callout(mo.md(t(*LLM_DOWN) + ' ' + t('Bis dahin sehen Sie nur den Wert für die Referenz von RAGAS.',
                                                      'Until then you see only the value for the RAGAS reference.')), kind='warn'),
            tiles([_exp_tile]),
        ], gap=1)
    else:
        _own = _sc['result']
        _diff = abs(_own['score'] - _exp['score'])
        _lines = []
        if _own['FP'] == 1:
            _lines.append(t('Gegen Ihre Referenz zählt eine Aussage der Antwort als zu viel.',
                            'Against your reference, one claim of the answer counts as too many.'))
        elif _own['FP']:
            _lines.append(t(f"Gegen Ihre Referenz zählen {_own['FP']} Aussagen der Antwort als zu viel.",
                            f"Against your reference, {_own['FP']} claims of the answer count as too many."))
        if _own['FN']:
            _lines.append(t('Eine Aussage Ihrer Referenz fehlt in der Antwort.', 'One claim of your reference is missing from the answer.')
                          if _own['FN'] == 1 else
                          t(f"{_own['FN']} Aussagen Ihrer Referenz fehlen in der Antwort.",
                            f"{_own['FN']} claims of your reference are missing from the answer."))
        _lines.append(t(f"Die Ähnlichkeit der Embeddings liegt bei {t.num(_own['similarity'])} statt {t.num(_exp['similarity'])}.",
                        f"The similarity of the embeddings is {t.num(_own['similarity'])} instead of {t.num(_exp['similarity'])}."))
        _view = mo.vstack([
            tiles([_exp_tile, (t.num(_own['score']), t('Answer Correctness, Ihre Referenz', 'Answer Correctness, your reference'), _cap(_own['TP'], _own['FP'], _own['FN']))]),
            mo.md(t('**Gleiche Antwort, andere Referenz, anderer Score.**', '**Same answer, different reference, different score.**')
                  if _diff >= 0.02 else
                  t('**Gleiche Antwort, andere Referenz, hier fast derselbe Score.**', '**Same answer, different reference, here almost the same score.**')),
            mo.accordion({t('Warum unterscheiden sich die Werte?', 'Why do the values differ?'): mo.vstack([
                mo.md(_why_static),
                mo.md(' '.join(_lines)),
                hint(t(f"Die Aussagen der Antwort sind in beiden Bewertungen dieselben; nur die Referenz wechselt. In einem früheren Lauf bekam dieselbe Antwort gegen die Referenz von RAGAS {t.num(R['answer_correctness_w3_01'])}; wie stark der Richter schwankt, zeigt Abschnitt H.",
                       f"The claims of the answer are the same in both scorings; only the reference changes. In an earlier run the same answer got {t.num(R['answer_correctness_w3_01'])} against the RAGAS reference; section H shows how much the judge varies.")),
            ], gap=0.75)}),
        ], gap=1)
    _view
    return


@app.cell(hide_code=True)
def _(B, bi_cell, card, cards, chunk_lines, data_label, hint, info, key, left, md_table, mo, reason, t, tiles):
    # Section D (row 8, 3e): Faithfulness on the same KISZ answer; the short claim list stays visible,
    # the strictness of the judge on the Kompendium (MUSS, SOLLTE) is folded
    _claims = B['faithfulness']['claims']
    _yes = sum(c['verdict'] for c in _claims)
    _n = len(_claims)
    _table = mo.md(left(md_table([{t('Aussage der Antwort', 'Claim of the answer'): bi_cell(c['statement']),
                                   '<span style="white-space:nowrap">' + t('belegt', 'supported') + '</span>':
                                       t('ja', 'yes') if c['verdict'] else f"<b style=\"white-space:nowrap\">{t('nein', 'no')}</b>",
                                   t('Begründung des Richters', "Judge's reason"): reason(c['reason'])} for c in _claims])))
    _kompendium = mo.vstack([
        cards([
            card('MUSS', t('Pflicht', 'obligation')),
            card('SOLLTE', t('Empfehlung', 'recommendation')),
            card('B, S, H', t('Basis-Anforderung, Standard-Anforderung, Anforderung bei erhöhtem Schutzbedarf',
                              'basic requirement, standard requirement, requirement for elevated protection needs')),
        ]),
        hint(t('So steht es im Grundschutz-Kompendium. Macht eine Antwort aus einem SOLLTE ein MUSS, gilt die Aussage als nicht belegt.',
               'This is how the Grundschutz-Kompendium uses them. If an answer turns a SOLLTE into a MUSS, the claim counts as not supported.')),
    ], gap=0.75)
    mo.vstack([
        mo.md(t('## D. Faithfulness: steht das in den Chunks?', '## D. Faithfulness: is it in the chunks?')),
        data_label('kisz'),
        key(t('Faithfulness fragt nicht, ob die Antwort richtig ist, sondern ob sie belegt ist.',
              'Faithfulness does not ask whether the answer is correct, but whether it is supported.')),
        hint(t('Der Richter prüft jede Aussage der Antwort gegen die gefundenen Chunks, nicht gegen die Referenz. Das ist Context Recall aus Notebook 2, nur mit der Antwort statt der Referenzantwort.',
               'The judge checks each claim of the answer against the retrieved chunks, not against the reference. This is Context Recall from notebook 2, only with the answer instead of the reference answer.')),
        tiles([(str(_n), t('Aussagen der Antwort', 'claims in the answer')), (str(_yes), t('davon belegt', 'of them supported')),
               (t.num(B['faithfulness']['score']), 'Faithfulness', f'{_yes} / {_n}')]),
        mo.hstack([
            info(t('Die drei gefundenen Chunks', 'The three retrieved chunks'), chunk_lines(B)),
            mo.vstack([_table, hint(t('Aussagen und Begründungen so, wie RAGAS sie erzeugt hat.',
                                      'Claims and reasons as RAGAS produced them.'))], gap=0.5),
        ], widths=[2, 3], gap=2, align='start'),
        mo.md(t('**Klingt richtig, ist aber zum größten Teil nicht belegt.** <span class="hpi-corpus">"Schulungen und Trainings"</span> und <span class="hpi-corpus">"technische Grundlagen"</span> stehen in keinem der drei Chunks.',
                '**Sounds right, but is mostly not supported.** <span class="hpi-corpus">"Schulungen und Trainings"</span> (courses and trainings) and <span class="hpi-corpus">"technische Grundlagen"</span> (technical foundations) are in none of the three chunks.')),
        hint(t('Der Richter liest genau: Was plausibel klingt, aber nicht dasteht, gilt als nicht belegt.',
               'The judge reads closely: what sounds plausible but is not there counts as not supported.')),
        mo.accordion({t('Im Grundschutz-Kompendium gilt zusätzlich ...', 'In the Grundschutz-Kompendium, there is more ...'): _kompendium}),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Row 9: quiz before the matrix task (question 1 only, section 3a)
    ui_quiz1 = mo.ui.radio(
        {'Answer Correctness': 'ac', 'Faithfulness': 'faith', t('beide', 'both'): 'both'},
        label=t('Die Suche findet falsche Chunks, und die Antwort gibt genau diese Chunks wieder. Was ist hoch?',
                'The search finds the wrong chunks, and the answer reproduces exactly these chunks. What is high?'))
    return (ui_quiz1,)


@app.cell(hide_code=True)
def _(data_label, key, mo, quiz_feedback, t, task, ui_quiz1):
    # Section E, first part: key sentence and the quiz
    mo.vstack([
        mo.md(t('## E. Richtig, belegt, beides, keins?', '## E. Correct, supported, both, neither?')),
        data_label('kisz'),
        key(t('Richtig heißt: passt zur Referenzantwort. Belegt heißt: steht in den gefundenen Chunks.',
              'Correct means: matches the reference answer. Supported means: is in the retrieved chunks.')),
        mo.md(t('### Kurz geprüft', '### Quick check')),
        task('quiz1', '', t(['Beantworten Sie die Frage.', 'Lesen Sie die Rückmeldung darunter.'],
                            ['Answer the question.', 'Read the feedback below it.']),
             t('etwa 1 Minute', 'about 1 minute')),
        ui_quiz1,
        quiz_feedback(ui_quiz1.value, 'faith', t(
            'Die Antwort ist belegt, weil sie wiedergibt, was in den Chunks steht. Richtig ist sie trotzdem nicht.',
            'The answer is supported because it reproduces what is in the chunks. It is still not correct.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(KZ, MATRIX, kisz_card, mo, t, task):
    # Row 10 (3e): two KISZ answers in the matrix; the answer from section B is the comparison in the resolution
    CARDS = KZ['cards']
    ui_cards = mo.ui.array([mo.ui.dropdown(MATRIX, label=t('Ihre Einordnung', 'Your choice')) for _ in CARDS])
    mo.vstack([
        task('matrix', t('Zwei Antworten einordnen', 'Place two answers'), t(
            ['Vergleichen Sie auf jeder Karte die Antwort mit der Referenzantwort: richtig oder nicht?',
             'Prüfen Sie an den drei Chunks, ob die Antwort dort steht: belegt oder nicht?',
             'Wählen Sie unter jeder Karte Ihre Einordnung; die Auflösung erscheint, sobald beide eingeordnet sind.'],
            ['On each card, compare the answer with the reference answer: correct or not?',
             'Check with the three chunks whether the answer is in them: supported or not?',
             'Choose your placement under each card; the solution appears once both are placed.']), t('etwa 3 Minuten', 'about 3 minutes')),
        *[kisz_card(t(f'Antwort {n + 1}', f'Answer {n + 1}'), e, ui_cards[n]) for n, e in enumerate(CARDS)],
    ], gap=1)
    return CARDS, ui_cards


@app.cell(hide_code=True)
def _(B, CARDS, MATRIX, frame, judge_class, mo, t, ui_cards):
    # The solution appears as soon as both answers are placed (meta plan section 10, point 4: choices need no button)
    mo.stop(any(v is None for v in ui_cards.value))

    def _entry(label, e):
        return (f"{label}<br>AC {t.num(e['answer_correctness'])}, Faithfulness {t.num(e['faithfulness_score'])}")

    _cells = {k: [] for k in MATRIX.values()}
    for _n, _e in enumerate(CARDS):
        _cells[judge_class(_e)].append(_entry(t(f'Antwort {_n + 1}', f'Answer {_n + 1}'), _e))
    _cells[judge_class(B)].append(_entry(t('zum Vergleich: die Antwort aus Abschnitt B', 'for comparison: the answer from section B'), B))
    _grid = '\n'.join([
        f"| | {t('richtig (AC ab 0,5)', 'correct (AC from 0.5)')} | {t('falsch (AC unter 0,5)', 'wrong (AC below 0.5)')} |",
        '|:---|:---|:---|',
        f"| **{t('belegt (Faithfulness ab 0,5)', 'supported (Faithfulness from 0.5)')}** | {'<br><br>'.join(_cells['both'])} | {'<br><br>'.join(_cells['supported_only'])} |",
        f"| **{t('nicht belegt (Faithfulness unter 0,5)', 'not supported (Faithfulness below 0.5)')}** | {'<br><br>'.join(_cells['correct_only'])} | {'<br><br>'.join(_cells['neither'])} |",
    ])
    _hits = sum(v == judge_class(e) for v, e in zip(ui_cards.value, CARDS))
    _wrong, _short = CARDS
    mo.vstack([
        mo.md(_grid),
        mo.md(t(f'**Sie haben {_hits} von {len(CARDS)} Antworten so eingeordnet wie der Richter.** Eine feste Grenze gibt es nicht; wir lesen hier unter 0,5 als niedrig, darüber als hoch.',
                f'**You placed {_hits} of {len(CARDS)} answers the way the judge did.** There is no fixed threshold; here we read below 0.5 as low and above as high.')),
        frame(t('Antwort 1: belegt, aber falsch', 'Answer 1: supported, but wrong'), t(
            f"Die Suche hat die Zeilen zu Sprechstunden und Pilotprojekten nicht gefunden (Context Recall {t.num(_wrong['context_recall'])}). Die Antwort gibt treu wieder, was gefunden wurde.",
            f"The search did not find the lines on consultation hours and pilot projects (Context Recall {t.num(_wrong['context_recall'])}). The answer faithfully reproduces what was found.")),
        frame(t('Antwort 2 und die Antwort aus Abschnitt B', 'Answer 2 and the answer from section B'), t(
            f"Beide beantworten dieselbe Frage aus denselben Chunks. Die kurze Antwort ({_short['words_answer']} Wörter) ist richtig und belegt; die ausführliche ({B['words_answer']} Wörter) verliert bei beiden Kennzahlen.",
            f"Both answer the same question from the same chunks. The short answer ({_short['words_answer']} words) is correct and supported; the detailed one ({B['words_answer']} words) loses on both metrics.")),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(Counter, OV, card, cards, data_label, en_of, hint, key, left, md_table, mo, np, pd, plt, t, theme, tiles):
    # Section F (row 11, 3e): all four metrics on the gold set; tiles, figure 1 with two panels, the table folded
    _metrics = [('context_precision', 'Context Precision'), ('context_recall', 'Context Recall'),
                ('answer_correctness', 'Answer Correctness'), ('faithfulness', 'Faithfulness')]
    _df = pd.DataFrame(OV)
    _all_f = float(_df['faithfulness'].mean())  # with the probably misjudged question 22, for the second mean (FAQ)
    _df.loc[_df['not_scorable'], 'faithfulness'] = np.nan
    _cols = {k: f"{n} ({t('Mittel', 'mean')} {t.num(float(_df[k].mean()))})" for k, n in _metrics}
    _cols['faithfulness'] = (f"Faithfulness ({t('Mittel', 'mean')} {t.num(float(_df['faithfulness'].mean()))}, "
                             f"{t('mit Frage 22', 'with question 22')} {t.num(_all_f)})")
    _table = pd.DataFrame({'#': _df['index'] + 1, t('Frage', 'Question'): _df['question']})  # numbered from 1 as in notebooks 1 and 2
    if t.lang == 'en':
        _table['English'] = [en_of(q) or '' for q in _df['question']]
    for _key, _ in _metrics:
        _table[_cols[_key]] = _df[_key].round(2)
    _table[t('Hinweis', 'Note')] = [
        t('Faithfulness vermutlich fehlbewertet, siehe Häufige Fragen', 'Faithfulness probably misjudged, see FAQ') if ns else ''
        for ns in _df['not_scorable']]

    def _fmt(v):
        return '' if v is None or (isinstance(v, float) and np.isnan(v)) else t.num(v)

    _widget = mo.ui.table(
        _table, selection=None, page_size=10, show_download=False, show_column_summaries=False,
        show_data_types=False, format_mapping={c: _fmt for c in _cols.values()},
        wrapped_columns=[c for c in (t('Frage', 'Question'), 'English') if c in _table],
        label=t('Klicken Sie auf eine Spaltenüberschrift, um zu sortieren.', 'Click a column header to sort.'))

    # Figure 1: one point per question in both panels; the raw values incl. question 22 (marked in the cards).
    _shown = [3, 32, 5, 21]
    _q = t('Frage', 'Question')
    _offsets = {'search': {5: (7, 7), 3: (-7, -9), 32: (-9, 8)}, 'answer': {3: (-7, -9), 5: (-7, -9), 32: (6, -11), 21: (7, 7)}}
    _merged = {32: t('Fragen 21 und 32', 'Questions 21 and 32')}  # same point in the search panel

    def _panel(ax, xk, yk, title, which):
        _pts = Counter((round(r[xk], 2), round(r[yk], 2)) for r in OV)
        _xs, _ys = zip(*_pts.keys())
        ax.scatter(_xs, _ys, s=[18 + 14 * (c - 1) for c in _pts.values()], facecolors='white',
                   edgecolors=theme.GREYS[1], linewidths=0.9, zorder=2)
        for _i in _shown:
            ax.scatter([OV[_i][xk]], [OV[_i][yk]], s=24, color='#000000', zorder=3)
        for _i, _off in _offsets[which].items():
            ax.annotate(_merged[_i] if which == 'search' and _i in _merged else f'{_q} {_i}', (OV[_i][xk], OV[_i][yk]),
                        xytext=_off, textcoords='offset points', ha='left' if _off[0] > 0 else 'right',
                        va='center', fontsize=8.5, color=theme.INK)
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: t.num(v, 1)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: t.num(v, 1)))
        ax.set_xlabel(dict(_metrics)[xk])
        ax.set_ylabel(dict(_metrics)[yk])
        ax.set_title(title)

    _fig, _axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    _panel(_axes[0], 'context_recall', 'context_precision', t('Suche', 'Retrieval'), 'search')
    _panel(_axes[1], 'faithfulness', 'answer_correctness', t('Antwort', 'Answer'), 'answer')
    _fig.tight_layout(w_pad=3)

    def _v(i, k):
        return t.num(OV[i][k])

    _mean = t('Mittel über 43 Fragen', 'mean over 43 questions')
    _k, _part, _what = t('Kennzahl', 'Metric'), t('Teil', 'Part'), t('Misst', 'Measures')
    _defs = mo.md(left(md_table([
        {_k: 'Context Precision', _part: t('Suche', 'Retrieval'),
         _what: t('Stehen die nützlichen Chunks oben in der Trefferliste?', 'Are the useful chunks at the top of the hit list?')},
        {_k: 'Context Recall', _part: t('Suche', 'Retrieval'),
         _what: t('Anteil der Aussagen aus der Referenzantwort, die in den gefundenen Chunks stehen.',
                  'Share of the reference claims that are stated in the retrieved chunks.')},
        {_k: 'Answer Correctness', _part: t('Antwort', 'Answer'),
         _what: t('Wie gut passt die Antwort zur Referenz? 0,75 × F1 der Aussagen + 0,25 × Ähnlichkeit der Embeddings.',
                  'How well does the answer match the reference? 0.75 × F1 of the claims + 0.25 × embedding similarity.')},
        {_k: 'Faithfulness', _part: t('Antwort', 'Answer'),
         _what: t('Anteil der Aussagen der Antwort, die in den gefundenen Chunks stehen.',
                  'Share of the answer claims that are stated in the retrieved chunks.')},
    ])))
    mo.vstack([
        mo.md(t('## F. Alle vier Kennzahlen', '## F. All four metrics')),
        data_label('gold'),
        key(t('Bisher einzelne Antworten, jetzt das ganze System: Suche und Antwort scheitern unabhängig voneinander, darum misst man sie getrennt.',
              'So far single answers, now the whole system: retrieval and answer fail independently of each other, which is why they are measured separately.')),
        hint(t('Ab hier messen wir am Gold-Datensatz: 43 Fragen von Fachleuten zum IT-Grundschutz-Kompendium.',
               'From here on we measure on the gold data set: 43 questions by experts about the IT-Grundschutz-Kompendium.')),
        _defs,
        tiles([(t.num(float(_df[k].mean())), n, _mean) for k, n in _metrics[:3]]
              + [(t.num(float(_df['faithfulness'].mean())), 'Faithfulness',
                  t(f'ohne Frage 22; mit Frage 22: {t.num(_all_f)}', f'without question 22; with question 22: {t.num(_all_f)}'))]),
        hint(t('Richter gpt-oss-120b, TOP_K 5', 'Judge gpt-oss-120b, TOP_K 5')),
        _fig,
        hint(t('Abbildung 1. Jeder Kreis ist eine der 43 Fragen, links ihre Suche, rechts ihre Antwort; oben rechts ist gut. Größere Kreise stehen für mehrere Fragen am selben Punkt, schwarz sind die vier Fragen darunter.',
               'Figure 1. Each circle is one of the 43 questions, its retrieval on the left, its answer on the right; top right is good. Larger circles stand for several questions at the same point; the four questions below are black.')),
        cards([
            card(f'{_q} 3', t(f"Suche und Antwort gut (Context Recall {_v(3, 'context_recall')}, Answer Correctness {_v(3, 'answer_correctness')}).",
                              f"Retrieval and answer good (Context Recall {_v(3, 'context_recall')}, Answer Correctness {_v(3, 'answer_correctness')}).")),
            card(f'{_q} 32', t(f"Suche gut (Context Recall {_v(32, 'context_recall')}), die Antwort passt nicht zur Referenz (Answer Correctness {_v(32, 'answer_correctness')}).",
                               f"Retrieval good (Context Recall {_v(32, 'context_recall')}), the answer does not match the reference (Answer Correctness {_v(32, 'answer_correctness')}).")),
            card(f'{_q} 5', t(f"Suche fand fast nichts (Context Recall {_v(5, 'context_recall')}); die Antwort gibt treu wieder, was gefunden wurde (Faithfulness {_v(5, 'faithfulness')}).",
                              f"Retrieval found almost nothing (Context Recall {_v(5, 'context_recall')}); the answer faithfully reproduces what was found (Faithfulness {_v(5, 'faithfulness')}).")),
            card(f'{_q} 21', t('Faithfulness 0, vermutlich fehlbewertet; siehe Häufige Fragen.', 'Faithfulness 0, probably misjudged; see FAQ.')),
        ]),
        mo.accordion({t('Alle 43 Fragen als Tabelle', 'All 43 questions as a table'): _widget}),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(frame, hint, key, left, md_table, mo, t):
    # Section G (new in 3e): what to do when the numbers are off; first check the measurement, then turn a knob
    _b, _k, _s = t('Befund', 'Finding'), t('Stellschraube', 'Lever'), t('gesehen in', 'seen in')
    _rows = [
        {_b: t('Context Recall niedrig', 'Context Recall low'),
         _k: t('TOP_K erhöhen, Chunking prüfen, Suche verbessern (zum Beispiel hybride Suche)', 'raise TOP_K, check the chunking, improve the search (for example hybrid search)'),
         _s: t('Notebook 2, TOP_K', 'notebook 2, TOP_K')},
        {_b: t('Context Precision niedrig', 'Context Precision low'),
         _k: t('Embedding-Modell, Reranking, weniger Chunks', 'embedding model, reranking, fewer chunks'),
         _s: t('Notebook 2, TOP_K und Zusatzaufgabe', 'notebook 2, TOP_K and extra task')},
        {_b: t('Faithfulness niedrig', 'Faithfulness low'),
         _k: t('System-Prompt "nur aus dem Kontext", weniger störende Chunks', 'system prompt "only from the context", fewer distracting chunks'),
         _s: t('Abschnitt I, offener Prompt', 'section I, open prompt')},
        {_b: t('Answer Correctness niedrig, Faithfulness hoch', 'Answer Correctness low, Faithfulness high'),
         _k: t('zuerst Referenzantworten und Antwortlänge prüfen, dann den Prompt', 'first check reference answers and answer length, then the prompt'),
         _s: t('Abschnitte B und C', 'sections B and C')},
    ]
    mo.vstack([
        mo.md(t('## G. Was tun, wenn die Zahlen nicht stimmen?', '## G. What to do when the numbers are off?')),
        key(t('Erst prüfen, ob die Messung stimmt; dann an der passenden Stellschraube drehen.',
              'First check whether the measurement holds; then turn the matching lever.')),
        frame(t('Schritt 1: Stimmt die Messung?', 'Step 1: does the measurement hold?'), t(
            '- Stichprobe gegen die Quelle (Notebook 2; hier Abschnitt D)\n- Streuung des Richters (Notebook 2; hier Abschnitt H)\n- derselbe Richter für alle Vergleiche (Notebook 2)',
            '- spot check against the source (notebook 2; here section D)\n- spread of the judge (notebook 2; here section H)\n- the same judge for all comparisons (notebook 2)')),
        mo.md(t('### Schritt 2: Befund und Stellschraube', '### Step 2: finding and lever')),
        mo.md(left(md_table(_rows))),
        hint(t('Eine Stellschraube probieren Sie gleich selbst aus: den System-Prompt in Abschnitt I.',
               'You try one lever yourself in a moment: the system prompt in section I.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Row 12: quiz before the prompt experiment
    ui_quiz2 = mo.ui.radio(
        {t('ja, um 0,05', 'yes, by 0.05'): 'yes',
         t('nur wenn der Unterschied größer ist als die Streuung des Richters',
           "only if the difference is larger than the judge's spread"): 'spread',
         t('nein, 0,05 ist immer zu wenig', 'no, 0.05 is always too little'): 'no'},
        label=t('Ein neuer Prompt erhöht Answer Correctness auf drei Fragen von 0,40 auf 0,45. Ist der Prompt besser?',
                'A new prompt raises Answer Correctness on three questions from 0.40 to 0.45. Is the prompt better?'))
    return (ui_quiz2,)


@app.cell(hide_code=True)
def _(data_label, hint, key, mo, quiz_feedback, t, task, ui_quiz2):
    # Section H (row 12): spread as the yardstick, then the quiz
    mo.vstack([
        mo.md(t('## H. Wann zählt ein Unterschied?', '## H. When does a difference count?')),
        data_label('gold'),
        key(t('Ein Unterschied zählt erst, wenn er größer ist als die Streuung des Richters.',
              "A difference only counts when it is larger than the judge's spread.")),
        hint(t('An der Antwort kann man drehen, zum Beispiel am System-Prompt. In Notebook 2 hat derselbe Richter dieselbe Frage unterschiedlich bewertet.',
               'You can turn knobs on the answer, for example the system prompt. In notebook 2 the same judge scored the same question differently.')),
        mo.md(t('### Kurz geprüft', '### Quick check')),
        task('quiz2', '', t(['Beantworten Sie die Frage.', 'Lesen Sie die Rückmeldung darunter.'],
                            ['Answer the question.', 'Read the feedback below it.']),
             t('etwa 1 Minute', 'about 1 minute')),
        ui_quiz2,
        quiz_feedback(ui_quiz2.value, 'spread', t(
            '0,05 kann allein daher kommen, dass der Richter beim zweiten Mal anders urteilt.',
            '0.05 can come from the judge alone, scoring differently the second time.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(EXP, mo, t):
    # Row 13: the prompt choice; the text field lives apart so a rebuilt button keeps what was typed
    _added = EXP['prompts']['kurz'][len(EXP['prompts']['original']):].strip()
    PROMPT_LABELS = {
        'original': 'Original',
        # The label shows what the short prompt adds to the original, word for word.
        'kurz': t(f'Kurz: Original plus "{_added}"',
                  'Short: original plus "Answer the question in short sentences. If possible, answer the question in at most 1 to 5 sentences. The shorter and clearer your answer, the better. Only give information that is relevant to the question."'),
        'offen': t('Offen: Ergänze den Kontext um eigenes Expertenwissen.',
                   'Open: add your own expert knowledge to the context.'),
        'own': t('Eigener Prompt', 'Your own prompt'),
    }
    # English renderings of the German prompts, shown in grey under them in the English version
    PROMPT_EN = {
        'original': 'You are a RAG assistant for IT-Grundschutz. Answer the question only on the basis of the provided context. If the information is not contained in the context, say so clearly.',
        'kurz': 'Like the original, plus: Answer the question in short sentences. If possible, answer the question in at most 1 to 5 sentences. The shorter and clearer your answer, the better. Only give information that is relevant to the question.',
        'offen': 'You are an expert in IT security and IT-Grundschutz. Answer the question in detail and comprehensively. Use the provided context as a starting point, but add your expert knowledge to give as complete an answer as possible. Also name best practices and recommendations that go beyond the context.',
    }
    _keys = ['original', 'kurz', 'offen'] + ([] if t.short else ['own'])
    ui_prompt = mo.ui.radio({PROMPT_LABELS[k]: k for k in _keys}, value=PROMPT_LABELS['kurz' if t.short else 'original'],
                            label=t('Welcher Prompt?', 'Which prompt?'))
    ui_own_prompt = mo.ui.text_area(value=EXP['prompts']['original'], rows=4, full_width=True,
                                    label=t('Ihr System-Prompt', 'Your system prompt'))
    return PROMPT_EN, PROMPT_LABELS, ui_own_prompt, ui_prompt


@app.cell(hide_code=True)
def _(mo):
    # Live results of row 13: written by the work cell, rendered by the display cell.
    get_live, set_live = mo.state({'runs': [], 'error': None})
    return get_live, set_live


@app.cell(hide_code=True)
def _(EXP, LIMIT, PROMPT_EN, data_label, get_used, hint, key, mo, t, task, theme, ui_own_prompt, ui_prompt):
    # Section I (row 13): the prompt experiment; the live button is prominent and has no counter in its label
    _spent = get_used() >= LIMIT  # shared with the reference button in section C
    ui_run = mo.ui.run_button(label=theme.cta_label(t('Alle drei Live-Läufe genutzt', 'All three live runs used') if _spent
                                                    else t('Eigenen Prompt testen', 'Test your own prompt')),
                              disabled=_spent, kind='success')

    def _prompt_view(k):
        _en = PROMPT_EN.get(k) if t.lang == 'en' else None
        _box = (f'<div class="hpi-corpus" style="border:1px solid {theme.RULE};border-radius:2px;padding:.6rem .9rem;background:#fff">'
                f'{EXP["prompts"][k]}</div>')
        return mo.vstack([mo.Html(_box), *([hint(_en)] if _en else [])], gap=0.25)

    _own = ui_prompt.value == 'own' and not t.short
    _steps = t.full(t(
        ['Lesen Sie den Original-Prompt.', 'Schauen Sie sich zuerst den kurzen und den offenen Prompt an.',
         "Wählen Sie dann 'Eigener Prompt', ändern Sie den Text und drücken Sie auf 'Eigenen Prompt testen'.",
         'Lesen Sie in Abbildung 2 ab, ob Ihr Ergebnis außerhalb des grauen Bereichs liegt, und antworten Sie unten.'],
        ['Read the original prompt.', 'First look at the short and the open prompt.',
         "Then choose 'Your own prompt', change the text and press 'Test your own prompt'.",
         'Read off figure 2 whether your result lies outside the grey band, and answer below.']), t(
        ['Lesen Sie den Original-Prompt.', 'Der kurze Prompt ist schon gewählt; vergleichen Sie danach den offenen.',
         'Lesen Sie in Abbildung 2 ab, ob das Ergebnis außerhalb des grauen Bereichs liegt, und antworten Sie unten.'],
        ['Read the original prompt.', 'The short prompt is already selected; compare the open one afterwards.',
         'Read off figure 2 whether the result lies outside the grey band, and answer below.']))
    mo.vstack([
        mo.md(t('## I. Ein eigener System-Prompt', '## I. A system prompt of your own')),
        data_label('gold'),
        key(t('Jetzt Ihr Prompt, mit der Streuung des Richters als Maßstab.', "Now your prompt, with the judge's spread as the yardstick.")),
        hint(t('Der System-Prompt sagt dem Antwortmodell, wie es antworten soll. Alle Antworten zum Gold-Datensatz entstanden mit dem Original-Prompt.',
               'The system prompt tells the answer model how to answer. All answers on the gold data set were produced with the original prompt.')),
        task('prompt', t('Einen Prompt prüfen', 'Check a prompt'), _steps,
             t('etwa 4 Minuten', 'about 4 minutes') if t.short else t('etwa 7 Minuten', 'about 7 minutes')),
        mo.md(f"**{t('Original-Prompt', 'Original prompt')}**"), _prompt_view('original'),
        ui_prompt,
        *([_prompt_view(ui_prompt.value)] if ui_prompt.value in ('kurz', 'offen') else []),
        *([ui_own_prompt, mo.Html(f'<div class="hpi-cta">{ui_run}</div>'),
           hint(t('Sie können bis zu dreimal live testen lassen, gemeinsam mit der Bewertung Ihrer Referenz in Abschnitt C. Ein Lauf heißt hier drei Antworten und sechs Bewertungen und dauert ein bis zwei Minuten.',
                  'You can have up to three live runs, shared with the scoring of your reference in section C. A run here means three answers and six scores and takes one to two minutes.'))] if _own else []),
    ], gap=1)
    return (ui_run,)


@app.cell(hide_code=True)
async def _(EXP, LIMIT, gen, get_used, mo, set_live, set_used, t, ui_own_prompt, ui_prompt, ui_run):
    mo.stop(not ui_run.value or ui_prompt.value != 'own' or get_used() >= LIMIT)
    _prompt = ui_own_prompt.value.strip()
    if not _prompt:
        set_live(lambda s: {**s, 'error': 'empty'})
    else:
        _rows = [{'question': q['question'], 'reference': q['reference'], 'contexts': [c['text'] for c in q['chunks']]}
                 for q in EXP['questions']]
        try:
            with mo.status.spinner(title=t('Antworten und Bewertungen laufen ...', 'Answers and scores are running ...'),
                                   subtitle=t('Frage 1 von 3', 'Question 1 of 3')) as _sp:
                _out = await gen.arun_prompt(_prompt, _rows, progress=lambda i: _sp.update(
                    subtitle=t(f'Frage {min(i + 1, 3)} von 3', f'Question {min(i + 1, 3)} of 3')))
        except Exception:  # noqa: BLE001 - any API failure gets the same sentence
            set_live(lambda s: {**s, 'error': {'prompt': _prompt}})
        else:
            set_live(lambda s: {'runs': s['runs'] + [{'prompt': _prompt, 'rows': _out}], 'error': None})
            set_used(lambda n: n + 1)
    return


@app.cell(hide_code=True)
def _(EXP, LLM_DOWN, PROMPT_LABELS, bi, get_live, hint, left, md_table, mo, np, plt, t, theme, ui_prompt):
    # Row 13, result: figure 2 with the judge's spread per question, the numbers, the answers folded
    _res = EXP['results']
    _qs = EXP['questions']

    def _median_runs(rows, key):
        """The median of the three judge runs per question."""
        return [float(np.median([r[key] for r in row['runs']])) for row in rows]

    def _nearest(prompt):
        """The prepared prompt with the largest word overlap with the own one."""
        _w = set(prompt.lower().split())
        return max(EXP['prompts'], key=lambda k: len(_w & set(EXP['prompts'][k].lower().split()))
                   / max(1, len(_w | set(EXP['prompts'][k].lower().split()))))

    _note, _sel, _answers, _label = None, None, None, PROMPT_LABELS.get(ui_prompt.value, '')
    _live = get_live()
    if ui_prompt.value == 'own' and not t.short:
        if isinstance(_live['error'], dict):
            _k = _nearest(_live['error']['prompt'])
            _note = mo.callout(mo.md(t(*LLM_DOWN) + ' ' + t(
                f'Bis dahin sehen Sie das Ergebnis des vorbereiteten Prompts, der Ihrem am nächsten kommt: *{PROMPT_LABELS[_k].split(":")[0]}*.',
                f'Until then you see the result of the prepared prompt closest to yours: *{PROMPT_LABELS[_k].split(":")[0]}*.')), kind='warn')
            # the original is already drawn as the hollow dot; a second marker would only repeat it
            _sel = None if _k == 'original' else {'ac': _median_runs(_res[_k], 'ac'), 'faithfulness': _median_runs(_res[_k], 'faithfulness')}
            _answers, _label = [r['answer'] for r in _res[_k]], PROMPT_LABELS[_k]
        elif _live['error'] == 'empty':
            _note = mo.callout(mo.md(t('Schreiben Sie zuerst einen Prompt.', 'Write a prompt first.')), kind='warn')
        elif _live['runs']:
            _run = _live['runs'][-1]['rows']
            _sel = {'ac': [r['ac'] for r in _run], 'faithfulness': [r['faithfulness'] for r in _run]}
            _answers = [r['answer'] for r in _run]
            _label = t('Ihr Prompt (ein Lauf)', 'Your prompt (one run)')
        else:
            _note = hint(t("Schreiben Sie einen Prompt und drücken Sie auf 'Eigenen Prompt testen'.", "Write a prompt and press 'Test your own prompt'."))
    elif ui_prompt.value in ('kurz', 'offen'):
        _sel = {'ac': _median_runs(_res[ui_prompt.value], 'ac'),
                'faithfulness': _median_runs(_res[ui_prompt.value], 'faithfulness')}
        _answers = [r['answer'] for r in _res[ui_prompt.value]]

    _orig = {k: _median_runs(_res['original'], k) for k in ('ac', 'faithfulness')}
    _band = {k: [(min(r[k] for r in row['runs']), max(r[k] for r in row['runs'])) for row in _res['original']]
             for k in ('ac', 'faithfulness')}
    _names = [t(f"Frage {q['index'] + 1}", f"Question {q['index'] + 1}") for q in _qs]

    _fig, _axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    for _ax, (_k, _title) in zip(_axes, [('ac', 'Answer Correctness'), ('faithfulness', 'Faithfulness')]):
        for _y, (_lo, _hi) in enumerate(_band[_k]):
            _ax.add_patch(plt.Rectangle((_lo, _y - 0.28), max(_hi - _lo, 0.008), 0.56, color=theme.RULE, lw=0))
        _ax.scatter(_orig[_k], range(3), s=34, facecolors='white', edgecolors=theme.GREYS[1], linewidths=1.1,
                    zorder=3, label=t('Original, Median aus drei Läufen', 'original, median of three runs'))
        if _sel is not None:
            _name = _label.split(':')[0]
            _ax.scatter(_sel[_k], range(3), s=34, color='#000000', zorder=4,
                        label=_name if '(' in _name else _name + t(', Median aus drei Läufen', ', median of three runs'))
        _ax.set_xlim(0, 1.02)
        _ax.set_title(_title)
        _ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: t.num(v, 1)))
    _axes[0].set_yticks(range(3), _names)
    _axes[0].invert_yaxis()
    _handles = [plt.Rectangle((0, 0), 1, 1, color=theme.RULE, lw=0), *_axes[0].collections]
    _fig.legend(_handles, [t('Streuung des Richters, Original', "judge's spread, original"),
                           *[c.get_label() for c in _axes[0].collections]],
                loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0))
    _fig.tight_layout(rect=(0, 0.12, 1, 1))

    _rows = []
    for _i, _n in enumerate(_names):
        _row = {t('Frage', 'Question'): _n}
        for _k, _short in (('ac', 'AC'), ('faithfulness', 'Faithfulness')):
            _lo, _hi = _band[_k][_i]
            _row[f"{_short}, {t('grauer Bereich', 'grey band')}"] = f'{t.num(_lo)} {t("bis", "to")} {t.num(_hi)}'
            _row[f"{_short}, {t('gewählt', 'chosen')}"] = t.num(_sel[_k][_i]) if _sel is not None else ''
        _rows.append(_row)

    _answers_view = None if _answers is None else mo.accordion({
        t('Die drei Antworten', 'The three answers'): mo.vstack([
            mo.vstack([mo.md(f"#### {_n}"), bi(q['question']), bi(a)])
            for _n, q, a in zip(_names, _qs, _answers)], gap=1.5),
        t('Fragen, Referenzantworten und Fundstellen', 'Questions, reference answers and gold passages'): mo.vstack([
            mo.vstack([mo.md(f"#### {_n}"), bi(q['question']), mo.md(f"*{t('Referenzantwort', 'Reference answer')}*"),
                       bi(q['reference']), mo.md(f"*{t('Fundstelle', 'Gold passage')}*"), bi(q['fundstelle'])])
            for _n, q in zip(_names, _qs)], gap=1.5),
    })
    _own_table = None
    if len(_live['runs']) > 1 and ui_prompt.value == 'own':
        _own_table = mo.md(left(md_table([
            {t('Lauf', 'Run'): str(n + 1), 'Prompt': f"<span class=\"hpi-corpus\">{r['prompt'].replace(chr(10), ' ').replace('|', '/')[:120]}</span>",
             t('AC im Mittel', 'mean AC'): t.num(float(np.mean([x['ac'] for x in r['rows']]))),
             t('Faithfulness im Mittel', 'mean Faithfulness'): t.num(float(np.mean([x['faithfulness'] for x in r['rows']])))}
            for n, r in enumerate(_live['runs'])])))
    mo.vstack([v for v in (
        _note, _fig,
        hint(t('Abbildung 2. Der graue Bereich reicht vom niedrigsten bis zum höchsten Wert aus drei Läufen des Richters auf denselben Antworten des Original-Prompts. Jeder Punkt ist der Median der drei Läufe zu einer Frage; bei einem eigenen Prompt ist der schwarze Punkt das Ergebnis eines Laufs.',
               'Figure 2. The grey band reaches from the lowest to the highest value of three judge runs on the same answers of the original prompt. Each dot is the median of the three runs for one question; for a prompt of your own, the black dot is the result of one run.')),
        mo.md(left(md_table(_rows))),
        hint(t('Der graue Bereich zeigt nur, wie stark der Richter schwankt. Würde das Modell noch einmal antworten, käme auch eine andere Antwort heraus; der echte Spielraum ist also noch größer.',
               'The grey band only shows how much the judge varies. If the model answered again, a different answer would come out too; the real margin is therefore even larger.')),
        _own_table, _answers_view,
    ) if v is not None], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Row 14: reflect, the last step of the prompt task
    ui_outside = mo.ui.radio({t('ja, deutlich', 'yes, clearly'): 'clear', t('knapp', 'barely'): 'barely',
                              t('nein', 'no'): 'no'}, inline=True,
                             label=t('Liegt Ihr Ergebnis außerhalb des grauen Bereichs?',
                                     'Is your result outside the grey band?'))
    ui_outside
    return (ui_outside,)


@app.cell(hide_code=True)
def _(EXP, frame, mo, np, quiz_feedback, t, ui_outside):
    # Row 14, resolution from the cached runs (medians against the band); short version: one paragraph
    mo.stop(ui_outside.value is None)
    _res, _qs = EXP['results'], EXP['questions']
    _ids = [q['index'] + 1 for q in _qs]  # shown numbers, from 1

    def _where(name, key):
        """Questions whose median lies below and above the band of the original prompt."""
        _lo, _hi = [], []
        for i, row, orig in zip(_ids, _res[name], _res['original']):
            _m = float(np.median([r[key] for r in row['runs']]))
            _band = [r[key] for r in orig['runs']]
            (_lo if _m < min(_band) else _hi if _m > max(_band) else []).append(i)
        return _lo, _hi

    def _qlist(ids):
        _and = t(' und ', ' and ')
        _n = str(ids[0]) if len(ids) == 1 else ', '.join(map(str, ids[:-1])) + _and + str(ids[-1])
        return t(f'Frage {_n}' if len(ids) == 1 else f'den Fragen {_n}', f'question {_n}' if len(ids) == 1 else f'questions {_n}')

    _k_ac, _k_f, _o_ac, _o_f = _where('kurz', 'ac'), _where('kurz', 'faithfulness'), _where('offen', 'ac'), _where('offen', 'faithfulness')
    _p1 = [t('Der kurze Prompt ändert Answer Correctness kaum: ', 'The short prompt hardly changes Answer Correctness: ')
           + (t('Bei allen drei Fragen bleibt der Median im grauen Bereich.', 'for all three questions the median stays inside the grey band.')
              if not any(_k_ac) else t('nur bei ', 'only for ') + _qlist(_k_ac[0] + _k_ac[1]) + t(' liegt er außerhalb.', ' does it fall outside.'))]
    if _k_f[0]:
        _p1.append(t(f'Die Faithfulness senkt er bei {_qlist(_k_f[0])} unter den grauen Bereich',
                     f'It lowers Faithfulness below the grey band for {_qlist(_k_f[0])}')
                   + (t(f', bei {_qlist(_k_f[1])} liegt sie knapp darüber.', f'; for {_qlist(_k_f[1])} it is slightly above.') if _k_f[1] else '.'))
    _p1.append(t(f'Der offene Prompt senkt die Faithfulness bei {_qlist(_o_f[0])} deutlich, weil das Modell Wissen ergänzt, das nicht in den Chunks steht, und Answer Correctness bei {_qlist(_o_ac[0])}.',
                 f'The open prompt lowers Faithfulness clearly for {_qlist(_o_f[0])}, because the model adds knowledge that is not in the chunks, and Answer Correctness for {_qlist(_o_ac[0])}.'))
    # The open answer on which the judge disagreed with itself most (Faithfulness, three runs).
    _q, _row = max(zip(_qs, _res['offen']),
                   key=lambda p: max(r['faithfulness'] for r in p[1]['runs']) - min(r['faithfulness'] for r in p[1]['runs']))
    _fs = ', '.join(t.num(r['faithfulness']) for r in _row['runs'])
    _top = t.num(max(r['faithfulness'] for r in _row['runs']))
    _p1.append(t(f"Die offene Antwort zu Frage {_q['index'] + 1} bekam in drei Läufen {_fs}; die {_top} ist ein Ausreißer des Richters.",
                 f"The open answer to question {_q['index'] + 1} got {_fs} in three runs; the {_top} is an outlier of the judge."))
    _c = EXP['claims']
    _tp = {k: sum(x['TP'] for x in _c[k]) for k in ('original', 'kurz')}
    _fp = {k: sum(x['FP'] for x in _c[k]) for k in ('original', 'kurz')}
    _p2 = t(
        f"**Warum Kürze hier nicht hilft:** Kürzere Antworten haben weniger Aussagen zu viel, aber auch weniger Treffer; F1 ändert sich dann kaum. In einer Zerlegung der drei Antworten hatten die kurzen zusammen {_fp['kurz']} statt {_fp['original']} Aussagen zu viel, aber auch {_tp['kurz']} statt {_tp['original']} Treffer. Bei drei Fragen bleibt jedes Ergebnis eine Tendenz; für eine Entscheidung braucht man alle 43, und ob die Antworten besser geworden sind, sagt Ihnen erst Ihr eigenes Urteil.",
        f"**Why brevity does not help here:** shorter answers have fewer claims too many, but also fewer hits; F1 then hardly changes. In one decomposition of the three answers, the short ones had {_fp['kurz']} instead of {_fp['original']} claims too many in total, but also {_tp['kurz']} instead of {_tp['original']} hits. With three questions every result remains a tendency; a decision needs all 43, and only your own judgment tells you whether the answers got better.")
    mo.vstack([
        quiz_feedback(ui_outside.value, None, t('Das hängt von Ihrem Prompt ab; maßgeblich ist der graue Bereich in Abbildung 2.',
                                                'That depends on your prompt; what counts is the grey band in figure 2.')),
        frame(t('Was die vorbereiteten Prompts zeigen', 'What the prepared prompts show'),
              '\n'.join(f'- {x}' for x in [_p1[0], ' '.join(_p1[1:-2]) if len(_p1) > 3 else '', _p1[-2], _p1[-1]] if x)
              + ('' if t.short else '\n\n' + _p2)),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(KZ, MATRIX, OV, bi, kisz_card, mo, t, task, theme, web_view):
    # Row 15 (normal only): extra task in one accordion; feedback renders below it,
    # so the accordion is never redrawn while someone works in it. Parts a and d on the KISZ website, c on the gold set.
    LOWER_RIGHT = [r for r in OV if r['context_recall'] >= 0.5 and r['answer_correctness'] < 0.5]
    _judge = {t('richtig und vollständig', 'correct and complete'): 'full',
              t('teilweise richtig', 'partly correct'): 'partly', t('falsch', 'wrong'): 'wrong'}
    ui_extra_quiz = mo.ui.radio(
        {t('Richtigkeit, gegen die Quelle auf der Webseite', 'correctness, against the source on the website'): 'correct',
         t('Belegtheit, gegen die Chunks', 'support, against the chunks'): 'supported',
         t('beides', 'both'): 'both'},
        label=t('Was bewerten Sie in Teil a, Richtigkeit oder Belegtheit?', 'What do you judge in part a, correctness or support?'))
    ui_extra_judge = mo.ui.array([mo.ui.radio(_judge, inline=True) for _ in KZ['extra_a']])
    ui_extra_why = mo.ui.text_area(label=t('Wo weichen Sie vom Richter ab, und warum?', 'Where do you differ from the judge, and why?'),
                                   rows=3, full_width=True)
    ui_why_ok = mo.ui.run_button(label=theme.cta_label(t('Begründung übernehmen', 'Use this reasoning')))
    ui_cause_q = mo.ui.dropdown({f"{r['index'] + 1}: {r['question']}": r['index'] for r in LOWER_RIGHT},
                                label=t('Frage mit guter Suche und niedriger Answer Correctness', 'Question with good retrieval and low Answer Correctness'),
                                full_width=True)
    ui_cause = mo.ui.radio({t('Referenz sehr kurz', 'reference very short'): 'short',
                            t('Antwort erfindet etwas', 'answer invents something'): 'invent',
                            t('Suche fand zu wenig', 'search found too little'): 'search',
                            t('Richter schwankt', 'judge varies'): 'judge'},
                           label=t('Was ist die Ursache?', 'What is the cause?'))
    ui_card_d = mo.ui.dropdown(MATRIX, label=t('Ihre Einordnung', 'Your choice'))

    def _part_a(n, e):
        return mo.vstack([
            mo.md(f"**{t('Frage', 'Question')}**"), bi(e['question']),
            mo.hstack([web_view(e),
                       mo.vstack([mo.md(f"**{t('Antwort der Pipeline', 'Answer of the pipeline')}**"), bi(e['answer'])])],
                      widths='equal', gap=2, align='start'),
            ui_extra_judge[n],
        ])

    t.full(mo.vstack([
        task('extra', '', t(
            ['Öffnen Sie den Kasten "Zusatzaufgabe" darunter.', 'Wählen Sie einen Teil (a bis d) und bearbeiten Sie ihn.',
             'Rückmeldungen finden Sie unter dem Kasten.', 'Alle anderen machen mit Abschnitt J weiter.'],
            ['Open the box "Extra task" below.', 'Pick a part (a to d) and work on it.',
             'Look for feedback below the box.', 'Everyone else continues with section J.']),
            t('so lange, bis die anderen fertig sind', 'until the others are done')),
        mo.accordion({t('Zusatzaufgabe', 'Extra task'): mo.vstack([
            ui_extra_quiz,
            mo.md(t('#### Teil a: zwei Antworten selbst bewerten\n\nBewerten Sie zwei weitere Antworten zur Webseite anhand der Quelle. Ist die Antwort fachlich richtig?',
                    '#### Part a: judge two answers yourself\n\nJudge two more answers about the website against the source. Is the answer correct on the subject matter?')),
            *[_part_a(n, e) for n, e in enumerate(KZ['extra_a'])],
            ui_extra_why,
            mo.Html(f'<div class="hpi-cta">{ui_why_ok}</div>'),
            mo.md(t('#### Teil b (optional): ein besserer Prompt\n\nFinden Sie einen Prompt, der Answer Correctness hebt, ohne Faithfulness zu senken. Nutzen Sie dafür "Eigener Prompt" in Abschnitt I; Ihre Läufe stehen dort in einer Tabelle.',
                    '#### Part b (optional): a better prompt\n\nFind a prompt that raises Answer Correctness without lowering Faithfulness. Use "Your own prompt" in section I; your runs appear there in a table.')),
            mo.md(t('#### Teil c: die Ursache finden\n\nWählen Sie eine Frage aus dem Gold-Datensatz, bei der die Suche viel gefunden hat (Context Recall ab 0,5) und Answer Correctness trotzdem unter 0,5 liegt. Was ist die Ursache? Werte, Fundstelle und Antwort erscheinen unter dem Kasten.',
                    '#### Part c: find the cause\n\nChoose a question from the gold data set where the search found a lot (Context Recall from 0.5) and Answer Correctness is still below 0.5. What is the cause? Values, gold passage and answer appear below the box.')),
            ui_cause_q, ui_cause,
            mo.md(t('#### Teil d: eine dritte Antwort einordnen\n\nOrdnen Sie eine dritte Antwort ein, wie in Abschnitt E. Die Auflösung erscheint unter dem Kasten.',
                    '#### Part d: place a third answer\n\nPlace a third answer, as in section E. The resolution appears below the box.')),
            kisz_card(t('Antwort 3', 'Answer 3'), KZ['extra_d'], ui_card_d),
        ])}),
    ], gap=1))
    return ui_card_d, ui_cause, ui_cause_q, ui_extra_judge, ui_extra_quiz, ui_extra_why, ui_why_ok


@app.cell(hide_code=True)
def _(mo):
    # The confirmed reasoning of extra task part a; session only, never cached
    get_why, set_why = mo.state('')
    return get_why, set_why


@app.cell(hide_code=True)
def _(mo, set_why, ui_extra_why, ui_why_ok):
    # Takes over the reasoning on "Begründung übernehmen"; typing alone shows nothing new
    mo.stop(not ui_why_ok.value)
    set_why(ui_extra_why.value.strip())
    return


@app.cell(hide_code=True)
def _(EXP, GOLD, KZ, MATRIX, OV, SRC, bi, flat, frame, fund_view, get_why, hint, info, judge_class, mo, nwords, quiz_feedback, t, tiles, ui_card_d, ui_cause, ui_cause_q, ui_extra_judge, ui_extra_quiz):
    mo.stop(t.short)
    _parts = [quiz_feedback(ui_extra_quiz.value, 'correct', t(
        'Sie vergleichen die Antwort mit der Quelle auf der Webseite, also bewerten Sie die Richtigkeit. Ob die Antwort belegt ist, prüft Faithfulness gegen die Chunks.',
        'You compare the answer with the source on the website, so you judge correctness. Whether the answer is supported is what Faithfulness checks against the chunks.'))]
    if all(v is not None for v in ui_extra_judge.value):
        _lab = {'full': t('richtig und vollständig', 'correct and complete'), 'partly': t('teilweise richtig', 'partly correct'),
                'wrong': t('falsch', 'wrong')}
        _parts.append(mo.md('\n'.join([
            f"| {t('Frage', 'Question')} | {t('Ihr Urteil', 'Your judgment')} | Answer Correctness | Faithfulness |",
            '|:---|:---|:---|:---|',
            *[f"| <span class=\"hpi-corpus\">{flat(e['question'])}</span> | {_lab[v]} | {t.num(e['answer_correctness'])} | {t.num(e['faithfulness_score'])} |"
              for e, v in zip(KZ['extra_a'], ui_extra_judge.value)],
        ])))
        if get_why():
            _parts.append(info(t('Ihre Begründung', 'Your reasoning'), mo.md(get_why())))
        else:
            _parts.append(hint(t('Schreiben Sie im Kasten auf, wo Sie vom Richter abweichen, und warum, und drücken Sie auf "Begründung übernehmen".',
                                 'Write down in the box where you differ from the judge, and why, and press "Use this reasoning".')))
    if ui_cause_q.value is not None:
        _i = ui_cause_q.value
        _r = next(r for r in OV if r['index'] == _i)
        _parts += [
            mo.md(f"#### {t('Teil c', 'Part c')}: {t('Frage', 'question')} {_i}"), bi(GOLD[_i]['question']),
            tiles([(t.num(_r['context_precision']), 'Context Precision'), (t.num(_r['context_recall']), 'Context Recall'),
                   (t.num(_r['answer_correctness']), 'Answer Correctness'),
                   ('' if _r['not_scorable'] else t.num(_r['faithfulness']), 'Faithfulness')]),
            mo.hstack([
                mo.vstack([mo.md(f"**{t('Referenzantwort', 'Reference answer')}**"), bi(GOLD[_i]['answer']), fund_view(GOLD[_i]['fundstelle'])]),
                mo.vstack([mo.md(f"**{t('Antwort der Pipeline', 'Answer of the pipeline')}**"), bi(SRC['answers'][_i])]),
            ], widths='equal', gap=2, align='start'),
        ]
        if ui_cause.value is not None:
            _wa, _wr = nwords(SRC['answers'][_i]), nwords(GOLD[_i]['answer'])
            _spread = max(max(r['ac'] for r in row['runs']) - min(r['ac'] for r in row['runs'])
                          for row in EXP['results']['original'])
            _lines = [t(f'Die Antwort hat {_wa} Wörter, die Referenz {_wr}.', f'The answer has {_wa} words, the reference {_wr}.')
                      + (' ' + t('Die Referenz ist sehr kurz; viele Aussagen der Antwort zählen als FP.',
                                 'The reference is very short; many claims of the answer count as FP.') if _wa > 5 * _wr else ''),
                      t(f"Context Recall {t.num(_r['context_recall'])}: Die Suche hat die Aussagen der Referenz zum größten Teil gefunden. An der Suche liegt es eher nicht.",
                        f"Context Recall {t.num(_r['context_recall'])}: the search found most of the claims of the reference. The search is probably not the cause.")]
            if _r['not_scorable']:
                _lines.append(t('Faithfulness ist hier vermutlich fehlbewertet, siehe Häufige Fragen.', 'Faithfulness is probably misjudged here, see FAQ.'))
            elif _r['faithfulness'] < 0.5:
                _lines.append(t(f"Faithfulness {t.num(_r['faithfulness'])}: Vieles in der Antwort steht nicht in den Chunks. Prüfen Sie, ob die Antwort etwas erfindet.",
                                f"Faithfulness {t.num(_r['faithfulness'])}: much of the answer is not in the chunks. Check whether the answer invents something."))
            else:
                _lines.append(t(f"Faithfulness {t.num(_r['faithfulness'])}: Die Antwort hält sich an die Chunks.",
                                f"Faithfulness {t.num(_r['faithfulness'])}: the answer sticks to the chunks."))
            _lines.append(t(f'Ob der Richter schwankt, zeigt erst ein zweiter Lauf. Im Prompt-Experiment in Abschnitt I lag Answer Correctness bei derselben Antwort bis zu {t.num(_spread)} auseinander.',
                            f'Whether the judge varies only shows in a second run. In the prompt experiment in section I, Answer Correctness for the same answer was up to {t.num(_spread)} apart.'))
            _parts.append(frame(t('Einordnung aus den Daten', 'What the data say'), '\n\n'.join(_lines)))
    if ui_card_d.value is not None:
        _d = KZ['extra_d']
        _label = {v: k for k, v in MATRIX.items()}
        _parts.append(frame(t(f"Teil d: {_label[judge_class(_d)]}", f"Part d: {_label[judge_class(_d)]}"), t(
            f"Sie haben *{_label[ui_card_d.value]}* gewählt, der Richter landet bei *{_label[judge_class(_d)]}* (Answer Correctness {t.num(_d['answer_correctness'])}, Faithfulness {t.num(_d['faithfulness_score'])}).\n\nAnders als bei Antwort 1 in Abschnitt E hat die Suche die richtige Zeile gefunden (Context Recall {t.num(_d['context_recall'])}). Die Antwort liest sie aber anders als die Fachleute: \"Ja\" statt \"Nein\". Weil sie sich dabei auf die Chunks stützt, gilt sie als belegt.",
            f"You chose *{_label[ui_card_d.value]}*, the judge ends up at *{_label[judge_class(_d)]}* (Answer Correctness {t.num(_d['answer_correctness'])}, Faithfulness {t.num(_d['faithfulness_score'])}).\n\nUnlike answer 1 in section E, the search found the right line (Context Recall {t.num(_d['context_recall'])}). The answer reads it differently from the experts, though: \"Ja\" (yes) instead of \"Nein\" (no). Because it relies on the chunks while doing so, it counts as supported.",
        )))
    mo.vstack(_parts, gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Section J (3e): final quiz over part III; six questions, three in the short version
    FINAL = [
        {"key": 'data', "short": False, "right": 'drafts',
             "q": t('Wofür taugen Testdaten, die RAGAS erzeugt?', 'What are test data generated by RAGAS good for?'),
             "options": {t('als alleinige Grundlage der Bewertung', 'as the sole basis of the evaluation'): 'sole',
                      t('als Entwürfe, die Fachleute prüfen und korrigieren', 'as drafts that experts check and correct'): 'drafts',
                      t('für gar nichts', 'for nothing at all'): 'none'},
             "fb": t('Ein Gold-Datensatz von Fachleuten bleibt das Fundament; RAGAS liefert schnell viele Entwürfe. Notebook 1, Abschnitt E.',
                  'A gold data set by experts remains the foundation; RAGAS quickly delivers many drafts. Notebook 1, section E.')},
        {"key": 'needs', "short": False, "right": 'reference',
             "q": t('Was braucht der Richter zu jeder Testfrage?', 'What does the judge need for each test question?'),
             "options": {t('die markierte Fundstelle', 'the marked gold passage'): 'passage',
                      t('eine Referenzantwort', 'a reference answer'): 'reference',
                      t('nichts, er kennt die Antwort selbst', 'nothing, it knows the answer itself'): 'nothing'},
             "fb": t('Gegen die Referenzantwort urteilt er bei drei der vier Kennzahlen; nur Faithfulness kommt ohne sie aus. Notebook 1, Abschnitte C und E.',
                  'It judges against the reference answer for three of the four metrics; only Faithfulness does without it. Notebook 1, sections C and E.')},
        {"key": 'cpcr', "short": True, "right": 'cp',
             "q": t('Die Suche findet alle nötigen Stellen, aber erst hinter mehreren nutzlosen Chunks. Welche Kennzahl ist niedrig?',
                 'The search finds all the passages needed, but only behind several useless chunks. Which metric is low?'),
             "options": {'Context Precision': 'cp', 'Context Recall': 'cr', t('beide', 'both'): 'both'},
             "fb": t('Context Recall fragt, ob alles gefunden wurde; Context Precision, ob die nützlichen Chunks oben stehen. Notebook 2.',
                  'Context Recall asks whether everything was found; Context Precision, whether the useful chunks are at the top. Notebook 2.')},
        {"key": 'acf', "short": True, "right": 'ac',
             "q": t('Eine Antwort stimmt mit der Referenz überein, aber nichts davon steht in den gefundenen Chunks. Was ist hoch?',
                 'An answer agrees with the reference, but none of it is in the retrieved chunks. What is high?'),
             "options": {'Answer Correctness': 'ac', 'Faithfulness': 'faith', t('beide', 'both'): 'both'},
             "fb": t('Answer Correctness vergleicht mit der Referenz, Faithfulness mit den Chunks. Notebook 3, Abschnitte C bis E.',
                  'Answer Correctness compares with the reference, Faithfulness with the chunks. Notebook 3, sections C to E.')},
        {"key": 'judge', "short": False, "right": 'measure',
             "q": t('Mit einem anderen Richter steigen alle Werte. Was hat sich verbessert?',
                 'With a different judge, all values go up. What has improved?'),
             "options": {t('das System', 'the system'): 'system',
                      t('nichts am System; nur die Messung ist anders', 'nothing in the system; only the measurement differs'): 'measure',
                      t('beides', 'both'): 'both'},
             "fb": t('Ein anderer Richter ändert die Messung, nicht das System; vergleichen Sie nur Werte desselben Richters. Auch derselbe Richter streut, darum zählt ein Unterschied erst jenseits der Streuung. Notebook 2 und Notebook 3, Abschnitt H.',
                  'A different judge changes the measurement, not the system; compare only values of the same judge. Even the same judge varies, so a difference only counts beyond the spread. Notebook 2 and notebook 3, section H.')},
        {"key": 'low_cr', "short": True, "right": 'check',
             "q": t('Context Recall ist niedrig. Was tun Sie zuerst?', 'Context Recall is low. What do you do first?'),
             "options": {t('den System-Prompt ändern', 'change the system prompt'): 'prompt',
                      t('prüfen, ob die Messung stimmt, dann TOP_K erhöhen oder das Chunking prüfen', 'check whether the measurement holds, then raise TOP_K or check the chunking'): 'check',
                      t('den Richter wechseln', 'switch the judge'): 'judge'},
             "fb": t('Erst die Messung prüfen, dann an der Suche drehen; der Prompt ändert nur die Antwort. Notebook 3, Abschnitt G, und Notebook 2 zu TOP_K.',
                  'First check the measurement, then work on the search; the prompt changes only the answer. Notebook 3, section G, and notebook 2 on TOP_K.')},
    ]
    FINAL = [f for f in FINAL if f['short'] or not t.short]
    ui_final = mo.ui.array([mo.ui.radio(f['options'], label=f"**{n + 1}.** {f['q']}") for n, f in enumerate(FINAL)])
    return FINAL, ui_final


@app.cell(hide_code=True)
def _(FINAL, key, mo, quiz_feedback, t, task, ui_final):
    # Section J: the quiz as a counted task; feedback directly under each question, the count once all are answered
    _done = [v for v in ui_final.value if v is not None]
    _right = sum(v == f['right'] for v, f in zip(ui_final.value, FINAL))
    mo.vstack([
        mo.md(t('## J. Abschlussquiz: Teil III', '## J. Final quiz: part III')),
        key(t(f'{len(FINAL)} Fragen zu allen drei Notebooks; jede Rückmeldung sagt, wo es vorkam.',
              f'{len(FINAL)} questions about all three notebooks; each feedback says where it came up.')),
        mo.md(t('### Kurz geprüft', '### Quick check')),
        task('final', t('Abschlussquiz', 'Final quiz'), t(['Beantworten Sie die Fragen.', 'Lesen Sie jeweils die Rückmeldung darunter.'],
                                                          ['Answer the questions.', 'Read the feedback below each one.']),
             t('etwa 2 Minuten', 'about 2 minutes') if t.short else t('etwa 4 Minuten', 'about 4 minutes')),
        *[mo.vstack([ui_final[n], quiz_feedback(ui_final.value[n], f['right'], f['fb'])], gap=0.5) for n, f in enumerate(FINAL)],
        *([mo.md(t(f'**Sie haben {_right} von {len(FINAL)} Fragen richtig beantwortet.**',
                   f'**You answered {_right} of {len(FINAL)} questions correctly.**'))] if len(_done) == len(FINAL) else []),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(card, cards, frame, hint, key, mo, t):
    # Section K (row 16): summary slide with three cards, key takeaway, outlook, next step and the checkpoint question
    mo.vstack([
        mo.md(t('## K. Zusammenfassung', '## K. Summary')),
        key(t('Sie haben eine Antwort selbst beurteilt, Answer Correctness von Faithfulness unterschieden, alle vier Kennzahlen gesehen und einen Prompt gegen die Streuung geprüft.',
              'You judged an answer yourself, told Answer Correctness apart from Faithfulness, saw all four metrics and checked a prompt against the spread.')),
        cards([
            card('Answer Correctness', t('Stimmt die Antwort mit der Referenzantwort überein?', 'Does the answer agree with the reference answer?')),
            card('Faithfulness', t('Steht, was die Antwort sagt, in den Chunks?', 'Is what the answer says in the chunks?')),
            card(t('Streuung', 'Spread'), t('Ein Unterschied zählt erst, wenn er größer ist als die Streuung des Richters.',
                                            "A difference only counts when it is larger than the judge's spread.")),
        ]),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'Answer Correctness misst Übereinstimmung mit der Referenz, nicht fachliche Qualität. Faithfulness misst, ob die Antwort belegt ist, nicht ob sie richtig ist.',
            'Answer Correctness measures agreement with the reference, not subject-matter quality. Faithfulness measures whether the answer is supported, not whether it is correct.')),
        frame(t('Zum Prüfen', 'Check yourself'), t('Können Sie in je einem Satz sagen, was Answer Correctness und was Faithfulness misst?',
                                                  'Can you say in one sentence each what Answer Correctness and what Faithfulness measure?')),
        mo.hstack([
            mo.vstack([mo.md(t('#### Ausblick', '#### Outlook')),
                       mo.md(t('Notebook 4 zeigt, wie Urteile wie die des Richters genutzt werden, um das Sprachmodell selbst zu verbessern.',
                               "Notebook 4 shows how judgments like the judge's are used to improve the language model itself."))]),
            mo.vstack([mo.md(t('#### Nächster Schritt', '#### Next step')),
                       mo.md(t('Weiter mit Notebook 4 zu RLHF und RLAIF.', 'Continue with notebook 4 on RLHF and RLAIF.')),
                       mo.md('`uv run marimo run w3_04_rlhf_rlaif.py`'),
                       hint(t('Im Ordner notebooks.', 'In the notebooks folder.'))]),
        ], widths='equal', gap=2, align='start'),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(B, EX, mo, re, t):
    # FAQ F1 to F6; F4 and F6 in both versions because question 22 is probably misjudged (sections 3a and 3b)
    _c21 = EX['check21']['faithfulness']['claims']
    _n21 = len(_c21)
    # claims that turn a SOLLTE into a MUSS; 'angepasst werden muss' in a subordinate clause is no such case
    _muss21 = sum(bool(re.search(r'\bmuss\b', c['statement'])) and 'werden muss' not in c['statement'] for c in _c21)
    _faq = {
        t('Warum hat dieselbe Antwort unterschiedlich viele Aussagen?', 'Why does the same answer have a different number of claims?'): mo.md(t(
            f"Answer Correctness und Faithfulness zerlegen die Antwort mit je eigenem Prompt, und jeder Aufruf kann anders ausfallen. Deshalb weicht die Zahl der Aussagen zwischen den Kennzahlen und zwischen Läufen ab. Bei der Antwort aus Abschnitt B waren es {len(B['ac']['answer_claims'])} Aussagen für Answer Correctness und {len(B['faithfulness']['claims'])} für Faithfulness.",
            f"Answer Correctness and Faithfulness split the answer with a prompt of their own each, and every call can turn out differently. That is why the number of claims differs between the metrics and between runs. For the answer from section B there were {len(B['ac']['answer_claims'])} claims for Answer Correctness and {len(B['faithfulness']['claims'])} for Faithfulness.")),
        **t.full({t('Woher kommen die Gewichte 75 und 25?', 'Where do the weights 75 and 25 come from?'): mo.md(t(
            'Das ist die Voreinstellung von RAGAS. Man kann die Gewichte beim Anlegen der Kennzahl ändern, zum Beispiel stärker auf die Ähnlichkeit setzen, wenn Ihre Referenzantworten kurz sind. Vergleichen dürfen Sie dann nur Werte mit denselben Gewichten.',
            'This is the RAGAS default. You can change the weights when you create the metric, for example rely more on the similarity if your reference answers are short. You may then only compare values with the same weights.'))}, {}),
        t('Warum sind die Antworten zum Gold-Datensatz so lang?', 'Why are the answers on the gold data set so long?'): mo.md(t(
            'Das Antwortmodell gpt-oss-120b ist ein Reasoning-Modell und antwortet ausführlich, oft mit Tabellen und Quellen. Der Original-Prompt verlangt keine Kürze.',
            'The answer model gpt-oss-120b is a reasoning model and answers at length, often with tables and sources. The original prompt does not ask for brevity.')),
        t('Warum ist Faithfulness bei Frage 22 vermutlich fehlbewertet?', 'Why is Faithfulness for question 22 probably misjudged?'): mo.md(t(
            f"Der Richter hat alle {_n21} Aussagen dieser Antwort abgelehnt. Die meisten geben eine Anforderung beschreibend wieder ('werden aktualisiert' statt 'SOLLTEN aktualisiert werden'), und der Richter liest das als Behauptung über den Ist-Zustand. Das ist vermutlich ein Fehlurteil; es trat in zwei getrennten Läufen auf. {_muss21} andere Aussagen dieser Antwort stufen aber SOLLTE zu MUSS hoch, zum Beispiel 'Der Anbieter muss nachweislich bestätigen ...', und die lehnt er zu Recht ab. Die Tabelle in Abschnitt F zeigt den Mittelwert ohne und mit dieser Frage.",
            f"The judge rejected all {_n21} claims of this answer. Most of them restate a requirement descriptively (<span class='hpi-corpus'>'werden aktualisiert'</span>, are updated, instead of <span class='hpi-corpus'>'SOLLTEN aktualisiert werden'</span>, should be updated), and the judge reads that as a statement about the current state. This is probably a misjudgment; it occurred in two separate runs. {_muss21} other claims of this answer, however, raise SOLLTE to MUSS, for example <span class='hpi-corpus'>'Der Anbieter muss nachweislich bestätigen ...'</span> (the provider must verifiably confirm), and those it rejects rightly. The table in section F shows the mean without and with this question.")),
        t('Darf ich Werte selbst als fehlbewertet markieren?', 'May I mark values as misjudged myself?'): mo.md(t(
            'Nur mit einer vorher festgelegten Regel, nach einer Stichprobe gegen die Chunks, und im Bericht mit beiden Mittelwerten, mit und ohne diese Fälle.',
            'Only with a rule fixed in advance, after a spot check against the chunks, and in the report with both means, with and without these cases.')),
        **t.full({t('Warum ist der Richter dasselbe Modell wie das Antwortmodell?', 'Why is the judge the same model as the answer model?'): mo.md(t(
            'Hier aus praktischen Gründen: Es ist das stärkste verfügbare Modell. In der Praxis nimmt man oft ein anderes Modell als Richter, damit es nicht die eigenen Formulierungen bevorzugt.',
            'Here for practical reasons: it is the strongest model available. In practice one often takes a different model as judge, so that it does not favor its own wording.'))}, {}),
    }
    mo.vstack([mo.md(t('## Häufige Fragen', '## Frequently asked questions')), mo.accordion(_faq)])
    return


if __name__ == "__main__":
    app.run()
