# ruff: noqa: PLR1711  marimo cells end with an explicit return

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Wie lernt ein Modell, was eine gute Antwort ist?",
    css_file="ragkit/theme.css",
    html_head_file="ragkit/head.html",
)


@app.cell(hide_code=True)
def _(mo, t, theme):
    # Row 0: header; SECTIONS_MARKER makes every H2 section collapsible (script in ragkit/head.html)
    mo.Html(theme.header_html(
        t('Wie lernt ein Modell, was eine gute Antwort ist?', 'How does a model learn what a good answer is?'),
        t('RAG III, Notebook 4. RLHF und RLAIF: wie Urteile über Antworten ein Sprachmodell verbessern, und wann sich das lohnt.',
          'RAG III, notebook 4. RLHF and RLAIF: how judgments about answers improve a language model, and when that pays off.'),
    ) + theme.SECTIONS_MARKER)
    return


@app.cell(hide_code=True)
def _():
    import json
    from collections import Counter  # majority verdict of the judge runs

    import marimo as mo  # the reactive notebook itself

    from ragkit import i18n, theme  # language and scope switches, HPI look
    from ragkit.config import DATA_DIR
    from ragkit.viz import md_table
    return Counter, DATA_DIR, i18n, json, md_table, mo, theme


@app.cell(hide_code=True)
def _(i18n, mo):
    # Row 0: language and scope, preset from `-- --lang en --short` or `?lang=en&short=1`
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
        """Immediate feedback under a quiz radio; nothing before a choice. correct=None: no right answer."""
        if value is None:
            return mo.md('')
        return feedback(None if correct is None else value == correct, explanation)
    return feedback, quiz_feedback, t


@app.cell(hide_code=True)
def _(mo, t, theme):
    # Slide building blocks as in w3_01, shared in ragkit/theme.py: key sentence, cards, grey hint lines,
    # information and task boxes. One red mark per view (the title rule).
    key, card, cards, hint, frame, info, left = (theme.key_sentence, theme.card, theme.cards, theme.hint,
                                                 theme.frame, theme.info, theme.left)

    def tiles(items):
        """A row of stat tiles; items are (value, label) or (value, label, caption)."""
        return mo.hstack([mo.stat(v, label=lab, caption=(rest[0] if rest else None), bordered=True)
                          for v, lab, *rest in items], justify='start', gap=1, wrap=True)

    # Meta plan 10.4: free-text fields are confirmed with a green button (CTA style); choices reveal directly.
    def confirm(label, source):
        """A green confirm button (as in w3_02). Its value is None before the first click, then {'v': ...} with the
        value of `source` (a function) at the moment of the click."""
        return mo.ui.button(value=None, on_click=lambda _: {'v': source()}, label=theme.cta_label(label))

    def cta(button):
        """A button in the CTA frame of w3_01."""
        return mo.Html(f'<div class="hpi-cta">{button}</div>')

    def confirmed(button):
        """The value confirmed with `button`: None before the first click."""
        return None if button.value is None else button.value['v']

    # Every place where participants act gets a numbered task box; each quiz is its own counted step,
    # as in w3_01 to w3_03. The prediction of E2 exists in the normal version only.
    TASK_ORDER = ['quiz_where', 'pair', *t.full(['predict'], []), 'quiz_worth', 'scenarios', 'final']

    def task(name, title, steps, time):
        """The task box of a numbered task, or of the extra task (name='extra')."""
        return theme.numbered_task(t, TASK_ORDER, name, title, steps, time)

    def quiz_task(name, time=None):
        """H3 "Kurz geprüft" and the counted task box of a quiz, as in w3_02."""
        return mo.vstack([
            mo.md(t('### Kurz geprüft', '### Quick check')),
            task(name, '', t(['Beantworten Sie die Frage.', 'Lesen Sie die Rückmeldung darunter.'],
                             ['Answer the question.', 'Read the feedback below it.']),
                 time or t('etwa 1 Minute', 'about 1 minute')),
        ], gap=1)
    return card, cards, confirm, confirmed, cta, frame, hint, info, key, left, quiz_task, task, tiles


@app.cell(hide_code=True)
def _(DATA_DIR, json, mo, t):
    # Row 0: the cache of tools/precompute_w3_04.py; a red callout only when it is missing
    CACHE = DATA_DIR / 'cache' / 'w3_04'
    _files = [CACHE / 'pair.json', CACHE / 'verdicts.json']
    cache_ok = all(p.exists() for p in _files)

    def _load(name):
        _p = CACHE / name
        return json.loads(_p.read_text(encoding='utf-8')) if _p.exists() else {}

    PAIR, VERDICTS = _load('pair.json'), _load('verdicts.json')
    # German corpus text -> English, English version only: website and answers of notebooks 1 and 3, judge reasons
    TR = {**_load('translations_en.json'),
          **json.loads((DATA_DIR / 'cache' / 'w3_01' / 'translations_en.json').read_text(encoding='utf-8')),
          **json.loads((DATA_DIR / 'cache' / 'w3_03' / 'kisz' / 'translations_en.json').read_text(encoding='utf-8'))}
    (None if cache_ok else mo.callout(mo.md(t(
        'Der Cache mit den Urteilen des Richters fehlt. Bauen Sie ihn einmal mit `uv run python -m tools.precompute_w3_04` im Ordner `notebooks`.',
        "The cache with the judge's verdicts is missing. Build it once with `uv run python -m tools.precompute_w3_04` in the `notebooks` folder.",
    )), kind='danger'))
    return PAIR, TR, VERDICTS, cache_ok


@app.cell(hide_code=True)
def _(Counter, PAIR, TR, VERDICTS, cache_ok, mo, t, theme):
    mo.stop(not cache_ok)

    def en_of(text):
        """English translation of a cached German text; only in the English version, only when it differs."""
        if t.lang != 'en':
            return None
        _en = TR.get(text)
        return _en if _en and _en.strip() != text.strip() else None

    def bi(text):
        """Corpus text as it is (class hpi-corpus for the language check), English translation below in grey."""
        _orig = mo.Html(f'<div class="hpi-corpus">{mo.md(text).text}</div>')
        _en = en_of(text)
        return _orig if _en is None else mo.vstack([_orig, theme.hint(_en)], gap=0.25)

    # The pair of section D: A is the short answer of notebook 1, B the fuller one of notebook 3, section B.
    ANSWERS = {'A': PAIR['short'], 'B': PAIR['full']}
    KIND_TO_LABEL = {'short': 'A', 'full': 'B'}
    REASONS = {'richtiger': ('richtiger', 'more correct'), 'vollständiger': ('vollständiger', 'more complete'),
               'kürzer': ('kürzer', 'shorter'), 'belegter': ('besser belegt', 'better supported')}

    def reason_label(code):
        return t(*REASONS[code])

    def tally(part, order):
        """Runs of one judge setting and one order: how often each answer won, the majority, its most common reason."""
        _runs = VERDICTS[part][f'{order}_first']
        _wins = Counter(KIND_TO_LABEL[r['winner']] for r in _runs)
        _top = _wins.most_common(1)[0][0]
        _major = [r for r in _runs if KIND_TO_LABEL[r['winner']] == _top]
        return {'runs': _runs, 'n': len(_runs), 'wins': _wins, 'top': _top, 'top_n': _wins[_top],
                'reason': Counter(r['reason'] for r in _major).most_common(1)[0][0], 'example': _major[0]}

    def answer_cards(widths='equal'):
        """Answers A and B side by side, each a hairline card, corpus text with grey English below."""
        _style = {'border': f'1px solid {theme.RULE}', 'border-radius': '2px', 'padding': '.75rem 1rem',
                  'height': '100%', 'background': '#ffffff'}
        return mo.hstack([
            mo.vstack([mo.md(f"**{t('Antwort', 'Answer')} {k}**"), bi(ANSWERS[k]['answer']),
                       theme.hint(t(f"{ANSWERS[k]['words']} Wörter", f"{ANSWERS[k]['words']} words"))], gap=0.5).style(_style)
            for k in ('A', 'B')], widths=widths, gap=1, align='stretch')
    return ANSWERS, answer_cards, bi, reason_label, tally


@app.cell(hide_code=True)
def _(card, cards, hint, key, mo, t):
    # Section A (agenda slide): key sentence, the route as five cards, the bridge to the judge of notebooks 2 and 3
    mo.vstack([
        mo.md(t('## A. Worum es geht', '## A. What this is about')),
        key(t('Bisher haben wir gemessen, wie gut Antworten sind. RLHF und RLAIF nutzen solche Urteile, um das Modell selbst zu verbessern.',
              'So far we have measured how good answers are. RLHF and RLAIF use such judgments to improve the model itself.')),
        mo.md(t('### Ihr Weg in diesem Notebook', '### Your route through this notebook')),
        cards([
            card(t('B. Wofür?', 'B. What for?'), t('Wofür RLHF und RLAIF eingesetzt werden.',
                                                 'What RLHF and RLAIF are used for.')),
            card(t('C. Wo?', 'C. Where?'), t('Wo RLHF in der Pipeline liegt, und was Sie selbst ändern können.',
                                              'Where RLHF sits in the pipeline, and what you can change yourself.')),
            card(t('D. Wie?', 'D. How?'), t('Die drei Schritte, und Sie bewerten selbst ein Paar von Antworten.',
                                            'The three steps, and you judge a pair of answers yourself.')),
            card(t('E. Wer bewertet?', 'E. Who judges?'), t('Ihr Urteil neben dem Urteil eines Sprachmodells.',
                                                           "Your judgment next to a language model's judgment.")),
            card(t('F. Wann?', 'F. When?'), t('Wann sich der Aufwand lohnt, an drei Fällen.',
                                              'When the effort pays off, in three cases.')),
        ]),
        hint(t('RLHF: Reinforcement Learning from Human Feedback. RLAIF: Reinforcement Learning from AI Feedback.',
               'RLHF: reinforcement learning from human feedback. RLAIF: reinforcement learning from AI feedback.')),
        hint(t('Den Richter kennen Sie aus Notebook 2 und 3. Bei RLAIF bewertet er nicht Ihr System, sondern liefert die Urteile, aus denen ein Modell lernt.',
               'You know the judge from notebooks 2 and 3. In RLAIF it does not score your system; it supplies the judgments a model learns from.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(card, cards, hint, key, mo, t):
    # Section B (slide, Jill 2026-10-19): what RLHF and RLAIF are used for, four cards and the link to the models of part III
    mo.vstack([
        mo.md(t('## B. Wofür wird es benutzt?', '## B. What is it used for?')),
        key(t('RLHF und RLAIF bringen einem Modell bei, so zu antworten, wie Menschen es bevorzugen.',
              'RLHF and RLAIF teach a model to answer the way people prefer.')),
        cards([
            card(t('Anweisungen befolgen', 'Following instructions'),
                 t('Aus einem Modell, das nur Text fortsetzt, wird ein Assistent, der auf Fragen antwortet. So entstanden Chat-Modelle wie ChatGPT, Claude und Llama-Chat.',
                   'A model that only continues text becomes an assistant that answers questions. This is how chat models such as ChatGPT, Claude and Llama-Chat were made.')),
            card(t('Sicher antworten', 'Answering safely'),
                 t('Schädliche Anfragen ablehnen, harmlose trotzdem beantworten.',
                   'Refusing harmful requests while still answering harmless ones.')),
            card(t('Weniger Erfundenes', 'Less made up'),
                 t('Bei InstructGPT wurden die Antworten wahrheitsgetreuer und seltener toxisch.',
                   'With InstructGPT, answers became more truthful and less often toxic.')),
            card(t('Ton und Form', 'Tone and format'),
                 t('Höflich, knapp, gut gegliedert: so, wie Menschen es in Vergleichen bevorzugen.',
                   'Polite, concise, well structured: the way people prefer it in comparisons.')),
        ]),
        hint(t('Llama 3.3, in Notebook 2 der zweite Richter, wurde laut Meta mit Feinabstimmung und RLHF nachtrainiert. Das Chat-Modell in Ihrem RAG-System ist also schon so ausgerichtet.',
               'Llama 3.3, the second judge in notebook 2, was post-trained with fine-tuning and RLHF according to Meta. So the chat model in your RAG system is already aligned this way.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t, theme):
    def pipeline_html():
        """Section C: the chain from pretraining to your RAG system in two groups, styled like the chains of w3_01
        (white panel, greys); RLHF or RLAIF dark, the provider's part and your part each with a grey caption."""
        _box = ('display:inline-block;padding:.35rem .7rem;border:1.2px solid {b};border-radius:3px;'
                'background:{bg};color:{c};font-size:.85rem;white-space:nowrap')
        _arrow = f'<span style="color:{theme.INK};padding:0 .35rem">&#8594;</span>'

        def _chain(steps, dark=None):
            return _arrow.join(
                f'<span style="{_box.format(b=theme.INK, bg=theme.INK if i == dark else "#ffffff", c="#ffffff" if i == dark else theme.INK)}">{s}</span>'
                for i, s in enumerate(steps))

        def _group(caption, inner, border):
            return (f'<div style="border:1.2px {border} {theme.GREYS[3]};border-radius:4px;padding:.45rem .6rem">'
                    f'<div style="color:{theme.GREY};font-size:.78rem;margin-bottom:.35rem">{caption}</div>'
                    f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:.25rem 0">{inner}</div></div>')

        _provider = _chain(t(['Vortraining', 'Feinabstimmung mit Beispielen (SFT)', 'RLHF oder RLAIF', 'fertiges Sprachmodell'],
                             ['pretraining', 'fine-tuning with examples (SFT)', 'RLHF or RLAIF', 'finished language model']), dark=2)
        _yours = _chain(t(['Suche', 'Prompt', 'Antwort', 'Bewertung mit RAGAS'],
                          ['retrieval', 'prompt', 'answer', 'evaluation with RAGAS']))
        return mo.Html(
            f'<div style="display:flex;flex-direction:column;align-items:flex-start;gap:.3rem;background:#ffffff;'
            f'padding:.6rem .8rem;border-radius:6px;border:1px solid {theme.RULE}">'
            + _group(t('macht der Anbieter des Modells, einmal', 'done by the model provider, once'), _provider, 'dashed')
            + f'<span style="color:{theme.INK};padding-left:1.2rem;font-size:1.1rem">&#8595;</span>'
            + _group(t('Ihr RAG-System: läuft bei jeder Anfrage, Sie stellen es selbst ein', 'your RAG system: runs on every request, you set it up yourself'), _yours, 'solid')
            + '</div>')
    return (pipeline_html,)


@app.cell(hide_code=True)
def _(hint, key, mo, pipeline_html, t):
    # Section C (slide): the chain, the key sentence
    mo.vstack([
        mo.md(t('## C. Wo in der Pipeline?', '## C. Where in the pipeline?')),
        key(t('RLHF passiert im Training des Modells, nicht bei jeder Anfrage.',
              'RLHF happens while the model is trained, not on every request.')),
        pipeline_html(),
        hint(t('Abbildung 1. Oben der Weg zum fertigen Sprachmodell, unten Ihr RAG-System, das dieses Modell nutzt.',
               'Figure 1. At the top the path to the finished language model, below it your RAG system, which uses this model.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Quiz before task "pair": several answers are right (multiple choice as checkboxes)
    QUIZ_WHERE = [('topk', 'TOP_K', True), ('prompt', t('System-Prompt', 'system prompt'), True),
                  ('chunking', 'Chunking', True), ('pref', t('Präferenzen des Modells', "the model's preferences"), False)]
    ui_where = mo.ui.array([mo.ui.checkbox(label=lab) for _k, lab, _r in QUIZ_WHERE])
    return QUIZ_WHERE, ui_where


@app.cell(hide_code=True)
def _(QUIZ_WHERE, feedback, mo, quiz_task, t, ui_where):
    _chosen = list(ui_where.value)
    _ok = all(c == r for c, (_k, _l, r) in zip(_chosen, QUIZ_WHERE))
    _marks = ', '.join(f"{lab} {'&#10003;' if c == r else '&#10007;'}" for c, (_k, lab, r) in zip(_chosen, QUIZ_WHERE))
    mo.vstack([
        quiz_task('quiz_where'),
        mo.md(t('**Was davon können Sie in Ihrem RAG-System selbst ändern, ohne das Modell neu zu trainieren?** Mehrere Antworten sind möglich.',
                "**Which of these can you change in your RAG system yourself, without training the model again?** Several answers are possible.")),
        mo.hstack(list(ui_where), justify='start', gap=2, wrap=True),
        *([feedback(_ok, t(
            f'{_marks}\n\nTOP_K, System-Prompt und Chunking gehören zu Ihrem RAG-System; Sie ändern sie in Minuten und messen die Wirkung mit RAGAS. Was das Modell für eine gute Antwort hält, legt das Training fest, also RLHF oder RLAIF beim Anbieter.',
            f'{_marks}\n\nTOP_K, system prompt and chunking belong to your RAG system; you change them in minutes and measure the effect with RAGAS. What the model considers a good answer is set in training, that is, by RLHF or RLAIF at the provider.'))]
          if any(_chosen) else []),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(card, cards, hint, key, left, md_table, mo, t):
    # Section D (slide 19): the three steps as cards; DPO only in the accordion
    _dpo = md_table([
        {'': 'PPO', t('Schritt 2', 'Step 2'): t('eigenes Bewertungsmodell', 'separate reward model'),
         t('Schritt 3', 'Step 3'): t('Modell wird optimiert, bis das Bewertungsmodell es belohnt', 'the model is optimized until the reward model rewards it')},
        {'': 'DPO', t('Schritt 2', 'Step 2'): t('entfällt als eigenes Modell', 'no separate model'),
         t('Schritt 3', 'Step 3'): t('Modell lernt direkt aus den Paaren', 'the model learns directly from the pairs')},
    ])
    _arrow = mo.md('&#8594;').style({'font-size': '1.4rem', 'color': '#5a6065', 'padding-top': '2.2rem'})
    mo.vstack([
        mo.md(t('## D. Wie funktioniert es?', '## D. How does it work?')),
        key(t('Ein Modell lernt in drei Schritten, welche Antworten Menschen bevorzugen.',
              'A model learns in three steps which answers people prefer.')),
        mo.hstack([
            card(t('1. Beispiele', '1. Examples'), t('Menschen schreiben Beispielantworten; das Modell wird damit feinabgestimmt (SFT).',
                                                    'People write example answers; the model is fine-tuned on them (SFT).')),
            _arrow,
            card(t('2. Bewertungsmodell', '2. Reward model'), t('Menschen vergleichen zwei Antworten: "A ist besser als B". Aus vielen solchen Paaren lernt ein Bewertungsmodell (Reward Model).',
                                                              'People compare two answers: "A is better than B". From many such pairs a reward model learns.')),
            _arrow,
            card(t('3. Optimierung', '3. Optimization'), t('Das Modell wird so optimiert, dass das Bewertungsmodell seine Antworten belohnt (PPO).',
                                                          'The model is optimized so that the reward model rewards its answers (PPO).')),
        ], widths=[4, 0.3, 4, 0.3, 4], gap=0.5, align='stretch'),
        hint(t('Schritt 2 probieren Sie gleich selbst aus.', 'You try step 2 yourself in a moment.')),
        mo.accordion({t('Und DPO?', 'And DPO?'): mo.vstack([
            mo.md(t('DPO (Direct Preference Optimization) spart Schritt 2 als eigenes Modell: Das Sprachmodell lernt direkt aus den Präferenzpaaren.',
                    'DPO (direct preference optimization) saves step 2 as a separate model: the language model learns directly from the preference pairs.')),
            mo.md(left(_dpo)),
        ])}),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Task "pair": the choice and the reason, defined apart so the view above never resets them
    ui_pick = mo.ui.radio({t('Antwort A', 'Answer A'): 'A', t('Antwort B', 'Answer B'): 'B'}, inline=True,
                          label=t('Welche Antwort ist besser?', 'Which answer is better?'))
    ui_why = mo.ui.radio({t('richtiger', 'more correct'): 'richtiger', t('vollständiger', 'more complete'): 'vollständiger',
                          t('kürzer', 'shorter'): 'kürzer', t('besser belegt', 'better supported'): 'belegter'}, inline=True,
                         label=t('Warum?', 'Why?'))
    return ui_pick, ui_why


@app.cell(hide_code=True)
def _(PAIR, answer_cards, bi, hint, info, mo, t, task, ui_pick, ui_why):
    # Task "pair" (row C): question and website line visible, answers A and B side by side
    mo.vstack([
        mo.md(t('### Ein Paar selbst bewerten', '### Judge a pair yourself')),
        hint(t('Daten: KISZ-Webseite', 'Data: website of the AI Service Center')),
        task('pair', t('Welche Antwort ist besser?', 'Which answer is better?'), t(
            ['Lesen Sie die Frage und die Stelle auf der Webseite.', 'Lesen Sie die Antworten A und B.',
             'Wählen Sie unten die bessere Antwort und einen Grund.'],
            ['Read the question and the passage on the website.', 'Read answers A and B.',
             'Choose the better answer and a reason below.']), t('etwa 3 Minuten', 'about 3 minutes')),
        mo.hstack([
            mo.vstack([mo.md(f"#### {t('Frage', 'Question')}"), bi(PAIR['question'])]),
            info(t('Das steht auf der Webseite', 'This is what the website says'),
                 mo.vstack([mo.vstack([mo.md(f"*{t('Zeile', 'Line')} {n}*"), bi(line)], gap=0.25)
                            for n, line in zip(PAIR['source_ids'], PAIR['source_lines'])])),
        ], widths='equal', gap=2, align='start'),
        answer_cards(),
        ui_pick,
        ui_why,
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(feedback, hint, mo, reason_label, t, ui_pick, ui_why):
    # Feedback on the own choice: no right or wrong; such pairs are the data of step 2
    mo.stop(ui_pick.value is None or ui_why.value is None)
    mo.vstack([
        feedback(None, t(
            f'Sie haben Antwort {ui_pick.value} gewählt, weil sie {reason_label(ui_why.value)} ist. Genau solche Paare sind die Daten für Schritt 2: Viele Menschen wählen zwischen zwei Antworten, und das Bewertungsmodell lernt daraus, welche Antworten bevorzugt werden.',
            f'You chose answer {ui_pick.value} because it is {reason_label(ui_why.value)}. Pairs like this are exactly the data for step 2: many people choose between two answers, and the reward model learns from that which answers are preferred.')),
        hint(t('A ist die kurze Antwort aus Notebook 1, B die ausführliche aus Notebook 3, Abschnitt B. Beide stützen sich auf dieselben drei Chunks.',
               'A is the short answer from notebook 1, B the detailed one from notebook 3, section B. Both rely on the same three chunks.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(VERDICTS, key, mo, t):
    # Section E (slide): the key sentence; the judge's verdict follows once the own choice exists
    mo.vstack([
        mo.md(t('## E. Mensch oder KI als Bewerter?', '## E. Human or AI as the judge?')),
        key(t('Bei RLAIF übernimmt ein Richter wie in Notebook 2 und 3 die Rolle des Menschen, mit denselben Schwächen.',
              'In RLAIF a judge like the one in notebooks 2 and 3 takes the place of the human, with the same weaknesses.')),
        mo.md(t(f"Dasselbe Paar hat der Richter ({VERDICTS.get('model', 'gpt-oss-120b')}) bewertet: gleiche Frage, gleiche Stelle der Webseite, gleiche Gründe zur Auswahl.",
                f"The judge ({VERDICTS.get('model', 'gpt-oss-120b')}) rated the same pair: same question, same passage of the website, same reasons to choose from.")),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(ANSWERS, bi, hint, info, mo, reason_label, t, tally, tiles, ui_pick, ui_why):
    # Row D: your verdict next to the judge's, as tiles; one reason of the judge in its own words
    mo.stop(ui_pick.value is None, hint(t('Wählen Sie zuerst in Abschnitt D eine Antwort; dann erscheint hier das Urteil des Richters.',
                                          "First choose an answer in section D; then the judge's verdict appears here.")))
    _p = tally('plain', 'short')  # the order participants see: the short answer as A
    mo.vstack([
        tiles([
            (t(f'Antwort {ui_pick.value}', f'Answer {ui_pick.value}'), t('Ihre Wahl', 'Your choice'),
             reason_label(ui_why.value) if ui_why.value else t('Grund noch offen', 'reason still open')),
            (t(f"Antwort {_p['top']}", f"Answer {_p['top']}"), t('Richter', 'Judge'),
             t(f"in {_p['top_n']} von {_p['n']} Läufen", f"in {_p['top_n']} of {_p['n']} runs")),
            (reason_label(_p['reason']), t('häufigster Grund des Richters', "judge's most frequent reason")),
        ]),
        info(t('Begründung des Richters, ein Lauf', "The judge's reasoning, one run"), bi(_p['example']['text'])),
        hint(t(f"Faithfulness aus Notebook 3: A {t.num(ANSWERS['A']['faithfulness'])}, B {t.num(ANSWERS['B']['faithfulness'])}. Die Erläuterungen zu Weiterbildung und Infrastruktur in B fand der Richter dort nicht in den Chunks.",
               f"Faithfulness from notebook 3: A {t.num(ANSWERS['A']['faithfulness'])}, B {t.num(ANSWERS['B']['faithfulness'])}. The judge there did not find B's explanations of training and infrastructure in the chunks.")),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(hint, left, md_table, mo, t):
    # Row D: RLHF against RLAIF after slide 20, five rows
    _h, _a = 'RLHF', 'RLAIF'
    _rows = [
        (t('Feedback', 'Feedback'), t('Menschen bewerten', 'people judge'), t('ein Sprachmodell bewertet (Richter)', 'a language model judges (judge)')),
        (t('Qualität', 'Quality'), t('sehr hoch, nah an menschlichen Präferenzen', 'very high, close to human preferences'),
         t('hängt vom Richter und seinem Prompt ab', 'depends on the judge and its prompt')),
        (t('Skalierbarkeit', 'Scalability'), t('begrenzt durch menschliche Arbeitszeit', 'limited by human working time'),
         t('gut, weil automatisiert', 'good, because automated')),
        (t('Kosten', 'Cost'), t('hoch durch Zeit und Aufwand der Menschen', 'high, from the time and effort of people'),
         t('hoch durch Rechenleistung, aber besser skalierbar', 'high, from compute, but scales better')),
        (t('Bias', 'Bias'), t('menschliche Vorurteile', 'human prejudices'), t('Vorlieben des Richter-Modells', "the judge model's leanings")),
    ]
    mo.vstack([
        mo.md(t('### RLHF und RLAIF im Vergleich', '### RLHF and RLAIF compared')),
        mo.md(left(md_table([{t('Aspekt', 'Aspect'): a, _h: h, _a: ai} for a, h, ai in _rows]))),
        hint(t('Optimiert wird in beiden Fällen mit PPO oder DPO.', 'In both cases the optimization uses PPO or DPO.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # E2 (normal only): prediction before the reversed order is revealed
    ui_predict = mo.ui.radio({t('seltener als vorher', 'less often than before'): 'less',
                              t('genauso oft', 'just as often'): 'same',
                              t('öfter', 'more often'): 'more'},
                             label=t('Jetzt steht die ausführliche Antwort vorn, als A. Wie oft wählt der Richter sie?',
                                     'Now the detailed answer comes first, as A. How often does the judge choose it?'))
    return (ui_predict,)


@app.cell(hide_code=True)
def _(mo, t, task, ui_predict):
    # E2 (normal only): the task box and the prediction
    t.full(mo.vstack([
        mo.md(t('### Vertauschte Reihenfolge', '### Reversed order')),
        mo.md(t('Bisher stand die kurze Antwort als A vorn. Wir haben denselben Richter noch einmal gefragt, mit vertauschten Antworten.',
                'So far the short answer came first, as A. We asked the same judge again, with the answers swapped.')),
        task('predict', t('Vorhersage', 'Prediction'), t(
            ['Überlegen Sie, ob die Reihenfolge eine Rolle spielt.', 'Wählen Sie unten Ihre Vorhersage.'],
            ['Consider whether the order matters.', 'Choose your prediction below.']), t('etwa 1 Minute', 'about 1 minute')),
        ui_predict,
    ], gap=1))
    return


@app.cell(hide_code=True)
def _(feedback, left, md_table, mo, t, tally, ui_predict):
    # E2: the counts of both orders; the right option comes from the data, not from an expectation
    mo.stop(t.short or ui_predict.value is None)
    _a, _b = tally('plain', 'short'), tally('plain', 'full')
    _before, _after = _a['wins']['B'], _b['wins']['B']  # wins of the detailed answer in both orders
    _truth = 'more' if _after > _before else 'less' if _after < _before else 'same'
    _table = md_table([
        {t('Reihenfolge', 'Order'): t('kurze Antwort vorn', 'short answer first'),
         t('ausführliche Antwort gewählt', 'detailed answer chosen'): t(f"{_before} von {_a['n']}", f"{_before} of {_a['n']}")},
        {t('Reihenfolge', 'Order'): t('ausführliche Antwort vorn', 'detailed answer first'),
         t('ausführliche Antwort gewählt', 'detailed answer chosen'): t(f"{_after} von {_b['n']}", f"{_after} of {_b['n']}")},
    ])
    _total, _n = _before + _after, _a['n'] + _b['n']
    feedback(ui_predict.value == _truth, left(_table) + '\n\n' + t(
        f'Insgesamt wählt der Richter die ausführliche Antwort in {_total} von {_n} Läufen, und öfter, wenn sie vorn steht. Das deutet auf zwei Vorlieben hin: für die längere Antwort und für die erste Position. Bei {_a["n"]} Läufen je Reihenfolge ist das ein Hinweis, kein Beweis. Ein Bewertungsmodell, das aus solchen Urteilen lernt, übernimmt diese Vorlieben.',
        f'Overall the judge chooses the detailed answer in {_total} of {_n} runs, and more often when it comes first. That points to two leanings: toward the longer answer and toward the first position. With {_a["n"]} runs per order this is a hint, not proof. A reward model that learns from such judgments takes on these leanings.'))
    return


@app.cell(hide_code=True)
def _(card, cards, hint, key, mo, t):
    # Section F (slide): when it pays off, effort against benefit as two cards
    mo.vstack([
        mo.md(t('## F. Wann lohnt sich das?', '## F. When does it pay off?')),
        key(t('RLHF lohnt sich für Anbieter, die ein Modell für viele Nutzende ausrichten. Für ein einzelnes RAG-System sind Suche, Prompt und Modellwahl fast immer die günstigeren Hebel.',
              'RLHF pays off for providers who align a model for many users. For a single RAG system, retrieval, prompt and choice of model are almost always the cheaper levers.')),
        cards([
            card(t('Aufwand', 'Effort'), t('- viele bewertete Antwortpaare (Präferenzdaten)\n- Rechenzeit für das Training\n- Fachwissen für Training und Bewertung',
                                           '- many rated answer pairs (preference data)\n- compute time for training\n- expertise for training and evaluation')),
            card(t('Nutzen', 'Benefit'), t('- Ton und Stil der Antworten\n- hilfsbereitere Antworten\n- weniger unerwünschte Antworten',
                                           '- tone and style of the answers\n- more helpful answers\n- fewer unwanted answers')),
        ]),
        hint(t('Die Stellschrauben aus Notebook 2 und 3 (TOP_K, Chunking, System-Prompt) ändern Sie in Minuten und messen die Wirkung mit RAGAS.',
               'You change the levers from notebooks 2 and 3 (TOP_K, chunking, system prompt) in minutes and measure the effect with RAGAS.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Quiz before task "scenarios"; the radio apart from its feedback
    ui_worth = mo.ui.radio(
        {t('TOP_K und Chunking prüfen', 'check TOP_K and chunking'): 'search',
         t('das Modell mit RLHF nachtrainieren', 'train the model further with RLHF'): 'rlhf',
         t('einen anderen Richter nehmen', 'use a different judge'): 'judge'},
        label=t('Ihr RAG-System findet oft die falschen Stellen. Was hilft am ehesten?',
                'Your RAG system often finds the wrong passages. What is most likely to help?'))
    return (ui_worth,)


@app.cell(hide_code=True)
def _(mo, quiz_feedback, quiz_task, t, ui_worth):
    mo.vstack([
        quiz_task('quiz_worth'),
        ui_worth,
        quiz_feedback(ui_worth.value, 'search', t(
            'Wenn die Suche die falschen Stellen findet, hilft kein besser ausgerichtetes Modell. Ein anderer Richter ändert nur die Messung, nicht das System.',
            'If the search finds the wrong passages, a better aligned model does not help. A different judge changes only the measurement, not the system.')),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Task "scenarios": three cases (short: cases 1 and 3); right is a set of options, None for no right answer
    _all = [
        {"key": 's1', "short": True, "right": {'no', 'other'},
         "case": t('Der Chatbot einer Behörde findet die richtigen Stellen in den Dokumenten nicht.',
                   "An agency's chatbot does not find the right passages in the documents."),
         "fb": t('Hier lohnt es sich nicht. Das liegt an der Suche, nicht am Sprachmodell. Prüfen Sie Context Recall, TOP_K und Chunking; RLHF ändert daran nichts.',
                 'It does not pay off here. That is down to the search, not the language model. Check Context Recall, TOP_K and chunking; RLHF does not change that.')},
        {"key": 's2', "short": False, "right": None,
         "case": t('Das Modell soll durchgehend in Leichter Sprache antworten; ein Prompt allein reicht nicht.',
                   'The model should answer in plain language throughout; a prompt alone is not enough.'),
         "fb": t('Liegen schon Präferenzdaten vor, etwa Paare aus Antworten in Leichter und in schwerer Sprache, kann eine Feinabstimmung mit DPO helfen. Sie kostet Rechenzeit und Fachwissen; vorher lohnt der Blick auf ein anderes Modell, das es schon kann.',
                 'If preference data already exist, for example pairs of answers in plain and in complex language, fine-tuning with DPO can help. It costs compute time and expertise; before that, it is worth looking for another model that can already do it.')},
        {"key": 's3', "short": True, "right": {'yes'},
         "case": t('Der Anbieter eines allgemeinen Sprachmodells will schädliche Antworten seltener machen.',
                   'The provider of a general language model wants harmful answers to become rarer.'),
         "fb": t('Hier lohnt es sich, genau dafür sind RLHF und RLAIF da: Ein Anbieter richtet ein Modell für sehr viele Nutzende aus, und der Aufwand verteilt sich. Constitutional AI in Abschnitt G ist ein Beispiel.',
                 'It pays off here; this is exactly what RLHF and RLAIF are for: a provider aligns a model for very many users, and the effort is spread across them. Constitutional AI in section G is an example.')},
    ]
    SCENARIOS = [s for s in _all if s['short'] or not t.short]
    _opts = {t('ja', 'yes'): 'yes', t('nein', 'no'): 'no', t('erst etwas anderes versuchen', 'try something else first'): 'other'}
    ui_scen = mo.ui.array([mo.ui.radio(_opts, inline=True, label=f"**{n + 1}.** {s['case']}") for n, s in enumerate(SCENARIOS)])
    return SCENARIOS, ui_scen


@app.cell(hide_code=True)
def _(SCENARIOS, feedback, mo, t, task, ui_scen):
    # Task "scenarios": each case with its feedback directly below it
    def _fb(v, s):
        if v is None:
            return mo.md('')
        return feedback(None if s['right'] is None else v in s['right'], s['fb'])

    mo.vstack([
        mo.md(t('### Lohnt sich RLHF oder RLAIF?', '### Does RLHF or RLAIF pay off?')),
        task('scenarios', t('Drei Fälle entscheiden', 'Decide three cases') if not t.short else t('Zwei Fälle entscheiden', 'Decide two cases'), t(
            ['Lesen Sie jeden Fall.', 'Wählen Sie: ja, nein oder erst etwas anderes versuchen.', 'Lesen Sie die Rückmeldung darunter.'],
            ['Read each case.', 'Choose: yes, no, or try something else first.', 'Read the feedback below it.']),
             t('etwa 3 Minuten', 'about 3 minutes') if t.short else t('etwa 4 Minuten', 'about 4 minutes')),
        *[mo.vstack([ui_scen[n], _fb(ui_scen.value[n], s)], gap=0.5) for n, s in enumerate(SCENARIOS)],
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(card, cards, hint, key, mo, t):
    # Section G (slide): two examples with sources; the short version keeps one sentence each
    _instruct = t.full(
        t('OpenAI hat GPT-3 mit RLHF ausgerichtet: Menschen schrieben Beispielantworten und ordneten Antworten des Modells nach Qualität. Menschen bevorzugten die Antworten des InstructGPT-Modells mit 1,3 Milliarden Parametern gegenüber denen von GPT-3 mit 175 Milliarden.\n\nChatGPT wurde mit denselben Methoden trainiert.',
          'OpenAI aligned GPT-3 with RLHF: people wrote example answers and ranked answers of the model by quality. People preferred the answers of the InstructGPT model with 1.3 billion parameters to those of GPT-3 with 175 billion.\n\nChatGPT was trained with the same methods.'),
        t('Mit RLHF ausgerichtet: Menschen bevorzugten Antworten des Modells mit 1,3 Milliarden Parametern gegenüber GPT-3 mit 175 Milliarden.',
          'Aligned with RLHF: people preferred the answers of the 1.3-billion-parameter model to GPT-3 with 175 billion.'))
    _cai = t.full(
        t('Anthropic hat ein Modell mit RLAIF trainiert: Ein Sprachmodell bewertet Antworten anhand einer Liste schriftlicher Grundsätze, der "Verfassung". Für die Harmlosigkeit sind diese Grundsätze die einzige menschliche Aufsicht; Menschen markieren keine schädlichen Antworten.',
          'Anthropic trained a model with RLAIF: a language model rates answers against a list of written principles, the "constitution". For harmlessness, these principles are the only human oversight; people do not label harmful answers.'),
        t('RLAIF: Ein Sprachmodell bewertet Antworten anhand schriftlicher Grundsätze.',
          'RLAIF: a language model rates answers against written principles.'))
    mo.vstack([
        mo.md(t('## G. Zwei Beispiele', '## G. Two examples')),
        key(t('RLHF und RLAIF stecken in den Sprachmodellen, die Sie in Ihrem RAG-System nutzen.',
              'RLHF and RLAIF are inside the language models you use in your RAG system.')),
        cards([
            card(t('InstructGPT und ChatGPT (OpenAI, 2022)', 'InstructGPT and ChatGPT (OpenAI, 2022)'), _instruct,
                 t.full(t('Quellen', 'Sources'), t('Quelle', 'Source'))
                 + ': [Ouyang et al. 2022, arXiv 2203.02155](https://arxiv.org/abs/2203.02155)'
                 + t.full(', [OpenAI 2022](https://openai.com/index/chatgpt/)', '')),
            card(t('Constitutional AI (Anthropic, 2022)', 'Constitutional AI (Anthropic, 2022)'), _cai,
                 t('Quelle: [Bai et al. 2022, arXiv 2212.08073](https://arxiv.org/abs/2212.08073)', 'Source: [Bai et al. 2022, arXiv 2212.08073](https://arxiv.org/abs/2212.08073)')),
        ]),
        *t.full([hint(t('Grundsätze wie bei Constitutional AI formulieren Sie in der Zusatzaufgabe selbst.',
                        'You write principles like those of Constitutional AI yourself in the extra task.'))], []),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(answer_cards, confirm, cta, hint, mo, t, theme):
    # Extra task (normal only): its own section as in w3_02, closed at the start like every section.
    # Inputs here, results in the cell below, so typing never redraws this box.
    mo.stop(t.short)
    ui_rules = mo.ui.text_area(label=t('Ihre Grundsätze', 'Your principles'), rows=4, full_width=True)
    ui_rule_pick = mo.ui.radio({t('Antwort A', 'Answer A'): 'A', t('Antwort B', 'Answer B'): 'B'}, inline=True,
                               label=t('Welche Antwort bevorzugen Ihre Grundsätze?', 'Which answer do your principles prefer?'))
    ui_rules_ok = confirm(t('Grundsätze übernehmen', 'Use these principles'),
                          lambda: {'rules': ui_rules.value.strip(), 'pick': ui_rule_pick.value})
    mo.vstack([
        mo.md(t('## Zusatzaufgabe, für alle, die schon fertig sind', '## Extra task, for everyone who is already done')),
        hint(t('Alle anderen machen mit Abschnitt H weiter.', 'Everyone else continues with section H.')),
        mo.Html(theme.task_box(
            t('Zusatzaufgabe', 'Extra task'),
            t(['Schreiben Sie zwei bis drei Grundsätze, nach denen ein Richter Antworten bewerten soll, wie bei Constitutional AI.',
               'Wählen Sie, welche der beiden Antworten aus Abschnitt D Ihre Grundsätze bevorzugen.',
               'Drücken Sie auf "Grundsätze übernehmen".'],
              ['Write two or three principles by which a judge should rate answers, as in Constitutional AI.',
               'Choose which of the two answers from section D your principles prefer.',
               'Press "Use these principles".']),
            t('so lange, bis die anderen fertig sind', 'until the others are done'),
            t('Eigene Grundsätze für den Richter', 'Your own principles for the judge'),
            kind='extra')),
        answer_cards(),
        ui_rules,
        ui_rule_pick,
        cta(ui_rules_ok),
    ], gap=1)
    return (ui_rules_ok,)


@app.cell(hide_code=True)
def _(VERDICTS, bi, confirmed, feedback, frame, info, mo, t, tally, ui_rules_ok):
    # Extra task: after the confirm button, your principles next to the example principles and the judge's verdicts
    mo.stop(t.short)
    _c = confirmed(ui_rules_ok)
    mo.stop(_c is None)
    mo.stop(not _c['rules'] or _c['pick'] is None, mo.callout(mo.md(t(
        'Schreiben Sie zuerst Grundsätze und wählen Sie eine Antwort.', 'Write principles and choose an answer first.')), kind='warn'))
    _pr = [tally('principle', o) for o in ('short', 'full')]
    _pl = [tally('plain', o) for o in ('short', 'full')]
    _n_pr, _n_pl = sum(x['n'] for x in _pr), sum(x['n'] for x in _pl)
    _a_pr = sum(x['wins']['A'] for x in _pr)
    _b_pl = sum(x['wins']['B'] for x in _pl)
    _rules = '\n'.join(f'{i + 1}. {r}' for i, r in enumerate(t(VERDICTS['principles_de'], VERDICTS['principles_en'])))
    mo.vstack([
        info(t('Ihre Grundsätze', 'Your principles'), mo.md(_c['rules'] + '\n\n' + t(f"Ihre Wahl: Antwort {_c['pick']}", f"Your choice: answer {_c['pick']}"))),
        frame(t('Beispiel-Grundsätze', 'Example principles'), _rules),
        feedback(None, t(
            f'Mit diesen beiden Grundsätzen wählte der Richter in {_a_pr} von {_n_pr} Läufen die kurze Antwort A, in beiden Reihenfolgen. Ohne Grundsätze wählte er in {_b_pl} von {_n_pl} Läufen die ausführliche Antwort B. Schriftliche Grundsätze lenken das Urteil; genau das nutzt Constitutional AI.',
            f'With these two principles the judge chose the short answer A in {_a_pr} of {_n_pr} runs, in both orders. Without principles it chose the detailed answer B in {_b_pl} of {_n_pl} runs. Written principles steer the verdict; that is exactly what Constitutional AI uses.')),
        info(t('Begründung des Richters mit den Grundsätzen, ein Lauf', "The judge's reasoning with the principles, one run"),
             bi(_pr[0]['example']['text'])),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo, t):
    # Section H: final quiz, three questions (short: two)
    _all = [
        {"short": True, "right": 'training',
         "q": t('Wo passiert RLHF?', 'Where does RLHF happen?'),
         "options": {t('bei jeder Anfrage an Ihr RAG-System', 'on every request to your RAG system'): 'request',
                     t('im Training des Sprachmodells', 'while the language model is trained'): 'training',
                     t('bei der Bewertung mit RAGAS', 'during the evaluation with RAGAS'): 'ragas'},
         "fb": t('RLHF gehört zum Training, beim Anbieter des Modells. Ihr RAG-System nutzt das fertige Modell. Abschnitt C.',
                 'RLHF belongs to training, at the model provider. Your RAG system uses the finished model. Section B.')},
        {"short": True, "right": 'pairs',
         "q": t('Woraus lernt das Bewertungsmodell in Schritt 2?', 'What does the reward model learn from in step 2?'),
         "options": {t('aus Paaren: Antwort A ist besser als B', 'from pairs: answer A is better than B'): 'pairs',
                     t('aus Beispielantworten von Menschen', 'from example answers by people'): 'examples',
                     t('aus den Chunks Ihres RAG-Systems', 'from the chunks of your RAG system'): 'chunks'},
         "fb": t('Beispielantworten sind Schritt 1 (SFT). Das Bewertungsmodell lernt aus Präferenzpaaren, wie Sie eines in Abschnitt D bewertet haben.',
                 'Example answers are step 1 (SFT). The reward model learns from preference pairs, like the one you judged in section D.')},
        {"short": False, "right": 'judge',
         "q": t('Was unterscheidet RLAIF von RLHF?', 'What sets RLAIF apart from RLHF?'),
         "options": {t('Ein Sprachmodell bewertet statt eines Menschen.', 'A language model judges instead of a person.'): 'judge',
                     t('RLAIF braucht kein Training.', 'RLAIF needs no training.'): 'notrain',
                     t('RLAIF hat keinen Bias.', 'RLAIF has no bias.'): 'nobias'},
         "fb": t('Bei RLAIF urteilt ein Richter. Er skaliert gut, bringt aber eigene Vorlieben mit, etwa für längere Antworten. Abschnitt E.',
                 'In RLAIF a judge rates. It scales well but brings leanings of its own, for example toward longer answers. Section D.')},
    ]
    FINAL = [f for f in _all if f['short'] or not t.short]
    ui_final = mo.ui.array([mo.ui.radio(f['options'], label=f"**{n + 1}.** {f['q']}") for n, f in enumerate(FINAL)])
    return FINAL, ui_final


@app.cell(hide_code=True)
def _(FINAL, key, mo, quiz_feedback, t, task, ui_final):
    # Section H: the quiz as a counted task, feedback directly under each question
    _done = [v for v in ui_final.value if v is not None]
    _right = sum(v == f['right'] for v, f in zip(ui_final.value, FINAL))
    mo.vstack([
        mo.md(t('## H. Abschluss', '## H. Wrap-up')),
        key(t(f'{len(FINAL)} Fragen zu diesem Notebook; jede Rückmeldung sagt, wo es vorkam.',
              f'{len(FINAL)} questions about this notebook; each feedback says where it came up.')),
        mo.md(t('### Kurz geprüft', '### Quick check')),
        task('final', t('Abschlussquiz', 'Final quiz'), t(['Beantworten Sie die Fragen.', 'Lesen Sie jeweils die Rückmeldung darunter.'],
                                                          ['Answer the questions.', 'Read the feedback below each one.']),
             t('etwa 1 Minute', 'about 1 minute') if t.short else t('etwa 2 Minuten', 'about 2 minutes')),
        *[mo.vstack([ui_final[n], quiz_feedback(ui_final.value[n], f['right'], f['fb'])], gap=0.5) for n, f in enumerate(FINAL)],
        *([mo.md(t(f'**Sie haben {_right} von {len(FINAL)} Fragen richtig beantwortet.**',
                   f'**You answered {_right} of {len(FINAL)} questions correctly.**'))] if len(_done) == len(FINAL) else []),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(card, cards, frame, mo, t):
    # Section H: summary slide with three cards, key takeaway, outlook and next step
    mo.vstack([
        mo.md(t('### Zusammenfassung', '### Summary')),
        cards([
            card(t('Wo', 'Where'), t('RLHF und RLAIF gehören ins Training des Sprachmodells, nicht in Ihr RAG-System.',
                                     'RLHF and RLAIF belong to the training of the language model, not to your RAG system.')),
            card(t('Wie', 'How'), t('Beispielantworten (SFT), ein Bewertungsmodell aus Präferenzpaaren, dann die Optimierung mit PPO oder DPO.',
                                    'Example answers (SFT), a reward model from preference pairs, then the optimization with PPO or DPO.')),
            card(t('Wer bewertet', 'Who judges'), t('Bei RLHF Menschen, bei RLAIF ein Richter; er skaliert, bringt aber eigene Vorlieben mit.',
                                                    'In RLHF people, in RLAIF a judge; it scales but brings leanings of its own.')),
        ]),
        frame(t('Kernaussage', 'Key takeaway'), t(
            'RLHF und RLAIF richten ein Modell für viele Nutzende aus. Für Ihr RAG-System sind Suche, Prompt und Modellwahl die günstigeren Hebel.',
            'RLHF and RLAIF align a model for many users. For your RAG system, retrieval, prompt and choice of model are the cheaper levers.')),
        frame(t('Zum Prüfen', 'Check yourself'), t('Können Sie in zwei Sätzen erklären, was RLHF tut und warum Sie es für Ihr RAG-System meist nicht selbst brauchen?',
                                                  'Can you explain in two sentences what RLHF does and why you usually do not need it yourself for your RAG system?')),
        mo.vstack([mo.md(t('#### Nächster Schritt', '#### Next step')),
                   mo.md(t('Zurück zu den Folien.', 'Back to the slides.'))]),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(VERDICTS, mo, t):
    # FAQ: the parking lot of the component plan (PPO and DPO, KL penalty, reward hacking) and how the verdicts came about
    _faq = {
        t('Was machen PPO und DPO genau?', 'What exactly do PPO and DPO do?'): mo.md(t(
            'PPO (Proximal Policy Optimization) ist ein Verfahren aus dem Reinforcement Learning. Das Modell erzeugt Antworten, das Bewertungsmodell vergibt eine Belohnung, und das Modell wird in kleinen Schritten so verändert, dass die Belohnung steigt. DPO (Direct Preference Optimization, [Rafailov et al. 2023, arXiv 2305.18290](https://arxiv.org/abs/2305.18290)) braucht kein eigenes Bewertungsmodell: Es verändert das Modell direkt so, dass die bevorzugte Antwort eines Paars wahrscheinlicher wird als die andere. DPO ist einfacher und günstiger, PPO flexibler.',
            'PPO (proximal policy optimization) is a reinforcement learning method. The model generates answers, the reward model gives a reward, and the model is changed in small steps so that the reward goes up. DPO (direct preference optimization, [Rafailov et al. 2023, arXiv 2305.18290](https://arxiv.org/abs/2305.18290)) needs no separate reward model: it changes the model directly so that the preferred answer of a pair becomes more likely than the other one. DPO is simpler and cheaper, PPO more flexible.')),
        t('Was ist der KL-Strafterm?', 'What is the KL penalty?'): mo.md(t(
            'Beim Optimieren in Schritt 3 soll sich das Modell nicht zu weit vom feinabgestimmten Modell aus Schritt 1 entfernen. Der KL-Strafterm misst den Abstand zwischen beiden und zieht ihn von der Belohnung ab. So bleibt die Sprache natürlich, und das Modell lernt keine seltsamen Antworten, die nur das Bewertungsmodell mag.',
            'During the optimization in step 3, the model should not drift too far from the fine-tuned model of step 1. The KL penalty measures the distance between the two and subtracts it from the reward. This keeps the language natural, and the model does not learn odd answers that only the reward model likes.')),
        t('Was ist Reward Hacking?', 'What is reward hacking?'): mo.md(t(
            'Das Modell findet Wege, eine hohe Belohnung zu bekommen, ohne besser zu werden. Bevorzugt das Bewertungsmodell zum Beispiel lange Antworten, lernt das Modell, lang zu antworten. Abschnitt E zeigt, dass unser Richter genau zu dieser Vorliebe neigt.',
            'The model finds ways to get a high reward without getting better. If the reward model prefers long answers, for example, the model learns to answer at length. Section D shows that our judge leans exactly this way.')),
        **({} if not VERDICTS else {t('Wie sind die Urteile des Richters entstanden?', "How did the judge's verdicts come about?"): mo.vstack([
            mo.md(t(
                f"Vorab berechnet mit {VERDICTS['model']} über die AISC-API, Temperatur {t.num(VERDICTS['temperature'], 1)}. Ohne Grundsätze fünf Läufe je Reihenfolge, mit den Beispiel-Grundsätzen drei. Der Richter sah, was Sie sehen: Frage, Zeile der Webseite, Antworten A und B. Das Werkzeug dazu ist `tools/precompute_w3_04.py`.",
                f"Precomputed with {VERDICTS['model']} via the AISC API, temperature {t.num(VERDICTS['temperature'], 1)}. Without principles five runs per order, with the example principles three. The judge saw what you see: question, line of the website, answers A and B. The tool for this is `tools/precompute_w3_04.py`.")),
            mo.accordion({t('Prompt des Richters', "The judge's prompt"): mo.md(f"```text\n{VERDICTS['prompt_plain']}\n```")}),
        ])}),
    }
    mo.vstack([mo.md(t('## Häufige Fragen', '## Frequently asked questions')), mo.accordion(_faq)])
    return


if __name__ == "__main__":
    app.run()
