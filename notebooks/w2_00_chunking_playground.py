# ruff: noqa: PLR1711, B018  marimo cells end with an explicit return and render a bare expression

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Chunking playground",
    css_file="ragkit/theme.css",
    html_head_file="ragkit/head.html",
)


@app.cell(hide_code=True)
def _(mo, theme):
    mo.Html(theme.header_html(
        'Chunking playground',
        'RAG II, notebook 0. Find the best way to cut a corpus, twice: once now, once at the end.',
    ))
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo  # the reactive notebook itself
    import pandas as pd  # the precomputed result tables

    from ragkit import submit, theme  # anonymous score submission, HPI look
    from ragkit.config import DATA_DIR
    from ragkit.viz import md_table

    theme.apply_mpl()
    return DATA_DIR, md_table, mo, pd, submit, theme


@app.cell(hide_code=True)
def _(DATA_DIR, mo, pd):
    GRID_DIR = DATA_DIR / 'grid'
    _missing = [f for f in ('grid_scores.parquet', 'grid_hits.parquet', 'grid_questions.parquet')
                if not (GRID_DIR / f).exists()]
    mo.stop(bool(_missing), mo.callout(mo.md(
        f"**The result tables are missing:** {', '.join(_missing)}.\n\n"
        'Build them once with `uv run python -m tools.run_grid --stage all` from `notebooks/`.'
    ), kind='warn'))

    scores = pd.read_parquet(GRID_DIR / 'grid_scores.parquet')
    hits = pd.read_parquet(GRID_DIR / 'grid_hits.parquet')
    questions = pd.read_parquet(GRID_DIR / 'grid_questions.parquet')
    BUDGET = 8
    return BUDGET, hits, questions, scores


@app.cell(hide_code=True)
def _(BUDGET, mo, questions, scores):
    mo.md(rf"""
    ## Your task

    A vector store does not retrieve documents, it retrieves **chunks**: the pieces of text that were cut out, embedded and indexed. Below you can cut the IT-Grundschutz-Kompendium in different ways and see how often the passage that answers a question ends up in the top five hits.

    - **Corpus**: all 111 Bausteine of the Kompendium, about 2.5 million characters.
    - **Questions**: {len(questions)} real questions with a known answer passage.
    - **Score**: three numbers, measured on the same questions. Your submission is ranked on Recall@5.
        - **Recall@5**: how often the passage that answers a question is among the five chunks that come back.
        - **MRR**: how high up that passage sits when it does come back, first place counting for more than fifth.
        - **nDCG@5**: the same idea as MRR, but it also rewards having several useful chunks among the five rather than only one.
    - **Budget**: {BUDGET} evaluations per round, because in real life every experiment costs time.

    You have five minutes. Find the best configuration you can, then submit your score.

    *There are {len(scores)} configurations in total, so you cannot try them all. Choose.*
    """)
    return


# The controls form a chain: each one offers only the values still reachable given the ones above
# it, so every configuration a participant can build is one the grid has measured. The chain needs
# one cell per control, because a marimo cell cannot read a variable it defines itself.


@app.cell(hide_code=True)
def _(mo, scores, submit):
    state = submit.load_state()

    def choose(frame, column, label_of=str, prefer=None, **kwargs):
        """A dropdown over what `frame` still offers, starting on `prefer` when that survives."""
        options = {label_of(v): v for v in sorted(frame[column].dropna().unique())}
        options = options or {'not applicable': None}
        start = prefer if prefer in options else next(iter(options))
        return mo.ui.dropdown(options, value=start, **kwargs)

    def narrow(frame, column, value):
        """The rows still reachable once `column` is fixed. A `None` value fixes nothing."""
        return frame if value is None else frame[frame[column] == value]

    def int_label(value):
        return str(int(value))

    def title_label(value):
        return 'yes' if value else 'no'

    ui_round = mo.ui.dropdown({'First try': '1', 'Second try': '2'}, value='First try', label='round')
    ui_strategy = choose(scores, 'strategy', prefer='chars', label='strategy')
    return choose, int_label, narrow, state, title_label, ui_round, ui_strategy


@app.cell(hide_code=True)
def _(choose, narrow, scores, ui_strategy):
    after_strategy = narrow(scores, 'strategy', ui_strategy.value)
    ui_model = choose(after_strategy, 'model', prefer='octen', label='model')
    return after_strategy, ui_model


@app.cell(hide_code=True)
def _(after_strategy, choose, int_label, narrow, ui_model):
    # `paragraph` cuts on the text's own breaks, so it has no size; the dropdown then reads
    # 'not applicable' and the lookup below leaves size out of the filter.
    after_model = narrow(after_strategy, 'model', ui_model.value)
    ui_size = choose(after_model, 'size', int_label, prefer='1200', label='size')
    return after_model, ui_size


@app.cell(hide_code=True)
def _(after_model, choose, int_label, narrow, ui_size):
    after_size = narrow(after_model, 'size', ui_size.value)
    ui_overlap = choose(after_size, 'overlap', int_label, prefer='0', label='overlap')
    return after_size, ui_overlap


@app.cell(hide_code=True)
def _(after_size, choose, narrow, title_label, ui_overlap):
    after_overlap = narrow(after_size, 'overlap', ui_overlap.value)
    ui_title = choose(after_overlap, 'prepend_title', title_label, prefer='no',
                      label='prepend the section title')
    return ui_title


@app.cell(hide_code=True)
def _(mo, state, ui_model, ui_overlap, ui_round, ui_size, ui_strategy, ui_title):
    ui_evaluate = mo.ui.run_button(label='Evaluate')

    mo.vstack([
        mo.md(f'You are **{state["handle"]}**. That name is all anyone will ever see.'),
        mo.hstack([ui_round], justify='start'),
        mo.hstack([ui_strategy, ui_model, ui_size, ui_overlap], justify='start'),
        mo.hstack([ui_title, ui_evaluate], justify='start'),
        mo.md('*Each control offers only what the ones before it leave available, so every '
              'configuration you can build here has been measured.*'),
    ])
    return (ui_evaluate,)


@app.cell(hide_code=True)
def _(mo):
    get_log, set_log = mo.state({})
    return get_log, set_log


@app.cell(hide_code=True)
def _(
    mo,
    scores,
    set_log,
    ui_evaluate,
    ui_model,
    ui_overlap,
    ui_round,
    ui_size,
    ui_strategy,
    ui_title,
):
    mo.stop(not ui_evaluate.value)

    # The cascading controls cannot produce a combination that is absent from the table, so this
    # always matches. `size` is left out when the strategy has none, which is the `paragraph` case.
    _match = scores[(scores.strategy == ui_strategy.value) & (scores.model == ui_model.value)
                    & (scores.overlap == ui_overlap.value) & (scores.prepend_title == ui_title.value)]
    if ui_size.value is not None:
        _match = _match[_match['size'] == ui_size.value]

    _round = ui_round.value
    _row = _match.iloc[0]
    set_log(lambda log: {**log, _round: log.get(_round, []) + [{
        'config_id': _row.config_id, 'Recall@5': float(_row['Recall@5']),
        'MRR': float(_row.MRR), 'nDCG@5': float(_row['nDCG@5']),
        'Recall@5_easy': float(_row['Recall@5_easy']),
        'Recall@5_complex': float(_row['Recall@5_complex']),
        'chunks': int(_row.chunks),
    }]})
    return


@app.cell(hide_code=True)
def _(BUDGET, get_log, md_table, mo, ui_round):
    _tried = get_log().get(ui_round.value, [])
    _left = BUDGET - len(_tried)

    if not _tried:
        _view = mo.md('*Choose a configuration and press Evaluate.*')
    elif _left < 0:
        _view = mo.callout(mo.md('**Budget spent.** The evaluations above are what you have; '
                                 'pick your best one and submit it.'), kind='warn')
    else:
        _last = _tried[-1]
        _best = max(_tried, key=lambda r: r['Recall@5'])
        _view = mo.vstack([
            mo.hstack([
                mo.stat(f"{_last['Recall@5']:.1%}", label='Recall@5', caption=_last['config_id']),
                mo.stat(f"{_last['MRR']:.3f}", label='MRR'),
                mo.stat(f"{_last['nDCG@5']:.3f}", label='nDCG@5'),
                mo.stat(f"{_last['chunks']:,}", label='chunks'),
            ], justify='start'),
            mo.md(f"Easy questions {_last['Recall@5_easy']:.1%}, complex questions "
                  f"{_last['Recall@5_complex']:.1%}. "
                  f"**Best so far: {_best['Recall@5']:.1%}** ({_best['config_id']}). "
                  f"**{max(_left, 0)} of {BUDGET} evaluations left.**"),
            mo.md(md_table([{k: r[k] for k in ('config_id', 'Recall@5', 'MRR', 'nDCG@5')}
                            for r in _tried])),
        ])
    _view
    return


@app.cell(hide_code=True)
def _(get_log, mo, questions, ui_round):
    _tried = get_log().get(ui_round.value, [])

    ui_question = mo.ui.dropdown(
        {f'{q.difficulty}: {q.question[:90]}': q.qid for q in questions.itertuples()},
        value=f'{questions.iloc[0].difficulty}: {questions.iloc[0].question[:90]}',
        label='inspect one question', full_width=True)
    ui_question if _tried else mo.md('')
    return (ui_question,)


@app.cell(hide_code=True)
def _(get_log, hits, mo, ui_question, ui_round):
    _tried = get_log().get(ui_round.value, [])
    mo.stop(not _tried or ui_question.value is None, mo.md(''))

    _config = _tried[-1]['config_id']
    _rows = hits[(hits.config_id == _config) & (hits.qid == ui_question.value)].nsmallest(5, 'rank')
    _items = []
    for _hit in _rows.itertuples():
        _mark = '**relevant**' if _hit.relevant else 'not relevant'
        _text = _hit.text.strip().replace('\n', ' ')[:400]
        _items.append(f'**{_hit.rank}.** ({_hit.score:.3f}, {_mark}) {_text}…')
    mo.vstack([
        mo.md(f'The top five chunks `{_config}` retrieved for this question:'),
        mo.md('\n\n'.join(_items)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Submit your score
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # What went in for each round, which is what greys the button out and what the message below
    # is drawn from. The workers write it; nothing else does.
    get_sent, set_sent = mo.state({})
    return get_sent, set_sent


@app.cell(hide_code=True)
def _(get_log, state):
    def best_payload(attempt):
        """The best evaluation of one round, in the shape the issue body takes."""
        tried = get_log().get(attempt, [])
        best = max(tried, key=lambda row: row['Recall@5'])
        return {'handle': state['handle'], 'try': attempt,
                'config': best['config_id'], 'recall_at_5': round(best['Recall@5'], 4),
                'mrr': round(best['MRR'], 4), 'ndcg_at_5': round(best['nDCG@5'], 4),
                'evaluations_used': len(tried)}
    return (best_payload,)


@app.cell(hide_code=True)
def _(get_log, get_sent, mo, ui_round):
    # Both controls are defined whatever happens and only the rendering is conditional: a cell
    # that stops before defining what its dependents read leaves them with a NameError as soon as
    # marimo runs them without re-running this one.
    _tried = get_log().get(ui_round.value, [])

    # One submission per round. Once this round has gone in the button is spent, and the way to
    # change the score is the overwrite button underneath.
    _done = get_sent().get(ui_round.value)
    ui_consent = mo.ui.checkbox(value=False, label='submit my best score for this round')
    ui_send = mo.ui.run_button(label='Submitted' if _done else 'Submit', disabled=bool(_done))
    (mo.hstack([ui_consent, ui_send], justify='start') if _tried
     else mo.md('*Evaluate something first.*'))
    return ui_consent, ui_send


@app.cell(hide_code=True)
def _(best_payload, mo, set_sent, state, submit, ui_consent, ui_round, ui_send):
    mo.stop(not ui_send.value or not ui_consent.value, mo.md(''))

    _payload = best_payload(ui_round.value)

    # marimo runs one cell at a time, so the button cannot grey out while this cell is working:
    # the spinner is what says the notebook is busy, and the search is what makes a second press
    # harmless whatever the buttons look like.
    with mo.status.spinner(title='Talking to GitHub...'):
        _found = submit.find_existing(state['handle'], ui_round.value, state=state)
        _created = None if _found else submit.submit_with_gh(_payload)

    if _found:
        _outcome = {**_found, 'status': 'existing'}
    elif _created:
        _outcome = {'issue': submit.issue_number(_created), 'url': _created, 'status': 'created'}
        submit.remember_submission(state, ui_round.value, _outcome['issue'], _created)
    else:
        _outcome = {'issue': None, 'url': None, 'status': 'nogh'}

    set_sent(lambda sent: {**sent, ui_round.value: {**_outcome, 'payload': _payload}})
    mo.md('')
    return


@app.cell(hide_code=True)
def _(get_sent, mo, submit, ui_round):
    # The outcome is rendered from the state rather than from the cell that did the work, because
    # that cell re-runs and stops as soon as the button it depends on is rebuilt.
    sent = get_sent().get(ui_round.value)
    ui_overwrite = (mo.ui.run_button(label='Overwrite my submission', kind='warn')
                    if sent and sent['issue'] else None)
    mo.stop(not sent, mo.md(''))

    if sent['status'] == 'nogh':
        _view = mo.callout(mo.md(
            f'[**Open the pre-filled issue and press the green button**]'
            f'({submit.issue_url(sent["payload"])})\n\n'
            'Without `gh` the notebook cannot tell whether you already submitted this round, so '
            'open that form once and no more.'), kind='info')
    elif sent['status'] == 'existing':
        _view = mo.callout(mo.md(
            f'**Try {ui_round.value} is already in.** It is issue [#{sent["issue"]}]({sent["url"]}), '
            'opened by you. Overwrite it if what you have now is better.'), kind='warn')
    elif sent['status'] == 'failed':
        _view = mo.callout(mo.md(
            f'**That did not go through.** Issue [#{sent["issue"]}]({sent["url"]}) is unchanged; '
            'edit it on GitHub, or try the button again.'), kind='danger')
    else:
        _word = 'Submitted' if sent['status'] == 'created' else 'Overwritten'
        _view = mo.md(f'{_word}: {sent["url"]}')

    mo.vstack([_view, mo.md(f'`{submit.encode_body(sent["payload"])}`')]
              + ([ui_overwrite] if ui_overwrite else []))
    return sent, ui_overwrite


@app.cell(hide_code=True)
def _(best_payload, mo, sent, set_sent, submit, ui_overwrite, ui_round):
    mo.stop(not sent or ui_overwrite is None or not ui_overwrite.value, mo.md(''))

    _payload = best_payload(ui_round.value)
    with mo.status.spinner(title='Updating your issue...'):
        _said = submit.update_with_gh(sent['issue'], _payload)

    set_sent(lambda state_: {**state_, ui_round.value: {
        **state_[ui_round.value], 'payload': _payload,
        'status': 'overwritten' if _said else 'failed'}})
    mo.md('')
    return


if __name__ == "__main__":
    app.run()
