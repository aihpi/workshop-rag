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
    import matplotlib.pyplot as plt  # the before-and-after plot
    import numpy as np  # medians
    import pandas as pd  # the precomputed result tables

    from ragkit import submit, theme  # anonymous score submission, HPI look
    from ragkit.config import DATA_DIR
    from ragkit.viz import md_table

    theme.apply_mpl()
    return DATA_DIR, md_table, mo, np, pd, plt, submit, theme


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
    - **Score**: Recall@5, the share of questions whose answer passage is in the top five.
    - **Budget**: {BUDGET} evaluations per round, because in real life every experiment costs time.

    You have five minutes. Find the best configuration you can, then submit your score.

    *There are {len(scores)} configurations in total, so you cannot try them all. Choose.*
    """)
    return


@app.cell(hide_code=True)
def _(mo, scores, submit):
    state = submit.load_state()

    def _dropdown(column, label_of=str, prefer=None, **kwargs):
        """A dropdown over what was actually measured, starting on `prefer` when it exists."""
        options = {label_of(v): v for v in sorted(scores[column].dropna().unique())}
        start = prefer if prefer in options else next(iter(options))
        return mo.ui.dropdown(options, value=start, **kwargs)

    def _int_label(value):
        return str(int(value))

    ui_round = mo.ui.dropdown({'First try': '1', 'Second try': '2'}, value='First try', label='round')
    ui_strategy = _dropdown('strategy', prefer='chars', label='strategy')
    ui_size = _dropdown('size', _int_label, prefer='1200', label='size')
    ui_overlap = _dropdown('overlap', _int_label, prefer='0', label='overlap')
    ui_model = _dropdown('model', prefer='octen', label='model')
    ui_title = mo.ui.checkbox(value=False, label='prepend the section title')
    ui_evaluate = mo.ui.run_button(label='Evaluate')

    mo.vstack([
        mo.md(f'You are **{state["handle"]}**. That name is all anyone will ever see.'),
        mo.hstack([ui_round], justify='start'),
        mo.hstack([ui_strategy, ui_size, ui_overlap, ui_model], justify='start'),
        mo.hstack([ui_title, ui_evaluate], justify='start'),
    ])
    return (
        state,
        ui_evaluate,
        ui_model,
        ui_overlap,
        ui_round,
        ui_size,
        ui_strategy,
        ui_title,
    )


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

    _match = scores[(scores.strategy == ui_strategy.value) & (scores.model == ui_model.value)
                    & (scores.overlap == ui_overlap.value) & (scores.prepend_title == ui_title.value)]
    if ui_strategy.value != 'paragraph':
        _match = _match[_match['size'] == ui_size.value]

    _round = ui_round.value
    if _match.empty:
        # Not every combination of the four controls was measured; saying so is better than
        # leaving the previous result on screen, and it costs no evaluation.
        _view = mo.callout(mo.md(
            'That combination was not measured, so it costs you nothing. Try another one.'),
            kind='neutral')
    else:
        _row = _match.iloc[0]
        set_log(lambda log: {**log, _round: log.get(_round, []) + [{
            'config_id': _row.config_id, 'Recall@5': float(_row['Recall@5']),
            'MRR': float(_row.MRR), 'nDCG@5': float(_row['nDCG@5']),
            'Recall@5_easy': float(_row['Recall@5_easy']),
            'Recall@5_complex': float(_row['Recall@5_complex']),
            'chunks': int(_row.chunks),
        }]})
        _view = mo.md('')
    _view
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
    mo.stop(not _tried, mo.md(''))

    ui_question = mo.ui.dropdown(
        {f'{q.difficulty}: {q.question[:90]}': q.qid for q in questions.itertuples()},
        value=f'{questions.iloc[0].difficulty}: {questions.iloc[0].question[:90]}',
        label='inspect one question', full_width=True)
    ui_question
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
def _(get_log, mo, ui_round):
    _tried = get_log().get(ui_round.value, [])
    mo.stop(not _tried, mo.md('*Evaluate something first.*'))

    ui_consent = mo.ui.checkbox(value=False, label='submit my best score for this round')
    ui_send = mo.ui.run_button(label='Submit')
    mo.hstack([ui_consent, ui_send], justify='start')
    return ui_consent, ui_send


@app.cell(hide_code=True)
def _(get_log, mo, state, submit, ui_consent, ui_round, ui_send):
    mo.stop(not ui_send.value or not ui_consent.value, mo.md(''))

    _tried = get_log().get(ui_round.value, [])
    _best = max(_tried, key=lambda r: r['Recall@5'])
    _payload = {'handle': state['handle'], 'try': ui_round.value,
                'config': _best['config_id'], 'recall_at_5': round(_best['Recall@5'], 4),
                'mrr': round(_best['MRR'], 4), 'ndcg_at_5': round(_best['nDCG@5'], 4),
                'evaluations_used': len(_tried)}
    _created = submit.submit_with_gh(_payload)
    _view = (mo.md(f'Submitted: {_created}') if _created else mo.md(
        f'[**Open the pre-filled issue and press the green button**]({submit.issue_url(_payload)})'))
    mo.vstack([_view, mo.md(f'`{submit.encode_body(_payload)}`')])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### How the room did

    For the instructor. A session appears here once its submissions carry a `session-YYYY-MM-DD` label, so a score sent in afterwards cannot change a past session.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ui_refresh = mo.ui.run_button(label='Load sessions')
    ui_refresh
    return (ui_refresh,)


@app.cell(hide_code=True)
def _(mo, submit, ui_refresh):
    mo.stop(not ui_refresh.value, mo.md('*Press Load sessions.*'))
    try:
        _labels = submit.session_labels()
    except OSError as exc:
        _labels = []
        mo.output.append(mo.callout(mo.md(f'GitHub is not reachable: `{exc}`'), kind='warn'))
    ui_session = mo.ui.dropdown({lab: lab for lab in _labels},
                                value=_labels[0] if _labels else None, label='session')
    ui_session
    return (ui_session,)


@app.cell(hide_code=True)
def _(mo, np, plt, scores, submit, theme, ui_session):
    mo.stop(ui_session.value is None, mo.md('*No confirmed session yet.*'))

    _paired = submit.pair_tries(submit.fetch_session(ui_session.value))
    mo.stop(not _paired, mo.md('*No paired submissions in this session yet.*'))

    _gold = float(scores['Recall@5'].max())
    _fig, _ax = plt.subplots(figsize=(6.4, 4.0))
    for _row in _paired:
        _ax.plot([0, 1], [_row['first'], _row['second']], color=theme.GREY, linewidth=1, alpha=0.7)
    _medians = [float(np.median([r['first'] for r in _paired])),
                float(np.median([r['second'] for r in _paired]))]
    _ax.plot([0, 1], _medians, color=theme.ACCENT, linewidth=2.5, marker='o', label='group median')
    _ax.axhline(_gold, linestyle='--', color=theme.INK, linewidth=1, label=f'best known {_gold:.1%}')
    _ax.set_xticks([0, 1], ['first try', 'second try'])
    _ax.set_xlim(-0.15, 1.15)
    _ax.set_ylabel('Recall@5')
    _ax.legend(frameon=False, loc='lower right')
    mo.vstack([
        _fig,
        mo.md(f'*{len(_paired)} participants. Median {_medians[0]:.1%} to {_medians[1]:.1%}, '
              f'best known configuration {_gold:.1%}.*'),
    ])
    return


if __name__ == "__main__":
    app.run()
