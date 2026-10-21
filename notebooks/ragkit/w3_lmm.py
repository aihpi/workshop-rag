"""Linear mixed models for w3_02: score ~ TOP_K + (1 | question), fitted once at precompute time.

    uv run python -m tools.precompute_w3_02 --stage lmm

writes data/cache/w3_02/lmm.json, which the notebook reads (section G and the extra task):

    {'topk': {metric: model}, 'configs': {name: {metric: model}}}

A model holds the fixed effects (`coef`), the variance of the random intercept per question (`group_var`) and the
residual variance (`resid_var`), the estimated mean per level (`means`), all pairwise contrasts with Holm-adjusted
p-values (`contrasts`) and the statsmodels summary as text. Scores are bounded in [0, 1]; the linear model is an
approximation that is good enough for the comparison of means over 43 questions.
"""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from . import w3_eval as ev

METRICS = ('context_precision', 'context_recall')
REFERENCE_TOPK = 3


def _fit(df: pd.DataFrame, metric: str, factor: str, reference, levels: list) -> dict:
    import statsmodels.formula.api as smf
    from scipy import stats

    ref = repr(reference)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = smf.mixedlm(f'{metric} ~ C({factor}, Treatment(reference={ref}))', df, groups=df['question']).fit(reml=True)
    k_fe = len(res.fe_params)
    cov = np.asarray(res.cov_params())[:k_fe, :k_fe]
    names = list(res.fe_params.index)

    def _idx(level):
        """Index of the dummy of `level` in the fixed effects; None for the reference level."""
        if level == reference:
            return None
        return names.index(next(n for n in names if n.endswith(f'[T.{level}]')))

    def _lin(weights):
        """Estimate, standard error, z, p and 95 % interval of a linear combination of the fixed effects."""
        w = np.zeros(k_fe)
        for i, v in weights:
            w[i] += v
        est = float(w @ res.fe_params.values)
        se = float(np.sqrt(w @ cov @ w))
        z = est / se if se else 0.0
        p = float(2 * stats.norm.sf(abs(z)))
        return {'estimate': est, 'se': se, 'z': z, 'p': p, 'lo': est - 1.96 * se, 'hi': est + 1.96 * se}

    coef = [{'term': n, 'level': None if n == 'Intercept' else type(reference)(n.split('[T.')[1].rstrip(']')),
             **_lin([(i, 1.0)])} for i, n in enumerate(names)]
    means = [{'level': lv, **_lin([(0, 1.0)] + ([(_idx(lv), 1.0)] if _idx(lv) is not None else []))} for lv in levels]
    contrasts = []
    for a, b in combinations(levels, 2):
        w = ([(_idx(b), 1.0)] if _idx(b) is not None else []) + ([(_idx(a), -1.0)] if _idx(a) is not None else [])
        contrasts.append({'a': a, 'b': b, **_lin(w)})
    # Holm correction over the pairwise contrasts of this model
    order = sorted(range(len(contrasts)), key=lambda i: contrasts[i]['p'])
    m, running = len(contrasts), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * contrasts[i]['p'])
        contrasts[i]['p_holm'] = min(1.0, running)
    return {'metric': metric, 'factor': factor, 'reference': reference, 'levels': levels, 'n_groups': int(res.model.n_groups),
            'n_obs': int(res.nobs), 'coef': coef, 'group_var': float(res.cov_re.iloc[0, 0]), 'resid_var': float(res.scale),
            'means': means, 'contrasts': contrasts, 'summary': str(res.summary())}


def topk_frame(name: str = 'standard') -> pd.DataFrame:
    topk = ev.load_config(name)['topk']
    return pd.DataFrame([{'question': q, 'topk': k, **{m: topk[str(k)][m][q] for m in METRICS}}
                         for k in ev.TOP_K_VALUES for q in range(43)])


def configs_frame(name: str, top_k: int = 5) -> pd.DataFrame:
    rows = []
    for cfg in ('standard', name):
        topk = ev.load_config(cfg)['topk'][str(top_k)]
        rows += [{'question': q, 'cfg': cfg, **{m: topk[m][q] for m in METRICS}} for q in range(43)]
    return pd.DataFrame(rows)


def fit_all() -> dict:
    """All models of w3_02: TOP_K for the standard configuration, and each other configuration against standard."""
    df = topk_frame()
    out = {'topk': {m: _fit(df, m, 'topk', REFERENCE_TOPK, list(ev.TOP_K_VALUES)) for m in METRICS}, 'configs': {}}
    for name in ev.CONFIGS:
        if name == 'standard':
            continue
        dfc = configs_frame(name)
        out['configs'][name] = {m: _fit(dfc, m, 'cfg', 'standard', ['standard', name]) for m in METRICS}
    return out
