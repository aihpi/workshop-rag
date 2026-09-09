"""Figure helpers.

The density estimator and the figure-to-base64 encoder that more than one cell
needs, plus the three w2_01 plots (length histograms, sorted scores, score
densities) that repeat for every chunking strategy. Plots that are themselves
the lesson stay in the notebooks, where participants read the plotting code.
"""

import base64
import io

import matplotlib.pyplot as plt
import numpy as np

from . import theme


def gaussian_kde(data, x_grid, bandwidth=None):
    """Gaussian kernel density estimate, using numpy only (no SciPy needed).

    Bandwidth defaults to Silverman's rule of thumb.
    """
    data = np.asarray(data, dtype=float)
    if bandwidth is None:
        bandwidth = 1.06 * np.std(data) * len(data) ** (-1 / 5)
    if bandwidth < 1e-6:
        bandwidth = 0.01
    kernels = np.exp(-0.5 * ((x_grid[:, None] - data[None, :]) / bandwidth) ** 2)
    return kernels.sum(axis=1) / (len(data) * bandwidth * np.sqrt(2 * np.pi))


def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a PNG base64 string, closing the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def md_table(rows) -> str:
    """Rows of dicts as a markdown table, taking the column order from the first row.

    `mo.ui.table` is a widget with its own scroll container, so a table taller than the
    cell gets a second scrollbar inside the page. A markdown table is plain HTML and
    grows with the page, which is what the notebooks want.
    """
    rows = list(rows)
    if not rows:
        return ''
    head = list(rows[0])
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    lines += ['| ' + ' | '.join(str(r[k]) for k in head) + ' |' for r in rows]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# w2_01 chunking figures
# ---------------------------------------------------------------------------

def length_hist_panels(lengths_by_strategy: dict[str, list[int]], bins: int = 40,
                       log_y: bool = True):
    """Small multiples of chunk-length histograms, one panel per strategy, shared x.

    All panels use the same bin edges so bar widths are comparable; the y axis
    is per panel (log by default, so the long tails stay visible).
    """
    names = list(lengths_by_strategy)
    fig, axes = plt.subplots(1, len(names), figsize=(min(6.5, 1.6 * len(names) + 0.6), 2.4),
                             sharex=True, squeeze=False, constrained_layout=True)
    longest = max(max(v) for v in lengths_by_strategy.values())
    edges = np.linspace(0, longest, bins + 1)
    for ax, name in zip(axes[0], names):
        ax.hist(lengths_by_strategy[name], bins=edges, color=theme.GREYS[2], edgecolor='none')
        ax.set_title(name)
        if log_y:
            ax.set_yscale('log')
    axes[0, 0].set_ylabel('chunks')
    fig.supxlabel('chunk length (characters)')
    return fig


def length_hist_panels_from_bins(bins_by_strategy: dict[str, tuple], log_y: bool = True):
    """The same small multiples, drawn from pre-binned counts rather than raw lengths.

    Each value is `(left_edges, right_edges, counts)`. This is what a notebook
    needs when the histogram was computed offline and only the bins were kept,
    so nothing has to be re-chunked to draw the figure.
    """
    names = list(bins_by_strategy)
    fig, axes = plt.subplots(1, len(names), figsize=(min(6.5, 1.6 * len(names) + 0.6), 2.4),
                             sharex=True, squeeze=False, constrained_layout=True)
    widest = max(max(right) for _, right, _ in bins_by_strategy.values())
    for ax, name in zip(axes[0], names):
        left, right, counts = bins_by_strategy[name]
        ax.bar(left, counts, width=[b - a for a, b in zip(left, right)], align='edge',
               color=theme.GREYS[2], edgecolor='none')
        ax.set_title(name)
        ax.set_xlim(0, widest)
        if log_y:
            ax.set_yscale('log')
    axes[0, 0].set_ylabel('chunks')
    fig.supxlabel('chunk length (characters)')
    return fig


def sorted_score_plot(scores, highlight_index: int | None = None, ax=None):
    """Scores sorted descending as a grey line with hollow markers; one index filled black.

    `highlight_index` refers to the position in `scores` (e.g. the gold chunk),
    not to its rank.
    """
    if ax is None:
        ax = plt.gca()
    order = np.argsort(scores)[::-1]
    ranked = np.asarray(scores, dtype=float)[order]
    ranks = np.arange(1, len(ranked) + 1)
    ax.plot(ranks, ranked, color=theme.GREY, marker='o', markerfacecolor='white',
            markeredgecolor=theme.GREY)
    if highlight_index is not None:
        pos = int(np.flatnonzero(order == highlight_index)[0])
        ax.plot(ranks[pos], ranked[pos], marker='o', color='#000000', linestyle='none')
    ax.set_xlabel('rank')
    ax.set_ylabel('score')
    return ax


def kde_plot(relevant_scores, other_scores, ax=None):
    """Score densities: relevant solid ink, other dashed grey, labelled at their peaks."""
    if ax is None:
        ax = plt.gca()
    relevant = np.asarray(relevant_scores, dtype=float)
    other = np.asarray(other_scores, dtype=float)
    lo, hi = min(relevant.min(), other.min()), max(relevant.max(), other.max())
    pad = 0.1 * (hi - lo) or 0.05
    grid = np.linspace(lo - pad, hi + pad, 300)
    for data, label, colour, style in ((other, 'other', theme.GREY, '--'),
                                       (relevant, 'relevant', theme.INK, '-')):
        density = gaussian_kde(data, grid)
        ax.plot(grid, density, color=colour, linestyle=style)
        peak = int(np.argmax(density))
        ax.annotate(label, (grid[peak], density[peak]), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom',
                    color=colour, fontsize=8.5)
    ax.set_xlabel('similarity score')
    ax.set_ylabel('density')
    return ax


def head_tail(text: str, n: int = 120) -> str:
    """First and last `n` characters joined by ' … ', to show where a chunk begins and ends."""
    if len(text) <= 2 * n:
        return text
    return f'{text[:n]} … {text[-n:]}'
