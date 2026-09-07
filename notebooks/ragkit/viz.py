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


def image_grid(query_path, hit_paths, scores, labels=None, credit: str | None = None):
    """Query photo on the left, the ranked hits in one row, similarity under each hit.

    `labels` (optional) are short names printed above the hits, e.g. the species.
    The query is framed in ink; a credit line goes under the figure when given.
    """
    from matplotlib.image import imread

    k = len(hit_paths)
    fig, axes = plt.subplots(1, k + 1, figsize=(6.5, 6.5 / (k + 1) + 0.5),
                             gridspec_kw={'wspace': 0.08, 'width_ratios': [1.15] + [1] * k})
    for ax in axes:
        ax.set_axis_off()
    axes[0].imshow(imread(query_path))
    axes[0].set_title('query', fontsize=8.5, color=theme.INK)
    axes[0].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[0].transAxes, fill=False,
                                    edgecolor=theme.INK, linewidth=1.5))
    for i, (ax, path, score) in enumerate(zip(axes[1:], hit_paths, scores)):
        ax.imshow(imread(path))
        title = f'{labels[i]}\n{score:.2f}' if labels else f'{score:.2f}'
        ax.set_title(title, fontsize=8, color=theme.GREY)
    if credit:
        fig.text(0.01, -0.02, credit, fontsize=7, color=theme.GREY)
    return fig


def thumbnail_sheet(paths, labels, cols: int = 13):
    """Small thumbnails in a grid with a label under each; for a corpus overview."""
    from matplotlib.image import imread

    rows = -(-len(paths) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.5, 6.5 / cols * rows * 1.25))
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, path, label in zip(axes.flat, paths, labels):
        ax.imshow(imread(path))
        ax.set_title(label, fontsize=6, color=theme.INK, pad=2)
    return fig
