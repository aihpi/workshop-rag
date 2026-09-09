"""Render the three metric animations w2_01 shows after participants have played with the dashboard.

Each loop changes one property of a hit list and draws the four correctness metrics beside it, so the
disagreement between them becomes visible rather than asserted. Written once, committed as GIFs, so
the notebook needs no animation library at runtime.

    uv run python -m tools.make_metric_animations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from ragkit import theme
from ragkit.search import ndcg_graded

OUT_DIR = Path(__file__).resolve().parents[1] / 'public' / 'img'
N_RANKS = 8
FPS = 1.2


def metrics(grades: list[int], k: int) -> dict[str, float]:
    positive = [i for i, g in enumerate(grades) if g > 0]
    inside = [i for i in positive if i < k]
    return {
        'MRR': 1.0 / (positive[0] + 1) if positive else 0.0,
        f'nDCG@{k}': ndcg_graded(grades, k),
        f'Recall@{k}': len(inside) / len(positive) if positive else 0.0,
        f'Precision@{k}': len(inside) / k,
    }


def draw(ax_list, ax_bars, grades, k, subtitle):
    ax_list.clear()
    ax_bars.clear()
    for i, g in enumerate(grades):
        shade = {0: 'white', 1: theme.GREYS[2], 2: theme.INK}[g]
        ax_list.add_patch(plt.Rectangle((i, 0), 0.86, 1, facecolor=shade,
                                        edgecolor=theme.INK, linewidth=0.8))
        ax_list.text(i + 0.43, -0.3, str(i + 1), ha='center', va='top', fontsize=8, color=theme.INK)
    ax_list.axvline(k - 0.07, color=theme.ACCENT, linewidth=1.6)
    ax_list.text(k - 0.07, 1.2, f'k = {k}', color=theme.ACCENT, fontsize=8.5, ha='center')
    ax_list.set_xlim(-0.1, N_RANKS + 0.4)
    ax_list.set_ylim(-0.8, 1.7)
    ax_list.set_title(subtitle, fontsize=9, color=theme.INK)
    ax_list.set_axis_off()

    values = metrics(grades, k)
    names = list(values)
    ax_bars.barh(range(len(names)), [values[n] for n in names], height=0.6,
                 color=theme.GREYS[1], edgecolor=theme.INK, linewidth=0.7)
    for i, name in enumerate(names):
        ax_bars.text(1.02, i, f'{values[name]:.2f}', va='center', fontsize=8.5, color=theme.INK)
    ax_bars.set_yticks(range(len(names)), names, fontsize=8.5)
    ax_bars.set_xlim(0, 1.18)
    ax_bars.set_xticks([0, 0.5, 1.0])
    ax_bars.invert_yaxis()
    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)


def animate(frames, filename, title):
    fig, (ax_list, ax_bars) = plt.subplots(2, 1, figsize=(5.6, 3.6), layout='constrained',
                                           gridspec_kw={'height_ratios': [1, 1.5]})
    fig.suptitle(title, fontsize=10, color=theme.INK)

    def update(index):
        grades, k, subtitle = frames[index]
        draw(ax_list, ax_bars, grades, k, subtitle)

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    animation.save(OUT_DIR / filename, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f'wrote {OUT_DIR / filename}', flush=True)


def main():
    theme.apply_mpl()

    sinking = []
    for position in range(N_RANKS):
        grades = [0] * N_RANKS
        grades[position] = 2
        sinking.append((grades, 5, f'the one relevant chunk is at rank {position + 1}'))
    animate(sinking, 'metric_rank_sinks.gif',
            'One relevant chunk, sinking down the list')

    moving_k = []
    base = [0, 0, 2, 0, 0, 1, 0, 0]
    for k in range(1, N_RANKS + 1):
        moving_k.append((base, k, f'the cut-off moves to k = {k}'))
    animate(moving_k, 'metric_cutoff_moves.gif',
            'The same hit list, a different cut-off')

    piling = []
    grades = [0] * N_RANKS
    for step, position in enumerate([1, 3, 4, 6]):
        grades = list(grades)
        grades[position] = 2
        piling.append((grades, 5, f'{step + 1} relevant chunks in the corpus'))
    animate(piling, 'metric_more_relevant.gif',
            'More chunks become relevant, one at a time')


if __name__ == '__main__':
    main()
