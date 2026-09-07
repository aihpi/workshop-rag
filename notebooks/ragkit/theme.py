"""HPI corporate design for the workshop notebooks.

Palette and rules follow the HPI Corporate Design Manual (May 2025). Figures are
grey-first: `palette()` hands out greys before any hue, and `apply_mpl()` sets
matplotlib to thin axes, no grid and Inter (or the nearest installed sans).
marimo notebooks pass `CSS_FILE` and `HEAD_FILE` to `marimo.App`.
"""

from pathlib import Path

import matplotlib as mpl

# --- Palette -----------------------------------------------------------------
PRIMARY = {'red': '#b1063a', 'orange': '#dd6108', 'yellow': '#f6a800', 'grey': '#5a6065'}
SECONDARY = {'violet': '#7664a0', 'blue': '#007a9e', 'light_green': '#779a0b', 'dark_green': '#007f6f'}
INK = '#212427'        # body text (HPI Dark Grey)
GREY = PRIMARY['grey']
RULE = '#d6d7d8'       # HPI Grey at 25 %
ACCENT = PRIMARY['red']
NOTE = '#fde9bf'       # HPI Yellow at 25 %
GREYS = ['#212427', '#5a6065', '#83888c', '#acb0b2', '#d6d7d8']

_HERE = Path(__file__).resolve().parent
CSS_FILE = _HERE / 'theme.css'
HEAD_FILE = _HERE / 'head.html'


def tint(hex_colour: str, percent: int) -> str:
    """Mix a colour with white; `percent` is the share of colour (100, 75, 50, 25)."""
    if not 0 <= percent <= 100:
        raise ValueError('percent must be within 0..100')
    h = hex_colour.lstrip('#')
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = 1 - percent / 100
    return '#' + ''.join(f'{round(c + f * (255 - c)):02x}' for c in rgb)


def palette(n: int, accent: str | None = None) -> list[str]:
    """`n` distinguishable colours, greys first; the optional accent goes last."""
    if n > len(GREYS) + (1 if accent else 0):
        raise ValueError(f'at most {len(GREYS)} grey classes per panel; use small multiples '
                         'or text labels, or ask the maintainer about colour')
    cols = GREYS[:n - 1] + [accent] if accent else GREYS[:n]
    return cols


def mpl_rc(accent: str | None = None) -> dict:
    """rcParams for the grey-first figure style."""
    cycle = GREYS[:4] if accent is None else [GREYS[0], accent, GREYS[2], GREYS[3]]
    return {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Verdana', 'Arial', 'DejaVu Sans'],
        'font.size': 9.5,
        'axes.titlesize': 10, 'axes.titleweight': 'medium', 'axes.titlelocation': 'left',
        'axes.labelsize': 9.5, 'axes.labelcolor': INK,
        'axes.edgecolor': GREY, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'axes.axisbelow': True,
        'axes.prop_cycle': mpl.cycler(color=cycle),
        'xtick.color': GREY, 'ytick.color': GREY,
        'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'lines.linewidth': 1.4, 'lines.markersize': 4.5,
        'patch.edgecolor': INK, 'patch.linewidth': 0.6,
        'hatch.linewidth': 0.6, 'hatch.color': INK,
        'legend.frameon': True, 'legend.edgecolor': RULE, 'legend.framealpha': 1,
        'legend.fontsize': 8.5,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'figure.figsize': (6.5, 3.4),
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'text.color': INK,
    }


def apply_mpl(accent: str | None = None) -> None:
    """Install the figure style process-wide."""
    mpl.rcParams.update(mpl_rc(accent))


def header_html(title: str, subtitle: str = '', logos: bool = True) -> str:
    """Notebook header: title, the single red rule, optional subtitle and logos.

    Logo files are served by marimo from `<notebook_dir>/public/img/`.
    """
    logo_html = (
        '<div class="hpi-logos">'
        '<img src="/public/img/logo_aisc_150dpi.png" alt="HPI AI Service Centre">'
        '<img src="/public/img/logo_bmftr_de.png" alt="BMFTR">'
        '</div>'
    ) if logos else ''
    sub = f'<p class="hpi-subtitle">{subtitle}</p>' if subtitle else ''
    return f'<div class="hpi-header">{logo_html}<h1>{title}</h1><div class="hpi-title-rule"></div>{sub}</div>'
