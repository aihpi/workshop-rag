"""HPI corporate design for the workshop notebooks.

Palette and rules follow the HPI Corporate Design Manual (May 2025). Figures are
grey-first: `palette()` hands out greys before any hue, and `apply_mpl()` sets
matplotlib to thin axes, no grid and Inter (or the nearest installed sans).
marimo notebooks pass `CSS_FILE` and `HEAD_FILE` to `marimo.App`.
"""

import base64
from functools import cache
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
_IMG_DIR = _HERE.parent / 'public' / 'img'


@cache
def _logo_uri(name: str) -> str:
    """A logo from public/img as a data URI, so it shows without marimo's /public/ route."""
    data = base64.b64encode((_IMG_DIR / name).read_bytes()).decode('ascii')
    return f'data:image/png;base64,{data}'


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


def task_box(label: str, steps: list[str], time: str = '', title: str = '', kind: str = '') -> str:
    """HTML of a task box for every place where participants do something themselves.

    `label` is the localized marker ("Ihre Aufgabe 1 von 4" / "Your task 1 of 4"), `steps` the
    imperative steps (one per line, numbered automatically), `time` a hint like "etwa 2 Minuten",
    `title` an optional short name of the task. `kind` picks the colour coding of the meta plan:
    'task' (HPI Dark Green) or 'extra' (HPI Orange); the default '' keeps the ink frame of the
    first version. Needs css_file="ragkit/theme.css". Show it with mo.Html(theme.task_box(...)).
    Text is escaped; pass plain text.
    """
    from html import escape

    head = (f'<div class="hpi-task-head"><span class="hpi-task-label">{escape(label)}</span>'
            + (f'<span class="hpi-task-time">{escape(time)}</span>' if time else '') + '</div>')
    ttl = f'<div class="hpi-task-title">{escape(title)}</div>' if title else ''
    items = ''.join(f'<li>{escape(s)}</li>' for s in steps)
    cls = 'hpi-task' + (f' hpi-box hpi-box-{kind}' if kind in ('task', 'extra') else '')
    return f'<div class="{cls}">{head}{ttl}<ol>{items}</ol></div>'


def info_box(label: str, body_html: str, kind: str = 'info') -> str:
    """HTML of a colour-coded information box (HPI Yellow by default): a label word, then the body.

    `body_html` is inserted as it is (for example `mo.md(text).text`); `label` is escaped. `kind`
    is 'info', 'task' or 'extra' as in task_box. Show it with mo.Html(theme.info_box(...)).
    """
    from html import escape

    return (f'<div class="hpi-box hpi-box-{kind}"><div class="hpi-box-head"><span class="hpi-box-label">{escape(label)}</span></div>'
            f'{body_html}</div>')


def feedback_box(ok: bool | None, explanation_html: str, t) -> str:
    """HTML of the feedback box for quiz answers and checkable predictions (meta plan 10.4a).

    `ok` True gives a large check mark and "Richtig" / "Correct", False a large cross and "Nicht ganz" /
    "Not quite", None (a choice without a right answer) "Hier gibt es kein Richtig oder Falsch." /
    "There is no right or wrong here." `explanation_html` follows as it is (for example
    `mo.md(text).text`; '' for none). `t` picks the language like ragkit.i18n.T. The box is neutral on
    purpose: white ground and a frame in HPI Dark Grey, no green (task boxes) and no yellow (information).
    Show it with mo.Html(theme.feedback_box(...)).
    """
    if ok is None:
        mark, verdict, state = '', t('Hier gibt es kein Richtig oder Falsch.', 'There is no right or wrong here.'), 'open'
    elif ok:
        mark, verdict, state = '&#10003;', t('Richtig', 'Correct'), 'ok'
    else:
        mark, verdict, state = '&#10007;', t('Nicht ganz', 'Not quite'), 'no'
    mark_html = f'<span class="hpi-feedback-mark" aria-hidden="true">{mark}</span>' if mark else ''
    body = f'<div class="hpi-feedback-body">{explanation_html}</div>' if explanation_html else ''
    return (f'<div class="hpi-feedback hpi-feedback-{state}" role="status">'
            f'<div class="hpi-feedback-head">{mark_html}<span class="hpi-feedback-verdict">{verdict}</span></div>'
            f'{body}</div>')


def cta_label(text: str) -> str:
    """Label for a call-to-action run button. The class lets theme.css find the button inside marimo's
    shadow root (button:has(.hpi-cta-label)) and colour it HPI Dark Green. See .hpi-cta in theme.css."""
    from html import escape

    return f'<span class="hpi-cta-label">{escape(text)}</span>'


def box_legend(t) -> str:
    """HTML of a small legend of the three box types, in their real style.

    `t` picks the language like ragkit.i18n.T: t(german, english). Show it once above the first
    section (outside any H2) with mo.Html(theme.box_legend(t)).
    """
    from html import escape

    items = [('info', t('Information', 'Information'), t('zum Lesen', 'to read')),
             ('task', t('Ihre Aufgabe', 'Your task'), t('hier sind Sie dran', 'your turn')),
             ('extra', t('Zusatzaufgabe, für alle, die schon fertig sind', 'Extra task, for everyone who is already done'),
              t('freiwillig, wenn Sie fertig sind', 'optional, once you are done'))]
    boxes = ''.join(f'<div class="hpi-box hpi-box-{k}"><span class="hpi-box-label">{escape(label)}</span>{escape(note)}</div>'
                    for k, label, note in items)
    return f'<div class="hpi-legend" role="note">{boxes}</div>'


# --- Slide building blocks for marimo ------------------------------------------
# Shared by the w3 notebooks: key sentence on the note ground, hairline cards, grey hint lines,
# information boxes and numbered task boxes. One red mark per view (the title rule), everything
# else ink, grey, rule, note. marimo is imported inside each function, so this module stays
# importable without it.

def key_sentence(text: str):
    """The one key sentence of a slide, large, on the note ground."""
    import marimo as mo

    return mo.md(text).style({'background': NOTE, 'padding': '.75rem 1rem', 'border-radius': '2px',
                              'border-left': f'7px solid {PRIMARY["yellow"]}', 'font-size': '1.15rem', 'line-height': '1.35'})


def card(title: str, body: str, foot: str | None = None, corpus: bool = False):
    """A small card: a keyword, one sentence, an optional grey foot line; corpus=True marks RAGAS or corpus text."""
    import marimo as mo

    if corpus:
        body = f'<span class="hpi-corpus">{body}</span>'
    md = f'**{title}**\n\n{body}' + (f'\n\n<span style="color:{GREY};font-size:.85rem">{foot}</span>' if foot else '')
    return mo.md(md).style({'border': f'1px solid {RULE}', 'border-radius': '2px', 'padding': '.75rem 1rem',
                            'height': '100%', 'background': '#ffffff'})


def cards(items: list, widths='equal'):
    """A row of cards of equal height."""
    import marimo as mo

    return mo.hstack(items, widths=widths, gap=1, align='stretch', wrap=False)


def hint(text: str):
    """A grey hint line; `text` is Markdown."""
    import marimo as mo

    return mo.Html(f'<div class="hpi-hint">{mo.md(text).text}</div>')


def frame(title: str, text: str):
    """An information box (HPI Yellow) with its label word, for a statement that must stand out; `text` is Markdown."""
    import marimo as mo

    return mo.Html(info_box(title, mo.md(text).text))


def info(label: str, content):
    """An information box around marimo content (source lines, gold passage, notes, solutions)."""
    import marimo as mo

    return mo.Html(info_box(label, content.text))


def left(table_md: str) -> str:
    """Left-align every column of a Markdown table (marimo right-aligns by default)."""
    return table_md.replace('---|', ':---|')


def numbered_task(t, order: list[str], name: str, title: str, steps: list[str], time: str):
    """The task box of a numbered task ("Ihre Aufgabe 2 von 5"), or of the extra task (name='extra').

    `t` picks the language like ragkit.i18n.T, `order` lists the task names of this notebook and
    version. A name missing from `order` (a task the short version hides) gives empty output.
    """
    import marimo as mo

    if name == 'extra':
        label = t('Zusatzaufgabe, für alle, die schon fertig sind', 'Extra task, for everyone who is already done')
    elif name not in order:  # not in this version; callers hide it with t.full, but arguments run first
        return mo.md('')
    else:
        label = t(f'Ihre Aufgabe {order.index(name) + 1} von {len(order)}',
                  f'Your task {order.index(name) + 1} of {len(order)}')
    return mo.Html(task_box(label, steps, time, title, kind='extra' if name == 'extra' else 'task'))


# Opt-in for collapsible H2 sections (script and style in head.html). Render it once, e.g.
# mo.Html(theme.header_html(...) + theme.SECTIONS_MARKER). Needs html_head_file="ragkit/head.html".
SECTIONS_MARKER = '<span data-hpi-sections hidden></span>'


def header_html(title: str, subtitle: str = '', logos: bool = True) -> str:
    """Notebook header: title, the single red rule, optional subtitle and logos.

    The logos are embedded as data URIs. marimo serves `/public/` only through a service worker
    that is not active on the very first page load, so linked logos were missing until a reload.
    """
    logo_html = (
        '<div class="hpi-logos">'
        f'<img src="{_logo_uri("logo_aisc_150dpi.png")}" alt="HPI AI Service Centre">'
        f'<img src="{_logo_uri("logo_bmftr_de.png")}" alt="BMFTR">'
        '</div>'
    ) if logos else ''
    sub = f'<p class="hpi-subtitle">{subtitle}</p>' if subtitle else ''
    return f'<div class="hpi-header">{logo_html}<h1>{title}</h1><div class="hpi-title-rule"></div>{sub}</div>'
