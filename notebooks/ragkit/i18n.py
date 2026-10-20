"""Language and scope switches shared by the w3 notebooks.

Every w3 notebook opens with two radios: language (Deutsch / English) and scope
(normal / kurz). Their start values come from the command line or the URL, so
whoever runs the workshop can fix a version:

    uv run marimo run w3_01_intro_end_to_end.py -- --lang en --short
    http://localhost:2718/?lang=en&short=1

Each cell then holds both languages side by side and asks `T` for the active
one; cells that exist only in the normal version wrap their output in `t.full`.

    ui_lang, ui_scope = i18n.switches(mo)          # cell 1, displays the radios
    t = i18n.T(ui_lang.value, ui_scope.value)       # cell 2
    mo.md(t('Text auf Deutsch.', 'Text in English.'))
    t.full(mo.md(t('Nur in der normalen Fassung.', 'Normal version only.')))
"""

LANGS = {'Deutsch': 'de', 'English': 'en'}
SCOPES = {'normal': 'normal', 'kurz (short)': 'kurz'}


def _truthy(value) -> bool:
    # `--short` without a value arrives as '' from marimo's argument parser.
    return value is True or str(value).strip().lower() in ('', '1', 'true', 'yes', 'ja')


def defaults(mo) -> tuple[str, str]:
    """('de' | 'en', 'normal' | 'kurz') from `--lang en`, `--short`, `?lang=en`, `?short=1`."""
    args, query = mo.cli_args(), mo.query_params()
    lang = str(args.get('lang') or query.get('lang') or 'de').lower()
    short = (('short' in args and _truthy(args.get('short')))
             or ('short' in query and _truthy(query.get('short')))
             or str(query.get('scope') or '').lower() in ('kurz', 'short'))
    return ('en' if lang.startswith('en') else 'de'), ('kurz' if short else 'normal')


def switches(mo):
    """The two radios, preset from `defaults`; display them with `mo.hstack`."""
    lang, scope = defaults(mo)
    ui_lang = mo.ui.radio(LANGS, value=next(k for k, v in LANGS.items() if v == lang),
                          inline=True, label='Sprache / Language')
    ui_scope = mo.ui.radio(SCOPES, value=next(k for k, v in SCOPES.items() if v == scope),
                           inline=True, label='Umfang / Scope')
    return ui_lang, ui_scope


class T:
    """Picks the active language and hides normal-only output in the short version."""

    def __init__(self, lang: str, scope: str):
        self.lang = lang
        self.short = scope == 'kurz'

    def __call__(self, de, en):
        """The German or the English variant; works for strings and any other object."""
        return en if self.lang == 'en' else de

    def num(self, x: float, digits: int = 2) -> str:
        """A score with a decimal comma in German and a decimal point in English."""
        s = f'{x:.{digits}f}'
        return s.replace('.', ',') if self.lang == 'de' else s

    def full(self, view, empty=None):
        """`view` in the normal version, `empty` (default: nothing) in the short one."""
        return empty if self.short else view
