"""Corpus text in the language of the notebook (w3_01 to w3_04).

The participant reads the notebook's language only. In the English version a cached
English translation is the text and the German original sits in an accordion below;
in the German version the original is the text. Texts without a translation are shown
as they are. The shown text carries the class hpi-corpus, so the language check of
the tools can tell corpus text from our own prose.

    ct = CorpusText(mo, t, translations)
    ct.show(chunk)            # a block, with the original in an accordion (English version)
    ct.cell(reason, n=22)     # one Markdown table cell, shortened to n words
"""

import re

ORIGINAL = 'Original (Deutsch)'


def flat(s) -> str:
    """One line for a Markdown table cell; a pipe would start a new column."""
    return ' '.join(str(s).split()).replace('|', '/')


def short(text, n: int = 22) -> str:
    w = flat(text).split()
    return ' '.join(w[:n]) + (' ...' if len(w) > n else '')


class CorpusText:
    def __init__(self, mo, t, translations: dict[str, str]):
        self.mo, self.t, self.tr = mo, t, translations

    def en_of(self, text):
        """The English translation of a cached German text, only in the English version, only when it differs."""
        if self.t.lang != 'en':
            return None
        en = self.tr.get(text)
        return en if en and en.strip() != text.strip() else None

    def primary(self, text) -> str:
        """The text the participant reads: the translation in the English version, else the original."""
        return self.en_of(text) or text

    def show(self, text, literal: bool = False, original: bool = True):
        """A block of corpus text. literal=True shows a leading '#' as a character, not as a heading.
        original=False leaves the accordion with the German original out (tables of many short lines)."""
        en = self.en_of(text)
        shown = en or text
        if literal:
            shown = re.sub(r'^#+', lambda m: '\\#' * len(m.group(0)), shown)
        block = self._block(shown)
        if en is None or not original:
            return block
        return self.mo.vstack([block, self.mo.accordion({ORIGINAL: self._block(text)})], gap=0.25)

    def cell(self, text, n: int | None = None) -> str:
        """One Markdown table cell, in the notebook's language, optionally shortened to n words."""
        cut = (lambda s: short(s, n)) if n else flat
        return f'<span class="hpi-corpus">{cut(self.primary(text))}</span>'

    def head_cell(self, text) -> str:
        """The first line of a chunk (its heading) as a table cell, in the notebook's language."""
        return f'<span class="hpi-corpus">{flat(self.primary(text).split(chr(10))[0])}</span>'

    def _block(self, text):
        return self.mo.Html(f'<div class="hpi-corpus">{self.mo.md(text).text}</div>')
