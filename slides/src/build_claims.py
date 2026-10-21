"""Build the claim slides of RAG III (Mentimeter rounds, definitions, breakdowns) from the AISC Light template.

    uv run --with python-pptx python slides/src/build_claims.py de slides/src/rag_iii_claims_german.pptx
    uv run --with python-pptx python slides/src/build_claims.py en slides/src/rag_iii_claims_english.pptx

HPI corporate design: white ground, Verdana, ink text, grey secondary text, hairlines, one red rule under the title,
no shadows, no cards, titles in sentence case.
"""

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

TEMPLATE = Path('/home/hanmul/.claude/plugins/synced/19464ea5-effb-41ee-92e0-029ec34f55ca_126a6649-ed73-4cb3-a3d3-fbe1ee1e4ac5/'
                'kisz-house-style/skills/hpi-corporate-design/assets/templates/202608_AISC-Template_Light.pptx')
INK, GREY, RULE, ACCENT = RGBColor(0x21, 0x24, 0x27), RGBColor(0x5a, 0x60, 0x65), RGBColor(0xd6, 0xd7, 0xd8), RGBColor(0xb1, 0x06, 0x3a)
FONT = 'Verdana'
LEFT, WIDTH = 0.77, 11.8  # content column in inches

T = {
    'de': {
        'menti1': 'Errate die Zahl',
        'menti2': 'Errate die Zahl, Runde 2',
        'menti_lead': 'Wie viele Aussagen stecken in jedem Text?',
        'code': 'menti.com, Code 7946 2006',
        'defs_title': 'Was ist eine Aussage?',
        'defs': [
            ('Alltag, ganz', 'Eine Aussage ist ein Satz, der etwas feststellt.'),
            ('Alltag, feiner', 'Jeder Teilsatz, der etwas feststellt, ist eine Aussage: Hauptsatz und Nebensatz zählen getrennt.'),
            ('Logik', 'Eine Aussage ist etwas, das für sich allein wahr oder falsch sein kann. Jedes Element einer Aufzählung zählt für jede Handlung, zu der es gehört.'),
            ('Arbeitsdefinition', 'Eine Aussage ist der kleinste Teil der Referenzantwort, der sich an einer Stelle im Kompendium prüfen lässt.'),
        ],
        'work_title': 'Arbeitsdefinition',
        'work': 'Eine Aussage ist der kleinste Teil der Referenzantwort, der sich an einer Stelle im Kompendium prüfen lässt.',
        'work_rules': ['Jede Handlung (dokumentieren, ersetzen) ist eine eigene Aussage.',
                       'Jede Begründung (weil, da) und jeder Zweck (damit, um zu) ist eine eigene Aussage.',
                       'Eine Aufzählung, die das Kompendium als eine Liste nennt, bleibt eine Aussage.',
                       'Wiederholungen und Füllwörter zählen nicht.'],
        'break_title': 'Beispiel {n}: Wie viele Aussagen?',
        'rows': ['Alltag, ganz', 'Alltag, feiner', 'Logik', 'Arbeitsdefinition'],
        'whole': 'der ganze Satz',
        'ragas_title': 'Was RAGAS dem Richter sagt',
        'ragas_lead': 'Die Anweisung an den Richter, wörtlich:',
        'ragas_none': 'Keine Definition. Der Richter entscheidet selbst, was eine Aussage ist.',
        'ragas_cols': [('gpt-oss-120b', '2 Aussagen', 'zerlegt meist je Satz'),
                       ('llama-3-3-70b', '4 Aussagen', 'zerlegt feiner')],
        'ragas_foot': 'Dieselbe Referenzantwort aus Notebook 2, Abschnitt E b. Der Nenner von Context Recall hängt davon ab, wer zerlegt.',
        'ref': 'Unternehmen können sich für Pilotprojekte bewerben und erhalten 2-3 Tage pro Woche aktive Projektbegleitung, um gemeinsam einen Prototypen zu entwickeln. Dabei können sie auch die Recheninfrastruktur nutzen.',
    },
    'en': {
        'menti1': 'Guess the number',
        'menti2': 'Guess the number, round 2',
        'menti_lead': 'How many claims are in each text?',
        'code': 'menti.com, code 7946 2006',
        'defs_title': 'What is a claim?',
        'defs': [
            ('Everyday, whole', 'A claim is a sentence that states something.'),
            ('Everyday, granular', 'Every clause that states something is a claim: main clause and subordinate clause count separately.'),
            ('Logic', 'A claim is something that can be true or false on its own. Every element of a list counts for every action it belongs to.'),
            ('Working definition', 'A claim is the smallest part of the reference answer that can be checked at one place in the Kompendium.'),
        ],
        'work_title': 'Working definition',
        'work': 'A claim is the smallest part of the reference answer that can be checked at one place in the Kompendium.',
        'work_rules': ['Every action (document, replace) is its own claim.',
                       'Every reason (because) and every purpose (so that) is its own claim.',
                       'A list that the Kompendium states as one list stays one claim.',
                       'Repetitions and filler do not count.'],
        'break_title': 'Example {n}: how many claims?',
        'rows': ['Everyday, whole', 'Everyday, granular', 'Logic', 'Working definition'],
        'whole': 'the whole sentence',
        'ragas_title': 'What RAGAS tells the judge',
        'ragas_lead': 'The instruction to the judge, verbatim:',
        'ragas_none': 'No definition. The judge decides for itself what a claim is.',
        'ragas_cols': [('gpt-oss-120b', '2 claims', 'mostly splits per sentence'),
                       ('llama-3-3-70b', '4 claims', 'splits finer')],
        'ragas_foot': 'The same reference answer from notebook 2, section E b. The denominator of Context Recall depends on who splits.',
        'ref': 'Companies can apply for pilot projects and receive 2 to 3 days per week of active project support to develop a prototype together. They can also use the computing infrastructure.',
    },
}
RAGAS_INSTRUCTION = ('Given a context, and an answer, analyze each statement in the answer and classify if the statement '
                     'can be attributed to the given context or not.')

# The three texts (gold 34, 28, 30) and their breakdowns: per row the claims and the count shown at the right.
# The first row (everyday, whole) is always the whole sentence and is filled in by the build.
EXAMPLES = {
    'de': [
        ('Die Anforderungen müssen dokumentiert werden, damit sie identifiziert und bei Planung, Betrieb und Beschaffung berücksichtigt werden können.',
         [(['1) Die Anforderungen müssen dokumentiert werden', '2) damit sie identifiziert und bei Planung, Betrieb und Beschaffung berücksichtigt werden können'], 2),
          (['1) Die Anforderungen müssen dokumentiert werden.', '2) Dokumentierte Anforderungen können identifiziert werden.',
            '3) Dokumentierte Anforderungen können bei der Planung berücksichtigt werden.', '4) ... beim Betrieb ...', '5) ... bei der Beschaffung ...'], 5),
          (['1) Anforderungen müssen dokumentiert werden.', '2) Zweck: Anforderungen identifizieren.', '3) Zweck: bei der Planung berücksichtigen.',
            '4) Zweck: beim Betrieb berücksichtigen.', '5) Zweck: bei der Beschaffung berücksichtigen.'], 5)]),
        ('Ohne verbindliche Passwortregeln besteht die Gefahr schwacher oder mehrfach genutzter Passwörter, die leicht kompromittiert werden können.',
         [(['1) Ohne verbindliche Passwortregeln besteht die Gefahr schwacher oder mehrfach genutzter Passwörter', '2) die leicht kompromittiert werden können'], 2),
          (['1) Ohne verbindliche Passwortregeln besteht die Gefahr schwacher Passwörter.', '2) Ohne verbindliche Passwortregeln besteht die Gefahr mehrfach genutzter Passwörter.',
            '3) Schwache Passwörter können leicht kompromittiert werden.', '4) Mehrfach genutzte Passwörter können leicht kompromittiert werden.'], 4),
          (['1) Ohne Passwortregeln: Gefahr schwacher Passwörter.', '2) Ohne Passwortregeln: Gefahr mehrfach genutzter Passwörter.',
            '3) Schwache oder mehrfach genutzte Passwörter sind leicht zu kompromittieren.'], 3)]),
        ('Ausgabe und Entzug von Zutritts-, Zugangs- und Zugriffsmitteln sind zu dokumentieren und bei Kompromittierung sofort zu ersetzen.',
         [(['1) Ausgabe und Entzug von Zutritts-, Zugangs- und Zugriffsmitteln sind zu dokumentieren', '2) und bei Kompromittierung sofort zu ersetzen'], 2),
          (['1) Die Ausgabe von Zutrittsmitteln ist zu dokumentieren.', '2) ... Zugangsmitteln ...', '3) ... Zugriffsmitteln ...',
            '4) Der Entzug von Zutrittsmitteln ist zu dokumentieren.', '5) ... Zugangsmitteln ...', '6) ... Zugriffsmitteln ...',
            '7) Zutrittsmittel sind bei Kompromittierung sofort zu ersetzen.', '8) Zugangsmittel ...', '9) Zugriffsmittel ...'], 9),
          (['1) Die Ausgabe von Zutritts-, Zugangs- und Zugriffsmitteln ist zu dokumentieren.', '2) Der Entzug von Zutritts-, Zugangs- und Zugriffsmitteln ist zu dokumentieren.',
            '3) Zutritts-, Zugangs- und Zugriffsmittel sind bei Kompromittierung sofort zu ersetzen.'], 3)]),
    ],
    'en': [
        ('The requirements must be documented so that they can be identified and taken into account during planning, operation, and procurement.',
         [(['1) The requirements must be documented', '2) so that they can be identified and taken into account during planning, operation, and procurement'], 2),
          (['1) The requirements must be documented.', '2) Documented requirements can be identified.',
            '3) Documented requirements can be taken into account during planning.', '4) ... during operation.', '5) ... during procurement.'], 5),
          (['1) Requirements must be documented.', '2) Purpose: identify requirements.', '3) Purpose: take into account during planning.',
            '4) Purpose: during operation.', '5) Purpose: during procurement.'], 5)]),
        ('Without binding password rules, there is a risk of weak or reused passwords that can be easily compromised.',
         [(['1) Without binding password rules, there is a risk of weak or reused passwords', '2) that can be easily compromised'], 2),
          (['1) Without binding password rules, there is a risk of weak passwords.', '2) Without binding password rules, there is a risk of reused passwords.',
            '3) Weak passwords can be easily compromised.', '4) Reused passwords can be easily compromised.'], 4),
          (['1) Without password rules: risk of weak passwords.', '2) Without password rules: risk of reused passwords.',
            '3) Weak or reused passwords are easy to compromise.'], 3)]),
        ('Issuance and revocation of entry, access, and usage media must be documented and replaced immediately if compromised.',
         [(['1) Issuance and revocation of entry, access, and usage media must be documented', '2) and replaced immediately if compromised'], 2),
          (['1) Issuance of entry media must be documented.', '2) ... access media ...', '3) ... usage media ...',
            '4) Revocation of entry media must be documented.', '5) ... access media ...', '6) ... usage media ...',
            '7) Entry media must be replaced immediately if compromised.', '8) Access media ...', '9) Usage media ...'], 9),
          (['1) Issuance of entry, access and usage media must be documented.', '2) Revocation of entry, access and usage media must be documented.',
            '3) Entry, access and usage media must be replaced immediately if compromised.'], 3)]),
    ],
}


def clear_slides(prs):
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        prs.part.drop_rel(sldId.rId)
        sldIdLst.remove(sldId)


def no_shadow(shape):
    shape.shadow.inherit = False


def new_slide(prs, title):
    layout = next(lay for lay in prs.slide_layouts if lay.name == 'Inhalt')
    s = prs.slides.add_slide(layout)
    for ph in list(s.placeholders):
        if ph.placeholder_format.type == 1:  # TITLE
            ph.text = title
            no_shadow(ph)
            for p in ph.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name = FONT
                    r.font.bold = False
                    r.font.size = Pt(28)
                    r.font.color.rgb = INK
                    r.font._element.set('cap', 'none')  # the master sets cap="all"; titles are sentence case
        else:
            ph._element.getparent().remove(ph._element)
    rule(s, LEFT, 1.22, 48 / 72, ACCENT, height=3 / 72)  # the one red element: 48 x 3 pt under the title
    return s


def rule(slide, x, y, w, color=RULE, height=0.75 / 72):
    """A hairline (or the red title rule): a filled rectangle without outline or shadow."""
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(height))
    no_shadow(shp)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp


def text(slide, x, y, w, h, s, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=0):
    """A plain text box: no fill, no outline, no shadow. `s` is one string or a list of paragraphs."""
    shp = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    no_shadow(shp)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    lines = s if isinstance(s, list) else [s]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        r = p.add_run()
        r.text = ln
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return shp


def menti_slide(prs, L, title, examples, with_definition):
    s = new_slide(prs, title)
    y = 1.35
    if with_definition:
        text(s, LEFT, y, WIDTH, 0.5, L['work'], size=14)
        text(s, LEFT, y + 0.5, WIDTH, 1.25, [f'{i + 1}. {r}' for i, r in enumerate(L['work_rules'])], size=11, color=GREY, space=2)
        y += 1.85
        rule(s, LEFT, y, WIDTH)
        y += 0.15
    else:
        text(s, LEFT, y, WIDTH, 0.4, L['menti_lead'], size=14, color=GREY)
        y += 0.55
    h = (6.3 - y) / 3
    for i, (txt, _) in enumerate(examples):
        text(s, LEFT, y, 0.6, h, 'ABC'[i], size=20, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
        text(s, LEFT + 0.7, y, WIDTH - 0.7, h, txt, size=14 if with_definition else 16, anchor=MSO_ANCHOR.MIDDLE)
        if i < 2:
            rule(s, LEFT, y + h, WIDTH)
        y += h
    text(s, LEFT + 7.8, 6.45, 4.0, 0.4, L['code'], size=14, color=GREY, align=PP_ALIGN.RIGHT)
    return s


def defs_slide(prs, L):
    s = new_slide(prs, L['defs_title'])
    y, row_h = 1.45, 1.15
    for i, (head, defn) in enumerate(L['defs']):
        text(s, LEFT, y, 2.9, row_h, head, size=14, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
        text(s, LEFT + 3.0, y, WIDTH - 3.0, row_h, defn, size=16, anchor=MSO_ANCHOR.MIDDLE)
        if i < len(L['defs']) - 1:
            rule(s, LEFT, y + row_h, WIDTH)
        y += row_h
    return s


def work_slide(prs, L):
    s = new_slide(prs, L['work_title'])
    text(s, LEFT, 1.5, WIDTH, 1.2, L['work'], size=22, anchor=MSO_ANCHOR.MIDDLE)
    rule(s, LEFT, 2.85, WIDTH)
    for i, r in enumerate(L['work_rules']):
        y = 3.05 + i * 0.8
        text(s, LEFT, y, 0.5, 0.7, str(i + 1), size=18, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
        text(s, LEFT + 0.6, y, WIDTH - 0.6, 0.7, r, size=16, anchor=MSO_ANCHOR.MIDDLE)
    return s


def breakdown_slide(prs, L, n, sentence, rows):
    s = new_slide(prs, L['break_title'].format(n='ABC'[n]))
    text(s, LEFT, 1.3, WIDTH, 0.75, sentence, size=15, anchor=MSO_ANCHOR.MIDDLE)
    rule(s, LEFT, 2.1, WIDTH)
    all_rows = [([L['whole']], 1), *rows]
    y = 2.15
    heights = [0.5, 0.8, 1.7, 1.05]  # whole, granular, logic, working definition
    for label, (claims, count), row_h in zip(L['rows'], all_rows, heights):
        size = 9 if len(claims) > 5 else 10 if len(claims) > 3 else 11
        text(s, LEFT, y, 2.3, row_h, label, size=12, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
        text(s, LEFT + 2.4, y, WIDTH - 3.5, row_h, claims, size=size, anchor=MSO_ANCHOR.MIDDLE, space=1)
        text(s, LEFT + WIDTH - 1.0, y, 1.0, row_h, str(count), size=26, color=GREY, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
        y += row_h
        rule(s, LEFT, y, WIDTH)
    return s


def ragas_slide(prs, L):
    s = new_slide(prs, L['ragas_title'])
    text(s, LEFT, 1.4, WIDTH, 0.4, L['ragas_lead'], size=14, color=GREY)
    text(s, LEFT, 1.8, WIDTH, 0.8, RAGAS_INSTRUCTION, size=16, anchor=MSO_ANCHOR.MIDDLE)
    rule(s, LEFT, 2.7, WIDTH)
    text(s, LEFT, 2.8, WIDTH, 0.45, L['ragas_none'], size=16)
    text(s, LEFT, 3.3, WIDTH, 0.7, L['ref'], size=13, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    rule(s, LEFT, 4.1, WIDTH)
    for i, (model, count, how) in enumerate(L['ragas_cols']):
        x = LEFT + i * 6.0
        text(s, x, 4.2, 5.8, 0.45, model, size=16)
        text(s, x, 4.65, 5.8, 0.6, count, size=28, color=GREY)
        text(s, x, 5.25, 5.8, 0.4, how, size=13, color=GREY)
    text(s, LEFT, 5.75, WIDTH, 0.4, L['ragas_foot'], size=12, color=GREY)
    return s


def build(lang, out):
    L, ex = T[lang], EXAMPLES[lang]
    prs = Presentation(TEMPLATE)
    clear_slides(prs)
    menti_slide(prs, L, L['menti1'], ex, with_definition=False)
    defs_slide(prs, L)
    work_slide(prs, L)
    for n, (sentence, rows) in enumerate(ex):
        breakdown_slide(prs, L, n, sentence, rows)
    menti_slide(prs, L, L['menti2'], ex, with_definition=True)
    ragas_slide(prs, L)
    prs.save(out)
    print(out, len(prs.slides), 'slides')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
