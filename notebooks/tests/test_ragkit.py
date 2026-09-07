"""Asserts over the pure parts of ragkit. No network, no Qdrant, no fixtures.

Runs on the standard library plus numpy, so it needs nothing installed beyond
what the notebooks already use. From the notebooks/ directory:

    python tests/test_ragkit.py

pytest picks it up as-is too, if you have it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.chunk import (
    _fix_german_umlauts,
    chunk_by_chars,
    chunk_by_paragraph,
    chunk_by_words,
    chunk_markdown_by_headers,
    count_umlaut_placeholders,
    format_citation,
    normalize_text,
    parser_aware_split,
    records_from_docling_json_structured_sections,
    records_from_markdown_header_chunks,
)
from ragkit.search import (
    cosine_sim,
    entropy,
    l2_normalise,
    ndcg_at_k,
    reciprocal_rank,
)

# --- normalisation -------------------------------------------------------

def test_fix_german_umlauts_handles_both_placeholder_forms():
    assert _fix_german_umlauts('B/C252rgschaft') == 'Bürgschaft'
    assert _fix_german_umlauts('BC252rgschaft') == 'Bürgschaft'


def test_normalize_text_collapses_blank_line_runs():
    assert normalize_text('a\n\n\n\n\nb') == 'a\n\nb'
    assert normalize_text('a\r\nb') == 'a\nb'


def test_count_umlaut_placeholders_walks_nested_structures():
    assert count_umlaut_placeholders({'a': ['x/C252y', 'clean']}) == 1


def test_normalize_text_can_leave_umlaut_placeholders_alone():
    assert normalize_text('a  b /C231') == 'a b ü'
    assert normalize_text('a  b /C231', fix_umlauts=False) == 'a b /C231'


def test_format_citation():
    assert format_citation([]) is None
    assert format_citation([7]) == 'p. 7'
    assert format_citation([7, 8, 9]) == 'p. 7-9'
    assert format_citation([7], lang='de') == 'S. 7'


def test_records_stamp_the_callers_pdf_and_chunk_sizes():
    md = '# One\n\nbody one\n\n## Two\n\nbody two'
    records = records_from_markdown_header_chunks(md, 'x/standard.pdf', max_chunk=50, overlap=10)
    assert records[0]['metadata']['source_file'] == 'standard.pdf'
    assert records[0]['metadata']['max_chunk'] == 50
    doc = {'texts': [{'label': 'section_header', 'text': 'H', 'prov': [{'page_no': 3}]},
                     {'label': 'text', 'text': 'body', 'prov': [{'page_no': 3}]}]}
    records = records_from_docling_json_structured_sections(doc, 'x/standard.pdf', lang='de')
    assert records[0]['metadata']['source_file'] == 'standard.pdf'
    assert records[0]['metadata']['citation_hint'] == 'S. 3'


# --- chunking ------------------------------------------------------------

def test_chunk_by_paragraph_drops_empties():
    assert chunk_by_paragraph('one\n\n\n\ntwo\n\n  \n\nthree') == ['one', 'two', 'three']


def test_chunk_by_words_respects_limit():
    chunks = chunk_by_words(' '.join(str(i) for i in range(250)), max_words=100)
    assert [len(c.split()) for c in chunks] == [100, 100, 50]


def test_chunk_by_chars_never_exceeds_limit_and_covers_input():
    text = ' '.join(f'word{i}' for i in range(400))
    chunks = chunk_by_chars(text, max_chars=200)
    assert chunks
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[0] == 'word0'
    assert chunks[-1].split()[-1] == 'word399'


def test_chunk_by_chars_overlap_repeats_content():
    text = ' '.join(f'word{i}' for i in range(200))
    assert len(chunk_by_chars(text, max_chars=200, overlap=50)) > \
           len(chunk_by_chars(text, max_chars=200, overlap=0))


def test_parser_aware_split_keeps_short_text_whole():
    assert parser_aware_split('short paragraph', max_chunk=1200) == ['short paragraph']


def test_parser_aware_split_splits_oversized_block():
    chunks = parser_aware_split('x' * 5000, max_chunk=1000, overlap=100)
    assert len(chunks) > 1


def test_parser_aware_split_rejects_overlap_at_or_above_max_chunk():
    # Without this guard the oversized-block loop never advances and hangs.
    try:
        parser_aware_split('x' * 5000, max_chunk=100, overlap=100)
    except ValueError:
        return
    raise AssertionError('expected ValueError for overlap >= max_chunk')


def test_chunk_markdown_by_headers_splits_at_headings():
    md = '# One\n\nbody one\n\n## Two\n\nbody two'
    chunks = chunk_markdown_by_headers(md)
    assert len(chunks) == 2
    assert chunks[0].startswith('# One')
    assert chunks[1].startswith('## Two')


def test_chunk_markdown_by_headers_on_empty_input():
    assert chunk_markdown_by_headers('   ') == []


# --- similarity and metrics ---------------------------------------------

def test_cosine_sim_extremes():
    assert abs(cosine_sim([1, 0], [1, 0]) - 1.0) < 1e-6
    assert abs(cosine_sim([1, 0], [0, 1]) - 0.0) < 1e-6
    assert abs(cosine_sim([1, 0], [-1, 0]) + 1.0) < 1e-6


def test_l2_normalise_gives_unit_rows():
    import numpy as np
    rows = l2_normalise(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_l2_normalise_survives_a_zero_row():
    import numpy as np
    assert np.all(np.isfinite(l2_normalise(np.zeros((1, 3)))))


def test_entropy_is_lower_when_one_score_dominates():
    peaked = [1.0] + [0.01] * 49
    uniform = [0.5] * 50
    assert entropy(peaked) < entropy(uniform)


def test_entropy_handles_negative_scores():
    # Cosine similarity can be negative; this must not produce nan.
    import numpy as np
    assert np.isfinite(entropy([-0.4, 0.1, 0.9]))
    assert np.isfinite(entropy([-0.4, 0.1, 0.9], shift_min=True))


def test_entropy_default_is_the_taught_formula_and_shift_min_differs():
    # w2_01 cell 30: p_i = s_i / sum(s). The peaked demo vector gives 2.76 bits.
    peaked = [1.0] + [0.01] * 49
    assert abs(entropy(peaked) - 2.76) < 0.01
    assert entropy(peaked, shift_min=True) < 0.01  # min-shift sends the 49 tail scores to p=0


def test_embed_rejects_the_placeholder_key():
    import os

    from ragkit.embed import embed
    saved = os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'
    try:
        embed('x')
    except RuntimeError as exc:
        assert 'placeholder' in str(exc)
    else:
        raise AssertionError('expected RuntimeError for placeholder key')
    finally:
        if saved is None:
            del os.environ['OPENAI_API_KEY']
        else:
            os.environ['OPENAI_API_KEY'] = saved


def test_entropy_top_n_sorts_first():
    scores = [0.1, 0.9, 0.5, 0.2]
    assert entropy(scores, top_n=2) == entropy([0.9, 0.5])


def test_reciprocal_rank():
    assert reciprocal_rank(['a', 'b', 'c'], {'a'}) == 1.0
    assert reciprocal_rank(['a', 'b', 'c'], {'b'}) == 0.5
    assert reciprocal_rank(['a', 'b', 'c'], {'z'}) == 0.0


def test_ndcg_at_k_is_one_for_a_perfect_ranking():
    assert abs(ndcg_at_k(['a', 'b', 'c'], {'a', 'b'}, k=3) - 1.0) < 1e-9


def test_ndcg_at_k_penalises_a_late_hit():
    assert ndcg_at_k(['x', 'y', 'a'], {'a'}, k=3) < ndcg_at_k(['a', 'y', 'x'], {'a'}, k=3)


def test_ndcg_at_k_with_no_relevant_docs():
    assert ndcg_at_k(['a', 'b'], set(), k=2) == 0.0


# ---------------------------------------------------------------------------
# config.setup, theme, crawl
# ---------------------------------------------------------------------------

import os

from ragkit import config, theme
from ragkit.crawl import _NOT_A_PHOTO, licence_ok


def test_setup_non_strict_reports_missing_key_and_file():
    saved = os.environ.pop('OPENAI_API_KEY', None)
    try:
        env = config.setup(require_qdrant=False, required_files=(Path('/nonexistent/file.md'),), strict=False)
        assert not env.ok
        assert any('OPENAI_API_KEY' in p for p in env.problems)
        assert env.missing_files == [Path('/nonexistent/file.md')]
        assert '✗' in env.summary_md() and 'nonexistent' in env.summary_md()
    finally:
        if saved is not None:
            os.environ['OPENAI_API_KEY'] = saved


def test_setup_strict_raises_with_fix_text():
    saved = os.environ.pop('OPENAI_API_KEY', None)
    try:
        try:
            config.setup(require_qdrant=False, strict=True)
        except RuntimeError as e:
            assert '.env' in str(e)
        else:
            raise AssertionError('strict setup should raise without an API key')
    finally:
        if saved is not None:
            os.environ['OPENAI_API_KEY'] = saved


def test_setup_ok_without_qdrant_when_key_present():
    saved = os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = 'test-key'
    try:
        env = config.setup(require_qdrant=False, strict=True)
        assert env.ok and env.api_key == 'test-key'
    finally:
        if saved is None:
            del os.environ['OPENAI_API_KEY']
        else:
            os.environ['OPENAI_API_KEY'] = saved


def test_tint_bounds_and_values():
    assert theme.tint('#5a6065', 100) == '#5a6065'
    assert theme.tint('#5a6065', 0) == '#ffffff'
    assert theme.tint('#5a6065', 25) == '#d6d7d8'
    try:
        theme.tint('#5a6065', 101)
    except ValueError:
        pass
    else:
        raise AssertionError('tint must reject percent > 100')


def test_mpl_rc_keys_are_valid():
    import matplotlib
    rc = theme.mpl_rc()
    for key, value in rc.items():
        matplotlib.rcParams[key] = value  # raises KeyError/ValueError on a bad key or value


def test_palette_greys_first_and_capped():
    assert theme.palette(2) == theme.GREYS[:2]
    assert theme.palette(3, accent='#b1063a')[-1] == '#b1063a'
    try:
        theme.palette(7)
    except ValueError:
        pass
    else:
        raise AssertionError('palette must refuse more than five grey classes')


def test_licence_filter():
    assert licence_ok('CC BY-SA 4.0') and licence_ok('CC0') and licence_ok('CC BY 2.0')
    assert licence_ok('Public domain')
    assert not licence_ok('CC BY-NC-SA 3.0') and not licence_ok('CC BY-ND 4.0')
    assert not licence_ok('Copyrighted free use') and not licence_ok('')


def test_photo_filename_filter():
    rejected = ['File:Juveniler Seeadler.jpg', 'File:Haliaeetus albicilla MHNT.ZOO.2010.11.96.1.jpg',
                'File:Erithacus rubecula distribution map.jpg', 'File:Book (1922) - strona 224.jpg']
    kept = ['File:Common Swift 2025 07 18 01.jpg', 'File:Rotkehlchen (April 2025) 1.jpg']
    assert all(_NOT_A_PHOTO.search(t) for t in rejected)
    assert not any(_NOT_A_PHOTO.search(t) for t in kept)


# ---------------------------------------------------------------------------
# IT-Grundschutz structure, gold labels, metrics and figures (w2_01)
# ---------------------------------------------------------------------------

import functools
from collections import Counter

import matplotlib

matplotlib.use('Agg')

from ragkit.chunk import (
    MODAL_RE,
    build_sections,
    chunk_stats,
    classify_header,
    dedupe,
    modality,
    parse_requirement_header,
    sections_to_text,
    strip_furniture,
    workshop_slice,
)
from ragkit.evaluate import (
    attach_section_keys,
    evaluate_ranking,
    gold_targets,
    load_gold,
    relevant_chunks,
    section_key,
    section_texts,
)
from ragkit.search import precision_at_k, recall_at_k
from ragkit.viz import head_tail, kde_plot, length_hist_panels, sorted_score_plot

SCHICHTEN = 'Hinweise zum Schichtenmodell und zur Modellierung'

SYNTHETIC_MD = f'''## Vorwort

Ein Vorwort.

<!-- image -->

## {SCHICHTEN}

Prozess-Bausteine gelten für alle.

## G 0.1 Feuer

Feuer brennt.

## APP.3.2 Webserver

## 1. Beschreibung

## 1.1. Einleitung

Ein Webserver liefert 'Seiten' aus.

## 3.1. Basis-Anforderungen

Die folgenden Anforderungen MÜSSEN vorrangig erfüllt werden.

## APP.3.2.A1 Sichere Konfiguration eines Webservers (B)

Der IT-Betrieb MUSS den Webserver sicher konfigurieren. Er DARF NUR nötige Dienste anbieten.
Er SOLLTE NICHT ohne Härtung laufen. Er SOLLTE protokollieren.

## APP.3.2.A3 ENTFALLEN (B)

Diese Anforderung ist entfallen.

## APP.3.2.A4 Protokollierung (B) [OT-Betrieb (Operational Technology, OT), Planende]

Der Webserver MUSS Ereignisse protokollieren.
'''


def test_classify_header_on_the_analysis_examples():
    cases = {
        '## ISMS.1 Sicherheitsmanagement': 'baustein',
        '## APP.3.2 Webserver': 'baustein',
        '## OPS.1.1.6 Software-Tests und -Freigaben': 'baustein',
        '## 1.1. Einleitung': 'template',
        '## 3.1. Basis-Anforderungen': 'template',
        '## ISMS.1.A1 Übernahme der Gesamtverantwortung für Informationssicherheit durch die '
        'Leitung (B) [Institutionsleitung]': 'requirement',
        '## APP.3.2.A1 Sichere Konfiguration eines Webservers (B)': 'requirement',
        '## ISMS.1.A14 ENTFALLEN (S)': 'requirement',
        '## G 0.1 Feuer': 'gefaehrdung',
        '## G 0.14 Ausspähen von Informationen (Spionage)': 'gefaehrdung',
        '## Vorwort': 'other',
        '## ORP: Organisation und Personal': 'other',
        '## Beispiele :': 'other',
        '## TGA-Anlage': 'other',
    }
    for line, kind in cases.items():
        assert classify_header(line) == kind, line


def test_parse_requirement_header_roles_and_entfallen():
    parsed = parse_requirement_header(
        '## IND.1.A18 Protokollierung (B) [Mitarbeitende, OT-Betrieb (Operational Technology, OT)]')
    assert parsed == {'req_id': 'IND.1.A18', 'title': 'Protokollierung', 'level': 'B',
                      'roles': ['Mitarbeitende', 'OT-Betrieb (Operational Technology, OT)']}
    parsed = parse_requirement_header('## ISMS.1.A14 ENTFALLEN (S)')
    assert parsed == {'req_id': 'ISMS.1.A14', 'title': 'ENTFALLEN', 'level': 'S', 'roles': []}
    assert parse_requirement_header('## 3.1. Basis-Anforderungen') is None


def test_modality_counts_the_five_classes():
    text = ('Es MUSS. Sie MÜSSEN. Er DARF NUR. Sie DÜRFEN NICHT. Er DARF KEINE. '
            'Er SOLLTE. Sie SOLLTEN. Er SOLLTE NICHT. Er SOLLTE KEINE. Er muss. Du MUSST.')
    assert modality(text) == {'MUSS': 2, 'DARF NUR': 1, 'DARF NICHT': 2,
                              'SOLLTE': 2, 'SOLLTE NICHT': 2}
    assert MODAL_RE.search('MUSST') is None


def test_strip_furniture_removes_image_markers_and_page_furniture():
    assert strip_furniture('a\n\n<!-- image -->\n\nb') == 'a\n\nb'
    assert strip_furniture('a\n42\nSeite 3\nIT-Grundschutz-Kompendium: Stand Februar 2023\nb') \
        == 'a\nb'


def test_build_sections_on_synthetic_markdown():
    sections = build_sections(SYNTHETIC_MD)
    assert [s['kind'] for s in sections] == ['other', 'other', 'gefaehrdung', 'baustein',
                                             'template', 'template', 'template',
                                             'requirement', 'requirement', 'requirement']
    assert [s['is_front_matter'] for s in sections] == [True] * 3 + [False] * 7
    assert all(s['baustein_id'] is None for s in sections[:3])
    assert '<!-- image -->' not in sections[0]['text']
    assert sections[2]['title'] == 'G 0.1 Feuer' and sections[2]['section_path'] == 'G 0.1 Feuer'
    baustein = sections[3]
    assert (baustein['baustein_id'], baustein['title'], baustein['section_path']) \
        == ('APP.3.2', 'Webserver', 'APP.3.2')
    assert baustein['text'] == '## APP.3.2 Webserver' and baustein['body'] == ''
    assert sections[4]['section_path'] == 'APP.3.2 > 1. Beschreibung'
    assert sections[5]['section_path'] == 'APP.3.2 > 1.1. Einleitung'
    a1 = sections[7]
    assert a1['req_id'] == 'APP.3.2.A1' and a1['level'] == 'B' and a1['roles'] == []
    assert a1['title'] == 'Sichere Konfiguration eines Webservers'
    assert a1['baustein_title'] == 'Webserver'
    assert a1['section_path'] == ('APP.3.2 > 3.1. Basis-Anforderungen > '
                                  'APP.3.2.A1 Sichere Konfiguration eines Webservers')
    assert a1['text'].startswith('## APP.3.2.A1 ') and a1['body'].startswith('Der IT-Betrieb')
    assert a1['modality'] == {'MUSS': 1, 'DARF NUR': 1, 'DARF NICHT': 0,
                              'SOLLTE': 1, 'SOLLTE NICHT': 1}
    assert a1['n_chars'] == len(a1['text']) and not a1['is_entfallen']
    assert sections[8]['is_entfallen']
    assert sections[9]['roles'] == ['OT-Betrieb (Operational Technology, OT)', 'Planende']


def test_sections_to_text_round_trips_the_cleaned_markdown():
    assert sections_to_text(build_sections(SYNTHETIC_MD)) == strip_furniture(SYNTHETIC_MD)


def test_workshop_slice_keeps_bausteine_and_named_front_matter():
    sections = build_sections(SYNTHETIC_MD)
    sliced = workshop_slice(sections, ['APP.3.2'], extra_titles=(SCHICHTEN,))
    assert [s['title'] for s in sliced][:2] == [SCHICHTEN, 'Webserver']
    assert len(sliced) == 8
    assert workshop_slice(sections, ['SYS.1.1']) == []


def test_section_key_prefers_the_requirement_id():
    sections = build_sections(SYNTHETIC_MD)
    assert section_key(sections[0]) == 'front::Vorwort'
    assert section_key(sections[6]) == 'APP.3.2::3.1. Basis-Anforderungen'
    assert section_key(sections[7]) == 'APP.3.2.A1'


def test_gold_targets_by_id_and_by_fuzzy_passage():
    sections = build_sections(SYNTHETIC_MD)
    texts = section_texts(sections)
    assert gold_targets('APP.3.2.A1 Sichere Konfiguration ... siehe auch APP.3.2.A4', sections) \
        == {'APP.3.2.A1', 'APP.3.2.A4'}
    assert gold_targets('Siehe SYS.1.1.A2 Rollentrennung', sections) == set()
    assert gold_targets('Ein Webserver liefert „Seiten“ aus.', sections, texts) \
        == {'APP.3.2::1.1. Einleitung'}
    assert gold_targets(f'{SCHICHTEN}\nProzess-Bausteine gelten für alle.', sections) \
        == {f'front::{SCHICHTEN}'}
    # a leading citation line that is not corpus text must not spoil the match
    assert gold_targets('APP.3.2, Kap. 1.1\n1.1. Einleitung\nEin Webserver liefert „Seiten“ aus.',
                        sections) == {'APP.3.2::1.1. Einleitung'}
    # an ID that does not exist falls back to the quoted text
    assert gold_targets('APP.3.2.A9 Einleitung\nEin Webserver liefert „Seiten“ aus.', sections) \
        == {'APP.3.2::1.1. Einleitung'}
    assert gold_targets('8.2.2 Vorgehen bei der Schutzbedarfsfeststellung (BSI-Standard 200-2)',
                        sections) == set()


TWO_SECTION_MD = '''## APP.1 Alpha

## 1.1. Einleitung

aaa bbb ccc ddd eee fff ggg hhh

## 1.2. Zielsetzung

iii jjj kkk lll mmm nnn ooo ppp
'''


def test_attach_section_keys_on_a_chunk_spanning_the_boundary():
    sections = build_sections(TWO_SECTION_MD)
    text = sections_to_text(sections)
    chunks = ['bbb ccc', 'fff ggg hhh\n\n## 1.2. Zielsetzung\n\niii jjj', 'ooo ppp']
    keys = [r['section_keys'] for r in attach_section_keys(chunks, sections)]
    assert keys == [['APP.1::1.1. Einleitung'],
                    ['APP.1::1.1. Einleitung', 'APP.1::1.2. Zielsetzung'],
                    ['APP.1::1.2. Zielsetzung']]
    # word chunks flatten newlines to spaces and must still be located
    from ragkit.chunk import chunk_by_words
    records = attach_section_keys(chunk_by_words(text, 5), sections)
    assert all(r['section_keys'] for r in records)
    assert records[0]['section_keys'][0] == 'APP.1::Alpha'
    # chunk 2 ('eee fff ggg hhh ##') already reaches into the 1.2 header
    assert relevant_chunks(records, {'APP.1::1.2. Zielsetzung'}) == {2, 3, 4}
    try:
        attach_section_keys(['not in the text at all'], sections)
    except ValueError:
        pass
    else:
        raise AssertionError('a chunk that is not a slice of the text must be rejected')


def test_relevant_chunks_intersects_section_keys():
    records = [{'section_keys': ['a']}, {'section_keys': ['b', 'c']}, {'section_keys': []}]
    assert relevant_chunks(records, {'c'}) == {1}
    assert relevant_chunks(records, {'a', 'b'}) == {0, 1}
    assert relevant_chunks(records, {'z'}) == set()


def test_recall_and_precision_at_k():
    assert recall_at_k(['a', 'b', 'c'], {'a', 'z'}, k=2) == 0.5
    assert precision_at_k(['a', 'b', 'c'], {'a', 'z'}, k=2) == 0.5
    assert recall_at_k(['a', 'b'], set(), k=2) == 0.0
    assert abs(precision_at_k(['a', 'b'], {'a', 'b'}, k=5) - 0.4) < 1e-9


def test_evaluate_ranking_values():
    import numpy as np
    result = evaluate_ranking(['x', 'a', 'y'], {'a'}, k=3)
    assert result['mrr'] == 0.5 and result['first_hit_rank'] == 2
    assert result['recall'] == 1.0 and abs(result['precision'] - 1 / 3) < 1e-9
    assert abs(result['ndcg'] - 1 / np.log2(3)) < 1e-9
    miss = evaluate_ranking(['x', 'y'], {'a'}, k=2)
    assert miss['mrr'] == 0.0 and miss['first_hit_rank'] is None and miss['recall'] == 0.0


def test_chunk_stats_and_dedupe():
    assert chunk_stats([5, 1, 3, 9, 7]) == {'n': 5, 'median': 5, 'p95': 9, 'max': 9, 'min': 1}
    assert chunk_stats(list(range(1, 21)))['p95'] == 19
    assert dedupe(['a', 'b', 'a', 'c', 'b']) == (['a', 'b', 'c'], 2)


def test_head_tail():
    assert head_tail('abcdefghij', n=3) == 'abc … hij'
    assert head_tail('abcdef', n=3) == 'abcdef'


def test_figure_helpers_smoke():
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    fig = length_hist_panels({'chars-300': [100, 250, 300, 300], 'section': [40, 900, 1400]})
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == 'chunks' and fig.axes[1].get_ylabel() == ''
    ax = sorted_score_plot([0.1, 0.9, 0.5], highlight_index=2)
    assert isinstance(ax, Axes) and len(ax.lines) == 2
    assert ax.lines[1].get_xdata()[0] == 2  # score 0.5 sits at rank 2
    plt.figure()
    ax = kde_plot([0.8, 0.85, 0.9], [0.1, 0.2, 0.3, 0.4])
    assert len(ax.lines) == 2 and len(ax.texts) == 2
    plt.close('all')


# --- integration on the real corpus (skipped when the data files are absent) --

CORPUS_MD = config.DATA_DIR / 'IT_Grundschutz_Kompendium_Edition2023.md'
WORKSHOP_BAUSTEINE = ['APP.3.2', 'APP.3.3', 'CON.3', 'CON.6', 'DER.3.1', 'ISMS.1', 'NET.1.1',
                      'OPS.1.1.6', 'OPS.1.2.2', 'OPS.1.2.4', 'OPS.1.2.5', 'OPS.2.2', 'OPS.2.3',
                      'ORP.1', 'ORP.4', 'ORP.5', 'SYS.1.1', 'SYS.1.2.3', 'SYS.1.8']


@functools.lru_cache(maxsize=1)
def _corpus_sections():
    return build_sections(CORPUS_MD.read_text(encoding='utf-8'))


def test_real_corpus_reproduces_the_analysis_counts():
    if not CORPUS_MD.exists():
        return
    sections = _corpus_sections()
    kinds = Counter(s['kind'] for s in sections)
    assert len(sections) == 4432
    assert (kinds['baustein'], kinds['requirement'], kinds['gefaehrdung'], kinds['template']) \
        == (111, 2123, 47, 1927)
    assert sum(s['is_entfallen'] for s in sections) == 289
    assert sum(s['is_front_matter'] for s in sections) == 213
    sliced = workshop_slice(sections, WORKSHOP_BAUSTEINE, extra_titles=(SCHICHTEN,))
    assert len({s['baustein_id'] for s in sliced} - {None}) == 19
    assert sum(s['kind'] == 'requirement' for s in sliced) == 385
    assert len(sections_to_text(workshop_slice(sections, WORKSHOP_BAUSTEINE))) == 391_899


def test_real_gold_rows_resolve_to_sections():
    if not CORPUS_MD.exists():
        return
    sections = _corpus_sections()
    texts = section_texts(sections)
    rows = load_gold('40_einfach')
    assert len(rows) == 40
    targets = [gold_targets(r['fundstelle'], sections, texts) for r in rows]
    assert all(targets)
    assert targets[0] == {f'front::{SCHICHTEN}'}
    assert targets[1] == {'APP.3.2.A1'}
    # 10 rows cite no requirement ID and resolve to a Baustein or front-matter section
    assert sum(1 for t in targets if any('::' in key for key in t)) == 10
    rows = load_gold('123_einfach')
    assert len(rows) == 123
    targets = [gold_targets(r['fundstelle'], sections, texts) for r in rows]
    # row 122 cites BSI-Standard 200-2, which is not in the Kompendium
    assert [i for i, t in enumerate(targets) if not t] == [122]
    # row 109 cites SYS.1.2.3.A5, which does not exist; its quoted text is SYS.1.2.2.A5
    assert targets[109] == {'SYS.1.2.2.A5'}


def test_entropy_softmax_separates_peaked_from_flat():
    peaked = entropy([0.9, 0.5, 0.4, 0.35, 0.3], temperature=0.05)
    flat = entropy([0.6, 0.6, 0.59, 0.59, 0.58], temperature=0.05)
    assert peaked < 0.3
    assert 2.0 < flat <= 2.33
    assert abs(entropy([1.0, 1.0, 1.0, 1.0], temperature=0.05) - 2.0) < 1e-9


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
    print(f'{len(tests)} tests passed')
