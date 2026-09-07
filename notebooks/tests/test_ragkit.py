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


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
    print(f'{len(tests)} tests passed')
