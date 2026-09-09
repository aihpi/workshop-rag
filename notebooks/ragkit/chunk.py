"""Text normalisation and chunking.

Splitting a document into retrievable units, plus the cleanup that has to
happen before any of it makes sense. Every function here is pure: no API
calls, no file access, no Qdrant. That makes them the part of the workshop
that is cheap to test and safe to experiment with.
"""

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Docling and OCR engines emit these placeholders instead of German umlauts.
_UMLAUT_MAP = {
    'C196': 'Ä',
    'C214': 'Ö',
    'C218': 'Ö',
    'C220': 'Ü',
    'C216': 'Ä',
    'C219': 'Ü',
    'C228': 'ä',
    'C229': 'ä',
    'C230': 'ö',
    'C231': 'ü',
    'C246': 'ö',
    'C252': 'ü',
    'C223': 'ß',
}


def _fix_german_umlauts(text: str) -> str:
    """Replace Docling/OCR placeholders like /C231 with real umlauts."""
    def repl(match: re.Match) -> str:
        code = match.group(1)
        if not code.startswith('C'):
            code = f'C{code}'
        return _UMLAUT_MAP.get(code, match.group(0))

    # Matches both /C231 and C231
    text = re.sub(r'/?(C\d{3})', repl, text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


def count_umlaut_placeholders(obj) -> int:
    """Count remaining umlaut placeholders anywhere in a JSON-serialisable object."""
    s = json.dumps(obj, ensure_ascii=False)
    return len(re.findall(r'/?C\d{3}', s))


def normalize_text(text: str, fix_umlauts: bool = True) -> str:
    """Normalise line endings, collapse whitespace and blank-line runs.

    With `fix_umlauts=True` (default) Docling/OCR placeholders like `/C231`
    are also repaired. w2_04 passes `False`: it compares OCR engines, so a
    placeholder there is a real quality difference that must stay visible.
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _fix_german_umlauts(text) if fix_umlauts else re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_json(doc_json: dict) -> dict:
    """Normalise only the content text fields of a Docling document, not metadata."""
    def walk(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in {'text', 'orig'} and isinstance(v, str):
                    out[k] = _fix_german_umlauts(v)
                else:
                    out[k] = walk(v)
            return out
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        return obj
    return walk(doc_json)


def format_citation(pages: list[int], lang: str = 'en'):
    """Render a page list as a citation hint, e.g. 'p. 12' or 'S. 12-15' (lang='de')."""
    if not pages:
        return None
    prefix = 'S.' if lang == 'de' else 'p.'
    if len(pages) == 1:
        return f'{prefix} {pages[0]}'
    return f'{prefix} {pages[0]}-{pages[-1]}'


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunk_by_paragraph(text: str) -> list[str]:
    """Split text into chunks at paragraph boundaries (blank lines)."""
    return [p.strip() for p in text.split('\n\n') if p.strip()]


def chunk_by_words(text: str, max_words: int = 100) -> list[str]:
    """Split text into chunks of at most max_words words."""
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def chunk_by_chars(text: str, max_chars: int = 1000, overlap: int = 0) -> list[str]:
    """Split text into chunks of at most max_chars characters.

    Breaks at the last whitespace before the limit to avoid splitting words.
    Optionally overlaps by `overlap` characters.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        # Find last space before the limit
        space_idx = text.rfind(' ', start, end)
        if space_idx <= start:
            space_idx = end  # no space found, hard cut
        chunk = text[start:space_idx].strip()
        if chunk:
            chunks.append(chunk)
        # Advance by stride, but respect the actual break point
        start = max(start + 1, space_idx - overlap)
    return chunks


def parser_aware_split(text: str, max_chunk: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks along paragraph boundaries.

    Also the fallback used by chunk_markdown_by_headers() to split oversized
    sections.

    Args:
        text: Text to split. Normalised internally.
        max_chunk: Maximum number of characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks with overlap applied.

    Raises:
        ValueError: If overlap >= max_chunk, which would prevent the
            oversized-block loop below from ever advancing.
    """
    if overlap >= max_chunk:
        raise ValueError('overlap must be smaller than max_chunk')

    text = normalize_text(text)
    if not text:
        return []

    blocks = text.split('\n\n')
    chunks = []
    current = ''

    for block in blocks:
        candidate = (current + '\n\n' + block).strip() if current else block
        if len(candidate) <= max_chunk:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(block) <= max_chunk:
            current = block
        else:
            start = 0
            while start < len(block):
                end = min(start + max_chunk, len(block))
                chunks.append(block[start:end])
                if end == len(block):
                    break
                start = end - overlap
            current = ''

    if current:
        chunks.append(current)

    if overlap > 0 and chunks:
        with_overlap = [chunks[0]]
        for i in range(1, len(chunks)):
            prefix = chunks[i - 1][-overlap:]
            with_overlap.append((prefix + '\n' + chunks[i]).strip())
        return with_overlap

    return chunks


def chunk_markdown_by_headers(markdown_text: str, max_chunk: int = 1200,
                              overlap: int = 200) -> list[str]:
    """Split along Markdown headings; oversized sections are split further."""
    text = normalize_text(markdown_text)
    if not text:
        return []

    parts = re.split(r'(?m)^(#{1,6}\s.+)$', text)
    sections = []
    if parts and parts[0].strip():
        sections.append(parts[0].strip())

    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ''
        sections.append(f'{header}\n\n{body}'.strip())

    chunks: list[str] = []
    for sec in sections:
        if len(sec) <= max_chunk:
            chunks.append(sec)
        else:
            chunks.extend(parser_aware_split(sec, max_chunk=max_chunk, overlap=overlap))

    return chunks


# ---------------------------------------------------------------------------
# Chunk records (text + metadata, ready for indexing)
# ---------------------------------------------------------------------------

def records_from_markdown_header_chunks(markdown_text: str, pdf_path: Path | str,
                                        max_chunk: int = 1200, overlap: int = 200) -> list[dict[str, Any]]:
    """Build indexable records by splitting Markdown at its headings.

    `pdf_path` is stamped into every record's metadata; the notebook that owns
    the document passes it, together with its own chunk-size constants.
    """
    pdf_path = Path(pdf_path)
    chunks = chunk_markdown_by_headers(markdown_text, max_chunk=max_chunk, overlap=overlap)
    return [
        {
            'chunk_id': i,
            'text': chunk_text,
            'metadata': {
                'source_file': pdf_path.name,
                'source_path': str(pdf_path),
                'doc_type': 'pdf',
                'converter': 'docling',
                'chunking_mode': 'markdown_headers',
                'max_chunk': max_chunk,
                'overlap': overlap,
                'total_chunks': len(chunks),
                # Exact page numbers usually cannot be derived reliably from plain Markdown
                'page_numbers': [],
                'citation_hint': None,
            },
        }
        for i, chunk_text in enumerate(chunks)
    ]


def records_from_docling_json_structured_sections(doc_json: dict, pdf_path: Path | str,
                                                  lang: str = 'en') -> list[dict[str, Any]]:
    """Structure-driven chunks: exactly from heading to heading, no max-chunk/overlap.

    `lang` selects the citation-hint language ('p. 12' vs 'S. 12').
    """
    pdf_path = Path(pdf_path)
    elements = doc_json.get('texts', [])
    if not isinstance(elements, list):
        return []

    # Labels that mark a new section boundary
    heading_labels = {'section_header', 'section_heading', 'heading', 'title'}

    def extract_page_numbers(node: dict) -> list[int]:
        pages = []
        prov = node.get('prov', [])
        if isinstance(prov, list):
            for p in prov:
                if isinstance(p, dict) and isinstance(p.get('page_no'), int):
                    pages.append(p['page_no'])
        return sorted(set(pages))

    items = []
    current_heading = ''
    current_parts: list[str] = []
    current_pages: list[int] = []

    def flush_section():
        nonlocal current_parts, current_pages
        if not current_parts:
            return

        body = normalize_text(' '.join(current_parts))
        if not body:
            current_parts = []
            current_pages = []
            return

        section_text = f'{current_heading}\n\n{body}'.strip() if current_heading else body
        page_numbers = sorted(set(current_pages))

        items.append({
            'text': section_text,
            'page_numbers': page_numbers,
            'citation_hint': format_citation(page_numbers, lang=lang),
        })

        current_parts = []
        current_pages = []

    for el in elements:
        if not isinstance(el, dict):
            continue

        # Exclude furniture (headers/footers/page numbers)
        if el.get('content_layer') == 'furniture':
            continue

        raw_text = el.get('text') or el.get('orig') or ''
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        text = normalize_text(raw_text)
        if not text:
            continue

        label = (el.get('label') or '').strip()
        pages = extract_page_numbers(el)

        if label in heading_labels or (label == 'page_header'
                                       and el.get('content_layer') != 'furniture'):
            flush_section()
            current_heading = text
            current_pages.extend(pages)
            continue

        current_parts.append(text)
        current_pages.extend(pages)

    flush_section()

    return [
        {
            'chunk_id': i,
            'text': item['text'],
            'metadata': {
                'source_file': pdf_path.name,
                'source_path': str(pdf_path),
                'doc_type': 'pdf',
                'converter': 'docling',
                'chunking_mode': 'json_structured_sections',
                'max_chunk': None,
                'overlap': 0,
                'total_chunks': len(items),
                'page_numbers': item['page_numbers'],
                'citation_hint': item['citation_hint'],
            },
        }
        for i, item in enumerate(items)
    ]


# ---------------------------------------------------------------------------
# IT-Grundschutz structure (w2_01)
# ---------------------------------------------------------------------------
# The Docling export of the Kompendium uses '## ' for every heading. Four header
# shapes carry the structure; everything else ('## Vorwort', glossary terms,
# layer separators) is 'other'. Verified on Edition 2023: 111 Bausteine, 2123
# requirements (289 of them ENTFALLEN), 47 Gefährdungen, 1927 template sections.

_BAUSTEIN_ID = r'[A-Z]{2,4}(?:\.\d{1,2}){1,3}'
REQ_ID_RE = re.compile(rf'\b({_BAUSTEIN_ID})\.A(\d{{1,2}})\b')
REQ_HEADER_RE = re.compile(
    rf'^## ({_BAUSTEIN_ID}\.A\d{{1,2}}) (.+?) \(([BSH])\)(?: \[([^\]]+)\])?$')
BAUSTEIN_HEADER_RE = re.compile(rf'^## ({_BAUSTEIN_ID}) (?!A\d)(\S.*)$')
GEFAEHRDUNG_HEADER_RE = re.compile(r'^## (G 0\.\d{1,2}) (.+)$')
TEMPLATE_HEADER_RE = re.compile(r'^## (\d{1,2}\.(?:\d{1,2}\.)?) (.+)$')
# Page furniture: Docling image placeholders plus the running-head and
# page-number shapes a PDF export usually leaves behind.
FURNITURE_RE = re.compile(
    r'^(<!-- image -->|IT-Grundschutz-Kompendium: Stand \w+ \d{4}.*|\d{1,4}|Seite \d+)$')
# BSI modal verbs as whole tokens. Negated forms come first in the alternation
# so that 'SOLLTE NICHT' is not also counted as 'SOLLTE'.
MODAL_RE = re.compile(
    r'\b(?:(?P<muss>MUSS|MÜSSEN)'
    r'|(?P<darf_nur>D(?:ARF|ÜRFEN) NUR)'
    r'|(?P<darf_nicht>D(?:ARF|ÜRFEN) (?:NICHT|KEIN\w*))'
    r'|(?P<sollte_nicht>SOLLTEN? (?:NICHT|KEIN\w*))'
    r'|(?P<sollte>SOLLTEN?))\b')
_MODAL_LABELS = {'muss': 'MUSS', 'darf_nur': 'DARF NUR', 'darf_nicht': 'DARF NICHT',
                 'sollte': 'SOLLTE', 'sollte_nicht': 'SOLLTE NICHT'}
_HEADER_KINDS = (('requirement', REQ_HEADER_RE), ('baustein', BAUSTEIN_HEADER_RE),
                 ('gefaehrdung', GEFAEHRDUNG_HEADER_RE), ('template', TEMPLATE_HEADER_RE))
# Roles are comma-separated, but one role may carry a parenthesised expansion
# with its own comma: '[OT-Betrieb (Operational Technology, OT), Planende]'.
# ponytail: one level of parentheses is all the corpus has.
_ROLE_SPLIT_RE = re.compile(r',\s*(?![^()]*\))')


def modality(text: str) -> dict[str, int]:
    """Count the BSI modal verbs in `text`, one entry per class."""
    counts = dict.fromkeys(_MODAL_LABELS.values(), 0)
    for m in MODAL_RE.finditer(text):
        counts[_MODAL_LABELS[m.lastgroup]] += 1
    return counts


def classify_header(line: str) -> str:
    """Kind of a '## ' line: requirement, baustein, gefaehrdung, template or other."""
    for kind, rx in _HEADER_KINDS:
        if rx.match(line):
            return kind
    return 'other'


def parse_requirement_header(line: str) -> dict[str, Any] | None:
    """Split '## APP.3.2.A1 Titel (B) [Rolle, Rolle]' into req_id, title, level and roles."""
    m = REQ_HEADER_RE.match(line)
    if m is None:
        return None
    req_id, title, level, roles = m.groups()
    return {'req_id': req_id, 'title': title, 'level': level,
            'roles': _ROLE_SPLIT_RE.split(roles) if roles else []}


def strip_furniture(md: str) -> str:
    """Drop furniture lines (image markers, running heads, page numbers), then normalise."""
    kept = [line for line in md.split('\n') if not FURNITURE_RE.match(line.strip())]
    return normalize_text('\n'.join(kept))


def build_sections(md: str) -> list[dict[str, Any]]:
    """One record per '## ' header: the header line, its body and where it sits.

    `md` may be the raw Docling export; furniture is stripped and the text
    normalised first. Everything before the first Baustein header is front
    matter (`baustein_id` None). `title` is the header text without '## ',
    except for requirements and Bausteine, whose ID moves into its own field.
    `section_path` is Baustein > nearest template section > requirement, e.g.
    'APP.3.2 > 3.1. Basis-Anforderungen > APP.3.2.A1 Sichere Konfiguration ...'.
    """
    lines = strip_furniture(md).split('\n')
    header_idx = [i for i, line in enumerate(lines) if line.startswith('## ')]
    sections: list[dict[str, Any]] = []
    baustein_id = baustein_title = template = None
    for k, i in enumerate(header_idx):
        end = header_idx[k + 1] if k + 1 < len(header_idx) else len(lines)
        header = lines[i]
        body = '\n'.join(lines[i + 1:end]).strip()
        kind = classify_header(header)
        req = parse_requirement_header(header) or {'req_id': None, 'level': None, 'roles': []}
        title = leaf = header[3:]
        if kind == 'baustein':
            baustein_id, title = BAUSTEIN_HEADER_RE.match(header).groups()
            baustein_title, template, leaf = title, None, None
        elif kind == 'template':
            template, leaf = title, None
        elif kind == 'requirement':
            title = req['title']
            leaf = f"{req['req_id']} {title}"
        text = f'{header}\n\n{body}' if body else header
        sections.append({
            'kind': kind, 'title': title, 'text': text, 'body': body,
            'baustein_id': baustein_id, 'baustein_title': baustein_title,
            'section_path': ' > '.join(p for p in (baustein_id, template, leaf) if p),
            'req_id': req['req_id'], 'level': req['level'], 'roles': req['roles'],
            'modality': modality(body),
            'is_entfallen': kind == 'requirement' and title == 'ENTFALLEN',
            'is_front_matter': baustein_id is None,
            'n_chars': len(text),
        })
    return sections


def workshop_slice(sections: list[dict[str, Any]], baustein_ids,
                   extra_titles=()) -> list[dict[str, Any]]:
    """Sections of the given Bausteine plus front-matter sections named in `extra_titles`."""
    ids, titles = set(baustein_ids), set(extra_titles)
    return [s for s in sections
            if s['baustein_id'] in ids or (s['is_front_matter'] and s['title'] in titles)]


def sections_to_text(sections: list[dict[str, Any]]) -> str:
    """Join section texts (header lines kept) into one document for the fixed-size chunkers."""
    return '\n\n'.join(s['text'] for s in sections)


def chunk_stats(lengths) -> dict[str, int | float]:
    """n, median, p95, max and min of a list of chunk lengths."""
    v = sorted(lengths)
    return {'n': len(v), 'median': statistics.median(v), 'p95': v[math.ceil(0.95 * len(v)) - 1],
            'max': v[-1], 'min': v[0]}


def dedupe(chunks: list[str]) -> tuple[list[str], int]:
    """Drop exact duplicates keeping first occurrences; also returns how many were dropped."""
    unique = list(dict.fromkeys(chunks))
    return unique, len(chunks) - len(unique)
