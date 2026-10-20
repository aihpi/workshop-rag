"""Loading the IT-Grundschutz-Kompendium into Qdrant, for instructors.

The workshop notebooks run on cached results and need no Qdrant. These functions rebuild the
collections that `tools/precompute_w3_02.py` and `tools/precompute_w3_03.py` read when the
caches are recomputed, and are used by `tools/make_qdrant_snapshot.py`. The steps are
the ones from Part II (w2_03): Docling, chunking, embedding, upload. What is new
is that two collections are built, one per chunking method, with the fixed
names that w3_02 and w3_03 expect, and that every step can be repeated: an
interrupted upload resumes with the chunks that are still missing.

The fallback is a Qdrant snapshot per collection, stored as
`data/qdrant_snapshot/<collection>.snapshot.gz` and uploaded through the Qdrant
REST API.
"""

import json
from pathlib import Path

from .config import (
    DATA_DIR,
    EMBED_BATCH_SIZE,
    EMBED_SHORT,
    MAX_CHUNK,
    OVERLAP,
    PDF_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
)

# One collection per chunking method; json_structured_sections is the one w3_02 and w3_03 use.
METHODS = ('json_structured_sections', 'markdown_headers')
COLLECTIONS = {m: f'grundschutz_chunks_{m}__{EMBED_SHORT}' for m in METHODS}

MARKDOWN_CACHE = DATA_DIR / f'{PDF_PATH.stem}.md'
JSON_CACHE = DATA_DIR / f'{PDF_PATH.stem}.json'
SNAPSHOT_DIR = DATA_DIR / 'qdrant_snapshot'
QDRANT_URL = f'http://{QDRANT_HOST}:{QDRANT_PORT}'


# ---------------------------------------------------------------------------
# Docling and chunking
# ---------------------------------------------------------------------------

def docling_cached() -> bool:
    """True when the Docling result of the compendium (Markdown and JSON) is on disk."""
    return MARKDOWN_CACHE.exists() and JSON_CACHE.exists()


def load_docling() -> tuple[str, dict]:
    """Markdown and JSON export of the compendium, from the cache or converted once (about 4 min)."""
    if docling_cached():
        return MARKDOWN_CACHE.read_text(encoding='utf-8'), json.loads(JSON_CACHE.read_text(encoding='utf-8'))
    from docling.document_converter import DocumentConverter

    doc = DocumentConverter().convert(str(PDF_PATH)).document
    markdown_text, doc_json = doc.export_to_markdown(), doc.export_to_dict()
    MARKDOWN_CACHE.write_text(markdown_text, encoding='utf-8')
    JSON_CACHE.write_text(json.dumps(doc_json, ensure_ascii=False, indent=2), encoding='utf-8')
    return markdown_text, doc_json


def build_records(markdown_text: str, doc_json: dict) -> dict[str, list[dict]]:
    """Chunk records `{chunk_id, text, metadata}` for both methods, keyed by method."""
    from .chunk import (
        normalize_json,
        records_from_docling_json_structured_sections,
        records_from_markdown_header_chunks,
    )

    return {
        'json_structured_sections': records_from_docling_json_structured_sections(
            normalize_json(doc_json), PDF_PATH, lang='de'),
        'markdown_headers': records_from_markdown_header_chunks(markdown_text, PDF_PATH, MAX_CHUNK, OVERLAP),
    }


_records: dict[str, list[dict]] | None = None


def cached_records() -> dict[str, list[dict]] | None:
    """`build_records` from the Docling cache (about half a second, then kept), or None without a cache."""
    global _records
    if _records is None and docling_cached():
        _records = build_records(*load_docling())
    return _records


def expected_counts() -> dict[str, int] | None:
    """Number of chunks each collection should hold, keyed by collection name."""
    records = cached_records()
    return None if records is None else {COLLECTIONS[m]: len(r) for m, r in records.items()}


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def qdrant():
    """A fresh Qdrant client for the configured host."""
    from qdrant_client import QdrantClient

    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)


def collection_counts(client=None) -> dict[str, int | None]:
    """Points per collection, None for a collection that does not exist."""
    client = client or qdrant()
    return {name: (client.get_collection(name).points_count if client.collection_exists(name) else None)
            for name in COLLECTIONS.values()}


def complete(counts: dict[str, int | None], expected: dict[str, int] | None) -> dict[str, bool]:
    """A collection is complete when it holds every chunk (or, without a Docling cache, any chunk)."""
    return {name: n is not None and n > 0 and (expected is None or n >= expected[name])
            for name, n in counts.items()}


def stored_ids(client, name: str) -> set[int]:
    """Ids of the points already in `name` (empty when the collection does not exist)."""
    if not client.collection_exists(name):
        return set()
    ids, offset = set(), None
    while True:
        points, offset = client.scroll(name, limit=2000, offset=offset, with_payload=False, with_vectors=False)
        ids.update(int(p.id) for p in points)
        if offset is None:
            return ids


def pending_batches(records: list[dict], done: set[int], batch_size: int = EMBED_BATCH_SIZE) -> list[list[dict]]:
    """The records not yet in Qdrant, in embedding batches."""
    todo = [r for r in records if r['chunk_id'] not in done]
    return [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]


def upload_batch(client, name: str, batch: list[dict]) -> None:
    """Embed one batch and upsert it; creates the collection on the first batch."""
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from .embed import embed

    vectors = embed([r['text'] for r in batch])
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
    client.upsert(name, points=[
        PointStruct(id=r['chunk_id'], vector=v.tolist(),
                    payload={'text': r['text'], 'chunk_id': r['chunk_id'], **r['metadata']})
        for r, v in zip(batch, vectors)
    ], wait=True)


def search(client, name: str, vector, limit: int = 3) -> list[dict]:
    """Top hits of one collection as plain dicts (score, text, page_numbers)."""
    points = client.query_points(name, query=list(map(float, vector)), limit=limit, with_payload=True).points
    return [{'score': p.score, 'text': (p.payload or {}).get('text', ''),
             'page_numbers': (p.payload or {}).get('page_numbers') or []} for p in points]


# ---------------------------------------------------------------------------
# Snapshots (fallback when the ingestion fails)
# ---------------------------------------------------------------------------
# A raw Qdrant snapshot of one collection is about 450 MB, mostly preallocated
# WAL and segment space; gzip brings it to about 70 MB. Files are stored
# gzipped and unpacked to a temporary file just before the upload.

def snapshot_path(name: str, directory: Path = SNAPSHOT_DIR) -> Path:
    return Path(directory) / f'{name}.snapshot.gz'


def snapshot_files(directory: Path = SNAPSHOT_DIR) -> dict[str, Path]:
    """Snapshot files on disk for the two collections (`.snapshot.gz` or `.snapshot`), keyed by name."""
    found = {}
    for name in COLLECTIONS.values():
        for path in (snapshot_path(name, directory), Path(directory) / f'{name}.snapshot'):
            if path.exists():
                found[name] = path
                break
    return found


def create_snapshot(name: str, directory: Path = SNAPSHOT_DIR, url: str = QDRANT_URL) -> Path:
    """Snapshot `name` on the server, download it gzipped to `directory`, then delete it on the server."""
    import gzip

    import requests

    client = qdrant()
    snap = client.create_snapshot(name, wait=True)
    target = snapshot_path(name, directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.part')
    with requests.get(f'{url}/collections/{name}/snapshots/{snap.name}', stream=True, timeout=(10, 600)) as r:
        r.raise_for_status()
        with gzip.open(tmp, 'wb', compresslevel=6) as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
    tmp.replace(target)
    client.delete_snapshot(name, snap.name, wait=True)
    return target


def restore_snapshot(path: Path, name: str, url: str = QDRANT_URL) -> int:
    """Upload a snapshot file as collection `name` (replacing it if present); returns its points."""
    import gzip
    import shutil
    import tempfile

    import requests

    path = Path(path)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp_dir:
        raw = path
        if path.suffix == '.gz':
            raw = Path(tmp_dir) / path.stem
            with gzip.open(path, 'rb') as src, open(raw, 'wb') as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
        with open(raw, 'rb') as f:
            r = requests.post(f'{url}/collections/{name}/snapshots/upload',
                              params={'priority': 'snapshot', 'wait': 'true'},
                              files={'snapshot': (raw.name, f, 'application/octet-stream')},
                              timeout=(10, 1800))
    r.raise_for_status()
    return qdrant().get_collection(name).points_count
