"""Shared helpers for the RAG workshop notebooks.

marimo puts the notebook directory on `sys.path`, so this package imports with
no install step:

    from ragkit.config import setup
    from ragkit import theme
    from ragkit.embed import embed, cached_embed
    from ragkit.chunk import chunk_by_paragraph, normalize_text
    from ragkit.search import rag_search, entropy

`config.setup()` validates key, Qdrant and data files once per notebook;
`theme` carries the HPI look for notebook chrome and figures; `evaluate` maps
gold Fundstellen and chunks onto Kompendium sections; `crawl` fetched the bird
dataset once and documents how such data can be collected.

`ragkit` holds the plumbing that every notebook repeats: API calls, batching,
caching, chunking, similarity, ranking metrics. It deliberately does not hold
the code that *is* the lesson, which stays visible in the notebook cells.

Submodules are not imported here, so `import ragkit` stays cheap and a notebook
that does no OCR never pays for the docling import.
"""

__all__ = ['chunk', 'config', 'crawl', 'embed', 'evaluate', 'search', 'theme', 'viz']
