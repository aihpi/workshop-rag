"""Shared helpers for the RAG workshop notebooks.

The notebooks run with `notebooks/` as the working directory, so this package
imports with no install step:

    from ragkit.embed import embed, cached_embed
    from ragkit.chunk import chunk_by_paragraph, normalize_text
    from ragkit.search import rag_search, entropy

`ragkit` holds the plumbing that every notebook repeats: API calls, batching,
caching, chunking, similarity, ranking metrics. It deliberately does not hold
the code that *is* the lesson, which stays visible in the notebook cells.

Submodules are not imported here, so `import ragkit` stays cheap and a notebook
that does no OCR never pays for the docling import.
"""

__all__ = ['chunk', 'config', 'embed', 'search', 'viz']
