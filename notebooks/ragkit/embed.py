"""Turning text and images into vectors.

`embed()` is the one function most notebooks need. The `embed_*` backends below
exist for the model comparison in w2_02, where the point is that different
models behave differently on the same input.

Everything that talks to the embedding API lives here, including the plumbing
around it: retries, batching, truncation, image downscaling and the on-disk
cache. Embeddings cost time and money, so nothing is computed twice.
"""

import base64
import io
import os
import time
from pathlib import Path

import numpy as np

from .config import (
    API_BASE_URL,
    EMBED_BATCH_SIZE,
    EMBED_MAX_CHARS,
    EMBED_MODEL_NAME,
    WORKSHOP_DIR,
)

MAX_RETRIES = 4
RETRY_BACKOFF = 2  # seconds; doubles each retry → 2, 4, 8, 16
IMG_MAX_PIXELS = 768  # resize images so the longest side is at most this

# Absolute, so the cache is the same directory no matter where Jupyter was
# launched from. Distinct from config.CACHE_DIR, which caches Workshop 3
# evaluation results rather than vectors.
CACHE_DIR = WORKSHOP_DIR / 'embedding_cache'

RECREATE = False  # set ragkit.embed.RECREATE = True to ignore the cache and re-embed


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

_client = None


def client():
    """Return the shared OpenAI-compatible client, created on first use.

    Deferred so that importing ragkit does not require credentials, and so a
    missing key fails with a clear message at the point of use.
    """
    global _client
    if _client is None:
        import openai
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OPENAI_API_KEY is missing. Set it in notebooks/.env')
        _client = openai.OpenAI(api_key=api_key, base_url=API_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def batched(iterable, batch_size):
    """Yield successive slices of `iterable` of at most `batch_size` items.

    This is `itertools.batched` from Python 3.12, kept local because
    pyproject.toml still supports 3.10.
    """
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def _retry(fn, *args, label='api', **kwargs):
    """Call `fn(*args, **kwargs)` with exponential-backoff retries on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            transient = (
                '500' in msg
                or 'connection' in msg          # aborted / reset / error
                or 'internalservererror' in msg
                or 'timeout' in msg or 'timed out' in msg
                or 'remote disconnected' in msg
            )
            if transient and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f'  ⟳ {label}: attempt {attempt}/{MAX_RETRIES} failed '
                      f'({exc.__class__.__name__}), retrying in {wait}s...')
                time.sleep(wait)
            else:
                raise


def _truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text to approximately `max_chars` characters (~200 tokens).

    miniLM has a hard 256-token limit; this keeps inputs safely below it.
    Truncates at the last space before the limit to avoid mid-word cuts.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated


def _extract_embedding(item):
    """Pull the vector out of one element of an embedding response."""
    if isinstance(item, dict):
        return item.get('embedding')
    return getattr(item, 'embedding', None)


# ---------------------------------------------------------------------------
# Image plumbing (images have to become base64 before they can be embedded)
# ---------------------------------------------------------------------------

def img_to_base64(path) -> str:
    """Read an image file and return it base64-encoded."""
    return base64.b64encode(Path(path).read_bytes()).decode('utf-8')


def _resize_image_b64(b64: str, max_side: int = IMG_MAX_PIXELS) -> str:
    """Resize a base64 image so its longest side is at most max_side pixels.

    Returns re-encoded base64, or the input unchanged if already small enough.
    This keeps token counts within the 32K context window of qwen3-vl.
    """
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    w, h = img.size
    if max(w, h) <= max_side:
        return b64
    scale = max_side / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Re-encode as JPEG (smaller than PNG) unless the image has alpha
    buf = io.BytesIO()
    fmt = 'PNG' if img.mode == 'RGBA' else 'JPEG'
    img.save(buf, format=fmt, quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _to_data_uri(b64: str) -> str:
    """Wrap a raw base64 string in a data URI if not already prefixed."""
    if b64.startswith('data:'):
        return b64
    return f'data:image/jpeg;base64,{b64}'


def pil_to_base64_data_url(img) -> str:
    """Encode a PIL image as a PNG data URI, ready for a vision model."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------

def embed(texts, model: str = EMBED_MODEL_NAME, batch_size: int = EMBED_BATCH_SIZE,
          max_chars: int = EMBED_MAX_CHARS):
    """Embed a string or list of strings. Returns a list of numpy arrays.

    The defaults come from config.py, so changing the model there changes it for
    every notebook at once, along with the batch size and character limit that
    model needs (miniLM caps at 350 characters, octen has no limit).
    """
    from litellm import embedding

    key = os.getenv('OPENAI_API_KEY', '')
    if not key or key.startswith('your_'):
        raise RuntimeError(
            'OPENAI_API_KEY is missing or still the placeholder. '
            'Copy notebooks/.env.example to notebooks/.env and insert your key.'
        )

    if isinstance(texts, str):
        texts = [texts]
    if max_chars:
        texts = [_truncate(t, max_chars) for t in texts]

    vectors = []
    for batch in batched(texts, batch_size):
        resp = _retry(
            embedding,
            model=model, input=batch, api_base=API_BASE_URL, encoding_format='float',
            label=f'embed {model}',
        )
        for item in resp.data:
            vec = _extract_embedding(item)
            if vec is None:
                raise RuntimeError('Received an embedding response without a vector.')
            vectors.append(np.asarray(vec, dtype=np.float32))
        if len(texts) > batch_size:  # single-query calls stay quiet
            print(f'  Embedded {len(vectors)}/{len(texts)}')
    return vectors


def cached_embed(key: str, fn, cache_dir=None, recreate: bool | None = None) -> np.ndarray:
    """Return cached vectors from <cache_dir>/<key>.npy, or compute via fn() and persist.

    Args:
        key: stable filename stub, e.g. "corpus_octen_4096".
        fn: zero-arg callable returning the array, or a list of arrays which is
            stacked automatically.
        cache_dir: override the module-level CACHE_DIR.
        recreate: override the module-level RECREATE toggle.

    Returns:
        A 2-D float32 array, one row per input text.
    """
    cache_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    recreate = RECREATE if recreate is None else recreate

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{key}.npy'
    if path.exists() and not recreate:
        print(f'  [cache hit ] {path.name}')
        return np.load(path)

    print(f'  [cache miss] {path.name} — computing ...')
    result = fn()
    if isinstance(result, list):
        rows = [np.asarray(x) for x in result]
        arr = np.vstack([r.reshape(1, -1) if r.ndim == 1 else r for r in rows])
    else:
        arr = np.asarray(result)
    arr = arr.astype(np.float32)
    np.save(path, arr)
    return arr


# ---------------------------------------------------------------------------
# Model-specific backends (w2_02 model comparison)
# ---------------------------------------------------------------------------

def embed_minilm(texts) -> list:
    """Embed texts with minilm-embedding (text-only, lightweight, 384 dimensions).

    miniLM has a 256-token limit, so long texts are truncated automatically.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [_truncate(t) for t in texts]
    resp = _retry(
        client().embeddings.create,
        input=texts, model='minilm-embedding', encoding_format='float',
        label='miniLM',
    )
    return [np.array(d.embedding) for d in resp.data]


def embed_octen(texts) -> list:
    """Embed texts with octen-embedding-8b (text-only, high-dimensional)."""
    if isinstance(texts, str):
        texts = [texts]
    resp = _retry(
        client().embeddings.create,
        input=texts, model='octen-embedding-8b', encoding_format='float',
        label='octen',
    )
    return [np.array(d.embedding) for d in resp.data]


def embed_dinov3(images_b64: list) -> list:
    """Embed images with dinov3-vit-large (image-only).

    Each element of images_b64 is a base64-encoded image.

    Uses a direct HTTP request because the OpenAI Python SDK automatically
    injects `encoding_format`, which this provider does not support. Images are
    resized first to keep the upload payload small.
    """
    import requests

    results = []
    for idx, b64 in enumerate(images_b64):
        payload_input = _to_data_uri(_resize_image_b64(b64))

        def _do_dinov3_request(payload=payload_input):
            resp = requests.post(
                f'{API_BASE_URL.rstrip("/")}/embeddings',
                json={'input': [payload], 'model': 'dinov3-vit-large'},
                headers={
                    'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                    'Content-Type': 'application/json',
                },
                timeout=(10, 120),  # (connect, read) — tuple avoids write stalls
            )
            if not resp.ok:
                raise RuntimeError(f'dinov3 error {resp.status_code}: {resp.text[:300]}')
            return resp

        resp = _retry(_do_dinov3_request, label=f'dinov3 {idx + 1}/{len(images_b64)}')
        results.append(np.array(resp.json()['data'][0]['embedding']))
    return results


def _check_image_was_seen(resp, uri_chars: int) -> None:
    """Raise when the endpoint embedded the data URI as text instead of as an image.

    A 512 px photo is a few hundred vision tokens; base64 text of the same photo
    is tens of thousands of tokens. Observed on the AISC proxy in September 2026:
    prompt_tokens grew with the URI length and unrelated photos came out 0.97
    similar, so every image "embedding" was in fact a text embedding of base64.
    """
    tokens = getattr(getattr(resp, 'usage', None), 'prompt_tokens', None)
    if tokens is not None and tokens > 2000 and tokens > uri_chars / 10:
        raise RuntimeError(
            f'qwen3-vl-embedding-8b counted {tokens} prompt tokens for one image: the endpoint '
            'embedded the data URI as text, not as a picture. Ask the API admins to enable '
            'multimodal input for this model before using image embeddings.')


def embed_qwen3vl(texts: list | None = None, images_b64: list | None = None) -> list:
    """Embed text and/or images with qwen3-vl-embedding-8b (multimodal).

    Provide texts, images, or both. When both are given they are paired 1:1.

    Images are resized, wrapped in a data URI, and sent ONE AT A TIME to stay
    within the 32K token context window. Text can be batched.
    """
    if texts and not images_b64:
        inputs = texts if isinstance(texts, list) else [texts]
        resp = _retry(
            client().embeddings.create,
            input=inputs, model='qwen3-vl-embedding-8b', encoding_format='float',
            label='qwen3-vl',
        )
        return [np.array(d.embedding) for d in resp.data]

    if images_b64 and not texts:
        imgs = images_b64 if isinstance(images_b64, list) else [images_b64]
        results = []
        for idx, b64 in enumerate(imgs):
            data_uri = _to_data_uri(_resize_image_b64(b64))
            resp = _retry(
                client().embeddings.create,
                input=[data_uri], model='qwen3-vl-embedding-8b', encoding_format='float',
                label=f'qwen3-vl img {idx + 1}/{len(imgs)}',
            )
            _check_image_was_seen(resp, len(data_uri))
            results.append(np.array(resp.data[0].embedding))
        return results

    if texts and images_b64:
        results = []
        for idx, (t, b) in enumerate(zip(texts, images_b64)):
            data_uri = _to_data_uri(_resize_image_b64(b))
            resp = _retry(
                client().embeddings.create,
                input=[t, data_uri], model='qwen3-vl-embedding-8b', encoding_format='float',
                label=f'qwen3-vl pair {idx + 1}',
            )
            results.append(np.array(resp.data[0].embedding))
        return results

    raise ValueError('Provide texts, images_b64, or both')


def embed_images_qwen3vl_safe(images: list, max_pixels: int = 512) -> tuple:
    """Embed images with qwen3-vl, retrying at smaller sizes on context-window errors.

    Args:
        images: dicts with 'base64' and 'name' keys.
        max_pixels: starting size; halved twice before giving up on an image.

    Returns:
        (list_of_vectors, list_of_successful_indices). Images that never fit are
        skipped, so the caller must use the indices to realign its labels.
    """
    vecs, ok_indices = [], []
    for idx, img_dict in enumerate(images):
        b64 = img_dict['base64']
        embedded = False
        for px in [max_pixels, max_pixels // 2, max_pixels // 4]:
            try:
                data_uri = _to_data_uri(_resize_image_b64(b64, max_side=px))
                resp = _retry(
                    client().embeddings.create,
                    input=[data_uri], model='qwen3-vl-embedding-8b', encoding_format='float',
                    label=f'qwen3-vl img {idx + 1}/{len(images)} ({px}px)',
                )
                vecs.append(np.array(resp.data[0].embedding))
                ok_indices.append(idx)
                embedded = True
                break
            except Exception as exc:
                if 'ContextWindow' in str(exc) or 'context length' in str(exc):
                    print(f'  ⚠ {img_dict["name"]}: too large at {px}px, retrying smaller...')
                    continue
                raise
        if not embedded:
            print(f'  ✗ Skipped {img_dict["name"]} — still exceeds context window '
                  f'at {max_pixels // 4}px')
    return vecs, ok_indices
