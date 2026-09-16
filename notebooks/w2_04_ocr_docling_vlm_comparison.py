# ruff: noqa: PLR1711,B018,BLE001,F841,RUF059  marimo returns; remaining codes are legacy notebook code kept as is
import marimo

__generated_with = "0.24.0"
app = marimo.App(css_file="ragkit/theme.css", html_head_file="ragkit/head.html")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 - OCR Comparison: Docling OCR vs. Docling VLM vs. External VLM

    To a computer, many PDFs are just **photos of text**: they contain pixels, but no (usable) searchable text. **OCR** (Optical Character Recognition) turns those pixels back into text — which makes it the very first step of almost every RAG pipeline. If this step is done sloppily, no chunking or embedding strategy, however good, can repair the damage later.

    Two philosophies compete here:
    - **Classical OCR** recognizes letter *shapes* one at a time — fast, free, runs locally.
    - A **VLM** (Vision-Language Model) looks at the page as a whole, the way a human does, and writes down what it sees — including tables and formulas. Better, but slower and (for external models) with API costs.

    This trade-off — **speed & cost versus structural and character fidelity** — is the core theme of this notebook.

    <img src="/public/img/w2_04_overview.svg" alt="One scanned PDF branches into four OCR approaches that converge into a comparison; an inset shows the speed-versus-fidelity trade-off" style="max-width:100%; width:860px">

    We compare four approaches on a hard-to-read PDF with tables and formulas:
    - `notebooks/raw_data/lstm_tables.pdf` (6 pages from the LSTM paper by Hochreiter & Schmidhuber)

    The four approaches:
    1. **Docling + RapidOCR** (classical local OCR)
    2. **Docling VLM pipeline** (local vision-language model, Granite preset)
    3. **External VLM** via LiteLLM (API, on page images)
    4. **Docling ApiVlmOptions** (same API, but with structured Docling output)

    At the end we compare the outputs quantitatively (metrics, **runtime**, similarity matrix) and take a close look at tables and formulas.

    **Workshop note:** all four outputs ship pre-computed (in `processed/ocr_compare/`) and load instantly by default. Set `RERUN_OCR = True` in section 1 to run the approaches yourself — the runtime column in section 7 only fills up for live runs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Configuration

    As in notebook 03, we collect all the **tuning knobs** in one place: paths, API access, and the model names of the three VLM variants. On Apple Silicon, the notebook automatically detects whether MLX acceleration is available for the local VLM.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import os
    import platform
    from pathlib import Path

    def auto_use_mlx() -> bool:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        try:
            import mlx  # noqa: F401
            return True
        except Exception:
            return False

    from ragkit import theme
    from ragkit.config import WORKSHOP_DIR, setup

    RAW_DIR = WORKSHOP_DIR / 'raw_data'
    OUT_DIR = WORKSHOP_DIR / 'processed' / 'ocr_compare'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH = RAW_DIR / 'lstm_tables.pdf'
    env = setup(required_files=(PDF_PATH,), strict=False)
    theme.apply_mpl()


    API_BASE_URL = os.getenv('OPENAI_API_BASE', 'https://api.aisc.hpi.de/')
    OPENAI_API_KEY_SET = bool(os.getenv('OPENAI_API_KEY'))

    # Adjust model names to your LiteLLM deployment if needed
    EXTERNAL_VLM_MODEL = os.getenv('EXTERNAL_VLM_MODEL', 'openai/gemma-4-31b')
    DOCLING_API_VLM_MODEL = os.getenv('DOCLING_API_VLM_MODEL', "gemma-4-31b")
    DOCLING_API_VLM_URL = os.getenv('DOCLING_API_VLM_URL', API_BASE_URL.rstrip('/') + '/v1/chat/completions')

    # Docling VLM configuration
    VLM_PRESET = os.getenv('DOCLING_VLM_PRESET', 'granite_docling')
    VLM_USE_MLX = auto_use_mlx()

    # Workshop pacing: by default the pre-computed OCR outputs in processed/ocr_compare/
    # load instantly. Set to True to run the four approaches live yourself
    # (requires a working API key and, for approach B, a local VLM).
    RERUN_OCR = False

    return (
        env,
        API_BASE_URL,
        DOCLING_API_VLM_MODEL,
        DOCLING_API_VLM_URL,
        EXTERNAL_VLM_MODEL,
        OUT_DIR,
        PDF_PATH,
        Path,
        RERUN_OCR,
        VLM_PRESET,
        VLM_USE_MLX,
        json,
        os,
    )



@app.cell(hide_code=True)
def _(env, mo):
    mo.md(env.summary_md())
    return

@app.cell(hide_code=True)
def _(
    PDF_PATH,
    show_panel,
):
    if not PDF_PATH.exists():
        raise FileNotFoundError(f'PDF not found: {PDF_PATH}')
    show_panel('Input document', f'{PDF_PATH.name}\nReady for OCR comparison.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Helper functions

    For the comparison to be fair, all four approaches need the same ground rules: the same text cleanup, the same storage format, and the same metrics. From now on we also stop the clock on every OCR run — **runtime** is half the truth in the speed-vs-quality trade-off.

    The simple metrics (`basic_metrics`) count characters, words, lines, plus math and table markers — a rough first impression only. Fair warning: these numbers are deliberately naive and **will mislead you** — section 8 shows exactly how. Treat them with skepticism from the start.
    """)
    return


@app.cell(hide_code=True)
def _(OUT_DIR, Path):
    import time

    from ragkit.chunk import normalize_text

    # Wall-clock runtimes of all OCR runs (seconds) – shown in the comparison (section 7)
    TIMINGS: dict[str, float] = {}


    def save_text(name: str, text: str) -> Path:
        p = OUT_DIR / f'{name}.md'
        p.write_text(normalize_text(text, fix_umlauts=False), encoding='utf-8')
        return p

    # Deliberately naive metrics — they WILL mislead you. Section 8 shows exactly how.
    def basic_metrics(text: str) -> dict:
        t = normalize_text(text, fix_umlauts=False)
        return {
            'chars': len(t),
            'words': len(t.split()),
            'lines': len(t.splitlines()),
            'math_markers': sum(t.count(x) for x in ['$', '\\(', '\\)', '\\[', '\\]']),
            'table_markers': sum(t.count(x) for x in ['|', '\t']),
        }

    # Pretty text previews (nicer than raw prints) — used by the OCR sections below
    import html as _html

    from IPython.display import HTML, display

    def text_card(title: str, text: str, max_chars: int = 1500, height: int = 260) -> str:
        """One scrollable preview card as an HTML string."""
        body = _html.escape(normalize_text(text, fix_umlauts=False)[:max_chars])
        return (
            f'<div style="border:1px solid #8b949e55;border-radius:8px;overflow:hidden;'
            f'font-family:system-ui,sans-serif;">'
            f'<div style="padding:6px 12px;font-size:12px;font-weight:600;opacity:0.75;'
            f'border-bottom:1px solid #8b949e55;">{_html.escape(title)}</div>'
            f'<pre style="margin:0;padding:10px 12px;max-height:{height}px;overflow:auto;'
            f'font-size:12.5px;line-height:1.45;white-space:pre-wrap;">{body}</pre></div>'
        )

    def show_text(title: str, text: str, max_chars: int = 1500, height: int = 260) -> None:
        display(HTML(f'<div style="margin:8px 0;">{text_card(title, text, max_chars, height)}</div>'))

    def show_panel(title: str, body: str) -> None:
        """Render text as a cell OUTPUT rather than printing it.

        print() goes to marimo's console channel: the editor tucks it into a small grey
        box and `marimo run` drops it altogether. Rendering makes it visible in both
        modes and in the exported HTML.
        """
        display(HTML(
            '<div style="border:1px solid #8b949e55;border-radius:8px;overflow:hidden;'
            'font-family:system-ui,sans-serif;margin:8px 0;">'
            '<div style="padding:6px 12px;font-size:12px;font-weight:600;opacity:0.75;'
            f'border-bottom:1px solid #8b949e55;">{_html.escape(title)}</div>'
            '<pre style="margin:0;padding:10px 12px;font-size:13px;line-height:1.55;'
            f'white-space:pre-wrap;overflow-x:auto;">{_html.escape(body)}</pre></div>'
        ))

    def metrics_body(name: str, text: str, timing_key: str) -> str:
        """Source (cache or live) plus the naive metrics, as panel text."""
        src = f"live run, {TIMINGS[timing_key]:.1f}s" if timing_key in TIMINGS else 'pre-computed cache'
        rows = [f'approach   {name}', f'source     {src}']
        rows += [f'{k:<10} {v}' for k, v in basic_metrics(text).items()]
        return chr(10).join(rows)

    return (
        HTML,
        TIMINGS,
        basic_metrics,
        display,
        metrics_body,
        show_panel,
        normalize_text,
        save_text,
        show_text,
        text_card,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) OCR A: Docling OCR (RapidOCR / optional OcrMacOptions)

    Classical OCR works like a very fast letter-shape detective: it scans the page image for familiar character *shapes* and assembles them into text. This is surprisingly fast and runs entirely locally — but it reaches its limits as soon as something unusual appears: umlauts, mathematical symbols, or table lines are easily confused or swallowed (the German city name "München" quickly turns into "Minchen").

    <img src="/public/img/w2_04_ocr_vs_vlm.svg" alt="Classic OCR matches letter shapes one by one and misreads München as Minchen; a VLM reads the whole page and preserves heading, table and formula" style="max-width:100%; width:860px">

    Strengths: consistent within the Docling workflow and flexible depending on the workshop hardware.
    Recommendation: `RapidOcrOptions` as the default, `OcrMacOptions` optionally for macOS (often faster).
    """)
    return


@app.cell(hide_code=True)
def _(
    basic_metrics,
    json,
    metrics_body,
    OUT_DIR,
    Path,
    PDF_PATH,
    RERUN_OCR,
    save_text,
    show_panel,
    show_text,
    time,
    TIMINGS: dict[str, float],
):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        OcrMacOptions,
        PdfPipelineOptions,
        RapidOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    def ocr_with_docling(pdf_path: Path, engine: str='rapidocr', start_page: int=1) -> tuple[str, dict]:
        opts = PdfPipelineOptions()
        opts.do_ocr = True
        if engine == 'mac':
            opts.ocr_options = OcrMacOptions(lang=['en-US'], force_full_page_ocr=True)
        elif engine == 'rapidocr':
            opts.ocr_options = RapidOcrOptions(lang=['english'], force_full_page_ocr=True)
        else:
            raise ValueError("engine must be 'rapidocr' or 'mac'")
        opts.do_table_structure = True
        opts.do_formula_enrichment = True
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        result = converter.convert(str(pdf_path))
        doc = result.document
        md = doc.export_to_markdown()
        doc_json = doc.export_to_dict()
        return (md, doc_json)
    OCR_ENGINE = 'rapidocr'
    _cache_md = OUT_DIR / f'04_s03_docling_{OCR_ENGINE}_ocr.md'
    _cache_json = OUT_DIR / f'04_s03_docling_{OCR_ENGINE}_ocr.json'
    if not RERUN_OCR and _cache_md.exists():
        docling_ocr_text = _cache_md.read_text(encoding='utf-8')
        docling_ocr_json = json.loads(_cache_json.read_text(encoding='utf-8')) if _cache_json.exists() else None
        print(f'[cache] loaded {_cache_md.name} (set RERUN_OCR = True to run live)')
    else:
        try:
    # Options: 'rapidocr' | 'mac'
            _t0 = time.perf_counter()
            docling_ocr_text, docling_ocr_json = ocr_with_docling(PDF_PATH, engine=OCR_ENGINE)
            TIMINGS[f'docling_{OCR_ENGINE}'] = time.perf_counter() - _t0
            save_text(f'04_s03_docling_{OCR_ENGINE}_ocr', docling_ocr_text)
            _cache_json.write_text(json.dumps(docling_ocr_json, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Duration: {TIMINGS[f'docling_{OCR_ENGINE}']:.1f}s")
        except Exception as e:
            print(f'Live run failed ({type(e).__name__}): {e}')
            print('Set RERUN_OCR = False to use the pre-computed results.')
            if _cache_md.exists():
                docling_ocr_text = _cache_md.read_text(encoding='utf-8')
                docling_ocr_json = json.loads(_cache_json.read_text(encoding='utf-8')) if _cache_json.exists() else None
                print(f'[cache] falling back to {_cache_md.name}')
            else:
                raise
    show_panel(f'docling_{OCR_ENGINE}', metrics_body(f'docling_{OCR_ENGINE}', docling_ocr_text, f'docling_{OCR_ENGINE}'))
    show_text(f'Preview — docling_{OCR_ENGINE}', docling_ocr_text, max_chars=600, height=200)
    return (
        DocumentConverter,
        InputFormat,
        OCR_ENGINE,
        PdfFormatOption,
        PdfPipelineOptions,
        docling_ocr_text,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) OCR B: Docling VLM (Granite preset)

    Now for the second philosophy — the right-hand panel of the diagram in section 3: instead of hunting for letter shapes, a **local VLM** looks at the whole page as an image and "reads it out loud" — the way a person would describe a page if you asked them to. The charm: everything runs on **your** machine, no document ever leaves it (important for confidential data). The price: small local models look less closely than large cloud models — we are about to see where that goes wrong.

    We use the `granite_docling` preset for the VLM-based extraction. On Apple Silicon, MLX acceleration is enabled automatically when available. If no suitable local VLM setup is available, use the external VLM step (section 5) as a fallback.

    If the next cell fails, the local VLM is simply not available on your hardware — that is fine: the cached output loads automatically, and three other approaches remain.
    """)
    return


@app.cell(hide_code=True)
def _(
    basic_metrics,
    DocumentConverter,
    InputFormat,
    json,
    metrics_body,
    OUT_DIR,
    Path,
    PDF_PATH,
    PdfFormatOption,
    RERUN_OCR,
    save_text,
    show_panel,
    show_text,
    time,
    TIMINGS: dict[str, float],
    VLM_PRESET,
    VLM_USE_MLX,
):
    from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
    from docling.pipeline.vlm_pipeline import VlmPipeline

    def ocr_with_docling_vlm(pdf_path: Path, preset: str='granite_docling', use_mlx: bool=False) -> tuple[str, dict]:
        if use_mlx:
            from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
            vlm_options = VlmConvertOptions.from_preset(preset, engine_options=MlxVlmEngineOptions())
        else:
            vlm_options = VlmConvertOptions.from_preset(preset)
        vlm_pipe_opts = VlmPipelineOptions(vlm_options=vlm_options)
        vlm_pipe_opts.force_backend_text = False
        vlm_pipe_opts.images_scale = 1.0
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=vlm_pipe_opts)})
        result = converter.convert(str(pdf_path))
        doc = result.document
        md = doc.export_to_markdown()
        doc_json = doc.export_to_dict()
        return (md, doc_json)
    docling_vlm_text = None
    docling_vlm_json = None
    _cache_md = OUT_DIR / '04_s04_docling_vlm_granite_ocr.md'
    _cache_json = OUT_DIR / '04_s04_docling_vlm_granite_ocr.json'
    if not RERUN_OCR and _cache_md.exists():
        docling_vlm_text = _cache_md.read_text(encoding='utf-8')
        docling_vlm_json = json.loads(_cache_json.read_text(encoding='utf-8')) if _cache_json.exists() else None
        print(f'[cache] loaded {_cache_md.name} (set RERUN_OCR = True to run live)')
    else:
        try:
            _t0 = time.perf_counter()
            docling_vlm_text, docling_vlm_json = ocr_with_docling_vlm(PDF_PATH, preset=VLM_PRESET, use_mlx=VLM_USE_MLX)
            TIMINGS['docling_vlm'] = time.perf_counter() - _t0
            save_text('04_s04_docling_vlm_granite_ocr', docling_vlm_text)
            _cache_json.write_text(json.dumps(docling_vlm_json, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Duration: {TIMINGS['docling_vlm']:.1f}s")
        except Exception as e:
            print(f'Live run failed ({type(e).__name__}): {e}')
            print('The local VLM may simply not be available on this hardware.')
            if _cache_md.exists():
                docling_vlm_text = _cache_md.read_text(encoding='utf-8')
                docling_vlm_json = json.loads(_cache_json.read_text(encoding='utf-8')) if _cache_json.exists() else None
                print(f'[cache] falling back to {_cache_md.name}')
    if docling_vlm_text:
        show_panel('docling_vlm (granite)', metrics_body('docling_vlm', docling_vlm_text, 'docling_vlm'))
        show_text('Preview — docling_vlm (granite)', docling_vlm_text, max_chars=600, height=200)
    return VlmPipeline, VlmPipelineOptions, docling_vlm_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) OCR C: External VLM (LiteLLM) on page images

    Third route: we take a snapshot of every PDF page and send the images to a **large VLM in the cloud**. This usually delivers the best quality — large models recognize umlauts, LaTeX formulas, and tables far more reliably. The flip side: every page costs one API call (time + money), and the documents leave your machine — for confidential material that is a real data-privacy trade-off!

    We compare two external variants:
    1. Direct LiteLLM call per page (image → text) — maximally simple and transparent
    2. Docling `ApiVlmOptions` with the LiteLLM endpoint — same API, but Docling orchestrates and additionally delivers structured JSON output

    Workshop default: `qwen3-vl-32b` on the LiteLLM endpoint.

    One caveat up front: our prompt asks for "plain markdown text only", but models do not always comply — `qwen3-vl` likes to return tables as raw LaTeX. That is not a bug in your code; section 8 shows the consequences.
    """)
    return


@app.cell(hide_code=True)
def _(
    API_BASE_URL,
    DOCLING_API_VLM_URL,
    DocumentConverter,
    InputFormat,
    os,
    Path,
    PdfFormatOption,
    show_panel,
    VlmPipeline,
    VlmPipelineOptions,
):

    from docling.datamodel.pipeline_options_vlm_model import (
        ApiVlmOptions,
        ResponseFormat,
    )
    from litellm import completion
    from pdf2image import convert_from_path

    from ragkit.embed import pil_to_base64_data_url

    def ocr_with_external_vlm(pdf_path: Path, model: str) -> str:
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY is missing for the external VLM.')
        images = convert_from_path(str(pdf_path))
        show_panel('Page images for the external VLM', f'{len(images)} page images extracted')
        outputs = []
        for i, img in enumerate(images, start=1):
            img_url = pil_to_base64_data_url(img)
            prompt = 'Extract the page text as faithfully as possible. Keep LaTeX/math and table structure if visible. Return plain markdown text only.'
            resp = completion(model=model, api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'), temperature=0.0, messages=[{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': img_url}}]}])
            page_text = resp.choices[0].message.content
            outputs.append(f'## Page {i}\n\n{page_text.strip()}')
        return '\n\n'.join(outputs)

    def ocr_with_docling_api_vlm(pdf_path: Path, model: str, annotate_pictures: bool=False) -> tuple[str, dict]:
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY is missing for Docling ApiVlmOptions.')
        auth_headers = {'Authorization': f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        api_opts = ApiVlmOptions(url=DOCLING_API_VLM_URL, headers=auth_headers, prompt='Extract page text faithfully. Preserve equations and tables. Return markdown.', response_format=ResponseFormat.MARKDOWN, params={'model': model, 'temperature': 0.0}, timeout=120, concurrency=2)
        pipe_opts = VlmPipelineOptions(enable_remote_services=True, vlm_options=api_opts)
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=pipe_opts)})
        result = converter.convert(str(pdf_path))
        doc = result.document
        md = doc.export_to_markdown()
        doc_json = doc.export_to_dict()
        return (md, doc_json)

    return (
        completion,
        ocr_with_docling_api_vlm,
        ocr_with_external_vlm,
        pil_to_base64_data_url,
    )


@app.cell(hide_code=True)
def _(
    basic_metrics,
    EXTERNAL_VLM_MODEL,
    metrics_body,
    ocr_with_external_vlm,
    OUT_DIR,
    PDF_PATH,
    RERUN_OCR,
    save_text,
    show_panel,
    show_text,
    time,
    TIMINGS: dict[str, float],
):
    # 1) direct LiteLLM call
    _cache_md = OUT_DIR / '04_s05_external_vlm_direct_ocr.md'
    if not RERUN_OCR and _cache_md.exists():
        external_vlm_text = _cache_md.read_text(encoding='utf-8')
        print(f'[cache] loaded {_cache_md.name} (set RERUN_OCR = True to run live)')
    else:
        try:
            _t0 = time.perf_counter()
            external_vlm_text = ocr_with_external_vlm(PDF_PATH, model=EXTERNAL_VLM_MODEL)
            TIMINGS['external_vlm'] = time.perf_counter() - _t0
            save_text('04_s05_external_vlm_direct_ocr', external_vlm_text)
            print(f"Duration: {TIMINGS['external_vlm']:.1f}s")
        except Exception as e:
            print(f'Live run failed ({type(e).__name__}): {e}')
            print('Set RERUN_OCR = False to use the pre-computed results.')
            if _cache_md.exists():
                external_vlm_text = _cache_md.read_text(encoding='utf-8')
                print(f'[cache] falling back to {_cache_md.name}')
            else:
                raise
    show_panel('external_vlm (direct LiteLLM call)', metrics_body('external_vlm', external_vlm_text, 'external_vlm'))
    show_text('Preview — external_vlm (direct)', external_vlm_text, max_chars=600, height=200)
    return (external_vlm_text,)


@app.cell(hide_code=True)
def _(
    basic_metrics,
    DOCLING_API_VLM_MODEL,
    json,
    metrics_body,
    ocr_with_docling_api_vlm,
    OUT_DIR,
    PDF_PATH,
    RERUN_OCR,
    save_text,
    show_panel,
    show_text,
    time,
    TIMINGS: dict[str, float],
):
    # 2) Docling ApiVlmOptions (incl. JSON output)
    _cache_md = OUT_DIR / '04_s05_docling_api_vlm_ocr.md'
    _cache_json = OUT_DIR / '04_s05_docling_api_vlm_ocr.json'
    if not RERUN_OCR and _cache_md.exists():
        docling_api_vlm_text = _cache_md.read_text(encoding='utf-8')
        docling_api_vlm_json = json.loads(_cache_json.read_text(encoding='utf-8')) if _cache_json.exists() else None
        print(f'[cache] loaded {_cache_md.name} (set RERUN_OCR = True to run live)')
    else:
        try:
            _t0 = time.perf_counter()
            docling_api_vlm_text, docling_api_vlm_json = ocr_with_docling_api_vlm(PDF_PATH, model=DOCLING_API_VLM_MODEL)
            TIMINGS['docling_api_vlm'] = time.perf_counter() - _t0
            save_text('04_s05_docling_api_vlm_ocr', docling_api_vlm_text)
            _cache_json.write_text(json.dumps(docling_api_vlm_json, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Duration: {TIMINGS['docling_api_vlm']:.1f}s")
        except Exception as e:
            print(f'Live run failed ({type(e).__name__}): {e}')
            print('Set RERUN_OCR = False to use the pre-computed results.')
            if _cache_md.exists():
                docling_api_vlm_text = _cache_md.read_text(encoding='utf-8')
                docling_api_vlm_json = json.loads(_cache_json.read_text(encoding='utf-8')) if _cache_json.exists() else None
                print(f'[cache] falling back to {_cache_md.name}')
            else:
                raise
    show_panel('docling_api_vlm (ApiVlmOptions)', metrics_body('docling_api_vlm', docling_api_vlm_text, 'docling_api_vlm'))
    show_text('Preview — docling_api_vlm', docling_api_vlm_text, max_chars=600, height=200)
    return docling_api_vlm_json, docling_api_vlm_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) Bonus: making images searchable (VLM image annotation)

    This section solves a **different problem** than the OCR text-quality comparison — feel free to skip it on a first pass and come back later.

    A blind spot of all approaches so far: **figures.** To text search, images are invisible — a question about "the architecture diagram" finds nothing if the diagram is never described in words anywhere. The solution: we have a VLM describe every image and insert the description as text into the document. That makes the image content searchable.

    <img src="/public/img/w2_04_image_blindspot.svg" alt="Text extraction leaves an invisible gap where a figure was; a VLM caption fills the gap with searchable text" style="max-width:100%; width:860px">

    For this we do **not** use the VLM picture description inside the VlmPipeline, but the more robust route:
    1. Extract images with the standard PDF pipeline (`generate_picture_images=True`)
    2. Describe each image externally with `qwen3-vl-32b`
    3. Insert the descriptions, in order, into a copy of the OCR markdown

    Naming convention: `04_sXX_<pipeline>_<artifact>.<ext>` for quick orientation during the workshop.
    """)
    return


@app.cell(hide_code=True)
def _(
    API_BASE_URL,
    DocumentConverter,
    InputFormat,
    Path,
    PdfFormatOption,
    PdfPipelineOptions,
    completion,
    os,
    pil_to_base64_data_url,
):
    from docling_core.types.doc import PictureItem

    def extract_docling_pictures(pdf_path: Path):
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.generate_picture_images = True
        opts.images_scale = 1.0
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        result = converter.convert(str(pdf_path))
        doc = result.document
        pictures = []
        idx = 0
        for item, _level in doc.iterate_items():
            if isinstance(item, PictureItem) and item.image is not None and (item.image.pil_image is not None):
                idx += 1
                pages = sorted({prov.page_no for prov in item.prov or [] if hasattr(prov, 'page_no')})
                pictures.append({'picture_index': idx, 'page_numbers': pages, 'pil_image': item.image.pil_image})
        return pictures

    def describe_picture_with_external_vlm(pil_img, model: str) -> str:
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY is missing for image annotation.')
        prompt = 'Describe this figure. If chart/diagram, explain structure and key takeaway. If formula image, transcribe math where possible. Return concise markdown.'
        resp = completion(model=model, api_base=API_BASE_URL, api_key=os.getenv('OPENAI_API_KEY'), temperature=0.0, messages=[{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': pil_to_base64_data_url(pil_img)}}]}])
        return (resp.choices[0].message.content or '').strip()

    def merge_image_descriptions_into_markdown(base_markdown: str, descriptions: list[dict]) -> str:
        parts = base_markdown.split('<!-- image -->')
        if len(parts) <= 1:
            return base_markdown
        merged = [parts[0]]
        for i in range(1, len(parts)):
            desc = descriptions[i - 1]['description'] if i - 1 < len(descriptions) else None
            page_txt = descriptions[i - 1].get('page_numbers', []) if i - 1 < len(descriptions) else []
            page_str = ', '.join(map(str, page_txt)) if page_txt else '-'
            if desc:
                merged.append(f'<!-- image -->\n\n**VLM image description (pages: {page_str})**\n\n{desc}\n')
            else:
                merged.append('<!-- image -->\n')
            merged.append(parts[i])
        return ''.join(merged)

    return (
        describe_picture_with_external_vlm,
        extract_docling_pictures,
        merge_image_descriptions_into_markdown,
    )


@app.cell(hide_code=True)
def _(
    describe_picture_with_external_vlm,
    docling_api_vlm_text,
    EXTERNAL_VLM_MODEL,
    extract_docling_pictures,
    json,
    merge_image_descriptions_into_markdown,
    OUT_DIR,
    PDF_PATH,
    RERUN_OCR,
    show_panel,
):
    # Base for the merge: OCR output from step 5
    base_md_for_merge = docling_api_vlm_text
    merge_out = OUT_DIR / '04_s06_docling_api_with_vlm_image_desc.md'
    desc_out = OUT_DIR / '04_s06_external_vlm_image_descriptions.json'
    if not RERUN_OCR and desc_out.exists():
        picture_descriptions = json.loads(desc_out.read_text(encoding='utf-8'))
        merged_md = merge_out.read_text(encoding='utf-8')
        _source = f'pre-computed cache ({desc_out.name})'
    else:
        pictures = extract_docling_pictures(PDF_PATH)
        _source = f'live run, {len(pictures)} picture items extracted'
        picture_descriptions = []
        failed = []
        for _pic in pictures:
            try:
                desc = describe_picture_with_external_vlm(_pic['pil_image'], model=EXTERNAL_VLM_MODEL)
            except Exception as exc:
                desc = f'(external VLM unavailable: {exc.__class__.__name__})'
                failed.append(_pic['picture_index'])
            picture_descriptions.append({'picture_index': _pic['picture_index'], 'page_numbers': _pic['page_numbers'], 'description': desc})
        merged_md = merge_image_descriptions_into_markdown(base_md_for_merge, picture_descriptions)
        # Never let a failed call overwrite good pre-computed descriptions: a placeholder
        # string is not a result. Fall back to the cache if there is one (issue #23).
        if failed and desc_out.exists():
            picture_descriptions = json.loads(desc_out.read_text(encoding='utf-8'))
            merged_md = merge_out.read_text(encoding='utf-8')
            _source = f'live run failed for pictures {failed}, kept the pre-computed descriptions'
        else:
            if failed:
                _source = f'live run failed for pictures {failed}, no cache to fall back to'
            merge_out.write_text(merged_md, encoding='utf-8')
            desc_out.write_text(json.dumps(picture_descriptions, ensure_ascii=False, indent=2), encoding='utf-8')
            _source += f'; saved {merge_out.name} and {desc_out.name}'
    show_panel('VLM image descriptions', f'source     {_source}\npictures   {len(picture_descriptions)}')
    for picture in picture_descriptions:
        show_panel(
            f"Picture {picture['picture_index']}  (pages {', '.join(map(str, picture['page_numbers']))})",
            picture['description'],
        )
    return (picture_descriptions,)


@app.cell(hide_code=True)
def _(
    docling_api_vlm_json,
    json,
    OUT_DIR,
    picture_descriptions,
    show_panel,
):
    enriched_json = dict(docling_api_vlm_json)  # copy
    enriched_json['external_vlm_picture_descriptions'] = picture_descriptions
    # 1) keep all annotations in one clear place
    pics = enriched_json.get('pictures', [])
    if isinstance(pics, list):
    # 2) optional: attach by picture index into picture items (if present)
        desc_by_idx = {d['picture_index']: d for d in picture_descriptions}
        for i, _pic in enumerate(pics, start=1):
            if i in desc_by_idx and isinstance(_pic, dict):
                _pic['vlm_description'] = desc_by_idx[i]['description']
                _pic['vlm_description_pages'] = desc_by_idx[i].get('page_numbers', [])
    json_out = OUT_DIR / '04_s06_docling_api_with_vlm_image_desc.json'
    json_out.write_text(json.dumps(enriched_json, ensure_ascii=False, indent=2), encoding='utf-8')
    # save
    show_panel('Enriched Docling JSON', f'{json_out.name}\nimage descriptions attached to {len(picture_descriptions)} picture items')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7) Comparison: metrics, runtime, and agreement

    Which approach is worth it when? Just like buying a car, there is no best answer without context — it depends on whether speed, price, or quality matters. We therefore compare the four outputs along the central trade-off **speed/cost vs. quality**: local OCR (RapidOCR) is fast and free, but loses special characters and formulas. External VLMs deliver the cleanest structure (LaTeX, tables), but cost API calls and waiting time per page.

    The table shows the simple metrics plus the **wall-clock runtime** of each approach.

    Caution: the times are orders of magnitude, not benchmarks — the first Docling run includes model downloads/warmup, and API latency depends on gateway load. For cached loads (`RERUN_OCR = False`) the `time_s` column stays empty (NaN): timings exist only for fresh runs.
    """)
    return


@app.cell(hide_code=True)
def _(
    OCR_ENGINE,
    TIMINGS: dict[str, float],
    basic_metrics,
    docling_api_vlm_text,
    docling_ocr_text,
    docling_vlm_text,
    external_vlm_text,
):
    import pandas as pd

    results = {}
    results[f'docling_{OCR_ENGINE}'] = docling_ocr_text
    if docling_vlm_text:
        results['docling_vlm'] = docling_vlm_text
    results['external_vlm'] = external_vlm_text
    if docling_api_vlm_text:
        results['docling_api_vlm'] = docling_api_vlm_text

    rows = []
    for name, text in results.items():
        m = basic_metrics(text)
        m['time_s'] = round(TIMINGS.get(name, float('nan')), 1)
        rows.append({'approach': name, **m})

    df_comparison = pd.DataFrame(rows).set_index('approach')
    df_comparison
    return pd, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The metrics do not tell us, however, whether the approaches recognized **the same text** — two outputs can be equally long and still completely different. For that we compute a pairwise **similarity matrix** (Levenshtein-based, via `rapidfuzz`): 100 = identical, lower = more deviation. How to read it: a score of 85 means roughly 85 % of the characters match once insertions, deletions, and substitutions are accounted for.

    Typical result: the two external VLM variants agree to more than 95% (same model, same images — reassuring!), while the local VLM pipeline deviates the most from all the others.
    """)
    return


@app.cell(hide_code=True)
def _(normalize_text, pd, results):
    from rapidfuzz import fuzz

    names = list(results.keys())
    # fix_umlauts=False: an engine that emits /C231 instead of ü must lose points here, not be repaired
    sim = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            sim.loc[a, b] = round(fuzz.ratio(normalize_text(results[a], fix_umlauts=False), normalize_text(results[b], fix_umlauts=False)), 1)

    sim.style.background_gradient(cmap='RdYlGn', vmin=70, vmax=100).format('{:.1f}')
    return (fuzz,)


@app.cell(hide_code=True)
def _(HTML, display, results, text_card):
    # The first ~2,000 characters of every output, side by side — scroll inside each card
    cards = ''.join(text_card(name, text, max_chars=2000, height=320) for name, text in results.items())
    display(HTML(
        '<div style="display:grid;grid-template-columns:repeat(2, minmax(0,1fr));'
        'gap:12px;margin:8px 0;">' + cards + '</div>'
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8) Tables & formulas in detail

    Metrics can deceive: `docling_api_vlm` contains **not a single Markdown table** — there, the tables sit in the output as raw LaTeX (`\begin{tabular}{|l|l|...}` with `&` as the column separator). Our pipe count (`table_markers`) misses this completely: it does count the LaTeX column definitions, but overlooks the actual table rows. For this approach, the value in the comparison table therefore says nothing about how well the tables were really extracted. If you only look at the numbers, you draw the wrong conclusions — which is why we now take a look ourselves.

    <img src="/public/img/w2_04_latex_tables.svg" alt="The same table as a Markdown table full of pipes and as LaTeX tabular with ampersand separators; a naive pipe count only detects the former" style="max-width:100%; width:860px"> We inspect the same passage in all four outputs directly side by side — the way you would compare two translations of the same sentence.

    One problem with that: every OCR mangles the text differently ("München" becomes "Minchen" in one output and "Munchen" in another). An exact text search would therefore find nothing. So we search for the passage **fuzzily** with `rapidfuzz.fuzz.partial_ratio`: we provide a reference snippet and find, in each output, the line that is most similar to it.

    Each spot below renders as a small table: one row per approach, showing the passage it produced. Everything that **deviates from the reference is highlighted in red** — so a clean row means a faithful output, and lots of red means heavy mangling. (The fuzzy search score only says how confidently we *located* the passage, not how good the output is — judge quality by the highlights.)

    Three revealing spots:
    1. **Umlauts** — the umlaut-heavy authors' address line ("Fakultät für Informatik … München")
    2. **Formulas** — the complexity statement `O(1)` in the abstract (does the LaTeX survive?)
    3. **Tables** — the first data row of Table 1 (`RTRL | 3 | ≈ 170 | …`)
    """)
    return


@app.cell(hide_code=True)
def _(HTML, display, fuzz, normalize_text, results):
    import html
    from difflib import SequenceMatcher

    def find_snippet(text: str, query: str, context: int=60) -> tuple[float, str]:
        """Find the line most similar to the reference snippet and cut out the
        matching passage (± context characters)."""
        min_len = max(10, len(query) // 2)
        lines = [l.strip() for l in normalize_text(text, fix_umlauts=False).splitlines() if len(l.strip()) >= min_len]
        best_score, best_line = (-1.0, '')  # drop mini-lines like '1'
        for line in lines:
            score = fuzz.partial_ratio(query.lower(), line.lower())
            if score > best_score:
                best_score, best_line = (score, line)
        aln = fuzz.partial_ratio_alignment(query.lower(), best_line.lower())
        start = max(0, aln.dest_start - context)
        end = min(len(best_line), aln.dest_end + context)
        prefix = '… ' if start > 0 else ''
        suffix = ' …' if end < len(best_line) else ''
        return (best_score, f'{prefix}{best_line[start:end]}{suffix}')

    def _highlight_deviations(reference: str, span: str) -> str:
        """Render `span` as HTML; every part that deviates from `reference` is marked red."""
        sm = SequenceMatcher(None, reference.lower(), span.lower(), autojunk=False)
        out = []
        for op, _i1, _i2, j1, j2 in sm.get_opcodes():
            piece = html.escape(span[j1:j2])
            if op != 'equal' and piece.strip():
                out.append(f'<mark style="background:#ffebe9;color:#cf222e;font-weight:600;padding:0 1px;border-radius:2px;">{piece}</mark>')
            else:
                out.append(piece)
        return ''.join(out)

    def side_by_side(query: str, title: str, context: int=25) -> None:
        """One table per test spot: the matched passage of every approach,
        with deviations from the reference highlighted in red."""
        td = 'padding:6px 10px;vertical-align:top;border-top:1px solid #8b949e55;'
        mono = 'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;'
        rows = []
        for name, text in results.items():
            min_len = max(10, len(query) // 2)
            lines = [l.strip() for l in normalize_text(text, fix_umlauts=False).splitlines() if len(l.strip()) >= min_len]
            score, line = max(((fuzz.partial_ratio(query.lower(), l.lower()), l) for l in lines), key=lambda t: t[0])
            aln = fuzz.partial_ratio_alignment(query.lower(), line.lower())
            before = html.escape(line[max(0, aln.dest_start - context):aln.dest_start])
            span = _highlight_deviations(query, line[aln.dest_start:aln.dest_end])
            after = html.escape(line[aln.dest_end:aln.dest_end + context])
            dim = 'opacity:0.55;'
            rows.append(f'<tr><td style="{td}white-space:nowrap;">{name}</td><td style="{td}{mono}"><span style="{dim}">…{before}</span>{span}<span style="{dim}">{after}…</span></td></tr>')
        display(HTML(f'''<div style="margin:10px 0 26px;font-family:system-ui,sans-serif;"><div style="font-size:15px;font-weight:600;margin-bottom:2px;">{html.escape(title)}</div><div style="font-size:12.5px;opacity:0.7;margin-bottom:6px;">reference: <code style="{mono}">{html.escape(query)}</code> &nbsp;·&nbsp; <mark style="background:#ffebe9;color:#cf222e;padding:0 3px;border-radius:2px;">red</mark> = deviates from the reference</div><table style="border-collapse:collapse;">{''.join(rows)}</table></div>'''))
    side_by_side('Fakultät für Informatik Technische Universität München', 'Umlauts in the address line')
    side_by_side('its computational complexity per time step and weight is O(1)', 'Formula in the abstract')
    side_by_side('RTRL 3 170 0.05 some fraction 173,000', 'Table 1, first data row')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What you should see:**
    - **Umlauts:** RapidOCR turns "für/München" into "fir/Minchen"; the local Granite VLM loses *all* umlauts ("Fakultt", "Munchen", "Jirger"). Only the external VLM variants reproduce the line without errors.
    - **Formulas:** RapidOCR and Granite output `O(1)` as plain text, the external VLMs as LaTeX `$O(1)$` — important if formulas are later meant to be rendered or searched for specifically.
    - **Tables:** Three approaches deliver a Markdown table (RapidOCR with broken quotation marks, Granite loses the "≈"), while `docling_api_vlm` returns raw LaTeX with HTML escapes (`&amp;`) — unusable for RAG chunking without post-processing.

    **Takeaway: high metric values ≠ good quality.** For German-language documents and formula/table preservation, the external VLM pays off; for fast full-text search across many documents, local OCR is often enough.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **By the way — this trick is useful beyond OCR.** `find_snippet` with `rapidfuzz.partial_ratio` is a general-purpose tool for messy text: deduplicating near-identical records, approximate search, or matching user queries against imperfectly extracted text. Worth keeping in your toolbox.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9) Conclusion

    What we have seen:
    - **Docling OCR (RapidOCR/OcrMacOptions):** the fastest, free local baseline — but it loses umlauts and formulas
    - **Docling VLM (Granite, local):** good structure and full data control (nothing leaves your machine) — but in our test the worst character errors (all umlauts lost)
    - **External VLM (direct via LiteLLM):** best text and formula fidelity, flexible model choice — costs API calls and time per page
    - **External VLM via Docling ApiVlmOptions:** same recognition quality plus structured Docling JSON output — but tables may come back as raw LaTeX

    Rule of thumb: **volume → local OCR, quality → external VLM.** For RAG on technical or German-language documents, the VLM quality usually pays off, because OCR errors propagate through the entire pipeline: misrecognized words become worse embeddings, which become worse retrieval hits, which become worse answers.

    **The middle ground (what production pipelines actually do):** most real workloads are neither pure volume nor pure quality. A common hybrid: run the fast local OCR over everything first, score each page (confidence values, character statistics, or checks like our umlaut count), and re-process only the low-confidence or business-critical pages with the external VLM. That buys VLM quality where it matters, at a fraction of the cost.

    The logical next step is ingesting the best OCR output into Qdrant, which closes the loop back to notebook 03.
    """)
    return


if __name__ == "__main__":
    app.run()
