# ruff: noqa: PLR1711,B018,BLE001  marimo returns; BLE001 for the harness' deliberate catch-all
import marimo

__generated_with = "0.24.0"
app = marimo.App(css_file="ragkit/theme.css", html_head_file="ragkit/head.html")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    # Rendering primitives, defined before the title so every section below can use them.
    # A cell that only defines functions renders nothing, so this stays invisible.
    import html

    from IPython.display import HTML, display

    from ragkit.chunk import normalize_text

    def text_card(title: str, text: str, max_chars: int = 1500, height: int = 260) -> str:
        """One scrollable preview card as an HTML string."""
        body = html.escape(normalize_text(text, fix_umlauts=False)[:max_chars])
        return (
            f'<div style="border:1px solid #8b949e55;border-radius:8px;overflow:hidden;'
            f'font-family:system-ui,sans-serif;">'
            f'<div style="padding:6px 12px;font-size:12px;font-weight:600;opacity:0.75;'
            f'border-bottom:1px solid #8b949e55;">{html.escape(title)}</div>'
            f'<pre style="margin:0;padding:10px 12px;max-height:{height}px;overflow:auto;'
            f'font-size:12.5px;line-height:1.45;white-space:pre-wrap;">{body}</pre></div>'
        )

    def show_text(title: str, text: str, max_chars: int = 1500, height: int = 260) -> None:
        display(
            HTML(f'<div style="margin:8px 0;">{text_card(title, text, max_chars, height)}</div>')
        )

    def show_panel(title: str, body: str) -> None:
        """Render text as a cell OUTPUT rather than printing it.

        print() goes to marimo's console channel: the editor tucks it into a small grey
        box and `marimo run` drops it altogether. Rendering makes it visible in both
        modes and in the exported HTML.
        """
        display(
            HTML(
                '<div style="border:1px solid #8b949e55;border-radius:8px;overflow:hidden;'
                'font-family:system-ui,sans-serif;margin:8px 0;">'
                '<div style="padding:6px 12px;font-size:12px;font-weight:600;opacity:0.75;'
                f'border-bottom:1px solid #8b949e55;">{html.escape(title)}</div>'
                '<pre style="margin:0;padding:10px 12px;font-size:13px;line-height:1.55;'
                f'white-space:pre-wrap;overflow-x:auto;">{html.escape(body)}</pre></div>'
            )
        )

    def show_markdown(title: str, md_text: str) -> None:
        """Render Markdown as formatted output. The VLM returns Markdown, so a <pre>
        panel would show raw ** and ### instead of headings and bold."""
        display(mo.md(f"**{title}**" + chr(10) * 2 + md_text))

    return (
        HTML,
        display,
        html,
        normalize_text,
        show_markdown,
        show_panel,
        show_text,
        text_card,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 - OCR Comparison: Docling OCR vs. Docling VLM vs. External VLM

    To a computer many PDFs are just **photos of text**: pixels, no searchable text. **OCR** turns those pixels back into text, which makes it the first step of almost every RAG pipeline. Do it sloppily and no chunking or embedding strategy can repair the damage later.

    Two competing philosophies:
    - **Classical OCR** recognises letter *shapes* one at a time. Fast, free, local.
    - A **VLM** looks at the whole page the way a person does and writes down what it sees, tables and formulas included. Better, but slower, and for external models it costs API calls.

    The trade-off is the theme of this notebook: **speed and cost against structural and character fidelity.**

    <img src="/public/img/w2_04_overview.svg" alt="One scanned PDF branches into four OCR approaches that converge into a comparison; an inset shows the speed-versus-fidelity trade-off" style="max-width:100%; width:860px">

    Four approaches on one hard PDF with tables and formulas, `notebooks/raw_data/lstm_tables.pdf` (6 pages from the LSTM paper by Hochreiter & Schmidhuber):

    1. **Docling + RapidOCR**, classical local OCR
    2. **Docling VLM pipeline**, local model, Granite preset
    3. **External VLM** via LiteLLM, on page images
    4. **Docling ApiVlmOptions**, same API, structured Docling output

    Then we compare them: metrics, runtime, similarity matrix, and a close look at tables and formulas.

    **Workshop note:** all four outputs ship pre-computed in `processed/ocr_compare/` and load instantly. Set `RERUN_OCR = True` in section 1 to run them live. The runtime column in section 7 only fills for live runs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Configuration

    Every tuning knob in one place: paths, API access, the OCR engine for approach A, and the model names for the three VLM variants. On Apple Silicon the notebook detects MLX acceleration for the local VLM automatically.

    **Before setting `RERUN_OCR = True`,** know what a live run actually needs. The panel below reports each one rather than assuming it:

    - approach A needs nothing beyond the installed packages
    - approach B needs a local VLM, which small or non-Apple machines may not manage
    - approach C rasterises the pages with **poppler** (`pdftoppm`), a system binary that is not installed by `uv sync`
    - approaches C and D both need a working API key and gateway budget, roughly one call per page
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import os
    import platform
    import shutil
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

    RAW_DIR = WORKSHOP_DIR / "raw_data"
    OUT_DIR = WORKSHOP_DIR / "processed" / "ocr_compare"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH = RAW_DIR / "lstm_tables.pdf"

    # This notebook compares OCR engines and never opens a collection, so a stopped Qdrant
    # is not a problem here; asking for it would report a failure nobody can act on.
    env = setup(require_qdrant=False, required_files=(PDF_PATH,), strict=False)
    theme.apply_mpl()

    API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.aisc.hpi.de/")

    # Adjust model names to your LiteLLM deployment if needed
    EXTERNAL_VLM_MODEL = os.getenv("EXTERNAL_VLM_MODEL", "openai/gemma-4-31b")
    DOCLING_API_VLM_MODEL = os.getenv("DOCLING_API_VLM_MODEL", "gemma-4-31b")
    DOCLING_API_VLM_URL = os.getenv(
        "DOCLING_API_VLM_URL", API_BASE_URL.rstrip("/") + "/v1/chat/completions"
    )

    # Docling VLM configuration
    VLM_PRESET = os.getenv("DOCLING_VLM_PRESET", "granite_docling")
    VLM_USE_MLX = auto_use_mlx()

    # Approach A's OCR engine: 'rapidocr' (cross-platform default) or 'mac' (Apple's Vision
    # framework, needs the ocrmac extra). Both have a pre-computed output on disk, so you
    # can switch this and see the difference without running anything.
    OCR_ENGINE = "rapidocr"

    # Approach C rasterises the PDF with poppler, a system binary uv does not install.
    POPPLER_FOUND = shutil.which("pdftoppm") is not None

    # Workshop pacing: by default the pre-computed OCR outputs in processed/ocr_compare/
    # load instantly. Set to True to run the four approaches live yourself.
    RERUN_OCR = False

    return (
        env,
        API_BASE_URL,
        DOCLING_API_VLM_MODEL,
        DOCLING_API_VLM_URL,
        EXTERNAL_VLM_MODEL,
        OCR_ENGINE,
        OUT_DIR,
        PDF_PATH,
        Path,
        POPPLER_FOUND,
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
    DOCLING_API_VLM_MODEL,
    DOCLING_API_VLM_URL,
    EXTERNAL_VLM_MODEL,
    OCR_ENGINE,
    OUT_DIR,
    PDF_PATH,
    POPPLER_FOUND,
    RERUN_OCR,
    show_panel,
    VLM_PRESET,
    VLM_USE_MLX,
):
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    # Everything the comparison was configured with, on screen. The API base, the key and
    # the data files are in the status table above, so they are not repeated here.
    _poppler = "found" if POPPLER_FOUND else "MISSING - approach C cannot run live"
    show_panel(
        "Configuration",
        "\n".join(
            [
                f"PDF                    {PDF_PATH}",
                f"PDF exists             {PDF_PATH.exists()}",
                f"OUT_DIR                {OUT_DIR}",
                f"OCR_ENGINE             {OCR_ENGINE}",
                f"EXTERNAL_VLM_MODEL     {EXTERNAL_VLM_MODEL}",
                f"DOCLING_API_VLM_MODEL  {DOCLING_API_VLM_MODEL}",
                f"DOCLING_API_VLM_URL    {DOCLING_API_VLM_URL}",
                f"Docling VLM preset     {VLM_PRESET}",
                f"Docling VLM uses MLX   {VLM_USE_MLX}",
                f"poppler (pdftoppm)     {_poppler}",
                f"RERUN_OCR              {RERUN_OCR}",
            ]
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) One harness for all four approaches

    A fair comparison needs the same ground rules everywhere: same text cleanup, same storage format, same metrics, same timing. So the plumbing lives here once, and each of the four sections below shows only what makes *that* approach different.

    `run_or_load(stem, name, run)` does the same six things every time: look for the pre-computed output, call `run()` if there is none or if `RERUN_OCR` is set, time it, save the markdown and any JSON, and render the metrics panel and preview.

    Every approach hands back an `OcrRun`: its text, its Docling JSON if it produced one, its runtime, and where the text came from. A live run that fails falls back to the cache. With no cache it returns **empty text instead of raising**, because in marimo an exception marks every dependent cell stale, so one unavailable approach would otherwise take sections 7 and 8 down with it.

    `basic_metrics` counts characters, words, lines, and math and table markers. A rough first impression only. These numbers are deliberately naive and **will mislead you**; section 8 shows exactly how. Distrust them from the start.
    """)
    return


@app.cell(hide_code=True)
def _(OUT_DIR, RERUN_OCR, json, normalize_text, show_panel, show_text):
    import time
    from typing import NamedTuple

    class OcrRun(NamedTuple):
        """What one approach produced. `seconds` is None when the text came from cache."""

        name: str
        text: str  # empty only when a live run failed and there was no cache
        json: dict | None  # None for approaches that return markdown only
        seconds: float | None
        source: str

    def save_text(stem: str, text: str) -> str:
        """Write the markdown and return exactly what was written.

        Returning it keeps a live run byte-identical to a later cache load. Without it the
        live path kept the raw text in memory while the cache path got the normalised one.
        """
        normalised = normalize_text(text, fix_umlauts=False)
        (OUT_DIR / f"{stem}.md").write_text(normalised, encoding="utf-8")
        return normalised

    # Deliberately naive metrics - they WILL mislead you. Section 8 shows exactly how.
    def basic_metrics(text: str) -> dict:
        t = normalize_text(text, fix_umlauts=False)
        return {
            "chars": len(t),
            "words": len(t.split()),
            "lines": len(t.splitlines()),
            "math_markers": sum(t.count(x) for x in ["$", "\\(", "\\)", "\\[", "\\]"]),
            "table_markers": sum(t.count(x) for x in ["|", "\t"]),
        }

    def metrics_body(run: OcrRun) -> str:
        """Where the text came from, plus the naive metrics, as panel text."""
        rows = [f"approach   {run.name}", f"source     {run.source}"]
        if run.text:
            rows += [f"{k:<10} {v}" for k, v in basic_metrics(run.text).items()]
        return chr(10).join(rows)

    def run_or_load(stem: str, name: str, run) -> OcrRun:
        """Load `stem` from the cache, or call `run()` and cache what it returns.

        `run` is a no-argument callable returning `(markdown, docling_json_or_None)`.
        That callable is the only thing the four approaches do differently.
        """
        md_path = OUT_DIR / f"{stem}.md"
        json_path = OUT_DIR / f"{stem}.json"

        def load(source: str) -> OcrRun:
            sidecar = (
                json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else None
            )
            return OcrRun(name, md_path.read_text(encoding="utf-8"), sidecar, None, source)

        if not RERUN_OCR and md_path.exists():
            result = load("pre-computed cache")
        else:
            try:
                t0 = time.perf_counter()
                text, sidecar = run()
                seconds = time.perf_counter() - t0
                text = save_text(stem, text)
                if sidecar is not None:
                    json_path.write_text(
                        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                result = OcrRun(name, text, sidecar, seconds, f"live run, {seconds:.1f}s")
            except Exception as exc:
                failed = f"live run failed ({type(exc).__name__}): {exc}"
                # Never re-raise: a raise here would mark sections 7 and 8 stale as well.
                result = (
                    load(f"{failed} - fell back to the cache")
                    if md_path.exists()
                    else OcrRun(name, "", None, None, failed)
                )

        show_panel(name, metrics_body(result))
        if result.text:
            show_text(f"Preview - {name}", result.text, max_chars=600, height=200)
        return result

    return basic_metrics, run_or_load


@app.cell(hide_code=True)
def _():
    # The Docling pieces sections 3 to 6 share. Each section below then adds only the
    # options that make its own approach different.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    return DocumentConverter, InputFormat, PdfFormatOption, PdfPipelineOptions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) OCR A: Docling OCR (RapidOCR / optional OcrMacOptions)

    Classical OCR is a fast letter-shape detective: it scans the page image for familiar character shapes and assembles them into text. Fast, and entirely local. It breaks as soon as anything unusual appears:

    - umlauts, where "München" becomes "Minchen"
    - mathematical symbols
    - table rules, easily confused or swallowed

    <img src="/public/img/w2_04_ocr_vs_vlm.svg" alt="Classic OCR matches letter shapes one by one and misreads München as Minchen; a VLM reads the whole page and preserves heading, table and formula" style="max-width:100%; width:860px">

    Consistent inside the Docling workflow and flexible about hardware. `RapidOcrOptions` is the default; `OcrMacOptions` is often faster on macOS. Switch `OCR_ENGINE` in section 1 to compare them: both outputs are already on disk.
    """)
    return


@app.cell(hide_code=True)
def _(
    DocumentConverter,
    InputFormat,
    OCR_ENGINE,
    Path,
    PDF_PATH,
    PdfFormatOption,
    PdfPipelineOptions,
    run_or_load,
):
    from docling.datamodel.pipeline_options import OcrMacOptions, RapidOcrOptions

    def ocr_with_docling(pdf_path: Path, engine: str) -> tuple[str, dict]:
        """The standard Docling PDF pipeline with an OCR engine in front of it."""
        opts = PdfPipelineOptions()
        opts.do_ocr = True
        if engine == "mac":
            opts.ocr_options = OcrMacOptions(lang=["en-US"], force_full_page_ocr=True)
        elif engine == "rapidocr":
            opts.ocr_options = RapidOcrOptions(lang=["english"], force_full_page_ocr=True)
        else:
            raise ValueError("engine must be 'rapidocr' or 'mac'")
        # force_full_page_ocr above: OCR the whole page even where the PDF has a text
        # layer, so every approach is judged on the same pixels.
        opts.do_table_structure = True
        opts.do_formula_enrichment = True
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        doc = converter.convert(str(pdf_path)).document
        return doc.export_to_markdown(), doc.export_to_dict()

    docling_ocr = run_or_load(
        f"04_s03_docling_{OCR_ENGINE}_ocr",
        f"docling_{OCR_ENGINE}",
        lambda: ocr_with_docling(PDF_PATH, engine=OCR_ENGINE),
    )
    return (docling_ocr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) OCR B: Docling VLM (Granite preset)

    The second philosophy, the right-hand panel of the diagram above. A **local VLM** looks at the whole page as an image and reads it out, the way a person would describe it.

    - **The appeal:** everything runs on your machine. No document leaves it, which matters for confidential data.
    - **The price:** small local models look less closely than large cloud ones. We are about to see where that goes wrong.

    Same Docling machinery as section 3, one swap: `VlmPipeline` replaces the OCR engine. MLX acceleration switches on automatically on Apple Silicon when available.

    If a live run fails here, the local VLM is simply unavailable on your hardware. That is fine: the cached output loads instead and three other approaches remain.
    """)
    return


@app.cell(hide_code=True)
def _(
    DocumentConverter,
    InputFormat,
    Path,
    PDF_PATH,
    PdfFormatOption,
    run_or_load,
    VLM_PRESET,
    VLM_USE_MLX,
):
    from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
    from docling.pipeline.vlm_pipeline import VlmPipeline

    def ocr_with_docling_vlm(pdf_path: Path, preset: str, use_mlx: bool) -> tuple[str, dict]:
        """Docling's own VLM pipeline: a local model reads the page images."""
        if use_mlx:
            from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions

            vlm_options = VlmConvertOptions.from_preset(
                preset, engine_options=MlxVlmEngineOptions()
            )
        else:
            vlm_options = VlmConvertOptions.from_preset(preset)
        vlm_pipe_opts = VlmPipelineOptions(vlm_options=vlm_options)
        # Trust the VLM's reading instead of merging the PDF's own text layer back in.
        vlm_pipe_opts.force_backend_text = False
        vlm_pipe_opts.images_scale = 1.0
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline, pipeline_options=vlm_pipe_opts
                )
            }
        )
        doc = converter.convert(str(pdf_path)).document
        return doc.export_to_markdown(), doc.export_to_dict()

    docling_vlm = run_or_load(
        "04_s04_docling_vlm_granite_ocr",
        "docling_vlm",
        lambda: ocr_with_docling_vlm(PDF_PATH, preset=VLM_PRESET, use_mlx=VLM_USE_MLX),
    )
    return VlmPipeline, VlmPipelineOptions, docling_vlm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) OCR C and D: External VLM (LiteLLM) on page images

    Third route: snapshot every PDF page and send the images to a **large VLM in the cloud**. Usually the best quality, because large models handle umlauts, LaTeX and tables far more reliably.

    The flip side: one API call per page in time and money, and the documents leave your machine. For confidential material that is a real privacy trade-off.

    Two external variants against the same endpoint and the same model:

    - **C, direct LiteLLM call** per page image. Maximally simple and transparent, markdown only.
    - **D, Docling `ApiVlmOptions`.** Docling orchestrates the calls and also returns structured JSON.

    Configured model: `gemma-4-31b`. The cached outputs below were produced with `qwen3-vl-32b`, which the gateway no longer serves.

    The two prompts are constants below rather than buried in the loop, because in a VLM pipeline **the prompt is part of the configuration** and is the first thing worth tuning. Note that both ask for markdown and neither reliably gets it: the cached `docling_api_vlm` output returns its tables as raw LaTeX. That is not a bug in your code; section 8 shows the consequences.
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

    EXTERNAL_VLM_PROMPT = (
        "Extract the page text as faithfully as possible. Keep LaTeX/math and table "
        "structure if visible. Return plain markdown text only."
    )
    DOCLING_API_VLM_PROMPT = (
        "Extract page text faithfully. Preserve equations and tables. Return markdown."
    )

    def ocr_with_external_vlm(pdf_path: Path, model: str) -> tuple[str, None]:
        """One VLM call per page image, concatenated. No Docling involved, so no JSON."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing for the external VLM.")
        images = convert_from_path(str(pdf_path))  # needs poppler; see section 1
        outputs = []
        for i, img in enumerate(images, start=1):
            resp = completion(
                model=model,
                api_base=API_BASE_URL,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTERNAL_VLM_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_base64_data_url(img)},
                            },
                        ],
                    }
                ],
            )
            outputs.append(f"## Page {i}\n\n{resp.choices[0].message.content.strip()}")
        return "\n\n".join(outputs), None

    def ocr_with_docling_api_vlm(pdf_path: Path, model: str) -> tuple[str, dict]:
        """The same pipeline as section 4, pointed at a remote model instead of a local one."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing for Docling ApiVlmOptions.")
        api_opts = ApiVlmOptions(
            url=DOCLING_API_VLM_URL,
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            prompt=DOCLING_API_VLM_PROMPT,
            response_format=ResponseFormat.MARKDOWN,
            params={"model": model, "temperature": 0.0},
            timeout=120,
            concurrency=2,
        )
        pipe_opts = VlmPipelineOptions(enable_remote_services=True, vlm_options=api_opts)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline, pipeline_options=pipe_opts
                )
            }
        )
        doc = converter.convert(str(pdf_path)).document
        return doc.export_to_markdown(), doc.export_to_dict()

    return (
        completion,
        ocr_with_docling_api_vlm,
        ocr_with_external_vlm,
        pil_to_base64_data_url,
    )


@app.cell(hide_code=True)
def _(EXTERNAL_VLM_MODEL, ocr_with_external_vlm, PDF_PATH, run_or_load):
    external_vlm = run_or_load(
        "04_s05_external_vlm_direct_ocr",
        "external_vlm",
        lambda: ocr_with_external_vlm(PDF_PATH, model=EXTERNAL_VLM_MODEL),
    )
    return (external_vlm,)


@app.cell(hide_code=True)
def _(DOCLING_API_VLM_MODEL, ocr_with_docling_api_vlm, PDF_PATH, run_or_load):
    docling_api_vlm = run_or_load(
        "04_s05_docling_api_vlm_ocr",
        "docling_api_vlm",
        lambda: ocr_with_docling_api_vlm(PDF_PATH, model=DOCLING_API_VLM_MODEL),
    )
    return (docling_api_vlm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) Bonus: making images searchable (VLM image annotation)

    A **different problem** from the OCR text-quality comparison. Skip it on a first pass and come back later if you like.

    Every approach so far shares one blind spot: **figures.** Images are invisible to text search, so a question about "the architecture diagram" finds nothing if the diagram is never described in words. So we have a VLM describe each image and insert that description into the document as text.

    <img src="/public/img/w2_04_image_blindspot.svg" alt="Text extraction leaves an invisible gap where a figure was; a VLM caption fills the gap with searchable text" style="max-width:100%; width:860px">

    Not via the VLM picture description inside the VlmPipeline, but the more robust route:

    1. Extract images with the standard PDF pipeline (`generate_picture_images=True`)
    2. Describe each image with the configured external VLM
    3. Insert the descriptions, in order, into a copy of approach D's markdown

    Naming convention: `04_sXX_<pipeline>_<artifact>.<ext>`.
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

    PICTURE_PROMPT = (
        "Describe this figure. If chart/diagram, explain structure and key takeaway. "
        "If formula image, transcribe math where possible. Return concise markdown."
    )

    def extract_docling_pictures(pdf_path: Path) -> list[dict]:
        """Pull the embedded images out of the PDF, with the page each one sits on."""
        opts = PdfPipelineOptions()
        opts.do_ocr = False  # we want the pictures, not the text
        opts.generate_picture_images = True
        opts.images_scale = 1.0
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        doc = converter.convert(str(pdf_path)).document
        pictures = []
        for item, _level in doc.iterate_items():
            if (
                isinstance(item, PictureItem)
                and item.image is not None
                and item.image.pil_image is not None
            ):
                pages = sorted(
                    {prov.page_no for prov in item.prov or [] if hasattr(prov, "page_no")}
                )
                pictures.append(
                    {
                        "picture_index": len(pictures) + 1,
                        "page_numbers": pages,
                        "pil_image": item.image.pil_image,
                    }
                )
        return pictures

    def describe_picture_with_external_vlm(pil_img, model: str) -> str:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing for image annotation.")
        resp = completion(
            model=model,
            api_base=API_BASE_URL,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PICTURE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_base64_data_url(pil_img)},
                        },
                    ],
                }
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def merge_image_descriptions_into_markdown(base_markdown: str, descriptions: list[dict]) -> str:
        """Docling leaves an `<!-- image -->` marker where every figure was. Splitting on
        it gives us exactly the slots to drop the descriptions into, in order."""
        parts = base_markdown.split("<!-- image -->")
        if len(parts) <= 1:
            return base_markdown
        merged = [parts[0]]
        for i in range(1, len(parts)):
            description = descriptions[i - 1] if i - 1 < len(descriptions) else {}
            desc = description.get("description")
            pages = description.get("page_numbers") or []
            page_str = ", ".join(map(str, pages)) if pages else "-"
            if desc:
                merged.append(
                    f"<!-- image -->\n\n**VLM image description (pages: {page_str})**\n\n{desc}\n"
                )
            else:
                merged.append("<!-- image -->\n")
            merged.append(parts[i])
        return "".join(merged)

    return (
        describe_picture_with_external_vlm,
        extract_docling_pictures,
        merge_image_descriptions_into_markdown,
    )


@app.cell(hide_code=True)
def _(
    describe_picture_with_external_vlm,
    docling_api_vlm,
    EXTERNAL_VLM_MODEL,
    extract_docling_pictures,
    json,
    merge_image_descriptions_into_markdown,
    OUT_DIR,
    PDF_PATH,
    RERUN_OCR,
    show_markdown,
    show_panel,
    show_text,
):
    merge_out = OUT_DIR / "04_s06_docling_api_with_vlm_image_desc.md"
    desc_out = OUT_DIR / "04_s06_external_vlm_image_descriptions.json"
    if not RERUN_OCR and desc_out.exists():
        picture_descriptions = json.loads(desc_out.read_text(encoding="utf-8"))
        merged_md = merge_out.read_text(encoding="utf-8")
        _source = f"pre-computed cache ({desc_out.name})"
    else:
        pictures = extract_docling_pictures(PDF_PATH)
        _source = f"live run, {len(pictures)} picture items extracted"
        picture_descriptions = []
        failed = []
        for _pic in pictures:
            try:
                desc = describe_picture_with_external_vlm(
                    _pic["pil_image"], model=EXTERNAL_VLM_MODEL
                )
            except Exception as exc:
                desc = f"(external VLM unavailable: {exc.__class__.__name__})"
                failed.append(_pic["picture_index"])
            picture_descriptions.append(
                {
                    "picture_index": _pic["picture_index"],
                    "page_numbers": _pic["page_numbers"],
                    "description": desc,
                }
            )
        # Base for the merge: approach D's markdown, which still carries the image markers.
        merged_md = merge_image_descriptions_into_markdown(
            docling_api_vlm.text, picture_descriptions
        )
        # Never let a failed call overwrite good pre-computed descriptions: a placeholder
        # string is not a result. Fall back to the cache if there is one (issue #23).
        if failed and desc_out.exists():
            picture_descriptions = json.loads(desc_out.read_text(encoding="utf-8"))
            merged_md = merge_out.read_text(encoding="utf-8")
            _source = f"live run failed for pictures {failed}, kept the pre-computed descriptions"
        else:
            if failed:
                _source = f"live run failed for pictures {failed}, no cache to fall back to"
            merge_out.write_text(merged_md, encoding="utf-8")
            desc_out.write_text(
                json.dumps(picture_descriptions, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _source += f"; saved {merge_out.name} and {desc_out.name}"

    show_panel(
        "VLM image descriptions", f"source     {_source}\npictures   {len(picture_descriptions)}"
    )
    for picture in picture_descriptions:
        show_markdown(
            f"Picture {picture['picture_index']}  (pages {', '.join(map(str, picture['page_numbers']))})",
            picture["description"],
        )

    # The merged document is the point of this section, so show it rather than only saving
    # it: a window around the first description, where the figure used to be a blank.
    _cut = max(0, merged_md.find("**VLM image description") - 400)
    show_text(
        f"{merge_out.name} - around the first inserted description",
        merged_md[_cut : _cut + 1800],
        max_chars=1800,
        height=300,
    )
    return (picture_descriptions,)


@app.cell(hide_code=True)
def _(
    docling_api_vlm,
    json,
    OUT_DIR,
    picture_descriptions,
    RERUN_OCR,
    show_panel,
):
    # Same descriptions, now inside the structured JSON: once in their own top-level key,
    # and once attached to the picture item they belong to.
    json_out = OUT_DIR / "04_s06_docling_api_with_vlm_image_desc.json"
    if docling_api_vlm.json is None:
        show_panel("Enriched Docling JSON", "skipped: approach D produced no JSON sidecar")
    else:
        enriched_json = dict(docling_api_vlm.json)
        enriched_json["external_vlm_picture_descriptions"] = picture_descriptions
        pics = enriched_json.get("pictures", [])
        if isinstance(pics, list):
            desc_by_idx = {d["picture_index"]: d for d in picture_descriptions}
            for i, _pic in enumerate(pics, start=1):
                if i in desc_by_idx and isinstance(_pic, dict):
                    _pic["vlm_description"] = desc_by_idx[i]["description"]
                    _pic["vlm_description_pages"] = desc_by_idx[i].get("page_numbers", [])
        # 2.2 MB of JSON: only rewrite it when there is a reason to.
        if RERUN_OCR or not json_out.exists():
            json_out.write_text(
                json.dumps(enriched_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _note = f"saved {json_out.name}"
        else:
            _note = f"{json_out.name} already on disk, left untouched"
        show_panel(
            "Enriched Docling JSON",
            f"{_note}\nimage descriptions attached to {len(picture_descriptions)} picture items",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7) Comparison: metrics, runtime, and agreement

    Which approach is worth it when? There is no best answer without context, only a trade-off between **speed and cost** on one side and **quality** on the other:

    - local OCR (RapidOCR) is fast and free, and loses special characters and formulas
    - external VLMs give the cleanest structure (LaTeX, tables), and cost an API call and a wait per page

    The table below holds the simple metrics plus the **wall-clock runtime** per approach.

    Treat the times as orders of magnitude, not benchmarks:
    - the first Docling run includes model download and warmup
    - API latency depends on gateway load
    - with `RERUN_OCR = False` the `time_s` column stays empty, because timings exist only for fresh runs
    """)
    return


@app.cell(hide_code=True)
def _(basic_metrics, docling_api_vlm, docling_ocr, docling_vlm, external_vlm):
    import pandas as pd

    # One filter for the whole comparison: an approach whose live run failed with no cache
    # has empty text, and drops out here rather than needing a guard in every cell below.
    ocr_runs = [r for r in (docling_ocr, docling_vlm, external_vlm, docling_api_vlm) if r.text]
    results = {r.name: r.text for r in ocr_runs}

    df_comparison = pd.DataFrame(
        [
            {
                "approach": r.name,
                **basic_metrics(r.text),
                "time_s": round(r.seconds, 1) if r.seconds is not None else float("nan"),
            }
            for r in ocr_runs
        ]
    ).set_index("approach")
    df_comparison
    return pd, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The metrics do not say whether the approaches recognised **the same text**. Two outputs can be equally long and completely different.

    So we compute a pairwise **similarity matrix** (Levenshtein, via `rapidfuzz`): 100 is identical, lower means more deviation. A score of 85 means roughly 85% of characters match once insertions, deletions and substitutions are counted.

    Typical result: the two external variants agree above 95% (same model, same images), while the local VLM pipeline deviates most from everything else.
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
            sim.loc[a, b] = round(
                fuzz.ratio(
                    normalize_text(results[a], fix_umlauts=False),
                    normalize_text(results[b], fix_umlauts=False),
                ),
                1,
            )

    sim.style.background_gradient(cmap="RdYlGn", vmin=70, vmax=100).format("{:.1f}")
    return (fuzz,)


@app.cell(hide_code=True)
def _(HTML, display, results, text_card):
    # The first ~2,000 characters of every output, side by side — scroll inside each card
    cards = "".join(
        text_card(name, text, max_chars=2000, height=320) for name, text in results.items()
    )
    display(
        HTML(
            '<div style="display:grid;grid-template-columns:repeat(2, minmax(0,1fr));'
            'gap:12px;margin:8px 0;">' + cards + "</div>"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8) Tables & formulas in detail

    Metrics deceive. `docling_api_vlm` contains **not a single Markdown table**: its tables arrive as raw LaTeX (`\begin{tabular}{|l|l|...}` with `&` separating columns). Our pipe count (`table_markers`) counts the LaTeX column definitions and misses the actual rows, so for that approach the number says nothing about how well tables were extracted. Read only the numbers and you draw the wrong conclusion.

    <img src="/public/img/w2_04_latex_tables.svg" alt="The same table as a Markdown table full of pipes and as LaTeX tabular with ampersand separators; a naive pipe count only detects the former" style="max-width:100%; width:860px">

    So we look ourselves, at the same passage in all four outputs side by side.

    One obstacle: every OCR mangles text differently. "München" becomes "Minchen" in one output and "Munchen" in another, so an exact search finds nothing. We search **fuzzily** with `rapidfuzz.fuzz.partial_ratio`: give it a reference snippet, and `find_snippet` returns the most similar line in each output together with where inside that line the match sits.

    How to read each spot, one row per approach:
    - **red** marks anything deviating from the reference
    - a clean row means a faithful output, a lot of red means heavy mangling
    - the **score** column says only how confidently we *located* the passage, not how good it is

    Three revealing spots:
    1. **Umlauts:** the authors' address line ("Fakultät für Informatik ... München")
    2. **Formulas:** the complexity statement `O(1)` in the abstract. Does the LaTeX survive?
    3. **Tables:** the first data row of Table 1 (`RTRL | 3 | ≈ 170 | ...`)
    """)
    return


@app.cell(hide_code=True)
def _(HTML, display, fuzz, html, normalize_text, results):
    from difflib import SequenceMatcher

    def find_snippet(text: str, query: str) -> tuple[float, str, int, int]:
        """Find the line most similar to `query`, and where inside it the match sits.

        Returns `(score, line, start, end)`. The caller decides how much context to show
        around `start:end` and what to do with the score.
        """
        min_len = max(10, len(query) // 2)  # drop mini-lines like '1'
        lines = [
            line.strip()
            for line in normalize_text(text, fix_umlauts=False).splitlines()
            if len(line.strip()) >= min_len
        ]
        best_score, best_line = -1.0, ""  # an output with no long-enough line at all
        for line in lines:
            score = fuzz.partial_ratio(query.lower(), line.lower())
            if score > best_score:
                best_score, best_line = score, line
        if not best_line:
            return 0.0, "(no comparable line in this output)", 0, 0
        aln = fuzz.partial_ratio_alignment(query.lower(), best_line.lower())
        return best_score, best_line, aln.dest_start, aln.dest_end

    def _highlight_deviations(reference: str, span: str) -> str:
        """Render `span` as HTML; every part that deviates from `reference` is marked red."""
        sm = SequenceMatcher(None, reference.lower(), span.lower(), autojunk=False)
        out = []
        for op, _i1, _i2, j1, j2 in sm.get_opcodes():
            piece = html.escape(span[j1:j2])
            if op != "equal" and piece.strip():
                out.append(
                    '<mark style="background:#ffebe9;color:#cf222e;font-weight:600;'
                    f'padding:0 1px;border-radius:2px;">{piece}</mark>'
                )
            else:
                out.append(piece)
        return "".join(out)

    def side_by_side(query: str, title: str, context: int = 25) -> None:
        """One table per test spot: the matched passage of every approach,
        with deviations from the reference highlighted in red."""
        td = "padding:6px 10px;vertical-align:top;border-top:1px solid #8b949e55;"
        mono = "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;"
        dim = "opacity:0.55;"
        rows = []
        for name, text in results.items():
            score, line, start, end = find_snippet(text, query)
            before = html.escape(line[max(0, start - context) : start])
            span = _highlight_deviations(query, line[start:end])
            after = html.escape(line[end : end + context])
            rows.append(
                f'<tr><td style="{td}white-space:nowrap;">{name}</td>'
                f'<td style="{td}{mono}opacity:0.7;text-align:right;">{score:.0f}</td>'
                f'<td style="{td}{mono}"><span style="{dim}">…{before}</span>{span}'
                f'<span style="{dim}">{after}…</span></td></tr>'
            )
        header = (
            f'<div style="font-size:15px;font-weight:600;margin-bottom:2px;">'
            f"{html.escape(title)}</div>"
            f'<div style="font-size:12.5px;opacity:0.7;margin-bottom:6px;">reference: '
            f'<code style="{mono}">{html.escape(query)}</code> &nbsp;·&nbsp; '
            f'<mark style="background:#ffebe9;color:#cf222e;padding:0 3px;border-radius:2px;">'
            f"red</mark> = deviates from the reference &nbsp;·&nbsp; score = match confidence</div>"
        )
        display(
            HTML(
                f'<div style="margin:10px 0 26px;font-family:system-ui,sans-serif;">{header}'
                f'<table style="border-collapse:collapse;">{"".join(rows)}</table></div>'
            )
        )

    side_by_side(
        "Fakultät für Informatik Technische Universität München", "Umlauts in the address line"
    )
    side_by_side(
        "its computational complexity per time step and weight is O(1)", "Formula in the abstract"
    )
    side_by_side("RTRL 3 170 0.05 some fraction 173,000", "Table 1, first data row")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What you should see:**

    - **Umlauts:** RapidOCR turns "für/München" into "fir/Minchen". The local Granite VLM loses *every* umlaut ("Fakultt", "Munchen", "Jirger"). Only the external variants reproduce the line correctly.
    - **Formulas:** RapidOCR and Granite give `O(1)` as plain text, the external VLMs as LaTeX `$O(1)$`. That matters if formulas are later rendered or searched for specifically.
    - **Tables:** three approaches give a Markdown table (RapidOCR with broken quotation marks, Granite losing the "≈"), while `docling_api_vlm` returns raw LaTeX with HTML escapes (`&amp;`), unusable for RAG chunking without post-processing.

    **A high metric value is not good quality.** For German documents, and for preserving formulas and tables, the external VLM pays off. For fast full-text search over many documents, local OCR is often enough.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **This trick is useful beyond OCR.** `find_snippet` with `rapidfuzz.partial_ratio` is a general tool for messy text: deduplicating near-identical records, approximate search, or matching user queries against imperfectly extracted text. Worth keeping in the toolbox.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9) Conclusion

    - **Docling OCR (RapidOCR/OcrMacOptions):** the fastest, free, local baseline. Loses umlauts and formulas.
    - **Docling VLM (Granite, local):** good structure and full data control, nothing leaves your machine. In our test, the worst character errors, with every umlaut lost.
    - **External VLM (direct via LiteLLM):** the best text and formula fidelity, and a free choice of model. Costs an API call and time per page.
    - **External VLM via Docling ApiVlmOptions:** the same recognition quality plus structured Docling JSON. Tables may come back as raw LaTeX.

    Rule of thumb: **volume means local OCR, quality means external VLM.** For RAG on technical or German documents the VLM quality usually pays, because OCR errors propagate through the whole pipeline: misrecognised words become worse embeddings, which become worse retrieval hits, which become worse answers.

    **What production pipelines actually do.** Most workloads are neither pure volume nor pure quality. A common hybrid: run the fast local OCR over everything, score each page (confidence values, character statistics, or a check like our umlaut count), then re-process only the low-confidence or business-critical pages with the external VLM. That buys VLM quality where it matters, at a fraction of the cost.

    Next step: ingest the best OCR output into Qdrant, which closes the loop back to notebook 03.
    """)
    return


if __name__ == "__main__":
    app.run()
