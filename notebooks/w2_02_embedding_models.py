# ruff: noqa: PLR1711  marimo cells end with an explicit return
import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Image and text embeddings",
    css_file="ragkit/theme.css",
    html_head_file="ragkit/head.html",
)


@app.cell(hide_code=True)
def _(mo, theme):
    mo.Html(theme.header_html(
        'Image and text embeddings',
        'RAG II, notebook 2. Finding pictures with pictures, and what it takes to find pictures with words.',
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 Embeddings are coordinates

    An embedding model turns an input into a list of numbers. Read the list as coordinates and every text or image becomes a point in a space with hundreds or thousands of dimensions. The model is trained so that inputs with similar meaning land close together and unrelated inputs land far apart. Retrieval is then a geometric question: which stored points lie closest to the query point? The distance used throughout this notebook is cosine similarity, the cosine of the angle between two vectors, which is 1 for identical directions and 0 for unrelated ones.

    For text this is familiar from the first notebook. Images work the same way once a model can map pixels into such a space. And when one model maps **both** pictures and words into the **same** space, a photo of a robin lands next to the sentence "small bird with a red breast", and three kinds of search become possible with a single index:

    - a picture finds similar pictures (image to image),
    - a sentence finds pictures (text to image),
    - a picture finds text that describes it (image to text).

    The data are 104 photographs of 26 European bird species from Wikimedia Commons and the German Wikipedia article on each species. The species is known for every photo, so we can measure how often a search returns the right bird.

    > **What this notebook can show today.** The first direction runs on a vision model (dinov3) and is measured in full. The other two need a multimodal embedding model that accepts pictures. The workshop API lists one (qwen3-vl-embedding-8b), but at the moment its endpoint reads an image as a string of characters, not as a picture; section 5 shows how we found out. The cells for those two directions are in the notebook and switch on once the endpoint is fixed.
    """)
    return


@app.cell(hide_code=True)
def _(mo, plt, theme):
    def _toy_map():
        points = {'robin': (0.8, 0.7), 'robin (2nd photo)': (0.86, 0.62), 'sparrow': (0.55, 0.75),
                  'eagle': (-0.7, -0.5), 'kite': (-0.55, -0.65), 'swan': (-0.6, 0.6)}
        queries = {'"small bird with red breast"': (0.7, 0.55), '"flying bird of prey"': (-0.8, -0.35)}
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for name, (x, y) in points.items():
            ax.plot(x, y, 'o', color=theme.INK, markersize=6)
            ax.annotate(name, (x, y), xytext=(6, 4), textcoords='offset points', fontsize=8)
        for name, (x, y) in queries.items():
            ax.plot(x, y, 's', markerfacecolor='white', markeredgecolor=theme.INK, markersize=7)
            ax.annotate(name, (x, y), xytext=(6, -12), textcoords='offset points', fontsize=8, color=theme.GREY)
        ax.set_xlim(-1.1, 1.4)
        ax.set_ylim(-1, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('embedding dimension 1 (of thousands)')
        ax.set_ylabel('embedding dimension 2')
        return fig

    mo.vstack([
        _toy_map(),
        mo.md('*Figure 1. A made-up two-dimensional picture of a shared embedding space. Filled dots are photos, hollow squares are text queries. Nearby points are similar; a query is answered by its nearest neighbours.*'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 Setup

    Expand the cell for one comment per library. The setup check below confirms that the API key and the data are in place; this notebook does not need Qdrant.
    """)
    return


@app.cell(hide_code=True)
def _():
    import csv  # the attribution list of the photos
    import hashlib  # cache keys that follow the photo files
    from pathlib import Path

    import marimo as mo  # the reactive notebook itself
    import matplotlib.pyplot as plt  # figures
    import numpy as np  # vector maths: cosine similarity, ranking, PCA
    import pandas as pd  # result tables
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from ragkit import theme  # HPI look for chrome and figures
    from ragkit.chunk import chunk_markdown_by_headers, wiki_headings_to_markdown
    from ragkit.config import WORKSHOP_DIR, setup
    from ragkit.embed import (  # API calls and the vector cache
        cached_embed,
        embed_dinov3,
        embed_images_qwen3vl_safe,
        embed_qwen3vl,
        img_to_base64,
    )
    from ragkit.search import entropy, l2_normalise, precision_at_k, reciprocal_rank
    from ragkit.viz import image_grid, thumbnail_sheet

    theme.apply_mpl()
    return (
        FancyArrowPatch, FancyBboxPatch, Path, WORKSHOP_DIR, cached_embed, chunk_markdown_by_headers,
        csv, embed_dinov3, embed_images_qwen3vl_safe, embed_qwen3vl, entropy, hashlib, image_grid, img_to_base64,
        l2_normalise, mo, np, pd, plt, precision_at_k, reciprocal_rank, setup, theme, thumbnail_sheet,
        wiki_headings_to_markdown,
    )


@app.cell(hide_code=True)
def _(WORKSHOP_DIR, mo, setup):
    BIRDS = WORKSHOP_DIR / 'raw_data' / 'birds'
    IMG_DIR, ART_DIR = BIRDS / 'images', BIRDS / 'articles'
    # Flip to True once the API embeds pictures as pictures (see section 5); the guard in
    # ragkit.embed raises if it still tokenises the data URI as text.
    MULTIMODAL_IMAGES_AVAILABLE = False
    env = setup(require_qdrant=False, required_files=(BIRDS / 'attribution.csv',), strict=False)
    mo.md(env.summary_md())
    return ART_DIR, BIRDS, IMG_DIR, MULTIMODAL_IMAGES_AVAILABLE, env


@app.cell(hide_code=True)
def _(env, mo):
    mo.stop(not env.api_key, mo.callout(mo.md(
        'The notebook needs the HPI API key to embed images and text. Follow the fix above, then re-run.'), kind='warn'))
    return


@app.cell(hide_code=True)
def _(ART_DIR, BIRDS, IMG_DIR, csv, mo, thumbnail_sheet):
    images = sorted(IMG_DIR.glob('*.jpg'))
    species = [p.stem.rsplit('_', 1)[0] for p in images]
    SPECIES = sorted(set(species))
    german_name = {p.stem: p.read_text(encoding='utf-8').split('\n')[0].lstrip('# ').split(' (')[0]
                   for p in sorted(ART_DIR.glob('*.md'))}
    with (BIRDS / 'attribution.csv').open(encoding='utf-8') as _f:
        attribution = {row['file']: row for row in csv.DictReader(_f)}

    def credit_line(paths) -> str:
        authors = sorted({attribution[p.name]['author'] for p in paths})
        return 'Photos: ' + ', '.join(authors) + ' (Wikimedia Commons, CC licences; full list in section 2)'

    def _overview():
        first = [next(p for p in images if p.stem.startswith(s + '_')) for s in SPECIES]
        return thumbnail_sheet(first, [german_name[s] for s in SPECIES], cols=9)

    _credits = '\n'.join(f"- `{f}`: {r['author']}, {r['licence']}, [source]({r['source']})" for f, r in attribution.items())
    mo.vstack([
        mo.md(f'### The bird photos\n\n{len(images)} photographs, {len(SPECIES)} species, four photos per species; one per species below.'),
        _overview(),
        mo.md('*Figure 2. One photo per species. Photos are from Wikimedia Commons under CC0, CC BY or CC BY-SA licences; the full attribution list is below.*'),
        mo.accordion({'Attribution of all 104 photos': mo.md(_credits)}),
    ])
    return SPECIES, attribution, credit_line, german_name, images, species


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 Image to image: what a vision model captures

    **dinov3-vit-large** is a vision transformer trained on images alone, without any text. It learned to recognise that two crops of the same picture belong together and that different pictures do not, and in doing so it built a representation of shape, colour, texture and layout. It knows nothing about the word "robin"; it only knows what robins tend to look like.

    We embed all 104 photos once (cached, about a minute) and ask: given one photo, which other photos are closest?
    """)
    return


@app.cell(hide_code=True)
def _(cached_embed, embed_dinov3, hashlib, images, img_to_base64, l2_normalise, np):
    image_b64 = [img_to_base64(p) for p in images]
    _fingerprint = hashlib.md5(''.join(b[:64] for b in image_b64).encode()).hexdigest()[:8]  # cache key follows the photos
    emb_dinov3 = l2_normalise(cached_embed(f'w202_img_dinov3_{_fingerprint}', lambda: embed_dinov3(image_b64)))
    sims_dinov3 = emb_dinov3 @ emb_dinov3.T
    np.fill_diagonal(sims_dinov3, -1)  # a photo must not retrieve itself
    return emb_dinov3, image_b64, sims_dinov3


@app.cell(hide_code=True)
def _(credit_line, german_name, image_grid, images, mo, np, species):
    def gallery(query_index: int, sims: np.ndarray, k: int = 5):
        """Figure: the query photo and its k nearest photos under the given similarity matrix."""
        order = np.argsort(-sims[query_index])[:k]
        hits = [images[j] for j in order]
        return image_grid(images[query_index], hits, [float(sims[query_index, j]) for j in order],
                          labels=[german_name[species[j]] for j in order],
                          credit=credit_line([images[query_index], *hits]))

    def index_of(stem: str) -> int:
        return next(i for i, p in enumerate(images) if p.stem == stem)

    mo.md('')
    return gallery, index_of


@app.cell(hide_code=True)
def _(gallery, index_of, mo, sims_dinov3):
    mo.vstack([
        gallery(index_of('rotkehlchen_1'), sims_dinov3),
        mo.md('*Figure 3. Image-to-image search with dinov3: the robin photo on the left and its five nearest photos with cosine similarity. Species names come from the file names, not from the model.*'),
        gallery(index_of('seeadler_1'), sims_dinov3),
        gallery(index_of('eisvogel_2'), sims_dinov3),
        gallery(index_of('elster_1'), sims_dinov3),
        mo.md(r"""
        *Figure 4. Three more queries: a sea eagle, a kingfisher and a magpie.*

        The neighbours share pose, background and colour with the query as much as species. That is what a vision-only model measures: visual similarity. It works well when a species looks alike across photos, and it is misled by a similar-looking bird on a similar-looking branch. Notice also the similarity values: the same-species hits sit around 0.7 to 0.8, the first wrong species often only a little lower. Section 4 measures how often the model is right across all 104 photos.

        **A text model cannot do this at all.** miniLM and octen from the first notebook accept strings only. To search images with them we would first have to describe every image in words (captioning) and search the captions, which throws away everything the caption does not mention.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(SPECIES, german_name, images, mo, np, pd, plt, precision_at_k, reciprocal_rank, sims_dinov3, species, theme):
    def _leave_one_out(sims):
        rows = []
        for i in range(len(images)):
            ranked = [int(j) for j in np.argsort(-sims[i])[:10]]
            relevant = {j for j, s in enumerate(species) if s == species[i] and j != i}
            rows.append({'species': species[i], 'p3': precision_at_k(ranked, relevant, 3),
                         'rr': reciprocal_rank(ranked, relevant), 'top1': species[ranked[0]]})
        return pd.DataFrame(rows)

    loo = _leave_one_out(sims_dinov3)
    per_species = loo.groupby('species')[['p3', 'rr']].mean().sort_values('p3')

    def _per_species_figure():
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        y = np.arange(len(per_species))
        ax.barh(y, per_species['p3'], color=theme.GREYS[2], edgecolor=theme.INK, linewidth=0.5, height=0.7)
        ax.plot(per_species['rr'], y, 'o', markerfacecolor='white', markeredgecolor=theme.INK, markersize=4.5)
        ax.set_yticks(y)
        ax.set_yticklabels([german_name[s] for s in per_species.index], fontsize=7.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Precision@3 (bars) and mean reciprocal rank (hollow markers)')
        ax.axvline(loo['p3'].mean(), color=theme.INK, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(loo['p3'].mean() + 0.01, len(per_species) - 0.6, f"mean P@3 = {loo['p3'].mean():.2f}", fontsize=8, color=theme.INK)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        return fig

    mo.vstack([
        mo.md(rf"""
        ## 4 Measuring it

        The species label of every photo is our ground truth. Each photo in turn is the query, the other 103 are ranked, and we ask how many of the top three show the same species (Precision@3; three other photos of the species exist, so 1.0 is possible) and where the first correct photo appears (reciprocal rank, averaged to MRR). Over all {len(images)} photos dinov3 reaches **Precision@3 {loo['p3'].mean():.2f}** and **MRR {loo['rr'].mean():.2f}**: for {int((loo['rr'] == 1).sum())} of the {len(images)} queries the nearest photo shows the right species.
        """),
        _per_species_figure(),
        mo.md('*Figure 5. Retrieval quality per species (dinov3, leave one out). Species with distinctive shape or colour are found almost perfectly; small brown songbirds and the two large raptors get mixed up with their look-alikes.*'),
    ])
    return loo, per_species


@app.cell(hide_code=True)
def _(SPECIES, german_name, loo, mo, np, plt, theme):
    def _confusion():
        idx = {s: i for i, s in enumerate(SPECIES)}
        mat = np.zeros((len(SPECIES), len(SPECIES)))
        for _, row in loo.iterrows():
            mat[idx[row['species']], idx[row['top1']]] += 1
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.imshow(mat, cmap='Greys', vmin=0, vmax=4)
        labels = [german_name[s] for s in SPECIES]
        ax.set_xticks(range(len(SPECIES)))
        ax.set_yticks(range(len(SPECIES)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel('species of the nearest photo')
        ax.set_ylabel('species of the query photo')
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        return fig, int(mat.sum() - np.trace(mat))

    _fig, _off = _confusion()
    mo.vstack([
        _fig,
        mo.md(f'*Figure 6. Which species the nearest photo belongs to (dinov3, top 1). The diagonal holds the correct answers; {_off} of 104 queries land on another species, mostly on a bird of similar shape or colour.*'),
    ])
    return


@app.cell(hide_code=True)
def _(SPECIES, emb_dinov3, german_name, mo, np, plt, species, theme):
    def _pca():
        centred = emb_dinov3 - emb_dinov3.mean(axis=0)
        _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
        coords = centred @ vt[:2].T
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.scatter(coords[:, 0], coords[:, 1], s=14, color=theme.GREYS[2], linewidths=0)
        for sp in SPECIES:
            pts = coords[[i for i, s in enumerate(species) if s == sp]]
            ax.annotate(german_name[sp], pts.mean(axis=0), fontsize=6.5, ha='center', va='center', color=theme.INK)
        ax.set_xlabel('principal component 1')
        ax.set_ylabel('principal component 2')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    mo.vstack([
        mo.md(r"""
        ### A map of the space

        A thousand dimensions cannot be drawn, but the two directions along which the photo vectors vary most can (principal component analysis, computed here with a singular value decomposition in numpy). Each dot is a photo and the species name sits at the centre of its four photos. Two dimensions keep only a small part of the distances the model uses, so overlapping names in the middle are expected; the outer species are the ones that look unlike everything else.
        """),
        _pca(),
        mo.md('*Figure 7. The first two principal components of the dinov3 photo embeddings.*'),
    ])
    return


@app.cell(hide_code=True)
def _(FancyArrowPatch, FancyBboxPatch, mo, plt, theme):
    def _directions_figure():
        fig, ax = plt.subplots(figsize=(6.5, 2.3))
        ax.set_axis_off()
        boxes = {'photo': (0.05, 0.55), 'photo (other)': (0.72, 0.55), 'text': (0.72, 0.05), 'shared space': (0.385, 0.3)}
        for name, (x, y) in boxes.items():
            centre = name == 'shared space'
            ax.add_patch(FancyBboxPatch((x, y), 0.23, 0.3 if centre else 0.25, boxstyle='round,pad=0,rounding_size=0.02',
                                        facecolor=theme.INK if centre else 'white', edgecolor=theme.INK, linewidth=0.8))
            ax.text(x + 0.115, y + (0.15 if centre else 0.125), name, ha='center', va='center', fontsize=8.5,
                    color='white' if centre else theme.INK)
        arrows = [((0.28, 0.67), (0.385, 0.5)), ((0.615, 0.5), (0.72, 0.67)), ((0.615, 0.4), (0.72, 0.2))]
        for start, end in arrows:
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=9, color=theme.GREY, linewidth=0.9))
        ax.text(0.5, 0.93, 'image to image', ha='center', fontsize=8, color=theme.GREY)
        ax.text(0.5, -0.02, 'image to text (and text to image, the reverse arrow)', ha='center', fontsize=8, color=theme.GREY)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1)
        return fig

    mo.vstack([
        mo.md(r"""
        ## 5 One space for pictures and words

        A **multimodal embedding model** such as qwen3-vl-embedding-8b is trained on pairs of images and texts so that a picture and a description of it land close together. Every input, whether pixels or characters, is mapped into the same 4 096-dimensional space. A query can then be a photo or a sentence, and one index can hold photos, article chunks or both.
        """),
        _directions_figure(),
        mo.md('*Figure 8. Three searches, one index. Photos and texts are embedded into the shared space; the query is whatever we have at hand.*'),
        mo.md(r"""
        ### Checking that an endpoint does what its name says

        The workshop API offers this model behind the same OpenAI-style `/embeddings` route as the text models. Sending a photo means sending it as a `data:image/jpeg;base64,...` string. Whether the server decodes that string into a picture or embeds the characters as text is invisible in the response, unless you look. Three cheap checks told us what happened in September 2026:

        | check | expected for a picture | observed |
        |---|---|---|
        | `usage.prompt_tokens` for one 512 px photo | a few hundred vision tokens | 31 964, growing with the length of the base64 string |
        | cosine similarity between a robin and a swan | clearly below same-species pairs | 0.97 |
        | cosine similarity between "kleiner Vogel mit roter Brust" and any photo | high for robins, low for others | 0.39 to 0.41 for every photo |

        So the endpoint embedded the base64 text, and every "image" vector was a text vector of gibberish. The lesson generalises: whenever a model is wrapped in a generic API, verify one known case before indexing thousands of items. `ragkit.embed` now raises if the token count reveals this situation, and the two cells below stay switched off until the endpoint accepts pictures.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(MULTIMODAL_IMAGES_AVAILABLE, cached_embed, embed_images_qwen3vl_safe, image_b64, images, l2_normalise, mo, np):
    mo.stop(not MULTIMODAL_IMAGES_AVAILABLE, mo.callout(mo.md(
        '**Text to image and image to text are switched off** (`MULTIMODAL_IMAGES_AVAILABLE = False` in the setup cell) '
        'because the API embeds pictures as text at the moment. Set the flag to True once the endpoint is fixed; the cells '
        'below then embed the photos with qwen3-vl and run the two remaining search directions.'), kind='info'))

    def _embed_images_qwen():
        dicts = [{'base64': b, 'name': p.name} for b, p in zip(image_b64, images)]
        vectors, ok = embed_images_qwen3vl_safe(dicts, max_pixels=512)
        if len(ok) != len(images):
            raise RuntimeError(f'{len(images) - len(ok)} images could not be embedded')
        return vectors

    emb_qwen_img = l2_normalise(cached_embed('w202_img_qwen3vl', _embed_images_qwen))
    sims_qwen = emb_qwen_img @ emb_qwen_img.T
    np.fill_diagonal(sims_qwen, -1)
    mo.md(f'qwen3-vl embedded {len(images)} photos into {emb_qwen_img.shape[1]:,} dimensions.')
    return emb_qwen_img, sims_qwen


@app.cell(hide_code=True)
def _(ART_DIR, cached_embed, chunk_markdown_by_headers, embed_qwen3vl, l2_normalise, mo, wiki_headings_to_markdown):
    article_chunks, chunk_species = [], []
    for _path in sorted(ART_DIR.glob('*.md')):
        for _chunk in chunk_markdown_by_headers(wiki_headings_to_markdown(_path.read_text(encoding='utf-8')), 1200, 200):
            article_chunks.append(_chunk)
            chunk_species.append(_path.stem)

    def _embed_chunks():
        out = []
        for i in range(0, len(article_chunks), 4):  # 4 x 1 400 chars stays far below the 32k-token context
            out.extend(embed_qwen3vl(texts=article_chunks[i:i + 4]))
        return out

    emb_chunks = l2_normalise(cached_embed('w202_txt_qwen3vl_chunks', _embed_chunks))
    mo.md(rf"""
    ## 6 The text side: searching the articles

    The 26 Wikipedia articles are split at their headings into {len(article_chunks):,} chunks of at most 1 200 characters (the same chunker as in the first notebook) and embedded with qwen3-vl as text, which the endpoint does correctly. This is the index that a photo would query in the image-to-text direction. Until then it answers text questions, which also shows that the multimodal model is a capable text model: a description of a bird should retrieve the article of that bird.
    """)
    return article_chunks, chunk_species, emb_chunks


@app.cell(hide_code=True)
def _(SPECIES, article_chunks, chunk_species, emb_chunks, embed_qwen3vl, german_name, l2_normalise, mo, np, pd, reciprocal_rank):
    TEXT_QUERIES = ['kleiner Vogel mit roter Brust', 'großer Greifvogel, der Fische jagt', 'schwarz-weißer Vogel mit langem Schwanz',
                    'großer weißer Wasservogel', 'tiny bird with red breast']
    text_query_emb = l2_normalise(np.asarray(embed_qwen3vl(texts=TEXT_QUERIES)))

    def text_to_articles(qvec, k: int = 3):
        scores = emb_chunks @ qvec
        order = np.argsort(-scores)[:k]
        return [{'rank': r + 1, 'score': round(float(scores[j]), 3), 'article': german_name[chunk_species[j]],
                 'chunk': article_chunks[j][:160].replace('\n', ' ') + ' …'} for r, j in enumerate(order)]

    def _name_query_mrr():
        names = [german_name[s] for s in SPECIES]
        qvecs = l2_normalise(np.asarray(embed_qwen3vl(texts=[f'Der Vogel {n}' for n in names])))
        scores = qvecs @ emb_chunks.T
        rr = [reciprocal_rank([int(j) for j in np.argsort(-scores[i])[:20]], {j for j, s in enumerate(chunk_species) if s == sp})
              for i, sp in enumerate(SPECIES)]
        return pd.DataFrame({'species': names, 'reciprocal rank': np.round(rr, 2)}), float(np.mean(rr))

    name_table, name_mrr = _name_query_mrr()
    _blocks = [mo.vstack([mo.md(f'**"{q}"**'), mo.ui.table(text_to_articles(v), selection=None)]) for q, v in zip(TEXT_QUERIES, text_query_emb)]
    mo.vstack([
        *_blocks,
        mo.md(f'Descriptions in German and English retrieve the matching article in most cases; the model was trained on many languages. Querying with each species name retrieves a chunk of the right article at mean reciprocal rank **{name_mrr:.2f}** over the 26 species.'),
        mo.accordion({'Reciprocal rank per species name': mo.ui.table(name_table.to_dict('records'), selection=None)}),
    ])
    return TEXT_QUERIES, name_mrr, name_table, text_query_emb, text_to_articles


@app.cell(hide_code=True)
def _(MULTIMODAL_IMAGES_AVAILABLE, credit_line, emb_chunks, emb_qwen_img, german_name, image_grid, images, mo, np, species, text_query_emb, TEXT_QUERIES, article_chunks, chunk_species, index_of):
    mo.stop(not MULTIMODAL_IMAGES_AVAILABLE, mo.md('*Text to image and image to text: switched off, see section 5.*'))

    def text_gallery(query: str, qvec, k: int = 5):
        scores = emb_qwen_img @ qvec
        order = np.argsort(-scores)[:k]
        hits = [images[j] for j in order]
        fig = image_grid(hits[0], hits, [float(scores[j]) for j in order],
                         labels=[german_name[species[j]] for j in order], credit=credit_line(hits))
        fig.axes[0].images[0].set_visible(False)
        fig.axes[0].patches[0].set_visible(False)
        fig.suptitle(f'"{query}"', fontsize=9, x=0.01, ha='left')
        return fig

    def image_to_text(stem: str, k: int = 3):
        i = index_of(stem)
        scores = emb_chunks @ emb_qwen_img[i]
        order = np.argsort(-scores)[:k]
        return [{'rank': r + 1, 'score': round(float(scores[j]), 3), 'article': german_name[chunk_species[j]],
                 'chunk': article_chunks[j][:200].replace('\n', ' ') + ' …'} for r, j in enumerate(order)]

    mo.vstack([
        mo.md('## 7 Text to image and image to text (multimodal)'),
        *[text_gallery(q, v) for q, v in zip(TEXT_QUERIES, text_query_emb)],
        mo.md('*Text queries and their five best photos.*'),
        mo.md('**Robin photo, nearest article chunks:**'),
        mo.ui.table(image_to_text('rotkehlchen_1'), selection=None),
    ])
    return image_to_text, text_gallery


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 Your own searches

    Pick a query photo (image to image with dinov3) or type a description (text search over the Wikipedia articles), choose how many hits to show, then press Search.
    """)
    return


@app.cell(hide_code=True)
def _(images, mo):
    ui_mode = mo.ui.dropdown(options=['photo', 'text'], value='photo', label='query type')
    ui_image = mo.ui.dropdown(options=[p.name for p in images], value='rotkehlchen_1.jpg', label='query photo', searchable=True)
    ui_text = mo.ui.text(value='ein Vogel mit blauem Rücken, der am Wasser jagt', label='query text', full_width=True)
    ui_k = mo.ui.slider(start=1, stop=10, step=1, value=5, label='hits')
    ui_go = mo.ui.run_button(label='Search')
    mo.vstack([mo.hstack([ui_mode, ui_k], justify='start'), ui_image, ui_text, ui_go])
    return ui_go, ui_image, ui_k, ui_mode, ui_text


@app.cell(hide_code=True)
def _(embed_qwen3vl, entropy, gallery, index_of, l2_normalise, mo, np, sims_dinov3, text_to_articles, ui_go, ui_image, ui_k, ui_mode, ui_text):
    mo.stop(not ui_go.value, mo.md('*Press Search.*'))

    def _run():
        k = ui_k.value
        if ui_mode.value == 'photo':
            i = index_of(ui_image.value.removesuffix('.jpg'))
            top = np.sort(sims_dinov3[i])[::-1][:max(k, 5)]
            return gallery(i, sims_dinov3, k), mo.md(f'entropy of the top scores: {entropy(top, temperature=0.05):.2f} bit (0 = one clear winner, {np.log2(len(top)):.2f} = all alike)')
        qvec = l2_normalise(np.asarray(embed_qwen3vl(texts=[ui_text.value])))[0]
        return mo.ui.table(text_to_articles(qvec, k), selection=None), mo.md('')

    _out, _extra = _run()
    mo.vstack([_out, _extra])
    return


if __name__ == "__main__":
    app.run()
