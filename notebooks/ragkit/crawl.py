"""Fetch the bird dataset for the multimodal notebook (run once by maintainers).

For each species: the German Wikipedia article as plain text and a few
photographs from the matching Wikimedia Commons category, restricted to
CC0 / CC BY / CC BY-SA licences. Everything lands in `raw_data/birds/` with
an `attribution.csv`, and that folder is committed, so participants never
crawl. The module doubles as the workshop's example of how data *could* be
collected from the web.

    python -m ragkit.crawl --dry-run   # list what would be fetched
    python -m ragkit.crawl             # fetch and write
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import time
from pathlib import Path

import requests

from .config import WORKSHOP_DIR

OUT_DIR = WORKSHOP_DIR / 'raw_data' / 'birds'
USER_AGENT = 'HPI-RAG-Workshop/1.0 (https://github.com/aihpi/workshop-ragV2; teaching material)'
COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
DEWIKI_API = 'https://de.wikipedia.org/w/api.php'
ALLOWED_LICENCES = ('cc0', 'cc by', 'cc-by', 'public domain', 'pd')
IMAGES_PER_SPECIES = 4
MAX_SIDE = 800

# (slug, German Wikipedia title, Commons category = scientific name)
SPECIES = [
    ('rotkehlchen', 'Rotkehlchen', 'Erithacus rubecula'),
    ('amsel', 'Amsel', 'Turdus merula'),
    ('kohlmeise', 'Kohlmeise', 'Parus major'),
    ('blaumeise', 'Blaumeise', 'Cyanistes caeruleus'),
    ('haussperling', 'Haussperling', 'Passer domesticus'),
    ('buchfink', 'Buchfink', 'Fringilla coelebs'),
    ('gruenfink', 'Grünfink', 'Chloris chloris'),
    ('stieglitz', 'Stieglitz', 'Carduelis carduelis'),
    ('star', 'Star (Art)', 'Sturnus vulgaris'),
    ('elster', 'Elster', 'Pica pica'),
    ('eichelhaeher', 'Eichelhäher', 'Garrulus glandarius'),
    ('rabenkraehe', 'Rabenkrähe', 'Corvus corone'),
    ('ringeltaube', 'Ringeltaube', 'Columba palumbus'),
    ('mauersegler', 'Mauersegler', 'Apus apus'),
    ('rauchschwalbe', 'Rauchschwalbe', 'Hirundo rustica'),
    ('buntspecht', 'Buntspecht', 'Dendrocopos major'),
    ('zaunkoenig', 'Zaunkönig', 'Troglodytes troglodytes'),
    ('eisvogel', 'Eisvogel', 'Alcedo atthis'),
    ('stockente', 'Stockente', 'Anas platyrhynchos'),
    ('hoeckerschwan', 'Höckerschwan', 'Cygnus olor'),
    ('graureiher', 'Graureiher', 'Ardea cinerea'),
    ('weissstorch', 'Weißstorch', 'Ciconia ciconia'),
    ('maeusebussard', 'Mäusebussard', 'Buteo buteo'),
    ('rotmilan', 'Rotmilan', 'Milvus milvus'),
    ('seeadler', 'Seeadler (Art)', 'Haliaeetus albicilla'),
    ('turmfalke', 'Turmfalke', 'Falco tinnunculus'),
]

_NOT_A_PHOTO = re.compile(r'map|range|distribution|verbreitung|spectro|sonogram|egg|eier|\bei\b|skull|skelet|'
                          r'\.svg|\.png|\.gif|drawing|illustration|zeichnung|stamp|karte|nestling|chick|'
                          r'küken|feather|feder|footprint|museum|specimen|präparat|MHNT|juvenil|immature|'
                          r'jungvogel|nest|strona|page|plate|tafel|\b1[89]\d\d\b|young|jonge|duckling|młode|'
                          r'natural.history|cemetery|brunnen|palace|kraftwerk|cowbird', re.IGNORECASE)


# Dropped after a visual review of the first crawl: distant subjects, look-alike
# duplicates of an already chosen photo, or the bird is not the main subject.
EXCLUDE_TITLES = {
    'File:Amsel, Männchen (Mai 2025) 2.jpg',
    'File:Buntspecht (März 2025).jpg',
    'File:Buntspecht Männchen (April 2025).jpg',
    'File:A grey heron on a boat.jpg',
    'File:Bayreuth Kriegsgräberstätte St. Georgen-20190324-RM-170540.jpg',
    'File:20210930 Falco tinnunculus.jpg',
    'File:20240815 Falco tinnunculus.jpg',
}


def _get(url: str, **params) -> dict:
    r = requests.get(url, params={'format': 'json', **params},
                     headers={'User-Agent': USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json()


def licence_ok(short_name: str) -> bool:
    """Accept CC0, CC BY, CC BY-SA and public domain; reject everything else (incl. NC/ND)."""
    s = short_name.lower()
    if '-nc' in s or '-nd' in s or 'nc ' in s or 'nd ' in s:
        return False
    return any(s.startswith(a) for a in ALLOWED_LICENCES)


def fetch_extract(title: str) -> str:
    """Plain-text article from German Wikipedia (section headings as == ... ==)."""
    data = _get(DEWIKI_API, action='query', prop='extracts', explaintext=1, redirects=1,
                titles=title)
    page = next(iter(data['query']['pages'].values()))
    return page.get('extract', '')


WIKIDATA_API = 'https://www.wikidata.org/w/api.php'


def _wikidata_qid(scientific_name: str) -> str | None:
    """Wikidata item of the species (the first hit described as a species)."""
    data = _get(WIKIDATA_API, action='wbsearchentities', search=scientific_name, language='en',
                type='item', limit=5)
    for hit in data.get('search', []):
        if 'species' in hit.get('description', ''):
            return hit['id']
    return None


def _search_titles(query: str, limit: int = 30) -> list[str]:
    """File titles from Commons full-text search (namespace 6 = File)."""
    data = _get(COMMONS_API, action='query', list='search', srsearch=query, srnamespace=6,
                srlimit=limit)
    return [hit['title'] for hit in data.get('query', {}).get('search', [])]


def _imageinfo(titles: list[str]) -> list[dict]:
    if not titles:
        return []
    data = _get(COMMONS_API, action='query', titles='|'.join(titles[:50]),
                prop='imageinfo', iiprop='url|mime|extmetadata|size',
                iiextmetadatafilter='LicenseShortName|Artist')
    return list(data.get('query', {}).get('pages', {}).values())


def _category_pages(category: str) -> list[dict]:
    data = _get(COMMONS_API, action='query', generator='categorymembers',
                gcmtitle=f'Category:{category}', gcmtype='file', gcmlimit=50,
                prop='imageinfo', iiprop='url|mime|extmetadata|size',
                iiextmetadatafilter='LicenseShortName|Artist')
    return list(data.get('query', {}).get('pages', {}).values())


def _usable(page: dict) -> dict | None:
    info = (page.get('imageinfo') or [None])[0]
    if not info:
        return None
    meta = info.get('extmetadata', {})
    lic = meta.get('LicenseShortName', {}).get('value', '')
    title = page['title']
    if info.get('mime') != 'image/jpeg' or _NOT_A_PHOTO.search(title) or not licence_ok(lic):
        return None
    if title in EXCLUDE_TITLES or info.get('width', 0) < 600:
        return None
    return {
        'title': title, 'url': info['url'], 'licence': lic,
        'author': re.sub(r'<[^>]+>', '', meta.get('Artist', {}).get('value', '')).strip(),
        'source': info.get('descriptionurl', ''),
    }


def fetch_image_candidates(category: str, want: int = IMAGES_PER_SPECIES,
                           exclude: set[str] = frozenset()) -> list[dict]:
    """Photos for one species, best sources first.

    1. Commons files whose structured data says they *depict* the species (Wikidata
       item) and that carry the community's Quality-image assessment.
    2. Any file depicting the species.
    3. Files in the species' Commons category (alphabetical, so least curated).
    JPEG, licence, size and file-name filters apply to all three; `exclude` holds
    titles already taken by another species (a photo of an eagle chased by crows
    depicts both, and must not become a photo of each).
    """
    pages: list[dict] = []
    qid = _wikidata_qid(category)
    if qid:
        depicts = f'haswbstatement:P180={qid} filemime:image/jpeg'
        pages += _imageinfo(_search_titles(f'{depicts} hastemplate:"Quality image"'))
        pages += _imageinfo(_search_titles(depicts))
    pages += _category_pages(category)
    out, seen = [], set(exclude)
    for page in pages:
        cand = _usable(page)
        if cand and cand['title'] not in seen:
            seen.add(cand['title'])
            out.append(cand)
        if len(out) >= want:
            break
    return out


def resize_jpeg(data: bytes, max_side: int = MAX_SIDE) -> bytes:
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert('RGB')
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=88, optimize=True)
    return buf.getvalue()


def _slugify(title: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')


def crawl(out_dir: Path = OUT_DIR, dry_run: bool = False) -> list[dict]:
    """Fetch articles and images for every species; return the attribution rows."""
    img_dir, art_dir = out_dir / 'images', out_dir / 'articles'
    rows, used_titles = [], set()
    for slug, de_title, category in SPECIES:
        cands = fetch_image_candidates(category, exclude=used_titles)
        used_titles.update(c['title'] for c in cands)
        print(f'{de_title:14s} {len(cands)} photos', end='')
        if not dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            art_dir.mkdir(parents=True, exist_ok=True)
            text = fetch_extract(de_title)
            (art_dir / f'{slug}.md').write_text(f'# {de_title} ({category})\n\n{text}\n', encoding='utf-8')
            print(f', article {len(text):,} chars', end='')
        for i, c in enumerate(cands, 1):
            fname = f'{slug}_{i}.jpg'
            if not dry_run:
                data = requests.get(c['url'], headers={'User-Agent': USER_AGENT}, timeout=60).content
                (img_dir / fname).write_bytes(resize_jpeg(data))
                time.sleep(0.5)
            rows.append({'file': fname, 'species': de_title, 'scientific_name': category,
                         'author': c['author'], 'licence': c['licence'], 'source': c['source']})
        print()
    if not dry_run:
        with (out_dir / 'attribution.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='list candidates, write nothing')
    ap.add_argument('--out', type=Path, default=OUT_DIR)
    args = ap.parse_args()
    rows = crawl(args.out, dry_run=args.dry_run)
    print(f'{len(rows)} images{" (dry run)" if args.dry_run else " written to " + str(args.out)}')
