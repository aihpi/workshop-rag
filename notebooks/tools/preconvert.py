"""Rewrite the Jupyter sources so that `marimo convert` produces a runnable, themed notebook.

Applied to a copy of each .ipynb before conversion:
- `from ragkit.config import *` becomes an explicit import of the config names the notebook uses
  (marimo cannot analyse star imports).
- `setup()` returns a WorkshopEnv now; the notebook shows its Markdown summary in a new cell.
- the logo header cell becomes `mo.Html(theme.header_html(...))`; other image paths point at /public/img.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit import config

CONFIG_NAMES = sorted(n for n in dir(config) if not n.startswith('_') and n not in ('os', 'Path', 'dataclass', 'field', 'load_dotenv'))


def rewrite(nb_path: Path, out_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    used = [n for n in CONFIG_NAMES if re.search(rf'\b{n}\b', code.replace('from ragkit.config import *', ''))]
    new_cells = []
    for cell in nb['cells']:
        src = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown' and '00_aisc/img/logo' in src:
            title = re.search(r'<h1>\s*(.*?)\s*(</h1>|\n)', src)
            cell = {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
                    'source': f"mo.Html(theme.header_html({(title.group(1).strip() if title else 'Workshop')!r}))"}
        elif cell['cell_type'] == 'markdown':
            cell['source'] = src.replace('../00_aisc/img/', '/public/img/').replace('src="img/', 'src="/public/img/').replace('](img/', '](/public/img/')
        elif 'from ragkit.config import *' in src:
            names = ', '.join(sorted(set(used + ['setup'])))
            src = src.replace('from ragkit.config import *', f'from ragkit import theme\nfrom ragkit.config import {names}')
            src = re.sub(r'^setup\(\)[^\n]*$', 'env = setup(strict=False)\ntheme.apply_mpl()', src, flags=re.MULTILINE)
            cell['source'] = src
            new_cells.append(cell)
            new_cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
                              'source': 'mo.md(env.summary_md())'})
            continue
        new_cells.append(cell)
    nb['cells'] = new_cells
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(exist_ok=True)
    for arg in sys.argv[2:]:
        rewrite(Path(arg), out_dir / Path(arg).name)
        print('rewrote', arg)
