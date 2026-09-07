"""Convert a Jupyter notebook to a themed marimo notebook with all code hidden.

    uv run python tools/convert_nb.py w2_03_real_world_datentypen.ipynb

Runs `marimo convert`, then post-processes the result: every cell gets
`hide_code=True`, `plt.show()` becomes the figure as the cell's last
expression, and the App gets the ragkit theme files. The remaining manual
work (IPython.display calls, relative paths, duplicate names) is listed at
the end so nothing is forgotten.
"""

import re
import subprocess
import sys
from pathlib import Path

THEME = 'css_file="ragkit/theme.css", html_head_file="ragkit/head.html"'


def convert(src: Path) -> Path:
    dst = src.with_suffix('.py')
    subprocess.run([sys.executable, '-m', 'marimo', 'convert', str(src), '-o', str(dst)], check=True)
    s = dst.read_text(encoding='utf-8')
    s = re.sub(r'@app\.cell\n', '@app.cell(hide_code=True)\n', s)
    s = re.sub(r'@app\.cell\((?!.*hide_code)(.*?)\)\n', r'@app.cell(\1, hide_code=True)\n', s)
    s = re.sub(r'^(\s*)plt\.show\(\)\s*$', r'\1plt.gcf()', s, flags=re.MULTILINE)
    s = re.sub(r'app = marimo\.App\((.*?)\)', lambda m: f'app = marimo.App({m.group(1) + ", " if m.group(1) else ""}{THEME})', s, count=1)
    dst.write_text(s, encoding='utf-8')
    return dst


def report(dst: Path) -> None:
    s = dst.read_text(encoding='utf-8')
    checks = {
        'IPython.display / display( / HTML(': len(re.findall(r'IPython\.display|\bdisplay\(|\bHTML\(', s)),
        'relative paths (raw_data/, processed/, data/, img/)': len(re.findall(r'["\'](raw_data|processed|data|img|\.\./)', s)),
        'renamed duplicates (_name)': len(re.findall(r'\b_[a-z]\w* = ', s)),
        'magics/shell left (%, !)': len(re.findall(r'^\s*[%!]', s, flags=re.MULTILINE)),
    }
    print(f'{dst}: converted')
    for k, v in checks.items():
        print(f'  {v:3d}  {k}')


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        report(convert(Path(arg)))
