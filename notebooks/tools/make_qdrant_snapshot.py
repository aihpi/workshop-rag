"""Save the two w3 collections as Qdrant snapshots (instructors only).

Run in the `notebooks` folder while Qdrant holds both collections:

    uv run python -m tools.make_qdrant_snapshot

The files land in `notebooks/data/qdrant_snapshot/<collection>.snapshot.gz`
(gitignored, gzipped, about 120 MB together). `ragkit.ingest` can restore them, so the
collections need not be rebuilt from the PDF when the caches are recomputed.
"""

import argparse
import sys
from pathlib import Path

from ragkit import ingest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=ingest.SNAPSHOT_DIR, help='target folder')
    args = parser.parse_args()

    counts = ingest.collection_counts()
    missing = [name for name, n in counts.items() if not n]
    if missing:
        print('Missing or empty in Qdrant, build them with ragkit.ingest first: ' + ', '.join(missing))
        return 1
    for name, n in counts.items():
        print(f'{name}: {n} points, creating snapshot ...', flush=True)
        path = ingest.create_snapshot(name, args.out)
        print(f'  -> {path} ({path.stat().st_size / 1e6:.0f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
