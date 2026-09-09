"""Anonymous score submission for the workshop playground.

Participants have a GitHub account but no write access to any workshop
repository, so a submission is an issue in the public repository named by
`SCORES_REPO`: anyone with an account can open one, nothing can conflict, and
no permission has to be granted before the workshop or revoked after it.

Identity stays out of the data. The notebook draws a handle once, keeps it in
a gitignored file and reuses it for both rounds, which is what lets the two
tries of one person be paired without knowing who that person is.

The instructor confirms a session by applying a `session-YYYY-MM-DD` label to
the issues that belong to it; unlabelled issues are ignored, so a submission
sent in after the workshop cannot inflate a past session.
"""

from __future__ import annotations

import json
import random
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

SCORES_REPO = 'aihpi/workshop-rag-scores'
API = 'https://api.github.com'
SESSION_PREFIX = 'session-'
STATE_PATH = Path(__file__).resolve().parent.parent / '.playground' / 'state.json'

_ADJECTIVES = ('teal', 'amber', 'lime', 'slate', 'copper', 'violet', 'coral', 'olive', 'indigo', 'rust')
_ANIMALS = ('otter', 'crane', 'lynx', 'marten', 'heron', 'ibex', 'shrew', 'grebe', 'stoat', 'raven')


# --- local state ------------------------------------------------------------

def load_state(path: Path | None = None) -> dict[str, Any]:
    """The participant's handle and their per-round history, created on first use."""
    path = path or STATE_PATH
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    state = {'handle': f'{random.choice(_ADJECTIVES)}-{random.choice(_ANIMALS)}-{random.randrange(100):02d}',
             'rounds': {}}
    save_state(state, path)
    return state


def save_state(state: dict[str, Any], path: Path | None = None) -> None:
    path = path or STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1), encoding='utf-8')


# --- the payload ------------------------------------------------------------

def encode_body(payload: dict[str, Any]) -> str:
    """One `key: value` per line, which is readable in the issue and trivial to parse back."""
    return '\n'.join(f'{k}: {v}' for k, v in payload.items())


def decode_body(body: str) -> dict[str, str]:
    """The inverse of `encode_body`, ignoring anything a participant typed around it."""
    out = {}
    for line in (body or '').splitlines():
        key, sep, value = line.partition(':')
        if sep and key.strip() and ' ' not in key.strip():
            out[key.strip()] = value.strip()
    return out


def issue_url(payload: dict[str, Any], repo: str = SCORES_REPO) -> str:
    """A link that opens GitHub's new-issue form with everything already filled in."""
    query = urlencode({'title': f"score {payload['handle']} try{payload['try']}",
                       'body': encode_body(payload), 'labels': 'score'})
    return f'https://github.com/{repo}/issues/new?{query}'


def submit_with_gh(payload: dict[str, Any], repo: str = SCORES_REPO) -> str | None:
    """Open the issue through an authenticated `gh`, or None if that is not available."""
    try:
        done = subprocess.run(
            ['gh', 'issue', 'create', '--repo', repo, '--label', 'score',
             '--title', f"score {payload['handle']} try{payload['try']}",
             '--body', encode_body(payload)],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


# --- reading a session back -------------------------------------------------

def _get(path: str) -> Any:
    """GitHub REST through `gh` when it is authenticated (5 000 requests an hour), else plain HTTP (60)."""
    try:
        done = subprocess.run(['gh', 'api', path], capture_output=True, text=True, timeout=30, check=False)
        if done.returncode == 0:
            return json.loads(done.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    request = urllib.request.Request(f'{API}/{path}', headers={'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def session_labels(repo: str = SCORES_REPO) -> list[str]:
    """Every confirmed session, newest first."""
    labels = [lab['name'] for lab in _get(f'repos/{repo}/labels?per_page=100')]
    return sorted((lab for lab in labels if lab.startswith(SESSION_PREFIX)), reverse=True)


def fetch_session(label: str, repo: str = SCORES_REPO) -> list[dict[str, str]]:
    """The decoded submissions of one confirmed session."""
    issues = _get(f'repos/{repo}/issues?labels={label}&state=all&per_page=100')
    return [decode_body(issue.get('body', '')) for issue in issues if 'pull_request' not in issue]


def pair_tries(submissions: list[dict[str, str]]) -> list[dict[str, Any]]:
    """One row per handle with its first and second try, dropping anything unpaired."""
    by_handle: dict[str, dict[str, float]] = {}
    for row in submissions:
        handle, attempt = row.get('handle'), row.get('try')
        if not handle or attempt not in ('1', '2'):
            continue
        try:
            by_handle.setdefault(handle, {})[attempt] = float(row['recall_at_5'])
        except (KeyError, ValueError):
            continue
    return [{'handle': h, 'first': v['1'], 'second': v['2']}
            for h, v in sorted(by_handle.items()) if '1' in v and '2' in v]


def today_label() -> str:
    """The label the instructor applies to confirm today's submissions."""
    return f'{SESSION_PREFIX}{datetime.now().astimezone().date().isoformat()}'
