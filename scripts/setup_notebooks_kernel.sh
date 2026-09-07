#!/usr/bin/env bash
# Create/use a repository-level notebooks venv and register it as a Jupyter kernel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

KERNEL_NAME="${KERNEL_NAME:-workshop-ragv2}"
DISPLAY_NAME="${DISPLAY_NAME:-Python (workshop-ragV2)}"
VENV_DIR="${VENV_DIR:-.venv-notebooks}"
SKIP_SYNC=false
EXTRAS=()

usage() {
    cat <<'EOF'
Usage: ./scripts/setup_notebooks_kernel.sh [--extra EXTRA]... [--skip-sync]

Options:
  --extra EXTRA     Optional notebooks dependency extra (e.g. vlm, mac_vlm). Can be repeated.
  --skip-sync       Skip `uv sync` and only (re)register the Jupyter kernel.
  -h, --help        Show this help.

Environment overrides:
  KERNEL_NAME       Kernel name (default: workshop-ragv2)
  DISPLAY_NAME      Kernel display name (default: Python (workshop-ragV2))
  VENV_DIR          Virtual environment path (default: .venv-notebooks)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extra)
            if [[ $# -lt 2 ]]; then
                echo "Error: --extra requires a value." >&2
                exit 1
            fi
            EXTRAS+=("$2")
            shift 2
            ;;
        --skip-sync)
            SKIP_SYNC=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'." >&2
            usage
            exit 1
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

cd "$REPO_ROOT"

if [[ "$VENV_DIR" != ".venv" && -d ".venv" ]]; then
    echo "Note: found legacy environment at $REPO_ROOT/.venv."
    echo "If unused, you can remove it to avoid confusion."
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating notebooks virtual environment at $REPO_ROOT/$VENV_DIR ..."
    uv venv "$VENV_DIR"
else
    echo "Using existing notebooks virtual environment at $REPO_ROOT/$VENV_DIR ..."
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/$VENV_DIR/bin/activate"

if [[ "$SKIP_SYNC" == false ]]; then
    echo "Syncing notebook dependencies from notebooks/pyproject.toml ..."
    SYNC_CMD=(uv sync --project notebooks --active)
    if (( ${#EXTRAS[@]} > 0 )); then
        for extra in "${EXTRAS[@]}"; do
            SYNC_CMD+=(--extra "$extra")
        done
    fi
    "${SYNC_CMD[@]}"
else
    echo "Skipping dependency sync (--skip-sync)."
fi

echo "Registering Jupyter kernel '$KERNEL_NAME' ..."
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

echo ""
echo "Done."
echo "Kernel name: $KERNEL_NAME"
echo "Interpreter: $REPO_ROOT/$VENV_DIR/bin/python"
echo ""
echo "Next steps:"
echo "1) Copy notebooks/.env.example to notebooks/.env and insert your API key"
echo "2) Start Qdrant (Docker Desktop or ./scripts/start_qdrant.sh)"
echo "3) Open a notebook, e.g. notebooks/w2_01_chunking_and_retrieval.ipynb"
echo "4) Top-right: Select Kernel -> Jupyter Kernel..."
echo "5) Choose '$DISPLAY_NAME' ($VENV_DIR/bin/python)"
