#!/usr/bin/env bash
# Export Marp slides (HTML/PDF). Requires the shared "slides" conda env.
# Usage (from repo root):
#   examples/summary/export.sh marp-html examples/summary/june18_2026.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${SLIDES_CONDA_ENV:-slides}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
if [ ! -d "${CONDA_ROOT}/envs/${ENV_NAME}" ] && [ -d "${HOME}/miniconda3/envs/${ENV_NAME}" ]; then
  CONDA_ROOT="${HOME}/miniconda3"
fi
SLIDES_BIN="${CONDA_ROOT}/envs/${ENV_NAME}/bin"
MODE="${1:-marp-html}"
INPUT="${2:-${SCRIPT_DIR}/june18_2026.md}"
STEM="$(basename "${INPUT}" .md)"
OUT_DIR="$(dirname "${INPUT}")"

if [ ! -x "${SLIDES_BIN}/marp" ]; then
  echo "Slides env not found at ${SLIDES_BIN}" >&2
  echo "Create slides env with marp-cli, or set SLIDES_CONDA_ENV." >&2
  exit 1
fi

export PATH="${SLIDES_BIN}:${PATH}"

if [ ! -f "${INPUT}" ]; then
  echo "Input not found: ${INPUT}" >&2
  exit 1
fi

case "${MODE}" in
  marp-html)
    OUT_HTML="${OUT_DIR}/${STEM}.html"
    marp --no-stdin --allow-local-files "${INPUT}" -o "${OUT_HTML}"
    python3 "${SCRIPT_DIR}/embed_images.py" "${OUT_HTML}"
    echo "Wrote ${OUT_HTML} (images embedded)"
    ;;
  marp-pdf)
    if marp --no-stdin --allow-local-files "${INPUT}" -o "${OUT_DIR}/${STEM}.pdf"; then
      echo "Wrote ${OUT_DIR}/${STEM}.pdf"
    else
      echo "Headless PDF failed. Try: $0 marp-html ${INPUT}" >&2
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 {marp-html|marp-pdf} [input.md]" >&2
    exit 1
    ;;
esac
