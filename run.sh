#!/bin/sh
set -eu
LAB_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORKSPACE_DIR=$(CDPATH= cd -- "$LAB_DIR/../.." && pwd)
if [ -n "${INFERENCE_LAB_PYTHON:-}" ]; then
    LAB_PYTHON="$INFERENCE_LAB_PYTHON"
elif [ -x "$LAB_DIR/.venv/bin/python" ]; then
    LAB_PYTHON="$LAB_DIR/.venv/bin/python"
elif [ -x "$WORKSPACE_DIR/work/recovery-venv/bin/python" ]; then
    LAB_PYTHON="$WORKSPACE_DIR/work/recovery-venv/bin/python"
else
    LAB_PYTHON=python3
fi
export PYTHONPATH="$LAB_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$LAB_PYTHON" -m inference_lab.cli "$@"
