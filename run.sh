#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh - IFC(BIM) -> textured FBX + RGB LAS/LAZ point cloud converter
#
#   ./run.sh                     convert every IFC in input/ with defaults
#   ./run.sh --spacing 0.03      any convert_ifc_to_las.py option is passed through
#   ./run.sh --no-fx --no-fbx    (see "Key Options" in README.md)
#   ./run.sh viewer              web viewer only, no conversion (default port 5013)
#   ./run.sh textures            pre-download textures only
#
# Python is auto-detected from the venv_lmm conda env. To use another one:
#   PYTHON=/path/to/python ./run.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PYTHON:-}" ]; then
    for cand in \
        "/c/ProgramData/miniconda3/envs/venv_lmm/python.exe" \
        "C:/ProgramData/miniconda3/envs/venv_lmm/python.exe" \
        "$HOME/miniconda3/envs/venv_lmm/bin/python" \
        "$HOME/anaconda3/envs/venv_lmm/bin/python"
    do
        if [ -x "$cand" ]; then PYTHON="$cand"; break; fi
    done
fi
PYTHON="${PYTHON:-python}"

if ! command -v "$PYTHON" >/dev/null 2>&1 && [ ! -x "$PYTHON" ]; then
    echo "python not found: $PYTHON" >&2
    echo "Specify one with: PYTHON=/path/to/python ./run.sh" >&2
    exit 1
fi

case "${1:-}" in
    viewer)
        shift
        exec "$PYTHON" webviewer.py -o ./output -c ./config.json "$@"
        ;;
    textures)
        shift
        exec "$PYTHON" texture_manager.py -c ./config.json -t ./textures "$@"
        ;;
esac

# No arguments -> default conversion. Otherwise pass everything through.
if [ "$#" -eq 0 ]; then
    set -- -i ./input -o ./output -c ./config.json
fi

exec "$PYTHON" convert_ifc_to_las.py "$@"
