#!/usr/bin/env bash
# day1, day2 配下の marimo notebook (notebook*.py) を Jupyter Notebook (.ipynb) に変換

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# notebook*.py を探索（.venv 配下は除外）
mapfile -t notebooks < <(
    find "$ROOT_DIR/day1" "$ROOT_DIR/day2" \
        -type d -name .venv -prune -o \
        -type f -name 'notebook*.py' -print | sort
)

if [ "${#notebooks[@]}" -eq 0 ]; then
    echo "No notebook*.py found under day1/ or day2/"
    exit 0
fi

for notebook in "${notebooks[@]}"; do
    dir="$(dirname "$notebook")"
    filename="$(basename "$notebook")"
    base="${filename%.py}"

    echo "==> Exporting: ${notebook#"$ROOT_DIR/"}"
    (cd "$dir" && uv run --with nbformat marimo export ipynb "$filename" -o "$base.ipynb")
done

echo "Done."
