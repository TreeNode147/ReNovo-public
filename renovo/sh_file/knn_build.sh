#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${RENOVO_PYTHON:-python}
export MKL_THREADING_LAYER=GNU PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/tmp"
mkdir -p "$root/tmp" "$root/logs/model" "$root/datastore" "$root/saved_model"
cd "$root"
exec "$python_bin" -m renovo build \
  --build_data="$1/train.mgf" --build_data="$1/valid.mgf" \
  --datastore_path="$root/datastore/${2}_db" \
  --model="$3" --config="$4" --output="$root/logs/model/$5"
