#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${RENOVO_PYTHON:-python}
export MKL_THREADING_LAYER=GNU PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/tmp"
mkdir -p "$root/tmp" "$root/logs/model" "$root/datastore" "$root/saved_model"
cd "$root"
exec "$python_bin" -m renovo train \
  --train_peak_path="$root/dataset/nine_species/exc_$1/train.mgf" \
  --validation_peak_path="$root/dataset/nine_species/exc_$1/valid.mgf" \
  --config="$3" --output="$root/logs/model/$2"
