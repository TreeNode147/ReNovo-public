#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 7 ]]; then
  echo 'Usage: knn_infer.sh <data> <datastore> <checkpoint> <config> <log-name> <k> <temperature>' >&2
  exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${RENOVO_PYTHON:-python}
export MKL_THREADING_LAYER=GNU PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/tmp"
mkdir -p "$root/tmp" "$root/logs/model" "$root/datastore" "$root/saved_model"
cd "$root"
exec "$python_bin" -m renovo evaluate \
  --annotated_peak_path="$1/test.mgf" --is_knn_inference --datastore_path="$2" \
  --model="$3" --config="$4" --output="$root/logs/model/$5" \
  --knn_k="$6" --knn_temperature="$7"
