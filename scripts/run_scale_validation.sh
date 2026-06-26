#!/usr/bin/env bash
# Scale-validation: does counter+RMS keep parity with AdamW as the model grows, and how does
# its training-peak memory compare to memory-efficient baselines? This is the question the
# evidence is currently weakest on (micro/tiny only). Run it on a CUDA GPU.
#
#   ./run_scale_validation.sh [config] [steps] [device]
#   ./run_scale_validation.sh s512 2000 cuda
#
# Saves logs under results/ next to this package. On CPU it still runs (smaller configs) but
# without a real memory peak -- use a GPU for the numbers that matter.
set -u

here="$(cd "$(dirname "$0")" && pwd)"
pkg="$(cd "$here/.." && pwd)"
out="$pkg/results"
mkdir -p "$out"

config="${1:-s512}"
steps="${2:-2000}"
device="${3:-cuda}"
data="${DATA_PATH:-}"
data_arg=()
[ -n "$data" ] && data_arg=(--data-path "$data")

stamp="${config}_${device}"
echo "scale-validation: config=$config steps=$steps device=$device -> $out/parity_${stamp}.log"

# Parity across kinds, and the dense baseline under memory-efficient optimizers.
memory-native-charlm --config "$config" --steps "$steps" --device "$device" \
  --kinds dense,qat,counter,counter_rms "${data_arg[@]}" \
  2>&1 | tee "$out/parity_${stamp}.log"

for opt in adamw galore lomo bnb8; do
  echo "--- dense + $opt ---"
  memory-native-charlm --config "$config" --steps "$steps" --device "$device" \
    --kinds dense --optimizer "$opt" "${data_arg[@]}" \
    2>&1 | tee "$out/dense_${opt}_${stamp}.log"
done

# Training-peak memory: counter_rms vs dense across optimizers (real peak on CUDA).
memory-native-memgate --config "$config" --device "$device" \
  --optimizers adamw,galore,lomo,bnb8 \
  2>&1 | tee "$out/memgate_${stamp}.log"

echo "done. Commit the results/*.log files and note the GPU/driver used."
