#!/usr/bin/env bash
# One-shot run on a Colab GPU runtime (or any machine with nvcc and an NVIDIA GPU).
#   !bash colab_run.sh                 # run from the mmaprobe/ directory
# Produces run_<config>_<arch>/{ledger.jsonl,report.json}. Commit ledger.jsonl BEFORE the
# held-out step if you want an outside reader to trust the order of events (see README).
set -uo pipefail

nvidia-smi -L || { echo "no GPU visible"; exit 1; }
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${CAP}"
echo "arch: ${ARCH}"
nvcc --version | tail -2

FP8=1
if ! nvcc -O2 -arch="${ARCH}" -o probe_mma cuda/probe_mma.cu 2> build.log; then
  echo "build failed; retrying without the FP8 path (needs nvcc >= 12.4 and sm_89+)"
  cat build.log
  FP8=0
  nvcc -O2 -arch="${ARCH}" -DNO_FP8 -o probe_mma cuda/probe_mma.cu || exit 1
fi

./probe_mma --info

CONFIGS="f16_k8"
if [ "${CAP}" -ge 89 ] && [ "${FP8}" = 1 ]; then CONFIGS="e4m3_k32 f16_k8"; fi

for cfg in ${CONFIGS}; do
  echo "=== ${cfg} on ${ARCH} ==="
  python3 -m mmaprobe run --config "${cfg}" --device cuda --binary ./probe_mma \
      --out "run_${cfg}_${ARCH}" --seed "${SEED:-1234}" --n-heldout "${N_HELDOUT:-20000}" || \
      echo "run for ${cfg} exited non-zero (a failed layout self-test is the usual cause)"
done
