#!/bin/bash
#SBATCH --job-name=mdip_radial_cine
#SBATCH -t 0-04:00
#SBATCH -p batch
#SBATCH -q batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH -o slurm-out/%j.out
#SBATCH -e slurm-out/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=evelyngcuevaj@gmail.com

set -euo pipefail

module load conda
ENV_NAME="${ENV_NAME:-m-dip}"
conda activate "${ENV_NAME}"

REPO_ROOT="${REPO_ROOT:-/home/egcuevaj/repositories/cine-MRI-models}"
SCRIPT="${REPO_ROOT}/M-DIP-AA.py"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATA_NAME="${DATA_NAME:-AA}"
RAW_FOLDER="${RAW_FOLDER:-/mnt/researchers/claudia-prieto/datasets/pulseqCINE/DATA_0.55T/${DATA_NAME}/traindata}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/${DATA_NAME}_mdip_radial}"

# -------------------------
# Fixed parameters
# -------------------------
N_ITER="${N_ITER:-1000}"
BATCH_SIZE="${BATCH_SIZE:-96}"
SAVE_EVERY="${SAVE_EVERY:-100}"
MONITOR_EVERY="${MONITOR_EVERY:-50}"
CUDA_NUM="${CUDA_NUM:-0}"
NO_FLOW="${NO_FLOW:-0}"
ACTIVATE_FLOW_AFTER="${ACTIVATE_FLOW_AFTER:-0}"
KSP_SCALE="${KSP_SCALE:-100}"
RADIAL_OPERATOR="${RADIAL_OPERATOR:-grid}"

ZS_CHANS="${ZS_CHANS:-2}"
ZT_CHANS="${ZT_CHANS:-4}"
ZT_INIT="${ZT_INIT:-periodic}"
P_DROPOUT="${P_DROPOUT:-0}"
NOISE_REG="${NOISE_REG:-0.05}"
LR_MIN="${LR_MIN:-1e-6}"
LR_STATIC_FACTOR="${LR_STATIC_FACTOR:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
LAMBDA_ZT="${LAMBDA_ZT:-0}"
LAMBDA_BASIS="${LAMBDA_BASIS:-0}"
LAMBDA_SUPPORT="${LAMBDA_SUPPORT:-0}"

# -------------------------
# Grid parameters
# -------------------------
FILENAMES=(
  "slice_1_8_nbins30.npz"
)
LEARNING_RATES=(
  "1e-3"
)
LAMBDA_FLOW_SPATIALS=(
  "0.10"
)
LAMBDA_FLOW_TEMPORALS=(
  "0.05"
)
N_BASES_LIST=(
  "4"
)

mkdir -p "${RESULTS_ROOT}"
mkdir -p "${REPO_ROOT}/slurm-out"

echo "Repository: ${REPO_ROOT}"
echo "Script:     ${SCRIPT}"
echo "Raw folder: ${RAW_FOLDER}"
echo "Results:    ${RESULTS_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

EXP_ID="${EXP_START:-2}"

for FILENAME in "${FILENAMES[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    for LAMBDA_FLOW_SPATIAL in "${LAMBDA_FLOW_SPATIALS[@]}"; do
      for LAMBDA_FLOW_TEMPORAL in "${LAMBDA_FLOW_TEMPORALS[@]}"; do
        for N_BASES in "${N_BASES_LIST[@]}"; do

          EXP_ID=$((EXP_ID + 1))
          EXP_NAME=$(printf "exp_%03d" "${EXP_ID}")
          EXP_DIR="${RESULTS_ROOT}/${EXP_NAME}"
          mkdir -p "${EXP_DIR}"

          echo "==============================================="
          echo "Running ${EXP_NAME}"
          echo "Filename:              ${FILENAME}"
          echo "Learning rate:         ${LR}"
          echo "N bases:               ${N_BASES}"
          echo "Lambda flow spatial:   ${LAMBDA_FLOW_SPATIAL}"
          echo "Lambda flow temporal:  ${LAMBDA_FLOW_TEMPORAL}"
          echo "Iterations:            ${N_ITER}"
          echo "Output dir:            ${EXP_DIR}"
          echo "==============================================="

          EXTRA_ARGS=()
          if [ "${NO_FLOW}" -eq 1 ]; then
            EXTRA_ARGS+=(--no-flow)
          fi

          cat > "${EXP_DIR}/config_run.txt" <<EOF
DATA_NAME=${DATA_NAME}
RAW_FOLDER=${RAW_FOLDER}
SCRIPT=${SCRIPT}
FILENAME=${FILENAME}

LEARNING_RATE=${LR}
LR_MIN=${LR_MIN}
LR_STATIC_FACTOR=${LR_STATIC_FACTOR}
WEIGHT_DECAY=${WEIGHT_DECAY}
N_ITER=${N_ITER}
BATCH_SIZE=${BATCH_SIZE}
SAVE_EVERY=${SAVE_EVERY}
MONITOR_EVERY=${MONITOR_EVERY}
KSP_SCALE=${KSP_SCALE}
RADIAL_OPERATOR=${RADIAL_OPERATOR}
NO_FLOW=${NO_FLOW}
ACTIVATE_FLOW_AFTER=${ACTIVATE_FLOW_AFTER}

N_BASES=${N_BASES}
ZS_CHANS=${ZS_CHANS}
ZT_CHANS=${ZT_CHANS}
ZT_INIT=${ZT_INIT}
P_DROPOUT=${P_DROPOUT}
NOISE_REG=${NOISE_REG}
LAMBDA_FLOW_SPATIAL=${LAMBDA_FLOW_SPATIAL}
LAMBDA_FLOW_TEMPORAL=${LAMBDA_FLOW_TEMPORAL}
LAMBDA_ZT=${LAMBDA_ZT}
LAMBDA_BASIS=${LAMBDA_BASIS}
LAMBDA_SUPPORT=${LAMBDA_SUPPORT}
EOF

          python "${SCRIPT}" \
            --raw-folder "${RAW_FOLDER}" \
            --filename "${FILENAME}" \
            --out-folder "${EXP_DIR}" \
            --n-iter "${N_ITER}" \
            --batch-size "${BATCH_SIZE}" \
            --save-every "${SAVE_EVERY}" \
            --monitor-every "${MONITOR_EVERY}" \
            --cuda-num "${CUDA_NUM}" \
            --lr-max "${LR}" \
            --lr-min "${LR_MIN}" \
            --lr-static-factor "${LR_STATIC_FACTOR}" \
            --weight-decay "${WEIGHT_DECAY}" \
            --n-bases "${N_BASES}" \
            --zs-chans "${ZS_CHANS}" \
            --zt-chans "${ZT_CHANS}" \
            --zt-init "${ZT_INIT}" \
            --p-dropout "${P_DROPOUT}" \
            --noise-reg "${NOISE_REG}" \
            --lambda-flow-spatial "${LAMBDA_FLOW_SPATIAL}" \
            --lambda-flow-temporal "${LAMBDA_FLOW_TEMPORAL}" \
            --lambda-zt "${LAMBDA_ZT}" \
            --lambda-basis "${LAMBDA_BASIS}" \
            --lambda-support "${LAMBDA_SUPPORT}" \
            --ksp-scale "${KSP_SCALE}" \
            --radial-operator "${RADIAL_OPERATOR}" \
            --activate-flow-after "${ACTIVATE_FLOW_AFTER}" \
            "${EXTRA_ARGS[@]}"

        done
      done
    done
  done
done

echo "All experiments finished."
