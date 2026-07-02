#!/bin/bash
#SBATCH --job-name=hbv_run
#SBATCH --account=project_2014823
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_2014823/hbv/logs/%j_%a.out
#SBATCH --error=/scratch/project_2014823/hbv/logs/%j_%a.err
# Array size set by API: sbatch --array=0-N hbv_run_puhti.sh

module load apptainer

SIF=/scratch/project_2014823/hbv/hbv-compute.sif
TOTAL=${SLURM_ARRAY_TASK_COUNT}
TASK=${SLURM_ARRAY_TASK_ID}

mkdir -p "${HBV_OUTPUT_DIR}"
mkdir -p /scratch/project_2014823/hbv/logs

echo "[SLURM] job=${SLURM_JOB_ID} task=${TASK}/${TOTAL} user=${HBV_USER} started on $(hostname)"

apptainer exec "${SIF}" \
    python /app/hbv_worker.py \
    --job-id     "${HBV_JOB_ID}" \
    --task        "${TASK}" \
    --total       "${TOTAL}" \
    --output-dir  "${HBV_OUTPUT_DIR}" \
    --config      "${HBV_CONFIG_PATH}"

echo "[SLURM] task=${TASK} finished (exit $?)"
