#!/bin/bash
#SBATCH --job-name=hbv_run
#SBATCH --account=project_2014823
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_2014823/hbv/logs/%j_%a.out
#SBATCH --error=/scratch/project_2014823/hbv/logs/%j_%a.err
# Array size set by API: sbatch --array=0-N hbv_run_puhti.sh

export MODULEPATH=/appl/modulefiles:$MODULEPATH
module load apptainer 2>/dev/null || true

SIF=/scratch/project_2014823/hbv/hbv-compute.sif
SCRATCH=/scratch/project_2014823/hbv
TOTAL=${SLURM_ARRAY_TASK_COUNT}
TASK=${SLURM_ARRAY_TASK_ID}

# Remap /data/hbv paths → Puhti scratch
remap() { echo "${1/\/data\/hbv/$SCRATCH}"; }

OUTPUT_DIR=$(remap "${HBV_OUTPUT_DIR}")
SHAPEFILE=$(remap "${HBV_SHAPEFILE}")
PRECIP_NC=$(remap "${HBV_PRECIP_NC}")
ET_NC=$(remap "${HBV_ET_NC}")
TEMP_NC=$(remap "${HBV_TEMP_NC}")
URBAN=$(remap "${HBV_URBAN_PATH}")
AGRI=$(remap "${HBV_AGRI_PATH}")
PARA=$(remap "${HBV_PARA_PATH}")

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH}/logs"

echo "[SLURM] job=${SLURM_JOB_ID} task=${TASK}/${TOTAL} user=${HBV_USER} started on $(hostname)"

echo "[DEBUG] SHAPEFILE=${SHAPEFILE} OUTPUT=${OUTPUT_DIR}"
apptainer exec --bind /scratch:/scratch "${SIF}" \
    python /app/hbv_worker.py \
    --job-id    "${HBV_JOB_ID}" \
    --task       "${TASK}" \
    --total      "${TOTAL}" \
    --shapefile  "${SHAPEFILE}" \
    --precip-nc  "${PRECIP_NC}" \
    --et-nc      "${ET_NC}" \
    --temp-nc    "${TEMP_NC}" \
    --urban-path "${URBAN}" \
    --agri-path  "${AGRI}" \
    --para-csv   "${PARA}" \
    --catchments "${HBV_CATCHMENT_IDS}" \
    --id-col     "${HBV_ID_COL:-TASO_ID}" \
    --output     "${OUTPUT_DIR}"

echo "[SLURM] task=${TASK} finished (exit $?)"
