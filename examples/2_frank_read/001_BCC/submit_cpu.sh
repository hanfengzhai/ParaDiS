#!/bin/bash
#SBATCH -J paradis-frank-001-bcc-cpu
#SBATCH -o bash_logs/frank_read_cpu.%j.out
#SBATCH -e bash_logs/frank_read_cpu.%j.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH -t 01:00:00

set -euo pipefail

mkdir -p bash_logs

module purge
module load gnu12/12.3.0
module load openmpi4/4.1.6

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE="$(basename "${SCRIPT_DIR}")"
CASE_REL="examples/2_frank_read/${CASE}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/frank_read.ctrl" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    CASE="$(basename "${SCRIPT_DIR}")"
    CASE_REL="examples/2_frank_read/${CASE}"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/frank_read.ctrl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
    echo "ERROR: cannot locate Frank-Read case directory" >&2
    exit 1
fi
cd "$REPO_ROOT"

EXE="${REPO_ROOT}/bin/paradis"
DAT="${CASE_REL}/frank_read.data"
CTL="${CASE_REL}/frank_read.ctrl"
LOG="${CASE_REL}/frank_read_cpu.log"
RESULTS="${CASE_REL}/frank_read_results"

NDOMS=8

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Repo: ${REPO_ROOT}"
echo "Case: ${CASE_REL}"

if [ ! -x "${EXE}" ]; then
    echo "Building ParaDiS (CPU) on ${HOSTNAME}..."
    mkdir -p "${REPO_ROOT}/obj/p" "${REPO_ROOT}/obj/s" "${REPO_ROOT}/bin"
    make SYS=linux
fi

if [ ! -x "${EXE}" ]; then
    echo "ERROR: ${EXE} not found after build"
    exit 1
fi

rm -rf "${RESULTS}" "${LOG}"

echo "Launching ${NDOMS} MPI tasks..."
export OMPI_MCA_hwloc_base_binding_policy=none
srun --cpu-bind=none -n "${NDOMS}" "${EXE}" -d "${DAT}" "${CTL}" | tee -a "${LOG}"

echo "Running visualization..."
python3 examples/utils/visualize.py --example-dir "${REPO_ROOT}/${CASE_REL}"

echo "Job finished: $(date)"
