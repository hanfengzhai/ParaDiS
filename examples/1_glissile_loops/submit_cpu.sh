#!/bin/bash
#SBATCH -J paradis-glissile-loop-cpu
#SBATCH -o bash_logs/glissile_loop_cpu.%j.out
#SBATCH -e bash_logs/glissile_loop_cpu.%j.err
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

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/glissile_loop.ctrl" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/1_glissile_loops/glissile_loop.ctrl" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -f "$(dirname "$0")/glissile_loop.ctrl" ]]; then
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
else
    echo "ERROR: cannot locate ParaDiS repository root" >&2
    echo "Submit from examples/1_glissile_loops:  cd examples/1_glissile_loops && sbatch submit_cpu.sh" >&2
    exit 1
fi
cd "$REPO_ROOT"

EXE="${REPO_ROOT}/bin/paradis"
DAT="examples/1_glissile_loops/glissile_loop.data"
CTL="examples/1_glissile_loops/glissile_loop.ctrl"
LOG="examples/1_glissile_loops/glissile_loop_cpu.log"
RESULTS="examples/1_glissile_loops/glissile_loop_results"

NDOMS=8

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Repo: ${REPO_ROOT}"

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
srun -n "${NDOMS}" "${EXE}" -d "${DAT}" "${CTL}" | tee -a "${LOG}"

echo "Job finished: $(date)"
