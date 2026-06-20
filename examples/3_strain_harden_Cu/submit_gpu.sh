#!/bin/bash
#SBATCH -J paradis-strain-harden-cu
#SBATCH -o bash_logs/strain_harden_Cu.%j.out
#SBATCH -e bash_logs/strain_harden_Cu.%j.err
#SBATCH -p gpu-ampere
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00

set -euo pipefail

mkdir -p bash_logs

module purge
module load gnu12/12.3.0
module load openmpi4/4.1.6
module load cuda/12.5

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/strain_harden_Cu.ctrl" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/3_strain_harden_Cu/strain_harden_Cu.ctrl" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -f "$(dirname "$0")/strain_harden_Cu.ctrl" ]]; then
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
else
    echo "ERROR: cannot locate ParaDiS repository root" >&2
    echo "Submit from examples/3_strain_harden_Cu:  cd examples/3_strain_harden_Cu && sbatch submit_gpu.sh" >&2
    exit 1
fi
cd "$REPO_ROOT"

EXE="${REPO_ROOT}/bin/paradis"
DAT="examples/3_strain_harden_Cu/180chains_16.10e.data"
CTL="examples/3_strain_harden_Cu/strain_harden_Cu.ctrl"
LOG="examples/3_strain_harden_Cu/strain_harden_Cu.log"
RESULTS="examples/3_strain_harden_Cu/strain_harden_Cu_results"

NDOMS=8

export CUDA_PATH=/usr/local/cuda-12.5
export CUDA_LIBS=/usr/local/cuda-12.5/lib64
export NVCC="${CUDA_PATH}/bin/nvcc"
export NVCC_FLAGS="-O3 -g -rdc=true -Wno-deprecated-gpu-targets -gencode arch=compute_80,code=sm_80"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Repo: ${REPO_ROOT}"

if [ ! -x "${EXE}" ]; then
    echo "Building ParaDiS with GPU support on ${HOSTNAME}..."
    mkdir -p "${REPO_ROOT}/obj/p" "${REPO_ROOT}/obj/s" "${REPO_ROOT}/bin"
    make SYS=linux GPU_ENABLED=ON
fi

if [ ! -x "${EXE}" ]; then
    echo "ERROR: ${EXE} not found after build"
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
else
    echo "WARNING: nvidia-smi not available on this node"
fi

rm -rf "${RESULTS}" "${LOG}" slurm*.out

echo "Launching ${NDOMS} MPI tasks..."
export OMPI_MCA_hwloc_base_binding_policy=none
srun --cpu-bind=none -n "${NDOMS}" "${EXE}" -d "${DAT}" "${CTL}" | tee -a "${LOG}"

echo "Job finished: $(date)"
