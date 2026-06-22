#!/usr/bin/env python3
"""Build glissile-loop cases: (001) and (111) planes × mobility law.

Geometry is rotated from tests/shearloopBCC_LinCS.data.  Each case lives in
<plane>_<mobility>/ with glissile_loop.data, .ctrl, and submit scripts.

Usage (from repo root):
    python examples/1_glissile_loops/make_loop_data.py
    python examples/1_glissile_loops/make_loop_data.py --case 001_BCC_glide
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DATA = os.path.join(REPO_ROOT, "tests", "shearloopBCC_LinCS.data")
BOX_HALF = 2000.0
PLANE_NORMAL_OLD = np.array([1.0, -1.0, 0.0])

CASES = (
    "001_BCC_Linear",
    "111_BCC_Linear",
    "001_BCC_glide",
    "111_BCC_glide",
    "001_FCC",
    "111_FCC",
)

PLANE_CONFIGS = {
    "001": {"normal": np.array([0.0, 0.0, 1.0]), "label": "(001)"},
    "111": {"normal": np.array([1.0, 1.0, 1.0]), "label": "(111)"},
}

MOBILITY_CONFIGS = {
    "BCC_Linear": {
        "label": "BCC_Linear",
        "mobilityLaw": "BCC_Linear",
        "MobScrew": "1.000000e+00",
        "MobEdge": "1.000000e+02",
        "MobClimb": "1.000000e-02",
        "rc": "2.000000",
        "enableCrossSlip": 0,
        "enforceGlidePlanes": 1,
    },
    "BCC_glide": {
        "label": "BCC_glide",
        "mobilityLaw": "BCC_glide",
        "MobScrew": "1.000000e+00",
        "MobEdge": "1.000000e+02",
        "MobClimb": "1.000000e-03",
        "MobClimbByPlane": {
            "001": "1.000000e-03",
            "111": "3.000000e-03",
        },
        "rc": "2.000000",
        "enableCrossSlip": 0,
        "enforceGlidePlanes": 1,
    },
    "FCC": {
        "label": "FCC_linear",
        "mobilityLaw": "FCC_linear",
        "MobScrew": "5.000000e+04",
        "MobEdge": "5.000000e+04",
        "MobClimb": "1.000000e+00",
        "rc": "5.000000",
        "enableCrossSlip": 0,
        "enforceGlidePlanes": 1,
    },
}

STRESS_VOIGT = {
    "001": [
        0.000000e00,
        0.000000e00,
        0.000000e00,
        6.522095e07,
        1.119015e07,
        0.000000e00,
    ],
    "111": [
        -5.403080e07,
        5.403080e07,
        -1.676381e-08,
        2.701540e07,
        -2.701540e07,
        -3.422610e-08,
    ],
}

PRIMARY_RE = re.compile(
    r"^\s+0,(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+(\d+)\s+(\d+)"
)
ARM_RE = re.compile(
    r"^\s+0,(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$"
)
NORMAL_RE = re.compile(
    r"^\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$"
)

SUBMIT_CPU = """\
#!/bin/bash
#SBATCH -J paradis-gloop-{case_slug}-cpu
#SBATCH -o bash_logs/glissile_loop_cpu.%j.out
#SBATCH -e bash_logs/glissile_loop_cpu.%j.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH -t 02:00:00

set -euo pipefail

mkdir -p bash_logs

module purge
module load gnu12/12.3.0
module load openmpi4/4.1.6

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE="$(basename "${{SCRIPT_DIR}}")"
CASE_REL="examples/1_glissile_loops/${{CASE}}"

if [[ -n "${{SLURM_SUBMIT_DIR:-}}" && -f "${{SLURM_SUBMIT_DIR}}/glissile_loop.ctrl" ]]; then
    SCRIPT_DIR="${{SLURM_SUBMIT_DIR}}"
    CASE="$(basename "${{SCRIPT_DIR}}")"
    CASE_REL="examples/1_glissile_loops/${{CASE}}"
    REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../../.." && pwd)"
elif [[ -f "${{SCRIPT_DIR}}/glissile_loop.ctrl" ]]; then
    REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../../.." && pwd)"
else
    echo "ERROR: cannot locate glissile loop case directory" >&2
    exit 1
fi
cd "$REPO_ROOT"

EXE="${{REPO_ROOT}}/bin/paradis"
DAT="${{CASE_REL}}/glissile_loop.data"
CTL="${{CASE_REL}}/glissile_loop.ctrl"
LOG="${{CASE_REL}}/glissile_loop_cpu.log"
RESULTS="${{CASE_REL}}/glissile_loop_results"

NDOMS=8

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Repo: ${{REPO_ROOT}}"
echo "Case: ${{CASE_REL}}"

if [ ! -x "${{EXE}}" ]; then
    echo "Building ParaDiS (CPU) on ${{HOSTNAME}}..."
    mkdir -p "${{REPO_ROOT}}/obj/p" "${{REPO_ROOT}}/obj/s" "${{REPO_ROOT}}/bin"
    make SYS=linux
fi

if [ ! -x "${{EXE}}" ]; then
    echo "ERROR: ${{EXE}} not found after build"
    exit 1
fi

rm -rf "${{RESULTS}}" "${{LOG}}"

echo "Launching ${{NDOMS}} MPI tasks..."
export OMPI_MCA_hwloc_base_binding_policy=none
srun --cpu-bind=none -n "${{NDOMS}}" "${{EXE}}" -d "${{DAT}}" "${{CTL}}" | tee -a "${{LOG}}"

echo "Running visualization..."
python3 examples/utils/visualize.py --example-dir "${{REPO_ROOT}}/${{CASE_REL}}"

echo "Job finished: $(date)"
"""

SUBMIT_GPU = """\
#!/bin/bash
#SBATCH -J paradis-gloop-{case_slug}
#SBATCH -o bash_logs/glissile_loop.%j.out
#SBATCH -e bash_logs/glissile_loop.%j.err
#SBATCH -p gpu-L40S
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00

set -euo pipefail

mkdir -p bash_logs

module purge
module load gnu12/12.3.0
module load openmpi4/4.1.6
module load cuda/12.5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE="$(basename "${{SCRIPT_DIR}}")"
CASE_REL="examples/1_glissile_loops/${{CASE}}"

if [[ -n "${{SLURM_SUBMIT_DIR:-}}" && -f "${{SLURM_SUBMIT_DIR}}/glissile_loop.ctrl" ]]; then
    SCRIPT_DIR="${{SLURM_SUBMIT_DIR}}"
    CASE="$(basename "${{SCRIPT_DIR}}")"
    CASE_REL="examples/1_glissile_loops/${{CASE}}"
    REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../../.." && pwd)"
elif [[ -f "${{SCRIPT_DIR}}/glissile_loop.ctrl" ]]; then
    REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../../.." && pwd)"
else
    echo "ERROR: cannot locate glissile loop case directory" >&2
    exit 1
fi
cd "$REPO_ROOT"

EXE="${{REPO_ROOT}}/bin/paradis"
DAT="${{CASE_REL}}/glissile_loop.data"
CTL="${{CASE_REL}}/glissile_loop.ctrl"
LOG="${{CASE_REL}}/glissile_loop.log"
RESULTS="${{CASE_REL}}/glissile_loop_results"

NDOMS=8

export CUDA_PATH=/usr/local/cuda-12.5
export CUDA_LIBS=/usr/local/cuda-12.5/lib64
export NVCC="${{CUDA_PATH}}/bin/nvcc"
export NVCC_FLAGS="-O3 -g -rdc=true -Wno-deprecated-gpu-targets -gencode arch=compute_89,code=sm_89"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Repo: ${{REPO_ROOT}}"
echo "Case: ${{CASE_REL}}"

if [ ! -x "${{EXE}}" ]; then
    echo "Building ParaDiS with GPU support on ${{HOSTNAME}}..."
    mkdir -p "${{REPO_ROOT}}/obj/p" "${{REPO_ROOT}}/obj/s" "${{REPO_ROOT}}/bin"
    make SYS=linux GPU_ENABLED=ON
fi

if [ ! -x "${{EXE}}" ]; then
    echo "ERROR: ${{EXE}} not found after build"
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
else
    echo "WARNING: nvidia-smi not available on this node"
fi

rm -rf "${{RESULTS}}" "${{LOG}}" slurm*.out

echo "Launching ${{NDOMS}} MPI tasks..."
export OMPI_MCA_hwloc_base_binding_policy=none
srun --cpu-bind=none -n "${{NDOMS}}" "${{EXE}}" -d "${{DAT}}" "${{CTL}}" | tee -a "${{LOG}}"

echo "Running visualization..."
python3 examples/utils/visualize.py --example-dir "${{REPO_ROOT}}/${{CASE_REL}}"

echo "Job finished: $(date)"
"""

CLEAR_RERUN = """\
rm bash_logs/*
rm glissile_loop.log
rm glissile_loop_cpu.log
rm -r glissile_loop_results/*
rm output/*
"""


def parse_case(case_name: str):
    for plane in PLANE_CONFIGS:
        prefix = plane + "_"
        if not case_name.startswith(prefix):
            continue
        mobility = case_name[len(prefix):]
        if mobility not in MOBILITY_CONFIGS:
            break
        return plane, mobility
    raise SystemExit("Unknown case {!r}".format(case_name))


def rotation_matrix_from_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    a = np.asarray(vec_from, dtype=float)
    b = np.asarray(vec_to, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3) if c > 0.0 else -np.eye(3)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def parse_shearloop(path: str):
    nodes = []
    in_nodal = False

    with open(path) as fh:
        for line in fh:
            if line.strip().startswith("nodalData"):
                in_nodal = True
                continue
            if not in_nodal:
                continue

            primary = PRIMARY_RE.match(line)
            if primary:
                nodes.append(
                    {
                        "id": int(primary.group(1)),
                        "pos": np.array(
                            [float(primary.group(2)), float(primary.group(3)),
                             float(primary.group(4))]
                        ),
                        "n_arms": int(primary.group(5)),
                        "constraint": int(primary.group(6)),
                        "arms": [],
                    }
                )
                continue

            arm = ARM_RE.match(line)
            if arm and nodes:
                burg = np.array(
                    [float(arm.group(2)), float(arm.group(3)), float(arm.group(4))]
                )
                nodes[-1]["arms"].append(
                    {"nbr": int(arm.group(1)), "burg": burg, "normal": None}
                )
                continue

            normal = NORMAL_RE.match(line)
            if normal and nodes and nodes[-1]["arms"]:
                nvec = np.array(
                    [float(normal.group(1)), float(normal.group(2)), float(normal.group(3))]
                )
                nodes[-1]["arms"][-1]["normal"] = nvec

    return nodes


def transform_nodes(nodes, rotation: np.ndarray):
    for node in nodes:
        node["pos"] = rotation @ node["pos"]
        for arm in node["arms"]:
            arm["burg"] = rotation @ arm["burg"]
            arm["normal"] = rotation @ arm["normal"]
            arm["normal"] /= np.linalg.norm(arm["normal"])


def build_data_text(plane_key: str, nodes) -> str:
    label = PLANE_CONFIGS[plane_key]["label"]
    lines = [
        "#",
        "#  Glissile loop on the {} plane (from shearloopBCC_LinCS.data).".format(label),
        "#",
        "dataFileVersion =   4",
        "numFileSegments =   1",
        "minCoordinates = [",
        "  -{:.1f}".format(BOX_HALF),
        "  -{:.1f}".format(BOX_HALF),
        "  -{:.1f}".format(BOX_HALF),
        "  ]",
        "maxCoordinates = [",
        "   {:.1f}".format(BOX_HALF),
        "   {:.1f}".format(BOX_HALF),
        "   {:.1f}".format(BOX_HALF),
        "  ]",
        "nodeCount =   {}".format(len(nodes)),
        "dataDecompType =   1",
        "dataDecompGeometry = [",
        "  1",
        "  1",
        "  1",
        "  ]",
        "",
        "#",
        "#  END OF DATA FILE PARAMETERS",
        "#",
        "",
        "domainDecomposition = ",
        " -{:.1f}".format(BOX_HALF),
        "     -{:.1f}".format(BOX_HALF),
        "         -{:.1f}".format(BOX_HALF),
        "          {:.1f}".format(BOX_HALF),
        "      {:.1f}".format(BOX_HALF),
        "  {:.1f}".format(BOX_HALF),
        "",
        "nodalData = ",
        "#  Primary lines: node_tag, x, y, z, num_arms, constraint",
        "#  Secondary lines: arm_tag, burgx, burgy, burgz, nx, ny, nz",
    ]

    for node in nodes:
        pos = node["pos"]
        lines.append(
            "     0,{:<4d} {:16.4f} {:16.4f} {:16.4f} {}   {}".format(
                node["id"], pos[0], pos[1], pos[2], node["n_arms"], node["constraint"]
            )
        )
        for arm in node["arms"]:
            burg = arm["burg"]
            normal = arm["normal"]
            lines.append(
                "           0,{:<4d} {:18.10f} {:18.10f} {:18.10f}".format(
                    arm["nbr"], burg[0], burg[1], burg[2]
                )
            )
            lines.append(
                "                    {:18.10f} {:18.10f} {:18.10f}".format(
                    normal[0], normal[1], normal[2]
                )
            )

    return "\n".join(lines) + "\n"


def build_plane_nodes(plane_key: str):
    rotation = rotation_matrix_from_vectors(
        PLANE_NORMAL_OLD, PLANE_CONFIGS[plane_key]["normal"]
    )
    nodes = parse_shearloop(SRC_DATA)
    transform_nodes(nodes, rotation)
    return nodes


def write_ctrl(case_name: str):
    plane, mobility_key = parse_case(case_name)
    mob = MOBILITY_CONFIGS[mobility_key]
    stress = STRESS_VOIGT[plane]
    case_rel = "examples/1_glissile_loops/{}".format(case_name)
    plane_label = PLANE_CONFIGS[plane]["label"]

    header = [
        "#",
        "#  Glissile loop on {} with {} mobility.".format(plane_label, mob["label"]),
    ]
    lines = header + [
        "#",
        "#  Run from the ParaDiS repository root:",
        "#    python examples/1_glissile_loops/make_loop_data.py --case {}".format(case_name),
        "#    mpirun -n 8 ./bin/paradis -d {}/glissile_loop.data \\".format(case_rel),
        "#                           {}/glissile_loop.ctrl".format(case_rel),
        "#",
        'dirname = "{}/glissile_loop_results"'.format(case_rel),
        "",
        "numXdoms = 2",
        "numYdoms = 2",
        "numZdoms = 2",
        "",
        "numXcells = 4",
        "numYcells = 4",
        "numZcells = 4",
        "",
        "fmEnabled = 0",
        "",
        "maxstep = 4000",
        "remeshRule = 2",
        "minSeg = 10.000000",
        "maxSeg = 200.000000",
        "rTol = 1.000000e+00",
        'timestepIntegrator = "trapezoid"',
        "maxDT = 1.000000e-05",
        "",
        "enforceGlidePlanes = {}".format(mob.get("enforceGlidePlanes", 1)),
        "enableCrossSlip = {}".format(mob["enableCrossSlip"]),
        "",
        "rc = {}".format(mob["rc"]),
        "MobScrew = {}".format(mob["MobScrew"]),
        "MobEdge = {}".format(mob["MobEdge"]),
        "MobClimb = {}".format(
            mob.get("MobClimbByPlane", {}).get(plane, mob["MobClimb"])
        ),
        'mobilityLaw = "{}"'.format(mob["mobilityLaw"]),
        "",
        "elasticinteraction = 1",
        "",
    ]
    if mob.get("loadType", 0) == 1:
        lines.extend([
            "loadType = 1",
            "eRate = {}".format(mob["eRate"]),
            "edotdir = [",
            "  0.0",
            "  0.0",
            "  1.0",
            "  ]",
            "",
        ])
    else:
        lines.extend([
            "loadType = 0",
            "edotdir = [",
            "  0.0",
            "  0.0",
            "  1.0",
            "  ]",
            "#  Voigt (xx, yy, zz, yz, zx, xy): rotated from shearloopBCC_LinCS.ctrl.",
            "appliedStress = [",
        ])
        for val in stress:
            lines.append("  {:.6e}".format(val))
        lines.append("  ]")
        lines.append("")
    lines.extend([
        "savetimers = 0",
        "savecn = 1",
        "savecnfreq = 200",
        "savecncounter = 0",
        "gnuplot = 1",
        "gnuplotfreq = 200",
        "gnuplotcounter = 0",
        "povray = 0",
        "povrayfreq = 200",
        "povraycounter = 0",
        "saveprop = 1",
        "savepropfreq = 10",
        "",
    ])
    return "\n".join(lines)


def write_case_scripts(case_name: str, case_dir: str):
    slug = case_name.lower().replace("_", "-")
    for name, template in (("submit_cpu.sh", SUBMIT_CPU), ("submit_gpu.sh", SUBMIT_GPU)):
        path = os.path.join(case_dir, name)
        with open(path, "w") as fh:
            fh.write(template.format(case_slug=slug))
        os.chmod(path, 0o755)

    clear_path = os.path.join(case_dir, "clear_rerun.sh")
    with open(clear_path, "w") as fh:
        fh.write(CLEAR_RERUN)
    os.chmod(clear_path, 0o755)


def setup_case(case_name: str, plane_data_cache: dict):
    plane, _mobility = parse_case(case_name)
    case_dir = os.path.join(SCRIPT_DIR, case_name)
    os.makedirs(case_dir, exist_ok=True)

    data_path = os.path.join(case_dir, "glissile_loop.data")
    with open(data_path, "w") as fh:
        fh.write(plane_data_cache[plane])

    ctrl_path = os.path.join(case_dir, "glissile_loop.ctrl")
    with open(ctrl_path, "w") as fh:
        fh.write(write_ctrl(case_name))

    write_case_scripts(case_name, case_dir)
    print("Setup {}".format(case_name))


def main():
    parser = argparse.ArgumentParser(description="Generate glissile loop case files")
    parser.add_argument(
        "--case",
        choices=list(CASES) + ["all"],
        default="all",
        help="Case to (re)generate (default: all)",
    )
    args = parser.parse_args()
    selected = list(CASES) if args.case == "all" else [args.case]

    plane_data_cache = {}
    for plane in PLANE_CONFIGS:
        plane_data_cache[plane] = build_data_text(plane, build_plane_nodes(plane))

    for case_name in selected:
        setup_case(case_name, plane_data_cache)


if __name__ == "__main__":
    main()
