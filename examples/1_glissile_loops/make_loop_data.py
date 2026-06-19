#!/usr/bin/env python3
"""Build a closed glissile loop on the (001) plane from the proven shearloop test.

The reference loop in tests/shearloopBCC_LinCS.data lies in a {110} glide plane
(normal (1,-1,0)).  We rotate positions, Burgers vectors, and glide normals so
the loop plane becomes z = 0 ((001)), then write glissile_loop.data.
"""

from __future__ import annotations

import os
import re

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DATA = os.path.join(REPO_ROOT, "tests", "shearloopBCC_LinCS.data")
OUT_PATH = os.path.join(SCRIPT_DIR, "glissile_loop.data")

BOX_HALF = 2000.0
PLANE_NORMAL_OLD = np.array([1.0, -1.0, 0.0])
PLANE_NORMAL_NEW = np.array([0.0, 0.0, 1.0])

PRIMARY_RE = re.compile(
    r"^\s+0,(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+(\d+)\s+(\d+)"
)
ARM_RE = re.compile(
    r"^\s+0,(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$"
)
NORMAL_RE = re.compile(
    r"^\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$"
)


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


def voigt_to_tensor(voigt):
    return np.array(
        [
            [voigt[0], voigt[5], voigt[4]],
            [voigt[5], voigt[1], voigt[3]],
            [voigt[4], voigt[3], voigt[2]],
        ],
        dtype=float,
    )


def tensor_to_voigt(tensor):
    return [
        tensor[0, 0],
        tensor[1, 1],
        tensor[2, 2],
        tensor[1, 2],
        tensor[0, 2],
        tensor[0, 1],
    ]


def rotated_applied_stress(rotation: np.ndarray) -> list[float]:
    """Voigt stress from tests/shearloopBCC_LinCS.ctrl, rotated into (001) frame."""
    voigt = [5.40308e7, -5.40308e7, 0.0, -2.70154e7, 2.70154e7, 0.0]
    sigma = voigt_to_tensor(voigt)
    sigma_rot = rotation @ sigma @ rotation.T
    return tensor_to_voigt(sigma_rot)


def parse_shearloop(path: str):
    header_lines = []
    nodes = []
    in_nodal = False

    with open(path) as fh:
        for line in fh:
            if line.strip().startswith("nodalData"):
                header_lines.append(line.rstrip("\n"))
                in_nodal = True
                continue

            if not in_nodal:
                header_lines.append(line.rstrip("\n"))
                continue

            primary = PRIMARY_RE.match(line)
            if primary:
                node_id = int(primary.group(1))
                pos = np.array(
                    [float(primary.group(2)), float(primary.group(3)), float(primary.group(4))]
                )
                n_arms = int(primary.group(5))
                constraint = int(primary.group(6))
                nodes.append(
                    {
                        "id": node_id,
                        "pos": pos,
                        "n_arms": n_arms,
                        "constraint": constraint,
                        "arms": [],
                    }
                )
                continue

            arm = ARM_RE.match(line)
            if arm and nodes:
                burg = np.array(
                    [float(arm.group(2)), float(arm.group(3)), float(arm.group(4))]
                )
                nodes[-1]["arms"].append({"nbr": int(arm.group(1)), "burg": burg, "normal": None})
                continue

            normal = NORMAL_RE.match(line)
            if normal and nodes and nodes[-1]["arms"]:
                nvec = np.array(
                    [float(normal.group(1)), float(normal.group(2)), float(normal.group(3))]
                )
                nodes[-1]["arms"][-1]["normal"] = nvec

    return header_lines, nodes


def _format_float(value: float) -> str:
    return "{:.14e}".format(value)


def transform_nodes(nodes, rotation: np.ndarray):
    for node in nodes:
        node["pos"] = rotation @ node["pos"]
        for arm in node["arms"]:
            arm["burg"] = rotation @ arm["burg"]
            arm["normal"] = rotation @ arm["normal"]
            arm["normal"] /= np.linalg.norm(arm["normal"])


def build_data_text(rotation: np.ndarray):
    _, nodes = parse_shearloop(SRC_DATA)
    transform_nodes(nodes, rotation)

    lines = [
        "#",
        "#  Closed BCC glissile loop on the (001) plane.",
        "#  Rotated from tests/shearloopBCC_LinCS.data ({110} glide plane).",
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

    return "\n".join(lines) + "\n", nodes


def main():
    rotation = rotation_matrix_from_vectors(PLANE_NORMAL_OLD, PLANE_NORMAL_NEW)
    text, nodes = build_data_text(rotation)

    positions = np.array([node["pos"] for node in nodes])
    z_span = positions[:, 2].max() - positions[:, 2].min()
    xy_radius = np.linalg.norm(positions[:, :2], axis=1).max()
    stress = rotated_applied_stress(rotation)

    with open(OUT_PATH, "w") as fh:
        fh.write(text)

    print("Wrote {} ({} nodes)".format(OUT_PATH, len(nodes)))
    print("Loop on (001): z span = {:.3e}, xy radius = {:.1f} b".format(z_span, xy_radius))
    print(
        "Rotated appliedStress (Voigt): "
        + ", ".join("{:.6e}".format(v) for v in stress)
    )


if __name__ == "__main__":
    main()
