#!/usr/bin/env python3
"""Visualize ParaDiS Frank-Read gnuplot frames and properties files.

Writes PNG frames, summary plots, and MP4/MOV animation under output/.

Usage (from examples/frank_read/):
    python visualize.py
    python visualize.py --results frank_read_results --fps 2
"""

from __future__ import annotations

import argparse, os, re, sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from tqdm import tqdm

plotting_params = {"font.family": "serif", "font.serif": ["Libertinus Serif"],
                   "mathtext.fontset": "cm", "xtick.labelsize": 15, "ytick.labelsize": 15,
                   "axes.labelsize": 15, "legend.fontsize": 15, "axes.unicode_minus": False}
plt.rcParams.update(plotting_params)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(SCRIPT_DIR, "frank_read_results")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "output")
PROP_NAMES = ("time_Plastic_strain", "density", "alleps")
FRAME_GLOB_PREFIX = "0t"
FIG_DPI = 200
FONT_SIZE = 15
DISL_COLOR = "#1f4fd8"
BOX_COLOR = "#1a1a1a"
LINE_COLOR = "darkblue"
SPATIAL_AXIS_LABELS = (r"$X$ [$\mu$m]", r"$Y$ [$\mu$m]", r"$Z$ [$\mu$m]")
DEFAULT_BURGMAG_M = 2.8754e-10  # Ta BCC, matches frank_read.log


class FrankReadVisualizer:
    """Render dislocation frames and properties from a ParaDiS run."""

    def __init__(self, results_dir, output_dir, fps=2, dpi=FIG_DPI, burgmag_m=None):
        self.results_dir = os.path.abspath(results_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.gnu_dir = os.path.join(self.results_dir, "gnuplot")
        self.props_dir = os.path.join(self.results_dir, "properties")
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.fps = fps
        self.dpi = dpi
        self.burgmag_m = burgmag_m
        self.um_per_b = None
        self.box_limits = None

    def run(self):
        if not os.path.isdir(self.gnu_dir):
            raise SystemExit("gnuplot directory not found: {}".format(self.gnu_dir))
        box_path = os.path.join(self.gnu_dir, "box.in")
        box_edges = self._parse_box(box_path) if os.path.isfile(box_path) else np.zeros((0, 2, 3))
        self.um_per_b = self._resolve_um_per_b()
        box_edges = self._to_um(box_edges)
        self.box_limits = self._box_limits(box_edges)
        frame_files = self._list_gnuplot_frames()
        if not frame_files:
            raise SystemExit("No gnuplot frames found in {}".format(self.gnu_dir))
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        png_paths = self._render_frames(frame_files, box_edges)
        if os.path.isdir(self.props_dir):
            tqdm.write("Plotting properties...")
            self._plot_properties(self._load_properties())
        self._write_animations(png_paths)
        self._write_summary(frame_files, box_edges)
        tqdm.write("Done. Outputs in {}".format(self.output_dir))

    def _list_gnuplot_frames(self):
        names = sorted(n for n in os.listdir(self.gnu_dir)
                       if n.startswith(FRAME_GLOB_PREFIX) and not n.endswith(".final"))
        return [os.path.join(self.gnu_dir, n) for n in names]

    def _parse_box(self, path):
        pts = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        arr = np.asarray(pts, dtype=float)
        if len(arr) < 2:
            return np.zeros((0, 2, 3))
        return arr.reshape(-1, 2, 3)

    def _parse_gnuplot_frame(self, path):
        segments, pending = [], None
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                pt = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                if pending is None:
                    pending = pt
                else:
                    segments.append((pending, pt))
                    pending = None
        return segments

    def _resolve_um_per_b(self):
        if self.burgmag_m is not None:
            return float(self.burgmag_m) * 1.0e6
        log_candidates = [
            os.path.join(SCRIPT_DIR, "frank_read.log"),
            os.path.join(os.path.dirname(self.results_dir), "frank_read.log"),
            os.path.join(self.results_dir, "frank_read.log"),
        ]
        for path in log_candidates:
            if not os.path.isfile(path):
                continue
            with open(path) as fh:
                text = fh.read()
            match = re.search(r"burgmag\s*:\s*([\d.eE+-]+)", text)
            if match:
                return float(match.group(1)) * 1.0e6
            match = re.search(r"\(\s*(\d+)b,\s*([\d.]+)um\)", text)
            if match:
                return float(match.group(2)) / float(match.group(1))
        return DEFAULT_BURGMAG_M * 1.0e6

    def _to_um(self, coords):
        return np.asarray(coords, dtype=float) * self.um_per_b

    def _scale_segments(self, segments):
        scale = self.um_per_b
        return [(p1 * scale, p2 * scale) for p1, p2 in segments]

    def _box_limits(self, box_edges):
        if box_edges.size == 0:
            return None
        pts = box_edges.reshape(-1, 3)
        return pts.min(axis=0), pts.max(axis=0)

    def _apply_sim_box(self, ax):
        if self.box_limits is None:
            return
        mins, maxs = self.box_limits
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_box_aspect((1, 1, 1))

    def _draw_segments(self, ax, segments, box_edges):
        segments = self._scale_segments(segments)
        for p1, p2 in segments:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=DISL_COLOR, linewidth=1.5, alpha=0.95,
                    solid_capstyle="round")
        for edge in box_edges:
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2],
                    color=BOX_COLOR, linewidth=0.6, alpha=0.35)
        self._apply_sim_box(ax)
        ax.set_xlabel(SPATIAL_AXIS_LABELS[0], fontsize=FONT_SIZE)
        ax.set_ylabel(SPATIAL_AXIS_LABELS[1], fontsize=FONT_SIZE)
        ax.set_zlabel(SPATIAL_AXIS_LABELS[2], fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE)
        ax.view_init(elev=22, azim=-58)

    def _render_frame(self, segments, box_edges, title, out_path):
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        self._draw_segments(ax, segments, box_edges)
        ax.set_title(title, fontsize=FONT_SIZE)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _frame_label(self, path):
        name = os.path.basename(path)
        match = re.search(r"0t(\d+)", name)
        step = int(match.group(1)) if match else 0
        return "Frank-Read source, step {}".format(step)

    def _render_frames(self, frame_files, box_edges):
        png_paths = []
        for frame_path in tqdm(frame_files, desc="frames"):
            out_png = os.path.join(self.frames_dir, "{}.png".format(os.path.basename(frame_path)))
            self._render_frame(self._parse_gnuplot_frame(frame_path),
                               box_edges, self._frame_label(frame_path), out_png)
            png_paths.append(out_png)
            tqdm.write("  {}".format(os.path.basename(out_png)))
        return png_paths

    def _load_properties(self):
        data = {}
        for name in PROP_NAMES:
            path = os.path.join(self.props_dir, name)
            if os.path.isfile(path):
                data[name] = np.loadtxt(path)
        return data

    def _save_line_plot(self, x, y, xlabel, ylabel, title, out_name):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, y, "o-", color=LINE_COLOR, linewidth=2, markersize=4,
                markerfacecolor=LINE_COLOR, markeredgecolor="black", markeredgewidth=0.4)
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
        ax.set_title(title, fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, out_name), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_properties(self, props):
        if "time_Plastic_strain" in props:
            t_eps = props["time_Plastic_strain"]
            self._save_line_plot(t_eps[:, 0], t_eps[:, 1],
                                 "Simulation time (s)", "Plastic strain",
                                 "Plastic strain vs time", "plastic_strain_vs_time.png")
        if "density" not in props:
            return
        dens = props["density"]
        self._save_line_plot(dens[:, 1], dens[:, 2],
                             "Strain", r"Dislocation density ($\mathrm{m}^{-2}$)",
                             "Dislocation density vs strain", "density_vs_strain.png")
        if "time_Plastic_strain" in props:
            t_eps = props["time_Plastic_strain"]
            self._save_line_plot(t_eps[:, 0], dens[:, 2],
                                 "Simulation time (s)", r"Dislocation density ($\mathrm{m}^{-2}$)",
                                 "Dislocation density vs time", "density_vs_time.png")

    def _video_frame(self, img):
        h, w = img.shape[:2]
        pad_h, pad_w = h & 1, w & 1
        if pad_h or pad_w:
            return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        return img

    def _write_video(self, frame_paths, out_path):
        if not frame_paths:
            return False
        try:
            writer = imageio.get_writer(
                out_path, fps=self.fps, codec="libx264", quality=8, pixelformat="yuv420p")
            for frame_path in frame_paths:
                writer.append_data(self._video_frame(imageio.imread(frame_path)))
            writer.close()
            return os.path.getsize(out_path) > 0
        except Exception:
            return False

    def _write_animations(self, png_paths):
        tqdm.write("Writing animation...")
        for ext in (".mp4", ".mov"):
            out_path = os.path.join(self.output_dir, "dislocation_network" + ext)
            if self._write_video(png_paths, out_path):
                tqdm.write("  {}".format(out_path))
                return
        tqdm.write("  video skipped (mp4/mov write failed)")

    def _write_summary(self, frame_files, box_edges):
        if len(frame_files) < 2:
            return
        fig = plt.figure(figsize=(14, 6))
        panels = [(frame_files[0], "Initial"), (frame_files[-1], "Final")]
        for idx, (frame_path, label) in enumerate(panels, start=1):
            ax = fig.add_subplot(1, 2, idx, projection="3d")
            self._draw_segments(ax, self._parse_gnuplot_frame(frame_path), box_edges)
            ax.set_title("{}, {}".format(label, self._frame_label(frame_path)),
                         fontsize=FONT_SIZE)
        fig.suptitle("Frank-Read dislocation network", fontsize=FONT_SIZE)
        fig.tight_layout()
        summary = os.path.join(self.output_dir, "first_vs_final.png")
        fig.savefig(summary, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        tqdm.write("  {}".format(summary))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize Frank-Read ParaDiS results")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="Simulation results directory (default: frank_read_results)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output directory for PNGs and videos (default: output)")
    parser.add_argument("--fps", type=int, default=2, help="Animation frames per second")
    parser.add_argument("--dpi", type=int, default=FIG_DPI, help="PNG resolution")
    parser.add_argument("--burgmag", type=float, default=None,
                        help="Burgers vector magnitude in meters (default: read from log)")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    FrankReadVisualizer(args.results, args.output, fps=args.fps, dpi=args.dpi,
                        burgmag_m=args.burgmag).run()
