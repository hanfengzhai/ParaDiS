#!/usr/bin/env python3
"""Visualize ParaDiS gnuplot frames and properties for any example.

Writes PNG frames, summary plots, and MP4/MOV animation under output/.
Intermediate frame PNGs under output/frames/ are removed after a video is written.

Usage (from an example directory):
    python ../utils/visualize.py
    python ../utils/visualize.py --results frank_read_results --fps 2

Usage (from repository root):
    python examples/utils/visualize.py --example-dir examples/2_frank_read/111_BCC
"""

from __future__ import annotations

import argparse
import os
import re
import shutil

import imageio.v2 as imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator
from tqdm import tqdm

plotting_params = {"font.family": "serif", "font.serif": ["Libertinus Serif"],
                   "mathtext.fontset": "cm", "xtick.labelsize": 15, "ytick.labelsize": 15,
                   "axes.labelsize": 15, "legend.fontsize": 15, "axes.unicode_minus": False}
plt.rcParams.update(plotting_params)

PROP_NAMES = ("time_Plastic_strain", "density", "alleps")
FRAME_GLOB_PREFIX = "0t"
FINAL_FRAME_NAME = "gnuplot.final"
DEFAULT_GNUPLOT_FREQ = 100
FIG_DPI = 200
FONT_SIZE = 15
DISL_COLOR = "#1f4fd8"
BOX_COLOR = "#1a1a1a"
LINE_COLOR = "darkblue"
SPATIAL_AXIS_LABELS = (r"$X$ [$\mu$m]", r"$Y$ [$\mu$m]", r"$Z$ [$\mu$m]")
DEFAULT_BURGMAG_M = 2.8754e-10  # Ta BCC
# Oblique 3D views distort in-plane ellipses; glissile loops use plane-normal views.
GLISSILE_VIEWS = {
    "001": {"elev": 90, "azim": -90},
    "111": {"elev": 35.26, "azim": 45.0},
}


class ParadiSVisualizer:
    """Render dislocation frames and properties from a ParaDiS run."""

    def __init__(self, example_dir, results_dir=None, output_dir=None, example_name=None,
                 fps=2, dpi=FIG_DPI, burgmag_m=None):
        self.example_dir = os.path.abspath(example_dir)
        self.example_name = example_name or self._detect_example_name()
        self.results_dir = os.path.abspath(
            results_dir or os.path.join(self.example_dir, self.example_name + "_results"))
        self.output_dir = os.path.abspath(
            output_dir or os.path.join(self.example_dir, "output"))
        self.gnu_dir = os.path.join(self.results_dir, "gnuplot")
        self.props_dir = os.path.join(self.results_dir, "properties")
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.fps = fps
        self.dpi = dpi
        self.burgmag_m = burgmag_m
        self.um_per_b = None
        self.gnuplotfreq = DEFAULT_GNUPLOT_FREQ
        self.box_limits = None

    def _detect_example_name(self):
        ctrl_names = sorted(
            name[:-5] for name in os.listdir(self.example_dir) if name.endswith(".ctrl"))
        if len(ctrl_names) == 1:
            return ctrl_names[0]
        if not ctrl_names:
            raise SystemExit("No .ctrl file found in {}".format(self.example_dir))
        raise SystemExit("Multiple .ctrl files in {} — pass --name".format(self.example_dir))

    def run(self):
        if not os.path.isdir(self.gnu_dir):
            raise SystemExit("gnuplot directory not found: {}".format(self.gnu_dir))
        self.gnuplotfreq = self._resolve_gnuplotfreq()
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

    def _ctrl_basename(self):
        return self.example_name + ".ctrl"

    def _log_basename(self):
        return self.example_name + ".log"

    def _resolve_gnuplotfreq(self):
        for path in self._ctrl_paths():
            if not os.path.isfile(path):
                continue
            with open(path) as fh:
                text = fh.read()
            match = re.search(r"^\s*gnuplotfreq\s*=\s*(\d+)", text, re.MULTILINE)
            if match:
                return int(match.group(1))
        return DEFAULT_GNUPLOT_FREQ

    def _resolve_max_cycle(self):
        for path in self._ctrl_paths():
            if not os.path.isfile(path):
                continue
            with open(path) as fh:
                text = fh.read()
            match = re.search(r"^\s*maxstep\s*=\s*(\d+)", text, re.MULTILINE)
            if match:
                return int(match.group(1))
        for path in self._log_paths():
            if not os.path.isfile(path):
                continue
            with open(path) as fh:
                text = fh.read()
            match = re.search(r"last\s+cycle\s*:\s*(\d+)", text)
            if match:
                return int(match.group(1))
        return None

    def _ctrl_paths(self):
        basename = self._ctrl_basename()
        return [
            os.path.join(self.example_dir, basename),
            os.path.join(os.path.dirname(self.results_dir), basename),
        ]

    def _log_paths(self):
        basename = self._log_basename()
        names = [basename, basename.replace(".log", "_cpu.log")]
        paths = []
        for name in names:
            paths.extend([
                os.path.join(self.example_dir, name),
                os.path.join(os.path.dirname(self.results_dir), name),
                os.path.join(self.results_dir, name),
            ])
        return paths

    def _list_gnuplot_frames(self):
        names = sorted(n for n in os.listdir(self.gnu_dir)
                       if n.startswith(FRAME_GLOB_PREFIX) and not n.endswith(".final"))
        return [os.path.join(self.gnu_dir, n) for n in names]

    def _final_frame_path(self):
        path = os.path.join(self.gnu_dir, FINAL_FRAME_NAME)
        return path if os.path.isfile(path) else None

    def _frame_cycle(self, path):
        name = os.path.basename(path)
        if name == FINAL_FRAME_NAME:
            return None
        match = re.search(r"0t(\d+)", name)
        if not match:
            return None
        return int(match.group(1)) * self.gnuplotfreq

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
        for path in self._log_paths():
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
        ax.set_xlabel(SPATIAL_AXIS_LABELS[0], fontsize=FONT_SIZE, labelpad=8)
        ax.set_ylabel(SPATIAL_AXIS_LABELS[1], fontsize=FONT_SIZE, labelpad=8)
        ax.set_zlabel(SPATIAL_AXIS_LABELS[2], fontsize=FONT_SIZE, labelpad=12)
        ax.tick_params(labelsize=FONT_SIZE)
        view = self._view_angles()
        ax.view_init(elev=view["elev"], azim=view["azim"])
        self._apply_scientific_ticks_3d(ax)

    def _glissile_plane(self):
        if self.example_name != "glissile_loop":
            return None
        base = os.path.basename(os.path.normpath(self.example_dir))
        if base.startswith("001"):
            return "001"
        if base.startswith("111"):
            return "111"
        return None

    def _glissile_case_label(self):
        base = os.path.basename(os.path.normpath(self.example_dir))
        if self.example_name != "glissile_loop":
            return None
        parts = base.split("_", 1)
        if len(parts) == 2:
            return "{} {}".format(parts[0], parts[1])
        return base

    def _view_angles(self):
        plane = self._glissile_plane()
        if plane is not None:
            return GLISSILE_VIEWS[plane]
        return {"elev": 22, "azim": -58}

    def _trim_image_whitespace(self, path, threshold=240, margins=(10, 10, 10, 55)):
        """Crop near-white borders; margins are (top, bottom, left, right) in pixels."""
        img = imageio.imread(path)
        if img.ndim == 2:
            mask = img < threshold
        else:
            mask = np.any(img[:, :, :3] < threshold, axis=2)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        top, bottom, left, right = margins
        h, w = img.shape[:2]
        ymin = max(0, ymin - top)
        xmin = max(0, xmin - left)
        ymax = min(h, ymax + bottom + 1)
        xmax = min(w, xmax + right + 1)
        imageio.imwrite(path, img[ymin:ymax, xmin:xmax])

    def _save_3d_figure(self, fig, out_path, layout=None, adjust_layout=True, trim=False):
        if adjust_layout:
            if layout:
                fig.subplots_adjust(**layout)
            else:
                fig.subplots_adjust(left=0.08, right=0.82, bottom=0.08, top=0.92)
        fig.savefig(out_path, dpi=self.dpi, pad_inches=0.15 if trim else 0.5)
        plt.close(fig)
        if trim:
            self._trim_image_whitespace(out_path)

    def _render_frame(self, segments, box_edges, out_path):
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        self._draw_segments(ax, segments, box_edges)
        self._save_3d_figure(fig, out_path)

    def _render_frames(self, frame_files, box_edges):
        png_paths = []
        for frame_path in tqdm(frame_files, desc="frames"):
            out_png = os.path.join(self.frames_dir, "{}.png".format(os.path.basename(frame_path)))
            self._render_frame(self._parse_gnuplot_frame(frame_path), box_edges, out_png)
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

    def _load_property_file(self, name):
        path = os.path.join(self.props_dir, name)
        if os.path.isfile(path):
            return np.loadtxt(path)
        return None

    def _nice_spatial_tick(self, val):
        """Round a spatial coordinate to a clean axis tick (e.g. 1.423 -> 1.0)."""
        if abs(val) < 1e-15:
            return 0.0
        av = abs(float(val))
        step = 1.0 if av >= 0.5 else 0.5
        return float(np.sign(val) * step * round(av / step))

    def _spatial_tick_label(self, val):
        if abs(val) < 1e-15:
            return r"$0$"
        nice = self._nice_spatial_tick(val)
        if abs(nice - round(nice)) < 1e-9:
            return r"${:.0f}$".format(nice)
        return r"${:.1f}$".format(nice)

    def _sci_scale(self, values):
        vals = np.asarray(values, dtype=float)
        nonzero = vals[np.abs(vals) > 1e-15]
        if nonzero.size == 0:
            return vals, 0
        exp = int(np.floor(np.log10(np.max(np.abs(nonzero)))))
        return vals / (10.0 ** exp), exp

    def _mantissa_plain(self, val):
        if abs(val) < 1e-15:
            return "0"
        return "{:.3g}".format(val)

    def _mantissa_tick_labels(self, values, exp):
        labels = []
        for val in np.asarray(values, dtype=float):
            if abs(val) < 1e-15:
                labels.append(r"$0$")
            else:
                scaled = val / (10.0 ** exp)
                labels.append(r"${}$".format(self._mantissa_plain(scaled)))
        return labels

    def _place_sci_multiplier(self, ax, axis, exp):
        if exp == 0:
            return
        text = self._sci_offset_text(exp)
        if axis is ax.xaxis:
            ax.text(1.0, 0.0, text, transform=ax.transAxes,
                    ha="left", va="top", fontsize=FONT_SIZE, clip_on=False)
        elif axis is ax.yaxis:
            ax.text(0.0, 1.0, text, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=FONT_SIZE, clip_on=False)

    def _sci_offset_text(self, exp):
        return r"$\times 10^{{{}}}$".format(exp)

    def _data_limits_with_margin(self, vals, margin_frac=0.04):
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if lo == hi:
            span = abs(lo) or 1.0
            lo, hi = lo - 0.5 * span, hi + 0.5 * span
        pad = (hi - lo) * margin_frac
        lo, hi = lo - pad, hi + pad
        if float(np.min(vals)) >= 0.0:
            lo = max(lo, 0.0)
        return lo, hi

    def _ticks_in_range(self, lo, hi, nbins=6):
        from matplotlib.ticker import MaxNLocator
        ticks = MaxNLocator(nbins=nbins).tick_values(lo, hi)
        return ticks[(ticks >= lo - 1e-15) & (ticks <= hi + 1e-15)]

    def _set_axis_sci_ticks(self, axis, tick_values, exp=None, corner_ax=None):
        tick_values = np.asarray(tick_values, dtype=float)
        if exp is None:
            _, exp = self._sci_scale(tick_values)
        axis.set_ticks(tick_values)
        axis.set_ticklabels(self._mantissa_tick_labels(tick_values, exp))
        axis.get_offset_text().set_visible(False)
        if corner_ax is not None:
            self._place_sci_multiplier(corner_ax, axis, exp)

    def _apply_scientific_ticks(self, ax):
        for axis, getlim in ((ax.xaxis, ax.get_xlim), (ax.yaxis, ax.get_ylim)):
            lo, hi = getlim()
            ticks = self._ticks_in_range(lo, hi)
            if len(ticks):
                self._set_axis_sci_ticks(axis, ticks, corner_ax=ax)

    def _axis_min_mid_max_ticks(self, vmin, vmax):
        lo = self._nice_spatial_tick(vmin)
        hi = self._nice_spatial_tick(vmax)
        if lo > hi:
            lo, hi = hi, lo
        if lo < -1e-15 < hi:
            mid = 0.0
        else:
            mid = self._nice_spatial_tick(0.5 * (float(vmin) + float(vmax)))
        ticks = [lo, mid, hi]
        unique = []
        tol = 1e-9 * max(1.0, abs(hi - lo))
        for tick in ticks:
            if not unique or abs(tick - unique[-1]) > tol:
                unique.append(tick)
        return unique

    def _apply_scientific_ticks_3d(self, ax):
        if self.box_limits is None:
            return
        mins, maxs = self.box_limits
        tick_sets = []
        for idx, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis)):
            ticks = self._axis_min_mid_max_ticks(mins[idx], maxs[idx])
            tick_sets.append((axis, ticks))
            lo, hi = ticks[0], ticks[-1]
            if idx == 0:
                ax.set_xlim(lo, hi)
            elif idx == 1:
                ax.set_ylim(lo, hi)
            else:
                ax.set_zlim(lo, hi)
        for axis, ticks in tick_sets:
            axis.set_ticks(ticks)
            axis.set_ticklabels([self._spatial_tick_label(t) for t in ticks])
            axis.get_offset_text().set_visible(False)
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_minor_locator(NullLocator())
            axis.set_tick_params(label1On=True, label2On=False, labelsize=FONT_SIZE)

    def _scale_time_axis(self, time_s):
        time_s = np.asarray(time_s, dtype=float)
        t_max = float(np.max(time_s))
        if t_max < 1e-3:
            return time_s * 1e6, r"Simulation time [$\mu$s]"
        if t_max < 1.0:
            return time_s * 1e3, r"Simulation time [ms]"
        return time_s, "Simulation time [s]"

    def _save_line_plot(self, x, y, xlabel, ylabel, out_name):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, y, "o-", color=LINE_COLOR, linewidth=2, markersize=4,
                markerfacecolor=LINE_COLOR, markeredgecolor="black", markeredgewidth=0.4)
        ax.set_xlim(self._data_limits_with_margin(x))
        ax.set_ylim(self._data_limits_with_margin(y))
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.3)
        self._apply_scientific_ticks(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, out_name), dpi=self.dpi,
                    bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

    def _plot_stress_strain(self):
        plastic = self._load_property_file("stress_Plastic_strain")
        if plastic is not None:
            self._save_line_plot(plastic[:, 0] * 100.0, plastic[:, 1] / 1e6,
                                 "Plastic strain [%]", "Stress [MPa]",
                                 "stress_vs_plastic_strain.png")
        total = self._load_property_file("stress_Total_strain")
        if total is not None:
            self._save_line_plot(total[:, 0] * 100.0, total[:, 1] / 1e6,
                                 "Total strain [%]", "Stress [MPa]",
                                 "stress_vs_total_strain.png")

    def _plot_properties(self, props):
        self._plot_stress_strain()
        if "time_Plastic_strain" in props:
            t_eps = props["time_Plastic_strain"]
            time_x, time_label = self._scale_time_axis(t_eps[:, 0])
            self._save_line_plot(time_x, t_eps[:, 1] * 100.0,
                                 time_label, "Plastic strain [%]",
                                 "plastic_strain_vs_time.png")
        if "density" not in props:
            return
        dens = props["density"]
        self._save_line_plot(dens[:, 0] * 100.0, dens[:, 2],
                             "Strain [%]", r"Dislocation density [$\mathrm{m}^{-2}$]",
                             "density_vs_strain.png")
        if "time_Plastic_strain" in props:
            t_eps = props["time_Plastic_strain"]
            time_x, time_label = self._scale_time_axis(t_eps[:, 0])
            self._save_line_plot(time_x, dens[:, 2],
                                 time_label, r"Dislocation density [$\mathrm{m}^{-2}$]",
                                 "density_vs_time.png")

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

    def _remove_gif_outputs(self):
        for name in os.listdir(self.output_dir):
            if name.lower().endswith(".gif"):
                os.remove(os.path.join(self.output_dir, name))

    def _remove_frame_pngs(self):
        if not os.path.isdir(self.frames_dir):
            return
        shutil.rmtree(self.frames_dir)
        tqdm.write("  removed {}".format(self.frames_dir))

    def _write_animations(self, png_paths):
        self._remove_gif_outputs()
        tqdm.write("Writing animation...")
        for ext in (".mp4", ".mov"):
            out_path = os.path.join(self.output_dir, "dislocation_network" + ext)
            if self._write_video(png_paths, out_path):
                tqdm.write("  {}".format(out_path))
                self._remove_frame_pngs()
                return
        tqdm.write("  video skipped (mp4/mov write failed)")

    def _annotate_panel(self, ax, label, cycle):
        if cycle is not None:
            text = "{}, cycle {}".format(label, cycle)
        else:
            text = label
        label = self._glissile_case_label()
        if label is not None:
            text += " ({} view)".format(label)
        ax.text2D(0.5, 0.02, text, transform=ax.transAxes,
                  fontsize=FONT_SIZE, va="bottom", ha="center",
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0))

    def _write_summary(self, frame_files, box_edges):
        if not frame_files:
            return
        final_path = self._final_frame_path()
        if final_path is None and len(frame_files) < 2:
            return
        initial_path = frame_files[0]
        summary_final = final_path if final_path else frame_files[-1]
        init_cycle = self._frame_cycle(initial_path)
        if summary_final.endswith(FINAL_FRAME_NAME):
            final_cycle = self._resolve_max_cycle()
        else:
            final_cycle = self._frame_cycle(summary_final)
        fig = plt.figure(figsize=(12, 5.5))
        gs = gridspec.GridSpec(1, 2, figure=fig, left=0.04, right=0.94,
                               bottom=0.12, top=0.96, wspace=0.28)
        for ax_idx, (frame_path, label, cycle) in enumerate(
                ((initial_path, "Initial", init_cycle),
                 (summary_final, "Final", final_cycle))):
            ax = fig.add_subplot(gs[ax_idx], projection="3d")
            self._draw_segments(ax, self._parse_gnuplot_frame(frame_path), box_edges)
            self._annotate_panel(ax, label, cycle)
        summary = os.path.join(self.output_dir, "first_vs_final.png")
        self._save_3d_figure(fig, summary, adjust_layout=False, trim=True)
        tqdm.write("  {} (cycle {}) vs {} (cycle {})".format(
            os.path.basename(initial_path), init_cycle,
            os.path.basename(summary_final), final_cycle))
        tqdm.write("  {}".format(summary))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize ParaDiS example results")
    parser.add_argument("--example-dir", default=os.getcwd(),
                        help="Example directory containing .ctrl and output/ (default: cwd)")
    parser.add_argument("--name", default=None,
                        help="Example basename, e.g. frank_read (default: detect from .ctrl)")
    parser.add_argument("--results", default=None,
                        help="Results directory (default: <name>_results under example-dir)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: output under example-dir)")
    parser.add_argument("--fps", type=int, default=2, help="Animation frames per second")
    parser.add_argument("--dpi", type=int, default=FIG_DPI, help="PNG resolution")
    parser.add_argument("--burgmag", type=float, default=None,
                        help="Burgers vector magnitude in meters (default: read from log)")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    ParadiSVisualizer(args.example_dir, results_dir=args.results, output_dir=args.output,
                      example_name=args.name, fps=args.fps, dpi=args.dpi,
                      burgmag_m=args.burgmag).run()
