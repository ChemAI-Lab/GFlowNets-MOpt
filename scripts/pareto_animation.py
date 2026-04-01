from gflow_vqe.utils import *
from gflow_vqe.hamiltonians import *
from gflow_vqe.gflow_utils import *
from gflow_vqe.result_analysis import *
from gflow_vqe.training import *
from openfermion import commutator
import os
import pickle
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import animation
from matplotlib.gridspec import GridSpec


# User options
fig_name = None  # Use None to default to the molecule name.
animation_format = "mp4"  # "mp4" (recommended) or "gif"
animation_fps = 24
animation_frame_step = 1   # Use >1 to skip snapshots (e.g., 5 or 10)
animation_max_frames = None  # e.g., 600 to cap total frames
animation_dpi = 120
points_cache_enabled = True
save_kde_animation = True

# Plot configuration matching animation.py
joint_xlim = (0.55, 1.8)
joint_ylim = (9, 27)
joint_height = 7.5
x_hist_nbins = 36
y_hist_bins = np.arange(joint_ylim[0] - 0.5, joint_ylim[1] + 1.5, 1.0)
si_reference_point = (0.614, 18)


def update_pareto_front(front, point, tiny=1e-12):
    """Update a 2D minimization Pareto front with one new point."""
    px, py = point

    for fx, fy in front:
        if fx <= px + tiny and fy <= py + tiny and (fx < px - tiny or fy < py - tiny):
            return front

    updated_front = []
    for fx, fy in front:
        dominated = px <= fx + tiny and py <= fy + tiny and (px < fx - tiny or py < fy - tiny)
        if not dominated:
            updated_front.append((fx, fy))

    duplicate = any(abs(fx - px) < tiny and abs(fy - py) < tiny for fx, fy in updated_front)
    if not duplicate:
        updated_front.append((px, py))

    updated_front.sort(key=lambda item: (item[0], item[1]))
    return updated_front


def build_frame_indices(n_points, frame_step, max_frames):
    if n_points <= 0:
        return []
    if frame_step < 1:
        raise ValueError("animation_frame_step must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("animation_max_frames must be None or >= 1")

    frame_indices = list(range(1, n_points + 1, frame_step))
    if frame_indices[-1] != n_points:
        frame_indices.append(n_points)

    if max_frames is not None and len(frame_indices) > max_frames:
        selected = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=int)
        frame_indices = [frame_indices[i] for i in selected]
        if frame_indices[-1] != n_points:
            frame_indices[-1] = n_points

    return frame_indices


def build_pareto_snapshots(points, frame_indices):
    """Cache Pareto fronts only for the frames that will be rendered."""
    frame_lookup = set(frame_indices)
    snapshots = {}
    front = []

    for idx, point in enumerate(points, start=1):
        front = update_pareto_front(front, tuple(point))
        if idx in frame_lookup:
            snapshots[idx] = np.array(front, dtype=float)

    return snapshots


def build_points(sampled_graphs, fci_wfn, n_q, n_paulis):
    points = []
    for graph in sampled_graphs:
        measurement = get_groups_measurement(graph, fci_wfn, n_q)
        color_objective = max_color(graph)
        points.append((measurement, color_objective))
    return np.asarray(points, dtype=float)


def load_or_build_points(cache_path, sampled_graphs_path, sampled_graphs, fci_wfn, n_q, n_paulis):
    sampled_graphs_mtime_ns = os.path.getmtime(sampled_graphs_path)
    if points_cache_enabled and os.path.exists(cache_path):
        with np.load(cache_path) as cached:
            cache_valid = (
                float(cached["sampled_graphs_mtime_ns"]) == float(sampled_graphs_mtime_ns)
                and int(cached["n_graphs"]) == len(sampled_graphs)
                and int(cached["n_qubits"]) == int(n_q)
                and int(cached["n_paulis"]) == int(n_paulis)
            )
            if cache_valid:
                print("Loaded cached Pareto points from {}".format(cache_path))
                return cached["points"]

    start_time = time.perf_counter()
    points = build_points(sampled_graphs, fci_wfn, n_q, n_paulis)
    elapsed = time.perf_counter() - start_time
    print("Computed Pareto points in {:.3f}s".format(elapsed))

    if points_cache_enabled:
        np.savez_compressed(
            cache_path,
            points=points,
            sampled_graphs_mtime_ns=sampled_graphs_mtime_ns,
            n_graphs=len(sampled_graphs),
            n_qubits=int(n_q),
            n_paulis=int(n_paulis),
        )
        print("Saved Pareto point cache to {}".format(cache_path))

    return points


def _bin_indices(values, bin_edges):
    indices = np.searchsorted(bin_edges, values, side="right") - 1
    valid = (indices >= 0) & (indices < len(bin_edges) - 1)
    return indices, valid


def build_histogram_snapshots(points, frame_indices, x_bins, y_bins):
    frame_lookup = set(frame_indices)
    x_indices, x_valid = _bin_indices(points[:, 0], x_bins)
    y_indices, y_valid = _bin_indices(points[:, 1], y_bins)

    hist_x = np.zeros(len(x_bins) - 1, dtype=int)
    hist_y = np.zeros(len(y_bins) - 1, dtype=int)
    snapshots = {}

    for idx in range(len(points)):
        if x_valid[idx]:
            hist_x[x_indices[idx]] += 1
        if y_valid[idx]:
            hist_y[y_indices[idx]] += 1
        frame_count = idx + 1
        if frame_count in frame_lookup:
            snapshots[frame_count] = (hist_x.copy(), hist_y.copy())

    return snapshots


def resolve_writer(format_name, fps):
    format_name = format_name.lower()
    if format_name == "mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("mp4 output requires ffmpeg to be available to Matplotlib.")
        return animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            extra_args=["-preset", "ultrafast", "-pix_fmt", "yuv420p"],
        )
    if format_name == "gif":
        return animation.PillowWriter(fps=fps)
    raise ValueError("Unsupported animation_format '{}'. Use 'mp4' or 'gif'.".format(format_name))


def render_histogram_animation(
    points,
    frame_indices,
    pareto_snapshots,
    output_path,
):
    x = points[:, 0]
    y = points[:, 1]
    x_bins = np.linspace(joint_xlim[0], joint_xlim[1], x_hist_nbins + 1)
    hist_snapshots = build_histogram_snapshots(points, frame_indices, x_bins, y_hist_bins)
    full_hist_x, _ = np.histogram(x, bins=x_bins)
    full_hist_y, _ = np.histogram(y, bins=y_hist_bins)
    max_hist_x = max(1, int(full_hist_x.max()))
    max_hist_y = max(1, int(full_hist_y.max()))

    fig = plt.figure(figsize=(joint_height, joint_height))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        wspace=0.05,
        hspace=0.05,
    )

    ax_marg_x = fig.add_subplot(gs[0, 0])
    ax_joint = fig.add_subplot(gs[1, 0], sharex=ax_marg_x)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    scatter = ax_joint.scatter([], [], alpha=0.5, s=30, edgecolors="none")
    pareto_line, = ax_joint.plot(
        [],
        [],
        color="orange",
        marker="o",
        markersize=6,
        linewidth=1.5,
        label="Pareto front",
    )
    ax_joint.plot(
        si_reference_point[0],
        si_reference_point[1],
        marker="D",
        color="red",
        markersize=7,
        linestyle="None",
        label="SI",
    )
    frame_text = ax_joint.text(
        0.02,
        0.98,
        "",
        transform=ax_joint.transAxes,
        va="top",
        ha="left",
    )

    x_bar_widths = np.diff(x_bins)
    x_bars = ax_marg_x.bar(
        x_bins[:-1],
        np.zeros_like(x_bins[:-1]),
        width=x_bar_widths,
        align="edge",
        color="purple",
        alpha=0.55,
        edgecolor="purple",
    )
    y_bar_heights = np.diff(y_hist_bins)
    y_bars = ax_marg_y.barh(
        y_hist_bins[:-1],
        np.zeros_like(y_hist_bins[:-1]),
        height=y_bar_heights,
        align="edge",
        color="green",
        alpha=0.55,
        edgecolor="green",
    )

    ax_joint.set_xlabel(r"$\epsilon^2M(x)$", fontsize=14)
    ax_joint.set_ylabel(r"$N_G(x)$", fontsize=14)
    ax_joint.set_xlim(*joint_xlim)
    ax_joint.set_ylim(*joint_ylim)
    ax_joint.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fixed_xticks = ax_joint.get_xticks()
    fixed_yticks = ax_joint.get_yticks()
    ax_joint.set_xticks(fixed_xticks)
    ax_joint.set_yticks(fixed_yticks)
    ax_joint.legend(loc="best")

    ax_marg_x.set_xlim(*joint_xlim)
    ax_marg_x.set_ylim(0, max_hist_x * 1.05)
    ax_marg_x.set_xticks(fixed_xticks)
    ax_marg_x.set_ylabel("")
    ax_marg_x.tick_params(axis="x", labelbottom=False)
    ax_marg_x.tick_params(axis="y", labelleft=False)

    ax_marg_y.set_ylim(*joint_ylim)
    ax_marg_y.set_xlim(0, max_hist_y * 1.05)
    ax_marg_y.set_yticks(fixed_yticks)
    ax_marg_y.set_xlabel("")
    ax_marg_y.tick_params(axis="x", labelbottom=False)
    ax_marg_y.tick_params(axis="y", labelleft=False)

    def update(frame_count):
        current_points = points[:frame_count]
        current_front = pareto_snapshots[frame_count]
        hist_x, hist_y = hist_snapshots[frame_count]

        scatter.set_offsets(current_points)
        if current_front.size > 0:
            pareto_line.set_data(current_front[:, 0], current_front[:, 1])
        else:
            pareto_line.set_data([], [])
        frame_text.set_text("Samples: {}/{}".format(frame_count, len(points)))

        for patch, height in zip(x_bars, hist_x):
            patch.set_height(height)
        for patch, width in zip(y_bars, hist_y):
            patch.set_width(width)

        return [scatter, pareto_line, frame_text, *x_bars, *y_bars]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / max(animation_fps, 1),
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )

    writer = resolve_writer(animation_format, animation_fps)
    print("Saving animation to {}".format(output_path))
    save_start_time = time.perf_counter()
    ani.save(output_path, writer=writer, dpi=animation_dpi)
    print("Animation encode time: {:.3f}s".format(time.perf_counter() - save_start_time))
    plt.close(fig)
    print("Animation saved.")


def render_kde_animation(
    points,
    frame_indices,
    pareto_snapshots,
    output_path,
):
    sns.set_theme(style="whitegrid")
    joint_grid = sns.JointGrid(x=points[:1, 0], y=points[:1, 1], height=joint_height, space=0)
    fig = joint_grid.figure

    def update(frame_count):
        current_points = points[:frame_count]
        current_front = pareto_snapshots[frame_count]
        x = current_points[:, 0]
        y = current_points[:, 1]

        joint_grid.ax_joint.cla()
        joint_grid.ax_marg_x.cla()
        joint_grid.ax_marg_y.cla()

        joint_grid.set_axis_labels(r"$\epsilon^2M(x)$", r"$N_G(x)$", fontsize=14)
        joint_grid.ax_joint.scatter(x, y, alpha=0.5, s=30, edgecolors="none")

        if current_front.size > 0:
            joint_grid.ax_joint.plot(
                current_front[:, 0],
                current_front[:, 1],
                color="orange",
                marker="o",
                markersize=6,
                linewidth=1.5,
                label="Pareto front",
            )

        if len(x) > 1 and np.std(x) > 0:
            sns.kdeplot(x=x, ax=joint_grid.ax_marg_x, fill=True, color="purple", bw_adjust=0.8)
        if len(y) > 1 and np.std(y) > 0:
            sns.kdeplot(y=y, ax=joint_grid.ax_marg_y, fill=True, color="green", bw_adjust=0.8)

        joint_grid.ax_joint.plot(
            si_reference_point[0],
            si_reference_point[1],
            marker="D",
            color="red",
            markersize=7,
            linestyle="None",
            label="SI",
        )
        joint_grid.ax_joint.text(
            0.02,
            0.98,
            "Samples: {}/{}".format(frame_count, len(points)),
            transform=joint_grid.ax_joint.transAxes,
            va="top",
            ha="left",
        )
        joint_grid.ax_joint.legend(loc="best")
        joint_grid.ax_joint.set_xlim(*joint_xlim)
        joint_grid.ax_joint.set_ylim(*joint_ylim)
        joint_grid.ax_joint.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        joint_grid.ax_marg_x.set_xlim(joint_grid.ax_joint.get_xlim())
        joint_grid.ax_marg_y.set_ylim(joint_grid.ax_joint.get_ylim())
        joint_grid.ax_marg_x.set_ylabel("")
        joint_grid.ax_marg_y.set_xlabel("")
        joint_grid.ax_marg_x.tick_params(axis="x", labelbottom=False)
        joint_grid.ax_marg_y.tick_params(axis="y", labelleft=False)
        joint_grid.ax_marg_y.set_xticks([])
        joint_grid.ax_marg_x.set_yticks([])
        joint_grid.ax_marg_y.set_xticklabels([])
        joint_grid.ax_marg_x.set_yticklabels([])
        fig.subplots_adjust(bottom=0.10, left=0.10, right=0.98, top=0.98)

        return (joint_grid.ax_joint, joint_grid.ax_marg_x, joint_grid.ax_marg_y)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / max(animation_fps, 1),
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )

    writer = resolve_writer(animation_format, animation_fps)
    print("Saving KDE animation to {}".format(output_path))
    save_start_time = time.perf_counter()
    ani.save(output_path, writer=writer, dpi=animation_dpi)
    print("KDE animation encode time: {:.3f}s".format(time.perf_counter() - save_start_time))
    plt.close(fig)
    print("KDE animation saved.")


def main():
    driver_args = parse_driver_args()
    molecule = driver_args.func
    molecule_name = driver_args.func_name
    if molecule is None:
        raise ValueError("Unknown molecule '{}'".format(molecule_name))

    current_fig_name = fig_name if fig_name is not None else molecule_name
    sampled_graphs_path = current_fig_name + "_sampled_graphs.p"
    points_cache_path = current_fig_name + "_pareto_points_cache.npz"
    output_path = "{}_pareto_animation.{}".format(current_fig_name, animation_format.lower())
    output_path_kde = "{}_pareto_animation_KDE.{}".format(current_fig_name, animation_format.lower())

    mol, H, Hferm, n_paulis, Hq = molecule()
    print("Number of Pauli products to measure: {}".format(n_paulis))

    sparse_hamiltonian = get_sparse_operator(Hq)
    energy, fci_wfn = get_ground_state(sparse_hamiltonian)
    n_q = count_qubits(Hq)

    with open(sampled_graphs_path, "rb") as f:
        sampled_graphs = pickle.load(f)

    sampled_graphs = [g for g in sampled_graphs if color_reward(g) > 0]
    if len(sampled_graphs) == 0:
        raise RuntimeError("No valid sampled graphs were found in '{}'.".format(sampled_graphs_path))

    print("Number of valid graphs in file: {}".format(len(sampled_graphs)))
    points = load_or_build_points(points_cache_path, sampled_graphs_path, sampled_graphs, fci_wfn, n_q, n_paulis)
    frame_indices = build_frame_indices(len(points), animation_frame_step, animation_max_frames)
    pareto_snapshots = build_pareto_snapshots(points, frame_indices)
    sns.set_theme(style="whitegrid")
    render_histogram_animation(points, frame_indices, pareto_snapshots, output_path)
    if save_kde_animation:
        render_kde_animation(points, frame_indices, pareto_snapshots, output_path_kde)


if __name__ == "__main__":
    main()
