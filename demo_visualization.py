from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sample_points(points, limit, rng):
    if points.shape[0] <= limit:
        return points
    return points[rng.choice(points.shape[0], size=limit, replace=False)]


def set_equal_axes(axis, points):
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def draw_overlay(axis, target, source, source_label, source_color, title):
    axis.scatter(target[:, 0], target[:, 1], target[:, 2], s=0.4, c='#777777', label='target')
    axis.scatter(
        source[:, 0], source[:, 1], source[:, 2], s=0.4, c=source_color, label=source_label
    )
    set_equal_axes(axis, np.vstack([target, source]))
    axis.set_title(title)
    axis.set_xlabel('x (m)')
    axis.set_ylabel('y (m)')
    axis.set_zlabel('z (m)')
    axis.grid(True)
    axis.legend(loc='upper right', markerscale=7)


def save_before_after_demo(target, initial, aligned, title, output, plot_points=6000, seed=13):
    rng = np.random.default_rng(seed)
    target_plot = sample_points(target, plot_points, rng)
    initial_plot = sample_points(initial, plot_points, rng)
    aligned_plot = sample_points(aligned, plot_points, rng)

    figure = plt.figure(figsize=(12, 5.2), dpi=180)
    before = figure.add_subplot(121, projection='3d')
    after = figure.add_subplot(122, projection='3d')
    draw_overlay(before, target_plot, initial_plot, 'initial source', '#D9483B', 'Before ICP')
    draw_overlay(after, target_plot, aligned_plot, 'aligned source', '#1B9E77', 'After Point-to-Plane ICP')
    figure.suptitle(title)
    figure.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches='tight')
    plt.close(figure)
    return output
