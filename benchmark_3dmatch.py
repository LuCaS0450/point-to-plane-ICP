import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

import icp
from download_3dmatch import SCENES

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit('Open3D is required. Install it with: python -m pip install open3d') from exc


def parse_gt_log(path):
    entries = []
    with Path(path).open('r') as f:
        lines = [line.strip() for line in f if line.strip()]

    index = 0
    while index < len(lines):
        header = lines[index].split()
        if len(header) < 3:
            raise ValueError(f'bad gt.log header at line {index + 1}: {lines[index]}')
        src_target = (int(header[0]), int(header[1]))
        index += 1
        matrix = []
        for _ in range(4):
            matrix.append([float(value) for value in lines[index].split()])
            index += 1
        entries.append((src_target[0], src_target[1], np.asarray(matrix, dtype=float)))
    return entries


def find_scene_paths(dataset_root, scene):
    root = Path(dataset_root)
    fragment_dirs = []
    for path in root.rglob('cloud_bin_0.ply'):
        if scene in str(path.parent):
            fragment_dirs.append(path.parent)
    gt_logs = [path for path in root.rglob('gt.log') if scene in str(path.parent)]

    if not fragment_dirs:
        raise FileNotFoundError(f'could not find fragments for {scene} under {root}')
    if not gt_logs:
        raise FileNotFoundError(f'could not find gt.log for {scene} under {root}')

    fragment_dirs.sort(key=lambda p: (len(str(p)), str(p)))
    gt_logs.sort(key=lambda p: (len(str(p)), str(p)))
    return fragment_dirs[0], gt_logs[0]


def load_fragment(fragment_dir, fragment_id, voxel_size, normal_radius, normal_max_nn):
    path = Path(fragment_dir) / f'cloud_bin_{fragment_id}.ply'
    if not path.exists():
        raise FileNotFoundError(path)

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f'empty point cloud: {path}')
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    points = np.asarray(pcd.points, dtype=float)
    normals = np.asarray(pcd.normals, dtype=float)
    return points, normals


def random_transform(rng, max_rotation_deg, max_translation):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = math.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))
    translation = rng.uniform(-max_translation, max_translation, size=3)

    T = np.identity(4)
    T[:3, :3] = icp.rotation_vector_to_matrix(axis * angle)
    T[:3, 3] = translation
    return T


def rotation_error_deg(T_est, T_gt):
    R_delta = np.dot(T_est[:3, :3], T_gt[:3, :3].T)
    value = (np.trace(R_delta) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return math.degrees(math.acos(value))


def translation_error(T_est, T_gt):
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def transform_rmse(points, T_est, T_gt):
    est = icp.transform_points(points, T_est)
    gt = icp.transform_points(points, T_gt)
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))


def mean_nn_distance(points, target):
    distances, _ = icp.nearest_neighbor(points, target)
    return float(np.mean(distances))


def evaluate_pair(method, source, target, target_normals, gt_transform, init_pose, args):
    start = time.perf_counter()
    if method == 'point_to_plane':
        T, distances, iterations, history = icp.point_to_plane_icp(
            source,
            target,
            target_normals=target_normals,
            init_pose=init_pose,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            max_correspondence_distance=args.max_correspondence_distance,
            return_history=True,
        )
        mean_residual = history[-1]['mean_abs_point_to_plane_error'] if history else float('nan')
    elif method == 'point_to_point':
        T, distances, iterations = icp.icp(
            source,
            target,
            init_pose=init_pose,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
        )
        mean_residual = float('nan')
    else:
        raise ValueError(method)

    runtime = time.perf_counter() - start
    aligned = icp.transform_points(source, T)
    return {
        'method': method,
        'rre_deg': rotation_error_deg(T, gt_transform),
        'rte_m': translation_error(T, gt_transform),
        'rmse_m': transform_rmse(source, T, gt_transform),
        'mean_nn_distance_m': mean_nn_distance(aligned, target),
        'mean_euclidean_distance_m': float(np.mean(distances)),
        'mean_abs_point_to_plane_error_m': mean_residual,
        'iterations': int(iterations + 1),
        'runtime_s': runtime,
    }


def summarize(rows, rmse_threshold):
    summary = {}
    for method in sorted({row['method'] for row in rows}):
        method_rows = [row for row in rows if row['method'] == method]
        if not method_rows:
            continue
        rmse = np.array([row['rmse_m'] for row in method_rows], dtype=float)
        rre = np.array([row['rre_deg'] for row in method_rows], dtype=float)
        rte = np.array([row['rte_m'] for row in method_rows], dtype=float)
        runtime = np.array([row['runtime_s'] for row in method_rows], dtype=float)
        success = rmse < rmse_threshold
        summary[method] = {
            'pairs': int(len(method_rows)),
            'registration_recall': float(np.mean(success)),
            'precision_on_evaluated_positive_pairs': 1.0,
            'rmse_threshold_m': float(rmse_threshold),
            'mean_rmse_m': float(np.mean(rmse)),
            'median_rmse_m': float(np.median(rmse)),
            'mean_rre_deg': float(np.mean(rre)),
            'mean_rte_m': float(np.mean(rte)),
            'mean_runtime_s': float(np.mean(runtime)),
        }
    return summary


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'scene', 'source_id', 'target_id', 'method', 'initial_rmse_m',
        'rre_deg', 'rte_m', 'rmse_m', 'mean_nn_distance_m',
        'mean_euclidean_distance_m', 'mean_abs_point_to_plane_error_m',
        'iterations', 'runtime_s',
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Evaluate point-to-plane ICP on 3DMatch local refinement pairs.')
    parser.add_argument('--dataset', default='data/3dmatch')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--scenes', default=','.join(SCENES), help='comma-separated scene names')
    parser.add_argument('--method', choices=['point_to_plane', 'point_to_point', 'both'], default='both')
    parser.add_argument('--voxel-size', type=float, default=0.05)
    parser.add_argument('--normal-radius', type=float, default=0.12)
    parser.add_argument('--normal-max-nn', type=int, default=30)
    parser.add_argument('--max-iterations', type=int, default=50)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--max-correspondence-distance', type=float, default=0.20)
    parser.add_argument('--rmse-threshold', type=float, default=0.20)
    parser.add_argument('--perturb-rot-deg', type=float, default=5.0)
    parser.add_argument('--perturb-trans', type=float, default=0.20)
    parser.add_argument('--max-pairs-per-scene', type=int, default=0, help='0 means all pairs')
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()

    scenes = [scene.strip() for scene in args.scenes.split(',') if scene.strip()]
    methods = ['point_to_plane', 'point_to_point'] if args.method == 'both' else [args.method]
    rng = np.random.default_rng(args.seed)
    cache = {}
    rows = []

    for scene in scenes:
        fragment_dir, gt_log = find_scene_paths(args.dataset, scene)
        pairs = parse_gt_log(gt_log)
        if args.max_pairs_per_scene > 0:
            pairs = pairs[:args.max_pairs_per_scene]
        print(f'{scene}: {len(pairs)} pairs')

        for target_id, source_id, gt_transform in pairs:
            for fragment_id in (target_id, source_id):
                key = (str(fragment_dir), fragment_id, args.voxel_size)
                if key not in cache:
                    cache[key] = load_fragment(
                        fragment_dir,
                        fragment_id,
                        args.voxel_size,
                        args.normal_radius,
                        args.normal_max_nn,
                    )

            target, target_normals = cache[(str(fragment_dir), target_id, args.voxel_size)]
            source, _ = cache[(str(fragment_dir), source_id, args.voxel_size)]
            perturb = random_transform(rng, args.perturb_rot_deg, args.perturb_trans)
            init_pose = np.dot(perturb, gt_transform)
            initial_rmse = transform_rmse(source, init_pose, gt_transform)

            for method in methods:
                metrics = evaluate_pair(method, source, target, target_normals, gt_transform, init_pose, args)
                metrics.update({
                    'scene': scene,
                    'source_id': int(source_id),
                    'target_id': int(target_id),
                    'initial_rmse_m': initial_rmse,
                })
                rows.append(metrics)

    results_dir = Path(args.results_dir)
    csv_path = results_dir / '3dmatch_point_to_plane_metrics.csv'
    json_path = results_dir / '3dmatch_point_to_plane_summary.json'
    write_csv(rows, csv_path)
    summary = {
        'dataset': str(Path(args.dataset)),
        'scenes': scenes,
        'protocol': '3DMatch local refinement from perturbed ground-truth transforms',
        'parameters': vars(args),
        'summary': summarize(rows, args.rmse_threshold),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'wrote {csv_path}')
    print(f'wrote {json_path}')
    print(json.dumps(summary['summary'], indent=2))


if __name__ == '__main__':
    main()
