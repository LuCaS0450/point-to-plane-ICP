import argparse
import math

import numpy as np

import icp
from benchmark_3dmatch import find_scene_paths, load_fragment, parse_gt_log, transform_rmse

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit('Open3D is required. Install it with: python -m pip install open3d') from exc


def make_transform(rotation_deg, translation):
    axis = np.array([0.3, -0.5, 0.8], dtype=float)
    axis /= np.linalg.norm(axis)
    T = np.identity(4)
    T[:3, :3] = icp.rotation_vector_to_matrix(axis * math.radians(rotation_deg))
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def to_pcd(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def main():
    parser = argparse.ArgumentParser(description='Visualize point-to-plane ICP alignment on a 3DMatch fragment pair.')
    parser.add_argument('--dataset', default='data/3dmatch')
    parser.add_argument('--scene', default='7-scenes-redkitchen')
    parser.add_argument('--pair-index', type=int, default=0)
    parser.add_argument('--voxel-size', type=float, default=0.04)
    parser.add_argument('--normal-radius', type=float, default=0.12)
    parser.add_argument('--normal-max-nn', type=int, default=30)
    parser.add_argument('--max-iterations', type=int, default=50)
    parser.add_argument('--max-correspondence-distance', type=float, default=0.20)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--no-window', action='store_true', help='run ICP and print metrics without opening a viewer')
    args = parser.parse_args()

    fragment_dir, gt_log = find_scene_paths(args.dataset, args.scene)
    pairs = parse_gt_log(gt_log)
    if args.pair_index < 0 or args.pair_index >= len(pairs):
        raise ValueError(f'pair-index must be in [0, {len(pairs) - 1}]')

    target_id, source_id, gt_transform = pairs[args.pair_index]
    target, target_normals = load_fragment(
        fragment_dir, target_id, args.voxel_size, args.normal_radius, args.normal_max_nn
    )
    source, _ = load_fragment(
        fragment_dir, source_id, args.voxel_size, args.normal_radius, args.normal_max_nn
    )

    init_pose = np.dot(make_transform(5.0, [0.12, -0.08, 0.06]), gt_transform)
    initial_source = icp.transform_points(source, init_pose)
    T, distances, iterations = icp.point_to_plane_icp(
        source,
        target,
        target_normals=target_normals,
        init_pose=init_pose,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        max_correspondence_distance=args.max_correspondence_distance,
    )
    aligned_source = icp.transform_points(source, T)

    print(f'scene: {args.scene}')
    print(f'pair: cloud_bin_{source_id}.ply -> cloud_bin_{target_id}.ply')
    print(f'initial RMSE to ground truth: {transform_rmse(source, init_pose, gt_transform):.4f} m')
    print(f'final RMSE to ground truth:   {transform_rmse(source, T, gt_transform):.4f} m')
    print(f'final mean NN distance:       {float(np.mean(distances)):.4f} m')
    print(f'iterations:                   {iterations + 1}')

    if args.no_window:
        return

    target_pcd = to_pcd(target, [0.65, 0.65, 0.65])
    init_pcd = to_pcd(initial_source, [0.90, 0.15, 0.10])
    aligned_pcd = to_pcd(aligned_source, [0.10, 0.70, 0.25])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
    o3d.visualization.draw_geometries(
        [target_pcd, init_pcd, aligned_pcd, frame],
        window_name='Point-to-plane ICP: target gray, initial source red, aligned source green',
        width=1280,
        height=800,
    )


if __name__ == '__main__':
    main()
