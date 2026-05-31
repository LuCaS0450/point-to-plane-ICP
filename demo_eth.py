import argparse
import csv
import json
from pathlib import Path

import icp
from benchmark_eth import (
    load_protocol,
    load_scan,
    load_scanner_poses,
    validate_ground_truth_direction,
)
from demo_visualization import save_before_after_demo
from download_eth import find_apartment_paths
from eval_utils import transform_rmse


def choose_protocol_index(metrics_path, min_overlap):
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return 0
    candidates = []
    with metrics_path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['method'] != 'point_to_plane':
                continue
            if float(row['overlap_ratio']) < min_overlap or row['success'].lower() != 'true':
                continue
            candidates.append((float(row['rmse_m']), int(row['protocol_index'])))
    return min(candidates)[1] if candidates else 0


def load_selected_parameters(summary_path, args):
    summary_path = Path(summary_path)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        return summary['parameters']['selected_point_to_plane_parameters']
    return {
        'voxel_size': args.voxel_size,
        'normal_radius': args.normal_radius,
        'max_correspondence_distance': args.max_correspondence_distance,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate an ETH Apartment point-to-plane ICP demo PNG.')
    parser.add_argument('--dataset', default='data/eth')
    parser.add_argument('--summary', default='results/eth/apartment_eth_summary.json')
    parser.add_argument('--metrics', default='results/eth/apartment_eth_metrics.csv')
    parser.add_argument('--output', default='results/eth/apartment_demo.png')
    parser.add_argument('--protocol-index', type=int, default=-1, help='negative means best successful pair')
    parser.add_argument('--min-overlap', type=float, default=0.30)
    parser.add_argument('--voxel-size', type=float, default=0.10)
    parser.add_argument('--normal-radius', type=float, default=0.30)
    parser.add_argument('--normal-max-nn', type=int, default=30)
    parser.add_argument('--min-distance', type=float, default=1.0)
    parser.add_argument('--max-correspondence-distance', type=float, default=0.40)
    parser.add_argument('--max-iterations', type=int, default=50)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--plot-points', type=int, default=6000)
    args = parser.parse_args()

    paths = find_apartment_paths(args.dataset)
    records = load_protocol(paths['protocol_csv'], paths['validation_csv'])
    poses = load_scanner_poses(paths['pose_csv'])
    validate_ground_truth_direction(records, poses)
    protocol_index = (
        args.protocol_index
        if args.protocol_index >= 0
        else choose_protocol_index(args.metrics, args.min_overlap)
    )
    if protocol_index >= len(records):
        raise ValueError(f'protocol-index must be below {len(records)}')
    record = records[protocol_index]
    params = load_selected_parameters(args.summary, args)

    target, target_normals, target_index = load_scan(
        paths['scan_dir'],
        record['reference'],
        params['voxel_size'],
        args.min_distance,
        params['normal_radius'],
        args.normal_max_nn,
    )
    source, _, _ = load_scan(
        paths['scan_dir'],
        record['reading'],
        params['voxel_size'],
        args.min_distance,
        params['normal_radius'],
        args.normal_max_nn,
    )
    transform, _, iterations = icp.point_to_plane_icp(
        source,
        target,
        target_normals=target_normals,
        init_pose=record['init_pose'],
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        max_correspondence_distance=params['max_correspondence_distance'],
        target_index=target_index,
    )
    initial = icp.transform_points(source, record['init_pose'])
    aligned = icp.transform_points(source, transform)

    output = save_before_after_demo(
        target,
        initial,
        aligned,
        (
            f"ETH Apartment: {record['reading']} -> {record['reference']} "
            f"(overlap {record['overlap_ratio']:.3f})"
        ),
        args.output,
        plot_points=args.plot_points,
    )

    print(f'protocol index: {protocol_index}')
    print(f"pair: {record['reading']} -> {record['reference']}")
    print(f"overlap ratio: {record['overlap_ratio']:.6f}")
    print(f"initial RMSE: {transform_rmse(source, record['init_pose'], record['gt_transform']):.6f} m")
    print(f"final RMSE:   {transform_rmse(source, transform, record['gt_transform']):.6f} m")
    print(f'iterations:   {iterations + 1}')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
