import argparse
import csv
import itertools
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import icp
from download_eth import find_apartment_paths
from eval_utils import (
    mean_nn_distance,
    rotation_error_deg,
    summarize,
    transform_rmse,
    translation_error,
)

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit('Open3D is required. Install it with: python -m pip install open3d') from exc


SCAN_ID_RE = re.compile(r'Hokuyo_(\d+)\.csv$', re.IGNORECASE)


def parse_matrix(row, prefix):
    return np.asarray(
        [[float(row[f'{prefix}{row_index}{col_index}']) for col_index in range(4)]
         for row_index in range(4)],
        dtype=float,
    )


def scan_id_from_name(name):
    match = SCAN_ID_RE.search(Path(name).name)
    if not match:
        raise ValueError(f'bad ETH scan filename: {name}')
    return int(match.group(1))


def load_scanner_poses(path):
    poses = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            pose_id = int(row['poseId'])
            poses[pose_id] = parse_matrix(row, 'T')
    if not poses:
        raise ValueError(f'no scanner poses found in {path}')
    return poses


def load_protocol(protocol_path, validation_path):
    with Path(protocol_path).open(newline='', encoding='utf-8-sig') as handle:
        protocol_rows = list(csv.DictReader(handle, skipinitialspace=True))
    with Path(validation_path).open(newline='', encoding='utf-8-sig') as handle:
        validation_rows = list(csv.DictReader(handle, skipinitialspace=True))
    if len(protocol_rows) != len(validation_rows):
        raise ValueError(
            f'protocol and validation row counts differ: {len(protocol_rows)} != {len(validation_rows)}'
        )

    records = []
    for index, (protocol, validation) in enumerate(zip(protocol_rows, validation_rows)):
        records.append({
            'protocol_index': index,
            'reference': protocol['reference'].strip(),
            'reading': protocol['reading'].strip(),
            'init_pose': parse_matrix(protocol, 'iT'),
            'gt_transform': parse_matrix(validation, 'gT'),
            'overlap_ratio': float(validation['overlap_ratio']),
            'perturbation_type': validation['perturbation_type'].strip(),
        })
    return records


def validate_ground_truth_direction(records, poses, tolerance=3e-2):
    # The archived scanner poses and validation transforms are rounded independently.
    maximum_error = 0.0
    checked_pairs = set()
    for record in records:
        pair = (record['reference'], record['reading'])
        if pair in checked_pairs:
            continue
        checked_pairs.add(pair)
        reference_id = scan_id_from_name(record['reference'])
        reading_id = scan_id_from_name(record['reading'])
        expected = np.dot(np.linalg.inv(poses[reference_id]), poses[reading_id])
        error = float(np.max(np.abs(expected - record['gt_transform'])))
        maximum_error = max(maximum_error, error)
        if error > tolerance:
            raise ValueError(
                f'ground-truth direction mismatch for {pair}: max abs error {error:.8f} > {tolerance}'
            )
    return {'checked_pairs': len(checked_pairs), 'max_abs_error': maximum_error}


def load_scan(scan_dir, filename, voxel_size, min_distance, normal_radius, normal_max_nn):
    path = Path(scan_dir) / filename
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open('r', encoding='utf-8-sig') as handle:
        header = [column.strip().lower() for column in handle.readline().split(',')]
    try:
        xyz_columns = tuple(header.index(axis) for axis in ('x', 'y', 'z'))
    except ValueError as exc:
        raise ValueError(f'{path} must contain x, y, z columns; got {header}') from exc

    points = np.loadtxt(path, delimiter=',', skiprows=1, usecols=xyz_columns, dtype=float)
    points = np.atleast_2d(points)
    points = points[np.all(np.isfinite(points), axis=1)]
    if min_distance > 0:
        points = points[np.linalg.norm(points, axis=1) >= min_distance]
    if points.shape[0] < 3:
        raise ValueError(f'not enough valid points in {path}')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    points = np.asarray(pcd.points, dtype=float)
    normals = np.asarray(pcd.normals, dtype=float)
    return points, normals, icp.build_nearest_neighbor_index(points)


def evaluate_pair(method, source, target, target_normals, target_index, record, params, args):
    start = time.perf_counter()
    if method == 'point_to_plane':
        transform, distances, iterations, history = icp.point_to_plane_icp(
            source,
            target,
            target_normals=target_normals,
            init_pose=record['init_pose'],
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            max_correspondence_distance=params['max_correspondence_distance'],
            return_history=True,
            target_index=target_index,
        )
        residual = history[-1]['mean_abs_point_to_plane_error'] if history else float('nan')
    elif method == 'point_to_point':
        transform, distances, iterations = icp.icp(
            source,
            target,
            init_pose=record['init_pose'],
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            target_index=target_index,
        )
        residual = float('nan')
    else:
        raise ValueError(method)

    runtime = time.perf_counter() - start
    aligned = icp.transform_points(source, transform)
    rmse = transform_rmse(source, transform, record['gt_transform'])
    return {
        'protocol_index': int(record['protocol_index']),
        'reference': record['reference'],
        'reading': record['reading'],
        'perturbation_type': record['perturbation_type'],
        'overlap_ratio': float(record['overlap_ratio']),
        'method': method,
        'initial_rmse_m': transform_rmse(source, record['init_pose'], record['gt_transform']),
        'rre_deg': rotation_error_deg(transform, record['gt_transform']),
        'rte_m': translation_error(transform, record['gt_transform']),
        'rmse_m': rmse,
        'success': bool(rmse < args.rmse_threshold),
        'mean_nn_distance_m': mean_nn_distance(aligned, target),
        'mean_euclidean_distance_m': float(np.mean(distances)),
        'mean_abs_point_to_plane_error_m': residual,
        'iterations': int(iterations + 1),
        'runtime_s': runtime,
    }


def evaluate_records_serial(records, scan_dir, methods, params, args, label):
    cache = {}
    rows = []
    for offset, record in enumerate(records, start=1):
        for filename in (record['reference'], record['reading']):
            key = (
                filename,
                params['voxel_size'],
                args.min_distance,
                params['normal_radius'],
                args.normal_max_nn,
            )
            if key not in cache:
                cache[key] = load_scan(
                    scan_dir,
                    filename,
                    params['voxel_size'],
                    args.min_distance,
                    params['normal_radius'],
                    args.normal_max_nn,
                )

        target, target_normals, target_index = cache[(
            record['reference'],
            params['voxel_size'],
            args.min_distance,
            params['normal_radius'],
            args.normal_max_nn,
        )]
        source, _, _ = cache[(
            record['reading'],
            params['voxel_size'],
            args.min_distance,
            params['normal_radius'],
            args.normal_max_nn,
        )]
        for method in methods:
            rows.append(
                evaluate_pair(
                    method, source, target, target_normals, target_index, record, params, args
                )
            )
        if args.progress_every > 0 and (offset % args.progress_every == 0 or offset == len(records)):
            print(f'{label}: {offset}/{len(records)} protocol rows')
    return rows


def evaluate_record_group(job):
    records, scan_dir, methods, params, args, label = job
    return evaluate_records_serial(records, scan_dir, methods, params, args, label)


def evaluate_records(records, scan_dir, methods, params, args, label):
    if args.workers <= 1 or len(records) <= 1:
        return evaluate_records_serial(records, scan_dir, methods, params, args, label)

    groups = {}
    for record in records:
        key = (record['reference'], record['reading'])
        groups.setdefault(key, []).append(record)
    jobs = [
        (group, scan_dir, methods, params, args, f'{label} {index}/{len(groups)}')
        for index, group in enumerate(groups.values(), start=1)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, group_rows in enumerate(executor.map(evaluate_record_group, jobs), start=1):
            rows.extend(group_rows)
            print(f'{label}: completed pair group {index}/{len(jobs)}')
    rows.sort(key=lambda row: (row['protocol_index'], row['method']))
    return rows


def select_tuning_records(records, min_overlap, per_type):
    selected = []
    counts = {}
    for record in records:
        if record['overlap_ratio'] < min_overlap:
            continue
        kind = record['perturbation_type']
        if counts.get(kind, 0) >= per_type:
            continue
        selected.append(record)
        counts[kind] = counts.get(kind, 0) + 1
    return selected


def tune_parameters(records, scan_dir, args):
    tuning_records = select_tuning_records(records, args.min_overlap, args.tune_per_type)
    if not tuning_records:
        raise ValueError('no protocol rows available for tuning')
    candidates = []
    for voxel_size, radius_factor, max_correspondence_distance in itertools.product(
        (0.05, 0.10, 0.20),
        (2.0, 3.0),
        (0.20, 0.40, 0.80),
    ):
        params = {
            'voxel_size': voxel_size,
            'normal_radius': voxel_size * radius_factor,
            'max_correspondence_distance': max_correspondence_distance,
        }
        label = (
            f"tune voxel={voxel_size:.2f} radius={params['normal_radius']:.2f} "
            f"correspondence={max_correspondence_distance:.2f}"
        )
        rows = evaluate_records(tuning_records, scan_dir, ['point_to_plane'], params, args, label)
        result = summarize(rows, args.rmse_threshold)['point_to_plane']
        candidates.append({'parameters': params, 'summary': result})
        print(
            f"{label}: recall={result['registration_recall']:.4f}, "
            f"median_rmse={result['median_rmse_m']:.4f}, "
            f"runtime={result['mean_runtime_s']:.4f}s"
        )
    candidates.sort(key=lambda item: (
        -item['summary']['registration_recall'],
        item['summary']['median_rmse_m'],
        item['summary']['mean_runtime_s'],
    ))
    return {
        'selection_rule': 'highest recall, then lowest median RMSE, then lowest mean runtime',
        'subset_rows': len(tuning_records),
        'per_perturbation_type_limit': args.tune_per_type,
        'candidates': candidates,
        'selected_parameters': candidates[0]['parameters'],
    }


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'protocol_index', 'reference', 'reading', 'perturbation_type', 'overlap_ratio',
        'method', 'initial_rmse_m', 'rre_deg', 'rte_m', 'rmse_m', 'success',
        'mean_nn_distance_m', 'mean_euclidean_distance_m',
        'mean_abs_point_to_plane_error_m', 'iterations', 'runtime_s',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ICP on the official ETH Apartment protocol.')
    parser.add_argument('--dataset', default='data/eth')
    parser.add_argument('--results-dir', default='results/eth')
    parser.add_argument('--method', choices=['point_to_plane', 'point_to_point', 'both'], default='both')
    parser.add_argument('--voxel-size', type=float, default=0.10)
    parser.add_argument('--normal-radius', type=float, default=0.30)
    parser.add_argument('--normal-max-nn', type=int, default=30)
    parser.add_argument('--min-distance', type=float, default=1.0)
    parser.add_argument('--max-correspondence-distance', type=float, default=0.40)
    parser.add_argument('--max-iterations', type=int, default=50)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--rmse-threshold', type=float, default=0.20)
    parser.add_argument('--min-overlap', type=float, default=0.30)
    parser.add_argument('--tune', action='store_true', help='select point-to-plane parameters on a fixed subset')
    parser.add_argument('--tune-only', action='store_true', help='save fixed-subset tuning result and stop')
    parser.add_argument('--tuning-json', help='reuse selected parameters from a previous tune-only run')
    parser.add_argument('--tune-per-type', type=int, default=8)
    parser.add_argument('--max-rows', type=int, default=0, help='0 means all official protocol rows')
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--workers', type=int, default=1, help='parallel scan-pair worker processes')
    args = parser.parse_args()

    paths = find_apartment_paths(args.dataset)
    records = load_protocol(paths['protocol_csv'], paths['validation_csv'])
    poses = load_scanner_poses(paths['pose_csv'])
    direction_check = validate_ground_truth_direction(records, poses)
    print(
        f"ground-truth direction passed: {direction_check['checked_pairs']} unique pairs, "
        f"max abs error {direction_check['max_abs_error']:.8f}"
    )
    protocol_count = len(records)
    filtered_protocol_count = sum(row['overlap_ratio'] >= args.min_overlap for row in records)
    if args.max_rows > 0:
        records = records[:args.max_rows]

    results_dir = Path(args.results_dir)
    tuning = None
    params = {
        'voxel_size': args.voxel_size,
        'normal_radius': args.normal_radius,
        'max_correspondence_distance': args.max_correspondence_distance,
    }
    if args.tuning_json:
        tuning = json.loads(Path(args.tuning_json).read_text(encoding='utf-8'))
        params = tuning['selected_parameters']
        print(f"loaded selected parameters from {args.tuning_json}: {json.dumps(params, sort_keys=True)}")
    elif args.tune or args.tune_only:
        tuning = tune_parameters(records, paths['scan_dir'], args)
        params = tuning['selected_parameters']
        print(f'selected parameters: {json.dumps(params, sort_keys=True)}')
        results_dir.mkdir(parents=True, exist_ok=True)
        tuning_path = results_dir / 'apartment_eth_tuning.json'
        tuning_path.write_text(json.dumps(tuning, indent=2), encoding='utf-8')
        print(f'wrote {tuning_path}')
        if args.tune_only:
            return

    methods = ['point_to_plane', 'point_to_point'] if args.method == 'both' else [args.method]
    rows = evaluate_records(records, paths['scan_dir'], methods, params, args, 'benchmark')
    filtered_rows = [row for row in rows if row['overlap_ratio'] >= args.min_overlap]

    csv_path = results_dir / 'apartment_eth_metrics.csv'
    json_path = results_dir / 'apartment_eth_summary.json'
    write_csv(rows, csv_path)
    summary = {
        'dataset': str(Path(args.dataset)),
        'scene': 'Apartment',
        'protocol': 'official ETH Apartment protocol iT initial poses with validation gT ground truth',
        'parameters': {
            **vars(args),
            'selected_point_to_plane_parameters': params,
        },
        'ground_truth_direction_check': direction_check,
        'counts': {
            'official_protocol_rows': protocol_count,
            'official_protocol_rows_meeting_overlap_threshold': filtered_protocol_count,
            'evaluated_protocol_rows': len(records),
            'evaluated_metric_rows': len(rows),
            'evaluated_metric_rows_meeting_overlap_threshold': len(filtered_rows),
        },
        'tuning': tuning,
        'summary_all': summarize(rows, args.rmse_threshold),
        'summary_overlap_filtered': summarize(filtered_rows, args.rmse_threshold),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'wrote {csv_path}')
    print(f'wrote {json_path}')
    print(json.dumps(summary['summary_overlap_filtered'], indent=2))


if __name__ == '__main__':
    main()
