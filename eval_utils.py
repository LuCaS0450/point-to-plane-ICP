import math

import numpy as np

import icp


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
        successful_rre = rre[success]
        successful_rte = rte[success]

        def successful_stat(values, reducer):
            return float(reducer(values)) if values.size else None

        summary[method] = {
            'pairs': int(len(method_rows)),
            'registration_recall': float(np.mean(success)),
            'precision_on_evaluated_positive_pairs': 1.0,
            'rmse_threshold_m': float(rmse_threshold),
            'mean_rmse_m': float(np.mean(rmse)),
            'median_rmse_m': float(np.median(rmse)),
            'mean_rre_deg': float(np.mean(rre)),
            'median_rre_deg': float(np.median(rre)),
            'mean_rte_m': float(np.mean(rte)),
            'median_rte_m': float(np.median(rte)),
            'successful_pairs': int(np.sum(success)),
            'successful_pairs_mean_rre_deg': successful_stat(successful_rre, np.mean),
            'successful_pairs_median_rre_deg': successful_stat(successful_rre, np.median),
            'successful_pairs_mean_rte_m': successful_stat(successful_rte, np.mean),
            'successful_pairs_median_rte_m': successful_stat(successful_rte, np.median),
            'mean_runtime_s': float(np.mean(runtime)),
        }
    return summary
