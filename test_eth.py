import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark_eth import (
    load_protocol,
    load_scanner_poses,
    parse_matrix,
    select_tuning_records,
    validate_ground_truth_direction,
)
from eval_utils import summarize


def matrix_fields(prefix):
    return [f'{prefix}{row}{col}' for row in range(4) for col in range(4)]


def matrix_values(matrix):
    return [f'{value:.9f}' for value in np.asarray(matrix).reshape(-1)]


class EthBenchmarkTest(unittest.TestCase):
    def test_protocol_pose_direction_overlap_filter_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / 'protocol.csv'
            validation_path = root / 'validation.csv'
            pose_path = root / 'poses.csv'

            reference_pose = np.identity(4)
            reading_pose = np.identity(4)
            reading_pose[:3, 3] = [0.4, -0.2, 0.1]
            ground_truth = np.dot(np.linalg.inv(reference_pose), reading_pose)
            initial_pose = ground_truth.copy()
            initial_pose[0, 3] += 0.05

            with pose_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['poseId', 'timestamp', *matrix_fields('T')])
                writer.writerow([0, 1.0, *matrix_values(reference_pose)])
                writer.writerow([1, 2.0, *matrix_values(reading_pose)])

            with protocol_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['reference', 'reading', *matrix_fields('iT')])
                writer.writerow(['Hokuyo_0.csv', 'Hokuyo_1.csv', *matrix_values(initial_pose)])
                writer.writerow(['Hokuyo_0.csv', 'Hokuyo_1.csv', *matrix_values(initial_pose)])

            with validation_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['overlap_ratio', 'perturbation_type', *matrix_fields('gT')])
                writer.writerow([0.75, 'easyPoses', *matrix_values(ground_truth)])
                writer.writerow([0.20, 'hardPoses', *matrix_values(ground_truth)])

            records = load_protocol(protocol_path, validation_path)
            poses = load_scanner_poses(pose_path)
            check = validate_ground_truth_direction(records, poses)
            selected = select_tuning_records(records, min_overlap=0.30, per_type=8)

            self.assertEqual(len(records), 2)
            self.assertTrue(np.allclose(parse_matrix({
                key: value for key, value in zip(matrix_fields('gT'), matrix_values(ground_truth))
            }, 'gT'), ground_truth))
            self.assertEqual(check['checked_pairs'], 1)
            self.assertLess(check['max_abs_error'], 1e-8)
            self.assertEqual([row['protocol_index'] for row in selected], [0])

            rows = [
                {'method': 'point_to_plane', 'rmse_m': 0.10, 'rre_deg': 1.0, 'rte_m': 0.1, 'runtime_s': 0.2},
                {'method': 'point_to_plane', 'rmse_m': 0.30, 'rre_deg': 9.0, 'rte_m': 0.9, 'runtime_s': 0.4},
            ]
            result = summarize(rows, rmse_threshold=0.20)['point_to_plane']
            self.assertEqual(result['pairs'], 2)
            self.assertEqual(result['successful_pairs'], 1)
            self.assertAlmostEqual(result['registration_recall'], 0.5)
            self.assertAlmostEqual(result['successful_pairs_mean_rre_deg'], 1.0)


if __name__ == '__main__':
    unittest.main()
