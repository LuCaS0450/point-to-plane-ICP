# icp
Python implementation of m-dimensional Iterative Closest Point method.  ICP finds a best fit rigid body transformation between two point sets.  Correspondence between the points is not assumed. Included is an SVD-based least-squared best-fit algorithm for corresponding point sets.

## Point-to-plane ICP extension

This project also includes a 3D point-to-plane ICP implementation for local
rigid registration refinement:

```powershell
python test.py
python -m pip install open3d
python download_3dmatch.py --output data/3dmatch
python benchmark_3dmatch.py --dataset data/3dmatch
python demo_point_to_plane.py --dataset data/3dmatch --output results/3dmatch/redkitchen_demo.png

# Optionally open the legacy Open3D interactive viewer after saving the PNG
python demo_point_to_plane.py --dataset data/3dmatch --show-window
```

The 3DMatch benchmark script uses the local-refinement protocol described in
the report: each official ground-truth fragment transform is perturbed and then
refined with point-to-plane ICP. Results are written to `results/`. The
3DMatch and ETH demo scripts save matching before/after PNG figures with 3D
grid axes.

## ETH Apartment supplementary benchmark

The ETH supplementary experiment uses the official Apartment registration
protocol. The downloader fetches the official archive, extracts only the local
scanner-frame CSV files and required metadata, and validates that every scan
referenced by the protocol is available.

```powershell
python download_eth.py --output data/eth --keep-archives
python download_eth.py --output data/eth --validate-only
python test_eth.py

# Quick smoke run
python benchmark_eth.py --dataset data/eth --results-dir results/eth-smoke --max-rows 3

# Fixed-subset parameter tuning, then the full Apartment benchmark
python benchmark_eth.py --dataset data/eth --results-dir results/eth --tune-only
python benchmark_eth.py --dataset data/eth --results-dir results/eth --tuning-json results/eth/apartment_eth_tuning.json --workers 8

# Save a before/after Point-to-Plane ICP visualization
python demo_eth.py --dataset data/eth --output results/eth/apartment_demo.png
```

The ETH benchmark outputs `apartment_eth_metrics.csv` and
`apartment_eth_summary.json`. It verifies that the official validation
transform direction matches `inv(reference_pose) @ reading_pose`, reports both
all-protocol and overlap-filtered summaries, and includes successful-pairs-only
error statistics.
