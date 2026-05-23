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
python demo_point_to_plane.py --dataset data/3dmatch
```

The 3DMatch benchmark script uses the local-refinement protocol described in
the report: each official ground-truth fragment transform is perturbed and then
refined with point-to-plane ICP. Results are written to `results/`.
