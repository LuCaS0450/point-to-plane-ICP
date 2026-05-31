import numpy as np
from sklearn.neighbors import NearestNeighbors


def _check_points(points, name):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'{name} must be an Nx3 array')
    if points.shape[0] < 3:
        raise ValueError(f'{name} must contain at least three points')
    return points


def skew(vector):
    '''
    Return the 3x3 skew-symmetric matrix for a 3-vector.
    '''
    x, y, z = vector
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def rotation_vector_to_matrix(rotation_vector):
    '''
    Convert an axis-angle rotation vector to a 3x3 rotation matrix.
    '''
    rotation_vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    theta = np.linalg.norm(rotation_vector)
    if theta < 1e-12:
        return np.identity(3) + skew(rotation_vector)

    axis = rotation_vector / theta
    K = skew(axis)
    return (np.identity(3) +
            np.sin(theta) * K +
            (1.0 - np.cos(theta)) * np.dot(K, K))


def pose_from_increment(increment):
    '''
    Build a 4x4 homogeneous transform from a small se(3) increment.

    The first three values are the rotation vector and the final three
    values are translation.
    '''
    increment = np.asarray(increment, dtype=float).reshape(6)
    T = np.identity(4)
    T[:3, :3] = rotation_vector_to_matrix(increment[:3])
    T[:3, 3] = increment[3:]
    return T


def transform_points(points, transform):
    '''
    Apply a 4x4 homogeneous transform to an Nx3 point array.
    '''
    points = _check_points(points, 'points')
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError('transform must be a 4x4 matrix')
    return np.dot(transform[:3, :3], points.T).T + transform[:3, 3]


def estimate_normals(points, k=30):
    '''
    Estimate point normals with local PCA.

    Normals are not globally oriented. This is acceptable for point-to-plane
    ICP because each residual equation is unchanged by flipping a normal sign.
    '''
    points = _check_points(points, 'points')
    k = int(min(max(k, 3), points.shape[0]))

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points)
    _, indices = neigh.kneighbors(points, return_distance=True)

    normals = np.empty_like(points)
    for row, nn_indices in enumerate(indices):
        neighborhood = points[nn_indices]
        centered = neighborhood - np.mean(neighborhood, axis=0)
        covariance = np.dot(centered.T, centered)
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        norm = np.linalg.norm(normal)
        normals[row] = normal / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    return normals


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def build_nearest_neighbor_index(dst):
    '''
    Build a reusable nearest-neighbor index for a destination point set.
    '''
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    return neigh


def nearest_neighbor(src, dst, index=None):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    if src.shape[1] != dst.shape[1]:
        raise ValueError('src and dst must have the same point dimension')

    if index is not None:
        distances, indices = index.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001,
        target_index=None):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
        target_index: optional reusable nearest-neighbor index for B
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    if A.shape[1] != B.shape[1]:
        raise ValueError('A and B must have the same point dimension')

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    if target_index is None:
        target_index = build_nearest_neighbor_index(dst[:m,:].T)
    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T, target_index)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


def point_to_plane_icp(A, B, target_normals=None, init_pose=None,
                       max_iterations=50, tolerance=1e-6,
                       max_correspondence_distance=np.inf,
                       min_correspondences=6, return_history=False,
                       target_index=None):
    '''
    Point-to-plane Iterative Closest Point for 3D rigid registration.

    Input:
        A: Nx3 source points
        B: Mx3 target points
        target_normals: Mx3 target normals. If omitted, normals are estimated
            from B by local PCA.
        init_pose: optional 4x4 initial transformation from A to B
        max_iterations: maximum ICP iterations
        tolerance: convergence threshold on mean absolute point-to-plane error
        max_correspondence_distance: reject pairs above this Euclidean distance
        min_correspondences: minimum valid pairs required to solve the system
        return_history: if True, append a list of per-iteration diagnostics
        target_index: optional reusable nearest-neighbor index for B
    Output:
        T: 4x4 final homogeneous transformation mapping A onto B
        distances: final nearest-neighbor Euclidean distances
        i: number of completed iterations minus one, matching icp()
        history: optional list of dicts when return_history is True
    '''
    A = _check_points(A, 'A')
    B = _check_points(B, 'B')

    if target_normals is None:
        target_normals = estimate_normals(B)
    target_normals = np.asarray(target_normals, dtype=float)
    if target_normals.shape != B.shape:
        raise ValueError('target_normals must have the same shape as B')

    normal_lengths = np.linalg.norm(target_normals, axis=1)
    valid_normals = normal_lengths > 1e-12
    if not np.all(valid_normals):
        target_normals = target_normals.copy()
        target_normals[~valid_normals] = np.array([0.0, 0.0, 1.0])
        normal_lengths = np.linalg.norm(target_normals, axis=1)
    target_normals = target_normals / normal_lengths[:, None]

    if init_pose is None:
        T_total = np.identity(4)
    else:
        T_total = np.asarray(init_pose, dtype=float).copy()
        if T_total.shape != (4, 4):
            raise ValueError('init_pose must be a 4x4 matrix')

    src = transform_points(A, T_total)
    if target_index is None:
        target_index = build_nearest_neighbor_index(B)
    prev_error = None
    history = []
    distances = np.full(A.shape[0], np.inf)
    i = -1

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src, B, target_index)
        mask = distances <= max_correspondence_distance
        if np.count_nonzero(mask) < min_correspondences:
            break

        src_corr = src[mask]
        dst_corr = B[indices[mask]]
        normal_corr = target_normals[indices[mask]]
        residuals = np.sum(normal_corr * (src_corr - dst_corr), axis=1)

        # Linearized residual:
        # n.T * ((w x p) + v + p - q) = 0
        lhs = np.empty((src_corr.shape[0], 6))
        lhs[:, :3] = np.cross(src_corr, normal_corr)
        lhs[:, 3:] = normal_corr
        rhs = -residuals

        increment, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
        delta = pose_from_increment(increment)
        T_total = np.dot(delta, T_total)
        src = transform_points(A, T_total)

        mean_error = float(np.mean(np.abs(residuals)))
        history.append({
            'iteration': i,
            'mean_abs_point_to_plane_error': mean_error,
            'mean_euclidean_distance': float(np.mean(distances[mask])),
            'correspondences': int(np.count_nonzero(mask)),
            'increment_norm': float(np.linalg.norm(increment)),
        })

        if prev_error is not None and abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    distances, _ = nearest_neighbor(src, B, target_index)
    if return_history:
        return T_total, distances, i, history
    return T_total, distances, i
