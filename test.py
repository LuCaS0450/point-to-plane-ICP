import numpy as np
import time
import icp

# Constants
N = 10                                    # number of random points in the dataset
num_tests = 100                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .1                            # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set


def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def make_cube_points(samples_per_axis=12):
    values = np.linspace(-0.5, 0.5, samples_per_axis)
    aa, bb = np.meshgrid(values, values)

    points = []
    normals = []
    for sign in (-1.0, 1.0):
        points.append(np.column_stack([np.full(aa.size, 0.5 * sign), aa.ravel(), bb.ravel()]))
        normals.append(np.tile([sign, 0.0, 0.0], (aa.size, 1)))

        points.append(np.column_stack([aa.ravel(), np.full(aa.size, 0.5 * sign), bb.ravel()]))
        normals.append(np.tile([0.0, sign, 0.0], (aa.size, 1)))

        points.append(np.column_stack([aa.ravel(), bb.ravel(), np.full(aa.size, 0.5 * sign)]))
        normals.append(np.tile([0.0, 0.0, sign], (aa.size, 1)))

    return np.vstack(points), np.vstack(normals)


def test_best_fit():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return


def test_icp():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        np.random.shuffle(B)

        # Run ICP
        start = time.time()
        T, distances, iterations = icp.icp(B, A, tolerance=0.000001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T

        assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
        assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
        assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses

    print('icp time: {:.3}'.format(total_time/num_tests))

    return


def test_point_to_plane_icp():
    target, target_normals = make_cube_points(samples_per_axis=14)

    R = rotation_matrix(np.array([0.2, 0.7, 0.4]), 0.08)
    t = np.array([0.05, -0.03, 0.04])
    T_gt = make_transform(R, t)
    source = icp.transform_points(target, np.linalg.inv(T_gt))

    R_init = rotation_matrix(np.array([0.3, 0.1, 0.5]), 0.03)
    t_init = np.array([0.02, -0.01, 0.015])
    init_pose = np.dot(make_transform(R_init, t_init), T_gt)

    start = time.time()
    T, distances, iterations = icp.point_to_plane_icp(
        source,
        target,
        target_normals=target_normals,
        init_pose=init_pose,
        max_iterations=50,
        tolerance=1e-8,
        max_correspondence_distance=0.2,
    )
    elapsed = time.time() - start

    aligned = icp.transform_points(source, T)
    assert np.mean(distances) < 0.01
    assert np.mean(np.linalg.norm(aligned - target, axis=1)) < 0.01
    assert np.allclose(T, T_gt, atol=0.03)

    print('point-to-plane icp time: {:.3}, iterations: {}'.format(elapsed, iterations + 1))

    return


if __name__ == "__main__":
    test_best_fit()
    test_icp()
    test_point_to_plane_icp()
