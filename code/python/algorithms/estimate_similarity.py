import numpy as np
import quaternion
from scipy.optimize import least_squares as least_squares


def similarity_functor(x, *args, **kwargs):

    pass

def estimate_similarity(pa, pb, refine=False):
    """
    Estimate a similarity transformation from pa to pb
    :param pa:
    :param pb:
    :return: 3x4 matrix encoding the transformation
    """
    assert(pa.shape, pb.shape)
    # first compute an initial guess

    # scale is estimated by the distance of first and last point
    scale = np.linalg.norm((pb[-1] - pb[0])) / np.linalg.norm(pa[-1] - pa[0])
    pa *= scale
    translation = np.average(pb, axis=0) - np.average(pa, axis=0)
    pa += translation - np.average(pa, axis=0)
    pb -= np.average(pb, axis=0)
    U, s, V = np.linalg.svd(np.matmul(pa.transpose(), pb), True)
    rot = U * V

    print('estimated scale: ', scale)
    print('estimated translation: ', translation)
    print('estimated rotation: ', rot)

    result = np.zeros([3, 4], dtype=float)
    result[:3, :3] = rot
    result[:, 3] = translation
    return result * scale

if __name__ == '__main__':

    import random
    # test_scale = random.uniform(0, 1)
    test_scale = 1.0
    test_translation = np.random.random(3)
    test_rotation = quaternion.as_rotation_matrix(quaternion.quaternion(random.uniform(0, 0.2), random.uniform(0, 0.2),
                                                                        random.uniform(0, 0.2), random.uniform(0, 0.2)))

    print('test_scale: ', test_scale)
    print('test_translation: ', test_translation)
    print('test_rotation: ', test_rotation)

    target = np.random.random([100, 3])
    source = target.copy()

    for i in range(source.shape[0]):
        source[i] = test_scale * (np.matmul(test_rotation, target[i].transpose()) + test_translation).flatten()

    estimated_transform = estimate_similarity(source, target)

    # print('Estimated transform:')
    # print(estimated_transform)


