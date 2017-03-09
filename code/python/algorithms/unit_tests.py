import unittest

import numpy as np
import quaternion

import geometry

class AllTests(unittest.TestCase):

    def test_rotation_from_vectors(self):
        for i in range(100):
            v1 = np.random.random(3)
            v2 = np.random.random(3)
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            R = gen_dataset.rotationMatrixFromTwoVectors(v1, v2)
            print('--------\nTest {}\nv1:{}\nv2:{}\nR:{}'.format(i, v1, v2, R))
            self.assertEqual(np.dot(v1, R).all(), v2.all())

    def test_quaternion_from_two_vectors(self):
        for i in range(100):
            v1 = np.random.random(3)
            v2 = np.random.random(3)
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            q = gen_dataset.quaternionFromTwoVectors(v1, v2)
            print('--------\nTest {}\nv1:{}\nv2:{}\nq:{}'.format(i, v1, v2, q))
            q1 = quaternion.quaternion(0.0, *v1)
            self.assertEqual((q * q1 * q.conjugate()).vec.all(), v2.all())
