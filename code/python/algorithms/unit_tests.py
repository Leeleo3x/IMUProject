import unittest

import numpy as np
import quaternion

import geometry

class AllTests(unittest.TestCase):

    def test_quaternion_from_two_vectors(self):
        for i in range(100):
            v1 = np.random.random(3)
            v2 = np.random.random(3)
            q = geometry.quaternion_from_two_vectors(v1, v2)
            print('--------\nTest {}\nv1:{}\nv2:{}\nq:{}'.format(i, v1, v2, q))
            q1 = quaternion.quaternion(1.0, *v1)
            self.assertEqual((q * q1 * q.conjugate()).vec.all(), v2.all())
