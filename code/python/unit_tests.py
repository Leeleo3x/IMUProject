import unittest
import numpy as np
import genDataset


class AllTests(unittest.TestCase):

    def test_rotation_from_vectors(self):
        for i in range(100):
            v1 = np.random.random(3)
            v2 = np.random.random(3)
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            R = genDataset.rotationMatrixFromUnitVectors(v1, v2)
            print('--------\nTest {}\nv1:{}\nv2:{}\nR:{}'.format(i, v1, v2, R))
            self.assertEqual(np.dot(v1, R).all(), v2.all())
