import unittest
import numpy as np
import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the directory that is located before the current working directory to the Python path
sys.path.append(os.path.join(script_dir, '..'))

from zq import zq

class TestZQFunction(unittest.TestCase):
    def test_zq_function(self):
        M = 6
        gamma = 0.5
        # from matlab/zq.m results 
        expected_z = np.array([-1.8508   ,-0.8860  , -0.2859 ,   0.2859 ,   0.8860 ,   1.8508])
        expected_q = np.array([    0.1244 ,   0.0612 ,   0.3144 ,   0.3144 ,   0.0612 ,   0.1244])
        expected_vzq =      1

        expected_kzq = 3

        z, q, vzq, kzq = zq(M, gamma)

        np.testing.assert_array_almost_equal(z, expected_z)
        np.testing.assert_array_almost_equal(q, expected_q)
        self.assertAlmostEqual(vzq, expected_vzq)
        self.assertAlmostEqual(kzq, expected_kzq)

if __name__ == '__main__':
    unittest.main()
