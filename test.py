import unittest

from main import *

class TestDescentMethods(unittest.TestCase):
    def test_gradient_f(self):
        set_parameters(3)
        grad_f = grad(x[0]**3 + x[0]*x[1] + x[2]**2, (1, 1, 1))
        self.assertEquals([4, 1, 1], list(grad_f))

    def test_gesse_f(self):
        set_parameters(3)
        gesse_f = gessian(x[0]**3 + x[0]*x[1] + x[2]**2, (1, 1, 1))
        test_gesse = [[6, 1, 0], [1, 0, 0], [0, 0, 2]]
        for i in xrange(3):
            for j in xrange(3):
                self.assertEqual(test_gesse[i][j], gesse_f[i, j])

if __name__ == '__main__':
    unittest.main()
