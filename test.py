import unittest

from main import *

def test_function(x, y):
    return 0.5*(y-x**2)**2 + 0.5*(y + x**3 - 1)**2


class TestDescentMethods(unittest.TestCase):
    def test_gradient_f(self):
        grad_f = grad(lambda x, y, z: x**3 + x*y + z, (1, 1, 1))
        self.assertEquals([4, 1, 1], list(grad_f))

    def test_gesse_f(self):
        gesse_f = gesse(lambda x, y, z: x**3 + x*y + z**2, (1, 1, 1))
        test_gesse = [[6, 1, 0], [1, 0, 0], [0, 0, 2]]
        for i in xrange(len(gesse_f)):
            for j in xrange(len(gesse_f[i])):
                self.assertEqual(test_gesse[i][j], gesse_f[i][j])

    def test_descent_method(self):
        eps = 0.001
        res = descent(test_function, (1, 1), eps=eps, method=newtone_descent)
        self.assertLessEqual(res[1], eps)


if __name__ == '__main__':
    unittest.main()
