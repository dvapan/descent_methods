import sympy
import numpy


def grad(f, x):
    g = list()
    for i in xrange(len(x)):
        partial_derivative = [0]*len(x)
        partial_derivative[i] = 1
        g.append(sympy.mpmath.diff(f, x, partial_derivative))
    return numpy.array(g)


def gesse(f, x):
    g = list()
    for i in xrange(len(x)):
        for j in xrange(len(x)):
            partial_derivative = [0]*len(x)
            partial_derivative[i] = 1
            partial_derivative[j] += 1
            g.append(sympy.mpmath.diff(f, x, partial_derivative))
    return numpy.array(g).reshape(len(x), len(x))


def newtone_descent(f, x):
    gesse_inv = numpy.linalg.inv(gesse(f, x))
    return gesse_inv.dot(grad(f, x))


def descent(f, x0, eps=1e-3, max_iteration=1e5, min_al=1e-3,
            method=grad):
    k = 1
    al = 1.0
    xk = x0
    while k < max_iteration:
        grad_f = grad(f, xk)
        square_norm_grad_f = sum(grad_f**2)
        if square_norm_grad_f < eps:
            break
        while al > min_al:
            if f(*(xk - al*grad_f)) - f(*xk) < -0.5*al*square_norm_grad_f:
                break
            al /= 2
        k += 1
        xk -= al*method(f, xk)

    return xk, f(*xk), k, al
