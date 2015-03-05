import sympy
import numpy

PARAM_COUNT = 3
x = sympy.symbols('x:{0}'.format(PARAM_COUNT))


def set_parameters(n):
    global x, PARAM_COUNT
    x = sympy.symbols('x:{0}'.format(n))
    PARAM_COUNT = n


def eval_func(f, val):
    # x = sympy.symbols('x:{0}'.format(n))
    return f.subs([(x[i], val[i]) for i in range(PARAM_COUNT)])


def symbol_grad(f):
    return [f.diff(x[i]) for i in range(PARAM_COUNT)]


def grad(f, val):
    g = symbol_grad(f)
    evaluted_g = [eval_func(g[i], val) for i in range(PARAM_COUNT)]
    return numpy.array(evaluted_g)


def symbol_gessian(f):
    g = symbol_grad(f)
    gessian = [symbol_grad(g[i]) for i in range(PARAM_COUNT)]
    return gessian


def gessian(f, val):
    gessian = symbol_gessian(f)
    ev_gessian = [[eval_func(gessian[i][j], val) for i in range(PARAM_COUNT)]
                  for j in range(PARAM_COUNT)]
    return numpy.matrix(ev_gessian)


def newtone_descent(f, x):
    gesse_inv = numpy.linalg.inv(gessian(f, x))
    return numpy.array(gesse_inv.dot(grad(f, x)))[0]


def descent(obj_f, x0, eps=1e-3, max_iteration=1e5, min_al=1e-3,
            method=grad):
    def f(*args):
        return eval_func(obj_f, args)
    k = 1
    al = 1.0
    xk = x0
    while k < max_iteration:
        grad_f = grad(obj_f, xk)
        square_norm_grad_f = sum(grad_f**2)
        if square_norm_grad_f < eps:
            break
        al = 1.0
        while al > min_al:
            if f(*(xk - al*grad_f)) - f(*xk) < -0.5*al*square_norm_grad_f:
                break
            al /= 2
        k += 1
        xk -= al*method(obj_f, xk)

    return xk, f(*xk), k, al
