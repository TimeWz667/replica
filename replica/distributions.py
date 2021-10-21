import scipy.stats as sts
from scipy.optimize import least_squares
import numpy as np

__author__ = 'Chu-Chang Ku'
__all__ = ['Uniform', 'Triangle', 'LogNormal', 'fit_lognorm', 'fit_beta']


class Uniform:
    def __init__(self, a, b):
        assert a < b
        self.Opt = {'loc': a, 'scale': b - a}

    def rand(self):
        return sts.uniform.rvs(**self.Opt)

    def logpdf(self, x):
        return sts.uniform.logpdf(x, **self.Opt)


class Triangle:
    def __init__(self, c, a, b):
        assert a < c < b
        scale = b - a
        self.Opt = {
            'loc': a,
            'scale': scale,
            'c': (c - a) / scale
        }

    def rand(self):
        return sts.triang.rvs(**self.Opt)

    def logpdf(self, x):
        return sts.triang.logpdf(x, **self.Opt)


class LogNormal:
    def __init__(self, s, m):
        self.Opt = {'s': s, 'scale': np.exp(m)}

    def rand(self):
        return sts.lognorm.rvs(**self.Opt)

    def logpdf(self, x):
        return sts.lognorm.logpdf(x, **self.Opt)


def fit_lognorm(qs, vs):
    qs = np.array(qs)
    vs = np.array(vs)

    def fn(x):
        m, s = x
        scale = np.exp(m)
        return sts.lognorm.cdf(vs, s=s, scale=scale) - qs

    sol = least_squares(fn, [np.log(vs[1]), 0.1], bounds=([0, 1e-10], [np.Inf, np.Inf]))

    if not sol.success:
        raise ValueError('No solution found')

    m, s = sol.x
    return sts.lognorm(s=s, scale=np.exp(m))


def fit_beta(qs, vs):
    qs = np.array(qs)
    vs = np.array(vs)

    def fn(x):
        a, b = x
        return sts.beta.cdf(vs, a=a, b=b) - qs

    sol = least_squares(fn, [1, 1], bounds=([1, 1], [np.Inf, np.Inf]))

    if not sol.success:
        raise ValueError('No solution found')

    a, b = sol.x
    return sts.beta(a=a, b=b)


if __name__ == '__main__':
    test_qs = [0.025, 0.5, 0.975]
    test_vs = 217 * np.array([0.8, 1, 1.2])

    dist = fit_lognorm(test_qs, test_vs)
    print("Data: ")
    print(test_vs)
    print("Stats from the estimated distribution")
    print(dist.ppf(test_qs))

    d = Uniform(1, 5)
    print(d)
    print(d.rand())

    test_vs = np.array([0.16, 0.2, 0.24])

    dist = fit_beta(test_qs, test_vs)
    print("Data: ")
    print(test_vs)
    print("Stats from the estimated distribution")
    print(dist.ppf(test_qs))
