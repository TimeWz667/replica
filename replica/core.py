import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os
import json
from numba import njit

__author__ = 'Chu-Chang Ku'
__all__ = ['Parameters', 'trm2dy', 'simulate', 'output_posterior']


@njit
def trm2dy(trm, y):
    dy = np.zeros_like(y)
    ns = len(y)

    for src in range(ns):
        for tar in range(ns):
            flow = y[src] * trm[src, tar]
            dy[src] -= flow
            dy[tar] += flow

    return dy


class Parameters:
    def __init__(self, pars, transformed):
        self.Pars = pars
        self.Transformed = transformed

    def __getitem__(self, item):
        try:
            return self.Pars[item]
        except KeyError:
            return self.Transformed[item]

    def list_variables(self):
        return list(self.Pars.keys()) + list(self.Transformed.keys())

    def to_json(self):
        return dict(self.Pars)


def simulate(model, pars, y0, t_out, t_warmup=200, dfe=None):
    times = np.array(t_out)
    time0 = min(times)

    ys_wp = solve_ivp(model, [time0 - t_warmup, time0], y0, args=(pars, ), events=dfe, method="RK23")

    if len(ys_wp.t_events[0]) > 0 or not ys_wp.success:
        return None, None, {'succ': False, 'res': 'DFE reached'}

    y0 = ys_wp.y[:, -1]

    ys = solve_ivp(model, [time0, max(times)], y0, args=(pars,), events=dfe, dense_output=True)

    if len(ys.t_events[0]) > 0 or not ys.success:
        return None, None, {'succ': False, 'res': 'DFE reached'}

    ms = pd.DataFrame([model.measure(t, ys.sol(t), pars) for t in times])

    ms = ms.set_index('Time')
    msg = {'succ': True}
    return ys, ms, msg


def output_posterior(post, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(out_path + '/post.json', 'w') as f:
        json.dump(post.to_json()['Posterior'], f)

    post.DF.to_csv(out_path + '/post.csv')
#    post.Message['Trace'].to_csv(out_path + '/post_trace.csv')
