from replica.vesga2019 import *
from replica import simulate
import numpy as np
import pandas as pd
import json
from joblib import Parallel, delayed

__author__ = 'Chu-Chang Ku'


params = ParameterDefinition()
model_ctrl = ModelIndia()
model_intv = ModelIndia()
model_intv.Intv = True

out_path = '../out/vesga2019'
t0 = 1970
t1 = 2036
t_out = np.linspace(t0, t1, num=int((t1 - t0) / 0.5) + 1)


def fn(p, model):
    p = params.draw(**p)
    _, ms, _ = simulate(model, p, Y0, t_out, dfe=dfe)
    return ms


if __name__ == '__main__':
    # Load posterior parameters
    with open(f'{out_path}/post.json', 'r') as f:
        post = json.load(f)

    with Parallel(n_jobs=5, verbose=8) as parallel:
        mss_ctrl = parallel(delayed(fn)(pars, model_ctrl) for pars in post)

    for i, ms in enumerate(mss_ctrl):
        ms['Key'] = i

    mss_ctrl = pd.concat(mss_ctrl)
    mss_ctrl.to_csv(f"{out_path}/Runs_Ctrl.csv")

    with Parallel(n_jobs=5, verbose=8) as parallel:
        mss_intv = parallel(delayed(fn)(pars, model_intv) for pars in post)

    for i, ms in enumerate(mss_intv):
        ms['Key'] = i

    mss_intv = pd.concat(mss_intv)
    mss_intv.to_csv(f"{out_path}/Runs_Intv.csv")
