import numpy as np
import pandas as pd
from replica.vesga2019.objective import Objective
from joblib import Parallel, delayed

__author__ = 'Chu-Chang Ku'

to_fit_ctrl = Objective(
    prior_path='../data/vesga2019/Prior.txt',
    data_path='../data/vesga2019/Targets.csv'
)

to_fit_intv = Objective(
    prior_path='../data/vesga2019/Prior.txt',
    data_path='../data/vesga2019/Targets.csv'
)
to_fit_intv.Model.Intv = True


def fn(p, to_fit):
    p = to_fit.serve(p)
    _, ms, _ = to_fit.simulate(p)
    return ms


if __name__ == '__main__':
    out_path = '../out/vesga2019'

    post = pd.read_csv('../out/vesga2019/Post.csv')
    post = [dict(p) for _, p in post.iterrows()]

    with Parallel(n_jobs=4, verbose=8) as parallel:
        mss = parallel(delayed(fn)(p, to_fit_ctrl) for p in post)

    for i, ms in enumerate(mss):
        ms['Key'] = i

    mss = pd.concat(mss)
    mss.to_csv(f'{out_path}/Runs_Ctrl.csv')


    with Parallel(n_jobs=4, verbose=8) as parallel:
        mss = parallel(delayed(fn)(p, to_fit_intv) for p in post)

    for i, ms in enumerate(mss):
        ms['Key'] = i

    mss = pd.concat(mss)
    mss.to_csv(f'{out_path}/Runs_Intv.csv')
