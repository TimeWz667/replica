import numpy as np
import pandas as pd
from replica.vesga2019.objective import Objective
from sims_pars.fitting.abcsmc import ApproxBayesComSMC
from joblib import Parallel, delayed

__author__ = 'Chu-Chang Ku'

to_fit = Objective(
    prior_path='../data/vesga2019/Prior.txt',
    data_path='../data/vesga2019/Targets.csv'
)


if __name__ == '__main__':
    out_path = "../out/vesga2019"

    smc = ApproxBayesComSMC(max_round=15, n_collect=150, n_core=5, verbose=8)

    smc.fit(to_fit)

    post = smc.Collector

    print(smc.Monitor.Trajectories)
    smc.Monitor.save_trajectories(f'{out_path}/Trace.csv')
    post.save_to_csv(f'{out_path}/Post.csv')

    t_out = np.linspace(1970, 2020, num=int((2020 - 1970) / 0.125) + 1)

    def fn(p):
        p = to_fit.serve_from_json(p)
        _, ms, _ = to_fit.simulate(p)
        return ms

    with Parallel(n_jobs=4, verbose=8) as parallel:
        mss = parallel(delayed(fn)(p.to_json()) for p in post.ParameterList)

    for i, ms in enumerate(mss):
        ms['Key'] = i

    mss = pd.concat(mss)
    mss.to_csv(f'{out_path}/Runs_Post.csv')
