import numpy as np
import pandas as pd
from replica import output_posterior
from replica.vesga2019.obj import Fittable
from fitter import ApproxBayesComSMC
from joblib import Parallel, delayed
import logging

__author__ = 'Chu-Chang Ku'

logging.basicConfig(level=logging.INFO,
                    datefmt='%H:%M:%S',
                    format='%(asctime)s-%(name)s-%(levelname)s: %(message)s')

objective = Fittable("../data/vesga2019/Targets.csv")


if __name__ == '__main__':
    out_path = "../out/vesga2019"

    smc = ApproxBayesComSMC(max_round=25, n_collect=200, n_core=5, verbose=8)
    smc.parallel_on()

    smc.fit(objective)

    post = smc.Posterior

    print(post.Message['Trace'])

    output_posterior(post, out_path)

    t_out = np.linspace(1970, 2020, num=int((2020 - 1970) / 0.125) + 1)

    def fn(p):
        _, ms, _ = objective.simulate(p)
        return ms

    with Parallel(n_jobs=4, verbose=8) as parallel:
        mss = parallel(delayed(fn)(p) for p in post.Posterior)

    for i, ms in enumerate(mss):
        ms['Key'] = i

    mss = pd.concat(mss)
    mss.to_csv(f'{out_path}/Runs_Post.csv')
