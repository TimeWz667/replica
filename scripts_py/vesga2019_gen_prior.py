from replica.vesga2019 import *
from replica import simulate
import numpy as np
import numpy.random as rd
import pandas as pd
from joblib import Parallel, delayed


params = ParameterDefinition()
model = ModelIndia()
t0 = 1970
t_out = np.linspace(t0, 2020, num=int((2020 - t0) / 0.125) + 1)


def fn(i):
    rd.seed(i)
    p = params.draw()
    _, ms, msg = simulate(model, p, Y0, t_out, dfe=dfe)

    while not msg['succ']:
        p = params.draw()
        _, ms, msg = simulate(model, p, Y0, t_out, dfe=dfe)
    return ms, msg


if __name__ == '__main__':
    n_sample = 300

    with Parallel(n_jobs=5, verbose=8) as parallel:
        res = parallel(delayed(fn)(i) for i in range(n_sample))

    mss = [ms for ms, _ in res]

    for i, ms in enumerate(mss):
        ms['Key'] = i

    mss = pd.concat(mss)

    out_path = "../out/vesga2019"
    mss.to_csv(f"{out_path}/Runs_Prior.csv")
