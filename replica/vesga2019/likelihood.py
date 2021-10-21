from replica.distributions import *
import numpy as np
import pandas as pd


__author__ = 'Chu-Chang Ku'
__all__ = ['find_likelihood_function']


def find_likelihood_function(path='../../data/Targets_original.csv', frac=True):
    dat = pd.read_csv(path)

    qs = np.array([0.025, 0.05, 0.975])
    ds = dict()
    for _, row in dat.iterrows():
        if row.Group == "Epi":
            ds[row.Index] = fit_lognorm(qs, np.array([row.L, row.M, row.U]) * 1e5)
        else:
            ds[row.Index] = fit_beta(qs, np.array([row.L, row.M, row.U]))

    def fn_li(ms, ti=2016):
        sv = ms[ms.index == ti]
        li = (ds['Prevalence'].logpdf(sv.PrevTB * 1e5) +
              ds['Incidence'].logpdf(sv.IncTB * 1e5) +
              ds['Mortality'].logpdf(sv.MorTB * 1e5))[0]

        if frac:
            li += (ds['PrAsym'].logpdf(sv.PrAsym) +
                   ds['PrSym'].logpdf(sv.PrSym) +
                   ds['PrCS'].logpdf(sv.PrCS))[0]
        return li

    return fn_li


if __name__ == '__main__':
    sims = pd.DataFrame({
        'Time': [2015, 2016],
        'PrevTB': 266 * 1e-5,
        'IncTB': 211 * 1e-5,
        'MorTB': 33 * 1e-5,
        'PrAsym': 0.2,
        'PrSym': 0.2,
        'PrCS': 0.2,
        'PrTxPub': 0.2,
        'PrTxPri': 0.2,
    })
    sims = sims.set_index('Time')

    fn_li = find_likelihood_function()

    print(sims)
    print(fn_li(sims, 2016))

    sims_low = pd.DataFrame({
        'Time': [2015, 2016],
        'PrevTB': 2,
        'IncTB': 2,
        'MorTB': 2,
        'PrAsym': 0.1,
        'PrSym': 0.1,
        'PrCS': 0.1,
        'PrTxPub': 0.1,
        'PrTxPri': 0.1,
    })
    sims_low = sims_low.set_index('Time')

    print(sims_low)
    print(fn_li(sims_low, 2016))
