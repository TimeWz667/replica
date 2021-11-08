import numpy as np
from replica.vesga2019.keys import *
from replica import trm2dy

__all__ = ['ModelIndia', 'dfe', 'Y0']


class ModelIndia:
    def __init__(self):
        self.T0_Growth = 1970
        self.T0_PubScaleUp = 1990
        self.T1_PubScaleUp = 1997
        self.T0_IntvScaleUp = 2018
        self.T1_IntvScaleUp = 2021

        self.Intv = False

    def calc_foi(self, y, pars):
        return (pars['trans'] * y).sum() / y.sum()

    def transmission(self, y, pars):
        dy = np.zeros(N_State)

        infection = self.calc_foi(y, pars) * pars['sus'] * y
        dy -= infection

        infection = infection.sum()

        dy[I_Lat] += infection * (1 - pars['p_fast'])
        dy[I_Asym] += infection * pars['p_fast']

        return dy

    def progression(self, y, pars):  # Natural history of TB
        return trm2dy(pars['Progression'], y)

    def healthcare(self, t, y, pars):
        if t <= self.T0_PubScaleUp:
            trm = pars['CareInitial']
        elif t <= self.T1_PubScaleUp:
            md = pars['Care'] - pars['CareInitial']
            scale = (t - self.T0_PubScaleUp) / (self.T1_PubScaleUp - self.T0_PubScaleUp)
            trm = pars['CareInitial'] + md * scale
        else:
            if not self.Intv:
                trm = pars['Care']
            else:  # Intervention scenario
                if t <= self.T0_IntvScaleUp:
                    trm = pars['Care']
                elif t <= self.T1_IntvScaleUp:
                    md = pars['CareIntv'] - pars['Care']
                    scale = (t - self.T0_IntvScaleUp) / (self.T1_IntvScaleUp - self.T0_IntvScaleUp)
                    trm = pars['Care'] + md * scale
                else:
                    trm = pars['CareIntv']

        dy = trm2dy(trm, y)
        return dy

    def demography(self, t, y, pars):
        dy = trm2dy(pars['Demography'], y)
        if t >= self.T0_Growth and pars['r_growth'] > 0:
            dy[I_U] += pars['r_growth'] * y.sum()
        return dy

    def __call__(self, t, y, pars):
        dy = self.demography(t, y, pars)
        dy += self.progression(y, pars)
        dy += self.healthcare(t, y, pars)
        dy += self.transmission(y, pars)
        return dy

    def measure(self, t, y, pars):
        n = y.sum()

        prev = y[I_Prevalent].sum()

        mea = {
            'Time': t,
            'N': n,
            'PrevTB': prev / n,
            'PrAsym': y[I_Asym] / prev,
            'PrSym': y[I_Sym] / prev,
            'PrCS': y[[I_DPub, I_DPri, I_E]].sum() / prev,
            'PrTxPub': y[I_TPub] / prev,
            'PrTxPri': y[I_TPri] / prev,
            'IncTB': y[I_Asym] * pars['r_sym'] / n,
            'MorTB': (y[I_Infectious] * pars['r_death_tb']).sum() / n,
            'NetDemo': self.demography(t, y, pars).sum()
        }
        return mea


def dfe(t, y, pars):
    return y[I_Infectious].sum()


dfe.terminal = True
dfe.direction = -1

# Initial population
seed = 1e-2
Y0 = np.zeros(N_State)
Y0[I_U] = 1 - seed
Y0[I_Lat] = seed
