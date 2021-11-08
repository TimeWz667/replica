from replica.vesga2019.model import ModelIndia, Y0, dfe
from replica.vesga2019.keys import *
from replica.vesga2019.likelihood import find_likelihood_function
from replica import simulate
from sims_pars import get_all_fixed_sc
from sims_pars.fitting import AbsObjectiveSC
import time
import numpy as np

__author__ = 'Chu-Chang Ku'
__all__ = ['Objective']


class TransformedPars:
    def __init__(self, ps):
        self.Pars = ps

        cs = self.trm_care(ps)

        self.Transformed = {
            'Progression': self.trm_natural_history(ps),
            'Care': cs,
            'CareInitial': self.trm_care_initial(ps, cs),
            'CareIntv': self.trm_care_full_intv(ps, cs),
            'Demography': self.trm_demo(ps),
        }

        # transmissibility and susceptibility
        sus = np.zeros(N_State)
        sus[I_U] = 1
        sus[I_LTBI] = ps['rr_sus']
        self.Transformed['sus'] = sus

        trans = np.zeros(N_State)
        trans[[I_Asym, I_Sym]] = ps['beta']
        trans[[I_DPub, I_DPri, I_E]] = ps['beta'] * ps['rr_inf_cs']
        self.Transformed['trans'] = trans

    def __getitem__(self, item):
        try:
            return self.Pars[item]
        except KeyError:
            return self.Transformed[item]

    @staticmethod
    def trm_natural_history(ps):
        # Natural history ----
        trm = np.zeros((N_State, N_State))

        # reinfection
        trm[I_Lat, I_Asym] = ps['r_react']
        trm[I_RLow, I_Asym] = ps['r_relapse_tc']
        trm[I_RHigh, I_Asym] = ps['r_relapse_td']
        trm[I_RStab, I_Asym] = ps['r_relapse_stab']

        # symptom onset
        trm[I_Asym, I_Sym] = ps['r_sym']

        # self-cure
        trm[I_Infectious, I_RStab] = ps['r_cure']

        # post-treatment stabilisation
        trm[I_RLow, I_RStab] = ps['r_stab']
        trm[I_RHigh, I_RStab] = ps['r_stab']

        return trm

    @staticmethod
    def trm_care(ps):
        # Care-seeking before public section scale-up
        trm = np.zeros((N_State, N_State))

        p_pub = ps['p_pub']
        p_sector = np.array([p_pub, 1 - p_pub])
        p_dx = np.array([ps['p_dx_pub'], ps['p_dx_pri']])
        p_tri = np.array([ps['p_tri_pub'], ps['p_tri_pri']])

        # initial care-seeking
        trm[I_Sym, I_D] = p_sector * ps['rr_csi']

        # initial LTFU
        trm[I_D, I_E] = ps['r_tri'] * (1 - p_dx * p_tri)

        # secondary care-seeking
        trm[I_E, I_D] = p_sector * ps['rr_tr']

        # diagnosis
        trm[I_D, I_T] = ps['r_tri'] * p_dx * p_tri

        # treatment outcome
        # trm[I_TPub, I_RHigh] = ps['p_default_pub'] * ps['r_rec']  # default
        # trm[I_TPub, I_RLow] = (1 - ps['p_default_pub']) * ps['r_rec']  # success
        # trm[I_TPri, I_RHigh] = ps['p_default_pri'] * ps['r_rec']  # default
        # trm[I_TPri, I_RLow] = (1 - ps['p_default_pri']) * ps['r_rec']  # success

        trm[I_TPub, I_RHigh] = ps['p_default_pub']  # default
        trm[I_TPub, I_RLow] = ps['r_rec']  # success
        trm[I_TPri, I_RHigh] = ps['p_default_pri']  # default
        trm[I_TPri, I_RLow] = ps['r_rec']  # success

        return trm

    @staticmethod
    def trm_care_initial(ps, trm):
        trm = trm.copy()
        p_pub = 0
        p_sector = np.array([p_pub, 1 - p_pub])

        # initial care-seeking
        trm[I_Sym, I_D] = p_sector * ps['rr_csi']
        trm[I_E, I_D] = p_sector * ps['rr_tr']
        return trm

    @staticmethod
    def trm_care_full_intv(ps, trm):
        trm = trm.copy()
        p_dx = 0.95
        p_tri = 0.95

        p_pub = ps['p_pub']
        p_sector = np.array([p_pub, 1 - p_pub])

        # initial care-seeking
        trm[I_Sym, I_D] = p_sector * ps['rr_csi'] / 0.75

        # initial LTFU
        trm[I_D, I_E] = ps['r_tri'] * (1 - p_dx * p_tri)

        # secondary care-seeking
        trm[I_E, I_D] = p_sector * ps['rr_tr'] / 0.75

        # diagnosis
        trm[I_D, I_T] = ps['r_tri'] * p_dx * p_tri

        # treatment outcome
        trm[I_TPub, I_RHigh] = 0.05 * ps['r_rec']  # default
        trm[I_TPub, I_RLow] = 0.95 * ps['r_rec']  # success
        trm[I_TPri, I_RHigh] = 0.05 * ps['r_rec']  # default
        trm[I_TPri, I_RLow] = 0.95 * ps['r_rec']  # success
        return trm

    @staticmethod
    def trm_demo(ps):
        trm = np.zeros((N_State, N_State))

        trm[I_Infectious, I_U] = ps['r_death_tb']
        trm[I_TPri, I_U] = ps['r_death_tb']
        trm[I_TPub, I_U] = ps['r_death_tb']
        trm[I_LTBI, I_U] = ps['r_death']
        return trm


class Objective(AbsObjectiveSC):
    def __init__(self, data_path, prior_path):
        with open(prior_path, 'r') as f:
            scr = f.read()
        sc = get_all_fixed_sc(scr)

        AbsObjectiveSC.__init__(self, sc)
        self.Model = ModelIndia()
        self.Likelihood = find_likelihood_function(data_path)
        self.DFE = dfe
        self.N_eval = 0

    def simulate(self, pars):
        time.sleep(0.002)

        t_out = np.linspace(1970, 2018, num=int((2018 - 1970) / 0.5) + 1)

        try:
            sim = simulate(self.Model, TransformedPars(pars), Y0, t_out, dfe=dfe)
            self.N_eval += 1
        except ValueError:
            sim = None, None, {'succ': False}

        return sim

    def link_likelihood(self, sim):
        ys, ms, msg = sim
        if not msg['succ'] or not ys.success:
            return - np.inf

        return self.Likelihood(ms)


if __name__ == '__main__':
    to_fit = Objective(
        prior_path='../../data/vesga2019/Prior.txt',
        data_path='../../data/vesga2019/Targets.csv'
    )

    pars = to_fit.sample_prior()
    print(pars)
    sim = to_fit.simulate(pars)

    print(sim[1])

    li = to_fit.evaluate(pars)
    print(li)

