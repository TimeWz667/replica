import numpy as np
from replica import Uniform, Parameters
from replica.vesga2019.keys import *

__all__ = ['ParameterDefinition']


FixedPars = {
        'p_fast': 0.14,
        'r_react': 0.001,  # Activation of Latent TB
        'r_relapse_tc': 0.032,  # relapse after treatment completion,
        'r_relapse_td': 0.14,  # relapse after treatment default,
        'r_relapse_stab': 0.0015,  # relapse after stabilisation,
        'r_stab': 0.5,  # stabilisation rate,
        'r_tri': 52,  # treatment initialisation rate,
        'r_rec': 2,  # first-line treatment duration
        'r_growth': 0.023,  # population growth
        'r_death': 0.015
    }


PriorPars = {
        'beta': Uniform(2.5, 11.2),
        'rr_inf_cs': Uniform(0.1, 10),
        'r_sym': Uniform(3, 5.5),
        'r_death_tb': Uniform(0.14, 0.18),
        'r_cure': Uniform(0.14, 0.18),  # self-cure
        'rr_sus': Uniform(0.25, 0.75),  # reduced susceptibility
        'rr_csi': Uniform(2, 3.3),  # initial care-seeking rate
        'rr_tr': Uniform(2.6, 5.2),  # transition between episode
        'p_pub': Uniform(0.4, 0.55),
        'p_dx_pub': Uniform(0.8, 0.84),
        'p_dx_pri': Uniform(0.5, 0.67),
        'p_tri_pub': Uniform(0.86, 0.89),
        'p_tri_pri': Uniform(0.32, 0.73),
        'p_default_pub': Uniform(0.13, 0.16),
        'p_default_pri': Uniform(0.13, 0.16)
    }


class ParameterDefinition:
    def __init__(self, **kwargs):
        self.Prior = {k: v for k, v in PriorPars.items() if k not in kwargs}
        self.Fixed = dict(FixedPars)
        self.Fixed.update(kwargs)

    def draw_free_pars(self):
        return {k: v.rand() for k, v in self.Prior.items()}

    def evaluate_prior(self, ps):
        return sum([v.logpdf(ps[k]) for k, v in self.Prior.items()])

    def draw(self, **kwargs):
        ps = dict(self.Fixed)

        for k, v in self.Prior.items():
            ps[k] = v.rand()
        ps.update(kwargs)

        cs = self.trm_care(ps)

        transformed = {
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
        transformed['sus'] = sus

        trans = np.zeros(N_State)
        trans[[I_Asym, I_Sym]] = ps['beta']
        trans[[I_DPub, I_DPri, I_E]] = ps['beta'] * ps['rr_inf_cs']
        transformed['trans'] = trans

        return Parameters(ps, transformed)

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
        #trm[I_TPub, I_RHigh] = ps['p_default_pub'] * ps['r_rec']  # default
        #trm[I_TPub, I_RLow] = (1 - ps['p_default_pub']) * ps['r_rec']  # success
        #trm[I_TPri, I_RHigh] = ps['p_default_pri'] * ps['r_rec']  # default
        #trm[I_TPri, I_RLow] = (1 - ps['p_default_pri']) * ps['r_rec']  # success

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


if __name__ == '__main__':
    Param = ParameterDefinition()

    p = Param.draw()

    for k, v in p.Pars.items():
        print('{}: {:.2g}'.format(k, v))

    print(p['CareInitial'])
