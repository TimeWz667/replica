import fitter.model as fm
from replica.vesga2019.parameters import ParameterDefinition
from replica.vesga2019.model import ModelIndia, Y0, dfe
from replica.vesga2019.likelihood import find_likelihood_function
from replica import simulate
import numpy as np
import time

__author__ = 'Chu-Chang Ku'
__all__ = ['Fittable']


class Fittable(fm.AbsTarget):
    def __init__(self, data_path):
        self.Params = ParameterDefinition()
        fm.AbsTarget.__init__(self)
        self.Model = ModelIndia()
        self.Likelihood = find_likelihood_function(data_path)
        self.DFE = dfe
        self.N_eval = 0

    def sample_prior(self) -> fm.Parameters:
        vs = self.Params.draw_free_pars()
        return fm.Parameters(vs)

    def calc_prior(self, pars: fm.Parameters):
        li = self.Params.evaluate_prior(pars.Pars)
        return li

    def simulate(self, pars: fm.Parameters):
        p = self.Params.draw(**pars.Pars)
        time.sleep(0.002)

        t_out = np.linspace(1970, 2018, num=int((2018 - 1970) / 0.5) + 1)

        try:
            sim = simulate(self.Model, p, Y0, t_out, dfe=dfe)
            self.N_eval += 1
        except ValueError:
            sim = None, None, {'succ': False}

        return sim

    def calc_likelihood(self, sim, pars: fm.Parameters):
        ys, ms, msg = sim
        if not msg['succ'] or not ys.success:
            return - np.inf

        return self.Likelihood(ms)


if __name__ == '__main__':
    to_fit = Fittable("../../data/vesga2019/Targets.csv")

    pars = to_fit.sample_prior()
    print(pars)
    sim = to_fit.simulate(pars)

    print(sim[1])

    li = to_fit.calc_likelihood(sim, pars)
    print(li)
