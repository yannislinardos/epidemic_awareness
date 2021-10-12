import numpy as np

class Awareness:
    """
    This class is used to define alternative (non-linear or discontinuous) awareness models for the simulations
    """

    def __init__(self, cL, cC, cG):
        self.cL = cL
        self.cC = cC
        self.cG = cG
        self.thr = 0.5
        self.upper_thr = 0.8
        self.lower_thr = 0.4
        self.activate_community_thr = -1
        self.activate_global_thr = False
        self.community_per_node = None

    def set_thr(self, thr):
        self.thr = thr

    def set_upper_thr(self, l):
        self.upper_thr = l

    def set_lower_thr(self, l):
        self.lower_thr = l

    def set_community_per_node(self, community_per_node):
        self.community_per_node = community_per_node

    def linear(self, iL, iH, i):
        alphas = (1 - self.cG * i) * (1 - self.cL * iL) * (1 - self.cC * iH)
        return alphas

    def exp(self, iL, iH, i):
        alphas = np.exp(- (self.cG * i + self.cL * iL + self.cC * iH))
        return alphas

    def threshold(self, iL, iH, i):

        # cL = (iL>=self.thr)*self.cL
        cC = (iH>=self.thr)*self.cC
        cG = self.cG if i>=self.thr else 0

        # alphas = (1 - cG * i) * (1 - cL * iL) * (1 - cC * iH)
        alphas = (1 - self.cL * iL) * (1 - cC) * (1 - cG)

        return alphas

    def constant(self, iL, iH, i):
        alphas = (1 - self.cC) * (1 - self.cG) * (1 - self.cL * iL)
        return alphas

    # cC is a list of coefficients
    def different_per_community(self, iL, iH, i):

        alphas = (1 - self.cG * i) * (1 - self.cL * iL) * (1 - np.array(self.cC)[self.community_per_node] * iH)

        return alphas

    def multiple_thresholds(self, iL, iH, i):

        if type(self.activate_community_thr) is int:
            self.activate_community_thr = np.zeros(iH.size)

        if i >= self.upper_thr:
            self.activate_global_thr = True
        if i <= self.lower_thr:
            self.activate_global_thr = False

        for j in range(iH.size):
            if iH[j] >= self.upper_thr:
                self.activate_community_thr[j] = True
            if iH[j] <= self.lower_thr:
                self.activate_community_thr[j] = False

        cC = self.activate_community_thr * self.cC
        cG = self.activate_global_thr * self.cG

        alphas = (1 - self.cL * iL) * (1 - cC) * (1 - cG)

        return alphas


if __name__ == '__main__':
    import simulations
    from HierarchicalConfigurationModel import *

    in_degrees = [2 * np.ones(1000), 2 * np.ones(1000)]
    out_degrees = np.ones(2000)
    HCM = HierarchicalConfigurationModel(in_degrees, out_degrees)
    N = HCM.get_number_of_nodes()

    sis = simulations.SIS(HCM)
    gamma = 1
    patient_zero = np.random.choice(list(HCM.graph.nodes), max(int(N*0.1), 1), replace=False)
    tau = 2
    t_max = 100
    awareness = (0, 1, 0)

    t, i , i_ = sis.simulate(tau, gamma, patient_zero, t_max, awareness, awareness_model='thr', thr=0.8)
    plt.plot(t, i)
    plt.show()
