import numpy as np
from matplotlib import pyplot as plt
from HierarchicalConfigurationModel import *
from utils import *
from scipy.optimize import fsolve, least_squares

class SIS:
    """
    G: network in HierarchicalConfigurationModel form
    tau: per edge transmission rate
    gamma: recovery rate
    initial_infections: initially infected nodes (usually one, the patient zero)
    t_max: maximum time that the simulation runs
    """

    def __init__(self, G):

        self.G = G

    #### getters and setters #####
    def set_G(self, G):
        self.G = G

    def get_G(self):
        return self.G


    # gets H, k_in, k_out and returns location in i vector
    def attr_to_location(self, H, k_in, k_out):

        max_k_out = self.G.p_out.shape[0] - 1
        max_k_in = self.G.p_in.shape[1] - 1

        loc = H * (max_k_out+1) * max_k_in + (k_in-1) * (max_k_out+1) + k_out
        return loc

    # takes location in i vector and returns H, k_in, k_out
    def location_to_attr(self, loc):

        max_k_out = self.G.p_out.shape[0] - 1
        max_k_in = self.G.p_in.shape[1] - 1

        # where each community starts and ends in the matrix
        community_separator = np.array([(max_k_out + 1) * max_k_in * index for index in range(1, self.G.n + 1)])

        H = np.searchsorted(community_separator, loc, side='right')

        start_of_subvector = 0 if H == 0 else community_separator[H - 1]
        # place in block vector
        block = loc - start_of_subvector

        k_in = block // (max_k_out + 1) + 1
        k_out = block % (max_k_out + 1)

        return H, k_in, k_out

    def from_node_to_compartment(self, node):

        H = self.G.get_community_per_node()[node]
        k_out = self.G.out_degrees[node]
        degree = len(np.array(self.G.get_graph()[node]))
        k_in = degree - k_out
        return H, k_in, k_out

    def find_stability_non_linear(self, tau, gamma, awareness, awareness_model='linear'):

        c_L, c_C, c_G = awareness

        max_k_out = self.G.p_out.shape[0] - 1
        max_k_in = self.G.p_in.shape[1] - 1
        n = self.G.get_number_of_communities()
        p_in = self.G.get_p_in()
        p_out = self.G.get_p_out()
        N = self.G.N
        N_H = self.G.N_H
        avg_k_in = np.sum(p_in * np.tile(np.arange(0, p_in.shape[1]), p_in.shape[0]).reshape(p_in.shape), axis=1)
        avg_k_out = np.sum(p_out * np.arange(0, p_out.shape[0]))
        p_total = self.G.p_total

        if awareness_model != 'different_per_community':
            c_C = [c_C for _ in range(n)]

        if awareness_model == 'different_per_community' and len(c_C) != 3:
            print('Check the community awareness')

        attr_to_loc = np.zeros((n + 1, max_k_in + 1, max_k_out + 1), dtype=int)
        for H in range(0, n):
            for k_in in range(1, p_in[H].size):
                for k_out in range(0, p_out.size):
                    attr_to_loc[H, k_in, k_out] = H * (max_k_out + 1) * max_k_in + (k_in - 1) * (max_k_out + 1) + k_out

        # takes location in i vector and returns H, k_in, k_out
        def location_to_attr(loc):
            # where each community starts and ends in the matrix
            community_separator = np.array([(max_k_out + 1) * max_k_in * index for index in range(1, self.G.n + 1)])
            H = np.searchsorted(community_separator, loc, side='right')
            start_of_subvector = 0 if H == 0 else community_separator[H - 1]
            # place in block vector
            block = loc - start_of_subvector
            k_in = block // (max_k_out + 1) + 1
            k_out = block % (max_k_out + 1)
            return H, k_in, k_out

        zero_attribute = list(zip(*np.where(p_total == 0)))
        number_of_equations = (max_k_out + 1) * n * max_k_in
        locations_of_zeros = []
        non_zero_to_normal_ordering = []
        for loc in range(number_of_equations):
            H, k_in, k_out = location_to_attr(loc)
            if (H, k_in, k_out) in zero_attribute:
                locations_of_zeros.append(loc)
            else:
                non_zero_to_normal_ordering.append(loc)

        non_zero_to_normal_ordering = np.array(non_zero_to_normal_ordering)
        number_of_non_zeros = number_of_equations - len(locations_of_zeros)

        # print('# of equations ', number_of_non_zeros)
        def attr_to_non_zero_location(H, k_in, k_out):
            loc = H * (max_k_out + 1) * max_k_in + (k_in - 1) * (max_k_out + 1) + k_out
            index = np.where(non_zero_to_normal_ordering == loc)[0]
            if index.size == 0:
                return 0
            else:
                return index[0]

        maximum_fraction = np.zeros(number_of_non_zeros)
        locs = []
        for H in range(0, n):
            for k_in in range(1, p_in[H].size):
                for k_out in range(0, p_out.size):
                    if attr_to_loc[H, k_in, k_out] not in locations_of_zeros:
                        loc = np.where(non_zero_to_normal_ordering == attr_to_loc[H, k_in, k_out])[0][0]
                        maximum_fraction[loc] = p_total[H, k_in, k_out]
                        locs.append(loc)

        def equations(y):

            theta_in = []
            for H in range(0, n):
                th = 0
                for k_in in range(1, p_in[H].size):
                    for k_out in range(0, p_out.size):
                        if attr_to_loc[H, k_in, k_out] not in locations_of_zeros:
                            loc = np.where(non_zero_to_normal_ordering == attr_to_loc[H, k_in, k_out])[0][0]
                            th += k_in * p_in[H, k_in] * y[loc]
                theta_in.append(th * N / (avg_k_in[H] * N_H[H]))

            theta_out = 0
            for H in range(0, n):
                for k_in in range(1, p_in[H].size):
                    for k_out in range(0, p_out.size):
                        if attr_to_loc[H, k_in, k_out] not in locations_of_zeros:
                            loc = np.where(non_zero_to_normal_ordering == attr_to_loc[H, k_in, k_out])[0][0]
                            theta_out += k_out * p_out[k_out] * y[loc]
            theta_out = theta_out / avg_k_out

            i_total = np.sum([y[attr_to_non_zero_location(H_j, s_in, s_out)] \
                    for H_j in range(0, n) for s_in in range(1, max_k_in + 1) for s_out
                    in range(0, max_k_out + 1)])
            eqs = []
            # infected
            for H in range(0, n):
                i_H = np.sum([y[attr_to_non_zero_location(H, s_in, s_out)]
                              for s_in in range(1, max_k_in + 1) for s_out in range(0, max_k_out + 1)])
                for k_in in range(1, p_in[H].size):
                    for k_out in range(0, p_out.size):
                        if attr_to_loc[H, k_in, k_out] not in locations_of_zeros:
                            loc = np.where(non_zero_to_normal_ordering == attr_to_loc[H, k_in, k_out])[0][0]
                            f = -gamma * y[loc] + tau * (p_total[H,k_in,k_out] - y[loc]) * \
                                (1 - c_C[H] * N / N_H[H] * i_H) * \
                                (1 - c_G * i_total) * \
                                (k_in * theta_in[H] + k_out * theta_out - c_L / (k_in + k_out) * \
                                 (k_out * theta_out * (1 - theta_out) + (k_out * theta_out) ** 2 +
                                  k_in * theta_in[H] * (1 - theta_in[H]) + (k_in * theta_in[H]) ** 2 + 2 * k_in *
                                  theta_in[H] * k_out * theta_out))

                            eqs.append(f)

            return np.array(eqs)

        y0 = np.copy(maximum_fraction)
        sol = least_squares(equations, y0/2, bounds=(np.zeros(number_of_non_zeros), maximum_fraction))
        cost = sol.cost

        if cost > 0.1:
            print('Numerical solution not found')
            return -1

        trials = 0
        while cost > 0.1 and trials < y0.size:
            not_zero_entries = np.where(y0!=0)[0]
            set_zero = np.random.choice(not_zero_entries)
            y0[set_zero] = 0
            sol = least_squares(equations, y0, bounds=(np.zeros(number_of_non_zeros), maximum_fraction))
            cost = sol.cost
            trials += 1

        if trials == y0.size and cost > 0.1:
            print('Numerical solution not found')

        i = np.sum(sol.x)

        inf_per_com = np.zeros(n)
        for j in range(sol.x.size):
            normal_loc = non_zero_to_normal_ordering[j]
            comm = location_to_attr(normal_loc)[0]
            inf_per_com[comm] += sol.x[j]

        return i, inf_per_com

    def find_stability_linear(self, tau, gamma, awareness):

        c_L, c_C, c_G = awareness

        max_k_out = self.G.p_out.shape[0] - 1
        max_k_in = self.G.p_in.shape[1] - 1
        n = self.G.get_number_of_communities()
        p_in = self.G.get_p_in()
        p_out = self.G.get_p_out()
        N = self.G.N
        N_H = self.G.N_H
        avg_k_in = np.sum(p_in * np.tile(np.arange(0, p_in.shape[1]), p_in.shape[0]).reshape(p_in.shape), axis=1)
        avg_k_out = np.sum(p_out * np.arange(0, p_out.shape[0]))
        p_total = self.G.p_total

        attr_to_loc = np.zeros((n + 1, max_k_in + 1, max_k_out + 1), dtype=int)
        for H in range(0, n):
            for k_in in range(1, p_in[H].size):
                for k_out in range(0, p_out.size):
                    attr_to_loc[H, k_in, k_out] = H * (max_k_out + 1) * max_k_in + (k_in - 1) * (max_k_out + 1) + k_out

        # takes location in i vector and returns H, k_in, k_out
        def location_to_attr(loc):
            # where each community starts and ends in the matrix
            community_separator = np.array([(max_k_out + 1) * max_k_in * index for index in range(1, self.G.n + 1)])
            H = np.searchsorted(community_separator, loc, side='right')
            start_of_subvector = 0 if H == 0 else community_separator[H - 1]
            # place in block vector
            block = loc - start_of_subvector
            k_in = block // (max_k_out + 1) + 1
            k_out = block % (max_k_out + 1)
            return H, k_in, k_out

        zero_attribute = list(zip(*np.where(p_total == 0)))
        number_of_equations = (max_k_out + 1) * n * max_k_in
        locations_of_zeros = []
        non_zero_to_normal_ordering = []
        for loc in range(number_of_equations):
            H, k_in, k_out = location_to_attr(loc)
            if (H, k_in, k_out) in zero_attribute:
                locations_of_zeros.append(loc)
            else:
                non_zero_to_normal_ordering.append(loc)

        non_zero_to_normal_ordering = np.array(non_zero_to_normal_ordering)
        number_of_non_zeros = number_of_equations - len(locations_of_zeros)

        maximum_fraction = np.zeros(number_of_non_zeros)
        for H in range(0, n):
            for k_in in range(1, p_in[H].size):
                for k_out in range(0, p_out.size):
                    if attr_to_loc[H, k_in, k_out] not in locations_of_zeros:
                        loc = np.where(non_zero_to_normal_ordering == attr_to_loc[H, k_in, k_out])[0][0]
                        maximum_fraction[loc] = p_total[H, k_in, k_out]

        def equations(y):

            # constant per community
            first_factors = []
            for H in range(0, n):
                th = 0
                for k_in in range(1, p_in[H].size):
                    for k_out in range(0, p_out.size):
                        if attr_to_loc[H, k_in, k_out] not in locations_of_zeros:
                            loc = np.where(non_zero_to_normal_ordering == attr_to_loc[H, k_in, k_out])[0][0]
                            th += k_in * p_in[H, k_in] * y[loc]
                            first_factors.append(th)

            second_factor = 0
            for H in range(0, n):
                for k_in in range(1, p_in[H].size):
                    for k_out in range(0, p_out.size):
                        if attr_to_loc[H, k_in, k_out] not in locations_of_zeros:
                            loc = np.where(non_zero_to_normal_ordering == attr_to_loc[H, k_in, k_out])[0][0]
                            second_factor += k_out * p_out[k_out] * y[loc]

            eqs = []
            # infected
            for H in range(0, n):
                for k_in in range(1, p_in[H].size):
                    for k_out in range(0, p_out.size):
                        if attr_to_loc[H, k_in, k_out] not in locations_of_zeros:
                            loc = np.where(non_zero_to_normal_ordering == attr_to_loc[H, k_in, k_out])[0][0]
                            f = -gamma * y[loc] + tau * p_total[H, k_in, k_out] * (1 - c_L / (k_in + k_out)) * (
                                    N * k_in / (N_H[H] * avg_k_in[H]) * first_factors[H]
                                    + k_out / avg_k_out * second_factor)

                            eqs.append(f)

            return np.array(eqs)

        y0 = maximum_fraction

        sol = least_squares(equations, y0, bounds=(np.zeros(number_of_non_zeros), y0))

        i = np.sum(sol.x)

        return i

    def find_stability(self, tau, gamma, awareness, linear=False, awareness_model='linear'):
        if linear:
            return self.find_stability_linear(tau, gamma, awareness, awareness_model=awareness_model)
        else:
            return self.find_stability_non_linear(tau, gamma, awareness, awareness_model=awareness_model)

    def calculate_epidemic_threshold(self, gamma, awareness, max_gamma=1, step=0.05, thr=0.0025, linear=False):

        tau_c = -1
        N = self.G.N

        for tau in np.arange(0, max_gamma, step):

            rho, _per_com = self.find_stability(tau, gamma, awareness, linear)
            if rho > max(thr, 1/N):
                tau_c = tau #- step
                break

        return tau_c

    def epidemic_threshold(self, gamma, max_tau=1, awareness_step=0.1, tau_step=0.05, thr=0.0025, linear=False):

        res = {}
        cs = np.arange(0, 1, awareness_step)

        print('Local Awareness')
        epidemic_thresholds_cL = []
        for c_L in tqdm(cs):
            epidemic_thresholds_cL.append(self.calculate_epidemic_threshold(gamma, (c_L, 0, 0), step=tau_step,
                                                                            max_gamma=max_tau, thr=thr, linear=linear))
        epidemic_thresholds_cL = np.array(epidemic_thresholds_cL)

        print('Community Awareness')
        epidemic_thresholds_cC = []
        for c_C in tqdm(cs):
            epidemic_thresholds_cC.append(self.calculate_epidemic_threshold(gamma, (0, c_C, 0), step=tau_step,
                                                                            max_gamma=max_tau, thr=thr, linear=linear))
        epidemic_thresholds_cC = np.array(epidemic_thresholds_cC)

        print('Global Awareness')
        epidemic_thresholds_cG = []
        for c_G in tqdm(cs):
            epidemic_thresholds_cG.append(self.calculate_epidemic_threshold(gamma, (0, 0, c_G), step=tau_step,
                                                                            max_gamma=max_tau, thr=thr, linear=linear))
        epidemic_thresholds_cG = np.array(epidemic_thresholds_cG)

        res['cs'] = cs
        res['epidemic_thresholds_cL'] = epidemic_thresholds_cL
        res['epidemic_thresholds_cC'] = epidemic_thresholds_cC
        res['epidemic_thresholds_cG'] = epidemic_thresholds_cG

        return res

    def epidemic_prevalence(self, gamma, tau, awareness, awareness_model='linear'):

        prevalence, per_community = self.find_stability(tau, gamma, awareness, False, awareness_model=awareness_model)
        return prevalence, per_community


if __name__ == '__main__':
    # # number of communities
    # n_com = 2
    # # average size of community
    # avg_size, sigma_size = 30, 0
    # # pareto variables
    # tau_in = 3.5
    # m_in = 2
    # tau_out = 3.5
    # m_out = 0.5
    # HCM = generate_random_HCM(n_com, avg_size, sigma_size, tau_in, m_in, tau_out, m_out)
    # N = HCM.get_number_of_nodes()
    N = 3000
    n = 3
    N_H = [N // n for _ in range(n)]

    in_degrees = [2 * np.ones(i) for i in N_H]
    out_degrees = 1 * np.ones(N)
    HCM = HierarchicalConfigurationModel(in_degrees, out_degrees)

    sis = SIS(HCM)
    gamma = 1
    tau = 1
    awareness = (0, 0, 0)

    for c_G in np.arange(0, 1, 0.1):
        i, inf_per_com = sis.epidemic_prevalence(tau, gamma, (0, 0.5, c_G))#, awareness_model='different_per_community')
        print(i, inf_per_com)

    # lin = sis.calculate_epidemic_threshold(gamma, awareness, linear=True)
    # non_lin = sis.calculate_epidemic_threshold_num(gamma, awareness, linear=False)
    # print('lin = %.2f, non_lin = %.2f' % (lin, non_lin))

    # lin = sis.epidemic_threshold(gamma, linear=True)
    # print(lin)
    # non_lin = sis.epidemic_threshold(gamma, linear=False)
    # print(non_lin)