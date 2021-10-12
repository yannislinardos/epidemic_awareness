import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from HierarchicalConfigurationModel import *
from utils import *
from awareness_models_simulation import *
import pickle
import itertools
from scipy.integrate import odeint, ode, solve_ivp
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

    # thr refers to the awareness model, not the epidemic threshold
    def simulate(self, tau, gamma, initial_infections, t_max, awareness, awareness_model='linear', thr=0.5,
                 reinfection=False, reinfection_fraction=0):
        # awareness = (c_L, c_C, c_G)
        c_L, c_C, c_G = awareness

        G = self.G.get_graph()
        n_com = self.G.get_number_of_communities()  # number of communities
        infected_per_community = np.zeros(n_com)
        community_sizes = self.G.get_community_sizes()
        # index is the node and value is the community
        community_per_node = self.G.get_community_per_node()
        N = self.G.get_number_of_nodes()
        degrees = np.zeros(N)
        degrees[np.array(G.degree)[:,0]] = np.array(G.degree)[:, 1]
        degrees[degrees == 0] = 1 # just to avoid division by zero

        model_class = Awareness(c_L, c_C, c_G)
        model = model_class.linear
        if awareness_model == 'linear':
            model = model_class.linear
        elif awareness_model == 'exp' or awareness_model == 'exponential':
            model = model_class.exp
        elif awareness_model == 'threshold' or awareness_model == 'thr':
            model_class.set_thr(thr)
            model = model_class.threshold
        elif awareness_model == 'mult_thr':
            model_class.set_lower_thr(thr[0])
            model_class.set_upper_thr(thr[1])
            model = model_class.multiple_thresholds
        elif awareness_model == 'constant' or awareness_model == 'const':
            model = model_class.constant
        elif awareness_model == 'different_per_community':
            model_class.set_community_per_node(community_per_node)
            model = model_class.different_per_community


        if type(initial_infections) is int:
            initial_infections = [initial_infections]

        S = np.ones(N)
        S[initial_infections] = 0
        I = np.zeros(N)
        I[initial_infections] = 1
        infected_neighbors = np.zeros(N)  # number of infected neighbours per node (including zeros)

        for i in initial_infections:
            infected_neighbors[np.array(G[i])] += 1
            infected_per_community[community_per_node[i]] += 1

        times = [0]
        infected_per_iter = [initial_infections.size]
        inf_per_com = [[infected_per_community[c]] for c in range(n_com)] # each list corresponds to one community

        at_risk_nodes = list(set(np.where(infected_neighbors != 0)[0]) - set(np.where(I == 1)[0]))

        infection_rates = np.zeros(N)

        infection_rates[at_risk_nodes] = tau * infected_neighbors[at_risk_nodes]

        # alphas = (1 - c_G * np.sum(I) / N) * (1 - c_L * infected_neighbors / degrees) * \
        #          (1 - c_C * infected_per_community[community_per_node] / community_sizes[community_per_node])
        alphas = model(infected_neighbors/degrees,
                       infected_per_community[community_per_node]/community_sizes[community_per_node],
                       np.sum(I)/N)

        effective_infection_rates = infection_rates * alphas

        total_infection_rate = np.sum(effective_infection_rates)

        total_recovery_rate = gamma * np.sum(I)
        total_rate = total_infection_rate + total_recovery_rate

        time = np.random.exponential(1 / total_rate)

        current_day = 0
        infected_today = 0
        infected_per_day = [0]

        while time < t_max and total_rate > 0:

            r = np.random.choice(['s->i', 'i->s'],
                                 p=np.array([total_infection_rate, total_recovery_rate]) / total_rate)

            if r == 'i->s':

                u = np.random.choice(np.where(I == 1)[0])
                I[u] = 0
                S[u] = 1

                if infected_neighbors[u] != 0:
                    at_risk_nodes.append(u)
                    infection_rates[u] = tau * infected_neighbors[u]

                susceptible_neighbors = np.intersect1d(np.array(G[u]), np.where(S == 1)[0])
                infection_rates[susceptible_neighbors] -= tau

                infected_neighbors[np.array(G[u])] -= 1
                infected_per_community[community_per_node[u]] -= 1

            elif r == 's->i':

                u = np.random.choice(at_risk_nodes, p=effective_infection_rates[at_risk_nodes] / total_infection_rate)
                at_risk_nodes.remove(u)
                S[u] = 0
                I[u] = 1
                infection_rates[u] = 0

                for v in np.intersect1d(np.array(G[u]), np.where(S == 1)[0]):  # for susceptible neighbors
                    infection_rates[v] += tau
                    if v not in at_risk_nodes:
                        at_risk_nodes.append(v)

                infected_neighbors[np.array(G[u])] += 1
                infected_per_community[community_per_node[u]] += 1
                infected_today += 1

            ## useful for the epidemic threshold
            if reinfection and np.sum(I) == 0:
                reinfected = np.random.choice(list(self.G.get_graph().nodes),
                                              max(int(reinfection_fraction*N),1), replace=False)
                I[reinfected] = 1
                S[reinfected] = 0
                for u in reinfected:
                    infection_rates[u] = 0
                    if u in at_risk_nodes:
                        at_risk_nodes.remove(u)

                    for v in np.intersect1d(np.array(G[u]), np.where(S == 1)[0]):  # for susceptible neighbors
                        infection_rates[v] += tau
                        if v not in at_risk_nodes:
                            at_risk_nodes.append(v)

                    infected_neighbors[np.array(G[u])] += 1
                    infected_per_community[community_per_node[u]] += 1
                    infected_today += 1

            times.append(time)
            infected_per_iter.append(np.sum(I))

            for com in range(n_com):
                inf_per_com[com].append(infected_per_community[com])

            # alphas = (1 - c_G * np.sum(I) / N) * (1 - c_L * infected_neighbors / degrees) * \
            #              (1 - c_C * infected_per_community[community_per_node] / community_sizes[community_per_node])
            alphas = model(infected_neighbors / degrees,
                           infected_per_community[community_per_node] / community_sizes[community_per_node],
                           np.sum(I) / N)
            effective_infection_rates = infection_rates * alphas

            effective_infection_rates[np.abs(effective_infection_rates) < 1e-4] = 0

            total_infection_rate = np.sum(effective_infection_rates)
            total_recovery_rate = gamma * np.sum(I)
            total_rate = total_infection_rate + total_recovery_rate

            if total_rate < 1e-8:
                break

            time += np.random.exponential(1 / total_rate)

            # print('\r t = %.5f / %d' % (time, t_max), end='')

            if 2 >= time - current_day >= 1:
                current_day += 1
                infected_per_day.append(infected_today)
                infected_today = 0
            elif time - current_day > 2 and t_max > time:
                infected_per_day += [0 for _ in range(int(time - current_day - 1))]
                infected_per_day.append(infected_today)
                infected_today = 0
                current_day += int(time - current_day)

        # for t in range(int(t_max)):
        #     timesteps = np.where(t <= np.array(times) <= t+1)


        if times[-1] < t_max and infected_per_iter[-1] == 0:
            times.append(t_max)
            infected_per_iter.append(0)

        infected_per_day += [0 for _ in range(int(t_max - len(infected_per_day)))]

        for com in range(n_com):
            inf_per_com[com] = np.array(inf_per_com[com])

        return np.array(times), np.array(infected_per_iter), np.array(infected_per_day), inf_per_com

    # def calculate_epidemic_threshold(self, gamma, awareness, max_tau=1, t_max=50,
    #                                  step=0.05, thr=0.0005, realizations=5, awareness_model='linear'):
    #
    #     N = self.G.get_number_of_nodes()
    #     # rhos = []
    #
    #     for tau in np.arange(0, max_tau, step):
    #
    #         fractions = np.zeros(realizations)
    #         for i in range(realizations):
    #             # patients_zero = np.random.choice(list(self.G.get_graph().nodes), int(N*0.01), replace=False)
    #             patients_zero = np.array(self.G.get_graph().nodes)
    #             times, infected_per_iter, infected_per_day, infected_per_community_per_time = \
    #                 self.simulate(tau, gamma, patients_zero, t_max, awareness, awareness_model=awareness_model)
    #             avg_infected = np.sum(infected_per_day[-int(t_max/10):])/int(t_max/10)
    #             fractions[i] = avg_infected/N
    #
    #         rho = np.mean(fractions)
    #         # rhos.append(fractions)
    #         if rho > thr:
    #             tau_c = tau - step
    #             return tau_c
    #     return -1

    # this is with the surviving runs method
    def calculate_epidemic_threshold_new(self, gamma, awareness, max_tau=2, t_max=50,
                                     step=0.05, thr=0.0025, realizations=10, max_realizations=15, awareness_model='linear'):

        N = self.G.get_number_of_nodes()

        for tau in np.arange(0, max_tau, step):

            fractions = []

            for _ in range(max_realizations):
                # patients_zero = np.random.choice(list(self.G.get_graph().nodes), int(N*0.01), replace=False)
                patients_zero = np.array(self.G.get_graph().nodes)
                times, infected_per_iter, infected_per_day, infected_per_community_per_time = \
                    self.simulate(tau, gamma, patients_zero, t_max, awareness, awareness_model=awareness_model)
                avg_infected = np.sum(infected_per_day[-int(t_max/10):])/int(t_max/10)/N

                if avg_infected > 1e-8:
                    fractions.append(avg_infected)

                if len(fractions) == realizations:
                    break

            # if they pass the threshold once, this is the tau_c
            if len(fractions) != 0:
                if np.mean(fractions) > max(thr, 1/N):
                    return tau

            #     if avg_infected != 0:
            #         fractions.append(avg_infected / N)
            #
            # if len(fractions) != 0 and np.mean(fractions) > thr:
            #     tau_c = tau - step
            #     return tau_c
        return -1


    def calculate_epidemic_threshold_reinfection(self, gamma, awareness, max_tau=2, t_max=100, reinfection_fraction=0.0005,
                                     step=0.05, thr=0.0025, realizations=5, awareness_model='linear'):

        N = self.G.get_number_of_nodes()
        chis = []
        taus = np.arange(0, max_tau, step)

        for tau in taus:

            rhos = []

            for _ in range(realizations):

                patients_zero = np.array(self.G.get_graph().nodes)
                times, infected_per_iter, infected_per_day, infected_per_community_per_time = \
                    self.simulate(tau, gamma, patients_zero, t_max, awareness, awareness_model=awareness_model,
                                  reinfection=True, reinfection_fraction=reinfection_fraction)

                rho = np.sum(infected_per_iter[-int(t_max / 10):]) / int(t_max / 10) / N
                rhos.append(rho)

            avg_rho = np.mean(rhos)
            avg_rho_squared = np.mean([r**2 for r in rhos])
            chi = N * (avg_rho_squared - avg_rho**2)/avg_rho
            chis.append(chi)

            if avg_rho > max(thr, 10/N):
                return tau

        return -1

        # plt.scatter(taus, chis)
        # plt.show()
        # tau_c = taus[np.argmax(chis)]
        # return tau_c

        #     if np.mean(rhos) > thr:

    def epidemic_threshold(self, gamma, max_tau=1, awareness_step=0.1,
                           tau_step=0.05, thr=.0025, awareness_model='linear'):

        res = {}
        cs = np.arange(0, 1, awareness_step)

        print('Local Awareness')
        epidemic_thresholds_cL = []
        for c_L in tqdm(cs):
            epidemic_thresholds_cL.append(self.calculate_epidemic_threshold_new(gamma, (c_L, 0, 0), max_tau=max_tau,
                                                                            step=tau_step, thr=thr, awareness_model=awareness_model))
        epidemic_thresholds_cL = np.array(epidemic_thresholds_cL)

        print('Community Awareness')
        epidemic_thresholds_cC = []
        for c_C in tqdm(cs):
            epidemic_thresholds_cC.append(self.calculate_epidemic_threshold_new(gamma, (0, c_C, 0), max_tau=max_tau,
                                                                            step=tau_step, thr=thr, awareness_model=awareness_model))
        epidemic_thresholds_cC = np.array(epidemic_thresholds_cC)

        print('Global Awareness')
        epidemic_thresholds_cG = []
        for c_G in tqdm(cs):
            epidemic_thresholds_cG.append(self.calculate_epidemic_threshold_new(gamma, (0, 0, c_G), max_tau=max_tau,
                                                                            step=tau_step, thr=thr, awareness_model=awareness_model))
        epidemic_thresholds_cG = np.array(epidemic_thresholds_cG)

        res['cs'] = cs
        res['epidemic_thresholds_cL'] = epidemic_thresholds_cL
        res['epidemic_thresholds_cC'] = epidemic_thresholds_cC
        res['epidemic_thresholds_cG'] = epidemic_thresholds_cG

        return res

    def epidemic_prevalence(self, gamma, tau, awareness, patient_zero, realizations=5, t_max=50, awareness_model='linear'):

        trials = []
        for _ in range(realizations):
            t, i, i_, infected_per_community_per_time = \
                self.simulate(tau, gamma, patient_zero, t_max, awareness, awareness_model=awareness_model)
            prevalence = i[-1] / self.G.N
            trials.append(prevalence)
        mean_prevalence = np.mean(trials)
        return mean_prevalence


if __name__ == '__main__':

    # # number of communities
    # n_com = 4
    # # average size of community
    # avg_size, sigma_size = 50, 0
    # # pareto variables
    # tau_in = 3.5
    # m_in = 2
    # tau_out = 3.5
    # m_out = 0.5
    # HCM = generate_random_HCM(n_com, avg_size, sigma_size, tau_in, m_in, tau_out, m_out)
    # N = HCM.get_number_of_nodes()

    in_degrees = [2*np.ones(100), 2*np.ones(100), 2*np.ones(100)]
    out_degrees = np.ones(300)
    HCM = HierarchicalConfigurationModel(in_degrees, out_degrees)
    N = HCM.get_number_of_nodes()

    sis = SIS(HCM)
    gamma = 1
    # patient_zero = np.random.choice(list(HCM.graph.nodes), max(int(N*0.1), 1), replace=False)
    patient_zero = np.array(HCM.graph.nodes)
    tau = 2
    t_max = 100
    # awareness = (0, 0, 0)

    t, i , i_, infected_per_community_per_time = \
        sis.simulate(tau, gamma, patient_zero, t_max, (0,(0, 0.5, 1), 1), reinfection=True,
                     awareness_model='different_per_community')

    for com in range(len(infected_per_community_per_time)):
        plt.plot(t, infected_per_community_per_time[com]/N, label='com '+str(com))

    plt.legend()
    plt.show()

    # plt.plot(t, i/N)
    # plt.show()

    # tau_c = sis.calculate_epidemic_threshold_reinfection(gamma, awareness, step=0.2)
    # res = sis.epidemic_threshold(gamma)
    # res = sis.epidemic_threshold(gamma)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 3)
    # ax[0].plot(res['cs'], res, label='numerical')
    # ax[0].plot(t_num, i_num, label='simulation')
    # ax.plot(t_sim, i_sim, label='simulation')
    # ax.legend()
    # plt.show()

