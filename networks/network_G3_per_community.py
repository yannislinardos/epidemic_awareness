from HierarchicalConfigurationModel import *
import numerical
import simulations
import numpy as np
from matplotlib import pyplot as plt

gamma = 1
N = 300
n = 30
N_H = [N//n for _ in range(n)]

in_degrees = [2*np.ones(i) for i in N_H]
out_degrees = 1*np.ones(N)

HCM = HierarchicalConfigurationModel(in_degrees, out_degrees)
sim = simulations.SIS(HCM)
num = numerical.SIS(HCM)

## infected per community

gamma = 1
patient_zero = np.random.choice(list(HCM.graph.nodes), max(int(N*0.1), 1), replace=False)
# patient_zero = np.array(HCM.graph.nodes)
tau = 2
t_max = 150
awareness = (0.5, 0.5, 0.5)

t, i, i_, inf_per_com = sim.simulate(tau, gamma, patient_zero, t_max, awareness)

for com in (range(n)):
    plt.plot(t, inf_per_com[com]/N)

plt.plot(t, i/N, label='network')
plt.ylabel('i')
plt.xlabel('t')
plt.legend()
plt.show()

avg_i = np.sum(i[-10:])/10/N
print('Avg i over the last 10 time steps ' + str(avg_i))

for i_com in inf_per_com:
    avg_i_per_community = np.sum(i_com[ -20:]) / 20 / N
    print('Avg i per community over the last 10 time steps ' + str(avg_i_per_community))

i_ss = num.epidemic_prevalence(gamma, tau, awareness)
print('The mean-field epidemic prevalence is ' + str(i_ss))
