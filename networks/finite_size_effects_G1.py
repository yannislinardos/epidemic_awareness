from HierarchicalConfigurationModel import *
import numerical
import simulations
import numpy as np
from matplotlib import pyplot as plt

gamma = 1
awareness = (0, 0, 0)

Ns = [30, 300, 900, 1800, 3000, 6000, 12000, 15000, 24000, 30000]
tau_cs = []
n = 3

for N in Ns:

    print('N = ', N)
    N_H = [N // n for _ in range(n)]
    in_degrees = [2 * np.ones(i) for i in N_H]
    out_degrees = 1 * np.ones(N)
    HCM = HierarchicalConfigurationModel(in_degrees, out_degrees)
    sim = simulations.SIS(HCM)
    tau_c = sim.calculate_epidemic_threshold(gamma, awareness, max_tau=2)
    print('tau_c = ', tau_c)
    tau_cs.append(tau_c)

plt.plot(Ns, tau_cs, marker='^')
plt.xlabel('Size')
plt.ylabel('tau_c')
plt.show()
