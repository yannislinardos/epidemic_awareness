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

print('Simulations')
res_sim = sim.epidemic_threshold(gamma, max_tau=2)
print(res_sim)
print('Numerical')
res_num = num.epidemic_threshold(gamma, max_tau=2, linear=False)
print(res_num)

fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.plot(res_sim['cs'], res_sim["epidemic_thresholds_cL"], label='simulation', marker='^', c='r')
ax1.plot(res_num['cs'], res_num["epidemic_thresholds_cL"], label='mean-field', marker='s', c='b')
ax1.set_title('Local awareness')
ax1.set_xlabel('cL')
ax1.set_ylabel('tau_c')
ax1.legend()
ax2.plot(res_sim['cs'], res_sim["epidemic_thresholds_cC"], label='simulation', marker='^', c='r')
ax2.plot(res_num['cs'], res_num["epidemic_thresholds_cC"], label='mean-field', marker='s', c='b')
ax2.set_title('Community awareness')
ax2.set_xlabel('cC')
ax2.set_ylabel('tau_c')
ax2.legend()
ax3.plot(res_sim['cs'], res_sim["epidemic_thresholds_cG"], label='simulation', marker='^', c='r')
ax3.plot(res_num['cs'], res_num["epidemic_thresholds_cG"], label='mean-field', marker='s', c='b')
ax3.set_title('Global awareness')
ax3.set_xlabel('cG')
ax3.set_ylabel('tau_c')
ax3.legend()
plt.tight_layout()
plt.show()

print('Local MSE = ' + str(np.mean((res_sim["epidemic_thresholds_cL"] - res_num["epidemic_thresholds_cL"])**2)))
print('Community MSE = ' + str(np.mean((res_sim["epidemic_thresholds_cC"] - res_num["epidemic_thresholds_cC"])**2)))
print('Global MSE = ' + str(np.mean((res_sim["epidemic_thresholds_cC"] - res_num["epidemic_thresholds_cC"])**2)))

