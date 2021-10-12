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

gamma = 1
tau = 1
patient_zero = np.random.choice(list(HCM.graph.nodes), max(int(N * 0.1), 1), replace=False)
t_max = 50
awareness_step = 0.1

cs = np.arange(0, 1, awareness_step)

print('!!!!NUMERICAL!!!!')

print('Local Awareness')
num_prevalence_cL = []
for c_L in tqdm(cs):
    num_prevalence, _ = num.epidemic_prevalence(gamma, tau, (c_L, 0, 0))
    num_prevalence_cL.append(num_prevalence)
num_prevalence_cL = np.array(num_prevalence_cL)

print('Community Awareness')
num_prevalence_cC = []
for c_C in tqdm(cs):
    num_prevalence, _ = num.epidemic_prevalence(gamma, tau, (0, c_C, 0))
    num_prevalence_cC.append(num_prevalence)
num_prevalence_cC = np.array(num_prevalence_cC)

print('Global Awareness')
num_prevalence_cG = []
for c_G in tqdm(cs):
    num_prevalence, _ = num.epidemic_prevalence(gamma, tau, (0, 0, c_G))
    num_prevalence_cG.append(num_prevalence)
num_prevalence_cG = np.array(num_prevalence_cG)

print('!!!!SIMULATIONS!!!!')
print('Local Awareness')
sim_prevalence_cL = []
for c_L in tqdm(cs):
    sim_prevalence_cL.append(sim.epidemic_prevalence(gamma, tau, (c_L, 0, 0), patient_zero))
sim_prevalence_cL = np.array(sim_prevalence_cL)

print('Community Awareness')
sim_prevalence_cC = []
for c_C in tqdm(cs):
    sim_prevalence_cC.append(sim.epidemic_prevalence(gamma, tau, (0, c_C, 0), patient_zero))
sim_prevalence_cC = np.array(sim_prevalence_cC)

print('Global Awareness')
sim_prevalence_cG = []
for c_G in tqdm(cs):
    sim_prevalence_cG.append(sim.epidemic_prevalence(gamma, tau, (0, 0, c_G), patient_zero))
sim_prevalence_cG = np.array(sim_prevalence_cG)

fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.plot(cs, sim_prevalence_cL, label='simulation', marker='^', c='r')
ax1.plot(cs, num_prevalence_cL, label='mean-field', marker='s', c='b')
ax1.set_title('Local awareness')
ax1.set_xlabel('c_L')
ax1.set_ylabel('i_ss')
ax1.legend()
ax2.plot(cs, sim_prevalence_cC, label='simulation', marker='^', c='r')
ax2.plot(cs, num_prevalence_cC, label='mean-field', marker='s', c='b')
ax2.set_title('Community awareness')
ax2.set_xlabel('c_c')
ax2.set_ylabel('i_ss')
ax2.legend()
ax3.plot(cs, sim_prevalence_cG, label='simulation', marker='^', c='r')
ax3.plot(cs, num_prevalence_cG, label='mean-field', marker='s', c='b')
ax3.set_title('Global awareness')
ax3.set_xlabel('c_G')
ax3.set_ylabel('i_ss')
ax3.legend()
plt.tight_layout()
plt.show()

print('Local MSE = ' + str(np.mean((sim_prevalence_cL-num_prevalence_cL)**2)))
print('Community MSE = ' + str(np.mean((sim_prevalence_cC-num_prevalence_cC)**2)))
print('Global MSE = ' + str(np.mean((sim_prevalence_cG-num_prevalence_cG)**2)))
