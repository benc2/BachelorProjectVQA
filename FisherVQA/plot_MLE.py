import numpy as np
import matplotlib.pyplot as plt
true_phi = np.ones(4)*np.pi/5
file_opt = "phi_opt_bogotasim.txt"
file_ghz = "phi_ghz_bogotasim.txt"

samples = 200000

filename_opt = workdir + folder + file_opt
data_opt = np.genfromtxt(filename_opt, delimiter=',', skip_header=1)

filename2 = workdir + folder + file_ghz
data_ghz = np.genfromtxt(filename2, delimiter=',', skip_header=1)

costs_opt = data_opt.T[1]
costs_ghz = data_ghz.T[1]

plt.plot(np.arange(len(costs_opt))+1, -costs_opt/samples, '.', label="Optimal", markersize=8)
plt.plot(np.arange(len(costs_opt))+1, -costs_opt/samples, '--', color="#BBBBBB", zorder=-10)
plt.plot(np.arange(len(costs_ghz))+1, -costs_ghz/samples, 'D', label="GHZ", markersize=4)
plt.plot(np.arange(len(costs_ghz))+1, -costs_ghz/samples, '--', color="#BBBBBB", zorder=-10)
plt.xlabel("Iterations")
plt.ylabel(r"$\log\mathcal{L}$")
legend = plt.legend()
legend.get_frame().set_alpha(None)
plt.show()


phis_opt = data_opt[:, 2:]
phis_ghz = data_ghz[:, 2:]
expanded_true_phi = np.expand_dims(true_phi, 0)
norms_opt = np.linalg.norm(phis_opt - expanded_true_phi, axis=1)
norms_ghz = np.linalg.norm(phis_ghz - expanded_true_phi, axis=1)

plt.plot(np.arange(len(costs_opt))+1, norms_opt, '.', markersize=8, label='Optimal')
plt.plot(np.arange(len(costs_opt))+1, norms_opt, '--', color="#BBBBBB", zorder=-10)
plt.plot(np.arange(len(costs_ghz))+1, norms_ghz, 'D', markersize=4, label='GHZ')
plt.plot(np.arange(len(costs_ghz))+1, norms_ghz, '--', color="#BBBBBB", zorder=-10)
plt.xlabel("Iterations")
plt.ylabel(r"$|\hat{\phi}_i - \phi|$")
legend = plt.legend()
legend.get_frame().set_alpha(None)

plt.show()
plt.close('all')


def g(phi):
    return np.mean(phi)


g_opt = np.abs(np.apply_along_axis(g, 1, phis_opt) - g(true_phi))
g_ghz = np.abs(np.apply_along_axis(g, 1, phis_ghz) - g(true_phi))
plt.plot(np.arange(len(costs_opt))+1, g_opt, '.', markersize=8, label='Optimal')
plt.plot(np.arange(len(costs_opt))+1, g_opt, '--', color="#BBBBBB", zorder=-10)
plt.plot(np.arange(len(costs_ghz))+1, g_ghz, 'D', markersize=4, label='GHZ')
plt.plot(np.arange(len(costs_opt))+1, g_ghz, '--', color="#BBBBBB", zorder=-10)
plt.xlabel("Iterations")
plt.ylabel(r"$|g(\hat{\phi}_i) - g(\phi)|$")
legend = plt.legend()
legend.get_frame().set_alpha(None)
plt.show()
