import matplotlib.pyplot as plt
import numpy as np

stds = np.zeros((9, 2))


for N in range(2, 11):
    data = np.genfromtxt(f"measurements_entangled_N={N}.csv", delimiter=',', skip_header=1)
    stds[N-2] = np.std(data.T, axis=1)


unentangled_data = np.genfromtxt(f"measurements_unentangled.csv", delimiter=',', skip_header=1)
unentangled_stds = np.std(unentangled_data.T, axis=1)


objs = plt.plot(np.arange(2,11), stds, '.', markersize=8)
obj_ghzideal = plt.plot([2,11], [unentangled_stds[0], unentangled_stds[0]], '--', label="GHZ ideal")
obj_ghznoise = plt.plot([2,11], [unentangled_stds[1], unentangled_stds[1]], '--', label="GHZ noise")
plt.xlabel("$n$")
plt.ylabel(r"$\sigma_{\theta}$")
legend = plt.legend([r'$\sigma_e$ ideal', r'$\sigma_e$ entangled', "GHZ noise", "GHZ ideal"])
legend.get_frame().set_alpha(None)
plt.plot(np.arange(2,11), stds, '--', color="#BBBBBB", zorder=-10)
plt.show()

plt.plot(np.arange(2,11), stds[:, 0]/unentangled_stds[0], '.', markersize=8, label=r"$\sigma_e/\sigma_u$")
plt.plot(np.arange(2,11), stds[:, 0]/unentangled_stds[0], '--', color="#BBBBBB", zorder=-10)
plt.plot(np.linspace(2,11, 200), 1/np.sqrt(np.linspace(2,11, 200)), label="$1/\sqrt{n}$")
plt.xlabel("$n$")
plt.ylabel(r"r")
legend = plt.legend()
legend.get_frame().set_alpha(None)
plt.show()

plt.plot(np.arange(2, 11), stds[:, 1]/unentangled_stds[1], '.', markersize=8, label=r"$\sigma_e/\sigma_u$")
plt.plot(np.arange(2,11), stds[:, 1]/unentangled_stds[1], '--', color="#BBBBBB", zorder=-10)
plt.plot(np.linspace(2, 11, 200), 1/np.sqrt(np.linspace(2,11, 200)), label="$1/\sqrt{n}$")
plt.xlabel("$n$")
plt.ylabel(r"r")
legend = plt.legend()
legend.get_frame().set_alpha(None)
plt.show()
