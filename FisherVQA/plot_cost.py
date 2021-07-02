import numpy as np
import matplotlib.pyplot as plt


# reference_cost =  3.734  # real
reference_cost = 9.452    # sim
true_phi = np.ones(4)*np.pi/5

filename = "test_4qubits_bogotasim_safe.csv" #"bogotareal.csv"

data = np.genfromtxt(filename, delimiter=',', skip_header=1)


costs = data.T[1]


# plt.subplot(121)

plt.plot(np.arange(len(costs))+1, costs, '.', markersize=8, zorder=10, label="VQA")
plt.plot(np.arange(len(costs))+1, costs, '--', color='#BBBBBB')
plt.plot([1, len(costs)], [reference_cost]*2, 'g--', label="GHZ")
plt.ylim([0, 20])
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.xticks(np.arange(0, len(costs)+2, 2))
legend = plt.legend()
legend.get_frame().set_alpha(None)
plt.show()
