import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.test.mock import FakeBogota
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.aer import AerSimulator
from fisher_vqa import FisherVQA, FisherVQA_IBMQ

# If you want to use an IBMQ computer, make an account and look for the token. Make a file called token.txt and put it
# in this folder. Without a reservation for a computer, this will be slow though.
# with open("token.txt") as tokenfile:
#     token = tokenfile.readline()
#
# IBMQ.enable_account(token)

# -- uncomment this for least busy small system --
# provider = IBMQ.get_provider(hub='ibm-q')
# small_filter = lambda x: x.configuration().n_qubits == 5 and not x.configuration().simulator
# devices = provider.backends(filters=lambda x: x.configuration().simulator)
# backend = qiskit.providers.ibmq.least_busy(devices)

# -- uncomment this for Bogota --
# provider = IBMQ.get_provider(hub='ibm-q')
# print(provider.backends())
# backend = provider.get_backend("ibmq_bogota")
# print(backend)

# -- uncomment this for Bogota simulator
backend = AerSimulator.from_backend(FakeBogota())

# -- uncomment this for ideal simulator
# backend = AerSimulator()


# define circuit to be used
def circ4qubits(theta, phi):
    circ = QuantumCircuit(4)
    circ.h(range(4))
    for i, t in enumerate(theta[:4]):
        circ.p(t, i)
    circ.h(range(1, 4))
    for i, t in enumerate(theta[4:]):
        circ.crx(t, i, i+1)
    for i, t in enumerate(phi):
        circ.p(t, i)
    circ.h(range(4))
    circ.measure_all()
    return circ


# show the circuit!
# circ4qubits(np.arange(7)+1, np.arange(4)+1).draw('mpl')
# plt.show()


ghz_theta = np.array([0.0]*4 + [np.pi]*3)
initial_theta = np.array([0]*4 + [np.pi/2]*3)
true_phi = np.array(4*[np.pi/5])
vqa_4qubits = FisherVQA(circ4qubits, "2222444", backend, 4, shots=8192)
# if you are using an IBMQ computer, replace FisherVQA by FisherVQA_IBMQ for significant performance improvements

print("GHZ cost:", vqa_4qubits.fisher_cost(ghz_theta, true_phi))

vqa_4qubits.optimize_theta(initial_theta, true_phi, maxiter=20)   # find optimal probe state
vqa_4qubits.optimize_phi(np.ones(4), true_phi, target_shots=8192, log_filename="phi_opt_bogotasim.txt", maxiter=20,
                         maxstep=.2, normalize=True)  # find phi using MLE
vqa_4qubits.theta_opt = ghz_theta
vqa_4qubits.optimize_phi(np.ones(4), true_phi, target_shots=8192, log_filename="phi_ghz_bogotasim.txt", maxiter=20,
                         maxstep=.2, normalize=True)  # compare to GHZ estimation

