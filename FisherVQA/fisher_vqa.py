import numpy as np

from qiskit import *
from qiskit.providers.aer import AerSimulator
from qiskit.providers.ibmq.managed import IBMQJobManager
from optimization import gradient_descent_backtrack_momentum


def counts_to_array(counts):
    n = counts.memory_slots
    counts = counts.int_outcomes()
    return np.array([counts.get(ell, 0) for ell in range(2**n)])


class CallbackFunction:  # for compatibility between function signatures
    """
    For compatibility between function signatures of FisherVQA and its _IBMQ version.
    """
    def __init__(self, job_function, do_function):
        self.job = job_function
        self.do = do_function

    def __call__(self, *args, **kwargs):
        job = self.job(*args, **kwargs)
        return self.do(job)

    def __iter__(self):
        yield self.job
        yield self.do


class FisherVQA:
    """
    A convenient class that packages all the data necessary for executing the VQA
    """
    def __init__(self, circ, shift_rules, backend, n_qubits, jacobian=None, shots=1024):
        """
        Args:
            circ:        The circuit you use. This must be the combined circuit of the probe preparation,
                            parameter encoding and measurement
            shift_rules: Which shift rules are to be used for which parameters. A string of '2's and '4's indicating
                            whether to use the 2- or 4-parameter shift rule
            backend:     Which backend to use for the simulation
            n_qubits:    The amount of qubits in the circuit
            jacobian:    The jacobian of the function of the parameters you want an optimal probe for
            shots:       The amount of times that each circuit is run
        """
        self.circ = circ
        self.n_qubits = n_qubits
        self.signature = shift_rules
        self.n_param = len(shift_rules)
        if jacobian is None:
            self.jacobian = np.ones(n_qubits)/n_qubits
        else:
            self.jacobian = jacobian
        self.backend = backend
        self.shots = shots
        self.theta_opt = None
        self.phi_opt = None

    def expvals_p(self, theta, phi):
        """
        Estimates p_l for all l and returns as array
        Args:
            theta:   circuit parameters
            phi:     parameters to be estimated
            shots:   amount of shots per quantum experiment
            backend: simulation backend
            circ:    circuit that produces the probabilities

        Returns: numpy array of probabilities
        """

        circ_ev = self.circ(theta, phi)
        tcirc = transpile(circ_ev, self.backend)
        counts = execute(tcirc, self.backend, shots=self.shots).result().get_counts()
        p = utils.counts_to_array(counts)/self.shots
        return p

    def gradients_phi(self, theta, phi):
        """
        Calculates the gradients of expvals_p with respect to the phi parameters
        """
        dphi_p_l = []
        for i in range(len(phi)):
            e = np.zeros(len(phi))
            e[i] = np.pi/2
            p_plus = self.expvals_p(theta, phi + e)
            p_minus = self.expvals_p(theta, phi - e)
            dphi_p_l.append((p_plus - p_minus)/2)
        return np.array(dphi_p_l)

    def gradients_theta(self, cost, theta, phi):  # specific to the U_3qubits circuit
        """
        Estimates the gradients of the given cost with respect to the components of theta, and returns an array of
        the gradients for each component
        Args:
            cost:    which cost to take the gradient of
            theta:   circuit parameters
            phi:     parameters to be estimated
            shots:   amount of shots per quantum experiment
            backend: simulation backend
            circ:    circuit that produces the probabilities
            signature: indicates which parameter shift rule to use for which parameter. 2 is for the 2 paramter rule,
                        4 for the 4 parameter rule

        Returns: numpy array of the gradients. If the cost returns an array, this function returns an array of one dimension
        higher, where the first index indicates which theta derivative was taken
        """
        cost_theta = lambda th: cost(th, phi)  # theta is only changing parameter, reduces clutter
        n = len(theta)
        grad_vector = []
        for i, s in enumerate(self.signature):
            if s == "2":
                e_i = np.zeros(n)
                e_i[i] = np.pi/2
                grad_component = 0.5*(cost_theta(theta + e_i) - cost_theta(theta - e_i))
            else:
                e_i = np.zeros(n)
                e_i[i] = 1
                a = np.pi/2
                b = 3*a
                c_plus = (np.sqrt(2) + 1) / 4 / np.sqrt(2)
                c_minus = (np.sqrt(2) - 1) / 4 / np.sqrt(2)
                grad_component = c_plus*(cost_theta(theta + a*e_i) - cost_theta(theta - a*e_i)) - \
                                 c_minus*(cost_theta(theta + b*e_i) - cost_theta(theta - b*e_i))
                # grad_component = 0.5*(cost_theta(theta + np.pi/2*e_i) - cost_theta(theta - np.pi/2*e_i)) - \
                #                  (np.sqrt(2) - 1)/4*(cost_theta(theta + np.pi*e_i) - cost_theta(theta - np.pi*e_i))
            grad_vector.append(grad_component)
        return np.array(grad_vector)

    def fisher_information(self, theta, phi):  # phi typically given, theta to optimize over
        fisher_matrix = np.zeros((self.n_qubits, self.n_qubits))
        p = self.expvals_p(theta, phi)
        d_p = self.gradients_phi(theta, phi)
        for j in range(self.n_qubits):
            for k in range(self.n_qubits):
                fisher_matrix[j, k] = sum(d_p[j, l]*d_p[k, l]/p[l] for l in range(8) if p[l] != 0)

        print(" done")
        return fisher_matrix


    def gradient_fisher_phi(self, theta, phi):
        n = len(phi)
        n_param = len(theta)
        p_l = self.expvals_p(theta, phi)
        dtheta_p_l = self.gradients_theta(self.expvals_p, theta, phi)
        dphi_p_l = self.gradients_phi(theta, phi)
        dtheta_dphi_p_l = self.gradients_theta(self.gradients_phi, theta, phi)  # array indices: theta, phi, p
        fisher_gradients = np.zeros((n_param, n, n))

        for i in range(n_param):
            for j in range(n):
                for k in range(n):
                    fisher_gradients[i, j, k] = np.nan_to_num(np.sum((dphi_p_l[j] * dtheta_dphi_p_l[i, k] +
                                                                      dtheta_dphi_p_l[i, j] * dphi_p_l[k])/p_l -
                                                                     dphi_p_l[j] * dphi_p_l[k] * dtheta_p_l[i] / p_l**2))
        return fisher_gradients

    def fisher_f(self, theta, phi):
        J = np.atleast_2d(self.jacobian.T).T  # jacobian that also works if jacobian is 1D (then reshaped into column vector)
        return J.T @ self.fisher_information(theta, phi) @ J

    def fisher_cost(self, theta, phi):
        I_f_inv = np.linalg.inv(self.fisher_f(theta, phi))
        n = I_f_inv.shape[0]  # output dimension of f
        return np.trace(np.ones((n, n)) @ I_f_inv)

    def gradient_fisher_f(self, theta, phi):
        denominator = self.fisher_f(theta, phi)
        gradient = []
        for i in range(self.n_param):
            gradient.append(-self.jacobian @ self.gradient_fisher_phi(theta, phi)[i] @ self.jacobian / denominator ** 2)
        return np.array(gradient)

    def gradient_fisher_cost(self, theta, phi):
        J = np.atleast_2d(self.jacobian.T).T
        n = J.shape[1]  # output dimension of f
        gradient = []
        I_f = self.fisher_f(theta, phi)
        I_f_inv = np.linalg.inv(I_f)
        dI_phi = self.gradient_fisher_phi(theta, phi)
        for i in range(self.n_param):
            dI_f = J.T @ dI_phi[i] @ J
            dI_f_inv = - I_f_inv @ dI_f @ I_f_inv
            gradient.append(np.trace(np.ones((n, n)) @ dI_f_inv))
        return np.array(gradient)

    def log_likelihood(self, theta, phi, target_probs):
        probabilities = self.expvals_p(theta, phi)
        return np.dot(np.nan_to_num(np.log(probabilities)), target_probs)

    def grad_log_likelihood(self, theta, phi, target_probs):
        probabilities = self.expvals_p(theta, phi)
        dp_phi = self.gradients_phi(theta, phi)
        return np.sum(np.nan_to_num(target_probs*dp_phi/probabilities), axis=1)

    def optimize_theta(self, initial_theta, true_phi, **kwargs):
        cost = lambda th: self.fisher_cost(th, true_phi)
        grad = lambda th: self.gradient_fisher_cost(th, true_phi)
        _, theta_opt = gradient_descent_backtrack_momentum(cost, grad, initial_theta, **kwargs)
        self.theta_opt = theta_opt
        return theta_opt

    def optimize_phi(self, initial_phi, true_phi, target_shots=8192, likelihood_shots=8192, **kwargs):
        shots_save = self.shots
        self.shots = likelihood_shots  # not a great solution, but eh
        target_counts = execute(self.circ(self.theta_opt, true_phi), self.backend, shots=target_shots).result().get_counts()
        target_counts = utils.counts_to_array(target_counts)
        cost = lambda phi: -self.log_likelihood(self.theta_opt, phi, target_counts)
        grad = lambda phi: -self.grad_log_likelihood(self.theta_opt, phi, target_counts)
        _, phi_opt = gradient_descent_backtrack_momentum(cost, grad, initial_phi, 1.9, **kwargs)
        self.phi_opt = phi_opt
        self.shots = shots_save
        return phi_opt


class FisherVQA_IBMQ(FisherVQA):
    """
    Version of FisherVQA optimized for use on IBMQ computers
    """
    def __init__(self, circ, shift_rules, backend, n_qubits, jacobian=None, shots=1024):
        super().__init__(circ, shift_rules, backend, n_qubits, jacobian, shots)
        self.job_manager = IBMQJobManager()
        self.expvals_p = CallbackFunction(self.expvals_p_job, self.expvals_p_do)
        self.gradients_phi = CallbackFunction(self.gradients_phi_job, self.gradients_phi_do)

    def multi_expvals_p_job(self, thetas, phis):
        circuits = [self.circ(theta, phi) for theta, phi in zip(thetas, phis)]
        transpiled_circuits = transpile(circuits, backend=self.backend)
        return self.job_manager.run(transpiled_circuits, backend=self.backend, shots=self.shots)

    def multi_expvals_p_do(self, job):
        return lambda i: utils.counts_to_array(job.results().get_counts(i))/self.shots

    def expvals_p_job(self, theta, phi):
        return self.multi_expvals_p_job([theta], [phi])

    def expvals_p_do(self, job):
        return self.multi_expvals_p_do(job)(0)

    def gradients_phi_job(self, theta, phi):
        thetas = [theta]*self.n_qubits*2  # duplicate theta vector since it's the same each time
        phis = []
        for i in range(self.n_qubits):
            e = np.zeros(self.n_qubits)
            e[i] = np.pi/2
            phis += [phi + e, phi - e]

        job = self.multi_expvals_p_job(thetas, phis)
        return job

    def gradients_phi_do(self, job):
        ps = self.multi_expvals_p_do(job)
        dphi_p_l = []
        for i in range(self.n_qubits):
            dphi_p_l.append((ps(2*i) - ps(2*i+1))/2)
        return np.array(dphi_p_l)

    def gradients_theta(self, cost_callback, theta, phi):
        """
        Estimates the gradients of the given cost with respect to the components of theta, and returns an array of
        the gradients for each component
        Args:
            cost_callback:    which cost to take the gradient of. Type: CallbackFunction
            theta:   circuit parameters
            phi:     parameters to be estimated
            shots:   amount of shots per quantum experiment
            backend: simulation backend
            circ:    circuit that produces the probabilities
            signature: indicates which parameter shift rule to use for which parameter. 2 is for the 2 paramter rule,
                        4 for the 4 parameter rule

        Returns: numpy array of the gradients. If the cost returns an array, this function returns an array of one dimension
        higher, where the first index indicates which theta derivative was taken
        """
        cost_job, cost_do = cost_callback
        cost_theta_job = lambda th: cost_job(th, phi)  # theta is only changing parameter, reduces clutter
        n = len(theta)
        jobs = []
        a = np.pi/2
        b = 3*a
        for i, s in enumerate(self.signature):
            if s == "2":
                e_i = np.zeros(n)
                e_i[i] = np.pi/2
                jobs.append([cost_theta_job(theta + e_i), cost_theta_job(theta - e_i)])
            elif s == "4":
                e_i = np.zeros(n)
                e_i[i] = 1

                jobs.append([cost_theta_job(theta + a*e_i), cost_theta_job(theta - a*e_i),
                             cost_theta_job(theta + b*e_i), cost_theta_job(theta - b*e_i)])
            else:
                raise ValueError(f"Invalid value {s} in signature")

        grad_vector = []
        c_plus = (np.sqrt(2) + 1) / 4 / np.sqrt(2)
        c_minus = (np.sqrt(2) - 1) / 4 / np.sqrt(2)
        for i, s in enumerate(self.signature):
            if s == "2":
                grad_component = 0.5*(cost_do(jobs[i][0]) - cost_do(jobs[i][1]))
            else:
                grad_component = c_plus*(cost_do(jobs[i][0]) - cost_do(jobs[i][1])) - \
                                 c_minus*(cost_do(jobs[i][2]) - cost_do(jobs[i][3]))
            grad_vector.append(grad_component)
        return np.array(grad_vector)

