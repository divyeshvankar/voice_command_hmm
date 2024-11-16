import numpy as np

class HMM:
    def __init__(self, n_states, n_obs):
        self.n_states = n_states
        self.n_obs = n_obs
        self.start_prob = np.random.dirichlet(np.ones(n_states))
        self.trans_prob = np.random.dirichlet(np.ones(n_states), size=n_states)
        self.emit_prob = np.random.dirichlet(np.ones(n_obs), size=n_states)

        print(f"Initialized start_prob: {self.start_prob}")
        print(f"Initialized trans_prob: {self.trans_prob}")
        print(f"Initialized emit_prob: {self.emit_prob}")

    def _forward(self, sequence):
        T = len(sequence)
        alpha = np.zeros((T, self.n_states))
        # Initialization
        alpha[0, :] = self.start_prob * self.emit_prob[:, sequence[0]]
        # Induction
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.trans_prob[:, j]) * self.emit_prob[j, sequence[t]]
        return alpha

    def _backward(self, sequence):
        T = len(sequence)
        beta = np.zeros((T, self.n_states))
        # Initialization
        beta[T - 1, :] = 1
        # Induction
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.trans_prob[i, :] * self.emit_prob[:, sequence[t + 1]] * beta[t + 1, :])
        return beta

    def _e_step(self, sequences):
        gamma_list = []
        xi_list = []

        for sequence in sequences:
            T = len(sequence)
            alpha = np.zeros((T, self.n_states))
            beta = np.zeros((T, self.n_states))
            gamma = np.zeros((T, self.n_states))
            xi = np.zeros((T - 1, self.n_states, self.n_states))

            # Forward Pass (Alpha)
            alpha[0, :] = self.start_prob * self.emit_prob[:, sequence[0]]  # Fixed indexing
            alpha[0, :] /= np.sum(alpha[0, :])  # Normalize
            for t in range(1, T):
                for j in range(self.n_states):
                    alpha[t, j] = np.sum(alpha[t - 1, :] * self.trans_prob[:, j]) * self.emit_prob[j, sequence[t]]
                alpha[t, :] /= np.sum(alpha[t, :])  # Normalize

            # Backward Pass (Beta)
            beta[-1, :] = 1
            for t in range(T - 2, -1, -1):
                for i in range(self.n_states):
                    beta[t, i] = np.sum(self.trans_prob[i, :] * self.emit_prob[:, sequence[t + 1]] * beta[t + 1, :])
                beta[t, :] /= np.sum(beta[t, :])  # Normalize

            # Gamma Calculation
            for t in range(T):
                gamma[t, :] = alpha[t, :] * beta[t, :]
                gamma[t, :] /= np.sum(gamma[t, :])  # Normalize

            # Xi Calculation
            for t in range(T - 1):
                denominator = np.sum(alpha[t, :] * beta[t, :])
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self.trans_prob[i, j] * self.emit_prob[j, sequence[t + 1]] * beta[t + 1, j]
                xi[t, :, :] /= denominator

            gamma_list.append(gamma)
            xi_list.append(xi)

        return gamma_list, xi_list




    def _m_step(self, sequences, gamma_list, xi_list):
        # Update start probabilities
        self.start_prob = np.mean([gamma[0, :] for gamma in gamma_list], axis=0)
        print(f"Updated start probabilities: {self.start_prob}")

        # Update transition probabilities
        numerator = np.zeros((self.n_states, self.n_states))
        denominator = np.zeros((self.n_states,))
        for xi in xi_list:
            numerator += np.sum(xi, axis=0)
            denominator += np.sum(xi, axis=(0, 2))

        if np.any(denominator == 0):
            print("Transition denominator contains zero values. Adding epsilon.")
            denominator += 1e-8  # Add epsilon for stability

        self.trans_prob = numerator / denominator[:, None]
        print(f"Transition numerator:\n{numerator}")
        print(f"Transition denominator:\n{denominator}")
        print(f"Updated transition probabilities:\n{self.trans_prob}")

        # Update emission probabilities
        numerator = np.zeros((self.n_states, self.n_obs))
        denominator = np.zeros((self.n_states,))
        for sequence, gamma in zip(sequences, gamma_list):
            for t in range(len(sequence)):
                numerator[:, sequence[t]] += gamma[t, :]
                denominator += gamma[t, :]

        if np.any(denominator == 0):
            print("Emission denominator contains zero values. Adding epsilon.")
            denominator += 1e-8  # Add epsilon for stability

        self.emit_prob = numerator / denominator[:, None]
        print(f"Emission numerator:\n{numerator}")
        print(f"Emission denominator:\n{denominator}")
        print(f"Updated emission probabilities:\n{self.emit_prob}")

        # State occupancy debugging
        state_occupancy = np.sum([np.sum(gamma, axis=0) for gamma in gamma_list], axis=0)
        print(f"State occupancy during training: {state_occupancy}")
        if np.any(state_occupancy == 0):
            print("Warning: Some states have zero occupancy.")




    def train(self, sequences, max_iter=100):
        for iteration in range(max_iter):
            # Expectation Step
            gamma_list, xi_list = self._e_step(sequences)

            # Maximization Step
            self._m_step(sequences, gamma_list, xi_list)
            print(f"Iteration {iteration + 1}/{max_iter} completed.")

    def predict(self, sequence):
        T = len(sequence)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        delta[0, :] = self.start_prob * self.emit_prob[:, sequence[0]]
        # Induction
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t - 1, :] * self.trans_prob[:, j]) * self.emit_prob[j, sequence[t]]
                psi[t, j] = np.argmax(delta[t - 1, :] * self.trans_prob[:, j])
        # Termination
        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(delta[T - 1, :])
        # Path backtracking
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states
