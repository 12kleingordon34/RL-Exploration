import logging

import numpy as np
from scipy.stats import poisson


logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DISCOUNT = 0.5


class bellman_agent():
    def __init__(self, max_vax, max_delivery):
        self.V = np.zeros((max_vax+1, max_vax+1))
        self.policy = np.zeros((max_vax+1, max_vax+1))
        self.max_vax = max_vax
        self.max_delivery = max_delivery
        self.is_policy_stable = False
        self.actions = np.arange(0, max_delivery+1)

    def policy_evaluation(self, centre, arrival_distribution, verbose=False):
        """
        i.e. Prediction
        """
        error = 0.01
        delta = 1
        num_policy_eval_iterations = 0
        while delta > error:
            num_policy_eval_iterations += 1
            delta = 0
            for i in range(self.V.shape[0]):
                for j in range(self.V.shape[1]):
                    v_old = self.V[i, j]
                    centre.reset((i, j))
                    if verbose:
                        logging.debug("pi(s) = {}, E[r] = {}".format(self.policy[i,j], self.expected_reward(
                            centre, self.policy[i,j], self.V, arrival_distribution)))
                    self.V[i, j] = self.expected_reward(
                        centre, self.policy[i,j], self.V, arrival_distribution
                    )
                    delta = max([delta, np.abs(v_old - self.V[i, j])])
            logging.debug(
                f'Num. Pol. Evals: {num_policy_eval_iterations} -- Delta: {np.round(delta, 4)}'
            )
                # if delta < error:
                #     break

    def policy_improvement(self, centre, arrival_distribution, verbose=False):
        is_policy_stable = True
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                if verbose:
                    logging.debug("state = {},{}".format(i,j))
                old_policy = self.policy[i, j]
                action_rewards = np.zeros((self.max_delivery+1))
                centre.reset((i, j))
                for delivery in self.actions:
                    if verbose:
                        logging.debug("a = {}, E[r] = {}".format(delivery, self.expected_reward(
                            centre, delivery, self.V, arrival_distribution)))
                    action_rewards[delivery] = self.expected_reward(
                        centre, delivery, self.V, arrival_distribution
                    )
                self.policy[i, j] = self.actions[int(np.argmax(action_rewards))]
                if is_policy_stable and (old_policy != self.policy[i, j]):
                    is_policy_stable = False
        self.is_policy_stable = is_policy_stable

    @staticmethod
    def expected_reward(centre, action, V, dist):
        global DISCOUNT

        temp_centre = centre.copy()
        old_state = temp_centre.get_state()
        V_s = 0
        for patient_no in range(dist.max_arrivals+1):
            temp_centre.reset(old_state)
            prob = dist.call(patient_no)
            temp_centre.treat_patients(patient_no)
            temp_centre.delivery(action)
            new_state = temp_centre.get_state()
            reward = temp_centre.get_reward()

            V_s += prob * (reward + DISCOUNT * V[new_state[0], new_state[1]])
        return V_s


class tabular_model_free_agent():
    def __init__(self, max_vax, max_delivery, alpha, behaviour_policy, target_policy):
        self.q = np.zeros((max_vax+1, max_vax+1, max_delivery+1))
        self.behaviour_policy = behaviour_policy
        self.target_policy = target_policy
        self.alpha = alpha
        self.max_vax = max_vax

        self.max_delivery = max_delivery
        # Randomly assign the first action to be delivering no vaccines
        self.last_action = 0
        self.old_state = (0,0)

    def step(self, reward, next_state, discount=DISCOUNT):
        old_vaccines = next_state[0]; new_vaccines = next_state[1]
        next_action = self.behaviour_policy(
            self.q[old_vaccines, new_vaccines, :]
        )

        # Return a policy vector over all action outcomes
        target_policy = self.target_policy(
            self.q[old_vaccines, new_vaccines, :],
            next_action
        )

        delta = reward + discount * np.dot(
            target_policy,
            self.q[old_vaccines, new_vaccines, :]
        ) - self.q[old_vaccines, new_vaccines, next_action]

        self.q[
            self.old_state[0],
            self.old_state[1],
            self.last_action
        ] += self.alpha * delta

        self.last_action = next_action
        self.old_state = next_state
        return next_action

