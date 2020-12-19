import numpy as np
from scipy.stats import poisson
import logging


DISCOUNT = 0.6

#test 3

class drug_centre():
    def __init__(self, cost_vaccine, fee_vaccine, state=(0,0)):
        self.cost_vaccine = cost_vaccine
        self.fee_vaccine = fee_vaccine
        self.old_vaccines = state[0]
        self.new_vaccines = state[1]
        self.last_step_treated = 0
        self.last_step_expired = 0

    def delivery(self, load):
        """
        Updates state, and returns a loss corresponding
        to the expired vaccines.
        """
        self.old_vaccines = self.new_vaccines
        self.new_vaccines = load


    def treat_patients(self, patient_no):
        """
        Treats a number of patients using the vaccines
        stored in the centre.

        Returns the reward corresponding to the
        """
        if patient_no < self.old_vaccines:
            patients_treated = patient_no
            self.old_vaccines -= patient_no
            self.last_step_treated = patients_treated
        elif patient_no < (self.old_vaccines + self.new_vaccines):
            patients_treated = patient_no
            patient_no -= self.old_vaccines
            self.old_vaccines = 0
            self.new_vaccines -= patient_no
            self.last_step_treated = patients_treated
        else:
            patients_treated = self.old_vaccines + self.new_vaccines
            self.new_vaccines = 0
            self.old_vaccines = 0
            self.last_step_treated = patients_treated

        # at end of treatment count expired stock and bin it (set old to 0)
        self.last_step_expired = self.old_vaccines
        self.old_vaccines = 0

    def get_reward(self):
        return (
            self.fee_vaccine * self.last_step_treated
            - self.cost_vaccine * self.last_step_expired
        )


    def get_state(self):
        return (int(self.old_vaccines), int(self.new_vaccines))


    def reset(self, state=(0,0)):
        self.old_vaccines = state[0]
        self.new_vaccines = state[1]
        self.last_step_treated = 0
        self.last_step_expired = 0


class truncated_patient_arrival_distribution():
    def __init__(self, max_arrivals, rate):
        self.rate = rate
        self.max_arrivals = max_arrivals
        self.dist = poisson.pmf(np.arange(0, self.max_arrivals+1, 1), mu=self.rate)
        self.dist = self.dist / self.dist.sum()

    def call(self, num_arrivals):
        assert num_arrivals >= 0
        if 0 <= num_arrivals <= self.max_arrivals:
            return self.dist[num_arrivals]
        else:
            return 0


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
        while delta > error:
            delta = 0
            for i in range(self.V.shape[0]):
                for j in range(self.V.shape[1]):
                    v_old = self.V[i, j]
                    centre.reset((i, j))
                    if verbose:
                        print("pi(s) = {}, E[r] = {}".format(self.policy[i,j], self.expected_reward(
                            centre, self.policy[i,j], self.V, arrival_distribution)))
                    self.V[i, j] = self.expected_reward(
                        centre, self.policy[i,j], self.V, arrival_distribution
                    )
                    delta = max([delta, np.abs(v_old - self.V[i, j])])
                    logging.info(f'delta = {delta}')
                # if delta < error:
                #     break

    def policy_improvement(self, centre, arrival_distribution, verbose=False):
        is_policy_stable = True
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                if verbose:
                    print("state = {},{}".format(i,j))
                old_policy = self.policy[i, j]
                action_rewards = np.zeros((self.max_delivery+1))
                centre.reset((i, j))
                for delivery in self.actions:
                    if verbose:
                        print("a = {}, E[r] = {}".format(delivery, self.expected_reward(
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

        old_state = centre.get_state()
        V_s = 0
        for patient_no in range(dist.max_arrivals+1):
            centre.reset(old_state)
            prob = dist.call(patient_no)
            centre.treat_patients(patient_no)
            centre.delivery(action)
            new_state = centre.get_state()
            reward = centre.get_reward()

            V_s += prob * (reward + DISCOUNT * V[new_state[0], new_state[1]])
        return V_s
