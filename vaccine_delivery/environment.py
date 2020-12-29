import logging

import numpy as np
from scipy.stats import poisson


logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DISCOUNT = 0.5


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

    def copy(self):
        return drug_centre(
            cost_vaccine=self.cost_vaccine,
            fee_vaccine=self.fee_vaccine,
            state=self.get_state()
        )


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


