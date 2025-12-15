
import numpy as np
import pickle
import os

BASE = "/content/drive/MyDrive/teamforge"
STATE_PATH = os.path.join(BASE, "bandit_state.pkl")

class ThompsonBandit:
    def __init__(self, decay=0.98):
        self.decay = decay
        self.arms = {}
        self._load()

    def _load(self):
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "rb") as f:
                    self.arms = pickle.load(f)
            except Exception:
                self.arms = {}

    def _save(self):
        with open(STATE_PATH, "wb") as f:
            pickle.dump(self.arms, f)

    def sample(self, arm_id):
        arm_id = int(arm_id)
        if arm_id not in self.arms:
            self.arms[arm_id] = [1.0, 1.0]  # alpha, beta
        a, b = self.arms[arm_id]
        return np.random.beta(a, b)

    def update(self, arm_id, reward):
        arm_id = int(arm_id)
        if arm_id not in self.arms:
            self.arms[arm_id] = [1.0, 1.0]

        # decay old belief
        self.arms[arm_id][0] *= self.decay
        self.arms[arm_id][1] *= self.decay

        # update with new reward
        if reward > 0:
            self.arms[arm_id][0] += reward
        else:
            self.arms[arm_id][1] += 1.0

        self._save()

# global singleton
bandit = ThompsonBandit()
