import unittest
import pdb
from reacher_env import ReacherEnvironment
from unityagents import UnityEnvironment
import torch
import numpy as np

class TestReacherEnv(unittest.TestCase):

    def setUp(self):
        self.env = ReacherEnvironment()

    def tearDown(self):
        self.env.close()

    def test_can_create_environment(self):
        state = self.env.reset()
        self.assertEqual(state.shape, (20,33))

    def test_can_step_environment(self):
        actions = np.zeros(self.env.action_shape)
        next_states, rewards, dones = self.env.step(actions)
        self.assertEqual(next_states.shape, (20,33))
        self.assertEqual(rewards.shape, (20,))
        self.assertEqual(dones.shape, (20,))