import pdb
import unittest
from reacher_env import ReacherEnvironment
from model import GaussianActorCritic
from agent_ppo import Agent
import torch

class TestAgentPPO(unittest.TestCase):

    def setUp(self):
        self.env = ReacherEnvironment()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = GaussianActorCritic(self.env.state_size, self.env.action_size).to(self.device)
        
    def tearDown(self):
        self.env.close()

    def test_can_create_model(self):
        state = self.env.reset()
        action, log_prob, entropy, v = self.net.forward(torch.from_numpy(state).float().to(self.device))
        self.assertEqual(action.shape, (self.env.num_agents, self.env.action_size))
        self.assertEqual(log_prob.shape, (self.env.num_agents, self.env.action_size))
        self.assertEqual(entropy.shape, (self.env.num_agents, self.env.action_size))
        self.assertEqual(v.shape, (self.env.num_agents,1))

    def test_can_step_environment(self):
        state = self.env.reset()
        action, _, _, _ = self.net.forward(torch.from_numpy(state).float().to(self.device))
        action = action.detach().cpu().numpy()
        next_state, reward, done = self.env.step(action)
        self.assertEqual(next_state.shape, state.shape)

    def test_can_step_agent(self):
        self.agent = Agent(self.env, self.net, epochs=1)
        self.assertEqual(self.agent.steps, 0)
        self.agent.step()
        self.assertGreater(self.agent.steps, 0)




