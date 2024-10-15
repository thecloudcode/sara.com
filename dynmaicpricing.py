import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class PriceOptimizationEnv:
    def __init__(self, base_demand, elasticity, competitors_price, max_sales):
        self.base_demand = base_demand
        self.elasticity = elasticity
        self.competitors_price = competitors_price
        self.max_sales = max_sales
        self.state = [base_demand, competitors_price]

    def reset(self):
        self.state = [self.base_demand, self.competitors_price]
        return np.array(self.state)

    def step(self, action):
        price = action
        demand = self.base_demand - self.elasticity * (price - self.competitors_price)
        sales_volume = max(0, min(self.max_sales, demand))
        revenue = price * sales_volume
        reward = revenue

        self.state = [self.base_demand, self.competitors_price]
        return np.array(self.state), reward, False

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.batch_size:
            experiences = random.sample(self.memory, self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states = zip(*experiences)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next)

        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        loss = self.criterion(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for t in range(100):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.update_epsilon()
        print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward}")


env = PriceOptimizationEnv(base_demand=100, elasticity=2, competitors_price=10, max_sales=150)
state_size = 2
action_size = 101
agent = DQNAgent(state_size, action_size)

train_agent(env, agent)