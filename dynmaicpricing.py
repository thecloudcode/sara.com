import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Environment simulating sales volume based on pricing
class PriceOptimizationEnv:
    def __init__(self, base_demand, elasticity, competitors_price, max_sales):
        self.base_demand = base_demand
        self.elasticity = elasticity
        self.competitors_price = competitors_price
        self.max_sales = max_sales  # Maximum number of units that can be sold
        self.state = [base_demand, competitors_price]

    def reset(self):
        # Reset the environment to its initial state
        self.state = [self.base_demand, self.competitors_price]
        return np.array(self.state)

    def step(self, action):
        # The action is the price selected by the agent
        price = action
        # Demand decreases as price increases (elasticity effect)
        demand = self.base_demand - self.elasticity * (price - self.competitors_price)
        # Limit sales to a maximum value (real-world capacity limit or inventory constraint)
        sales_volume = max(0, min(self.max_sales, demand))
        # Revenue = price * sales volume
        revenue = price * sales_volume
        # Reward is the revenue, which the agent tries to maximize
        reward = revenue

        # Environment state can be dynamic (e.g., change competitor prices or base demand)
        self.state = [self.base_demand, self.competitors_price]

        return np.array(self.state), reward, False

# Q-Network (DQN)
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

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.batch_size = 64

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.batch_size:
            # Sample a batch of experiences and learn from them
            experiences = random.sample(self.memory, self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        # Convert batch of experiences into tensors
        states, actions, rewards, next_states = zip(*experiences)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Get max predicted Q-values for next states from the target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for the current states
        Q_targets = rewards + (self.gamma * Q_targets_next)

        # Get expected Q-values from the local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        # Compute loss
        loss = self.criterion(Q_expected, Q_targets)

        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        # Reduce epsilon for less exploration as the agent learns
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training function
def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()  # Reset the environment for a new episode
        total_reward = 0
        for t in range(100):  # Max steps per episode
            action = agent.act(state)  # Select action (price)
            next_state, reward, done = env.step(action)  # Take action in the environment
            agent.step(state, action, reward, next_state)  # Store experience and train
            state = next_state  # Update state
            total_reward += reward  # Accumulate rewards
            if done:
                break
        agent.update_epsilon()  # Decay exploration rate
        print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward}")

# Initialize environment and agent
env = PriceOptimizationEnv(base_demand=100, elasticity=2, competitors_price=10, max_sales=150)
state_size = 2  # Demand and competitor price
action_size = 101  # Discrete price points between 0 and 100
agent = DQNAgent(state_size, action_size)

# Train the agent
train_agent(env, agent)
