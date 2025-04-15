import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 设计Encoder的网络框架
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 16)

    def forward(self, depth_image):
        """
        input:(batch_size,1,64,64)
        output:(batch_size,16)
        """
        x = self.pool(F.relu(self.conv1(depth_image)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        feature = self.fc2(x)

        return feature


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.Encoder = Encoder()  # 卷积层

        self.l1 = nn.Linear(state_dim + 16, 128)  # 16是feature的维度
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.max_action = max_action

    def forward(self, state_, depth_image):
        feature = self.Encoder(depth_image)  # depth_image:(batch_size,1,64,64)
        state = torch.cat([feature, state_], 1)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.Encoder = Encoder()  # 卷积层

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim + 16, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim + 16, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 1)

    def forward(self, state_, action, depth_image):
        """
        Inputs:(batch_size,state_dim + action_dim)  (batch_size,1,64,64)
        Outputs:(batch_size,1)  (batch_size,1)
        """
        feature = self.Encoder(depth_image)  # depth_image:(batch_size,1,64,64)
        state = torch.cat([feature, state_], 1)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state_, action, depth_image):
        """
        Inputs:(batch_size,state_dim + action_dim)  (batch_size,1,64,64)
        Outputs:(batch_size,1)
        """
        feature = self.Encoder(depth_image)
        state = torch.cat([feature, state_], 1)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action=1,
            discount=0.99,
            tau=0.05,
            policy_noise=0.12,
            noise_clip=0.5,
            policy_freq=3
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)  # Actor转移到GPU计算
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-5)

        self.critic = Critic(state_dim, action_dim).to(device)  # Critic转移到GPU计算
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-5)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state, depth_image):
        # state是(28,)-ndarray
        # depth_image是(64,64)-ndarray
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        depth_image = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0).to(device)
        return self.actor(state, depth_image).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        # 提取touch_value和other_info拼接成state_
        state_ = torch.cat((state[:, :1], state[:, -27:]), dim=1)
        next_state_ = torch.cat((next_state[:, :1], next_state[:, -27:]), dim=1)
        # 提取range_value并转换成depth_image
        depth_image = state[:, 1:1 + 64 * 64].reshape(-1, 1, 64, 64)
        next_depth_image = next_state[:, 1:1 + 64 * 64].reshape(-1, 1, 64, 64)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                    self.actor_target(next_state_, next_depth_image) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state_, next_action, next_depth_image)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_, action, depth_image)

        # Compute loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state, depth_image), depth_image).mean()  # 策略是梯度上升更新，因此在loss前加负号
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
