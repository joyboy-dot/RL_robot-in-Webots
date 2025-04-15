import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0  # 记录位置的指针
        self.size = 0  # 缓冲区经验数量
        # state和next_state留出64*64存储range_value
        self.state = np.zeros((max_size, state_dim + 64 * 64))  # state_dim + 64x64
        self.next_state = np.zeros((max_size, state_dim + 64 * 64))  # next_state_dim + 64x64

        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        """
        state: np.ndarray 或者类似的可迭代对象（例如列表或元组）
        action: np.ndarray 或者类似的可迭代对象（例如列表或元组）
        next_state: np.ndarray 或者类似的可迭代对象（例如列表或元组）
        reward: float值
        done: bool(0/1)
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
