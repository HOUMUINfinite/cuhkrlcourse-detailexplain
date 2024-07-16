###
分布式Q学习是强化学习中的一种方法，它不仅估计单一的Q值（动作价值），还估计Q值的分布。
这种方法能够更好地捕捉环境中的不确定性，提供更丰富的信息，用于决策和策略优化。
具体来说，分布式Q学习的一个著名实现是 Categorical DQN，它通过计算不同回报的概率分布来改进传统的DQN。

传统的DQN通过神经网络估计每个状态-动作对的Q值，即预期回报的期望值。
然而，对于某些环境，回报可能具有很大的不确定性，仅仅估计期望值是不够的。分布式Q学习则通过估计回报的分布，提供更加详细的统计信息。
它的主要思想是将Q值的回报分布在一组离散的支持点（atoms）上，并计算每个支持点的概率
###

import numpy as np

import torch

from agents.DQN import Model as DQN_Agent
from networks.network_bodies import SimpleBody, AtariBody
from networks.networks import CategoricalDQN


class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None):
        self.atoms = config.ATOMS # 离散化的支持数量
        self.v_max = config.V_MAX # 支持的最大值
        self.v_min = config.V_MIN # 支持的最小值
        self.supports = torch.linspace(self.v_min, self.v_max, self.atoms).view(1, 1, self.atoms).to(config.device) # 离散化后的支持值。
        self.delta = (self.v_max - self.v_min) / (self.atoms - 1) # 支持的间隔

        super(Model, self).__init__(static_policy, env, config)

    def declare_networks(self):
        self.model = CategoricalDQN(self.env.observation_space.shape, self.env.action_space.n, noisy=self.noisy, sigma_init=self.sigma_init, atoms=self.atoms)
        self.target_model = CategoricalDQN(self.env.observation_space.shape, self.env.action_space.n, noisy=self.noisy, sigma_init=self.sigma_init, atoms=self.atoms)

    def projection_distribution(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        with torch.no_grad():
            max_next_dist = torch.zeros((self.batch_size, 1, self.atoms), device=self.device, dtype=torch.float) + 1. / self.atoms
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                self.target_model.sample_noise()
                max_next_dist[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
                max_next_dist = max_next_dist.squeeze()

            Tz = batch_reward.view(-1, 1) + (self.gamma**self.nsteps) * self.supports.view(1, -1) * non_final_mask.to(torch.float).view(-1, 1)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1
            
            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).unsqueeze(dim=1).expand(self.batch_size, self.atoms).to(batch_action)
            m = batch_state.new_zeros(self.batch_size, self.atoms)
            m.view(-1).index_add_(0, (l + offset).view(-1), (max_next_dist * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (max_next_dist * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        return m #投影后的分布
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.atoms)
        batch_reward = batch_reward.view(-1, 1, 1)

        # estimate
        self.model.sample_noise()
        current_dist = self.model(batch_state).gather(1, batch_action).squeeze()

        target_prob = self.projection_distribution(batch_vars)
          
        loss = -(target_prob * current_dist.log()).sum(-1) #KL散度损失，用于衡量当前分布与目标分布之间的差异
        if self.priority_replay:
            self.memory.update_priorities(indices, loss.detach().squeeze().abs().cpu().numpy().tolist())
            loss = loss * weights
        loss = loss.mean()

        return loss

    def get_action(self, s, eps): #根据状态 s 选择动作，使用分布的期望值来选择最大Q值的动作。
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy or self.noisy:
                X = torch.tensor([s], device=self.device, dtype=torch.float) 
                self.model.sample_noise()
                a = self.model(X) * self.supports
                a = a.sum(dim=2).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def get_max_next_state_action(self, next_states): #计算下一状态中Q值最大的动作的分布。
        next_dist = self.target_model(next_states) * self.supports
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.atoms)
