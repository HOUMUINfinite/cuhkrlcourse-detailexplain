#策略网络（Actor）和价值网络（Critic

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from agents.BaseAgent import BaseAgent
from networks.networks import ActorCritic #Actor-Critic网络架构。
from utils.RolloutStorage import RolloutStorage #用于存储和处理经验数据


from timeit import default_timer as timer

class Model(BaseAgent):
    # Model 类继承自 BaseAgent，并扩展了Actor-Critic的功能。
    def __init__(self, static_policy=False, env=None, config=None):
        super(Model, self).__init__()
        self.device = config.device

        self.noisy=config.USE_NOISY_NETS
        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.learn_start = config.LEARN_START
        self.sigma_init= config.SIGMA_INIT
        self.num_agents = config.num_agents
        self.value_loss_weight = config.value_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight
        self.rollout = config.rollout
        self.grad_norm_max = config.grad_norm_max

        self.static_policy = static_policy #是否使用静态策略
        self.num_feats = env.observation_space.shape
        self.num_feats = (self.num_feats[0] * 4, *self.num_feats[1:])
        self.num_actions = env.action_space.n
        self.env = env #
        # config：配置对象，包含各种超参数。
        

        self.declare_networks()
            
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.99, eps=1e-5)
        
        #move to correct device
        self.model = self.model.to(self.device)

        if self.static_policy:
            self.model.eval()
        else:
            self.model.train()
            
        # Rollout存储：初始化RolloutStorage，用于存储和处理经验数据。
        self.rollouts = RolloutStorage(self.rollout, self.num_agents,
            self.num_feats, self.env.action_space, self.device, config.USE_GAE, config.gae_tau)

        self.value_losses = []
        self.entropy_losses = []
        self.policy_losses = []


    def declare_networks(self):
        self.model = ActorCritic(self.num_feats, self.num_actions)
        # ActorCritic：定义Actor-Critic网络，用于估计策略和价值函数。

    def get_action(self, s, deterministic=False):
        # 根据状态 s 选择动作，返回值函数、动作和动作的对数概率。
        # deterministic：是否使用确定性策略。
        logits, values = self.model(s) #通过模型计算动作的logits和状态的价值。
        dist = torch.distributions.Categorical(logits=logits) #创建一个Categorical分布。

        if deterministic:
            actions = dist.probs.argmax(dim=1, keepdim=True)
        else:
            actions = dist.sample().view(-1, 1) #actions = dist.sample()：从分布中采样动作。

        log_probs = F.log_softmax(logits, dim=1) #计算动作的对数概率。
        action_log_probs = log_probs.gather(1, actions) #获取所选动作的对数概率。

        return values, actions, action_log_probs
        

    def evaluate_actions(self, s, actions):
        # 评估给定状态和动作的值函数、动作对数概率和策略熵。
        logits, values = self.model(s)

        dist = torch.distributions.Categorical(logits=logits)

        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = dist.entropy().mean() #计算策略的熵，用于鼓励探索。

        return values, action_log_probs, dist_entropy

    def get_values(self, s):
        #仅返回给定状态的值函数
        _, values = self.model(s)

        return values

    def compute_loss(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy = self.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values #优势函数，用于衡量当前策略的好坏。
        value_loss = advantages.pow(2).mean() #值函数损失。

        action_loss = -(advantages.detach() * action_log_probs).mean() #策略损失。、

        # dist_entropy：策略的熵，用于鼓励探索

        loss = action_loss + self.value_loss_weight * value_loss - self.entropy_loss_weight * dist_entropy

        return loss, action_loss, value_loss, dist_entropy

    def update(self, rollout):
        loss, action_loss, value_loss, dist_entropy = self.compute_loss(rollout)

        self.optimizer.zero_grad()
        loss.backward() #计算梯度。
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_max) #梯度裁剪，防止梯度爆炸。
        self.optimizer.step()

        self.save_loss(loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item())
        #self.save_sigma_param_magnitudes()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def save_loss(self, loss, policy_loss, value_loss, entropy_loss): #保存每个训练步骤的损失，用于后续分析。
        super(Model, self).save_loss(loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)
