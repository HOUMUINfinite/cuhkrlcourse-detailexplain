# 使用了递归神经网络（RNN）来处理序列数据。这种方法适用于那些状态具有时间依赖性的环境，如部分可观察的马尔可夫决策过程（POMDP）。
import numpy as np

import torch
import torch.optim as optim

from agents.DQN import Model as DQN_Agent
from networks.networks import DRQN
from utils.ReplayMemory import RecurrentExperienceReplayMemory
from utils.hyperparameters import Config
from networks.network_bodies import AtariBody, SimpleBody

class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None):
        self.sequence_length=config.SEQUENCE_LENGTH #从配置中读取序列长度

        super(Model, self).__init__(static_policy, env, config)

        self.reset_hx() #初始化序列缓冲区
    
    def declare_networks(self):
        self.model = DRQN(self.num_feats, self.num_actions, body=SimpleBody)
        self.target_model = DRQN(self.num_feats, self.num_actions, body=SimpleBody)

    def declare_memory(self):
        self.memory = RecurrentExperienceReplayMemory(self.experience_replay_size, self.sequence_length) #DRQN需要存储和采样序列数据，因此使用递归经验回放缓冲区。

python

        #self.memory = ExperienceReplayMemory(self.experience_replay_size)

    def prep_minibatch(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (self.batch_size,self.sequence_length)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).view(self.batch_size, self.sequence_length, -1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).view(self.batch_size, self.sequence_length)
        #get set of next states for end of each sequence
        batch_next_state = tuple([batch_next_state[i] for i in range(len(batch_next_state)) if (i+1)%(self.sequence_length)==0])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).unsqueeze(dim=1)
            non_final_next_states = torch.cat([batch_state[non_final_mask, 1:, :], non_final_next_states], dim=1)
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars):
        #在训练过程中，隐藏状态需要在每个序列开始时重置，并在每个序列的时间步之间传递
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # 初始化隐藏状态
        _ = None
        
        #estimate
        current_q_values, _ = self.model(batch_state) #_表示在前向传播过程中传递隐藏状态
        current_q_values = current_q_values.gather(2, batch_action).squeeze()
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros((self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
            if not empty_next_state_values:
                max_next, _ = self.target_model(non_final_next_states) #在计算目标 Q 值时传递隐藏状态。
                max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
            expected_q_values = batch_reward + ((self.gamma**self.nsteps)*max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        loss = loss.mean()

        return loss

    def get_action(self, s, eps=0.1):
        #在动作选择过程中，隐藏状态也需要在序列之间传递，以确保动作的选择能够基于时间序列数据。
        with torch.no_grad():
            self.seq.pop(0)
            self.seq.append(s)

            # 初始化隐藏状态
            _ = None
        
            if np.random.random() >= eps or self.static_policy or self.noisy:
                X = torch.tensor([self.seq], device=self.device, dtype=torch.float) 
                self.model.sample_noise()
                a, _ = self.model(X) # 传递隐藏状态
           
                a = a[:, -1, :] #select last element of seq
                a = a.max(1)[1]
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    #def get_max_next_state_action(self, next_states, hx):
    #    max_next, _ = self.target_model(next_states, hx)
    #    return max_next.max(dim=1)[1].view(-1, 1)'''

    def reset_hx(self):
        #self.action_hx = self.model.init_hidden(1)
        self.seq = [np.zeros(self.num_feats) for j in range(self.sequence_length)]

    
