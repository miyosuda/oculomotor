import os
import numpy as np
import brica

from .fef import GRID_DIVISION
from .pfc import Task, Phase, InternalPhase, TARGETS

from oculoenv.environment import CAMERA_VERTICAL_ANGLE_MAX
from oculoenv.environment import CAMERA_HORIZONTAL_ANGLE_MAX

### learning
from itertools import count
from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torchvision
import torchvision.transforms as transforms
###

ACTION_PHASE_SIZE = len([Phase.AVOID, Phase.WANDER, Phase.EXPLORE])
ACTION_TARGET_SIZE = len(TARGETS)
ACTION_COEFF_SIZE = 4 + 1 # template_coeff, direction_coeff, change_coeff, search_coeff and +1 # see _coeff_num_to_list


"""
This is an example implemention of BG (Basal ganglia) module.
You can change this as you like.
"""

class BG(object):
    def __init__(self):
        self.timing = brica.Timing(5, 1, 0)

        self.bgrl = BGRL()
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()
        self.log_prob = None
        self.value = None

        #for i in range(72):
        #    phase, target, coeff = self._action_num_to_list(i)
        #    self._action_list_to_num([phase, target, coeff])


    def __call__(self, inputs):
        if 'from_environment' not in inputs:
            raise Exception('BG did not recieve from Environment')
        if 'from_pfc' not in inputs:
            raise Exception('BG did not recieve from PFC')
        if 'from_fef' not in inputs:
            raise Exception('BG did not recieve from FEF')

        reward, done = inputs['from_environment']
        task, phase, internal_phase, target = inputs['from_pfc']
        fef_data = inputs['from_fef']
        assert fef_data.shape == (256, 3)

        # convert (256, 3) to (8, 8), ..., (8, 8)
        template_likelihood, direction_likelihood, change_likelihood, search_likelihood = self._fef_data_to_likelihood_matrix(fef_data)

        # step (training)
        direction_index = self._direction_likelihood_to_index(direction_likelihood)
        obs = np.array([[task, phase, internal_phase, target, direction_index], np.max(template_likelihood), np.max(change_likelihood)])
        action_num = self._step_rule(obs, reward, done)
        #action_num = self._step_rl(obs, reward, done)

        next_phase, next_target, coeff = self._action_num_to_list(action_num)
        template_coeff, direction_coeff, change_coeff, search_coeff = self._coeff_num_to_list(coeff)

        template_thresholds = np.ones_like(template_likelihood) * template_coeff
        direction_thresholds = np.ones_like(direction_likelihood) * direction_coeff
        change_thresholds = np.ones_like(change_likelihood) * change_coeff
        search_thresholds = np.ones_like(search_likelihood) * search_coeff

        # convert (8, 8), ..., (8, 8) to (256,)
        likelihood_thresholds = self._likelihood_matrix_to_sc_data(template_thresholds, direction_thresholds, change_thresholds, search_thresholds)

        return dict(to_pfc=(reward, done, next_phase, next_target),
                    to_fef=None,
                    to_sc=likelihood_thresholds)

    def save_model(self, f):
        self.bgrl.save_model(f)

    def load_model(self, f):
        self.bgrl.load_model(f)

    def _step_rl(self, obs, reward, done):
        [task, phase, internal_phase, target, direction_index], max_template_likelihood, max_change_likelihood = obs

        if self.log_prob is not None and self.value is not None:
            self.bgrl.append_memory(self.log_prob, self.value, reward)
        if done:
            print('done', len(self.bgrl.memory))
            self.bgrl.train_model(self.gamma, self.eps)
        obs = np.array([task, phase, internal_phase, target, direction_index, max_template_likelihood, max_change_likelihood])
        action, log_prob, value = self.bgrl.get_action(self.bgrl.numpy_to_tensor(obs))
        self.log_prob = log_prob
        self.value = value
        return action.item()

    def _step_rule(self, obs, reward, done):
        [task, phase, internal_phase, target, direction_index], max_template_likelihood, max_change_likelihood = obs

        next_phase = phase
        next_target = target
        coeff = 0 # target

        if phase == Phase.AVOID:
            next_target = 0 # center
        elif phase == Phase.WANDER:
            if task == Task.POINT_TO_TARGET:
                coeff = 4 # search
                if max_template_likelihood > 0.2:
                    next_phase = Phase.EXPLORE
            elif task == Task.ODD_ONE_OUT:
                if internal_phase == InternalPhase.EVALUATION:
                    coeff = 4 # search
            elif task == Task.VISUAL_SEARCH:
                coeff = 4 # search
                if max_template_likelihood > 0.2:
                    next_phase = Phase.EXPLORE
                    next_target = 1 # bottom YES button
                elif internal_phase == InternalPhase.EVALUATION:
                    next_phase = Phase.EXPLORE
                    next_target = 2 # bottom NO button
            elif task == Task.RANDOM_DOT_MOTION_DISCRIMINATION:
                if internal_phase == InternalPhase.EVALUATION:
                    next_phase = Phase.EXPLORE
                    next_target = 5 + direction_index # direction target
            elif task == Task.CHANGE_DETECTION:
                if internal_phase == InternalPhase.EVALUATION:
                    next_phase = Phase.EXPLORE
                    if max_change_likelihood > 0.5:
                        next_target = 3 # CD YES button
                    else:
                        next_target = 4 # CD NO button
            elif task == Task.MULTIPLE_OBJECT_TRACKING:
                if internal_phase == InternalPhase.LEARNING:
                    coeff = 1 # template
                elif internal_phase == InternalPhase.INTERVAL:
                    coeff = 1 # template
                elif internal_phase == InternalPhase.EVALUATION:
                    next_phase = Phase.EXPLORE
                    if max_template_likelihood > 0.2:
                        next_target = 1 # bottom YES button
                    else:
                        next_target = 2 # bottom NO button
            else:
                assert task in (Task.POINT_TO_TARGET, Task.CHANGE_DETECTION, Task.ODD_ONE_OUT, Task.VISUAL_SEARCH, Task.MULTIPLE_OBJECT_TRACKING, Task.RANDOM_DOT_MOTION_DISCRIMINATION)
        elif phase == Phase.EXPLORE:
            if task == Task.POINT_TO_TARGET:
                coeff = 1 # template
                if max_template_likelihood < 0.01:
                    next_phase = Phase.AVOID
            elif task == Task.VISUAL_SEARCH:
                if max_template_likelihood > 0.2:
                    next_target = 1 # bottom YES button
            else:
                pass
        else:
            assert phase in (Phase.AVOID, Phase.WANDER, Phase.EXPLORE)

        return self._action_list_to_num(np.array([next_phase, next_target, coeff]))

    def _action_num_to_list(self, action_num):
        phase, target, coeff = np.unravel_index(action_num, (ACTION_PHASE_SIZE, ACTION_TARGET_SIZE, ACTION_COEFF_SIZE))
        return phase+1, target, coeff # phase+1 because Phase.AVOID == 1, not zero

    def _action_list_to_num(self, action_list):
        max_action = ACTION_PHASE_SIZE * ACTION_TARGET_SIZE * ACTION_COEFF_SIZE
        next_phase, next_target, coeff = action_list
        action_nums = np.array([ i for i in range(max_action) ]).reshape(ACTION_PHASE_SIZE, ACTION_TARGET_SIZE, ACTION_COEFF_SIZE)
        return action_nums[next_phase-1,  next_target, coeff] # next_phase-1 because Phase.AVOID == 1, not zero

    def _coeff_num_to_list(self, coeff_num):
        template_coeff, direction_coeff, change_coeff, search_coeff = 1.0, 1.0, 1.0, 1.0
        if coeff_num == 0:
            pass
        if coeff_num == 1:
            template_coeff = 0.1
        elif coeff_num == 2:
            direction_coeff = 0.1
        elif coeff_num == 3:
            change_coeff = 0.1
        elif coeff_num == 4:
            search_coeff = 0.1
        else:
            assert coeff_num in (0, 1, 2, 3, 4)
        return template_coeff, direction_coeff, change_coeff, search_coeff

    def _fef_data_to_likelihood_matrix(self, fef_data): # convert (256, 3) to (8, 8), ..., (8, 8)
        g_g = GRID_DIVISION * GRID_DIVISION # 64
        template_accumulator = fef_data[g_g*0:g_g*1]
        direction_accumulator = fef_data[g_g*1:g_g*2]
        change_accumulator = fef_data[g_g*2:g_g*3]
        search_accumulator = fef_data[g_g*3:g_g*4]
        template_likelihood = template_accumulator[:, 0].reshape(GRID_DIVISION, GRID_DIVISION).T # (8, 8)
        direction_likelihood = direction_accumulator[:, 0].reshape(GRID_DIVISION, GRID_DIVISION).T # (8, 8)
        change_likelihood = change_accumulator[:, 0].reshape(GRID_DIVISION, GRID_DIVISION).T # (8, 8)
        search_likelihood = search_accumulator[:, 0].reshape(GRID_DIVISION, GRID_DIVISION).T # (8, 8)
        return np.array([template_likelihood, direction_likelihood, change_likelihood, search_likelihood])

    def _likelihood_matrix_to_sc_data(self, template_thresholds, direction_thresholds, change_thresholds, search_thresholds): # convert (8, 8), ..., (8, 8) to (256,)
        g_g = GRID_DIVISION * GRID_DIVISION # 64
        # TODO: np.hstack
        likelihood_thresholds = np.array([])
        likelihood_thresholds = np.append(likelihood_thresholds, template_thresholds.T.reshape(g_g))
        likelihood_thresholds = np.append(likelihood_thresholds, direction_thresholds.T.reshape(g_g))
        likelihood_thresholds = np.append(likelihood_thresholds, change_thresholds.T.reshape(g_g))
        likelihood_thresholds = np.append(likelihood_thresholds, search_thresholds.T.reshape(g_g))
        return likelihood_thresholds

    def _direction_likelihood_to_index(self, direction_likelihood):
        w, h = direction_likelihood.shape
        x, y = np.unravel_index(direction_likelihood.argmax(), direction_likelihood.shape) # max index
        theta = np.arctan2(y - 0.5 * h, x - 0.5 * w) # [-pi, pi]
        theta = theta + 0.125 * np.pi # theta + pi / 8
        theta = (theta + 2 * np.pi) % (2 * np.pi) # [0, 2 * pi]
        index = np.digitize(theta, bins=np.linspace(0, 2 * np.pi, 8 + 1)[1:-1])
        return np.reshape(index, (1))[0] # TODO

### learning start ###
class BGNet(nn.Module):
    def __init__(self):
        super(BGNet, self).__init__()
        self.affine1 = nn.Linear(7, 128)
        self.action_head = nn.Linear(128, ACTION_PHASE_SIZE * ACTION_TARGET_SIZE * ACTION_COEFF_SIZE)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

class BGRL(object):
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #print(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.net = BGNet()
        self.net.to(self.device)
        #self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08) # default parameters
        self.memory = []

    def load_model(self, file_path):
        module_dir, _ = os.path.split(os.path.realpath(__file__))
        absolute_path = os.path.join(module_dir, file_path)
        return self.net.load_state_dict(torch.load(absolute_path, map_location=self.device))

    def save_model(self, file_path):
        module_dir, _ = os.path.split(os.path.realpath(__file__))
        absolute_path = os.path.join(module_dir, file_path)
        return torch.save(self.net.state_dict(), absolute_path)

    def step(self, state):
        return self.net(state)

    def get_action(self, state):
        policy, value = self.step(state)
        action, log_prob = self.policy_to_action(policy)
        return action, log_prob, value

    def policy_to_action(self, policy):
        p = Categorical(policy)
        action = p.sample()
        log_prob = p.log_prob(action)
        return action, log_prob
        
    def numpy_to_tensor(self, state):
        states = torch.from_numpy(state).float()
        states = states.to(self.device)
        return states

    def append_memory(self, log_prob, value, reward):
        self.memory.append([log_prob, value, reward])

    def clear_memory(self):
        del self.memory[:]
        # self.memory = []

    def get_returns(self, rewards, gamma, eps):
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns

    def train_model(self, gamma, eps):
        policy_losses = []
        value_losses = []
        mem = np.array(self.memory)
        mem[:,2] = self.get_returns(mem[:,2], gamma, eps) # mem[:,2] == rewards
        for log_prob, value, r in mem:
            policy_losses.append(-log_prob * (r - value))
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()
        self.clear_memory()
### learning end ###
