import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from collections import namedtuple, deque
import numpy as np
import random
import math
from PIL import Image

IMAGE_SIZE = 256
NUM_ACTIONS = 5
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        x = x / 255.0  # Normalize the input to [0, 1]
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(x.size()[0], -1)
        return self.fc_layers(conv_out)

def process_image(video_frame):
    transform1 = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = np.reshape(video_frame.pixels, (IMAGE_SIZE, IMAGE_SIZE, 3))
    image_tensor = np.mean(image_tensor, axis=2)
    image_tensor = np.where(image_tensor < 128, 0, 256)
    image_tensor = transform1(image_tensor).reshape((1,1,IMAGE_SIZE,IMAGE_SIZE)).float()
    return image_tensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DropperAgent:
    def __init__(self, agent_host):
        self.agent_host = agent_host
        self.action_list = ["movenorth","movesouth", "moveeast", "movewest", "move 0"] #forward, backward, left, right, standstill
    def setAction(self, action):
        if action != 4: # If action requested isn't to stand
            self.agent_host.sendCommand(self.action_list[action])

def get_reward(waterX, waterZ, playerX, playerZ, action):
    reward = (playerX - waterX)**2 + (playerZ - waterZ)**2
    if math.sqrt(reward) < 0.8:
        if action == 4:
            return 200
        else:
            return 100
    return -reward

def generate_water(size):
    waterSize = size
    waterX = random.randrange(-8, 8)
    waterZ = random.randrange(-8, 8)
    waterX1 = math.floor(waterX - waterSize/2)
    waterX2 = math.floor(waterX + waterSize/2)
    waterZ1 = math.floor(waterZ - waterSize/2)
    waterZ2 = math.floor(waterZ + waterSize/2)
    return waterX1, waterX2, waterZ1, waterZ2

steps_done = 0
def select_action(state, policy_net, device):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(NUM_ACTIONS)]], device=device, dtype=torch.long)

def optimize_model(policy_net, target_net, optimizer, device, memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()