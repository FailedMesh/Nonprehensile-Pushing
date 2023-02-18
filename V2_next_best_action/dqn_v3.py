'''Complete DQN 3
Consumes the current state, object size, and the next state as input, and spits out Q values for each of the 16 standard actions, 
    for each of the two actions types
    1. Action type 1: Maximize Position Change
    2. Action type 2: Maximize Orientation Change
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable

ACTION_DESCRIPTION = {
    'POSMAX': "Set of 16 actions, where each corresponds to pushing along a particular direction. Here, each \
                we choose the POC (push point of contact) that maximizes the change in position",
    'ORNMAX': "Set of 16 actions, where each corresponds to pushing along a particular direction. Here, each \
                we choose the POC (push point of contact) that maximizes the change in orientation",            
}

POC_DICT = {
    'POSMAX': 0,
    'ORNMAX': 1
}

ANGLE_DICT = { }
for i in range(4):
    ANGLE_DICT[str(i*4)] = (np.pi/2)*i + 0
    ANGLE_DICT[str(1+i*4)] = (np.pi/2)*i + np.pi/6
    ANGLE_DICT[str(2+i*4)] = (np.pi/2)*i + np.pi/4
    ANGLE_DICT[str(3+i*4)] = (np.pi/2)*i + np.pi/3

class pushDQN3(nn.Module):
    '''Input Output Description
    Input Space: (x1, y1, theta1, x2, y2, theta2, length, breadth)

    Output: 0-31 (Q values for push actions (pushing in the 16 standard directions for each of the push type))
    '''
    def __init__(self, n_observations=8, n_actions=32, use_cuda=True) -> None:
        super(pushDQN3, self).__init__()
        self.use_cuda = use_cuda

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
        
    def forward(self, x, is_volatile=False):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        output_probs = self.layer3(x)
    
        return output_probs
