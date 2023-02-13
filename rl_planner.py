'''Complete Information DQN (CIdqn)
'''

import glob
import imp
import math
import gc
import os
from sre_constants import SUCCESS
import time
import datetime
import pybullet as p
import cv2
import numpy as np
from graphviz import Digraph
import argparse
import random
import torch
import matplotlib.pyplot as plt
from time import sleep
import copy

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from Environments.utils import sample_goal, get_pose_distance


from Config.constants import (
    GRIPPER_PUSH_RADIUS,
    PIXEL_SIZE,
    PUSH_DISTANCE,
    WORKSPACE_LIMITS,
    TARGET_LOWER,
    TARGET_UPPER,
    orange_lower,
    orange_upper,
    BG_THRESHOLD,
    MIN_GRASP_THRESHOLDS
)

from Environments.environment_sim2 import Environment
import Environments.utils as env_utils
from V1_destination_prediction.Test_cases.tc3_no_bottom import TestCase1

from create_env import get_push_start, get_max_extent_of_target_from_bottom, get_push_start_POSMAX


from collections import namedtuple, deque

from V2_next_best_action.models.dqn_v2 import pushDQN2

torch.cuda.empty_cache()

def select_action(state):
    '''Select the next best action 
    state: tensor(shape=(6))
    '''
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)

def get_reward(prev_state, current_state):
    '''
    prev_state: (x1, y1, theta1, x2, y2, theta2)
    current_state: (x3, y3, theta3, _, _, _)
    '''
    reward = np.exp(-1 * np.sqrt(np.linalg.norm(current_state[0:3] - prev_state[3:6])))
    return reward

def get_reward2(prev_state, current_state):
    '''
    prev_state: (x1, y1, theta1, x2, y2, theta2)
    current_state: (x3, y3, theta3, _, _, _)
    '''

    print("Rewarding ----------------------------")
    pos_diff = np.linalg.norm(prev_state[0:2] - prev_state[3:5]) - np.linalg.norm(current_state[0:2] - prev_state[3:5])
    orn_diff = np.linalg.norm(prev_state[2:3] - prev_state[5:6]) - np.linalg.norm(current_state[2:3] - prev_state[5:6])
    reward = pos_diff + 0.1*orn_diff  #np.linalg.norm(prev_state[0:3] - prev_state[3:6]) - np.linalg.norm(current_state[0:3] - prev_state[3:6]) # prev distance - current distance
    print(f"Position Diff: {pos_diff}\tOrn Diff: {0.1*orn_diff}\nReward Aggregate: {reward}")
    return reward

def get_reward4(prev_state, current_state):
    '''
    prev_state: (x1, y1, theta1, x2, y2, theta2)
    current_state: (x3, y3, theta3, _, _, _)
    '''

    # print("Rewarding ----------------------------")
    pos_diff = np.linalg.norm(prev_state[0:2] - prev_state[3:5]) - np.linalg.norm(current_state[0:2] - prev_state[3:5])
    orn_diff = np.linalg.norm(prev_state[2:3] - prev_state[5:6]) - np.linalg.norm(current_state[2:3] - prev_state[5:6])
    reward = pos_diff + 0.1*orn_diff  #np.linalg.norm(prev_state[0:3] - prev_state[3:6]) - np.linalg.norm(current_state[0:3] - prev_state[3:6]) # prev distance - current distance
    # print(f"Position Diff: {pos_diff}\tOrn Diff: {0.1*orn_diff}\nReward Aggregate: {reward}")
    return reward

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_observations = 6 # 3 for initial state, 3 for goal state
n_actions = 16 # 16 push + 1 grasp

policy_net = pushDQN2(n_observations, n_actions, use_cuda=True).to(device)

checkpoint = torch.load('./V2_next_best_action/models/model_checkpoints/dqnv2/model7/5350.pt',map_location=torch.device('cpu'))
policy_net.load_state_dict(checkpoint)
policy_net.eval()
MAX_PUSH_ITER = 100 # 20

push_directions = [0, np.pi/8, np.pi/4, 3*np.pi/8, 
                    np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, 
                    np.pi, 9*np.pi/8, 5*np.pi/4, 11*np.pi/8,  
                    3*np.pi/2, 13*np.pi/8, 7*np.pi/4, 15*np.pi/8] # 16 standard directions

# max_timesteps = 250
# max_per_episode_timesteps = 50
# timestep = 0

def compute_l2_loss(joint_pose_list):
    if len(joint_pose_list) <= 1:
        return 0
    print(f"Joint pose list shape: {joint_pose_list.shape}")
    diffs = np.diff(joint_pose_list,axis=0)
    diff_sq = np.square(diffs)
    norm = np.sum(np.sqrt(np.sum(diff_sq, axis=1)), axis=0)
    return norm



def get_loss(sample_trajectory):
    env = Environment(gui = False)
    env.reset()
    threshold_d = 0.05 # Threshold pose distance
    testcase1 = TestCase1(env)
    start_pt = sample_trajectory[0]
    body_ids, sucess = testcase1.create_specific_test_case(targetPos=start_pt)
    targetPos, targetOrn = p.getBasePositionAndOrientation(body_ids[0])
    marker_poses = []
    marker_yaws = []
    joint_cost = 0

    for i in range(len(sample_trajectory)-1):
        marker_pos = copy.deepcopy(np.array(targetPos))
        marker_pos[0] = sample_trajectory[i+1][0]
        marker_pos[1] = sample_trajectory[i+1][1]
        marker_poses.append(marker_pos)
        marker_yaw = np.random.uniform(low=0, high=np.pi)
        marker_yaws.append(marker_yaw)


    for i in range(len(sample_trajectory)-1):
        dist_pos = 100
        marker_pos = marker_poses[i] # sample_trajectory[i+1]
        targetPos, targetOrn = p.getBasePositionAndOrientation(body_ids[0])
        dist_pos = np.linalg.norm(marker_pos[0:2] - np.array(targetPos)[0:2])
        
        marker_orn = p.getQuaternionFromEuler([0, 0, marker_yaw])
        marker_obj, goal_suc = testcase1.add_marker_obj(marker_pos, marker_orn, half_extents=testcase1.current_target_size/2)
        body_ids.append(marker_obj)

        PUSH_ITER = 0

        # if not(marker_pos[0] > WORKSPACE_LIMITS[0][0] and marker_pos[0] < WORKSPACE_LIMITS[0][1] and
        #         marker_pos[1] > WORKSPACE_LIMITS[1][0] and marker_pos[1] < WORKSPACE_LIMITS[1][1]):
        #         print ("Bad")
        #         return 1000

        ob1_start_pos, ob1_start_orn = p.getBasePositionAndOrientation(body_ids[-2])
        ob2_start_pos, ob2_start_orn = p.getBasePositionAndOrientation(body_ids[-1])

        ob1_start_pos = np.array(ob1_start_pos)
        ob1_start_orn = np.array(ob1_start_orn)
        ob2_start_pos = np.array(ob2_start_pos)
        ob2_start_orn = np.array(ob2_start_orn)
        
        while dist_pos > threshold_d and PUSH_ITER < MAX_PUSH_ITER:
            targetPos, targetOrn = p.getBasePositionAndOrientation(body_ids[0])
            # print ("TargetPos:",targetPos)
            # print ("WS Limits:",WORKSPACE_LIMITS)
            # if not(targetPos[0] > WORKSPACE_LIMITS[0][0] + 0.05 and targetPos[0] < WORKSPACE_LIMITS[0][1] - 0.05 and
            #     targetPos[1] > WORKSPACE_LIMITS[1][0] + 0.05 and targetPos[1] < WORKSPACE_LIMITS[1][1] - 0.05):
            # if not(targetPos[0] > WORKSPACE_LIMITS[0][0] and targetPos[0] < WORKSPACE_LIMITS[0][1] and
            #     targetPos[1] > WORKSPACE_LIMITS[1][0] and targetPos[1] < WORKSPACE_LIMITS[1][1]):
            #     print ("Bad")
            #     return 1000

            target_euler = p.getEulerFromQuaternion(targetOrn)
            marker_yaw = marker_yaws[i]

            cur_target_st = np.array([targetPos[0], targetPos[1], target_euler[2]], dtype=np.float64)
            cur_target_goal = np.array([marker_pos[0], marker_pos[1], marker_yaw], dtype=np.float64)
            cur_state = np.hstack((cur_target_st, cur_target_goal))
            state = {
                'cur_state': torch.tensor(cur_state, dtype=torch.float, device=device).unsqueeze(0),
            }
            action = select_action(state['cur_state'])
            if action.item() in range(0, 16):
                bottomPos, bottomOrn = [], [] # p.getBasePositionAndOrientation(body_ids[0])
                targetPos, targetOrn = p.getBasePositionAndOrientation(body_ids[0])
                current_target_obj_size = testcase1.current_target_size

                push_dir = push_directions[(action.item())%16] # Sample push directions
                # push_type = 0 # 0 indicates position maximization 
                # if action.item() >= 16:
                    # push_type = 1 # 1 indicates orientation maximization
                push_start, push_end = get_push_start_POSMAX(push_dir, bottomPos, bottomOrn,
                                                targetPos, targetOrn, current_target_obj_size, is_viz=False)
                # push_start, push_end = 
                # push_start, push_end = get_push_start(push_dir, target_mask, body_ids[1])
                _, all_joints = env.push(push_start, push_end)

                # collect reward
                new_target_pos, new_target_orn = p.getBasePositionAndOrientation(body_ids[0])
                target_euler = p.getEulerFromQuaternion(new_target_orn)

                new_target_st = np.array([new_target_pos[0], new_target_pos[1], target_euler[2]], dtype=float)
                # new_target_goal = new_target_st + np.random.uniform(low=[-5, -5, -2*np.pi], high=[5, 5, 2*np.pi], size=(3,))
                new_state = np.hstack((new_target_st, cur_target_goal))
                next_state = {
                    'cur_state': torch.tensor(new_state, dtype=torch.float, device=device).unsqueeze(0)
                }
                state=next_state
                # reward = get_reward2(current_state=new_state, prev_state=state['cur_state'].squeeze().cpu().numpy())
                # print(f"Reward: {reward}")

                dist_pos = np.linalg.norm(cur_target_goal[0:2] - new_target_st[0:2])

                # print(f'Goal: {i}\tDist pos: {dist_pos}')
                wp_to_wp_loss = compute_l2_loss(all_joints)
                joint_cost += wp_to_wp_loss
                #print ("Joint poses:", all_joints)
            PUSH_ITER += 1

        # ------------ Collision Cost -------------- #

        ob1_final_pos, ob1_final_orn = p.getBasePositionAndOrientation(body_ids[-2])
        ob2_final_pos, ob2_final_orn = p.getBasePositionAndOrientation(body_ids[-1])

        ob1_final_pos = np.array(ob1_final_pos)
        ob1_final_orn = np.array(ob1_final_orn)
        ob2_final_pos = np.array(ob2_final_pos)
        ob2_final_orn = np.array(ob2_final_orn)

        ob1_pos_diff = ob1_final_pos - ob1_start_pos
        ob2_pos_diff = ob2_final_pos - ob2_start_pos
        ob1_orn_diff = ob1_final_orn - ob1_start_orn
        ob2_orn_diff = ob2_final_orn - ob2_start_orn

        obstacle_movement = np.concatenate((ob1_pos_diff, ob1_orn_diff, ob2_pos_diff, ob2_orn_diff))
        collision_cost = 1000 * np.linalg.norm(obstacle_movement)

        # -------------------------------------------- #

        tot_loss = joint_cost + collision_cost

    return tot_loss


def vpsto_wrapper(candidates):
    sample_trajectory = candidates['pos']
    print (np.array(sample_trajectory).shape)
    costs = []
    for traj in sample_trajectory:
        print ("New traj:",traj)
        for i in range(len(traj)):
            traj[i] = np.array(traj[i])
        cost_traj = get_loss(traj)
        costs.append(cost_traj)
    return costs



# sample_trajectory = []
# dist_interval = 0.05
# start_pt = np.array([0.35, -0.05])
# sample_trajectory.append(start_pt)
# current_pt = copy.deepcopy(start_pt)
# for i in range(3):
#     print(i)
#     yaw = np.random.uniform(low=-np.pi/2, high=np.pi/2)
#     next_pt = None
#     succ = False
#     while not succ:
#         next_pt = current_pt + np.array([np.cos(yaw), np.sin(yaw)])*dist_interval
#         if next_pt[0] > WORKSPACE_LIMITS[0][0] and next_pt[0] < WORKSPACE_LIMITS[0][1] \
#             and next_pt[1] > WORKSPACE_LIMITS[1][0] and next_pt[1] < WORKSPACE_LIMITS[1][1]:
#             succ = True
#     sample_trajectory.append(next_pt)
#     current_pt = copy.deepcopy(next_pt)

# print(sample_trajectory)


