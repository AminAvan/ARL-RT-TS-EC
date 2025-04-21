from Lib.platform import system
from edge_sim_py import *
import os
import networkx as nx
import msgpack
import itertools
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import requests
import ast
import time
import psutil
from functools import wraps
# from pyrapl import measurement
import subprocess

### rl down ###
import gymnasium as gym
from builtins import map
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
import dill
import multiprocessing
import multiprocessing.reduction as reduction
reduction.ForkingPickler = dill.Pickler

from collections import namedtuple, deque
import random, math, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F
from itertools import count

import csv
import sys
from datetime import datetime

import builtins
sys.modules['__builtin__'] = builtins

## [for server which poses GPU
import GPUtil
import nvidia_smi
import psutil, os
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown
## for server which poses GPU]

from typing import List, Tuple, Any

####################################################
# Generate the filename with the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
filename = f"Results_{current_time}.csv"
file = open(filename, 'w')  # Open the file
file.write("Results:\n")  # Header
#####################################################
#####################################################

class ResourceTracker:
    def __init__(self):
        self.total_memory = 0
        self.total_power = 0
        self.call_count = 0
        self.start_time = time.time()

    def update(self, memory):
        self.total_memory += memory
        self.call_count += 1

        # Define your CPU's idle and max power consumption (in watts)
        P_idle = 5  # Example idle power in watts (adjust based on your CPU)
        P_max = 98  # Example max power in watts at full load (adjust based on your CPU)

        # # Estimate power consumption based on CPU usage
        # cpu_percent = psutil.cpu_percent(interval=0.1)
        # # Apply the CPU power consumption formula
        # power_estimate = P_idle + (P_max - P_idle) * (cpu_percent / 100)
        # # Add the power estimate to the total power consumption
        # self.total_power += power_estimate

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                ### Values for NVIDIA GeForce GTX 1070
                P_idle = 10  # Idle power for GTX 1070 (in watts)
                P_max = 150  # Maximum power for GTX 1070 (in watts)


                info_gpu = subprocess.run(
                  ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                  stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE,
                  text=True
                )
                gpu_utilization_percentage = float(info_gpu.stdout.strip())
                # Apply the GPU power consumption formula (similar to CPU)
                power_estimate = P_idle + (P_max - P_idle) * (gpu_utilization_percentage / 100)
                self.total_power += power_estimate
            else:
                raise ValueError("No GPU found.")
        except (ImportError, ValueError):
            # Fallback to CPU utilization if GPU is not available
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # Apply the CPU power consumption formula
            power_estimate = P_idle + (P_max - P_idle) * (cpu_percent / 100)
            self.total_power += power_estimate

    def report(self):
        total_time = time.time() - self.start_time
        print(f"runtime: {total_time:.2f} seconds")
        file.write(f"runtime: {total_time:.2f} seconds\n")
        print(f"memory consumption: {self.total_memory / (1024 * 1024):.2f} MB until {total_time:.2f} seconds")
        file.write(f"memory consumption: {self.total_memory / (1024 * 1024):.2f} MB until {total_time:.2f} seconds\n")
        print(f"power consumption: {self.total_power:.2f} Watt-seconds until {total_time:.2f} seconds")
        file.write(f"power consumption: {self.total_power:.2f} Watt-seconds until {total_time:.2f} seconds\n")


    def final_report(self):
        print(f"Total memory consumption: {self.total_memory / (1024 * 1024):.2f} MB")
        file.write(f"Total memory consumption: {self.total_memory / (1024 * 1024):.2f} MB\n")
        print(f"Total power consumption: {self.total_power:.2f} Watt-seconds")
        file.write(f"Total power consumption: {self.total_power:.2f} Watt-seconds\n")

# resource_tracker = ResourceTracker() ## was for normal use rather 31 runs


#################################################################
# Adjust display settings to show all columns and increase width
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# Set the maximum number of rows to display
pd.set_option('display.max_rows', 100000)  # Set this to the desired number, or None for unlimited rows
# Adjust the print options to display the full array
np.set_printoptions(threshold=np.inf)
#################################################################


# The function "has_capacity_to_host_only_resources(self, service)" is a modified version of the original
# "has_capacity_to_host" function in edge_sim_py/components/edge_server.py. Unlike "has_capacity_to_host",
# which only considers the number of cores and assumes a service requires a specific number of cores,
# "has_capacity_to_host_only_resources(self, service)" function also accounts for the clock frequency of the cores.
# This modification provides a more fair way of comparison of the original method in EdgeSimPy with our new method
# of investigating the proper edge server for the edge user application.
def has_capacity_to_host_only_resources(self, service):
    # Calculating the additional disk demand that would be incurred to the edge server
    additional_disk_demand = self._get_disk_demand_delta(service=service)

    # Calculating the edge server's free resources
    free_processing_power = self.processing_power - self.processing_power_demand
    free_memory = self.memory - self.memory_demand
    free_disk = self.disk - self.disk_demand

    # Checking if the host would have resources to host
    if (free_processing_power >= service.processing_power_demand and free_memory >= service.memory_demand and free_disk >= additional_disk_demand):
        EdgeServer.is_potential_host = EdgeServer.is_potential_host + 1
        self.execution_time_of_service[str(service.id)] = (service.processing_power_demand / self.processing_power)
        can_host = True
    else:
        can_host = False

    return can_host


############################
## BestFit implementation ##
def Best_Fit_Service_Provisioning(parameters):
    # Override 'has_capacity_to_host'
    EdgeServer.has_capacity_to_host = has_capacity_to_host_only_resources

    # We can always call the 'all()' method to get a list with all created instances of a given class
    for service in Service.all():
        # We don't want to migrate services are already being migrated
        if service.server == None and not service.being_provisioned:

            edge_servers = sorted(
            EdgeServer.all(),
            key=lambda s: ((s.processing_power - s.processing_power_demand) * (s.memory - s.memory_demand) * (s.disk - s.disk_demand)) ** (1 / 3),
            reverse=False,
            )

            # Let's iterate over the list of edge servers to find a suitable host for our service
            for edge_server in edge_servers:

                # We must check if the edge server has enough resources to host the service
                if edge_server.has_capacity_to_host(service=service):
                    # Start provisioning the service in the edge server
                    service.provision(target_server=edge_server)
                    # After start migrating the service we can move on to the next service
                    break


# Our proposed mechanism to check properly with considering multi-attributes in a proper way to realize if the candidate edge server poses enough resources (including: processor, memory, locak disk, communication latency) for the services of applications of edge users.
# The total amount of 'processing power demand' of a service is calculated by 'processor cycles per megabyte * required memory size'.
# Then, we calculate the 'exection time' of the service by 'processing_power_demand / free_processing_power', where the 'free_processing_power' is the total processing power of edge server that is obtained by 'self.cpu_cycle * self.cpu' (where self.cpu is the number of core of each edge server)
# Then, by having the 'exection time' of the service on this specific edge server, we calculate the service_utilization based on 'exection time / deadline'
# now we have the 'service utilization' now we check if the processor of the edge server has enough space for the service, simultanesouly the memory demand of the service and the disk demand of the service would be check to see if the edge server has enough space
# after all, if the edge server has enough capacity regarding processor, memory, and disk; from this function
def has_capacity_to_host_proposed(self, service: object) -> bool:
    # Calculating the additional disk demand that would be incurred to the edge server
    additional_disk_demand = self._get_disk_demand_delta(service=service)

    # Calculating the edge server's free resources
    free_memory = self.memory - self.memory_demand
    free_disk = self.disk - self.disk_demand
    free_processing_power = self.processing_power
    

    user_service_deadline = next(iter(service.application.users[0].delay_slas.values()))
    user_service_exe_time = (service.processing_power_demand / free_processing_power)
    user_service_utilization = (user_service_exe_time / user_service_deadline)
    free_cpu_utilization = self.total_cpu_utilization + user_service_utilization

    # Checking if the host would have resources to host the registry and its (additional) layers
    if (free_cpu_utilization <= 1 and free_memory >= service.memory_demand and free_disk >= additional_disk_demand):
        # calculating true runtime of service on the host server
        self.execution_time_of_service[str(service.id)] = user_service_exe_time
        self.total_cpu_utilization = (self.total_cpu_utilization + user_service_utilization)
        self.total_memory_utilization = (self.memory_demand + service.memory_demand) / self.memory

        EdgeServer.is_potential_host = EdgeServer.is_potential_host + 1
        if(len(service.all()) == EdgeServer.is_potential_host):
            s_total_cpu_util = 0
            s_total_mem_util = 0
            for s in EdgeServer.all():
                s_total_cpu_util += round(s.total_cpu_utilization, 2)
                s_total_mem_util += round((s.memory_demand/s.memory), 2)


        can_host = True
    else:
        can_host = False
    return can_host


##################################################
## Earliest Deadline First (EDF) implementation ##
def EDF_algorithm(parameters):
    # Override 'has_capacity_to_host' for all instances of the EdgeServer class
    EdgeServer.has_capacity_to_host = has_capacity_to_host_only_resources  ## this for baseline

    priorities_list = []
    for usr in User.all():
        # Calculating the urgency of each user's deadline
        priority = 1 / list(usr.delay_slas.values())[0]

        # Assign users along sith their deadline-priority
        priorities_list.append((usr, priority))

    # Sort the priorities_list based on deadline
    sorted_priorities_list = sorted(priorities_list, key=lambda x: (x[1]), reverse=True)
    for user in sorted_priorities_list:

        for service in user[0].applications[0].services:
            # We don't want to migrate services are already being migrated
            if service.server == None and not service.being_provisioned:

                # Let's iterate over the list of edge servers to find a suitable host for our service
                for edge_server in EdgeServer.all():

                    # We must check if the edge server has enough resources to host the service
                    if edge_server.has_capacity_to_host(service=service):
                        # Start provisioning the service in the edge server
                        service.provision(target_server=edge_server)

                        # After start migrating the service we can move on to the next service
                        break


#####################################
## Vanilla RL (vRL) implementation ##
def v_RL(parameters):
    # Override 'has_capacity_to_host' for all instances of the EdgeServer class
    EdgeServer.has_capacity_to_host = has_capacity_to_host_proposed

    ## for 31runs
    resource_tracker = ResourceTracker()

    """
    Convergence threshold is considered if the objective value (i.e., correct services allocation by scheduler in edge computing)
        exceeds less than 0.02% of the optimal value [2].
    
    [2]: Yu, Ming, et al. "Convergent policy optimization for safe reinforcement learning." Advances in Neural Information Processing Systems 32 (2019).
    """
    sliding_window = 100  # Number of consecutive episodes checking the objective's threshold
    objective_value_threshold = (0.98 * len(User.all()))  ## Determining a threshold for the 'hit-ratio' objective
    average_value_for_allocation, total_allocations_records = [], []
    num_completely_scheduled = 0
    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        unassigned_services_indices = [
            1 if service.server == server or service.being_provisioned else 0
            for service in Service.all()
        ]

        servers_range_indices = list(range(1, len(EdgeServer.all()) + 1))


        output = policy_net(state)

        if not unassigned_services_indices:
            raise ValueError("No unassigned tasks available for selection.")

        if sample > eps_threshold:
            with torch.no_grad():
                # Exploitation: Choose the best action based on policy_net
                return map_action_to_task_server(policy_net(state).max(1).indices.view(1, 1).item())
        else:
            # Exploration: Randomly select from unassigned tasks
            random_service_idx = random.randint(1,len(unassigned_services_indices))
            random_server_idx = random.randint(1,len(servers_range_indices))
            return torch.tensor([[random_service_idx,random_server_idx]], device=device, dtype=torch.long)


    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    class ReplayMemory(object):

        def __init__(self, capacity):
            self.rl_memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.rl_memory.append(Transition(*args))

        def sample(self, batch_size):
            return random.sample(self.rl_memory, batch_size)

        def __len__(self):
            return len(self.rl_memory)

    class DQN(nn.Module): ## was

        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 512)
            self.layer2 = nn.Linear(512, 512)
            self.layer3 = nn.Linear(512, n_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)


    BATCH_SIZE = 1024
    GAMMA = 0.995
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = ((len(Service.all())*600)/10)
    TAU = 0.005
    LR = 5e-4

    ## according to the update/modification that I did in 'def select_action(state)',
    # it seems there is no need to use this function anymore
    def map_action_to_task_server(action):
        """
        Maps an action index to a task and server.

        Args:
            action (int): The action index.
            total_num_tasks (int): Total number of tasks.
            total_num_servers (int): Total number of servers.

        Returns:
            (int, int): A tuple of (task_index, server_index).
        """

        translated_action = action + 1
        total_num_tasks = len(Service.all())
        total_num_servers = len(EdgeServer.all())
        # Determine the task and server indices
        task_index = ((translated_action - 1) // total_num_servers + 1) ## the task(service) 0 represents the first service which its ID is '1'
        server_index = ((translated_action - 1) % total_num_servers + 1) ## (action % total_num_servers) + 1 ## the server 0 represents the first server which its ID is '1'

        # Validate indices
        if task_index > total_num_tasks:
            raise ValueError("Action index out of bounds for the given number of tasks and servers.")

        return torch.tensor([[task_index, server_index]], device=device)


    # Get number of actions from EdgeSimPy converted action-space
    n_actions = (len(Service.all())*len(EdgeServer.all()))

    ## initial state for the RL-agent
    state = [0,0]
    ## number of observation is equal to the number of action that can be taken!?
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    rl_memory = ReplayMemory(500000)

    episode_durations = []
    episode_allocated_service = []
    episode_crtc_allc_services = []
    episodes_user_miss_deadline = []  ## number of users that miss their deadline in each episode

    def is_service_allocated_before(wanted_to_go_state):
        """
        Check the state to see if the selected_service is chosen before and is in its procedure of allocation or not

        Args:
            state (list): The original list of 0s.
            id (int): The index (1-based) to update in the list.

        Returns:
            bool
        """

        unassigned_services_indices = [
            1 if service.server == server or service.being_provisioned else 0
            for service in Service.all()
        ]

        if (unassigned_services_indices[(wanted_to_go_state[0]-1)] == 1):
            return True
        elif (unassigned_services_indices[(wanted_to_go_state[0]-1)] == 0):
            return False
        else:
            print("Error: id is out of range")


    def update_state(state, id):
        """
        Creates a new list by updating the n-th item to '1' based on the input id
        without modifying the original list.

        Args:
            state (list): The original list of 0s.
            id (int): The index (1-based) to update in the list.

        Returns:
            list: A new list with the n-th item set to '1'.
        """
        # Create a copy of the original list
        updated_state = state[:]
        # Convert 1-based index to 0-based index
        updated_state[id] = 1

        return updated_state

    def get_service_criticality_level(input_value):
        """
        Determine the processing level based on the input value.

        Args:
            input_value (int): The input integer value.

        Returns:
            float: The processing level as a string.
        """
        # Define the valid ranges and their corresponding outputs
        valid_ranges = {
            (22, 23): "3",
            (44, 46): "2",
            (2800, 4000): "1.2",
            (5600, 8000): "1",
        }

        # Check which range the input_value belongs to
        for (lower, upper), output in valid_ranges.items():
            if lower <= input_value <= upper:
                return float(output)

        # If no matching range is found
        return 0

    def compute_reward(not_redundant, enough_capacity, service_deadline_met, cpu_utilization_factor,
                       memory_utilization_factor, deadline_critical_level, response_time_factor, num_crtc_alloc_services, missed_tasks):
        """
        Compute the reward for the RL agent in a real-time task scheduling scenario.

        Args:
            enough_capacity (bool): Whether the selected server had enough capacity to host the service.
            service_deadline_met (bool): Whether the service's deadline was met.
            cpu_utilization_factor (float): CPU utilization factor of the server.
            memory_utilization_factor (float): Memory utilization factor of the server.
            response_time_factor (float): Factor representing the response time (lower is better).
            deadline_critical_level (float): A severity factor representing how far a task missed its deadline

        Returns:
            float: The computed reward.
        """
        reward = 0
        penalty = 0

        ######################
        ## Positive Rewards ##
        ######################

        if (num_crtc_alloc_services == len(Service.all())):
            reward += len(Service.all()) * 10


        if (not_redundant == 1):
            # Reward for selecting the service with the earliest deadline
            reward += num_crtc_alloc_services


        # Reward for efficient resource utilization (CPU and memory within capacity)
        if (enough_capacity == 1):
            reward += (num_crtc_alloc_services * 2)


        # Reward for meeting service deadlines
        if (service_deadline_met == 1):
            reward += (num_crtc_alloc_services * 4)


        ######################
        ## Negative Rewards ##
        ######################

        if ((missed_tasks + num_crtc_alloc_services) == len(Service.all())):
            reward -= missed_tasks * 1000


        # Redundant decision
        if (not_redundant == -1):
            # Reward for selecting the service with the earliest deadline
            reward -= missed_tasks

        # Penalty for exceeding server capacity
        if (enough_capacity == -1):
            reward -= (missed_tasks*1.5)

        # Severe penalty for missing deadlines
        if (service_deadline_met == -1):
            reward -= (missed_tasks*2)

        return reward

    def plot_durations(show_result=False):
        plt.figure(1)

        allocated_t = torch.tensor(episodes_user_miss_deadline, dtype=torch.float)

        plt.title('Result' if show_result else 'Training...')
        plt.xlabel('Episode')
        plt.ylabel('Hit-ratio (%)')

        plt.plot(allocated_t.numpy(), label='1-episode hit-ratio')

        if len(allocated_t) > 10:
            means = allocated_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            plt.plot(means.numpy(), label='10-episode average')

        plt.legend()
        plt.pause(0.001)  # Update the figure

    def optimize_model():
        if len(rl_memory) < BATCH_SIZE:
            return
        transitions = rl_memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
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

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 2500
    else:
        num_episodes = 500

    ### Measuring the power and memory usages
    process = psutil.Process(os.getpid())

    num_step_in_last_time_completion = 0
    last_num_of_allocated_services = 0
    for i_episode in range(num_episodes):

        # Initialize the environment and get its state # Use the reset method
        for server in EdgeServer._instances:
            server.reset_attributes()


        ## initial state for the RL-agent
        state = [0, 0]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        num_likely_missed_deadline = 0
        num_likely_MEET_deadline = 0
        # a list of users that miss their deadline due to missing of a service in their application
        user_miss_deadline = []
        reward_is_zero = 0
        total_rewards = 0

        for t in count():
            action = select_action(state)
            rl_task, rl_server = action[0][0].item(), action[0][1].item()

            rl_selected_service = next((s for s in Service._instances if s.id == (rl_task)), None)

            rl_selected_application = next(
                (app for app in Application._instances if rl_task in [service.id for service in app.services]),
                None
            )
            rl_selected_user = next((user for user in User._instances if rl_selected_application in user.applications), None)
            rl_selected_server = next((s for s in EdgeServer._instances if s.id == (rl_server)), None)

            avoid_redundant_service = 0
            server_poses_capacity = 0
            service_deadline_likely_met = 0

            if not is_service_allocated_before(action.squeeze(0).tolist()):

                avoid_redundant_service = 1

                if rl_selected_server.has_capacity_to_host(service=rl_selected_service):
                    server_poses_capacity = 1 ## put some positive reward in reward-function

                    service_criticality_level = get_service_criticality_level(
                        list(rl_selected_user.delay_slas.values())[0])
                    ############## response time ##################
                    communication_paths = []
                    topology = Topology.first()
                    communication_chain = [rl_selected_user.base_station, rl_selected_server.base_station]
                    for i in range(len(communication_chain) - 1):

                        # Defining origin and target nodes
                        origin = communication_chain[i]
                        target = communication_chain[i + 1]

                        # Finding and storing the best communication path between the origin and target nodes
                        if origin == target:
                            path = []
                        else:
                            path = nx.shortest_path(
                                G=topology,
                                source=origin.network_switch,
                                target=target.network_switch,
                                weight="delay",
                                method="dijkstra",
                            )

                        # Adding the best path found to the communication path
                        communication_paths.append([network_switch.id for network_switch in path])
                        ########
                    delay = 0.0
                    roundtrip_time = 0.0
                    # Initializes the application's delay with the time it takes to communicate its client and his base station
                    delay = rl_selected_user.base_station.wireless_delay
                    for path in communication_paths:
                        delay += topology.calculate_path_delay(path=[NetworkSwitch.find_by_id(i) for i in path])

                    roundtrip_time = (2 * delay)
                    response_time_for_service = round(
                        (roundtrip_time + rl_selected_server.execution_time_of_service[str(rl_selected_service.id)]), 4)

                    #################################################
                    if (response_time_for_service < list(rl_selected_user.delay_slas.values())[0]):
                        service_deadline_likely_met = 1
                        num_likely_MEET_deadline += 1
                        observation = action.squeeze(0).tolist()
                    else:
                        service_deadline_likely_met = -1
                        response_time_for_service = -1
                        num_likely_missed_deadline += 1
                        if rl_selected_user.id not in user_miss_deadline:
                            user_miss_deadline.append(rl_selected_user.id)
                        service_criticality_level = get_service_criticality_level(list(rl_selected_user.delay_slas.values())[0])
                        observation = action.squeeze(0).tolist()
                else:
                    server_poses_capacity = -1

                    response_time_for_service = -1
                    num_likely_missed_deadline += 1

                    if rl_selected_user.id not in user_miss_deadline:
                        user_miss_deadline.append(rl_selected_user.id)
                    service_criticality_level = get_service_criticality_level(list(rl_selected_user.delay_slas.values())[0])

                    observation = action.squeeze(0).tolist()


            else:
                avoid_redundant_service = -1
                response_time_for_service = -1
                num_likely_missed_deadline += 1

                if rl_selected_user.id not in user_miss_deadline:
                    user_miss_deadline.append(rl_selected_user.id)
                service_criticality_level = get_service_criticality_level(list(rl_selected_user.delay_slas.values())[0])

                observation = action.squeeze(0).tolist()


            ## calculating the reward
            reward = compute_reward(avoid_redundant_service, server_poses_capacity, service_deadline_likely_met, rl_selected_server.total_cpu_utilization,
                           rl_selected_server.total_memory_utilization, service_criticality_level, response_time_for_service, num_likely_MEET_deadline, num_likely_missed_deadline)

            total_rewards += reward

            reward = torch.tensor([reward], device=device)

            if num_likely_MEET_deadline == len(Service.all()):
                terminated = True
                num_completely_scheduled += 1
            else:
                terminated = False

            if ((num_likely_missed_deadline+num_likely_MEET_deadline) >= len(Service.all())):
                truncated = True
            else:
                truncated = False

            if terminated or truncated:
                done = True
                total_allocations_records.append((len(User.all()) - len(user_miss_deadline)))
                ### Measuring memory & power usages of normal-RL
                resource_tracker.update(process.memory_info().rss)
            else:
                done = False

            if terminated:
                next_state = None
                num_step_in_last_time_completion = t
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


            # Store the transition in rl_memory
            rl_memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                if episode_durations:
                    average_duration = sum(episode_durations) / len(episode_durations)


                if next_state is not None:
                    count_ones = torch.sum(next_state == 1).item()
                else:
                    count_ones = len(Service.all())  # Handle the case where next_state is None
                episode_allocated_service.append(count_ones)
                episode_crtc_allc_services.append(num_likely_MEET_deadline)
                episodes_user_miss_deadline.append((((len(User.all()) - len(user_miss_deadline)) / len(User.all())) * 100))
                
                print(f"Episode {len(episode_allocated_service)} with duration: {episode_durations[-1]}, and total rewards: {total_rewards}")
                file.write(f"Episode {len(episode_allocated_service)} with duration: {episode_durations[-1]}, and total rewards: {total_rewards}\n")


                print(f"  Number of services that are missed their deadline:{num_likely_missed_deadline}")
                file.write(f"  Number of services that are missed their deadline:{num_likely_missed_deadline}\n")
                print(f"Users who miss deadline due to service failure: {user_miss_deadline}")
                file.write(f"Users who miss deadline due to service failure: {user_miss_deadline}\n")


                print(f"Hit-ratio: {round((((len(User.all()) - len(user_miss_deadline)) / len(User.all())) * 100),2)}%.")
                file.write(f"Hit-ratio: {round((((len(User.all()) - len(user_miss_deadline)) / len(User.all())) * 100),2)}%.\n")

                if len(total_allocations_records) >= 10:
                    last_10_items = total_allocations_records[-10:]  # Get the last 10 items
                    avg = (sum(last_10_items) / len(last_10_items))/(len(User.all()))  # Calculate the average
                    print(f"  Average of hit-ratio in last 10-episodes is: {round((avg*100),2)}%")
                    file.write(f"  Average of hit-ratio in last 10-episodes is: {round((avg*100),2)}%.\n")
                last_num_of_allocated_services = count_ones

                ### Reporting the measured memory & power usages of normal-RL
                resource_tracker.report()
                print(f"========================================")
                file.write(f"========================================\n")

                break


        # Check for convergence by users
        if len(total_allocations_records) >= sliding_window:
            avg_hit_ratio = sum(
                total_allocations_records[-sliding_window:]) / sliding_window  # Compute average reward
            average_value_for_allocation.append(avg_hit_ratio)

            # Ensures the agent's performance exceeds the threshold, varying by less than 0.02% of the optimal value.
            if (avg_hit_ratio >= objective_value_threshold) and len(average_value_for_allocation) > 1:
                # Checks that the agent's performance is stable and not fluctuating around the threshold.
                if abs(average_value_for_allocation[-1] - average_value_for_allocation[-2]) < 1e-3:
                    print(f"Policy converged after {i_episode} episodes.")
                    file.write(f"Policy converged after {i_episode} episodes.\n")
                    print(f"=========================")
                    file.write(f"=========================\n")
                    break


    print('Complete')
    file.write(f"Complete\n")
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


################################
## Agile (aRL) implementation ##
def a_RL(parameters):
    # Override 'has_capacity_to_host' for all instances of the EdgeServer class
    EdgeServer.has_capacity_to_host = has_capacity_to_host_proposed

    ## for 31runs
    resource_tracker = ResourceTracker()

    """
    Convergence threshold is considered if the objective value (i.e., correct services allocation by scheduler in edge computing)
        exceeds less than 0.02% of the optimal value [2].

    [2]: Yu, Ming, et al. "Convergent policy optimization for safe reinforcement learning." Advances in Neural Information Processing Systems 32 (2019).
    """
    # sliding_window = 100  # Number of consecutive episodes checking the objective's threshold  ## was
    sliding_window = 20 ## is due to checking the wireless_delay_fluctuation
    objective_value_threshold = 0.98  ## Determining a threshold for the 'hit-ratio' objective  ## was
    average_value_for_allocation, total_allocations_records = [], []
    num_completely_scheduled = 0
    steps_done = 0
    num_states = 0
    edf_service_history = []
    Hist_is_service_allocated_before = []
    response_time_deadline_log_dict = {}
    selected_task_log_dict = {}

    hi_from_edf = 0
    hi_from_dl_decision = 0

    def edf_idx():
        """
        Check the earliest unassigned tasks until now!

        Returns:
            index of the service
        """
        nonlocal edf_service_history
        # Sort users by their minimum delay_sla in ascending order
        sorted_users = sorted(User.all(), key=lambda user: min(user.delay_slas.values()))
        selected_users_def_edf_idx = []
        for user in sorted_users:
            # Iterate through the user's services to check for unallocated services
            for service_edf in user.applications[0].services:
                if service_edf.server != servers and not service_edf.being_provisioned:
                    # User has at least one unallocated service
                    selected_users_def_edf_idx.append(user)
                    break  # Move to the next user once an unallocated service is found

            # Stop if the required number of users is reached
            if len(selected_users_def_edf_idx) >= (len(User.all())):
                break

        for a in selected_users_def_edf_idx:
            for service_edf in a.applications[0].services:
                # if service_edf.id not in edf_service_history:
                #     edf_service_history.append(service_edf.id)
                #     return service_edf.id
                return service_edf.id

    def select_action(state):
        nonlocal steps_done, num_states, hi_from_edf, hi_from_dl_decision
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        num_states += 1

        unassigned_services_indices = [
            1 if service.server == server or service.being_provisioned else 0
            for service in Service.all()
        ]

        servers_range_indices = list(range(1, len(EdgeServer.all()) + 1))

        output = policy_net(state)

        if not unassigned_services_indices:
            raise ValueError("No unassigned tasks available for selection.")

        if sample > eps_threshold:
            with (torch.no_grad()):
                hi_from_dl_decision += 1
                # Exploitation: Choose the best action based on policy_net
                # Restricting to unassigned tasks is not necessary for exploitation
                # print(selected_task_log_dict)

                if f"{int(map_action_to_task_server(policy_net(state).max(1).indices.view(1, 1).item())[0][0])}-{int(map_action_to_task_server(policy_net(state).max(1).indices.view(1, 1).item())[0][1])}" in selected_task_log_dict:

                    red_act = 2
                    while (f"{int(map_action_to_task_server(policy_net(state).topk(red_act, dim=1).indices[0, (red_act-1)].item())[0][0])}-{int(map_action_to_task_server(policy_net(state).topk(red_act, dim=1).indices[0, (red_act-1)].item())[0][1])}" in selected_task_log_dict):
                        red_act += 1
                    # print(f"select_action: {int(map_action_to_task_server(policy_net(state).topk(red_act, dim=1).indices[0, (red_act-1)].item())[0][0])}")
                    # print(selected_task_log_dict)
                    return map_action_to_task_server(policy_net(state).topk(red_act, dim=1).indices[0, (red_act-1)].item())
                else:
                    # print(
                    #     f"ELSE_select_action: {int(map_action_to_task_server(policy_net(state).max(1).indices.view(1, 1).item())[0][0])}")
                    # print(f"else{selected_task_log_dict}")
                    return map_action_to_task_server(policy_net(state).max(1).indices.view(1, 1).item())

        else:
            hi_from_edf += 1
            # Exploration: Randomly select from unassigned tasks
            edf_service_idx = edf_idx()
            # while (edf_service_idx in selected_task_log_dict):
            #     edf_service_idx += 1
            # if (edf_service_idx > 262):
            #     edf_service_idx = 262
            # if (edf_service_idx in selected_task_log_dict):
            #     print(f"redundant_edf_service_idx: {edf_service_idx}")
            # else:
            #     print(f"edf_service_idx: {edf_service_idx}")
            edf_server_idx = random.randint(1, len(servers_range_indices))

            while (f"{edf_service_idx}-{edf_server_idx}" in selected_task_log_dict):
                seed_edf_service_idx = [x for x in range(1, 262) if x != int(edf_service_idx)]
                seed_edf_server_idx = [x for x in range(1, 4) if x != int(edf_server_idx)]

                edf_service_idx = random.choice(seed_edf_service_idx)
                edf_server_idx = random.choice(seed_edf_server_idx)


            return torch.tensor([[edf_service_idx, edf_server_idx]], device=device, dtype=torch.long)

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward', 'done'))

    # class ReplayMemory(object):
    #
    #     def __init__(self, capacity):
    #         self.rl_memory = deque([], maxlen=capacity)
    #
    #     def push(self, *args):
    #         """Save a transition"""
    #         self.rl_memory.append(Transition(*args))
    #
    #     def sample(self, batch_size):
    #         return random.sample(self.rl_memory, batch_size)
    #
    #     def __len__(self):
    #         return len(self.rl_memory)



    class ReplayMemory(object):
        """A prioritized experience replay buffer for reinforcement learning."""

        def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-5):
            """
            Initialize the replay memory with specified parameters.

            Args:
                capacity: Maximum number of transitions to store
                alpha: Prioritization factor (0 = uniform, 1 = full prioritization)
                epsilon: Small constant to avoid zero priority
            """
            if not isinstance(capacity, int) or capacity <= 0:
                raise ValueError("Capacity must be a positive integer")
            if not 0 <= alpha <= 1:
                raise ValueError("Alpha must be between 0 and 1")
            if epsilon <= 0:
                raise ValueError("Epsilon must be positive")

            self.capacity = capacity
            self.alpha = alpha
            self.epsilon = epsilon

            # Initialize storage
            self.memory = deque(maxlen=capacity)
            self.priorities = deque(maxlen=capacity)
            self.position = 0

        def push(self, *args) -> None:
            """
            Add a transition to memory with maximum priority.

            Args:
                *args: Transition components (state, action, reward, next_state, done)
            """
            transition = Transition(*args)

            # Get maximum priority (default to 1.0 if empty)
            max_priority = max(self.priorities) if self.priorities else 1.0

            if len(self.memory) < self.capacity:
                self.memory.append(transition)
                self.priorities.append(max_priority)
            else:
                self.memory[self.position] = transition
                self.priorities[self.position] = max_priority

            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size: int) -> Tuple[List[Any], List[int], np.ndarray]:
            """
            Sample a batch of transitions based on priorities.

            Args:
                batch_size: Number of transitions to sample

            Returns:
                Tuple of (transitions, indices, sampling_probabilities)
            """
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("Batch size must be a positive integer")
            if batch_size > len(self.memory):
                raise ValueError("Batch size cannot exceed current memory size")
            if not self.memory:
                return [], [], np.array([])

            # Calculate sampling probabilities
            priorities = np.array(self.priorities, dtype=np.float32)
            scaled_priorities = priorities ** self.alpha
            sampling_probs = scaled_priorities / scaled_priorities.sum()

            # Sample indices
            indices = np.random.choice(len(self.memory), batch_size, p=sampling_probs)
            transitions = [self.memory[idx] for idx in indices]

            return transitions, indices.tolist(), sampling_probs[indices]

        def update_priority(self, indices: List[int], errors: List[float]) -> None:
            """
            Update priorities for specified transitions.

            Args:
                indices: List of transition indices to update
                errors: List of corresponding error values
            """
            if len(indices) != len(errors):
                raise ValueError("Number of indices must match number of errors")
            if not all(0 <= idx < len(self.memory) for idx in indices):
                raise ValueError("Invalid index provided")

            for idx, error in zip(indices, errors):
                priority = abs(error) + self.epsilon
                self.priorities[idx] = priority

        def __len__(self) -> int:
            """Return current size of memory."""
            return len(self.memory)

        def clear(self) -> None:
            """Clear all transitions and priorities from memory."""
            self.memory.clear()
            self.priorities.clear()
            self.position = 0

    class DQN(nn.Module):  ## was

        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 512)
            self.layer2 = nn.Linear(512, 512)
            self.layer3 = nn.Linear(512, n_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)


    BATCH_SIZE = 1024
    GAMMA = 0.995
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 5000
    TAU = 0.005
    LR = 5e-4

    ## according to the update/modification that I did in 'def select_action(state)',
    # it seems there is no need to use this function anymore
    def map_action_to_task_server(action):
        """
        Maps an action index to a task and server.

        Args:
            action (int): The action index.
            total_num_tasks (int): Total number of tasks.
            total_num_servers (int): Total number of servers.

        Returns:
            (int, int): A tuple of (task_index, server_index).
        """

        translated_action = action + 1
        total_num_tasks = len(Service.all())
        total_num_servers = len(EdgeServer.all())

        # Determine the task and server indices
        task_index = ((
                                  translated_action - 1) // total_num_servers + 1)  ## the task(service) 0 represents the first service which its ID is '1'

        server_index = ((
                                    translated_action - 1) % total_num_servers + 1)  ## (action % total_num_servers) + 1 ## the server 0 represents the first server which its ID is '1'


        # Validate indices
        if task_index > total_num_tasks:
            raise ValueError("Action index out of bounds for the given number of tasks and servers.")

        # print(f"task_index:{task_index}")

        return torch.tensor([[task_index, server_index]], device=device)

    """
    Based on my understanding, it would be better to consider the state = [service, server], where the range of
    task=[1,..,262] and server=[1,..,4].
    """

    # Get number of actions from EdgeSimPy converted action-space
    n_actions = (len(Service.all()) * len(EdgeServer.all()))

    ## initial state for the RL-agent
    state = [0, 0]

    ## number of observation is equal to the number of action that can be taken!?
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    rl_memory = ReplayMemory(500000)

    episode_durations = []
    episode_allocated_service = []
    episode_crtc_allc_services = []
    episodes_user_miss_deadline = []  ## number of users that miss their deadline in each episode

    def is_service_allocated_before(wanted_to_go_state):
        """
        Check the state to see if the selected_service is chosen before and is in its procedure of allocation or not

        Args:
            state (list): The original list of 0s.
            id (int): The index (1-based) to update in the list.

        Returns:
            bool
        """

        if wanted_to_go_state in Hist_is_service_allocated_before:
            return True
        elif wanted_to_go_state in Hist_is_service_allocated_before:
            Hist_is_service_allocated_before.append(wanted_to_go_state)
            return False



    def update_state(state, id):
        """
        Creates a new list by updating the n-th item to '1' based on the input id
        without modifying the original list.

        Args:
            state (list): The original list of 0s.
            id (int): The index (1-based) to update in the list.

        Returns:
            list: A new list with the n-th item set to '1'.
        """
        # Create a copy of the original list
        updated_state = state[:]
        # Convert 1-based index to 0-based index
        updated_state[id] = 1

        return updated_state

    def get_service_criticality_level(input_value):
        """
        Determine the processing level based on the input value.

        Args:
            input_value (int): The input integer value.

        Returns:
            float: The processing level as a string.
        """
        # Define the valid ranges and their corresponding outputs
        valid_ranges = {
            (22, 23): "3",
            (44, 46): "2",
            (2800, 4000): "1.2",
            (5600, 8000): "1",
        }

        # Check which range the input_value belongs to
        for (lower, upper), output in valid_ranges.items():
            if lower <= input_value <= upper:
                return float(output)

        # If no matching range is found
        return 0

    def compute_reward(not_redundant, enough_capacity, service_deadline_met, cpu_utilization_factor,
                       memory_utilization_factor, deadline_critical_level, response_time_factor,
                       num_crtc_alloc_services, missed_tasks):
        """
        Compute the reward for the RL agent in a real-time task scheduling scenario.

        Args:
            enough_capacity (bool): Whether the selected server had enough capacity to host the service.
            service_deadline_met (bool): Whether the service's deadline was met.
            cpu_utilization_factor (float): CPU utilization factor of the server.
            memory_utilization_factor (float): Memory utilization factor of the server.
            response_time_factor (float): Factor representing the response time (lower is better).
            deadline_critical_level (float): A severity factor representing how far a task missed its deadline

        Returns:
            float: The computed reward.
        """
        reward = 0
        penalty = -missed_tasks

        ######################
        ## Positive Rewards ##
        ######################

        if (num_crtc_alloc_services == len(Service.all())):
            reward += len(Service.all()) * 10
        #     reward += 1

        if (not_redundant == 1):
            # Reward for selecting the service with the earliest deadline
            # reward += num_crtc_alloc_services
            # reward += len(Service.all())
            # reward += 1
            reward += 0.25
            # print(f"not_redundant:{not_redundant}, reward:{reward}")

        # Reward for efficient resource utilization (CPU and memory within capacity)
        if (enough_capacity == 1):
            # reward += (num_crtc_alloc_services * 2)
            reward += 0.25
            # print(f"enough_capacity:{enough_capacity}, reward:{reward}")

        # Reward for meeting service deadlines
        if (service_deadline_met == 1):
            # reward += (num_crtc_alloc_services * 3)
            reward += (((num_crtc_alloc_services)/(len(Service.all())))*100)
            # print(f"service_deadline_met:{service_deadline_met}, reward:{reward}")

        ######################
        ## Negative Rewards ##
        ######################

        # if ((missed_tasks + num_crtc_alloc_services) == len(Service.all())):
        #     reward += (num_crtc_alloc_services - missed_tasks)
        #     print(f"reward:{reward}")

        # Redundant decision
        if (not_redundant == -1):
            # Reward for selecting the service with the earliest deadline
            # reward -= len(Service.all())
            reward = reward - 1
            # print(f"not_redundant:{not_redundant}, reward:{reward}")

        if (response_time_factor == -1) or (service_deadline_met == -1):
            reward = reward - (((missed_tasks*60)/(len(Service.all())))*100)
            # print("penalty response_time_factor")

        # Penalty for exceeding server capacity
        if (enough_capacity == -1):
            # reward -= (missed_tasks*2)
            reward = reward - 0.5
            # print("penalty enough_capacity")

        # # Severe penalty for missing deadlines
        # if (service_deadline_met == -1):
        #     # reward -= (missed_tasks*4)
        #     reward = reward - 2
        #     # print("penalty service_deadline_met")

        # if (not_redundant == -1) or (response_time_factor == -1) or (enough_capacity == -1) or (service_deadline_met == -1):
        #     print()
        # reward += penalty
        # print(f"action reward:{reward}")
        rounded_reward = round(reward, 1)
        return rounded_reward

    def plot_durations(show_result=False):
        plt.figure(1)  # Work on figure #1

        allocated_t = torch.tensor(episodes_user_miss_deadline, dtype=torch.float)

        plt.title('Result' if show_result else 'Training...')
        plt.xlabel('Episode')
        plt.ylabel('Hit-ratio (%)')

        plt.plot(allocated_t.numpy(), label='1-episode hit-ratio')

        if len(allocated_t) > 10:
            means = allocated_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            plt.plot(means.numpy(), label='10-episode average')

        plt.legend()
        plt.pause(0.001)  # Update the figure

    # def optimize_model():
    #     if len(rl_memory) < BATCH_SIZE:
    #         return
    #     transitions = rl_memory.sample(BATCH_SIZE)
    #     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    #     # detailed explanation). This converts batch-array of Transitions
    #     # to Transition of batch-arrays.
    #     batch = Transition(*zip(*transitions))
    #
    #     # Compute a mask of non-final states and concatenate the batch elements
    #     # (a final state would've been the one after which simulation ended)
    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
    #                                   dtype=torch.bool)
    #     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    #     state_batch = torch.cat(batch.state)
    #     action_batch = torch.cat(batch.action)
    #     reward_batch = torch.cat(batch.reward)
    #
    #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    #     # columns of actions taken. These are the actions which would've been taken
    #     # for each batch state according to policy_net
    #     state_action_values = policy_net(state_batch).gather(1, action_batch)
    #
    #     # Compute V(s_{t+1}) for all next states.
    #     # Expected values of actions for non_final_next_states are computed based
    #     # on the "older" target_net; selecting their best reward with max(1).values
    #     # This is merged based on the mask, such that we'll have either the expected
    #     # state value or 0 in case the state was final.
    #     next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #     with torch.no_grad():
    #         next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    #     # Compute the expected Q values
    #     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    #
    #     # Compute Huber loss
    #     criterion = nn.SmoothL1Loss()
    #     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    #
    #     # Optimize the model
    #     optimizer.zero_grad()
    #     loss.backward()
    #     # In-place gradient clipping
    #     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    #     optimizer.step()

    def optimize_model():
        if len(rl_memory) < BATCH_SIZE:
            return

        transitions, indices, sampling_probabilities = rl_memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)  # [1024, 2]
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        q_values = policy_net(state_batch)  # [1024, num_actions]
        state_action_values = q_values.gather(1, action_batch)  # [1024, 2]

        # Compute V(s_{t+1})
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        # Compute expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # [1024]

        # Reduce state_action_values to match expected values
        state_action_values_mean = state_action_values.mean(dim=1)  # [1024], average over action dims

        # Compute TD errors for priority updates
        with torch.no_grad():
            td_errors = torch.abs(state_action_values_mean - expected_state_action_values).cpu().numpy()
        rl_memory.update_priority(indices, td_errors)

        # Compute importance sampling weights
        beta = 0.4
        is_weights = (1.0 / (len(rl_memory) * sampling_probabilities)) ** beta
        is_weights = torch.tensor(is_weights, device=device, dtype=torch.float32)
        is_weights /= is_weights.max()

        # Compute weighted Huber loss
        criterion = nn.SmoothL1Loss(reduction='none')
        loss_per_sample = criterion(state_action_values_mean, expected_state_action_values)  # [1024]
        weighted_loss = (loss_per_sample * is_weights).mean()

        # Optimize the model
        optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 2500
    else:
        num_episodes = 500

    ### Measuring the power and memory usages
    process = psutil.Process(os.getpid())

    num_step_in_last_time_completion = 0
    last_num_of_allocated_services = 0
    for i_episode in range(num_episodes):
        terminated, truncated, done = False, False, False
        # Create an empty dictionary to store your log messages
        response_time_deadline_log_dict = {}
        selected_task_log_dict = {}

        # Initialize the environment and get its state # Use the reset method
        for server in EdgeServer._instances:
            server.reset_attributes()

        ## initial state for the RL-agent
        state = [0, 0]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        num_likely_missed_deadline = 0
        num_likely_MEET_deadline = 0
        # a list of users that miss their deadline due to missing of a service in their application
        user_miss_deadline = []
        reward_is_zero = 0
        total_rewards = 0

        for t in count():
            action = select_action(state)
            rl_task, rl_server = action[0][0].item(), action[0][1].item()
            # print(f"rl_task:{rl_task}")
            rl_selected_service = next((s for s in Service._instances if s.id == (rl_task)), None)
            # print(f"rl_selected_service:{rl_selected_service}\n")

            rl_selected_application = next(
                (app for app in Application._instances if rl_task in [service.id for service in app.services]),
                None
            )
            rl_selected_user = next((user for user in User._instances if rl_selected_application in user.applications),
                                    None)
            rl_selected_server = next((s for s in EdgeServer._instances if s.id == (rl_server)), None)

            selected_task_log_dict[
                f"{rl_selected_service.id}-{rl_selected_server.id}"] = f"{rl_selected_service.id}-{rl_selected_server.id} is selected"
            # print(f"selected_task_log_dict:{selected_task_log_dict}\n")

            avoid_redundant_service = 0
            server_poses_capacity = 0
            service_deadline_likely_met = 0
            # print(f"rl_selected_service.id:{rl_selected_service.id}")
            if (not is_service_allocated_before(action.squeeze(0).tolist()[0])) and (not f"{rl_selected_service.id}-{rl_selected_server.id}" in response_time_deadline_log_dict):
                avoid_redundant_service = 1
                if rl_selected_server.has_capacity_to_host(service=rl_selected_service):
                    server_poses_capacity = 1  ## put some positive reward in reward-function

                    service_criticality_level = get_service_criticality_level(
                        list(rl_selected_user.delay_slas.values())[0])
                    ############## response time ##################
                    communication_paths = []
                    topology = Topology.first()
                    communication_chain = [rl_selected_user.base_station, rl_selected_server.base_station]
                    for i in range(len(communication_chain) - 1):

                        # Defining origin and target nodes
                        origin = communication_chain[i]
                        target = communication_chain[i + 1]

                        # Finding and storing the best communication path between the origin and target nodes
                        if origin == target:
                            path = []
                        else:
                            path = nx.shortest_path(
                                G=topology,
                                source=origin.network_switch,
                                target=target.network_switch,
                                weight="delay",
                                method="dijkstra",
                            )

                        # Adding the best path found to the communication path
                        communication_paths.append([network_switch.id for network_switch in path])
                        ########
                    delay = 0.0
                    roundtrip_time = 0.0
                    # Initializes the application's delay with the time it takes to communicate its client and his base station
                    delay = rl_selected_user.base_station.wireless_delay
                    for path in communication_paths:
                        delay += topology.calculate_path_delay(path=[NetworkSwitch.find_by_id(i) for i in path])

                    roundtrip_time = (2 * delay)
                    response_time_for_service = round(
                        (roundtrip_time + rl_selected_server.execution_time_of_service[str(rl_selected_service.id)]), 4)

                    #################################################
                    if (response_time_for_service < list(rl_selected_user.delay_slas.values())[0]):
                        # Build the message string
                        message = f"response_time {response_time_for_service}, deadline {list(rl_selected_user.delay_slas.values())[0]}"
                        # Store the string in the dictionary, using i or some unique key
                        response_time_deadline_log_dict[f"{rl_selected_service.id}-{rl_selected_server.id}"] = message

                        # print(f"taskService_{rl_selected_service.id}, response_time {response_time_for_service}, deadline {list(rl_selected_user.delay_slas.values())[0]}")
                        service_deadline_likely_met = 1
                        num_likely_MEET_deadline += 1
                        observation = action.squeeze(0).tolist()
                    else:
                        # print("no response time")
                        # if (len(episode_allocated_service) == 50):
                        #     print(
                        #     f" not meet response_time {response_time_for_service}, task deadline {list(rl_selected_user.delay_slas.values())[0]}")
                        service_deadline_likely_met = -1
                        response_time_for_service = -1
                        num_likely_missed_deadline += 1
                        if rl_selected_user.id not in user_miss_deadline:  ## was
                            user_miss_deadline.append(rl_selected_user.id) ## was
                        service_criticality_level = get_service_criticality_level(
                            list(rl_selected_user.delay_slas.values())[0])
                        observation = action.squeeze(0).tolist()
                else:
                    # print("no capacity")
                    server_poses_capacity = -1
                    response_time_for_service = -1
                    num_likely_missed_deadline += 1
                    if rl_selected_user.id not in user_miss_deadline:  ## was
                        user_miss_deadline.append(rl_selected_user.id) ## was
                    service_criticality_level = get_service_criticality_level(
                        list(rl_selected_user.delay_slas.values())[0])
                    observation = action.squeeze(0).tolist()

            else:
                # print("redundant action")
                print(response_time_deadline_log_dict)
                avoid_redundant_service = -1
                response_time_for_service = -1
                num_likely_missed_deadline += 1
                if rl_selected_user.id not in user_miss_deadline:  ## was
                    user_miss_deadline.append(rl_selected_user.id) ## was
                service_criticality_level = get_service_criticality_level(list(rl_selected_user.delay_slas.values())[0])
                observation = action.squeeze(0).tolist()


            reward = compute_reward(avoid_redundant_service, server_poses_capacity, service_deadline_likely_met,
                                    rl_selected_server.total_cpu_utilization,
                                    rl_selected_server.total_memory_utilization, service_criticality_level,
                                    response_time_for_service, num_likely_MEET_deadline, num_likely_missed_deadline)

            total_rewards += reward
            # print()
            reward = torch.tensor([reward], device=device)

            if num_likely_MEET_deadline == len(Service.all()):
                print(f"num_likely_MEET_deadline:{num_likely_MEET_deadline}, action taken:{t}")
                # print(f"len(Service.all()):{len(Service.all())}")
                terminated = True
                num_completely_scheduled += 1
            else:
                terminated = False

            # if ((num_likely_missed_deadline + num_likely_MEET_deadline) >= len(Service.all())):
            if (t > 280):
            #     print(f"action taken:{num_likely_missed_deadline + num_likely_MEET_deadline}")
            #     print(f"len(Service.all()):{len(Service.all())}")
                truncated = True
            else:
                truncated = False

            if terminated or truncated:
                done = True
                # print(f"terminated:{terminated}")
                # print(f"truncated:{truncated}")
                # print(f"done:{done}")
                # total_allocations_records.append(num_likely_MEET_deadline)
                # total_allocations_records.append((len(User.all()) - len(user_miss_deadline))) ## was
                if (num_likely_MEET_deadline == len(Service.all())):
                    num_likely_missed_deadline = 0
                    user_miss_deadline = []
                    total_allocations_records.append((len(User.all()) - len(user_miss_deadline))/len(User.all()))
                else:
                    total_allocations_records.append((len(User.all()) - len(user_miss_deadline))/len(User.all()))
                ### Measuring memory & power usages of normal-RL
                resource_tracker.update(process.memory_info().rss)
            else:
                done = False

            if terminated:
                next_state = None
                num_step_in_last_time_completion = t
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in rl_memory
            rl_memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                edf_service_history = []
                Hist_is_service_allocated_before = []
                episode_durations.append(t + 1)
                if episode_durations:
                    average_duration = sum(episode_durations) / len(episode_durations)

                if next_state is not None:
                    count_ones = torch.sum(next_state == 1).item()
                else:
                    count_ones = len(Service.all())  # Handle the case where next_state is None
                episode_allocated_service.append(count_ones)
                episode_crtc_allc_services.append(num_likely_MEET_deadline)
                episodes_user_miss_deadline.append(
                    (((len(User.all()) - len(user_miss_deadline)) / len(User.all())) * 100))

                # print(f"hi_from_edf:{hi_from_edf}")
                # print(f"hi_from_dl_decision:{hi_from_dl_decision}")
                hi_from_dl_decision = 0
                hi_from_edf = 0
                hi_from_dl_decision = 0

                print(
                    f"Episode {len(episode_allocated_service)} with duration: {episode_durations[-1]}, and total rewards: {total_rewards}")
                file.write(
                    f"Episode {len(episode_allocated_service)} with duration: {episode_durations[-1]}, and total rewards: {total_rewards}\n")


                print(f"  Number of services that are missed their deadline:{num_likely_missed_deadline}")
                file.write(f"  Number of services that are missed their deadline:{num_likely_missed_deadline}\n")
                print(f"Users who miss deadline due to service failure: {user_miss_deadline}")
                file.write(f"Users who miss deadline due to service failure: {user_miss_deadline}\n")

                # user based hit-ratio
                print(
                    f"Hit-ratio: {round((((len(User.all()) - len(user_miss_deadline)) / len(User.all())) * 100), 2)}%.")
                # if(round((((len(User.all()) - len(user_miss_deadline)) / len(User.all())) * 100), 2) > 95):
                #     print(f"len dict: {len(response_time_deadline_log_dict)}")
                    # for key, value in response_time_deadline_log_dict.items():
                    #     print(f"{key}: {value}")
                file.write(
                    f"Hit-ratio: {round((((len(User.all()) - len(user_miss_deadline)) / len(User.all())) * 100), 2)}%.\n")

                # ## task based hit-ratio
                # print(
                #     f"Hit-ratio: {round((((len(Service.all()) - num_likely_missed_deadline) / len(Service.all())) * 100), 2)}%.")
                # file.write(
                #     f"Hit-ratio: {round((((len(User.all()) - len(user_miss_deadline)) / len(User.all())) * 100), 2)}%.\n")

                print(f'number of state:{num_states}')
                num_states = 0

                if len(total_allocations_records) >= 10:
                    last_10_items = total_allocations_records[-10:]  # Get the last 10 items
                    avg = (sum(last_10_items) / len(last_10_items)) / (len(User.all()))  # Calculate the average
                    file.write(f"  Average of hit-ratio in last 10-episodes is: {round((avg * 100), 2)}%.\n")
                last_num_of_allocated_services = count_ones

                ### Reporting the measured memory & power usages of normal-RL
                resource_tracker.report()
                print(f"========================================")
                file.write(f"========================================\n")
                # sys.exit(0)
                break

        # Check for convergence by users
        if len(total_allocations_records) >= sliding_window:
            # print(f"len(total_allocations_records):{len(total_allocations_records)}, sliding_window:{sliding_window}")
            avg_hit_ratio = sum(
                total_allocations_records[-sliding_window:]) / sliding_window  # Compute average reward
            average_value_for_allocation.append(avg_hit_ratio)
            # print(f"total_allocations_records:{total_allocations_records}")
            # print(f"avg_hit_ratio:{avg_hit_ratio}")
            # if (len(total_allocations_records)>=20):
            #     print(f"avg_HR_{20}:{sum(total_allocations_records[-20:]) / 20}")
            # print(f"objective_value_threshold:{objective_value_threshold}\n")
            # Ensures the agent's performance exceeds the threshold, varying by less than 0.02% of the optimal value.
            if (avg_hit_ratio >= objective_value_threshold) and len(average_value_for_allocation) > 1:
                # Checks that the agent's performance is stable and not fluctuating around the threshold.
                # print(f"average_value_for_allocation[-1]:{average_value_for_allocation[-1]}, average_value_for_allocation[-2]:{average_value_for_allocation[-2]}")
                # if abs(average_value_for_allocation[-1] - average_value_for_allocation[-2]) < 1e-3:
                #     print(f"Policy converged after {i_episode} episodes.")
                #     file.write(f"Policy converged after {i_episode} episodes.\n")
                #     print(f"=========================")
                #     file.write(f"=========================\n")
                #     break
                print(f"Policy converged after {i_episode} episodes.")
                file.write(f"Policy converged after {i_episode} episodes.\n")
                print(f"=========================")
                file.write(f"=========================\n")
                break

    print('Complete')
    file.write(f"Complete\n")
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


# #############################################
# ## Distributed Agile (DaRL) implementation ##
# #############################################
# def D_a_RL(parameters):
#     import dill
#     import multiprocessing.reduction as reduction
#     dill.settings['recurse'] = True
#     reduction.ForkingPickler = dill.Pickler
#     EdgeServer.has_capacity_to_host = has_capacity_to_host_proposed
#     resource_tracker = ResourceTracker()
#
#     # for service in Service.all():
#     #     print(f"service is {service}") ## accessible in here
#
#     # Define hyperparameters early so nested functions can access them
#     BATCH_SIZE = 1024
#     GAMMA = 0.995
#     EPS_START = 1.0
#     EPS_END = 0.05
#     EPS_DECAY = 5000
#     TAU = 0.005
#     LR = 5e-4
#     NUM_PROCESSES = 4
#
#     sliding_window = 20  ## is due to checking the wireless_delay_fluctuation
#     objective_value_threshold = 0.98  ## Determining a threshold for the 'hit-ratio' objective  ## was
#     average_value_for_allocation, total_allocations_records = [], []
#     num_completely_scheduled = 0
#     steps_done = 0
#     num_states = 0
#     edf_service_history = []
#     Hist_is_service_allocated_before = []
#     response_time_deadline_log_dict = {}
#     selected_task_log_dict = {}
#
#     hi_from_edf = 0
#     hi_from_dl_decision = 0
#
#     def edf_idx(): # Check the earliest unassigned tasks until now! Returns:  index of the service
#         nonlocal edf_service_history
#         # Sort users by their minimum delay_sla in ascending order
#         sorted_users = sorted(User.all(), key=lambda user: min(user.delay_slas.values()))
#         selected_users_def_edf_idx = []
#         for user in sorted_users:
#             # Iterate through the user's services to check for unallocated services
#             for service_edf in user.applications[0].services:
#                 if service_edf.server != servers and not service_edf.being_provisioned:
#                     # User has at least one unallocated service
#                     selected_users_def_edf_idx.append(user)
#                     break  # Move to the next user once an unallocated service is found
#
#             # Stop if the required number of users is reached
#             if len(selected_users_def_edf_idx) >= (len(User.all())):
#                 break
#
#         for a in selected_users_def_edf_idx:
#             for service_edf in a.applications[0].services:
#                 # if service_edf.id not in edf_service_history:
#                 #     edf_service_history.append(service_edf.id)
#                 #     return service_edf.id
#                 return service_edf.id
#
#     # Define your Transition namedtuple
#     from collections import namedtuple, deque
#     Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
#
#     # Define ReplayMemory locally
#     import numpy as np
#     class ReplayMemory:
#         def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-5):
#             self.capacity = capacity
#             self.alpha = alpha
#             self.epsilon = epsilon
#             self.memory = deque(maxlen=capacity)
#             self.priorities = deque(maxlen=capacity)
#             self.position = 0
#
#         def push(self, *args):
#             transition = Transition(*args)
#             max_priority = max(self.priorities) if self.priorities else 1.0
#             if len(self.memory) < self.capacity:
#                 self.memory.append(transition)
#                 self.priorities.append(max_priority)
#             else:
#                 self.memory[self.position] = transition
#                 self.priorities[self.position] = max_priority
#             self.position = (self.position + 1) % self.capacity
#
#         def sample(self, batch_size: int):
#             if len(self.memory) < batch_size:
#                 return [], [], np.array([])
#             priorities = np.array(self.priorities, dtype=np.float32)
#             scaled_priorities = priorities ** self.alpha
#             sampling_probs = scaled_priorities / scaled_priorities.sum()
#             indices = np.random.choice(len(self.memory), batch_size, p=sampling_probs)
#             transitions = [self.memory[idx] for idx in indices]
#             return transitions, indices.tolist(), sampling_probs[indices]
#
#         def update_priority(self, indices, errors):
#             for idx, error in zip(indices, errors):
#                 priority = abs(error) + self.epsilon
#                 self.priorities[idx] = priority
#
#         def __len__(self):
#             return len(self.memory)
#
#     # Define DQN locally
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F
#     class DQN(nn.Module):
#         def __init__(self, n_observations, n_actions):
#             super(DQN, self).__init__()
#             self.layer1 = nn.Linear(n_observations, 512)
#             self.layer2 = nn.Linear(512, 512)
#             self.layer3 = nn.Linear(512, n_actions)
#
#         def forward(self, x):
#             x = F.relu(self.layer1(x))
#             x = F.relu(self.layer2(x))
#             return self.layer3(x)
#
#     # for service in Service.all():
#     #     print(f"service is {service}") ### si fine here
#
#     # Define WorkerLogic locally
#     import random, math
#     from itertools import count
#
#     class WorkerLogic:
#         @staticmethod
#         def run(worker_id, parameters, global_policy_net, global_target_net, global_optimizer,
#                 device, n_observations, n_actions, resource_tracker):
#             local_policy_net = DQN(n_observations, n_actions).to(device)
#             local_policy_net.train()
#             local_memory = ReplayMemory(500000)
#             steps_done = 0
#
#             print("Inside WorkerLogic.run()")
#             try:
#                 services = Service.all()
#                 print(f"Service.all() returned: {services}")
#                 for service in services:
#                     print(f"Inside class: service is {service}")
#             except Exception as e:
#                 print(f"Error accessing Service.all(): {e}")
#
#             for service in services:
#                 print(f"Worker {worker_id}: service is {service}")
#
#             # Nested helper function to select an action using epsilon-greedy policy
#             def select_action(state):
#                 nonlocal steps_done, num_states, hi_from_edf, hi_from_dl_decision
#                 sample = random.random()
#                 eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#                                 math.exp(-1. * steps_done / EPS_DECAY)
#                 steps_done += 1
#                 num_states += 1
#
#                 # unassigned_services_indices = [
#                 #     1 if service.server == server or service.being_provisioned else 0
#                 #     for service in Service.all()
#                 # ]
#                 print("hi before loooop")
#                 for service in Service.all():
#                     print(service)
#                     print("hi in loooop")
#
#                 print(f"len(User.all()):{len(User.all())}")
#
#                 print("hi 1938 loooop")
#                 for service in Service.all():
#                     print("hi")
#                     condition = (service.server == server or service.being_provisioned)
#                     print(
#                         f"Service: {service}, service.server: {service.server}, server: {server}, being_provisioned: {service.being_provisioned}, condition: {condition}")
#                     unassigned_services_indices.append(1 if condition else 0)
#
#
#
#                 servers_range_indices = list(range(1, len(EdgeServer.all()) + 1))
#
#                 output = local_policy_net(state)
#
#                 if not unassigned_services_indices:
#                     print(f"unassigned_services_indices:{unassigned_services_indices}")
#                     raise ValueError("No unassigned tasks available for selection.")
#
#                 if sample > eps_threshold:
#                     with (torch.no_grad()):
#                         hi_from_dl_decision += 1
#                         # Exploitation: Choose the best action based on policy_net
#                         # Restricting to unassigned tasks is not necessary for exploitation
#                         # print(selected_task_log_dict)
#
#                         if f"{int(map_action_to_task_server(policy_net(state).max(1).indices.view(1, 1).item())[0][0])}-{int(map_action_to_task_server(policy_net(state).max(1).indices.view(1, 1).item())[0][1])}" in selected_task_log_dict:
#
#                             red_act = 2
#                             while (
#                                     f"{int(map_action_to_task_server(policy_net(state).topk(red_act, dim=1).indices[0, (red_act - 1)].item())[0][0])}-{int(map_action_to_task_server(policy_net(state).topk(red_act, dim=1).indices[0, (red_act - 1)].item())[0][1])}" in selected_task_log_dict):
#                                 red_act += 1
#                             # print(f"select_action: {int(map_action_to_task_server(policy_net(state).topk(red_act, dim=1).indices[0, (red_act-1)].item())[0][0])}")
#                             # print(selected_task_log_dict)
#                             return map_action_to_task_server(
#                                 policy_net(state).topk(red_act, dim=1).indices[0, (red_act - 1)].item())
#                         else:
#                             # print(
#                             #     f"ELSE_select_action: {int(map_action_to_task_server(policy_net(state).max(1).indices.view(1, 1).item())[0][0])}")
#                             # print(f"else{selected_task_log_dict}")
#                             return map_action_to_task_server(policy_net(state).max(1).indices.view(1, 1).item())
#
#                 else:
#                     hi_from_edf += 1
#                     # Exploration: Randomly select from unassigned tasks
#                     edf_service_idx = edf_idx()
#                     # while (edf_service_idx in selected_task_log_dict):
#                     #     edf_service_idx += 1
#                     # if (edf_service_idx > 262):
#                     #     edf_service_idx = 262
#                     # if (edf_service_idx in selected_task_log_dict):
#                     #     print(f"redundant_edf_service_idx: {edf_service_idx}")
#                     # else:
#                     #     print(f"edf_service_idx: {edf_service_idx}")
#                     edf_server_idx = random.randint(1, len(servers_range_indices))
#
#                     while (f"{edf_service_idx}-{edf_server_idx}" in selected_task_log_dict):
#                         seed_edf_service_idx = [x for x in range(1, 262) if x != int(edf_service_idx)]
#                         seed_edf_server_idx = [x for x in range(1, 4) if x != int(edf_server_idx)]
#
#                         edf_service_idx = random.choice(seed_edf_service_idx)
#                         edf_server_idx = random.choice(seed_edf_server_idx)
#
#                     return torch.tensor([[edf_service_idx, edf_server_idx]], device=device, dtype=torch.long)
#
#             def optimize_model():
#                 # Placeholder: implement your optimization steps here.
#                 pass
#
#             num_episodes = 500  # or 2500 depending on device availability
#             for i_episode in range(num_episodes):
#                 local_policy_net.load_state_dict(global_policy_net.state_dict())
#                 state = torch.tensor([0, 0], dtype=torch.float32, device=device).unsqueeze(0)
#                 for t in count():
#                     action = select_action(state)
#                     # Insert your logic for action execution, reward calculation, etc.
#                     optimize_model()
#                     # For demonstration, break out of the inner loop after one iteration.
#                     break
#                 # print(f"Worker {worker_id} | Episode {i_episode} complete.")
#             # print(f"Worker {worker_id} finished execution.")
#
#     # Create global shared models using your local DQN
#     n_actions = len(Service.all()) * len(EdgeServer.all())
#     n_observations = 2  # [task, server]
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     global_policy_net = DQN(n_observations, n_actions).to(device)
#     global_target_net = DQN(n_observations, n_actions).to(device)
#     global_policy_net.share_memory()
#     global_target_net.share_memory()
#     global_target_net.load_state_dict(global_policy_net.state_dict())
#     import torch.optim as optim
#     global_optimizer = optim.AdamW(global_policy_net.parameters(), lr=LR, amsgrad=True)
#
#     # Set multiprocessing start method and spawn processes
#     import torch.multiprocessing as mp
#     mp.set_start_method('spawn', force=True)
#     processes = []
#     for i in range(NUM_PROCESSES):
#         p = mp.Process(target=WorkerLogic.run, args=(
#             i, parameters, global_policy_net, global_target_net, global_optimizer,
#             device, n_observations, n_actions, resource_tracker))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
#
#     print('Complete')

##########################################################################################################
##########################################################################################################

scheduling_time_exceeded = False
service_scheduling_duration = time.time()
old_provisioned_services = 0
def stopping_criterion(model: object):
    # Defining a variable that will help us to count the number of services successfully provisioned within the infrastructure
    provisioned_services = 0
    scheduling_time_limitation = (1 / 30) * (len(Service.all()) * len(EdgeServer.all()))
    global old_provisioned_services, service_scheduling_duration, scheduling_time_exceeded

    # Iterating over the list of services to count the number of services provisioned within the infrastructure
    for service in Service.all():
        # Initially, services are not hosted by any server (i.e., their "server" attribute is None).
        # Once that value changes, we know that it has been successfully provisioned inside an edge server.
        if service.server != None:
            provisioned_services += 1

    elapsed_time = time.time() - service_scheduling_duration
    if elapsed_time >= scheduling_time_limitation:
        scheduling_time_exceeded = True

    # This condition checks whether the service can be properly provisioned to the edge server based on
    # the decision made by the scheduling algorithm
    if (old_provisioned_services < provisioned_services):
        service_scheduling_duration = time.time()
        old_provisioned_services = provisioned_services
        print(f"{old_provisioned_services} out of {len(Service.all())} services are successfully scheduled.")
        resource_tracker.report()
        print()

    return (provisioned_services == Service.count()) or (provisioned_services == EdgeServer.is_potential_host) or (scheduling_time_exceeded == True)

##################################################
##################################################
## Determining the name of Scheduling Algorithm ##
##################################################

# Map algorithm names to functions
algorithm_functions = {
    "BestFit": Best_Fit_Service_Provisioning,
    "EDF": EDF_algorithm,
    "a_RL": a_RL,
    "D_a_RL": D_a_RL,
    "v_RL": v_RL,
}
# Define the name of the scheduling algorithm, that could be "BestFit", "EDF"
# scheduling_algorithm = "EDF"
# scheduling_algorithm = "BestFit"
# scheduling_algorithm = "v_RL"
# scheduling_algorithm = "a_RL"
scheduling_algorithm = "D_a_RL"


# @measure_memory
def wrapped_Service_Provisioning(parameters, algorithm_name=scheduling_algorithm):
    # multiprocessing.freeze_support()  # Useful for frozen executables
    # # for normal running
    # # Get the function based on the algorithm name
    # result = algorithm_functions[algorithm_name](parameters)
    # process = psutil.Process(os.getpid())
    # resource_tracker.update(process.memory_info().rss)
    # for testing RL for 31 runs
    # Get the function based on the algorithm name
    for i in range(31):
        result = algorithm_functions[algorithm_name](parameters)
        BaseStation.fluctuate_wireless_delay(BaseStation)
    process = psutil.Process(os.getpid())
    resource_tracker.update(process.memory_info().rss)
    return result

### following should be in main file ? ###
# Ensure that all process-spawning code is under this guard:
# if __name__ == '__main__':
# multiprocessing.freeze_support()  # Recommended to be here as well

logs_directory = f"logs/algorithm={scheduling_algorithm};dataset=dataset1;"  ## dataset

# Creating a Simulator object
simulator = Simulator(
    dump_interval=5,
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=wrapped_Service_Provisioning,
    logs_directory=logs_directory,
)

#########################
## Loading the dataset ##
#########################
# Get the directory of the current script
current_dir = os.path.dirname(__file__)
# Define the relative path to the dataset, using the script's directory as the base
dataset_path = os.path.join(current_dir, "dataset_generator", "datasets", "dataset1.json")
# Initialize the simulator with the absolute path
simulator.initialize(input_file=dataset_path)


# Start the timer
start_time_edgesimpy = time.time()

# Executing the simulation
simulator.run_model()

# End the timer and calculate the duration
end_time_edgesimpy = time.time()
duration_edgesimpy = end_time_edgesimpy - start_time_edgesimpy
print(f"Total runtime: {duration_edgesimpy:.2f} seconds")

resource_tracker.final_report()
file.close()