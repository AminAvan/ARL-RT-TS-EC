
---
# Overview of aRL
Task scheduling for soft real-time applications in edge computing presents significant challenges due to dynamic environments, large search spaces, and multiple conflicting objectives. Traditional heuristic & metaheuristic task schedulers often struggle to adapt to these conditions, motivating the adoption of reinforcement learning (RL). However, RL is often hindered by prolonged training times, primarily due to extensive exploration of irrelevant actions.

This GitHub repository provides the implementation of Agile Reinforcement Learning (aRL), an enhanced reinforcement learning approach that integrates informed exploration and action masking to prioritize relevant actions. These improvements enhance predictability, accelerate adaptation, and facilitate faster convergence.

Please find the aRL slides [here](CanadianAI2025_AminAvan.pptx), and aRL paper [here](https://caiac.pubpub.org/pub/xvm5a604).

The original repository of EdgeSimPy can be found [here](https://github.com/EdgeSimPy/EdgeSimPy). We have added features to the original source code, developing it into a framework for testing and experimenting with scheduling algorithms for real-time applications in edge computing.

---
# Agile reinforcement learning (aRL)
The primary objective is to minimize the runtime of the task scheduling algorithm when generating the task schedule while maximizing the hit-ratio of the resulting task schedule. Consequently, the task scheduling problem is formulated as an MDP, in which an action is represented as a two-element list, with the first element denoting the task and the second element representing the edge server.

An RL-agent receives positive rewards to satisfy each of the constraints, and negative rewards for each violation. Accordingly, the objective of the RL-agent is to identify a policy that maximizes cumulative rewards. aRL employs informed exploration and action masking to accelerate learning and reduce learning time.


* Action Masking: A decision matrix (G), defined in Eq. (4.2), tracks the assignment decisions of the RL-agent during each episode of the learning phase. At the beginning of each episode, all cells in the decision matrix (G) are initialized to '0'. When the RL-agent decides to assign a specific task to a particular edge server, the corresponding cell is updated to '1'. In decision matrix (G), columns represent edge servers and rows denote tasks.
  * 1st mechanism to mask the actions of RL-agent and facilitate the execution of relevant actions, an `action bound' inspired by a finite-horizon concept is introduced. Under the action bound, the number of actions per episode is limited by the total number of tasks. This restriction mitigates excessively long rollouts and ensures that each episode terminates within a specified timeframe, thereby enabling more reliable policy and value function updates. Therefore, an episode terminates either when the total number of actions equals the total number of tasks or when all tasks are assigned to edge servers while satisfying the constraints.
  * 2nd mechanism, referred to as the `single-assignment constraint', restricts the RL-agent to one assignment decision per task. After each action in an episode, the total number of possible actions (i.e., unassigned tasks) decreases, thereby reducing the state-action space over the course of the episode. In addition, the single-assignment constraint prevents the RL-agent from reassigning tasks that have already been assigned and directs its focus to the set of unassigned tasks.
* Informed Exploration: In exploration mode, aRL explores the action-state space using a modified version of the Earliest Deadline First (EDF) algorithm, adapted for edge computing, rather than relying solely on random exploration. As a result, aRL selects a task with the earliest deadline among unassigned tasks to be assigned to an available edge server subject to the conditions specified in Eqs. (3.11) and (3.17).
---
## Required Packages
All packages required by EdgeSimPy are listed in the [pyproject.toml](pyproject.toml) file, which should be installed using [Poetry](https://python-poetry.org/) in Python.

## Build Instructions
Assuming you have already installed a recent version of [Poetry](https://python-poetry.org/) ([how to install](https://python-poetry.org/docs/)) and your Python version is ```3.10.x```, you can install the required packages for EdgeSimPy using the following command:
```bash
poetry install
poetry shell
```
After completing the above step, you need to change the 'Python Interpreter' in your IDE to point to the 'Poetry Environment' location. The path was output by `poetry shell`, and you need to append `\Scripts\python.exe` to the end, resulting in a path like `[output from poetry shell]\Scripts\python.exe`.


## Selecting & Running Scheduling Algorithms
In the [main](edge_sim_py/__main__.py) file, the `algorithm_functions` function poses all available scheduling algorithms. Set your preferred algorithm name as the value of the `scheduling_algorithm` variable, located just after this function.

After selecting your preferred scheduling algorithm, run [main](edge_sim_py/__main__.py). While the file is running, the following information will be printed regularly as an online report of the scheduling process until completion.
```
'...' out of '...' services are successfully scheduled.
runtime: '...' seconds
memory consumption: '...' MB until '...' seconds
power consumption: '...' Watt-seconds until '...' seconds
```
* `runtime`: the time it takes for the algorithm to generate an optimal/near-optimal schedule for `'...' out of '...' services` based on service attributes & edge server resources, plus the time required to provision these services according to the schedule.
* `memory consumption`: the memory used by the algorithm during scheduling & provisioning for `'...' out of '...' services`.
* `power consumption`: the power consumed by the algorithm during scheduling & provisioning for `'...' out of '...' services`.

Upon completion of the scheduling process, the following information will be displayed:
```
Total runtime: '...' seconds
Total memory consumption: '...' MB
Total power consumption: '...' Watt-seconds
```

## Dataset
We modify & introduce new parameters in the [dataset generator](edge_sim_py/dataset_generator/create_dataset.py) to generate a dataset incorporating real-time parameters specific to the services within edge user applications.

This [dataset](edge_sim_py/dataset_generator/datasets/dataset1.json) simulates a video surveillance scenario with four applications:
1) Crowd counting
   2) services: alpine, python, nginx, redis, mobilenetssd.
2) Face recognition
   3) services: ubuntu, python, envoy, aerospike, yolov8.
3) Machine learning model development for crowd counting
   4) services: ubuntu, python, envoy, kafka, pytorch, mobilenetssd.
4) Machine learning model development for face recognition
   5) services: ubuntu, python, envoy, kafka, tensorflow, yolov8.

The data from [container_images](edge_sim_py/dataset_generator/container_images.json) defines the required disk space and container image layers, which are utilized during the provisioning process as part of the algorithm's `runtime`.

In [dataset](edge_sim_py/dataset_generator/datasets/dataset1.json), all services are characterized by a list of dictionaries named `service_demand_values`, which each dic includes the following information:
* `"label"`: name of service
* `"cpu"`: we set this variable to `1` for all services, as it was previously used to indicate CPU demand in terms of the number of cores. However, we now determine the CPU demand of a service based on the processor cycles needed per second for processing `1MB` of its data.
* `"memory: y"`: amount of memory (RAM) required by a service to operate.
* `"cpu_cycles_demand": (x * y)"`: The total required processor cycles are calculated by multiplying `x` (the cycles needed per second to process `1MB` of data) by `y`.

The distribution of application types & their respective services in [dataset](edge_sim_py/dataset_generator/datasets/dataset1.json) is detailed below. Each time a dataset is generated using [dataset generator](edge_sim_py/dataset_generator/create_dataset.py), the following information will be printed:
```
==== INFRASTRUCTURE OCCUPATION OVERVIEW ====
Edge Servers: 4
	Total CPU computational capacity: 61912000000.0 Hz
	Total RAM capacity: 49152 MB

Edge users: 52
	crowd counting [Critical Sensitivity]: 44 (84.62%)
	face recognition [High Sensitivity]: 6 (11.54%)
	crowd counting ml dev [Moderate Sensitivity]: 1 (1.92%)
	face recognition ml dev [Low Sensitivity]: 1 (1.92%)

Services of 52 edge users: 262
	Total CPU Cycles Demands: 9670960000
		[Critical Sensitivity]: 1924560000 (19.9%)
		[High Sensitivity]: 1315800000 (13.61%)
		[Moderate Sensitivity]: 3135550000 (32.42%)
		[Low Sensitivity]: 3295050000 (34.07%)
	Total RAM Demand: 45138 MB
		[Critical Sensitivity]: 30360 (67.26%)
		[High Sensitivity]: 9660 (21.4%)
		[Moderate Sensitivity]: 2384 (5.28%)
		[Low Sensitivity]: 2734 (6.06%)
```

## Monitoring

We can monitor the entity's state at the end of each time step of EdgeSimPy using [monitoring](edge_sim_py/monitoring.py). Simulation logs are stored in MessagePack, and you can customize which entity metrics are monitored at each time step by overriding the entity's `collect()` method.

Based on the current [monitoring](edge_sim_py/monitoring.py) code, various details about each edge server can be obtained, including its coordinates, its availability, RAM capacity, disk capacity, processor demand (based on cycles demand), memory demand, storage demand, and the IDs of services hosted on the server.

In addition, the current [monitoring](edge_sim_py/monitoring.py) provides details about each service, including its ID, availability status, associated application ID, the ID of the hosting edge-server, and migration status.

Finally, the current [monitoring](edge_sim_py/monitoring.py) prints `'m' out of 'M' services are experienced failures (missed/lost/failed), potentially affecting 'n' of the 'N' users.`