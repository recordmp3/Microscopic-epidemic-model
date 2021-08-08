## Requirements
python >= 3.7
pytorch >= 1.7.0
numpy >= 1.19
cython == 0.29.21
gym == 0.18.0
matplotlib == 3.3.2
baselines == 0.1.6
a g++ complier compatible with C++11 standard

## Code structure

### Microscopic Pandemic Simulator (MPS) 
**env2.pyx**,**EnvCWrapper.pxd**: Cython code; define the interface of MPS with python training module.
**environment.cpp**: The MPS implementation, which receives agents' action as input and returns observations and rewards.
**config.cpp**,**config.h**: Define the useful mappings, important hyperparameters and the setup with respect to the dataset for MPS.
**facility.cpp**,**facility.h**: Define the facility object with crucial properties such as capacity and types. 
**individual.cpp**,**individual.h**: Define the individual object with crucial properties such as health state, supply level, affiliation to facility and age.
**city.cpp**,**city.h**: Define the city object, which serves as a global counter for agents in different health states.

### Scalable Multi-Agent DQN (SMADQN)
**a2c_ppo_acktr/envs**: Calls baselines function to wrap the pyx class into a gym-compatible environment. Other files under a2c_ppo_acktr folder are deprecated.
**DQN.py**,**Storage.py**: The algorithm (and buffer) implementation of SMADQN.
**config.py**: Contains hyperparameters for RL training.


### Scripts and Main Code
**main.py**: Main code that integrates the training and simulation pipeline.
**utils.py**: Useful functions for python code.
**draw.py**: Visualization in python using debugging information from the C++ code of MPS. 
**build.py**: Script for Cython compilation.
**mkfl.sh**: This script calls for the compilation the C++ code and runs main.py; run this script for experiments.
### Dataset
**graph_sorted.txt**: The dataset for experiment.

## How to Obtain Results for Each Experiment
The hyperparameters controlling the current type of experiment are listed at the beginning of **environment.cpp**; alter these to get different results.
After modifying the code, run **mkfl.sh** to start the execution.