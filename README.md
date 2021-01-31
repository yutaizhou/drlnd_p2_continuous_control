Project 2 of the Udacity Deep Reinforcement Learning Nanodegree Program

**Project Details**

The envionrment is Reacher from Unity ML Agents. An agent, which is just a double-jointed arm, must move its arm around in 3D space towards a
target location, which can itself move around in 3D space. The goal of the agent is to keep the end of its arm inside the target location for as 
long as possible.
State Space: Continuous. 33 element vector describing states of the arm agent such as position, rotation, velocity, and angular velocity.

Action Space: Contiuous. 4 element vector that corresponds to the torque applicabble to both joints in order to move the arm.

Reward: +0.1 for each time step that the agent keeps the end of its arm inside the target location.

Episodic task, where each episode is 1000 steps through the environment. 

Parallelism: There are two versions of the environment, single agent and multiple agents. In the single agent setting, only one agent is 
trying to solve the task. In the multi-agent setting, 20 homogenous agents are simultaneously yet independently trying to solve the task. 
The agents are not interacting with each other in the world, and they each have their own target location to move towards. 
The agents do, however, share the same instantiation of the learning algorithm. This means a single function approximator or replay buffer is 
used to update parameters, choose actions, and store/sample experiences

The task is considered solved when all 20 agents are able to obtain an average total reward of at least 30 over 10 consecutve episodes.

**Repository Layout**
- results/: the latest run from the implemented algorithms, containing the score numpy file, plot of score, agent model, and progress log. 
- src/: source code for the agent, which is separated into code specific to agent itself, reacher environment, and utilities
- Continuous_Control.ipynb: this is simply a file copied over from the official repository. Safe to ignore.
- run.py: code run running, evaluating, and logging the agent. 

**Getting Started** 

Create a conda environment from the environment.yml file:
conda env create -f environment.yml

Activate the newly created environment:
conda activate drlnd

**Instructions**

At the time of writing (Jan 31, 2021), 1 algorithm has been implemented: DDPG.

To run DDPG without loading any weights, run the following:
```
python run.py DDPG
```
Note: unless the directory path has been changed in run.py, this will overwrite the previous results.

