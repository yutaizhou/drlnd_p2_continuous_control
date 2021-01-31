## Learning Algorithm

### Agent Architecture
The **Deep Deterministic Policy Gradient** algorithm as described in the [2015 paper][ddpg_paper] is used to solve the task. It is an actor-critic algoritm , so it has two deep neural networks (DNN), one to represent the actor, and one for the critic.

The DNN used to approximation the Q function (critic) is a simple 3 layer fully connected feedforward neural network, with a simple concatenation of the first layer representation of the state and the 4 element continuous action vector before feeding the concatenated vector into the second layer. Since the state space, |S| = 33, and action space, |A| = 4, the input size of the first layer is 33, and the input size of the second layer is 4 + the output size of the first layer after the nonlinear transformation of state. The output size of the network is 1, since the critic simply predicts the scalar Q-value of a given state-action pair.

The DNN used to approximation the policy (critic) is also a simple 3 layer fully connected feedforward neural network. With a slightly simpler architecture than the critic DNN, it has an input size of |S| = 33, and an output size of |A| = 4. 

### DDPG Theory
Although DDPG is introduced in the 2015 paper as an actor-critic algorithm, it is widely thought of as a continuous-action space extension of the [DQN](dqn_paper) algorithm. To estimate the Q-value, DQN uses the Q-learning update rule, which consists of calculating the TD (temporal difference) residual from the difference between TD target and current TD estimate. The TD target is calculated by taking the argmax over all actions of the Q-value of the next state. This argmax operator is simple to do in the discrete action case, which is the setting that Q-learning was developed in, but argmax in continuous action space is more tricky. 

To remedy this, DDPG uses an actor to represent a parametrized deterministic policy that replaces the argmax operator. For a given next state, the actor outputs a next action in the form of a |A|-dimensional vector, which is further perturbed by a stochastic process in the form of Ornstein–Uhlenbeck Process to add more randomness to the continuous action. This perturbed continuous action is then fed into the Q-value network to estimate the TD target. 

The idea of using a parametrized policy to choose the 'argmax action' is due to the fact that policy gradient algorithms are easier to work with in continuous action spaces. The [original derivation](pg_paper) of the policy gradient is for stochastic policies, but a [more recent derivation](dpg_paper) showed how to extend this to determnistic policies. Since the argmax operator is meant to be a deterministic discrete action selector (deterministic in lieu of epsilon greedy exploration), DDPG builds upon the ideas from deterministic policy gradients to use it as a deterministic continuous action selector, which makes it an effective replacement for the argmax operator to calculate TD target.

Epsilon-greedy exploration and epsilon decay were not implemented and are likely unnecessary for encouraging exploration since the Ornstein–Uhlenbeck Process already introduces randomness to the action selection. 


### Parallel Training Considerations
To make the default DDPG code work for 20 agents training in parallel, changes had to be made to the Ornstein–Uhlenbeck Process, the replay buffer, and the network updating frequency.

Ornstein–Uhlenbeck Process had to sampled independently for the action trajectory of each individual agent. This means 20 processes had to be started independently of each other. This is to break correlation in the noise process between parallel agents.

The replay buffer recives 20 interaction 5-tuples (state, action, reward, next_state, done). To add the tuples to its memory, The 20 tuples are added one at a time, rather than added once as a big batch. This is done to break correlation in time steps between parallel agents when the tuple is sampled to learn from.

For the network update, rather than updating once every time step, updates are done every 20 steps, but each update consists of 10 batches to repeatedly learn from. This means that for every 20 steps, the following are repeated 10 times:
1. Obtain one batch of experiences from the buffer
2. Update the critic by calculating MSE loss between TD target and current estimate, backprop, then SGD on critic parameters since we want to minimize the MSE loss
3. Update the actor by calculating the sampled policy gradient, backprop, then SGA on actor parameters since we want to maximize the expected return of the actions chosen by the actor. The expected return is approximated by the critic
4. Perform soft update between the target actor/critic and the local actor/critic

_Hyperparameters_
- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 256        # minibatch size
- GAMMA = 0.95            # discount factor
- LR_ACTOR = 1e-4         # learning rate of the actor 
- LR_CRITIC = 1e-3        # learning rate of the critic
- TAU = 1e-3              # for soft update of target parameters
- WEIGHT_DECAY = 0   # L2 weight decay
- TRAIN_FREQ = 20 # update net work after this many time steps

## Reward Plot

![Reward Plot][reward_plot]

The task was solved in 503 episodes.


## Future Work

DDPG could be further extended by n-step returns. The Q-learning update rule it uses calculates the TD residual using the TD target estimated by one step of actual reward. This x step of actual reward is a rather arbitrary choice, and can be easily changed as shown in the [A3C paper](a3c_paper). This concept of generalizing estimated returns is further refined in the [GAE paper](gae_paper). Both n-step return and GAE are worthwhile extensions to be considered to improve DDPG. 


DDPG is an off-policy actor-critic algorithm. Another state of the art off-policy algorithm is the [SAC](sac_paper), which is worth trying. It comes from the field of maximum entropy reinforcement learning. 

On-policy algorithms such as [PPO](ppo_paper) and [TRPO](trpo_paper) have shown major improvements in the performance since the DDPG paper, and are also worth implementing.

<!-- Links -->
[reward_plot]: https://github.com/yutaizhou/drlnd_p2_navigation/blob/master/results/DDPG/result.png
[ddpg_paper]: https://arxiv.org/abs/1509.02971
[dqn_paper]: https://www.nature.com/articles/nature14236
[pg_paper]: https://dl.acm.org/doi/10.5555/3009657.3009806
[dpg_paper]: http://proceedings.mlr.press/v32/silver14.html
[a3c_paper]: https://arxiv.org/abs/1602.01783
[gae_paper]: https://arxiv.org/abs/1506.02438
[sac_paper]: https://arxiv.org/abs/1801.01290
[trpo_paper]: https://arxiv.org/abs/1502.05477
[ppo_paper]: https://arxiv.org/abs/1707.06347

