# NNabla DRL Collections
Single file deep reinforcement learning implementations with NNabla.
All algorithms are written in 300 lines.
As each file does not have any dependencies among this repository, you can run all the algorithms just by copying one of them to anywhere you want.

## install
```
$ pip install -r requirements.txt
```
If you use GPU, see [here](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).

## algorithms (discrete action-space)
- [x] [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236)
- [x] [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [x] [Categorical DQN](https://arxiv.org/abs/1707.06887)
- [x] [NoisyNet-DQN](https://arxiv.org/abs/1706.10295)
- [ ] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

## algorithms (continuous action-space)
- [x] [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971)
- [x] [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/abs/1802.09477)
- [ ] [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)
- [ ] [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)

For `SAC` and `PPO`, I'm waiting for [my PR](https://github.com/sony/nnabla/pull/392) to be merged in order to implement multivariate normal distribution.

## blog posts
- [Deep Q-Network Implementation with SONY’s NNabla](https://towardsdatascience.com/deep-q-network-implementation-with-sonys-nnabla-490d945deb8e)
