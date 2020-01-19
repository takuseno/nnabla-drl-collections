# NNabla DRL Collections
Single file deep reinforcement learning implementations with NNabla.
All algorithms are written in 300 lines.

## install
```
$ pip install -r requirements.txt
```
If you use GPU, see [here](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).

## algorithms (discrete action-space)
- [x] [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236)
- [x] [Double DQN](https://arxiv.org/abs/1509.06461)
- [x] [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [x] [NoisyNet-DQN](https://arxiv.org/abs/1706.10295)
- [x] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) (Not tested)
- [x] [Categorical DQN](https://arxiv.org/abs/1707.06887)
- [x] [Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [ ] [Rainbow](https://arxiv.org/abs/1710.02298)
- [ ] [Quantile Regression DQN (QR DQN)](https://arxiv.org/abs/1710.10044)
- [ ] [Implicit Quantile Networks (IQN)](https://arxiv.org/abs/1806.06923)
- [x] [Advantage Actor-Critic (A2C)](https://arxiv.org/abs/1602.01783)
- [ ] [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)

## algorithms (continuous action-space)
- [x] [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971)
- [x] [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/abs/1802.09477)
- [ ] [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)
- [ ] [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [ ] [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [ ] [Actor-Critic with Experience Replay (ACER)](https://arxiv.org/abs/1611.01224)

For `SAC` and `PPO`, I'm waiting for [my PR](https://github.com/sony/nnabla/pull/392) to be merged in order to implement multivariate normal distribution.

## blog posts
- [Deep Q-Network Implementation with SONY’s NNabla](https://towardsdatascience.com/deep-q-network-implementation-with-sonys-nnabla-490d945deb8e)
