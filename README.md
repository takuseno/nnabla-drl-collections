# NNabla DRL Collections
Single file deep reinforcement learning implementations with NNabla.
All algorithms are written in 300 lines.

## install
```
$ pip install -r requirements.txt
```
If you use GPU, see [here](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).

## algorithms
- [x] Deep Q-Network (DQN)
- [x] Deep Deterministic Policy Gradients (DDPG)
- [x] Twin Delayed Deep Deterministic Policy Gradients (TD3)
- [ ] Soft Actor-Critic (SAC)
- [ ] Proximal Policy Optimization (PPO)

For `SAC` and `PPO`, I'm waiting for [my PR](https://github.com/sony/nnabla/pull/392) to be merged in order to implement multivariate normal distribution.

## blog posts
- [Deep Q-Network Implementation with SONY’s NNabla](https://towardsdatascience.com/deep-q-network-implementation-with-sonys-nnabla-490d945deb8e)
