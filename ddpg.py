import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import argparse
import random
import os
import gym

from datetime import datetime
from collections import deque
from nnabla.monitor import Monitor, MonitorSeries
from nnabla.ext_utils import get_extension_context


#------------------------------- neural network ------------------------------#
def q_network(obs, action, name):
    with nn.parameter_scope(name):
        out = PF.affine(obs, 64, name='fc1')
        out = F.tanh(out)
        out = F.concatenate(out, action, axis=1)
        out = PF.affine(out, 64, name='fc2')
        out = F.tanh(out)
        out = PF.affine(out, 1, name='fc3')
    return out


def policy_network(obs, action_size, name):
    with nn.parameter_scope(name):
        out = PF.affine(obs, 64, name='fc1')
        out = F.tanh(out)
        out = PF.affine(out, 64, name='fc2')
        out = F.tanh(out)
        out = PF.affine(out, action_size, name='fc3')
    return F.tanh(out)
#-----------------------------------------------------------------------------#

#-------------------------- DDPG algorithm -----------------------------------#
class DDPG:
    def __init__(self,
                 obs_shape,
                 action_size,
                 batch_size,
                 critic_lr,
                 actor_lr,
                 tau,
                 gamma):
        # inference
        self.infer_obs_t = nn.Variable((1,) + obs_shape)

        with nn.parameter_scope('trainable'):
            self.infer_policy_t = policy_network(
                self.infer_obs_t, action_size, 'actor')

        # training
        self.obs_t = nn.Variable((batch_size,) + obs_shape)
        self.actions_t = nn.Variable((batch_size, action_size))
        self.rewards_tp1 = nn.Variable((batch_size, 1))
        self.obs_tp1 = nn.Variable((batch_size,) + obs_shape)
        self.dones_tp1 = nn.Variable((batch_size, 1))

        # critic training
        with nn.parameter_scope('trainable'):
            q_t = q_network(self.obs_t, self.actions_t, 'critic')
        with nn.parameter_scope('target'):
            policy_tp1 = policy_network(
                self.obs_tp1, action_size, 'actor')
            q_tp1 = q_network(self.obs_tp1, policy_tp1, 'critic')
        y = self.rewards_tp1 + gamma * q_tp1 * (1.0 - self.dones_tp1)
        # stop backpropation to the target to prevent unncessary calculation
        unlinked_y = y.get_unlinked_variable(need_grad=False)
        self.critic_loss = F.mean(F.squared_error(q_t, unlinked_y))

        # actor training
        with nn.parameter_scope('trainable'):
            policy_t = policy_network(self.obs_t, action_size, 'actor')
            q_t_with_actor = q_network(self.obs_t, policy_t, 'critic')
        self.actor_loss = -F.mean(q_t_with_actor)

        # get neural network parameters
        with nn.parameter_scope('trainable'):
            with nn.parameter_scope('critic'):
                critic_params = nn.get_parameters()
            with nn.parameter_scope('actor'):
                actor_params = nn.get_parameters()

        # setup optimizers
        self.critic_solver = S.Adam(critic_lr)
        self.critic_solver.set_parameters(critic_params)
        self.actor_solver = S.Adam(actor_lr)
        self.actor_solver.set_parameters(actor_params)

        with nn.parameter_scope('trainable'):
            trainable_params = nn.get_parameters()
        with nn.parameter_scope('target'):
            target_params = nn.get_parameters()

        # build target update
        update_targets = []
        sync_targets = []
        for key, src in trainable_params.items():
            dst = target_params[key]
            update_targets.append(F.assign(dst, (1.0 - tau) * dst + tau * src))
            sync_targets.append(F.assign(dst, src))
        self.update_target_expr = F.sink(*update_targets)
        self.sync_target_expr = F.sink(*sync_targets)

    def infer(self, obs_t):
        self.infer_obs_t.d = np.array([obs_t])
        self.infer_policy_t.forward(clear_buffer=True)
        return self.infer_policy_t.d[0]

    def train_critic(self, obs_t, actions_t, rewards_tp1, obs_tp1, dones_tp1):
        self.obs_t.d = np.array(obs_t)
        self.actions_t.d = np.array(actions_t)
        self.rewards_tp1.d = np.array(rewards_tp1)
        self.obs_tp1.d = np.array(obs_tp1)
        self.dones_tp1.d = np.array(dones_tp1)
        self.critic_loss.forward()
        self.critic_solver.zero_grad()
        self.critic_loss.backward(clear_buffer=True)
        self.critic_solver.update()
        return self.critic_loss.d

    def train_actor(self, obs_t):
        self.obs_t.d = np.array(obs_t)
        self.actor_loss.forward()
        self.actor_solver.zero_grad()
        self.actor_loss.backward(clear_buffer=True)
        self.actor_solver.update()
        return self.actor_loss.d

    def update_target(self):
        self.update_target_expr.forward(clear_buffer=True)

    def sync_target(self):
        self.sync_target_expr.forward(clear_buffer=True)
#-----------------------------------------------------------------------------#

#----------------------------- replay buffer ---------------------------------#
class ReplayBuffer:
    def __init__(self, maxlen, batch_size):
        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size

    def append(self, obs_t, action_t, reward_tp1, obs_tp1, done_tp1):
        experience = dict(obs_t=obs_t, action_t=action_t,
                          reward_tp1=[reward_tp1], obs_tp1=obs_tp1,
                          done_tp1=[done_tp1])
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def size(self):
        return len(self.buffer)
#-----------------------------------------------------------------------------#

#---------------------------- exploration noise ------------------------------#
# from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mean, sigma, theta=.15, time=1e-2, init_x=None):
        self.theta = theta
        self.mean = mean
        self.sigma = sigma
        self.time = time
        self.init_x = init_x
        self.prev_x = None
        self.reset()

    def __call__(self):
        normal = np.random.normal(size=self.mean.shape)
        new_x = self.prev_x + self.theta * (self.mean - self.prev_x) \
            * self.time + self.sigma * np.sqrt(self.time) * normal
        self.prev_x = new_x
        return new_x

    def reset(self):
        if self.init_x is not None:
            self.prev_x = self.init_x
        else:
            self.prev_x = np.zeros_like(self.mean)
#-----------------------------------------------------------------------------#

#------------------------------ training loop --------------------------------#
def train(model, buffer):
    experiences = buffer.sample()
    obs_t = []
    actions_t = []
    rewards_tp1 = []
    obs_tp1 = []
    dones_tp1 = []
    for experience in experiences:
        obs_t.append(experience['obs_t'])
        actions_t.append(experience['action_t'])
        rewards_tp1.append(experience['reward_tp1'])
        obs_tp1.append(experience['obs_tp1'])
        dones_tp1.append(experience['done_tp1'])
    # train critic
    critic_loss = model.train_critic(
        obs_t, actions_t, rewards_tp1, obs_tp1, dones_tp1)
    # train actor
    actor_loss = model.train_actor(obs_t)
    # update target
    model.update_target()
    return critic_loss, actor_loss

def train_loop(env, model, noise, buffer, logdir, final_step, render):
    # monitors
    monitor = Monitor(logdir)
    actor_loss_monitor = MonitorSeries('actor_loss', monitor, interval=100)
    critic_loss_monitor = MonitorSeries('critic_loss', monitor, interval=100)
    reward_monitor = MonitorSeries('reward', monitor, interval=1)
    # copy parameters to target networks
    model.sync_target()

    step = 0
    while step < final_step:
        noise.reset()
        obs_t = env.reset()
        done = False
        cumulative_reward = 0.0
        while not done:
            # infer action
            action_t = np.clip(model.infer(obs_t) + noise(), -1.0, 1.0)
            # move environment
            obs_tp1, reward_tp1, done, _ = env.step(action_t)
            # append transition
            buffer.append(obs_t, action_t, reward_tp1, obs_tp1, done)

            # train
            if buffer.size() > buffer.batch_size:
                critic_loss, actor_loss = train(model, buffer)
                critic_loss_monitor.add(step, critic_loss)
                actor_loss_monitor.add(step, actor_loss)

            if step % 100000 == 0:
                path = os.path.join(logdir, 'model_{}.h5'.format(step))
                nn.save_parameters(path)

            if render:
                env.render()

            obs_t = obs_tp1
            step += 1
            cumulative_reward += reward_tp1
        reward_monitor.add(step, cumulative_reward)
#-----------------------------------------------------------------------------#

def main(args):
    env = gym.make(args.env)
    env.seed(args.seed)
    action_shape = env.action_space.shape

    # GPU
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id=str(args.device))
        nn.set_default_context(ctx)

    if args.load:
        nn.load_parameters(args.load)

    model = DDPG(env.observation_space.shape, action_shape[0], args.batch_size,
                 args.critic_lr, args.actor_lr, args.tau, args.gamma)

    noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_shape),
                                         np.ones(action_shape) * 0.2)

    buffer = ReplayBuffer(args.buffer_size, args.batch_size)

    # set log directory
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    logdir = os.path.join('logs', args.logdir + '_' + date)
    if os.path.exists(logdir):
        os.makedirs(logdir)

    train_loop(env, model, noise, buffer, logdir, args.final_step, args.render)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='ddpg')
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--buffer-size', type=int, default=10 ** 5)
    parser.add_argument('--final-step', type=int, default=10 ** 6)
    parser.add_argument('--load', type=str)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
