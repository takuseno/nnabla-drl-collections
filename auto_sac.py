import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.random as random
import math
import argparse
import gym

from nnabla.ext_utils import get_extension_context
from nnabla.parameter import get_parameter_or_create
from nnabla.initializer import ConstantInitializer
from common.buffer import ReplayBuffer
from common.log import prepare_monitor
from common.experiment import evaluate, train
from common.exploration import EmptyNoise
from sac import q_network, policy_network
from sac import _squash_action


class SAC:
    def __init__(self,
                 obs_shape,
                 action_size,
                 batch_size,
                 critic_lr,
                 actor_lr,
                 temp_lr,
                 tau,
                 gamma):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.temp_lr = temp_lr
        self.gamma = gamma
        self.tau = tau
        self._build()

    def _build(self):
        # inference graph
        self.infer_obs_t = nn.Variable((1,) + self.obs_shape)
        with nn.parameter_scope('trainable'):
            infer_dist = policy_network(self.infer_obs_t, self.action_size,
                                        'actor')
        self.infer_act_t, _ = _squash_action(infer_dist)

        # training graph
        self.obss_t = nn.Variable((self.batch_size,) + self.obs_shape)
        self.acts_t = nn.Variable((self.batch_size, self.action_size))
        self.rews_tp1 = nn.Variable((self.batch_size, 1))
        self.obss_tp1 = nn.Variable((self.batch_size,) + self.obs_shape)
        self.ters_tp1 = nn.Variable((self.batch_size, 1))

        with nn.parameter_scope('trainable'):
            self.log_temp = get_parameter_or_create('temp', [1, 1],
                                                    ConstantInitializer(0.0))
            dist_t = policy_network(self.obss_t, self.action_size, 'actor')
            dist_tp1 = policy_network(self.obss_tp1, self.action_size, 'actor')
            squashed_act_t, log_prob_t = _squash_action(dist_t)
            squashed_act_tp1, log_prob_tp1 = _squash_action(dist_tp1)
            q1_t = q_network(self.obss_t, self.acts_t, 'critic/1')
            q2_t = q_network(self.obss_t, self.acts_t, 'critic/2')
            q1_t_with_actor = q_network(self.obss_t, squashed_act_t, 'critic/1')
            q2_t_with_actor = q_network(self.obss_t, squashed_act_t, 'critic/2')

        with nn.parameter_scope('target'):
            q1_tp1 = q_network(self.obss_tp1, squashed_act_tp1, 'critic/1')
            q2_tp1 = q_network(self.obss_tp1, squashed_act_tp1, 'critic/2')

        # q function loss
        q_tp1 = F.minimum2(q1_tp1, q2_tp1)
        entropy_tp1 = F.exp(self.log_temp) * log_prob_tp1
        mask = (1.0 - self.ters_tp1)
        q_target = self.rews_tp1 + self.gamma * (q_tp1 - entropy_tp1) * mask
        q_target.need_grad = False
        q1_loss = 0.5 * F.mean(F.squared_error(q1_t, q_target))
        q2_loss = 0.5 * F.mean(F.squared_error(q2_t, q_target))
        self.critic_loss = q1_loss + q2_loss

        # policy function loss
        q_t = F.minimum2(q1_t_with_actor, q2_t_with_actor)
        entropy_t = F.exp(self.log_temp) * log_prob_t
        self.actor_loss = F.mean(entropy_t - q_t)

        # temperature loss
        temp_target = log_prob_t - self.action_size
        temp_target.need_grad = False
        self.temp_loss = -F.mean(F.exp(self.log_temp) * temp_target)

        # trainable parameters
        with nn.parameter_scope('trainable'):
            with nn.parameter_scope('critic'):
                critic_params = nn.get_parameters()
            with nn.parameter_scope('actor'):
                actor_params = nn.get_parameters()
        # target parameters
        with nn.parameter_scope('target/critic'):
            target_params = nn.get_parameters()

        # target update
        update_targets = []
        sync_targets = []
        for key, src in critic_params.items():
            dst = target_params[key]
            updated_dst = (1.0 - self.tau) * dst + self.tau * src
            update_targets.append(F.assign(dst, updated_dst))
            sync_targets.append(F.assign(dst, src))
        self.update_target_expr = F.sink(*update_targets)
        self.sync_target_expr = F.sink(*sync_targets)

        # setup solvers
        self.critic_solver = S.Adam(self.critic_lr)
        self.critic_solver.set_parameters(critic_params)
        self.actor_solver = S.Adam(self.actor_lr)
        self.actor_solver.set_parameters(actor_params)
        self.temp_solver = S.Adam(self.temp_lr)
        self.temp_solver.set_parameters({'temp': self.log_temp})

    def infer(self, obs_t):
        self.infer_obs_t.d = np.array([obs_t])
        self.infer_act_t.forward(clear_buffer=True)
        return np.clip(self.infer_act_t.d[0], -1.0, 1.0)

    def evaluate(self, obs_t):
        return self.infer(obs_t)

    def train_actor(self, obss_t):
        self.obss_t.d = np.array(obss_t)
        self.actor_loss.forward()
        self.actor_solver.zero_grad()
        self.actor_loss.backward(clear_buffer=True)
        self.actor_solver.update()
        return self.actor_loss.d

    def train_critic(self, obss_t, acts_t, rews_tp1, obss_tp1, ters_tp1):
        self.obss_t.d = np.array(obss_t)
        self.acts_t.d = np.array(acts_t)
        self.rews_tp1.d = np.array(rews_tp1)
        self.obss_tp1.d = np.array(obss_tp1)
        self.ters_tp1.d = np.array(ters_tp1)
        self.critic_loss.forward()
        self.critic_solver.zero_grad()
        self.critic_loss.backward(clear_buffer=True)
        self.critic_solver.update()
        return self.critic_loss.d

    def train_temp(self, obss_t):
        self.obss_t.d = np.array(obss_t)
        self.temp_loss.forward()
        self.temp_solver.zero_grad()
        self.temp_loss.backward(clear_buffer=True)
        self.temp_solver.update()
        return self.temp_loss.d

    def update_target(self):
        self.update_target_expr.forward(clear_buffer=True)

    def sync_target(self):
        self.sync_target_expr.forward(clear_buffer=True)

    def reset(self, step):
        pass

def update(model, buffer):
    def _func(step):
        experiences = buffer.sample()
        obss_t = []
        acts_t = []
        rews_tp1 = []
        obss_tp1 = []
        ters_tp1 = []
        for experience in experiences:
            obss_t.append(experience['obs_t'])
            acts_t.append(experience['act_t'][0])
            rews_tp1.append(experience['rew_tp1'])
            obss_tp1.append(experience['obs_tp1'])
            ters_tp1.append(experience['ter_tp1'])
        # train critic
        critic_loss = model.train_critic(obss_t, acts_t, rews_tp1, obss_tp1,
                                         ters_tp1)
        # train actor
        actor_loss = model.train_actor(obss_t)
        # train temp
        temp_loss = model.train_temp(obss_t)

        # update target parameters
        model.update_target()

        return critic_loss, actor_loss, temp_loss
    return _func


def main(args):
    env = gym.make(args.env)
    env.seed(args.seed)
    eval_env = gym.make(args.env)
    eval_env.seed(50)
    action_shape = env.action_space.shape

    # GPU
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id=str(args.device))
        nn.set_default_context(ctx)

    if args.load:
        nn.load_parameters(args.load)

    model = SAC(env.observation_space.shape, action_shape[0], args.batch_size,
                args.critic_lr, args.actor_lr, args.temp_lr, args.tau,
                args.gamma)
    model.sync_target()

    buffer = ReplayBuffer(args.buffer_size, args.batch_size)

    monitor = prepare_monitor(args.logdir)

    update_fn = update(model, buffer)

    eval_fn = evaluate(eval_env, model, render=args.render)

    train(env, model, buffer, EmptyNoise(), monitor, update_fn, eval_fn,
          args.final_step, args.batch_size, 1, args.save_interval,
          args.evaluate_interval, ['critic_loss', 'actor_loss', 'temp_loss'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='auto_sac')
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--temp-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--buffer-size', type=int, default=10 ** 6)
    parser.add_argument('--final-step', type=int, default=10 ** 6)
    parser.add_argument('--evaluate-interval', type=int, default=10 ** 5)
    parser.add_argument('--save-interval', type=int, default=10 ** 5)
    parser.add_argument('--load', type=str)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
