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
from nnabla.initializer import BaseInitializer
from common.distribution import Normal
from common.buffer import ReplayBuffer
from common.log import prepare_monitor
from common.experiment import evaluate, train
from common.exploration import EmptyNoise


def q_network(obs, action, name):
    with nn.parameter_scope(name):
        out = F.concatenate(obs, action, axis=1)
        out = PF.affine(out, 256, name='fc1')
        out = F.relu(out)
        out = PF.affine(out, 256, name='fc2')
        out = F.relu(out)
        return PF.affine(out, 1, name='fc3')


def policy_network(obs, action_size, name):
    with nn.parameter_scope(name):
        out = PF.affine(obs, 256, name='fc1')
        out = F.relu(out)
        out = PF.affine(out, 256, name='fc2')
        out = F.relu(out)
        mean = PF.affine(out, action_size, name='mean')
        logstd = PF.affine(out, action_size, name='logstd')
        clipped_logstd = F.clip_by_value(logstd, -20, 2)
    return Normal(mean, F.exp(clipped_logstd))


def v_network(obs, name):
    with nn.parameter_scope(name):
        out = PF.affine(obs, 256, name='fc1')
        out = F.relu(out)
        out = PF.affine(out, 256, name='fc2')
        out = F.relu(out)
        return PF.affine(out, 1, name='fc3')


# enforcing action bounds
# https://math.stackexchange.com/questions/3108216/change-of-variables-apply-tanh-to-the-gaussian-samples
def _squash_action(dist):
    raw_action = dist.sample()
    squashed_action = F.tanh(raw_action)
    jacob = 2.0 * (math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
    log_prob = F.sum(dist.log_prob(raw_action) - jacob, axis=1, keepdims=True)
    return squashed_action, log_prob


class SAC:
    def __init__(self,
                 obs_shape,
                 action_size,
                 batch_size,
                 critic_lr,
                 actor_lr,
                 value_lr,
                 tau,
                 gamma,
                 policy_reg,
                 reward_scale):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.value_lr = value_lr
        self.gamma = gamma
        self.tau = tau
        self.policy_reg = policy_reg
        self.reward_scale = reward_scale
        self._build()

    def _build(self):
        # inference graph
        self.infer_obs_t = nn.Variable((1,) + self.obs_shape)
        with nn.parameter_scope('trainable'):
            infer_dist = policy_network(self.infer_obs_t, self.action_size,
                                        'actor')
        self.infer_act_t, _ = _squash_action(infer_dist)
        self.deterministic_act_t = infer_dist.mean()

        # training graph
        self.obss_t = nn.Variable((self.batch_size,) + self.obs_shape)
        self.acts_t = nn.Variable((self.batch_size, self.action_size))
        self.rews_tp1 = nn.Variable((self.batch_size, 1))
        self.obss_tp1 = nn.Variable((self.batch_size,) + self.obs_shape)
        self.ters_tp1 = nn.Variable((self.batch_size, 1))

        with nn.parameter_scope('trainable'):
            dist = policy_network(self.obss_t, self.action_size, 'actor')
            squashed_act_t, log_prob_t = _squash_action(dist)
            v_t = v_network(self.obss_t, 'value')
            q_t1 = q_network(self.obss_t, self.acts_t, 'critic/1')
            q_t2 = q_network(self.obss_t, self.acts_t, 'critic/2')
            q_t1_with_actor = q_network(self.obss_t, squashed_act_t, 'critic/1')
            q_t2_with_actor = q_network(self.obss_t, squashed_act_t, 'critic/2')

        with nn.parameter_scope('target'):
            v_tp1 = v_network(self.obss_tp1, 'value')

        # value loss
        q_t = F.minimum2(q_t1_with_actor, q_t2_with_actor)
        v_target = q_t - log_prob_t
        v_target.need_grad = False
        self.value_loss = 0.5 * F.mean(F.squared_error(v_t, v_target))

        # q function loss
        scaled_rews_tp1 = self.rews_tp1 * self.reward_scale
        q_target = scaled_rews_tp1 + self.gamma * v_tp1 * (1.0 - self.ters_tp1)
        q_target.need_grad = False
        q1_loss = 0.5 * F.mean(F.squared_error(q_t1, q_target))
        q2_loss = 0.5 * F.mean(F.squared_error(q_t2, q_target))
        self.critic_loss = q1_loss + q2_loss

        # policy function loss
        mean_loss = 0.5 * F.mean(dist.mean() ** 2)
        logstd_loss = 0.5 * F.mean(F.log(dist.stddev()) ** 2)
        policy_reg_loss = self.policy_reg * (mean_loss + logstd_loss)
        self.objective_loss = F.mean(log_prob_t - q_t)
        self.actor_loss = self.objective_loss + policy_reg_loss

        # trainable parameters
        with nn.parameter_scope('trainable'):
            with nn.parameter_scope('value'):
                value_params = nn.get_parameters()
            with nn.parameter_scope('critic'):
                critic_params = nn.get_parameters()
            with nn.parameter_scope('actor'):
                actor_params = nn.get_parameters()
        # target parameters
        with nn.parameter_scope('target/value'):
            target_params = nn.get_parameters()

        # target update
        update_targets = []
        sync_targets = []
        for key, src in value_params.items():
            dst = target_params[key]
            updated_dst = (1.0 - self.tau) * dst + self.tau * src
            update_targets.append(F.assign(dst, updated_dst))
            sync_targets.append(F.assign(dst, src))
        self.update_target_expr = F.sink(*update_targets)
        self.sync_target_expr = F.sink(*sync_targets)

        # setup solvers
        self.value_solver = S.Adam(self.value_lr)
        self.value_solver.set_parameters(value_params)
        self.critic_solver = S.Adam(self.critic_lr)
        self.critic_solver.set_parameters(critic_params)
        self.actor_solver = S.Adam(self.actor_lr)
        self.actor_solver.set_parameters(actor_params)

    def infer(self, obs_t):
        self.infer_obs_t.d = np.array([obs_t])
        self.infer_act_t.forward(clear_buffer=True)
        return self.infer_act_t.d[0]

    def evaluate(self, obs_t):
        self.infer_obs_t.d = np.array([obs_t])
        self.deterministic_act_t.forward(clear_buffer=True)
        return np.clip(self.deterministic_act_t.d[0], -1.0, 1.0)

    def train_value(self, obss_t):
        self.obss_t.d = np.array(obss_t)
        self.value_loss.forward()
        self.value_solver.zero_grad()
        self.value_loss.backward(clear_buffer=True)
        self.value_solver.update()
        return self.value_loss.d

    def train_actor(self, obss_t):
        self.obss_t.d = np.array(obss_t)
        self.actor_loss.forward()
        loss = self.objective_loss.d.copy()
        self.actor_solver.zero_grad()
        self.actor_loss.backward(clear_buffer=True)
        self.actor_solver.update()
        return loss

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
        # train value network
        value_loss = model.train_value(obss_t)
        # train critic
        critic_loss = model.train_critic(obss_t, acts_t, rews_tp1, obss_tp1,
                                         ters_tp1)
        # train actor
        actor_loss = model.train_actor(obss_t)

        # update target parameters
        model.update_target()

        return value_loss, critic_loss, actor_loss
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
                args.critic_lr, args.actor_lr, args.value_lr, args.tau,
                args.gamma, args.policy_reg, args.reward_scale)
    model.sync_target()

    buffer = ReplayBuffer(args.buffer_size, args.batch_size)

    monitor = prepare_monitor(args.logdir)

    update_fn = update(model, buffer)

    eval_fn = evaluate(eval_env, model, render=args.render)

    train(env, model, buffer, EmptyNoise(), monitor, update_fn, eval_fn,
          args.final_step, args.batch_size, 1, args.save_interval,
          args.evaluate_interval, ['value_loss', 'critic_loss', 'actor_loss'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='sac')
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reward-scale', type=float, default=5.0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--value-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--policy-reg', type=float, default=0.001)
    parser.add_argument('--buffer-size', type=int, default=10 ** 6)
    parser.add_argument('--final-step', type=int, default=10 ** 6)
    parser.add_argument('--evaluate-interval', type=int, default=10 ** 5)
    parser.add_argument('--save-interval', type=int, default=10 ** 5)
    parser.add_argument('--load', type=str)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
