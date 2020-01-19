import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import argparse
import gym

from nnabla.ext_utils import get_extension_context
from common.buffer import ReplayBuffer
from common.log import prepare_monitor
from common.experiment import evaluate, train
from common.exploration import OrnsteinUhlenbeckActionNoise


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
        self.obss_t = nn.Variable((batch_size,) + obs_shape)
        self.acts_t = nn.Variable((batch_size, action_size))
        self.rews_tp1 = nn.Variable((batch_size, 1))
        self.obss_tp1 = nn.Variable((batch_size,) + obs_shape)
        self.ters_tp1 = nn.Variable((batch_size, 1))

        # critic training
        with nn.parameter_scope('trainable'):
            q_t = q_network(self.obss_t, self.acts_t, 'critic')
        with nn.parameter_scope('target'):
            policy_tp1 = policy_network(
                self.obss_tp1, action_size, 'actor')
            q_tp1 = q_network(self.obss_tp1, policy_tp1, 'critic')
        y = self.rews_tp1 + gamma * q_tp1 * (1.0 - self.ters_tp1)
        self.critic_loss = F.mean(F.squared_error(q_t, y))

        # actor training
        with nn.parameter_scope('trainable'):
            policy_t = policy_network(self.obss_t, action_size, 'actor')
            q_t_with_actor = q_network(self.obss_t, policy_t, 'critic')
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

    def evaluate(self, obs_t):
        return self.infer(obs_t)

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

    def train_actor(self, obss_t):
        self.obss_t.d = np.array(obss_t)
        self.actor_loss.forward()
        self.actor_solver.zero_grad()
        self.actor_loss.backward(clear_buffer=True)
        self.actor_solver.update()
        return self.actor_loss.d

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

        # update target parameters
        model.update_target()

        return critic_loss, actor_loss
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

    model = DDPG(env.observation_space.shape, action_shape[0], args.batch_size,
                 args.critic_lr, args.actor_lr, args.tau, args.gamma)
    model.sync_target()

    noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_shape),
                                         np.ones(action_shape) * 0.2)

    buffer = ReplayBuffer(args.buffer_size, args.batch_size)

    monitor = prepare_monitor(args.logdir)

    update_fn = update(model, buffer)

    eval_fn = evaluate(eval_env, model, render=args.render)

    train(env, model, buffer, noise, monitor, update_fn, eval_fn,
          args.final_step, args.batch_size, 1, args.save_interval,
          args.evaluate_interval, ['critic_loss', 'actor_loss'])


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
    parser.add_argument('--evaluate-interval', type=int, default=10 ** 5)
    parser.add_argument('--save-interval', type=int, default=10 ** 5)
    parser.add_argument('--load', type=str)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
