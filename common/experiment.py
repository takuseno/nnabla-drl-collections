import numpy as np
import nnabla as nn
import os

from nnabla.monitor import MonitorSeries, MonitorTimeElapsed


def train(env,
          model,
          buffer,
          exploration,
          monitor,
          update_fn,
          eval_fn,
          final_step,
          update_start,
          update_interval,
          save_interval,
          evaluate_interval,
          loss_labels=[]):
    reward_monitor = MonitorSeries('reward', monitor, interval=1)
    eval_reward_monitor = MonitorSeries('eval_reward', monitor, interval=1)
    time_monitor = MonitorTimeElapsed('time', monitor, interval=10000)
    loss_monitors = []
    for label in loss_labels:
        loss_monitors.append(MonitorSeries(label, monitor, interval=10000))

    step = 0
    while step <= final_step:
        obs_t = env.reset()
        ter_tp1 = False
        cumulative_reward = 0.0
        model.reset(step)
        while not ter_tp1:
            # select best action
            act_t = model.infer(obs_t)
            # add exploration noise
            act_t = exploration.get(step, act_t)
            # iterate environment
            obs_tp1, rew_tp1, ter_tp1, _ = env.step(act_t)
            # store transition
            buffer.append(obs_t, [act_t], rew_tp1, obs_tp1, ter_tp1)

            # update parameters
            if step > update_start and step % update_interval == 0:
                for i, loss in enumerate(update_fn(step)):
                    if loss is not None:
                        loss_monitors[i].add(step, loss)

            # save parameters
            if step % save_interval == 0:
                path = os.path.join(monitor.save_path, 'model_%d.h5' % step)
                nn.save_parameters(path)

            if step % evaluate_interval == 0:
                eval_reward_monitor.add(step, np.mean(eval_fn()))

            step += 1
            cumulative_reward += rew_tp1
            obs_t = obs_tp1
            time_monitor.add(step)

        # record metrics
        reward_monitor.add(step, cumulative_reward)


def evaluate(env, model, num_episode=10, render=False):
    def _func():
        episode = 0
        episode_rews = []
        while episode < num_episode:
            obs = env.reset()
            ter = False
            episode_rew = 0.0
            while not ter:
                act = model.evaluate(obs)
                obs, rew, ter, _ = env.step(act)
                episode_rew += rew
                if render:
                    env.render()
            episode_rews.append(episode_rew)
            episode += 1
        return episode_rews
    return _func
