def evaluate(env, model, num_episode=10, render=False):
    def _func():
        episode = 0
        episode_rews = []
        while episode < 10:
            obs = env.reset()
            ter = False
            episode_rew = 0.0
            while not ter:
                act = model.evaluate([obs])
                obs, rew, ter, _ = env.step(act)
                episode_rew += rew
                if render:
                    env.render()
            episode_rews.append(episode_rew)
            episode += 1
        return episode_rews
    return _func
