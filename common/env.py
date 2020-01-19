import numpy as np
import cv2


def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(gray, (210, 160))
    state = cv2.resize(state, (84, 110))
    state = state[18:102, :]
    return state

    return 

class AtariWrapper:
    def __init__(self,
                 env,
                 seed,
                 limit=108000,
                 skip=4,
                 episodic=True,
                 random_start=True,
                 with_screen=False):
        self.env = env
        self.rng = np.random.RandomState(seed)
        self.obs_stack = np.zeros((4, 84, 84), dtype=np.uint8)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.limit = limit
        self.lives = 0
        self.was_real_done = True
        self.skip = skip
        self.random_start = random_start
        self.t = 0
        self.episodic = episodic
        self.with_screen = with_screen

    def _max_and_skip_frames(self, action):
        total_reward = 0.0
        done = None
        shape = (2,) + self.observation_space.shape
        obs_buffer = np.zeros(shape, dtype=np.uint8)
        for i in range(self.skip):
            obs, reward, done, info = self._step_with_episodic_life(action)

            if i == self.skip - 2:
                obs_buffer[0] = obs
            elif i == self.skip - 1:
                obs_buffer[1] = obs

            total_reward += reward

            if done:
                break
        max_frame = obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def _step_with_episodic_life(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.episodic:
            self.was_real_done = done
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives and lives > 0:
                done = True
            self.lives = lives
        return obs, reward, done, info

    def step(self, action):
        self.t += 1
        obs, reward, done, info = self._max_and_skip_frames(action)
        if self.t == self.limit:
            done = True
        self.obs_stack = np.roll(self.obs_stack, 1, axis=0)
        self.obs_stack[0] = preprocess(obs)
        info = {}
        if self.with_screen:
            info['raw_obs'] = self.env.render('rgb_array')
        return self.obs_stack.copy(), reward, done, info

    def _reset_with_episodic_life(self):
        if not self.episodic or self.was_real_done:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.step(0)
        return obs

    def reset(self):
        obs = self._reset_with_episodic_life()

        # random initialization
        if self.random_start:
            for _ in range(self.rng.randint(30)):
                obs, _, done, _ = self._step_with_episodic_life(0)
                if done:
                    obs = self._reset_with_episodic_life()

        self.lives = self.env.unwrapped.ale.lives()
        self.obs_stack.fill(0)
        self.obs_stack = np.roll(self.obs_stack, 1, axis=0)
        self.obs_stack[0] = preprocess(obs)
        self.t = 0
        return self.obs_stack.copy()

    def render(self):
        self.env.render()
