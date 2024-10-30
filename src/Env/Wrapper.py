import gym

class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env, max_steps: int = 100) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self._max_episode_steps = max_steps

    def reset(self):
        self._elapsed_steps = 0
        return super().reset()

    def step(self, action):
        ob, rew, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info

class Wrapper(gym.Wrapper):
    def __init__(self, env, max_steps: int = 100) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self._max_episode_steps = max_steps

    def reset(self):
        self._elapsed_steps = 0
        return super().reset()

    def step(self, action):
        ob, rew, done, info = super().step(action)
        self._elapsed_steps += 1
        return ob, rew, done, info