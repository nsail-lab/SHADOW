from mediators.Mediator import Mediator


class TransparentMediator(Mediator):
    def reset(self, env):
        return
    def process_action(self, agent_action):
        return
    def process_observation(self, observation, info=None):
        return observation