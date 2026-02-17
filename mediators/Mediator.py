from abc import ABC, abstractmethod

class Mediator(ABC):
    @abstractmethod
    def reset(self, env):
        pass

    @abstractmethod
    def process_action(self, agent_action):
        pass

    @abstractmethod
    def process_observation(self, agent_state, info=None):
        pass