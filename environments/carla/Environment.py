import logging
from typing import Tuple

import cv2
import numpy as np
from gym import spaces

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
logger = logging.getLogger(__name__)
from environments.carla.autonomous_agent import Agent

NUM_AGENTS = 4

# Care for the following lines in rollout_worker...
# if not isinstance(real_env, (ExternalEnv, ExternalMultiAgentEnv)):
#     logger.info(
#         "The env you specified is not a supported (sub-)type of "
#         "ExternalEnv. Attempting to convert it automatically to "
#         "ExternalEnv.")
#
#     if isinstance(real_env, MultiAgentEnv):
#         external_cls = ExternalMultiAgentEnv
#     else:
#         external_cls = ExternalEnv

CONFIG = {
    "action_space": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),  # steering, throttle (+) / brake (-)
    "observation_space": spaces.Box(low=-5,high=5,shape=(128, 128, 4),dtype=np.float32)  # RGB image from front camera
}


class carlaSimulatorInterfaceEnv(MultiAgentEnv):
    _WORKER_ID = 0

    def __init__(self, config):
        self.config = {**config, **CONFIG}
        self.action_space = self.config["action_space"]
        self.observation_space = self.config["observation_space"]
        super().__init__()
        self.dones = set()
        self.agents = []
        for i in range(NUM_AGENTS):
            self.agents.append(Agent(self.config, i))

        # Reset entire env every this number of step calls.
        self.episode_horizon = 64#config['HORIZON']  # config['ROLLOUT_FRAGMENT_LENGTH']
        # Keep track of how many times we have called `step` so far.
        self.episode_timesteps = 0
        print("WARRNING COMPLETE INITIALIZATION !!!")
        #atexit.register(self.close)
    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Performs one multi-agent step through LGLSVL

        Args:
            action_dict (dict): Multi-agent action dict

        Returns:
            tuple:
                - obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                - rewards: Rewards dict matching `obs`.
                - dones: Done dict with only an __all__ multi-agent entry in
                    it. __all__=True, if episode is done for all agents.
                - infos: An (empty) info dict.
        """
        obss = dict()
        scalarInput = dict() #ToDo use this instead if the matrix work around
        rewards = dict()
        done = dict()
        infos = dict()
        done["__all__"] = False # this is important. Because you have soft and hard dones! If __all__ is True only then the env will be resetted 

        for _id, action in action_dict.items():
            obss[_id], rewards[_id], done[_id] = self.agents[_id]._get_observation(action)
            if done[_id]:
                self.dones.add(_id)
        #print("Debug done: ",done ) # Debug done:  {'__all__': False} #Debug done:  {'__all__': False, 0: True}
        
        done["__all__"] = len(self.dones) == len(self.agents) #  ToDo: Not needed anymore. If dones list are as long as agents list then True --> Finally done["__all__"] is equal True
        return obss, rewards, done, infos


    def reset(self) -> MultiAgentDict:
        obss = dict()
        self.dones = set()
        for _id, agent in enumerate(self.agents):
            print("resetting agent:", _id)
            obss[_id] = agent.reset()
        #showImage(obss[0], "ResetScreen")
        return obss

    def close(self):
        pass

def showImage(input_data, figureName='map'):
    #cv2.imshow('map', np.uint8(input_data))
    cv2.imshow(figureName,input_data/255)
    cv2.waitKey(1)