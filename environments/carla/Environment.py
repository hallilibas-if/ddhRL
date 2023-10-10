import logging
from typing import Tuple

import cv2
import numpy as np
from gym import spaces

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
logger = logging.getLogger(__name__)
from environments.carla.autonomous_agent import Agent
from train_constants import NUM_AGENTS


CONFIG = {
    "action_space": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),  # steering, throttle (+) / brake (-)
    "observation_space": spaces.Box(low=-5,high=5,shape=(4, 4, 2049),dtype=np.float32)  # RGB image from front camera
    
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
        self.episode_horizon = 64 #config['horizon']  # config['ROLLOUT_FRAGMENT_LENGTH']
        # Keep track of how many times we have called `step` so far.
        self.episode_timesteps = 0
    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Performs one multi-agent step through CARLA

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
        obs_img = dict()
        obs_scalar = dict() #ToDo use this instead if the matrix work around
        rewards = dict()
        done = dict()
        infos = dict()
        done["__all__"] = False # this is important. Because you have soft and hard dones! If __all__ is True only then the env will be resetted 

        print("Action dictionary: ", action_dict)
        for _id, action in action_dict.items():   
            if _id<NUM_AGENTS and (_id+NUM_AGENTS) in action_dict:  # if _id is 0 or 2 
                obs_img[_id], rewards[_id], done[_id] = self.agents[_id]._get_observation_conductor(action, action_dict[_id+NUM_AGENTS]) #Action: tspeed , #action_dict[_id+NUM_AGENTS]: Steering, Acceleration
                if done[_id]:
                    self.dones.add(_id)
                
                obs_img[_id+NUM_AGENTS], rewards[_id+NUM_AGENTS], done[_id+NUM_AGENTS] = self.agents[_id]._get_observation_controller()
                if done[_id+NUM_AGENTS]:
                    self.dones.add(_id+NUM_AGENTS)
        
        done["__all__"] = len(self.dones) == len(self.agents)*2 #  ToDo: Not needed anymore. If dones list are as long as agents list then True --> Finally done["__all__"] is equal True
        return obs_img, rewards, done, infos

    def reset(self) -> MultiAgentDict:
        obs_img = dict()
        obs_scalar = dict()
        self.dones = set()
        for agent in self.agents:
            _id = agent._id
            print("resetting agent:", _id)
            obs_img[_id]= agent.reset()
            _id_next = _id+NUM_AGENTS
            print("resetting agent:", _id_next )
            obs_img[_id_next] = obs_img[_id]
        return obs_img

    def close(self):
        pass

def showImage(input_data, figureName='map'):
    #cv2.imshow('map', np.uint8(input_data))
    cv2.imshow(figureName,input_data/255)
    cv2.waitKey(1)