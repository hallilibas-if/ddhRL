"""
A couple of articles on hierarchical reinforcement learning
https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/
https://towardsdatascience.com/hierarchical-reinforcement-learning-a2cca9b76097

"""
import numpy as np
from gym import spaces
#************************************************************
# Observation spaces
#************************************************************

high_level_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4, 4, 2049),
            dtype=np.float32
        )  # RGB image from front camera

# At the low level, you don't care about OCEAN, you've already chosen your chicken
low_level_obs_space =  spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4, 4, 2049),
            dtype=np.float32
        )  # RGB image from front camera


#************************************************************
# Action spaces
#************************************************************

high_level_action_space = spaces.Box(
            low=0,
            high=20,
            shape=(1,),
            dtype=np.float32
        )  


low_level_action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        ) 