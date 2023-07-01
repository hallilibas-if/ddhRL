"""
You pick a chicken to befriend and then get the location of that chicken

A couple of articles on hierarchical reinforcement learning
https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/
https://towardsdatascience.com/hierarchical-reinforcement-learning-a2cca9b76097

"""
import numpy as np

from gym import spaces

from train_constants import NUM_OCEAN_FEATURES, NUM_CHICKENS, NUM_DIRECTIONS


#************************************************************
# Observation spaces
#************************************************************

# 10 chickens follow the OCEAN model of personality
# I think it has no less validity here than in many applications
high_level_obs_space = spaces.Box(
            low=-5,
            high=5,
            shape=(128, 128, 4),
            dtype=np.float32
        )  # RGB image from front camera

# At the low level, you don't care about OCEAN, you've already chosen your chicken
low_level_obs_space =  spaces.Box(
            low=-5,
            high=5,
            shape=(128, 128, 4),
            dtype=np.float32
        )  # RGB image from front camera


#************************************************************
# Action spaces
#************************************************************

# Which chicken to choose
high_level_action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )  # RGB image from front camera

# go 8 directions N, NE, E, SE, S, SW, W, NW
low_level_action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )  # RGB image from front camera