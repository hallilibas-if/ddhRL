"""
See https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/policy/torch_policy.py
This is for PyTorch but TensorFlow is analogous
"""
from ray.rllib import TorchPolicy, TFPolicy
from CustomModel import CustomTFModel


class YourHighLevelPolicy(TFPolicy):

    def __init__(self, observation_space, action_space, config):
        your_model = CustomTFModel(observation_space, action_space,
                               num_outputs=1, model_config=config,
                               name='YourModel')
        self.action_space = action_space
        super().__init__(observation_space=observation_space,action_space=action_space,config=config)
        # if you don't pass it a model it will create one automatically


class YourLowLevelPolicy(YourHighLevelPolicy):
    pass