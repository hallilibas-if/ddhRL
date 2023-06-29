from typing import Dict

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from rllib_policy import YourHighLevelPolicy,YourLowLevelPolicy
from your_openai_spaces import high_level_obs_space, high_level_action_space, \
    low_level_obs_space,low_level_action_space
from ray.rllib.models import ModelCatalog
from CustomModel import CustomTFModel
from ray.rllib.models.tf.tf_action_dist import DiagGaussian
from environments.carla import carlaSimulatorInterfaceEnv
from train_constants import YOUR_ROOT

ModelCatalog.register_custom_model("our_model", CustomTFModel)
ModelCatalog.register_custom_action_dist("normal", DiagGaussian)

def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    """
    Maps agent_id to policy_id
    """
    if agent_id ==0:
        return 'high_level_policy'
    elif agent_id==1:
        return 'low_level_policy'
    else:
        raise RuntimeError(f'Invalid agent_id: {agent_id}')


def get_multiagent_policies() -> Dict[str,PolicySpec]:
    policies: Dict[str,PolicySpec] = {}  # policy_id to policy_spec
    config_high = {
          "model": {
            "custom_model": "our_model",
            "custom_model_config": nn_config_high,
            "custom_action_dist": "normal",
            "max_seq_len": 1,
            "dim": 42,
            "grayscale": False
          }
    }
    config_low = {
          "model": {
            "custom_model": "our_model",
            "custom_model_config": nn_config_low,
            "custom_action_dist": "normal",
            "max_seq_len": 1,
            "dim": 42,
            "grayscale": False
          }
    }
    policies['high_level_policy'] = PolicySpec(
                policy_class=None, # use default in trainer, or could be YourHighLevelPolicy
                observation_space=high_level_obs_space,
                action_space=high_level_action_space,
                config=config_high
    )

    policies['low_level_policy'] = PolicySpec(
        policy_class=None,  # use default in trainer, or could be YourLowLevelPolicy
        observation_space=low_level_obs_space,
        action_space=low_level_action_space,
        config=config_low
    )

    return policies


# see https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/agents/trainer.py
nn_config_high = {
        "NUM_DPC_FRAMES": 4,
        "FREEZE_CONV_LAYERS": True,
        "SHARED_CNN_LAYERS": 0,
        "ACTOR_CNN_LAYERS": 1,
        "ACTOR_CNN_FILTERS": [128, 64],
        "ACTOR_CNN_KERNEL_SIZE": [3, 3, 3],
        "ACTOR_CNN_STRIDES": [2, 1, 1],
        "ACTOR_FC_LAYERS": 2,
        "ACTOR_FC_UNITS" : [512, 128],
        "ACTOR_CNN_GRU_LAYERS": [0, 0, 0],
        "ACTOR_FC_GRU_LAYERS": [0, 0, 0],
        "ACTOR_GRU_STATE_SIZES": [512, 256, 128],
        "CRITIC_CNN_LAYERS": 1,
        "CRITIC_CNN_FILTERS": [128, 64],
        "CRITIC_CNN_KERNEL_SIZE": [3, 3, 3],
        "CRITIC_CNN_STRIDES": [2, 1, 1],
        "CRITIC_FC_LAYERS": 2,
        "CRITIC_FC_UNITS" : [512, 128],
        "CRITIC_CNN_GRU_LAYERS": [0, 0, 0],
        "CRITIC_FC_GRU_LAYERS": [0, 0, 0],
        "CRITIC_GRU_STATE_SIZES": [512, 256, 128],
        "OUTPUTS": 1,
        "REGULARIZER" : "l1_l2",
        "INITIALIZER" : "he_normal"
}

nn_config_low = {
        "NUM_DPC_FRAMES": 4,
        "FREEZE_CONV_LAYERS": True,
        "SHARED_CNN_LAYERS": 0,
        "ACTOR_CNN_LAYERS": 1,
        "ACTOR_CNN_FILTERS": [128, 64],
        "ACTOR_CNN_KERNEL_SIZE": [3, 3, 3],
        "ACTOR_CNN_STRIDES": [2, 1, 1],
        "ACTOR_FC_LAYERS": 2,
        "ACTOR_FC_UNITS" : [512, 128],
        "ACTOR_CNN_GRU_LAYERS": [0, 0, 0],
        "ACTOR_FC_GRU_LAYERS": [0, 0, 0],
        "ACTOR_GRU_STATE_SIZES": [512, 256, 128],
        "CRITIC_CNN_LAYERS": 1,
        "CRITIC_CNN_FILTERS": [128, 64],
        "CRITIC_CNN_KERNEL_SIZE": [3, 3, 3],
        "CRITIC_CNN_STRIDES": [2, 1, 1],
        "CRITIC_FC_LAYERS": 2,
        "CRITIC_FC_UNITS" : [512, 128],
        "CRITIC_CNN_GRU_LAYERS": [0, 0, 0],
        "CRITIC_FC_GRU_LAYERS": [0, 0, 0],
        "CRITIC_GRU_STATE_SIZES": [512, 256, 128],
        "OUTPUTS": 1,
        "REGULARIZER" : "l1_l2",
        "INITIALIZER" : "he_normal"
}


policies = get_multiagent_policies()
cust_config = {
        "simple_optimizer": True,
        "ignore_worker_failures": True,
        "batch_mode": "complete_episodes",
        "env": carlaSimulatorInterfaceEnv,
        "env_config": {
            "OUTPUTS": 1,
            "experiment_path": YOUR_ROOT
        },
        "framework": "tf2",
        "eager_tracing": False,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_map_fn,
            "policies_to_train": list(policies.keys()),
            "count_steps_by": "env_steps",
            "observation_fn": None,
            "replay_mode": "independent",
            "policy_map_cache": None,
            "policy_map_capacity": 100,
        },
    }