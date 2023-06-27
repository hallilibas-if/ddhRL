from typing import Dict

from ray.rllib.policy.policy import PolicySpec
from CARLA_environment import YourEnvironment
from rllib_policy import YourHighLevelPolicy,YourLowLevelPolicy
from your_openai_spaces import high_level_obs_space, high_level_action_space, \
    low_level_obs_space,low_level_action_space


def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    """
    Maps agent_id to policy_id
    """
    if 'high' in agent_id:
        return 'high_level_policy'
    elif 'low' in agent_id:
        return 'low_level_policy'
    else:
        raise RuntimeError(f'Invalid agent_id: {agent_id}')


def get_multiagent_policies() -> Dict[str,PolicySpec]:
    policies: Dict[str,PolicySpec] = {}  # policy_id to policy_spec

    policies['high_level_policy'] = PolicySpec(
                policy_class=YourHighLevelPolicy, # use default in trainer, or could be YourHighLevelPolicy
                observation_space=high_level_obs_space,
                action_space=high_level_action_space,
                config={}
    )

    policies['low_level_policy'] = PolicySpec(
        policy_class=YourLowLevelPolicy,  # use default in trainer, or could be YourLowLevelPolicy
        observation_space=low_level_obs_space,
        action_space=low_level_action_space,
        config={}
    )

    return policies


policies = get_multiagent_policies()

# see https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/agents/trainer.py
cust_config = {
        #"env": "logan_env",
        "simple_optimizer": True,
        "ignore_worker_failures": True,
        "batch_mode": "complete_episodes",
        "env": YourEnvironment,
        "env_config": {
            "is_use_visualization": False,
        },
        "framework": "tf2",
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

nn_config = {
        #"env": "logan_env",
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
        "NUM_DPC_FRAMES": 5,
        "OUTPUTS": 2,
        "REGULARIZER" : "l1_l2",
        "INITIALIZER" : "he_normal"
    }