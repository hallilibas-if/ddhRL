"""
This trainer is based off of PPO and points you in the direction of how to
customize it.

PPO is as a policy gradient method called Proximal Policy Optimization.
Policy gradient means you update the parameters of the policy directly instead of using
a Q table or something like that. Good for continuous actions.
"""
from typing import Optional, Type

from ray.rllib.policy.policy import Policy
from ray.rllib.agents.ppo import PPOTorchPolicy, PPOTrainer, PPOTF2Policy
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.agents.ppo import ppo
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict
from utils.train_constants import RESTORE_PATH
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# PPO default config builds on DEFAULT_CONFIG here
# https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/agents/trainer.py
config = ppo.DEFAULT_CONFIG.copy()

# Further update with config
config.update(
    {
    # The batch size collected for each worker
    "rollout_fragment_length": 64,
    "num_cpus_per_worker": 3,
    # Can be "complete_episodes" or "truncate_episodes"
    "batch_mode": "truncate_episodes",
    "horizon": 64,
    "simple_optimizer": True,
    "framework": "tf2",
    "train_batch_size" : 640,
    "disable_env_checking":True,
    "num_sgd_iter" : 3,
    "sgd_minibatch_size": 128,
    "lr": 0.000005,
    "no_done_at_end": False,
    "preprocessor_pref": "deepmind",
    "gamma" : 0.9,
    "lambda" : 0.95,
    "epsilon" : 0.1,
    "entropy_coeff" : 0.0,
    "kl_target" : 0.03,
    "kl_coeff" : 0.3
    })

if RESTORE_PATH != "" or RESTORE_PATH != None:
    config.update({
        "kl_coeff" : 0.03
    })

class YourTrainer(PPOTrainer):
    @classmethod
    @override(PPOTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return config

    @staticmethod
    @override(PPOTrainer)
    def execution_plan(workers, config, **kwargs):
        # Execution plans are a little in the weeds. They control what exactly
        # the workers do.
        assert len(kwargs) == 0, "Dummy execution_plan takes no parameters"
        rollouts = ParallelRollouts(workers, mode="bulk_sync")
        train_op = rollouts.combine(
            ConcatBatches(
                min_batch_size=config["train_batch_size"],
                count_steps_by=config["multiagent"]["count_steps_by"],
            )).for_each(
            TrainOneStep(workers, num_sgd_iter=config["num_sgd_iter"]))
        return StandardMetricsReporting(train_op, workers, config)

    @override(PPOTrainer)
    def get_default_policy_class(
            self, config: TrainerConfigDict
    ) -> Optional[Type[Policy]]:
        return PPOTF2Policy
