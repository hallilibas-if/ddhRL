"""
https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/models/torch/torch_modelv2.py
This is for PyTorch but TensorFlow is analogous
"""
import tensorflow as tf
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from TFModel import _tf_build_networks
import numpy as np
from ray.rllib.policy.view_requirement import ViewRequirement

class CustomTFModel(TFModelV2):
    """
    A TFModelV2 Policy model that uses the neural network structure proposed in Angelo's project.
    See: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_keras_model.py
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomTFModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # For logging actions mean an variance
        self.configuration = model_config["custom_model_config"]
        print("Shawan: ", self.configuration)
        build_networks = _tf_build_networks

        self.last_value_output = None  # Lazy init for most recent value of a forward pass

        self.dpc_encoder, self.actor_critic_shared_model, self.actor_model, self.critic_model = build_networks(self.configuration)
        
        #     logger.info("DPC Weights loaded from path: " + str(path))
        # else:
        #     logger.warn("No weights found to load from path: " + str(path))

        self.save_dummy = tf.keras.Model([self.dpc_encoder, self.actor_critic_shared_model, self.actor_model, self.critic_model])

        self.num_frames = self.configuration['NUM_DPC_FRAMES']

        for model in [self.actor_critic_shared_model, self.actor_model,
                      self.critic_model]:
            model.summary()

        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs",
            shift="-{}:0".format(self.num_frames - 1),
            space=obs_space)
        self.view_requirements["prev_n_rewards"] = ViewRequirement(
            data_col="rewards", shift="-{}:-1".format(self.num_frames))
        self.view_requirements["prev_n_actions"] = ViewRequirement(
            data_col="actions",
            shift="-{}:-1".format(self.num_frames),
            space=self.action_space)

    def forward(self, input_dict, state, seq_lens):
        """
        Implements the forward pass.
        See: https://ray.readthedocs.io/en/latest/rllib-models.html

        :param input_dict: {"obs”, “obs_flat”, “prev_action”, “prev_reward”, “is_training”}
        :param state: None
        :param seq_lens: None
        :return: (ouputs, state), outsputs of size [BATCH, num_outputs]
        """
        stacked_observations = input_dict["prev_n_obs"][:,:,:,:,0:3]
        print("Shape of stacked obs: ", stacked_observations.shape)
        inputs =  np.array(stacked_observations, dtype=np.float32) 
        inputs = inputs.transpose(0, 4, 1, 2, 3)
        print("Shape of stacked obs inside inputs: ", inputs.shape)
        #inputs = tf.cast(inputs, tf.float32)
        
        
        stacked_scalar = input_dict["prev_n_obs"][:,:,:,:,-1]
        #stacked_1_m = stacked_scalar[:,:,0:8,0:8] #ToDo Shawan:have to be dynamic! Here its hardcoded!
        #stacked_2_m = stacked_scalar[:,:,8:16,8:16] #ToDo Shawan:have to be dynamic! Here its hardcoded!
        
        stacked_1 = stacked_scalar[:,:,0,0] #Road speed --> ToDo Shawan:have to be dynamic! Here its hardcoded! #(?,5)
        stacked_2 = stacked_scalar[:,:,9,9] #Agent speed--> ToDo Shawan:have to be dynamic! Here its hardcoded! #(?,5)
 
        stacked_1 = tf.cast(stacked_1, tf.float32)
        stacked_2 = tf.cast(stacked_2, tf.float32)

        #stacked_1_m = tf.cast(stacked_1_m, tf.float32)
        #stacked_2_m = tf.cast(stacked_2_m, tf.float32)

        #print("Shape of stacked_1 : ", stacked_1.shape)

        #print("Batch size stacked_inputs: ", input_dict["prev_n_obs"].shape) #(1, 5, 128, 128, 4)
        #print("Batch size: ", stacked_observations.shape) #(1, 5, 128, 128, 3)
        #feature_map = self.dpc_encoder(inputs[1, ...])
        feature_map=self.dpc_encoder.run(inputs[None,:,:,:,:,:])
        print("Shape of the encoder output: ", np.array(feature_map).shape) #(1, 1, 8, 8, 128)

        feature_map = tf.squeeze(feature_map, [0])
        feature_map = tf.squeeze(feature_map, [1])
        print("Batch size feature_map: ", np.array(feature_map).shape) # (1, 8, 8, 128)
        
        
        
        if self.configuration['FREEZE_CONV_LAYERS']:
            feature_map = tf.stop_gradient(feature_map)
        stacked_1 = tf.stop_gradient(stacked_1)
        stacked_2 = tf.stop_gradient(stacked_2)


        
        shared_model_output = self.actor_critic_shared_model([feature_map])
        #print("Shape of the shared model output: ", shared_model_output.shape)

        logits = tf.concat(self.actor_model([shared_model_output,stacked_1,stacked_2]), axis=1, name="Concat_logits")
        #print("Actions: ", logits)
        #print("Output of the actor: ", logits.shape)

        self.last_value_output = tf.reshape(self.critic_model([shared_model_output,stacked_1,stacked_2]), [-1])

        return logits, []  # [] is empty state

    def value_function(self):
        """
        Use the last computed value from the forward pass operation. (see function self.forward())
        """
        return self.last_value_output