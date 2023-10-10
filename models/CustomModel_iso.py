"""
https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/models/torch/torch_modelv2.py
This is for PyTorch but TensorFlow is analogous
"""
import tensorflow as tf
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from models.TFModel_iso import _tf_build_networks
import numpy as np
from ray.rllib.policy.view_requirement import ViewRequirement
from utils.tensorboard_writer import tensorboard_writer

class CustomTFModel(TFModelV2):
    """
    A TFModelV2 Policy model that uses the neural network structure proposed in our papers.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomTFModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.configuration = model_config["custom_model_config"]
        # For logging actions mean an variance
        self.write_tensorboard = tensorboard_writer(self.configuration)

        build_networks = _tf_build_networks

        self.last_value_output = None  # Lazy init for most recent value of a forward pass

        self.actor_critic_shared_model, self.actor_model, self.critic_model = build_networks(self.configuration)

        self.save_dummy = tf.keras.Model([self.actor_model, self.critic_model])

        self.num_frames = self.configuration['NUM_DPC_FRAMES']

        for model in [self.actor_model,
                      self.critic_model]:
            model.summary()

        """
        #This is where the miracle of stacked data happens
        self.view_requirements["prev_obs_scalar"] = ViewRequirement(
            data_col=("obs", 1),
            shift="-{}:0".format(self.num_frames - 1),
            space=obs_space)
        
        self.view_requirements["prev_n_rewards"] = ViewRequirement(
            data_col="rewards", shift="-{}:-1".format(self.num_frames))
        self.view_requirements["prev_n_actions"] = ViewRequirement(
            data_col="actions",
            shift="-{}:-1".format(self.num_frames),
            space=self.action_space)
        """
    def forward(self, input_dict, state, seq_lens):
        """
        Implements the forward pass.
        See: https://ray.readthedocs.io/en/latest/rllib-models.html

        :param input_dict: {"obs”, “obs_flat”, “prev_action”, “prev_reward”, “is_training”}
        :param state: None
        :param seq_lens: None
        :return: (ouputs, state), outsputs of size [BATCH, num_outputs]
        """
 

        #Get current obs
        current_observation = input_dict["obs"][:,:,:,:-1] # (32, 4, 4, 2048) 
        current_observation = tf.cast(current_observation, tf.float32)
        print("Shape of stacked obs inside inputs: ", current_observation.shape)  #(32, 4, 4, 2048)

        obs_scalar = input_dict["obs"][:,:,:,-1:] # (BS,obs_scalar,sequence) (32,3,4)
        print("Shape of stacked_scalar_obs_scalar obs inside inputs: ", obs_scalar.shape)  #(32, 4, 4, 2048)


        stacked_1 = obs_scalar[:,0] # ToDo check shape -->(32,4) cycleSpeedCurrent
        stacked_2 = obs_scalar[:,1] # ToDo check shape -->(32,4) cycleSpeedLimit / cycleSpeedTarget
        stacked_3 = obs_scalar[:,2] # ToDo check shape -->(32,4) cycleAcc
        print("Shape of stacked_scalar_obs_scalar obs inside inputs: ", stacked_3.shape)  #(32, 4, 4, 2048)

        stacked_1 = tf.cast(stacked_1, tf.float32)
        stacked_2 = tf.cast(stacked_2, tf.float32)
        stacked_3 = tf.cast(stacked_3, tf.float32)

        #does this here works ? If not , use a normal for loop
        stacked_1_m = np.dstack((stacked_1, stacked_2, stacked_3, stacked_1))  # np.expand_dims(stacked_1["obs"], axis=1)
        stacked_1_m = np.expand_dims(stacked_1_m, axis=-1)

        stacked_2_m = np.dstack((stacked_2, stacked_3, stacked_1,stacked_2))
        stacked_2_m = np.expand_dims(stacked_2_m, axis=-1)

        stacked_3_m = np.dstack((stacked_3, stacked_1, stacked_2, stacked_3 ))
        stacked_3_m = np.expand_dims(stacked_3_m, axis=-1)
        
        stacked_1_m = tf.cast(stacked_1_m, tf.float32)
        stacked_2_m = tf.cast(stacked_2_m, tf.float32)
        stacked_3_m = tf.cast(stacked_3_m, tf.float32)

        print("Shape of stacked_1_m : ", stacked_1_m.shape)
    

        feature_map = tf.concat([current_observation, stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m,stacked_1_m,stacked_2_m,stacked_3_m],axis =-1)
     
        print("Batch size feature_map concat: ", feature_map.shape) #(1, 8, 8, 138) 
 

        #Here sensor fusion have to be happens ToDoShawan
  
  
        logits = tf.concat(self.actor_model([feature_map,stacked_1,stacked_2, stacked_3]), axis=1, name="Concat_logits")
        #print("Actions: ", logits)
        #print("Output of the actor: ", logits.shape)

        self.last_value_output = tf.reshape(self.critic_model([feature_map,stacked_1,stacked_2, stacked_3]), [-1])

        self.write_tensorboard.write(logits=logits)
        return logits, []  # [] is empty state
    

    def value_function(self):
        """
        Use the last computed value from the forward pass operation. (see function self.forward())
        """
        return self.last_value_output