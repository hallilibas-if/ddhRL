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

        self.dpc_encoder, self.actor_critic_shared_model, self.actor_model, self.critic_model = build_networks(self.configuration)
        
        self.input_name = self.dpc_encoder.get_inputs()[0].name

        self.save_dummy = tf.keras.Model([self.dpc_encoder, self.actor_model, self.critic_model])

        self.num_frames = self.configuration['NUM_DPC_FRAMES']

        for model in [self.actor_model,
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
  
        #Get previous obs
        stacked_observations = input_dict["prev_n_obs"][:,:,:,:,0:3] 
        #print("Shape of stacked obs: ", stacked_observations.shape) #(32, 4, 128, 128, 3)
        #inputs =  np.array(current_observation, dtype=np.float32)  
        #inputs = inputs.transpose(0, 4, 1, 2, 3) # 0: BS , 1:T , 2:H , 3:W , 4:RGB
        #print("Shape of stacked obs inside inputs: ", inputs.shape) #(32, 3, 4, 128, 128)

        #Get current obs
        current_obsRaw = np.expand_dims(input_dict["obs"], axis=1) #Before (32, 128, 128, 3) --> (32, 1, 128, 128, 3)
        current_observation = current_obsRaw[:,:,:,:,0:3]  #(32, 128, 128, 3)
        #print("Shape of stacked obs: ", current_observation.shape) #(32,1, 128, 128, 3)
        inputs =  np.array(current_observation, dtype=np.float32)  
        inputs = inputs.transpose(0, 4, 1, 2,3)
        #print("Shape of stacked obs inside inputs: ", inputs.shape) #(32, 3, 1, 128, 128)
        #inputs = tf.cast(inputs, tf.float32)

        stacked_scalar_prev = input_dict["prev_n_obs"][:,:,:,:,-1] 
        stacked_scalar = current_obsRaw[:,:,:,:,-1]

        stacked_scalar = np.concatenate((stacked_scalar, stacked_scalar_prev), axis=1)

        stacked_1_m = stacked_scalar[:,:,2:6,2:6] #ToDo Shawan:have to be dynamic! Here its hardcoded!
        stacked_2_m = stacked_scalar[:,:,10:14,10:14] #ToDo Shawan:have to be dynamic! Here its hardcoded!
        stacked_3_m = stacked_scalar[:,:,18:22,18:22] #ToDo Shawan:have to be dynamic! Here its hardcoded!

        stacked_1_m = tf.cast(stacked_1_m, tf.float32)
        stacked_2_m = tf.cast(stacked_2_m, tf.float32)
        stacked_3_m = tf.cast(stacked_3_m, tf.float32)

        print("stacked scalar: ",stacked_scalar.shape)
        #print("prev actions: ", input_dict.keys() )
        stacked_1 = stacked_scalar[:,:,2,2] #Road speed --> ToDo Shawan:have to be dynamic! Here its hardcoded! #(?,5)
        stacked_2 = stacked_scalar[:,:,10,10] #Agent speed--> ToDo Shawan:have to be dynamic! Here its hardcoded! #(?,5)
        stacked_3 = stacked_scalar[:,:,18,18]
        stacked_1 = tf.cast(stacked_1, tf.float32)
        stacked_2 = tf.cast(stacked_2, tf.float32)
        stacked_3 = tf.cast(stacked_3, tf.float32)


        for i in range(1):
            input_raw= inputs[:,:,i,:,:]
            input_raw = np.array(input_raw)          
            #print("The model expects input shape: ", self.input_name.shape)
            feature_map_one = self.dpc_encoder.run(None, {self.input_name: input_raw})[-1]
            feature_map_one = feature_map_one.transpose(0,3,2,1)
            print("Shape of the encoder output2: ", np.array(feature_map_one).shape) #(1, 1, 4, 4, 512) 
            if i>0:
                feature_map = np.concatenate((feature_map, feature_map_one), axis=-1)
            else:
                feature_map = feature_map_one

        print("Batch size feature_map: ", np.array(feature_map).shape) # Attention! We need 4 dims ! -->(BS, 4, 4, 2048) 
        print("Shape of stacked_1_m : ", stacked_1_m[:,0,:,:].shape)
    
        for x in range(len(stacked_1[0,:])):
            stacked_1New = tf.expand_dims(stacked_1_m[:,x,:,:], axis=3)
            stacked_2New = tf.expand_dims(stacked_2_m[:,x,:,:], axis=3)
            stacked_3New = tf.expand_dims(stacked_3_m[:,x,:,:], axis=3)
            #print("Iteration: ", x) # 0 ; 1;2;3;4 because 5 stacked images
            #print("Shape of stacked_1New : ", stacked_1New.shape) # (1, 8, 8, 1)
            feature_map = tf.concat([feature_map, stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New,stacked_2New,stacked_1New, stacked_3New],axis =-1)
     
        print("Batch size feature_map concat: ", feature_map.shape) #(1, 8, 8, 138) 
    
        
        if self.configuration['FREEZE_CONV_LAYERS']:
            feature_map = tf.stop_gradient(feature_map)
            print("Offline trained encoder is frozen!")
        stacked_1 = tf.stop_gradient(stacked_1)
        stacked_2 = tf.stop_gradient(stacked_2)
        stacked_3 = tf.stop_gradient(stacked_3)


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