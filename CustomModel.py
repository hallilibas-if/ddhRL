"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines an RLlib custom model.
https://rllib.readthedocs.io/en/latest/rllib-training.html
"""

import tensorflow as tf
import os
import logging

import torch
import torch.nn as nn
import torchvision.utils as vutils

import numpy as np
import cv2


from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.view_requirement import ViewRequirement


from TFModel import _tf_build_networks
from TorchModel import _torch_build_networks
from TorchModel import  load_encoder_weights
logger = logging.getLogger(__name__)

# For showing activations distributions
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


class CustomTFModel(TFModelV2):
    """
    A TFModelV2 Policy model that uses the neural network structure proposed in Angelo's project.
    See: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_keras_model.py
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomTFModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # For logging actions mean an variance
        self.writer_freq = 1
        self.writer_current_step = 0

        self.configuration = model_config['custom_model_config']

        build_networks = _tf_build_networks

        self.last_value_output = None  # Lazy init for most recent value of a forward pass

        self.dpc_encoder, self.actor_critic_shared_model, self.actor_model, self.critic_model = build_networks(
            self.configuration)
        
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

        """
        for x in range(len(stacked_1[0,:])):
            stacked_1New = tf.expand_dims(stacked_1_m[:,x,:,:], axis=3)
            stacked_2New = tf.expand_dims(stacked_2_m[:,x,:,:], axis=3)
            #print("Iteration: ", x) # 0 ; 1;2;3;4 because 5 stacked images
            #print("Shape of stacked_1New : ", stacked_1New.shape) # (1, 8, 8, 1)
            feature_map = tf.concat([feature_map, stacked_2New,stacked_1New],axis =-1)
        """
        #print("Batch size feature_map concat: ", feature_map.shape) #(1, 8, 8, 138)
        #Here sensor fusion have to be happens ToDoShawan
        shared_model_output = self.actor_critic_shared_model([feature_map])
        #print("Shape of the shared model output: ", shared_model_output.shape)

        logits = tf.concat(self.actor_model([shared_model_output,stacked_1,stacked_2]), axis=1, name="Concat_logits")
        #print("Actions: ", logits)
        #print("Output of the actor: ", logits.shape)

        self.last_value_output = tf.reshape(self.critic_model([shared_model_output,stacked_1,stacked_2]), [-1])


            # if enough data is available to compute mean and variance
        if stacked_observations.shape[0] > 32:
            # print("Training logits:", logits.shape)
            if self.configuration["OUTPUTS"] == 1:  # only steering
                try:
                    self.writer.add_histogram('Steering mean', logits[:, 0].numpy(), self.writer_current_step)
                    self.writer.add_histogram('Steering variance', tf.exp(logits[:, 1]).numpy(), self.writer_current_step)
                    #print("Stuff added to summary writer: ", logits[0, 0], logits[0, 1])
                except:
                    print("Warning: Nothing added to summary writer")
            elif self.configuration["OUTPUTS"] == 2:  # steering and acceleration
                mean, std = tf.split(logits, 2, axis=1)
                steering_mean = mean[:, 0].numpy()
                throttle_mean = mean[:, 1].numpy()
                steering_std = np.exp((std[:, 0]).numpy())
                throttle_std = np.exp((std[:, 1]).numpy())
                #print("steering mean shape: {}, mean: {}".format(steering_mean.shape, steering_mean))
                #print("steering std shape: {}, std: {}".format(steering_std.shape, steering_std))

                #print("throttle mean shape: {}, mean: {}".format(throttle_mean.shape, throttle_mean))
                #print("throttle std shape: {}, std: {}".format(throttle_std.shape, throttle_std))

                try:
                    self.writer.add_histogram('Steering action mean', steering_mean, self.writer_current_step)
                    self.writer.add_histogram('Steering action variance', steering_std,
                                              self.writer_current_step)
                    self.writer.add_histogram('Throttle action mean', throttle_mean, self.writer_current_step)
                    self.writer.add_histogram('Throttle action variance', throttle_std,
                                              self.writer_current_step)
                except:
                    print("Warning: Nothing added to summary writer")
            elif self.configuration["OUTPUTS"] == 3:  # steering, braking and throttle
                mean, std = tf.split(logits, 2, axis=1)
                steering_mean = mean[:, 0].numpy()
                throttle_mean = mean[:, 1].numpy()
                break_mean = mean[:, 2].numpy()
                steering_std = tf.exp(std[:, 0]).numpy()
                throttle_std = tf.exp(std[:, 1]).numpy()
                break_std = tf.exp(std[:, 2]).numpy()
                try:
                    self.writer.add_histogram('Steering action mean', steering_mean, self.writer_current_step)
                    self.writer.add_histogram('Steering action variance', steering_std,
                                              self.writer_current_step)
                    self.writer.add_histogram('Throttle action mean', throttle_mean, self.writer_current_step)
                    self.writer.add_histogram('Throttle action variance', throttle_std,
                                              self.writer_current_step)
                    self.writer.add_histogram('Break action mean', break_mean, self.writer_current_step)
                    self.writer.add_histogram('Break action variance', break_std, self.writer_current_step)
                except:
                    print("Warning: Nothing added to summary writer")

            self.writer_current_step += 1

        return logits, []  # [] is empty state

    def value_function(self):
        """
        Use the last computed value from the forward pass operation. (see function self.forward())
        """
        return self.last_value_output



class CustomTorchModel(TorchModelV2, nn.Module):
    """
    A TFModelV2 Policy model that uses the neural network structure proposed in Angelo's project.
    See: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_keras_model.py
    """
    writer = SummaryWriter(logdir="../results/")
    def __init__(self, obs_space, action_space, num_outputs, model_config, name="Torch_model"):
        nn.Module.__init__(self)
        #super(CustomTorchModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        #TorchModelV2.__init__(self, *args, **kwargs)
        #raise NotImplementedError("Hi Mohamed! Please implement me")

        # For logging actions mean an variance
        self.writer_freq = 1
        self.writer_current_step = 0

        self.configuration = model_config['custom_model_config']

        build_networks = _torch_build_networks
        
        self.last_value_output = None  # Lazy init for most recent value of a forward pass
        
        self.dpc_encoder, self.actor_critic_shared_model, self.actor_model, self.critic_model = build_networks(
            self.configuration)

        path = self.configuration["weights_path"]
        self.import_weights_from_path(path)

        #self.dpc_encoder.eval()
        #self.actor_critic_shared_model.train()
        #self.actor_model.train()
        #self.critic_model.train()

        for n, p in self.dpc_encoder.named_parameters():
            p.requires_grad = False

        self.num_frames = self.configuration['NUM_DPC_FRAMES']
        """
        for model in [self.dpc_encoder, self.actor_critic_shared_model, self.actor_model,
                      self.critic_model]:
            print(model)
        """
        #self.save_dummy = torch.nn.Sequential(
        #    *[self.dpc_encoder, self.actor_critic_shared_model, self.actor_model, self.critic_model])

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

        # TODO: delete this after debuging
        self.saved_img_i = 0

        # TODO: delete this after debuging
        self.last_dpc_weights = None

    def denormalize(self, x):
        return 255. * (x/2. + 0.5)

    def forward(self, input_dict, state, seq_lens):
        """
        Implements the forward pass.
        See: https://ray.readthedocs.io/en/latest/rllib-models.html

        :param input_dict: {"obs”, “obs_flat”, “prev_action”, “prev_reward”, “is_training”}
        :param state: None
        :param seq_lens: None
        :return: (ouputs, state), outsputs of size [BATCH, num_outputs]
        """


        """# TODO: delete after debuging
        if self.last_dpc_weights is None:  # parameters not saved yet
            self.last_dpc_weights = self.dpc_encoder.parameters()
        else:  # parmeters were
            for p1, p2 in zip(self.dpc_encoder.parameters(), self.last_dpc_weights):
                if p1 != p2:
                    raise ValueError("DPC ENCODER WEIGHTS CHANGED!!!")
        """
        stacked_observations = input_dict["prev_n_obs"][:,:,:,:,0:3]
       

        stacked_scalar = input_dict["prev_n_obs"][:,:,:,:,-1]
        stacked_1 = stacked_scalar[:,:,0:8,0:8] #ToDo Shawan:have to be dynamic! Here its hardcoded!
        stacked_2 = stacked_scalar[:,:,8:16,8:16] #ToDo Shawan:have to be dynamic! Here its hardcoded!

        """
        stacked_observations = input_dict["prev_n_obs"]  #.float()

        print("in custom model: obs type={}, max={}, min={}".format(type(stacked_observations),
                                                                    stacked_observations.max(),
                                                                    stacked_observations.min()))
        """
        """
        self.saved_img_i += 1  # TODO: DELETE THIS
        if stacked_observations.shape[0] ==1:
            print("Input max: {}, min: {}".format(stacked_observations.max(), stacked_observations.min()))
            print("Input shape:", stacked_observations.shape)
            grid_images = (self.denormalize(vutils.make_grid(
                stacked_observations.permute(0, 1, 4, 2, 3).contiguous().view(5, 3, 128, 128),
                nrow=1)))

            grid_images = grid_images.cpu().permute(1, 2, 0).numpy()
            #grid_images = np.transpose(grid_images.numpy(), (1, 2, 0))
            print("Images denormalized shape:", grid_images.shape)
            print("Images denormalized max: {}, min: {}".format(grid_images.max(), grid_images.min()))

            cv2.imwrite('/home/shawanmohammed/Desktop/deeprl-for-autonomous-driving/results/stacked_images/image{}.jpg'.format(self.saved_img_i),
                        grid_images)
            print("image saved sould be !!!")
            cv2.waitKey(100)
        """
        #if self.dpc_encoder.training:
            #raise ValueError("DPC model shouldn't be training")
        #    self.dpc_encoder.eval()

        feature_map = self.dpc_encoder(stacked_observations)
        feature_map = feature_map.squeeze(1) # 32,128,8,8
        print("Shape of the encoder output: ", feature_map.shape) #(1, 1, 8, 8, 128)
        print("stacked_1 :", stacked_1.shape)
        feature_map = torch.cat([feature_map, stacked_2,stacked_1],axis =1)
        

        
        if self.configuration['FREEZE_CONV_LAYERS']:
            feature_map = tf.stop_gradient(feature_map)
        
        #shared_model_output = torch.zeros_like(feature_map)#feature_map#self.actor_critic_shared_model(feature_map)#.detach())
        shared_model_output = feature_map
        """
        print("in custom model: dpc output: type={}, max={}, min={}".format(type(shared_model_output),
                                                                                shared_model_output.max(),
                                                                                shared_model_output.min()))

        print("Shape of the shared model output: ", shared_model_output.shape)
        print("shared model max: {}, min: {}".format(shared_model_output.max(), shared_model_output.min()))
        """
        actor_out = self.actor_model(shared_model_output)

        #print("Actor output shape", actor_out.shape)

        #logits = torch.squeeze(actor_out, 1)
        logits = actor_out
        #print("Logits output shape", logits.shape)

        #print("Logits from actor: ", logits)

        #print("Output of the actor: ", logits.shape)
        #logits = tf.concat(self.actor_model([shared_model_output]), axis=1, name="Concat_logits")
        #self.last_value_output = tf.reshape(self.critic_model([shared_model_output]), [-1])
        #print("shared model output shape", shared_model_output.shape)
        self.last_value_output = self.critic_model(shared_model_output).squeeze(1)#torch.reshape(self.critic_model(shared_model_output), (-1,))  #torch.squeeze(self.critic_model(shared_model_output), 0)# #
        #self.last_value_output = self.last_value_output.detach().cpu().numpy()

        """
        print("Critics Output:", self.last_value_output)
        print("Critics Output Shape:", self.last_value_output.shape)
        print("in custom model: critics output: type={}, max={}, min={}".format(type(self.last_value_output),
                                                                                self.last_value_output.max(),
                                                                                self.last_value_output.min()))
        """
        logits = torch.clamp(logits, -10., 10.)

        # add stuff to writer
        if self.writer_current_step % self.writer_freq == 0:
            print("shape is ", stacked_observations.shape)

            if stacked_observations.shape[0] > 2:
                # print("Training logits:", logits.shape)
                if self.configuration["OUTPUTS"] == 1:
                    try:
                        self.writer.add_histogram('action mean', logits[:,0], self.writer_current_step)
                        self.writer.add_histogram('action variance', logits[:,1], self.writer_current_step)
                        print("Stuff added to summary writer: ", logits[0,0], logits[0,1])
                    except:
                        print("Warning: Nothing added to summary writer")
                elif self.configuration["OUTPUTS"] == 2:
                    mean, std = torch.chunk(logits, 2, dim=1)
                    steering_mean = mean[:, 0]
                    throttle_mean = mean[:, 1]
                    steering_std = std[:, 0]
                    throttle_std = std[:, 1]
                    try:
                        self.writer.add_histogram('Steering action mean', steering_mean, self.writer_current_step)
                        self.writer.add_histogram('Steering action variance', steering_std, self.writer_current_step)
                        self.writer.add_histogram('Throttle action mean', throttle_mean, self.writer_current_step)
                        self.writer.add_histogram('Throttle action variance', throttle_std, self.writer_current_step)
                    except:
                        print("Warning: Nothing added to summary writer")
                elif self.configuration["OUTPUTS"] == 3:
                    mean, std = torch.chunk(logits, 2, dim=1)
                    steering_mean = mean[:, 0]
                    throttle_mean = mean[:, 1]
                    break_mean = mean[:,2]
                    steering_std = std[:, 0]
                    throttle_std = std[:, 1]
                    break_std = std[:, 2]
                    try:
                        self.writer.add_histogram('Steering action mean', steering_mean, self.writer_current_step)
                        self.writer.add_histogram('Steering action variance', steering_std, self.writer_current_step)
                        self.writer.add_histogram('Throttle action mean', throttle_mean, self.writer_current_step)
                        self.writer.add_histogram('Throttle action variance', throttle_std, self.writer_current_step)
                        self.writer.add_histogram('Break action mean', break_mean, self.writer_current_step)
                        self.writer.add_histogram('Break action variance', break_std, self.writer_current_step)
                    except:
                        print("Warning: Nothing added to summary writer")
        if stacked_observations.shape[0] > 2:
            self.writer_current_step += 1  # only increment if batch.shape[0] > 1

        """
        print("in custom model: actor's logits: type={}, max={}, min={}".format(type(logits),
                                                                    logits.max(),
                                                                    logits.min()))
        

        # Debgging: print model weights
        for name, param in self.actor_model.state_dict().items():
            print("Actor weights: name={}, max={}, min={}".format(name, param.max(), param.min()))
        for name, param in self.critic_model.state_dict().items():
            print("Critic weights: name={}, max={}, min={}".format(name, param.max(), param.min()))
        """
        return logits, []  # [] is empty state
    
    def value_function(self):
        """
        Use the last computed value from the forward pass operation. (see function self.forward())
        """
        return self.last_value_output # torch.squeeze(self.last_value_output, [-1])#
    
    def import_weights_from_path(self, weights_path: str) -> None:

        # print encoder state dict
        #print("DPC Encoder state_dict: ")
        print(self.dpc_encoder.state_dict())
        # load weights
        print("Loading DPC weights")
        load_encoder_weights(self.dpc_encoder,
                             pretrain_pth=weights_path,
                             from_reconstruction_model=self.configuration["DPC_TRAINED_FOR_RECONST"])

        # print encoder state dict
        #print("DPC Encoder new state_dict")
        #print(self.dpc_encoder.state_dict())

