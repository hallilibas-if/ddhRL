"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines a artificial neural networks that are used by our custom RLLib model to learn on an environment.
"""

import os
import warnings
import logging
import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer

import tensorflow.keras as keras
from tensorflow.keras.layers import ReLU, BatchNormalization, AveragePooling3D, MaxPooling3D, ZeroPadding3D, Activation
import math
import onnx
import tensorflow as tf
from torch.autograd import Variable
from onnx_tf.backend import prepare



import numpy as np
from onnx import numpy_helper
import onnxruntime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def _tf_build_networks(configuration):
    """
	Build a joined network for actor and critic.
	Return the network and it's sub-networks.
	The network structure is explained in Kemal's thesis.
	"""

 
    #r18_onnx = onnx.load("/home/shawan/Desktop/slowfast-model-load/onnxModels/r18_moco.onnx")
    #r18_onnx = prepare(r18_onnx)
    model_encoder_path = "/home/shawan/Desktop/slowfast-model-load/onnxModels/yolo-intermediary.onnx"
    r18_onnx = onnxruntime.InferenceSession(model_encoder_path, None)
    print( model_encoder_path )
 


    actor_critic_input_shape = (1,1, 4, 4, 512) # zuvor  für DPC(1, 1, 2, 2, 2048), wenn MoCo mit 128x128 dann (1, 1, 4, 4, 432)  
    print("Shared Output Shape: ", actor_critic_input_shape) # for DPC is (1, 1, 8, 8, 128)

    # PPO Actor and Critic:

    # PPO Actor and Critic Shared Layers

    actor_critic_feature_map_input = tf.keras.layers.Input(shape=[actor_critic_input_shape[-3],
																  actor_critic_input_shape[-2],
																  actor_critic_input_shape[-1]+240]) # actor_critic_input_shape[4]+10]) 2*seq_length! Because after dpc there is no seq_length! Everything is only a big qubic!ToDo Shawan: Changed here to a dynamic way of implementation!

    layer = actor_critic_feature_map_input
    for i in range(configuration['SHARED_CNN_LAYERS']):
        layer = tf.keras.layers.Conv2D(filters=configuration['SHARED_CNN_FILTERS'][i],
                                       kernel_size=configuration['SHARED_CNN_KERNEL_SIZE'][i],
                                       strides=configuration['SHARED_CNN_STRIDES'][i],
                                       activation=tf.nn.relu,
                                       name='Actor_Critic' + '_SharedCNNLayer_' + str(i),
                                       kernel_initializer=configuration['INITIALIZER'],
                                       padding='same',
                                       kernel_regularizer=configuration['REGULARIZER'])(layer)

    actor_critic_shared_cnn_output = layer

    # PPO Actor Critic Inputs
    # (8 , 8 , 128)
    actor_feature_map_input = tf.keras.layers.Input(shape=[actor_critic_shared_cnn_output.shape[1],
                                                           actor_critic_shared_cnn_output.shape[2],
                                                           actor_critic_shared_cnn_output.shape[3]])
    # (8 , 8 , 128)
    critic_feature_map_input = tf.keras.layers.Input(shape=[actor_critic_shared_cnn_output.shape[1],
                                                           actor_critic_shared_cnn_output.shape[2],
                                                           actor_critic_shared_cnn_output.shape[3]])
    
    road_speed_map_input = tf.keras.layers.Input(shape=[configuration['NUM_DPC_FRAMES']+1])
    agent_speed_map_input = tf.keras.layers.Input(shape=[configuration['NUM_DPC_FRAMES']+1])
    agent_action_map_input = tf.keras.layers.Input(shape=[configuration['NUM_DPC_FRAMES']+1])

    # Build lists of states, since we work with keras GRU cells
    # Since not every layer is stateful, we need only as many states as we have stateful layers

    # Critic state lists
    critic_cnn_gru_state_inputs = [(a * b or None) and tf.keras.layers.Input(shape=[a * b]) for a, b in
                                   zip(configuration.get("CRITIC_GRU_STATE_SIZES"),
                                       configuration.get("CRITIC_CNN_GRU_LAYERS"))]

    critic_fc_gru_state_inputs = [(a * b or None) and tf.keras.layers.Input(shape=[a * b]) for a, b in
                                  zip(configuration.get("CRITIC_GRU_STATE_SIZES"),
                                      configuration.get("CRITIC_FC_GRU_LAYERS"))]

    # Actor state lists
    actor_cnn_gru_state_inputs = [(a * b or None) and tf.keras.layers.Input(shape=[a * b]) for a, b in
                                  zip(configuration.get("ACTOR_GRU_STATE_SIZES"),
                                      configuration.get("ACTOR_CNN_GRU_LAYERS"))]

    actor_fc_gru_state_inputs = [(a * b or None) and tf.keras.layers.Input(shape=[a * b]) for a, b in
                                 zip(configuration.get("ACTOR_GRU_STATE_SIZES"),
                                     configuration.get("ACTOR_FC_GRU_LAYERS"))]

    value, critic_gru_fc_state_outputs, critic_gru_cnn_state_outputs = _tf_build_critic_net(x=critic_feature_map_input,
                                                                                        configuration=configuration,
                                                                                        agent_speed=agent_speed_map_input, 
                                                                                        road_speed =road_speed_map_input,
                                                                                        agent_actions=agent_action_map_input,
                                                                                        critic_cnn_gru_states=critic_cnn_gru_state_inputs,
                                                                                        critic_fc_gru_states=critic_fc_gru_state_inputs)

    action_layer, actor_gru_fc_state_outputs, actor_gru_cnn_state_outputs = _tf_build_actor_net(x=actor_feature_map_input, 
                                                                                            configuration=configuration,
                                                                                            agent_speed=agent_speed_map_input, 
                                                                                            road_speed =road_speed_map_input,
                                                                                            agent_actions=agent_action_map_input,
                                                                                            actor_cnn_gru_states=actor_cnn_gru_state_inputs,
                                                                                            actor_fc_gru_states=actor_fc_gru_state_inputs)


    actor_critic_shared = tf.keras.Model([actor_critic_feature_map_input],
                                         [actor_critic_shared_cnn_output],
                                         name='Actor_Critic_Shared_Model')

    # Reduce our state inputs
    actor_cnn_gru_state_inputs = [i for i in actor_cnn_gru_state_inputs if i is not None]
    actor_fc_gru_state_inputs = [i for i in actor_fc_gru_state_inputs if i is not None]
    actor = tf.keras.Model(inputs=[actor_feature_map_input, agent_speed_map_input, road_speed_map_input,agent_action_map_input,actor_cnn_gru_state_inputs, actor_fc_gru_state_inputs],
                           outputs=[action_layer, actor_gru_cnn_state_outputs, actor_gru_fc_state_outputs],
                           name='Actor_Model')

    # Reduce our state inputs
    critic_cnn_gru_state_inputs = [i for i in critic_cnn_gru_state_inputs if i is not None]
    critic_fc_gru_state_inputs = [i for i in critic_fc_gru_state_inputs if i is not None]
    critic = tf.keras.Model([critic_feature_map_input,agent_speed_map_input, road_speed_map_input, agent_action_map_input, critic_cnn_gru_state_inputs, critic_fc_gru_state_inputs],
                            [value, critic_gru_cnn_state_outputs, critic_gru_fc_state_outputs],
                            name='Critic_Model')

    return r18_onnx, actor_critic_shared, actor, critic


def _tf_build_actor_net(x, configuration, agent_speed, road_speed, agent_actions,actor_cnn_gru_states, actor_fc_gru_states):
    # These two lists tell us which layers should be stateful
    actor_rnn_cnn_layers = configuration.get("ACTOR_CNN_GRU_LAYERS") or [0] * 64
    actor_rnn_fc_layers = configuration.get("ACTOR_FC_GRU_LAYERS") or [0] * 64

    # Append actor CNN layers to graph
    gru_cnn_state_outputs = []
    for i in range(configuration['ACTOR_CNN_LAYERS']):
        actor_cnn_layer_params = {"filters": configuration['ACTOR_CNN_FILTERS'][i],
                                  "kernel_size": configuration['ACTOR_CNN_KERNEL_SIZE'][i],
                                  "strides": configuration['ACTOR_CNN_STRIDES'][i],
                                  "activation": tf.nn.relu,
                                  "name": "ActorCNNLayer_CNN_" + str(i),
                                  "kernel_initializer": configuration['INITIALIZER'],
                                  "kernel_regularizer": configuration['REGULARIZER'],
                                  "padding": 'same'}

        if actor_rnn_cnn_layers[i]:
            raise NotImplementedError("Currently, there is no Conv2D GRU Layer available. "
                                      "The available LSTM layer has not been tested.")
            x = tf.keras.layers.ConvLSTM2D(**actor_cnn_layer_params)(x)
        else:
            x = tf.keras.layers.Conv2D(**actor_cnn_layer_params)(x)

    x = tf.keras.layers.Flatten()(x) #ToDo concatinate speed information
    #agent_speed = tf.keras.layers.Dense(units=4, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(agent_speed)
    #road_speed = tf.keras.layers.Dense(units=4, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(road_speed)
    #add_both = tf.keras.layers.Concatenate(axis=-1)([road_speed,agent_speed])
    #dense_both = tf.keras.layers.Dense(units=32, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(add_both)
    #x = tf.keras.layers.Concatenate(axis=-1)([x, dense_both])
    #print("Tief drinne shape: ",x.shape)
    
    # Append actor FC layers to graph
    gru_fc_state_outputs = []
    recurrent_layer_counter = 0
    for i in range(configuration['ACTOR_FC_LAYERS']):
        actor_layer_params = {"units": configuration['ACTOR_FC_UNITS'][i],
                              "activation": tf.nn.relu,
                              "name": 'ActorLayer_Dense_' + str(i),
                              "kernel_initializer": configuration['INITIALIZER'],
                              "kernel_regularizer": configuration['REGULARIZER']}

        if actor_rnn_fc_layers[i]:
            actor_layer_params.update({"name": "ActorGRULayer_DenseGRU_" + str(i),
                                       "units": configuration["ACTOR_GRU_STATE_SIZES"][i]})
            state = actor_fc_gru_states[recurrent_layer_counter]

            x, state = tf.keras.layers.GRUCell(**actor_layer_params)(x, state)

            recurrent_layer_counter += 1
            gru_fc_state_outputs.append(state)
        else:
            x = tf.keras.layers.Dense(**actor_layer_params)(x)
    
    agent_speed = tf.keras.layers.Dense(units=5, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(agent_speed)
    road_speed = tf.keras.layers.Dense(units=5, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(road_speed)
    agent_actions = tf.keras.layers.Dense(units=5, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(agent_actions)

    add_all = tf.keras.layers.Concatenate(axis=-1)([x,road_speed,agent_speed, agent_actions])
    #x = tf.keras.layers.Concatenate(axis=-1)([x, dense_both])
    print("Tief drinne shape Actor: ",add_all.shape)
    x = tf.keras.layers.Dense(units=50, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(add_all)         
    mean = tf.keras.layers.Dense(configuration["OUTPUTS"],
                                 activation=tf.nn.tanh,
                              name='ActorLayer_last_mean',
                              kernel_initializer=normc_initializer(0.01),
                              kernel_regularizer=configuration['REGULARIZER'])(x)
    var = tf.keras.layers.Dense(configuration["OUTPUTS"],
                              name='ActorLayer_last_var',
                              kernel_initializer=normc_initializer(0.01),
                              kernel_regularizer=configuration['REGULARIZER'])(x)
    
    """
    correlations = tf.keras.layers.Dense(configuration["OUTPUTS"] * (configuration["OUTPUTS"] - 1) / 2,
                                         name='ActorLayer_last_correlations',
                                         kernel_initializer=normc_initializer(0.01),
                                         kernel_regularizer=configuration['REGULARIZER'])(x)
    
    x = tf.keras.layers.Concatenate(axis=-1)([mean, var, correlations])
    """
    x = tf.keras.layers.Concatenate(axis=-1)([mean, var])

    return x, gru_fc_state_outputs, gru_cnn_state_outputs


def _tf_build_critic_net(x, configuration,agent_speed, road_speed, agent_actions, critic_cnn_gru_states, critic_fc_gru_states):
    # These two lists tell us which layers should be stateful
    critic_rnn_cnn_layers = configuration.get("CRITIC_CNN_GRU_LAYERS") or [0] * 64
    critic_rnn_fc_layers = configuration.get("CRITIC_FC_GRU_LAYERS") or [0] * 64

    # Append critic CNN layers to graph
    gru_cnn_state_outputs = []
    for i in range(configuration['CRITIC_CNN_LAYERS']):
        critic_cnn_layer_params = {"filters": configuration['CRITIC_CNN_FILTERS'][i],
                                   "kernel_size": configuration['CRITIC_CNN_KERNEL_SIZE'][i],
                                   "strides": configuration['CRITIC_CNN_STRIDES'][i],
                                   "activation": tf.nn.relu,
                                   "name": "CriticCNNLayer_CNN_" + str(i),
                                   "kernel_initializer": configuration['INITIALIZER'],
                                   "kernel_regularizer": configuration['REGULARIZER'],
                                   "padding": 'same'}

        if critic_rnn_cnn_layers[i]:
            raise NotImplementedError("Currently, there is no Conv2D GRU Layer available. "
                                      "The available LSTM layer has not been tested.")
            x = tf.keras.layers.ConvLSTM2D(**critic_cnn_layer_params)(x)

        else:
            x = tf.keras.layers.Conv2D(**critic_cnn_layer_params)(x)

    x = tf.keras.layers.Flatten()(x)
    #agent_speed = tf.keras.layers.Dense(units=2, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(agent_speed)
    #road_speed = tf.keras.layers.Dense(units=2, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(road_speed)
    #add_both = tf.keras.layers.Concatenate(axis=-1)([road_speed,agent_speed])
    #dense_both = tf.keras.layers.Dense(units=32, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(add_both)
    #x = tf.keras.layers.Concatenate(axis=-1)([x, dense_both])
    
    # Append critic FC layers to graph
    gru_fc_state_outputs = []
    recurrent_layer_counter = 0
    for i in range(configuration['CRITIC_FC_LAYERS']):
        critic_layer_params = {"units": configuration['CRITIC_FC_UNITS'][i],
                               "activation": tf.nn.relu,
                               "name": 'CriticLayer_Dense_' + str(i),
                               "kernel_initializer": configuration['INITIALIZER'],
                               "kernel_regularizer": configuration['REGULARIZER']}

        if critic_rnn_fc_layers[i]:
            critic_layer_params.update({"name": "CriticGRULayer_DenseGRU_" + str(i),
                                        "units": configuration["CRITIC_GRU_STATE_SIZES"][i]})
            state = critic_fc_gru_states[recurrent_layer_counter]

            x, state = tf.keras.layers.GRUCell(**critic_layer_params)(x, state)

            recurrent_layer_counter += 1
            gru_fc_state_outputs.append(state)
        else:
            x = tf.keras.layers.Dense(**critic_layer_params)(x)
    
    agent_speed = tf.keras.layers.Dense(units=5, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(agent_speed)
    road_speed = tf.keras.layers.Dense(units=5, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(road_speed)
    agent_actions = tf.keras.layers.Dense(units=5, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(agent_actions)

    add_all = tf.keras.layers.Concatenate(axis=-1)([x,road_speed,agent_speed, agent_actions])
    #x = tf.keras.layers.Concatenate(axis=-1)([x, dense_both])
    print("Tief drinne shape Critic: ",add_all.shape)
    x = tf.keras.layers.Dense(units=50, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(add_all)    

    #x = tf.keras.layers.Concatenate(axis=-1)([x, dense_both])
    #x = tf.keras.layers.Dense(units=64, kernel_initializer="he_normal", kernel_regularizer="l1_l2", activation='relu')(x)
    value_func = tf.keras.layers.Dense(1, name='value', kernel_initializer=configuration['INITIALIZER'],
                                       kernel_regularizer=configuration['REGULARIZER'])(x)

    return value_func, gru_fc_state_outputs, gru_cnn_state_outputs