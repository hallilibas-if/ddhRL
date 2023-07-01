import ray
from ray import tune
from rllib_trainer import YourTrainer, config
from rllib_config import cust_config
from train_constants import YOUR_ROOT
from time import sleep
import numpy as np
import os

#configs
config.update(cust_config)
config['num_workers'] = 0   # when running on a big machine or multiple machines can run more workers
LOCAL_MODE = False # in local mode you can debug it. When False then agent is not finding ./results
from environments.carla.Environment import NUM_AGENTS
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NUM_GPUS=1
NUM_CPUS=6
OBJECT_STORE_MEMORY=30000000000 #30GB
RUN_WITH_TUNE = True
NUM_ITERATIONS = 10000  # 500 results in Tensorboard shown with 500 iterations (about an hour)

print(ray.init(num_gpus=NUM_GPUS,
            num_cpus=NUM_CPUS,
            object_store_memory=OBJECT_STORE_MEMORY,
            namespace="rllib_carla",
            local_mode=LOCAL_MODE))


@ray.remote(num_cpus=2)
class buffer_com(object):
    def __init__(self):
        self.agent_obs = np.zeros((NUM_AGENTS,182, 182, 3))
        self.agent_actions= np.zeros((NUM_AGENTS,3))
        self.rewards =[]
        for x in range(NUM_AGENTS):
            self.rewards.append({})
        
        self.agent_scalarInput =[]
        for x in range(NUM_AGENTS):
            self.agent_scalarInput.append({})

        self.done = np.zeros((NUM_AGENTS,1))
        self.key = np.zeros((NUM_AGENTS,1))
        self.mutexObs=np.zeros((NUM_AGENTS,1))
        self.mutexAction=np.zeros((NUM_AGENTS,1))

    def set_done(self, done):
        self.done = done

    def get_actions(self, ID ,key, obs,scalarInput, rewards, done):
        self.agent_obs[ID] = obs
        self.rewards[ID] = rewards
        self.agent_scalarInput[ID] = scalarInput
        self.done[ID] = done
        self.key[ID] = key
        self.mutexObs[ID] = 1
        #print("I am ego vehicle {} and I will give my sards to agent".format(ID))
        while self.mutexAction[ID]==0:
            sleep(0.01)
        self.mutexAction[ID]=0
        #print("I am ego vehicle {} and I got actions from agent: {}".format(ID, self.agent_actions[ID]))
        return self.agent_actions[ID]

    def get_sards(self, ID, key, actions):
        """
        Makes API call to simulator to capture a camera image which is saved to disk,
        loads the captured image from disk and returns it as an observation.
        """
        #print("I am Agent {} and I will give my actions to ego vehicle: {}".format(ID, actions))
        self.agent_actions[ID] = actions
        self.mutexAction[ID] = 1
        time_ran=0
        while self.mutexObs[ID]==0:
            sleep(0.01)
            if time_ran == 15000 or (self.key[ID]!= key and key!=0):
                self.done[ID]=True 
                self.agent_obs[ID] = np.zeros((182, 182, 3))
                self.agent_scalarInput[ID] = {}
                self.rewards[ID] = {}
            time_ran+=1
        self.mutexObs[ID] = 0
        #print("I am agent {} and I got my sards".format(ID))
        return self.agent_obs[ID], self.agent_scalarInput[ID],self.rewards[ID], self.done[ID], self.key[ID]


sard_buffer=buffer_com.options(name="carla_com",max_concurrency=2*NUM_AGENTS).remote() #ToDo Shawan change max_concurrency to 2*agent_number


# Tune is the system for keeping track of all of the running jobs, originally for
# hyperparameter tuning
if RUN_WITH_TUNE:

    tune.registry.register_trainable(YOUR_ROOT, YourTrainer)
    stop = {
            "training_iteration": NUM_ITERATIONS  # Each iteration is some number of episodes
        }
    results = tune.run(YOUR_ROOT, stop=stop, config=config, verbose=1, checkpoint_freq=10)

    # You can just do PPO or DQN but we wanted to show how to customize
    #results = tune.run("PPO", stop=stop, config=config, verbose=1, checkpoint_freq=10)

else:
    from CARLA_environment import YourEnvironment
    trainer = YourTrainer(config, env=YourEnvironment)

    # You can just do PPO or DQN but we wanted to show how to customize
    #from ray.rllib.agents.ppo import PPOTrainer
    #trainer = PPOTrainer(config, env=YourEnvironment)

    trainer.train()

    # Results at YOUR_ROOT/YourTrainer_YourEnvironment_YYYY_MM_DD_SS-NN-XXXXXXXXX
