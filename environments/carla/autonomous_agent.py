import logging
from mimetypes import init
from queue import Empty
from typing import Callable, Optional, Tuple

import os
import datetime
import gym
import csv
import cv2
import numpy as np
from gym import spaces
from gym.utils import seeding
from time import sleep



import ray
logger = logging.getLogger(__name__)



MIMIC = False

class controlAgent():
    def __init__(self) -> None:
        self.steering = 0
        self.throttle = 0
        self.braking = 0
        
def showImage(input_data, figureName='map'):
    #cv2.imshow('map', np.uint8(input_data))
    cv2.imshow(figureName,input_data/255)
    cv2.waitKey(1)


class Agent(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config, _id):
        self.control = controlAgent()
        self.action_space = config["action_space"]
        self.observation_space = config["observation_space"]
        self.output = config["OUTPUTS"]
        self.width = self.observation_space.shape[1]
        self.height = self.observation_space.shape[0]
        self._id = _id
        self.sard_buffer = ray.get_actor(name="carla_com", namespace="rllib_carla")  
        self.speedLimit = 0
        self.current_speed= 0
        self.oldKeepLane = 0

        #statistic cal.
        csv_path = 'route_statistics_' + str(self._id) + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.csv' # add datetime just in case it is overwritten by a failed trial
        self.file_path = os.path.join(config["experiment_path"], csv_path) #self.file_path = os.path.join("/home/shawan/Desktop/Shawan/ddhRL/results", csv_path)
        sleep(20)
        with open(self.file_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            header = ['Episode Number', 'Key' ,'Collision', 'Wrong lane change', 'Avg. distance to lane center', 'Route completed [%]','Max. distance traveled [steps]', 'Avg. speed compliance','Cumulativ reward']
            writer.writerow(header)
            f.close()

        self.allWrongChgLane=0
        self.routeCompletion = 0
        self.steps = 0
        self.num_episodes = 0
        self.sumSpeedCompliance=0
        self.episodeReward =0
        self.rand_speed_coeff= np.random.randint(0, 9)
        self.flag_speedLimit = 0
        self.acc_controle_coeff= 0.0 #0.8
        self.acceleration = 0.0
        self.key = 0


        #reward cal.
        self.reward = 0
        self.reward_speedLimit = 0
        self.reward_collision = 0
        self.reward_lane = 0
        self.reward_action = 0
        self.c_collision = 30.
        self.c_lane = 2 #2 #0.8
        self.c_action = 0#0.05
        self.c_speed= 0.05 #0.14 #0.06
        self.done = False
        self.acc0_old= 0
        self.acc1_old= 0

    def reset(self):
        """
        Resets environment for a new episode.
        """
        # print("Agent reset is called by {}".format(self._id))
        self.speedLimit = 0
        self.current_speed= 0

        #statistic cal.
        if self.done:
            self.route_statistics()
            self.steps = 0
            self.num_episodes += 1
            self.allWrongChgLane=0
            self.routeCompletion = 0
            self.sumSpeedCompliance=0
            self.oldKeepLane = 0
            self.episodeReward = 0
            if self.num_episodes <=0:
                self.acc_controle_coeff= 0.8 - 0.02 * self.num_episodes 
            else:
                self.acc_controle_coeff= 0
            self.acceleration = 0.0
            self.rand_speed_coeff= np.random.randint(0, 9)
            self.flag_speedLimit  = 0
            self.acc0_old= 0
            self.acc1_old= 0

        #reward cal.
        self.reward = 0
        self.reward_speedLimit = 0
        self.reward_collision = 0
        self.reward_lane = 0
        self.reward_action = 0
        self.done = False
        #self.rand_speed_coeff= np.random.randint(0, 9) #Totest if the agent can adjust the speed to new targets. Till know not working

    
        #print("resetting agent (in Agent class)", self._id)
        #im, _, _ = ray.get(self.sard_buffer.get_sards.remote([0,0,0])) This is not working, because sometimes the worker is resetting the agents by its own , e.g max amount of episode is reached
        im = np.zeros((self.width, self.height, 4))  #ToDo Shawan: Change here to a more dynamic approach using spaces.Box
        im = im.astype(np.float32)
        im = im/128 - 1.0
        return im  

    def _get_observation(self,actions):
        """
        Makes API call to simulator to capture a camera image which is saved to disk,
        loads the captured image from disk and returns it as an observation.
        """
        self._processActions(actions)
        im, scalarInput, rewards, self.done, self.key = ray.get(self.sard_buffer.get_sards.remote(self._id, self.key, [self.control.steering,self.control.throttle,self.control.braking]))
        if scalarInput:
            raw_speed= scalarInput[0]/10
            if raw_speed <0:
                raw_speed=0 
            self.speedLimit = raw_speed + self.rand_speed_coeff * 0.25
            self.current_speed= scalarInput[1]
            """
            if random.random() < 0.01 and self.flag_speedLimit==0:  # 01% chance to set speedLimit to 0
                self.flag_speedLimit = 1
            elif self.flag_speedLimit>0:
                self.speedLimit = 0
                print("STOP SIGN NR: ",self.flag_speedLimit)
                self.flag_speedLimit += 1
                if self.flag_speedLimit ==20:
                    self.flag_speedLimit = 0
            """
                
        
        #im = im[20:205, (360 - 244) // 2:(360 + 224) // 2]  # result is ~ 180x180
        im = cv2.resize(im, (self.width, self.height))
        showImage(im, str(self._id))
        im = im / (128.0)
        im = im - 1.0
        scalarArray = np.zeros((self.width, self.height,1))
        scalarArray[0:8,0:8]= np.full((8,8,1),round((self.speedLimit-4)/8 ,1) )
        scalarArray[8:16,8:16]= np.full((8,8,1),round((self.current_speed-4)/8 ,1) )
        scalarArray[16:24,16:24]= np.full((8,8,1), round(self.acceleration ,1))
        obs = np.concatenate((im,scalarArray), axis=-1)
        obs = obs.astype(np.float32)
        calRewards=self._calculate_reward(rewards)
        return obs, calRewards, self.done

    def _calculate_reward(self, rewards):
        """
        Reward is calculated based on distance travelled.
        Name of the leaderboard tests:
        -RouteCompletionTest
        -CollisionTest
        -InRouteTest
        -AgentBlockedTest
        -CheckKeepLane
        -CheckDiffVelocity
        """
        self.steps +=1
        if isinstance(rewards, dict):
            if len(rewards)!=0:
                #self.reward_speedLimit = rewards["CheckDiffVelocity"] #reward_speed_shaping = abs(self.ego.state.speed - 14.) #when reaching the goal speed then no punishment
                self.reward_collision = rewards["CollisionTest"]
                self.routeCompletion = rewards["RouteCompletionTest"]  
                if self.oldKeepLane != rewards["CheckKeepLane"]:
                    self.reward_lane = abs(self.oldKeepLane -rewards["CheckKeepLane"])
                    self.oldKeepLane = rewards["CheckKeepLane"]
                    self.allWrongChgLane+=1
                else:
                    self.reward_lane=0

        
        self.reward_speedLimit = abs(self.speedLimit - self.current_speed)
        self.sumSpeedCompliance+=self.reward_speedLimit 

        if self.reward_speedLimit < 0.3:
            self.reward_speedLimit = 0
        elif self.current_speed < 0.1:#(self.current_speed < 0.1 and self.flag_speedLimit==0) or (self.current_speed > 0.1 and self.flag_speedLimit>0):
            self.reward_speedLimit = 5 *self.reward_speedLimit
                    
        

        print("Agent has ID {} and is in episode {}".format(self._id,self.num_episodes))
        print("predicted steering: {}, throttle: {}, braking: {}".format(self.control.steering,self.control.throttle,self.control.braking))
        print("Used self.acc_controle_coeff", self.acc_controle_coeff)
        print("Agents current speed is: ", self.current_speed)
        print("The Target speed limit is: ", self.speedLimit)
        print("Previous acceleration was: ", self.acceleration)
        self.reward_action = abs(self.control.steering)
    
        print({'reward_type': ['coefficient', 'reward_value(not weighted)', 'reward_value_weighted'],
               'collision': [self.c_collision, self.reward_collision, self.c_collision * self.reward_collision],
               'lane': [self.c_lane, self.reward_lane, self.c_lane * self.reward_lane],
               'action': [self.c_action, self.reward_action, self.c_action * self.reward_action],
               'speedDiff': [self.c_speed, self.reward_speedLimit, self.c_speed * self.reward_speedLimit]})
        
        """'traveled distance': [self.c_traveled_dist, self.reward_traveled_dist,
                                     self.c_traveled_dist * self.reward_traveled_dist]
               })"""
        
        
        self.reward = -(self.c_collision * self.reward_collision) - (self.c_lane * self.reward_lane) - (self.c_speed * self.reward_speedLimit)
        self.reward = self.reward/5
        self.episodeReward += self.reward
        return self.reward

    def _processActions(self,action):
        jsonable = self.action_space.to_jsonable(action)
        steering = jsonable[0]
        self.control.steering = steering
        if self.output == 1:
            if self.current_speed < self.speedLimit/3:
                self.control.throttle = 1.5
                self.control.braking = 0.0
            elif self.current_speed < self.speedLimit/2:
                self.control.throttle = 1.0
                self.control.braking = 0.0
            elif self.current_speed < self.speedLimit:
                self.control.throttle = 0.5
                self.control.braking = 0.0
            elif self.current_speed/2 > self.speedLimit:
                self.control.throttle = 0.0
                self.control.braking = 0.2
            else:
                self.control.throttle = 0.0
                self.control.braking = 0.0
        elif self.output == 2 and MIMIC==False:
            self.acceleration =  0.5*self.acceleration + 0.5*jsonable[1] + 0.2 #+self.acc_controle_coeff
            if self.acceleration>=1:
                self.acceleration=1
            elif self.acceleration <=-1:
                self.acceleration=-1

            if self.acceleration >= 0:
                # positive acceleration
                self.control.throttle = np.abs(self.acceleration)
                self.control.braking = 0.0
            else:
                self.control.throttle = 0.0
                self.control.braking = np.abs(self.acceleration)  
        elif self.output == 2 and MIMIC==True:
            if self.current_speed*3 < self.speedLimit:
                newAcceleration = 1.5
            elif self.current_speed*2 < self.speedLimit:
                newAcceleration = 1
            elif self.current_speed < self.speedLimit:
                newAcceleration = 0.5
            elif self.current_speed/2 > self.speedLimit:
                newAcceleration = -0.2
            else:
                newAcceleration = 0.0

            #self.acceleration += jsonable[1]
            self.acceleration = newAcceleration+ jsonable[1] #+self.acceleration
            if self.acceleration>=2:
                self.acceleration=2
            elif self.acceleration <=-3:
                self.acceleration=-3

            if self.acceleration >= 0:
                # positive acceleration
                self.control.throttle = np.abs(self.acceleration)
                self.control.braking = 0.0
            else:
                self.control.throttle = 0.0
                self.control.braking = np.abs(self.acceleration)  
        elif self.output == 3 and MIMIC==False:
            acceleration_1 = jsonable[1] +self.acc_controle_coeff
            acceleration_2 = jsonable[2]
            self.control.throttle = acceleration_1
            self.control.braking = acceleration_2
        elif self.output == 4 and MIMIC==False:
            acc0=jsonable[1] 
            acc1=0.5*jsonable[2] + 0.5*self.acc0_old
            acc2=0.5*jsonable[3] + 0.5*self.acc1_old
            self.acceleration =  0.5*acc0 + 0.3*acc1 + 0.2*acc2 + 0.2 #+self.acc_controle_coeff
            if self.acceleration>=1:
                self.acceleration=1
            elif self.acceleration <=-1:
                self.acceleration=-1

            if self.acceleration >= 0:
                # positive acceleration
                self.control.throttle = np.abs(self.acceleration)
                self.control.braking = 0.0
            else:
                self.control.throttle = 0.0
                self.control.braking = np.abs(self.acceleration)  
            self.acc0_old= jsonable[1]
            self.acc1_old= jsonable[2]

        

    def route_statistics(self):
        #ToDo: Distance to the center of the lane must be implemented. Here are some works that deal with this issue: https://github.com/carla-simulator/carla/issues/992
        if self.sumSpeedCompliance !=0:
            speed_compliance = self.sumSpeedCompliance/self.steps
        else:
            speed_compliance =0

        data = [self.num_episodes,self.key, self.reward_collision, self.allWrongChgLane, 0, self.routeCompletion, self.steps, speed_compliance,self.episodeReward]
        with open(self.file_path, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)
            f.close()
