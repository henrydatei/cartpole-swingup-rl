import math
from multiprocessing.dummy import DummyProcess
from tkinter.messagebox import NO
from typing import Optional, Union
import torch
import numpy as np
import pygame
from pygame import gfxdraw

import gym
from gym import spaces, logger
from gym.utils import seeding


import time
import cv2
import RPi.GPIO as GPIO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.logger import configure

import os
import matplotlib.pyplot as plt
import pandas as pd
from torch import deg2rad
import random
from stable_baselines3.common.vec_env import DummyVecEnv
cam=cv2.VideoCapture(-1)
#writer=cv2.VideoWriter('test_with penlt_recent1.avi',cv2.VideoWriter_fourcc('M','J','P','G'),10,(640,480))


#cam.release()

class CartPoleEnv(gym.Env):  # 5angles in 
        
    def __init__(self):

      
        GPIO.setmode(GPIO.BCM)
        self.dirt=22
        self.stp=27
        self.button=12
        self.led=13
        GPIO.setup(self.dirt,GPIO.OUT)
        GPIO.setup(self.stp,GPIO.OUT)
        GPIO.setup(self.button, GPIO.IN, pull_up_down=GPIO.PUD_UP)    # set buttonPin to PULL UP INPUT mode
        GPIO.setup(self.led, GPIO.OUT)   # set the ledPin to OUTPUT mode
        GPIO.output(self.led, GPIO.LOW)
        
        self.x=0
        self.x_threshold = 0.4

        high = np.array(
            [
                self.x_threshold * 2,
                2*math.pi,
                2*math.pi,
                2*math.pi,
                2*math.pi,
                2*math.pi,
                
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                1
            ],
            dtype=np.float32,
        )
        low= np.array(
            [
                -1*self.x_threshold * 2,
                0,
                0,
                0,
                0,
                0,
                
                -1*np.finfo(np.float32).max,
                -1*np.finfo(np.float32).max,
                0
            ],
            dtype=np.float32,
        )
        self.low_a=np.array([-1.0],dtype=np.float32)
        self.high_a=np.array([1.0],dtype=np.float32)
        

        self.action_space = spaces.Box(low=self.low_a,high=self.high_a,dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        
        self.steps_beyond_done = None
        
        self.time1=[]
        self.anglelist=[]
        self.positionlist=[]

       
    def angle(self):
       
        self.camera=cam
        x_medium=314
        y_medium=379
        
        #count=0
        #while count <2:    
        #count +=1
        #self.time1.append(time.time())

                    
                    
        #for i in range(6):
        _,image=self.camera.read()
       
        hsv_frame=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        low_yellow=np.array([20,180,70],np.float32 )
        high_yellow=np.array([35,255,130],np.float32)

        yellow_mask=cv2.inRange(hsv_frame,low_yellow,high_yellow)
        
        low_pur = np.array([150,95,95])
        high_pur = np.array([180,255,255])
        
        pur_mask=cv2.inRange(hsv_frame,low_pur,high_pur)
     
        contours_y,_=cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours_y,key=lambda x:cv2.contourArea(x),reverse=True)
        
        if len(contours_y)==0:
            contours_p, _ = cv2.findContours(pur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours_p, key=lambda x:cv2.contourArea(x), reverse=True)
    
        for cnt in contours:
            (x,y,w,h)=cv2.boundingRect(cnt)

            x_medium=int((x+x+w)/2)
            y_medium=int((y+y+h)/2)

            break

        #cv2.line(image,(320,385),(x_medium,y_medium),(255,0,255),2)
        x_ori=322
        y_ori=372
        cv2.line(image ,(x_ori, y_ori), (x_medium,y_medium), (255,0,255), 2)
        #fname=f"img_{int(time.time())}.jpg"
        #cv2.imwrite(fname,image)
        #writer.write(image)
        if y_medium==y_ori and x_medium>x_ori:
        #print("X+")
            self.theta=math.pi/2
        elif y_medium==y_ori and x_medium<x_ori:
            #print("X-")
            self.theta=1.5*math.pi
        elif y_medium>y_ori and x_medium==x_ori:
            #print("Y-")
            
            self.theta=math.pi
        elif y_medium<y_ori and x_medium==x_ori:
            #print("Y+")
            
            self.theta=0
        
        elif y_medium<y_ori and x_medium>x_ori:
            #print("Q1")
            self.theta=math.atan((x_medium-x_ori)/(y_ori-y_medium))
        
        elif y_medium>y_ori and x_medium>x_ori:
            #print("Q2")
            
            self.theta=math.pi-math.atan((x_medium-x_ori)/(y_medium-y_ori))
        
        elif y_medium>y_ori and x_medium<x_ori:
            #print("Q3")
            
            self.theta=math.pi+math.atan((x_ori-x_medium)/(y_medium-y_ori))
        
        elif y_medium<y_ori and x_medium<x_ori:
        # print("Q4")
            
            self.theta=2*math.pi-math.atan((x_ori-x_medium)/(y_ori-y_medium))
        
        if len(contours_y)==0:
            self.theta=math.pi+self.theta
            if self.theta > 2*math.pi:
                self.theta=self.theta-2*math.pi
        
        #self.anglelist.append(self.theta)
        #if count>1:
            #   self.thetadot=(self.anglelist[-1]-self.anglelist[-2])/(self.time1[-1]-self.time1[-2])
            #print(theta,thetadot)

        # key=cv2.waitKey(1) & 0xFF
        
        #self.camera.release()
        #print(theta)
        #if key == ord("q"):
            #  cv2.destroyAllWindows()
            # break
        
        
        return self.theta 

    def movemotor(self): # to manually move cart to centre or any other position
       
        move1= input('please input distance')
        
        move1=float(move1)
        move=5*move1
        move=int(move)

        if move<0: #CCW (away from motor)
            move=abs(move)

            GPIO.output(self.dirt,GPIO.LOW)
        else:
            GPIO.output(self.dirt,GPIO.HIGH)

        for i in range(move):
            time.sleep(0.0005)
            GPIO.output(self.stp,GPIO.HIGH)
            time.sleep(0.0005)
            GPIO.output(self.stp,GPIO.LOW)
    
        return 'cart is in centre'

    def reset(self):
        self.cnt=0
        time.sleep(random.randint(15,60))
        #self.state = self.x,self.angle(),0,0
        #if self.x >0.420 or self.x < -0.42:
        if self.x >=0 or self.x < 0:

            #move1=-425 if self.x >.420 else 420
            #move1=-self.x*1000 if self.x >0 else -self.x*1000
            move1 = -872
            move1=float(move1)
            move=5*move1
            move=int(move)

            if move<0: #CCW (away from motor)
                move=abs(move)
                GPIO.output(self.dirt,GPIO.LOW)
            else:
                GPIO.output(self.dirt,GPIO.HIGH)

            for i in range(move):
                time.sleep(0.0005)
                GPIO.output(self.stp,GPIO.HIGH)
                time.sleep(0.0005)
                GPIO.output(self.stp,GPIO.LOW)
                if GPIO.input(self.button)==GPIO.LOW: # if button is pressed
                   # turn on led
                    print ('Button is presesd')
                    break
            
            move1 = 436
            move1=float(move1)
            move=5*move1
            move=int(move)

            if move<0: #CCW (away from motor)
                move=abs(move)
                GPIO.output(self.dirt,GPIO.LOW)
            else:
                GPIO.output(self.dirt,GPIO.HIGH)

            for i in range(move):
                time.sleep(0.0005)
                GPIO.output(self.stp,GPIO.HIGH)
                time.sleep(0.0005)
                GPIO.output(self.stp,GPIO.LOW)
                
            self.x=0

        #input('press enter to reset')
        ang=[]
        time1=[]
        for _ in range(10):
        
            ang.append(self.angle())
            time1.append(time.time())

        dtheta=[]
        for z in range(5,9,1):
            
            if ang[z]>1.5*np.pi and ang[z+1]<0.5*np.pi:
                w=ang[z]-2*np.pi 
                dtheta.append(ang[z+1]-w)
            
            elif ang[z]<0.5*np.pi and ang[z+1]>1.5*np.pi:
                w=ang[z+1]-2*np.pi 
                dtheta.append(w-ang[z])
            
            else:
                dtheta.append(ang[z+1]-ang[z])


        dt= [ time1[-5]- time1[-4], time1[-4]- time1[-3], time1[-3]- time1[-2], time1[-2]- time1[-1] ] 
        theta1,theta2,theta3,theta4,theta5 = ang[-5:]
        theta_dot=np.sum(np.divide(dtheta,dt))/len(dt)
        dtheta1=abs(np.array(dtheta))
        poleup=bool(
            any(_ > np.deg2rad(345) for _ in ang) 
            or any(_ < np.deg2rad(15) for _ in ang)
            and any(_ < np.deg2rad(10) for _ in dtheta1) )  
        self.state = self.x,theta1,theta2,theta3,theta4,theta5,0,theta_dot, poleup
        self.steps_beyond_done = None
        
        return np.array(self.state, dtype=np.float32)

    
    def step(self, action: np.ndarray):
        assert self.state is not None, "Call reset before using step method."
        self.x, theta1,theta2,theta3,theta4,theta5, x_dot, theta_dot,poleup = self.state
        self.time1.append(time.time())
        self.cnt+=1
        self.positionlist.append(self.x)
        ang1=[]
        #self.anglelist.append(theta2)
        #self.shutter=0.0005
        move1=min(max(action[0], self.low_a[0]), self.high_a[0] )
       
        move1=float(move1)
        move1=25*move1
       
        move1 = move1 if self.x +move1/1000 <0.425 and self.x +move1/1000 >-0.425 else 0
        
        if move1==0:
            for _ in range(5):
        
                ang1.append(self.angle())
                self.time1.append(time.time())
        else:

            if move1<1.5 and move1 > -1.5:
                move1=1.5 if move1<1.5 and move1 > 0 else -1.5
            
            self.shutter = 0.0125/abs(move1)
            
            move=5*move1
            move=int(move)
        
            if move<0: #CCW (away from motor)
                move=abs(move)

                GPIO.output(self.dirt,GPIO.LOW)
            else:
                GPIO.output(self.dirt,GPIO.HIGH)
        
            for i in range(move):
                time.sleep(self.shutter)
                GPIO.output(self.stp,GPIO.HIGH)
                time.sleep(self.shutter)
                GPIO.output(self.stp,GPIO.LOW)
                if i>(move-6):
                
                    self.time1.append(time.time())
                    #hello = time.time()
                    ang1.append(self.angle())
                    #print(time.time()-hello)
                    self.time1.append(time.time())
                
        self.x = self.x + move1/1000
        #self.x = self.x if self.x <=0.425 and self.x >=-0.425 else 0.425 if self.x >0.425 else -0.425
        #x_dot = x_dot + self.tau * xacc
        self.positionlist.append(self.x)
       # x_dot = 0.17 if action[1] >0 else -0.17
                
        theta1,theta2,theta3,theta4,theta5 = ang1
        
        
        dtheta=[]
       
        for z in range(len(ang1)-1):
            if ang1[z]>1.5*np.pi and ang1[z+1]<0.5*np.pi:
                w=ang1[z]-2*np.pi 
                dtheta.append(ang1[z+1]-w)
            
            elif ang1[z]<0.5*np.pi and ang1[z+1]>1.5*np.pi:
                w=ang1[z+1]-2*np.pi 
                dtheta.append(w-ang1[z])
            
            else:
                dtheta.append(ang1[z+1]-ang1[z])


        dt= [self.time1[-9]-self.time1[-7],self.time1[-7]-self.time1[-5],self.time1[-5]-self.time1[-3],self.time1[-3]-self.time1[-1] ] 
        
        #self.time1.append(time.time())
        x_dot=(self.positionlist[-1]-self.positionlist[-2])/(self.time1[-10]-self.time1[-11])
       # self.anglelist.append(theta)
        theta_dot=np.sum(np.divide(dtheta,dt))/len(dt)
        #theta_dot = (theta6-theta5)/(self.time1[-1]-self.time1[-3])
        #angtime= [self.time1[-13]-self.time1[-12],self.time1[-12]-self.time1[-11],self.time1[-11]-self.time1[-10],self.time1[-10]-self.time1[-9],self.time1[-9]-self.time1[-8],self.time1[-8]-self.time1[-7],self.time1[-7]-self.time1[-6],self.time1[-6]-self.time1[-5],self.time1[-5]-self.time1[-4],self.time1[-4]-self.time1[-3],self.time1[-3]-self.time1[-2],self.time1[-2]-self.time1[-1]]        
        dtheta1=abs(np.array(dtheta))
        poleup = bool(
            any(_ > np.deg2rad(345) for _ in ang1) 
            or any(_ < np.deg2rad(15) for _ in ang1)
            and any(_ < np.deg2rad(10) for _ in dtheta1) )
        
        self.state = (self.x, theta1,theta2, theta3,theta4,theta5, x_dot, theta_dot,poleup)
        if poleup==1:
            GPIO.output(self.led, GPIO.HIGH)
        else:
            GPIO.output(self.led, GPIO.LOW)  
                  
        done = bool(
            #self.x < -self.x_threshold
            #or self.x > self.x_threshold
            self.cnt>2047
            #or theta < -self.theta_threshold_radians
            #or theta > self.theta_threshold_radians
        )

        if not done:
            reward_fn = np.exp((math.cos(theta1)+math.cos(theta2)+math.cos(theta3)+math.cos(theta4)+math.cos(theta5))/2)
            
            #if any(ang1) >np.deg2rad(345) or any(ang1)< np.deg2rad(15) and any(dtheta1)>np.deg2rad(10):
            penalty_hit = -10 if self.x < -self.x_threshold or self.x > self.x_threshold else 0   
            penalty = reward_fn*(np.sum(dtheta1)/(len(dtheta1)*2*np.pi))
                
            #else:
             #   penalty=0
            reward = reward_fn-penalty + penalty_hit
            #if theta < math.radians(15) or theta >math.radians(345):
             #  reward = 12
            #else:
             #   reward = 0
            #elif theta < math.radians(45) or theta > math.radians(315):
             #   reward = 2+2*math.cos(theta)
            #else:    
             #   reward = 1+math.cos(theta)
        elif self.steps_beyond_done is None:
            
            self.steps_beyond_done = 0
            reward = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = -1

        return np.array(self.state, dtype=np.float32), reward, done,{}#[dtheta,dt,reward_fn,penalty]#angtime 

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 10 episodes
              mean_reward = np.mean(y[-10:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model at {} timesteps".format(x[-1]))
                    print("Saving new best model to {}.zip".format(self.save_path))
                  self.model.save(self.save_path)

        return True

#tmp_path='/tmp/ppo_loger310522'
#log_dir='/tmp/ppo_pole310522'

tmp_path='/home/pi/ppo_loger_allnew5_150722'

log_dir='/home/pi/ppo_pole_allnew5_150722'

os.makedirs(log_dir,exist_ok=True)

#env=CartPoleEnv3()
env=CartPoleEnv()
env=Monitor(env,log_dir)
new_logger=configure(tmp_path,['stdout','csv','tensorboard'])
callback = SaveOnBestTrainingRewardCallback(check_freq=2048, log_dir=log_dir, verbose=1)


env.movemotor()

model = PPO("MlpPolicy", env=env, verbose=0,batch_size=128,policy_kwargs=dict(net_arch=[dict(pi=[128,128], vf=[128,128])]))
model.set_logger(new_logger)
model.learn(total_timesteps=300000,reset_num_timesteps=False,callback=callback)

model=PPO.load('ppo_allnew4100k150722',env=env)
model.set_logger(new_logger)
model.learn(total_timesteps=50000,reset_num_timesteps=False,callback=callback)



model=PPO.load('ppo_3000k',env=env)
model1=PPO.load('ppo_3000k')
model1.set_env(env)
model1.set_logger(new_logger)
model1.learn(total_timesteps=200000,reset_num_timesteps=True,callback=callback)

model1.learning_rate
model1._total_timesteps
model.set_logger(new_logger)
model.learn(total_timesteps=100000,reset_num_timesteps=False)#,callback=callback)

obs_1000k280622=pd.DataFrame(env.get_allobservations())
rew_1000k280622=pd.DataFrame(env.get_allreward())
env.get_total_steps()
obs_1000k280622.to_csv('observations_allnew5400k190722.csv')
rew_1000k280622.to_csv('rewards_allnew5400k190722.csv')

model.save('ppo_allnew5400k190722')

model=PPO.load('ppo_allnew1600k010722',env=env)
for i in range(5):
    env.angle()

allaction=[]
all_episode_rewards=[]
timeact=[]
timeeps=[]
obss=[]
info1=[]
episode_rewards = []
model1=PPO.load('best_model1_2',env=env)
for i in range(1):
    
    done= False
    obs= env.reset()
    timeeps.append(time.time())
    for _ in range(500):
        timeact.append(time.time())
        
        action,_ = model1.predict(obs)
        #action=env.action_space.sample()
        #action=np.array([1],dtype=np.float32)
        t1=time.time()
        
        obs, reward, done, info = env.step(action)
        print(obs)
            #print(info)
        print(reward)#time.time()-t1
        #env.angle()
        episode_rewards.append(reward)
        allaction.append(action)
        obss.append(obs)
        #info1.append(info)
    
    all_episode_rewards.append(sum(episode_rewards))

all_episode_rewards[1]-all_episode_rewards[0]
model1.num_timesteps


actions=pd.DataFrame(allaction)
actions.to_csv('actions_3M.csv')
timeac=pd.DataFrame(timeact)
timeac.to_csv('timeact_3M.csv')

obser=pd.DataFrame(obss)
obser.to_csv('obs_3M.csv')
rewards=pd.DataFrame(episode_rewards)
rewards.to_csv('rewards_3M.csv')

import torch.nn as nn
class FeedForwardNN(nn.Module):

    def __init__(self, input, output, layers):
        super().__init__()
        
        layerlist = []
        
        for i in layers:
            layerlist.append(nn.Linear(input,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            input = i
        layerlist.append(nn.Linear(layers[-1],output))
            
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, X):
        x = self.layers(X)
        return x

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model11=torch.load('model_220922_2000',map_location=torch.device(device))
model11=torch.load('model_290922_2700',map_location=torch.device(device))

obs_min=np.array([-0.399998,-2.955726,-3.416839,-1.361638,-1.994463,-0.755379,-0.186696,-74.526435])
obs_max=np.array([0.408297,20.770255,9.668610,8.115521,6.994965,7.262312,0.193296,160.672284])

  
obs= env.reset()
episode_rewards=[]
allobs=[]
allaction=[]
for _ in range(2048):
    obs_inp=torch.tensor((obs[0:-1]-obs_min)/(obs_max-obs_min),dtype=torch.float) 

    with torch.no_grad():
        act = model11(obs_inp)  #range 0 to 1
    action=act*(2) -1
    #action=env.action_space.sample()
    #action=np.array([1],dtype=np.float32)
    #action=np.float(-0.5)
    obs, reward, done, info = env.step(action)
    #print(obs)
    
    print(reward,' ; ', action )
        #print(info)
   # print(reward)#time.time()-t1
    #env.angle()
    episode_rewards=+reward
    allaction.append(action)
    allobs.append(obs)

allobs[0]
allaction