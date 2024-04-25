import math
import torch
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
import time
import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2, Preview
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch import deg2rad
import random
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.logger import configure
from libcamera import controls
#cam=cv2.VideoCapture(-1)
cam = Picamera2()
#picam.preview_configuration.main.size=(1280,720)
cam.preview_configuration.main.format="RGB888"
cam.preview_configuration.align()
cam.configure("preview")
cam.start()
cam.set_controls({"AfMode":controls.AfModeEnum.Continuous})

#cam.release()

class CartPoleEnv(gym.Env):  # 1angle which will come after 6 angles 
        
    def __init__(self):

      
        GPIO.setmode(GPIO.BCM)
        self.dirt=22
        self.stp=27
        self.button=12
        self.led=13
        self.trigPin = 23
        self.echoPin = 24
        self.MAX_DISTANCE = 100         # define the maximum measuring distance, unit: cm
        self.timeOut = self.MAX_DISTANCE*60
        GPIO.setup(self.dirt,GPIO.OUT)
        GPIO.setup(self.stp,GPIO.OUT)
        GPIO.setup(self.button, GPIO.IN, pull_up_down=GPIO.PUD_UP)    # set buttonPin to PULL UP INPUT mode
        GPIO.setup(self.led, GPIO.OUT)   # set the ledPin to OUTPUT mode
        GPIO.output(self.led, GPIO.LOW)
        GPIO.setup(self.trigPin, GPIO.OUT)   # set trigPin to OUTPUT mode
        GPIO.setup(self.echoPin, GPIO.IN)
        
        self.x=0
        self.x_threshold = 0.425

        high = np.array(
            [
                self.x_threshold * 2,   #postion
                2*math.pi, 
                             
                2*math.pi,
                2*math.pi,
                2*math.pi,
                2*math.pi,             #angle
                np.finfo(np.float32).max,#cart_speed
                np.finfo(np.float32).max, #angular_speed
                1
            ],
            dtype=np.float32,
        )
        low= np.array(
            [
                -1*self.x_threshold * 2,
                0,
                0,0,0,0,
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
        x_medium=296
        y_medium=405
        
        #count=0
        #while count <2:    
        #count +=1
        #self.time1.append(time.time())

        #for i in range(6):
        #_,image=self.camera.read()
        image=cam.capture_array()
        hsv_frame=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        low_pur = np.array([20, 90, 90], np.uint8)
        high_pur = np.array([30, 255, 255], np.uint8)
        low_yellow = np.array([100, 150, 30], np.uint8)
        high_yellow = np.array([120, 255, 255], np.uint8)
        
        #low_yellow=np.array([20,180,70],np.float32 )
        #high_yellow=np.array([35,255,130],np.float32)

        yellow_mask=cv2.inRange(hsv_frame,low_yellow,high_yellow)
        
        #low_pur = np.array([150,95,95])
        #high_pur = np.array([180,255,255])
        
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
        x_ori=293
        y_ori=405
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
    
    def pulseIn(self,pin,level,timeOut): # obtain pulse time of a pin under timeOut
        t0 = time.time()
        while(GPIO.input(pin) != level):
            if((time.time() - t0) > timeOut*0.000001):
                return 0
        t0 = time.time()
        while(GPIO.input(pin) == level):
            if((time.time() - t0) > timeOut*0.000001):
                return 0
        pulseTime = (time.time() - t0)*1000000
        return pulseTime
    
    def getSonar(self):     # get the measurement results of ultrasonic module,with unit: cm
        GPIO.output(self.trigPin,GPIO.HIGH)      # make trigPin output 10us HIGH level 
        time.sleep(0.00001)     # 10us
        GPIO.output(self.trigPin,GPIO.LOW) # make trigPin output LOW level 
        pingTime = self.pulseIn(self.echoPin,GPIO.HIGH,self.timeOut)   # read plus time of echoPin
        distance = pingTime * 340.0 / 2.0 / 10000.0     # calculate distance with sound speed 340m/s 
        
        return (distance/100-0.035)
    
    def movemotor(self,move1): # to manually move cart to centre or any other position
       
        #move1= input('please input distance')
        
        move1=float(move1)
        move=int(5*move1)
    

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
        #time.sleep(random.randint(15,60))
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
        for _ in range(5):
        
            ang.append(self.angle())
            time1.append(time.time())
        
        dtheta=[]
        for z in range(len(ang)-1):
            
            if ang[z]>1.5*np.pi and ang[z+1]<0.5*np.pi:
                w=ang[z]-2*np.pi 
                dtheta.append(ang[z+1]-w)
            
            elif ang[z]<0.5*np.pi and ang[z+1]>1.5*np.pi:
                w=ang[z+1]-2*np.pi 
                dtheta.append(w-ang[z])
            
            else:
                dtheta.append(ang[z+1]-ang[z])
        

        dt= [ time1[-4]- time1[-5], time1[-3]- time1[-4], time1[-2]- time1[-3], time1[-1]- time1[-2] ] 
        #dt=time1[-2]-time1[-1]
        theta1,theta2,theta3,theta4,theta5 = ang
        theta_dot=np.sum(np.divide(dtheta,dt))/len(dt)
        dtheta1=abs(np.array(dtheta))
        poleup=bool(
            any(_ > np.deg2rad(345) for _ in ang) 
            or any(_ < np.deg2rad(15) for _ in ang)
            and any(_ < np.deg2rad(10) for _ in dtheta1) )  
        #self.x=self.getSonar()
        self.state = self.x,theta1,theta2,theta3,theta4,theta5,0,theta_dot, poleup
        self.steps_beyond_done = None
        self.nfo=np.copy(self.state)
        return np.array(self.state, dtype=np.float32)

    
    def step(self, action: np.ndarray):
        assert self.state is not None, "Call reset before using step method."
        self.x, theta1,theta2,theta3,theta4,theta5, x_dot, theta_dot,poleup = self.state
        self.time1=[]
        self.time1.append(time.time())
        self.cnt+=1
        self.positionlist.append(self.x)
        ang1=[] #old list
        ang2=[] # new list of 7 angles
        #self.anglelist.append(theta2)
        #self.shutter=0.0005
        move1=min(max(action[0], self.low_a[0]), self.high_a[0] )
       
        move1=float(move1)
        move1=25*move1
       
        move1 = move1 if self.x +move1/1000 <0.425 and self.x +move1/1000 >-0.425 else 0
        
        if move1==0:
            for _ in range(5):
                self.time1.append(time.time())
                ang1.append(self.angle())
                #self.time1.append(time.time())
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
                #print(i)
                #if  any(np.linspace(0,move,6)==i+1):
                if i>(move-6):
                    self.time1.append(time.time())
                    #hello = time.time()
                    ang1.append(self.angle())
                    #print(time.time()-hello)
                    #self.time1.append(time.time())

        #self.x=self.getSonar()      
        self.x = self.x + move1/1000
        self.x = self.x if self.x <=0.425 and self.x >=-0.425 else 0.425 if self.x >0.425 else -0.425
        #x_dot = x_dot + self.tau * xacc
        self.positionlist.append(self.x)
       # x_dot = 0.17 if action[1] >0 else -0.17
        #print('movement completed')
        #print(len(ang1))        
        theta1,theta2,theta3,theta4,theta5 = ang1
        #print(f'ang1:{ ang1}')
       # ang2=ang1[-2:]
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
        #dtheta4=dtheta[-4:]
        dt= [ self.time1[-4]- self.time1[-5], self.time1[-3]- self.time1[-4], self.time1[-2]- self.time1[-3], self.time1[-1]- self.time1[-2] ] 
        #dt=self.time1[-1]-self.time1[-3]
        #print(f'dtheta:{ dtheta}')
        #dtheta4.reverse()
        #dt.reverse()
        """for z in range(len(ang2)-1):
            if ang2[z]>1.5*np.pi and ang2[z+1]<0.5*np.pi:
                w=ang2[z]-2*np.pi 
                dtheta.append(ang2[z+1]-w)
            
            elif ang2[z]<0.5*np.pi and ang2[z+1]>1.5*np.pi:
                w=ang2[z+1]-2*np.pi 
                dtheta.append(w-ang2[z])
            
            else:
                dtheta.append(ang2[z+1]-ang2[z])

        dt= [self.time1[-13]-self.time1[-11]]
        """
        #self.time1.append(time.time())
        x_dot=(self.positionlist[-1]-self.positionlist[-2])/(self.time1[-6]-self.time1[-1])
       # self.anglelist.append(theta)
        theta_dot=np.sum(np.divide(dtheta,dt))/len(dt)
       # theta_dot=np.sum(np.divide(dtheta,dt))/len(dt)
        #theta_dot=theta_dot[0]
        #
        #print(f'theta_dot:{ theta_dot}')
        #wt=np.zeros(4)
        #for i in range(4):
         #   wt[i]= (np.sum(dt)-np.sum(dt[:i+1]))/(4*dt[0]+3*dt[1]+2*dt[2]+1*dt[3])
          #  if theta_dot1[i]>50:
           #     wt[i]=0
        #theta_dot=np.sum(np.multiply(wt,theta_dot1))

        #theta_dot = (theta6-theta5)/(self.time1[-1]-self.time1[-3])
        #angtime= [self.time1[-13]-self.time1[-12],self.time1[-12]-self.time1[-11],self.time1[-11]-self.time1[-10],self.time1[-10]-self.time1[-9],self.time1[-9]-self.time1[-8],self.time1[-8]-self.time1[-7],self.time1[-7]-self.time1[-6],self.time1[-6]-self.time1[-5],self.time1[-5]-self.time1[-4],self.time1[-4]-self.time1[-3],self.time1[-3]-self.time1[-2],self.time1[-2]-self.time1[-1]]        
        dtheta1=abs(np.array(dtheta))
        #print(f'dtheta1:{ dtheta1}')
        poleup = bool(
            any(_ > np.deg2rad(345) for _ in ang1) 
            or any(_ < np.deg2rad(15) for _ in ang1)
            and any(_ < np.deg2rad(10) for _ in dtheta1) )
        #print(f'poleup:{poleup}') 
        self.state = (self.x, theta1,theta2,theta3,theta4,theta5, x_dot, theta_dot,poleup)
        #print(f'state:{self.state}')
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
            penalty_hit = np.exp(abs(self.x/0.0425))/4250
            penalty = reward_fn*(np.sum(dtheta1)/(len(dtheta1)*2*np.pi))
                
            #else:
             #   penalty=0
            reward = reward_fn-penalty - penalty_hit
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

#tmp_path='/home/pi/New/CP2023/new_cam_5ang2/ppo_logger2_260423'

#log_dir='/home/pi/New/CP2023/new_cam_5ang2/ppo_pole2_260423'

os.makedirs(log_dir,exist_ok=True)

#env=CartPoleEnv3()
env=CartPoleEnv()
env=Monitor(env,log_dir)
new_logger=configure(tmp_path,['stdout','csv','tensorboard'])
callback = SaveOnBestTrainingRewardCallback(check_freq=2048, log_dir=log_dir, verbose=1)


model = PPO("MlpPolicy", env=env, verbose=0,batch_size=128,policy_kwargs=dict(net_arch=[dict(pi=[128,128], vf=[128,128])]))
#model=PPO.load('ppo_cp23_5ang_800k020523',env=env)
model.set_logger(new_logger)
time_start=time.time()
model.learn(total_timesteps=300000,reset_num_timesteps=False)
total_time=time.time()-time_start
obs_1400k270423=pd.DataFrame(env.get_allobservations())
rew_1400k270423=pd.DataFrame(env.get_allreward())
env.get_total_steps()
#obs_1400k270423 .to_csv('observations_cp23_5ang2_1400k070523.csv')
#rew_1400k270423.to_csv('rewards_cp23_5ang2_1400k070523.csv')

#model.save('ppo_cp23_5ang2_1400k070523')



#################Trial

"""
env.reset()
env.movemotor(40)
#env.getSonar()
env.angle()

check={}
for _ in range(0,885,15):
    
    check[f'{_}']=[]  
    for z in range(100):
        time.sleep(0.005)
        check[f'{_}'].append(env.getSonar())
    env.movemotor(15)


check_df=pd.DataFrame(check)
check_df.columns
check_df.head
check_df_describe= check_df.describe()
check_df_describe.index
pd.DataFrame(check_df_describe).iloc[1]

check_df.to_csv('sensor_readings200.csv')

plt.plot(check_df_describe.iloc[2])"""