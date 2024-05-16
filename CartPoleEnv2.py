import gym
from gym.spaces import Discrete, Box
import numpy as np
import math
import zmq
import time
import datetime
import os
import pandas as pd

from cartpole.Communicator import Communicator

class CartPoleEnv2(gym.Env):
    def __init__(self):
        self.max_revolutions_to_each_side = 9
        # self.angle_step = 10 # degrees
        self.max_time_steps = 2048 # episode length
        self.time_steps = 0
        self.overall_time_steps = 0

        # state
        self.angle1 = 0 # rad
        self.angle2 = 0 # rad
        self.angle3 = 0 # rad
        self.angle4 = 0 # rad
        self.angle5 = 0 # rad
        self.angle_velocity = 0 # rad/s
        self.position = 0 # comes from the stepper motor
        self.position_velocity = 0
        self.pole_up = 0

        # self.action_space = Discrete(2 * self.max_revolutions_to_each_side * 360/self.angle_step) # possible to go from one end to the other in one step
        self.action_space = Discrete(2) # 1 -> 60000 velocity, 2 -> -60000 velocity
        self.observation_space = Box(
            low=np.array([-math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -np.inf, -12800*self.max_revolutions_to_each_side, -60000, 0]), 
            high=np.array([math.pi, math.pi, math.pi, math.pi, math.pi, np.inf, 12800*self.max_revolutions_to_each_side, 60000, 1]), 
        )

        self.communicator = Communicator("/dev/ttyUSB0", 115200)
        self.communicator.send_message('h', 0)
        self.communicator.send_message('s', 12800)
        self.communicator.send_message('v', 0)
        self.communicator.send_message('a', 1000000)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1) # only keep the last message
        self.socket.connect("tcp://localhost:9999")
        self.socket.subscribe("")
        self.last_message = (-math.pi, -math.pi, -math.pi, -math.pi, -math.pi, 0, 0, 0)

        self.log_dir = "logs"
        self.all_observations = []
        self.all_rewards = []
        self.all_times = []
        self.all_delays = []

    def get_angle_and_velocity(self):
        message = self.socket.recv_string()
        angle1, angle2, angle3, angle4, angle5, angle_velocity, pole_up, angle_start_time = message.split(",")
        if angle1 == "0" and angle2 == "0" and angle3 == "0" and angle4 == "0" and angle5 == "0" and angle_velocity == "0" and pole_up == "0" and angle_start_time == "0":
            angle1, angle2, angle3, angle4, angle5, angle_velocity, pole_up, angle_start_time = self.last_message
        else:
            angle1, angle2, angle3, angle4, angle5, angle_velocity, pole_up, angle_start_time = float(angle1), float(angle2), float(angle3), float(angle4), float(angle5), float(angle_velocity), int(pole_up), float(angle_start_time)
            self.last_message = (angle1, angle2, angle3, angle4, angle5, angle_velocity, pole_up, angle_start_time)
        return angle1, angle2, angle3, angle4, angle5, angle_velocity, pole_up, angle_start_time
    
    def reward_simple(self, angle: float):
        """
        Reinforcement Learning in Continuous Time and Space
        by Kenji Doya, 2000
        similar version here: https://ieeexplore.ieee.org/abstract/document/1380086
        """
        return math.cos(angle)
    
    def reward_simple_position_penalty_clipping(self, theta: float, x: float, theta_dot: float):
        angle_reward = math.cos(theta)
        rotation_position = abs(x)/12800
        position_penalty = math.exp(rotation_position/self.max_revolutions_to_each_side) - 1

        if abs(theta_dot) > 2:
            angle_reward = max(angle_reward, 0.5)  # Clip reward to a maximum value

        return angle_reward - position_penalty
    
    def reward_ankit(self, x: float, theta1: float, theta2: float, theta3: float, theta4: float, theta5: float, overall_time: int):
        # dtheta1 = (math.pi-abs(theta1)) + (math.pi-abs(theta2))
        # dtheta2 = (math.pi-abs(theta2)) + (math.pi-abs(theta3))
        # dtheta3 = (math.pi-abs(theta3)) + (math.pi-abs(theta4))
        # dtheta4 = (math.pi-abs(theta4)) + (math.pi-abs(theta5))
        angle_reward = math.exp((math.cos(theta1) + math.cos(theta2) + math.cos(theta3) + math.cos(theta4) + math.cos(theta5))/2) # [0.08, 12.18]
        # rotation_position = abs(x)/12800
        # position_penalty = math.exp(rotation_position/self.max_revolutions_to_each_side) - 1 # [0, exp(1)-1]
        # if math.degrees(abs(theta5)) < 12:
        #     angular_velocity_penalty = angle_reward * (dtheta1 + dtheta2 + dtheta3 + dtheta4) / (4 * 2 * math.pi) # [0, 1] * angle_reward
        # else:
        #     angular_velocity_penalty = 0
        # if math.degrees(abs(theta5)) > 168:
        #     no_swing_up_penalty = overall_time/50000
        # else:
        #     no_swing_up_penalty = 0

        return angle_reward
    
    def reward_escobar_2020(self, x: float, theta: float, force: float):
        """
        A Parametric Study of a Deep Reinforcement Learning Control System Applied to the Swing-Up Problem of the Cart-Pole
        by Camilo Andrés Manrique Escobar, Carmine Maria Pappalardo and Domenico Guida
        """
        A = 0.01
        B = 0.1
        C = 5
        D = -0.01
        n = 2
        r1 = (A * abs(x)**n + B * abs(theta)**n + C * abs(force)**n) * D
        return r1
    
    def reward_kimura_1999(self, x: float, theta: float, theta_dot: float):
        """
        Stochastic Real-Valued Reinforcement Learning to Solve a non-Linear Control Problem
        by H. Kimura & S. Kobayashi
        """
        if abs(theta) >= 0.8*math.pi:
            return -1
        elif abs(theta_dot) >= 10:
            return -3
        elif abs(theta) < 0.133*math.pi and abs(theta_dot) < 2:
            return 1
        else:
            return 0
        
    def reward_swing_up_stabilization(self, theta: float, theta_dot: float):
        if math.degrees(abs(theta)) < 12:
            # stabilize: small angle
            return math.cos(theta)
        else:
            # swing up, high angular velocity
            return abs(theta_dot/100)

    def step(self, action: int):
        # done?
        if self.time_steps >= self.max_time_steps:
            print("Max time steps reached")
            done = True
        # elif self.position + action * 10 > 360*self.max_revolutions_to_each_side or self.position + action * 10 < -360*self.max_revolutions_to_each_side:
        #     done = True
        # elif abs(float(self.communicator.send_message('p', 0)[1])) >= 12800*self.max_revolutions_to_each_side:
        #     print("Position limit reached")
        #     done = True
        else:
            done = False
        self.time_steps += 1
        self.overall_time_steps += 1

        # update state
        # if not done:
        #     self.communicator.send_message('r', action * 10)
        #     self.position += action * 10

        if action == 0:
            self.position_velocity = 60000
        else:
            self.position_velocity = -60000

        if float(self.communicator.send_message('p', 0)[1]) >= 12800*self.max_revolutions_to_each_side and self.position_velocity > 0:
            self.position_velocity = 0
        if float(self.communicator.send_message('p', 0)[1]) <= -12800*self.max_revolutions_to_each_side and self.position_velocity < 0:
            self.position_velocity = 0

        self.communicator.send_message('v', self.position_velocity)
        if math.degrees(abs(self.get_angle_and_velocity()[4])) > 12:
            time.sleep(0.1)

        self.angle1, self.angle2, self.angle3, self.angle4, self.angle5, self.angle_velocity, self.pole_up, angle_start_time = self.get_angle_and_velocity()
        self.position = float(self.communicator.send_message('p', 0)[1])
        self.all_delays.append(time.time() - angle_start_time)
        observation = np.array([self.angle1, self.angle2, self.angle3, self.angle4, self.angle5, self.angle_velocity, self.position, self.position_velocity, self.pole_up])

        # reward
        if not done:
            # reward = self.reward_escobar_2020(self.position, self.angle, self.position_velocity/1000)
            # reward = self.reward_kimura_1999(self.position, self.angle, self.angle_velocity)
            # reward = self.reward_swing_up_stabilization(self.angle, self.angle_velocity)
            # reward = self.reward_simple(self.angle)
            # reward = self.reward_simple_position_penalty_clipping(self.angle, self.position, self.angle_velocity)
            reward = self.reward_ankit(self.position, self.angle1, self.angle2, self.angle3, self.angle4, self.angle5, self.overall_time_steps)
        else:
            reward = 0

        self.all_observations.append(observation)
        self.all_rewards.append(reward)
        self.all_times.append(time.time())

        return observation, reward, done, {}
    
    def office_empty(self):
        # check if the office is empty
        weekday = datetime.datetime.today().weekday()
        if weekday < 5:  # monday to friday
            # check if the time is between 8:30 and 18:30
            now = datetime.datetime.now().time()
            start = datetime.time(8, 30)
            end = datetime.time(18, 30)
            if start <= now <= end:
                return False  # office is occupied
        return True  # office is empty

    def reset(self):
        print("Resetting...")
        while not self.office_empty():
            print("Wait 10 minutes...")
            time.sleep(600) # wait 10 minutes
        
        # check if file exists and if yes, delete it
        if os.path.exists(os.path.join(self.log_dir, 'observations_rewards_times.csv')):
            os.remove(os.path.join(self.log_dir, 'observations_rewards_times.csv'))
        
        # save observations and rewards
        observations = pd.DataFrame(self.all_observations, columns=['angle1','angle2', 'angle3', 'angle4', 'angle5', 'angle_velocity', 'position', 'position_velocity', 'pole_up'])
        rewards = pd.DataFrame(self.all_rewards, columns=['reward'])
        times = pd.DataFrame(self.all_times, columns=['time'])
        delays = pd.DataFrame(self.all_delays, columns=['delay'])

        # merge observations, rewards and times
        observations = pd.concat([observations, rewards], axis=1)
        observations = pd.concat([observations, times], axis=1)
        observations = pd.concat([observations, delays], axis=1)
        observations.to_csv(os.path.join(self.log_dir, 'observations_rewards_times.csv'))

        self.communicator.send_message("a", 10000) #slow it down a bit
        self.communicator.send_message('m', 0)
        self.communicator.send_message('v', 0)
        self.communicator.send_message("a", 1000000)
        self.position = 0
        self.position_velocity = 0
        self.angle1 = 0
        self.angle2 = 0
        self.angle3 = 0
        self.angle4 = 0
        self.angle5 = 0
        self.angle_velocity = 0
        self.pole_up = 0
        self.time_steps = 0
        return np.array([self.angle1, self.angle2, self.angle3, self.angle4, self.angle5, self.angle_velocity, self.position, self.position_velocity, self.pole_up])
    
    def render(self):
        pass
    
    def close(self):
        pass

    
