import gym
from gym.spaces import Discrete, Box
import numpy as np
import math
import zmq
import time

from cartpole.Communicator import Communicator

class CartPoleEnv2(gym.Env):
    def __init__(self):
        self.max_revolutions_to_each_side = 9
        # self.angle_step = 10 # degrees
        self.max_time_steps = 2048 # episode length
        self.time_steps = 0

        # state
        self.angle = 0 # rad
        self.angle_velocity = 0 # rad/s
        self.position = 0 # comes from the stepper motor
        self.position_velocity = 0

        # self.action_space = Discrete(2 * self.max_revolutions_to_each_side * 360/self.angle_step) # possible to go from one end to the other in one step
        self.action_space = Discrete(51) # 0 -> 0 velocity, 1 -> 2000 velocity, 2 -> -2000 velocity, 3 -> 4000 velocity, ..., 49 -> 50000, 50 -> -50000 velocity
        self.observation_space = Box(
            low=np.array([-math.pi, -np.inf, -12800*self.max_revolutions_to_each_side, -50000]), 
            high=np.array([math.pi, np.inf, 12800*self.max_revolutions_to_each_side, 50000]), 
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
        self.last_message = (-math.pi, 0)

        self.all_observations = []
        self.all_rewards = []
        self.all_times = []

    def get_angle_and_velocity(self):
        message = self.socket.recv_string()
        angle, angle_velocity, pole_up = message.split(",")
        if angle == "0" and angle_velocity == "0" and pole_up == "0":
            angle, angle_velocity = self.last_message
        else:
            angle, angle_velocity = float(angle), float(angle_velocity)
            self.last_message = (angle, angle_velocity)
        return angle, angle_velocity
    
    def reward_simple(self, angle: float):
        """
        Reinforcement Learning in Continuous Time and Space
        by Kenji Doya, 2000
        similar version here: https://ieeexplore.ieee.org/abstract/document/1380086
        """
        return math.cos(angle)
    
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
        elif abs(float(self.communicator.send_message('p', 0)[1])) >= 12800*self.max_revolutions_to_each_side:
            print("Position limit reached")
            done = True
        else:
            done = False
        self.time_steps += 1

        # update state
        # if not done:
        #     self.communicator.send_message('r', action * 10)
        #     self.position += action * 10

        if math.degrees(abs(self.get_angle_and_velocity()[0])) > 12:
            # for swing up
            if action == 0:
                pass
            elif action % 2 == 0:
                self.position_velocity = 60000
            else:
                self.position_velocity = -60000
            self.communicator.send_message('v', self.position_velocity)
            time.sleep(0.1)
        else:
            # for stabilization
            self.position_velocity = math.ceil(action/2) * 2000 * (-1)**(action + 1)
            self.communicator.send_message('v', self.position_velocity)

        self.angle, self.angle_velocity = self.get_angle_and_velocity()
        self.position = float(self.communicator.send_message('p', 0)[1])
        observation = np.array([self.angle, self.angle_velocity, self.position, self.position_velocity])

        # reward
        if not done:
            # reward = self.reward_escobar_2020(self.position, self.angle, self.position_velocity/1000)
            # reward = self.reward_kimura_1999(self.position, self.angle, self.angle_velocity)
            # reward = self.reward_swing_up_stabilization(self.angle, self.angle_velocity)
            reward = self.reward_simple(self.angle)
        else:
            reward = 0

        self.all_observations.append(observation)
        self.all_rewards.append(reward)
        self.all_times.append(time.time())

        return observation, reward, done, {}
    
    def reset(self):
        print("Resetting...")
        self.communicator.send_message("a", 10000) #slow it down a bit
        self.communicator.send_message('m', 0)
        self.communicator.send_message('v', 0)
        self.communicator.send_message("a", 1000000)
        self.position = 0
        self.position_velocity = 0
        self.angle = 0
        self.angle_velocity = 0
        self.time_steps = 0
        return np.array([self.angle, self.angle_velocity, self.position, self.position_velocity])
    
    def render(self):
        pass
    
    def close(self):
        pass

    
