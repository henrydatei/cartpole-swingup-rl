from CartPoleEnv2 import CartPoleEnv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
import datetime
import os
import pandas as pd

myEnv = CartPoleEnv2()
# check_env(myEnv)

# for _ in range(100):
#     action = myEnv.action_space.sample()
#     observation, reward, done, info = myEnv.step(action)
#     print(observation, reward, done, info)

# env.reset()

model_name = "PPO_CartPoleEnv2_10k_smoothing_error_protection_reward_ankit_2step_faster_lessweight_2actions_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("logs", model_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs('models', exist_ok=True)

new_logger = configure(log_dir,['stdout','csv', 'tensorboard'])
model = PPO("MlpPolicy", myEnv, verbose=1)
model.set_logger(new_logger)
model.learn(total_timesteps=10000)

myEnv.reset()

# save model with current time
model.save(os.path.join("models", model_name))

# save observations and rewards
observations = pd.DataFrame(myEnv.all_observations, columns=['angle1','angle2', 'angle3', 'angle4', 'angle5', 'angle_velocity', 'position', 'position_velocity'])
rewards = pd.DataFrame(myEnv.all_rewards, columns=['reward'])
times = pd.DataFrame(myEnv.all_times, columns=['time'])

# merge observations, rewards and times
observations = pd.concat([observations, rewards], axis=1)
observations = pd.concat([observations, times], axis=1)
observations.to_csv(os.path.join(log_dir, 'observations_rewards_times.csv'))