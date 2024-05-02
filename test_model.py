from CartPoleEnv2 import CartPoleEnv2
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

myEnv = CartPoleEnv2()
model_name = "PPO_CartPoleEnv2 600k_smoothing_error_protection_reward_ankit_2step_faster_lessweight_3actions_2024-04-30_15-43-31"

model = PPO.load("models/" + model_name, env=myEnv, force_reset=True)
model.learn(total_timesteps=10000)