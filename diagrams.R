library(dplyr)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# display effect of angular velocity smoothing
prev = read.csv("logs/PPO_CartPoleEnv2_10k_2024-04-16_13-42-02/observations.csv")
now = read.csv("logs/PPO_CartPoleEnv2_10k_smoothing_error_protection_2024-04-17_10-27-34/observations_rewards_times.csv")

prev_sample = slice_sample(prev, n=5000)
now_sample = slice_sample(now, n=5000)
plot(abs(prev$X1), xlab = "Observations", ylab = "Angular Velocity", main = "Before smoothing")
plot(abs(now$angle_velocity), xlab = "Observations", ylab = "Angular Velocity", main = "After smoothing")

# display reduced actions and convergence on different reward functions
m1 = read.csv("logs/PPO_CartPoleEnv2_10m_smoothing_error_protection_reward_swingupstabilisation_2step_2024-04-18_16-37-07/progress.csv")
m2 = read.csv("logs/PPO_CartPoleEnv2_500k_smoothing_error_protection_reward_simple_2step_faster_lessweight_51actions_2024-04-23_17-51-38/progress.csv")
m3 = read.csv("logs/PPO_CartPoleEnv2 600k_smoothing_error_protection_reward_ankit_2step_faster_lessweight_3actions_2024-04-30_15-43-31/progress.csv")

plot(m1$time.total_timesteps, m1$rollout.ep_rew_mean, type = "l", xlab = "Timesteps", ylab = "Reward per Episode", main = "High Velocity Reward, 101 Actions")
plot(m2$time.total_timesteps, m2$rollout.ep_rew_mean, type = "l", xlab = "Timesteps", ylab = "Reward per Episode", main = "Simple Reward, 51 Actions")
plot(m3$time.total_timesteps, m3$rollout.ep_rew_mean, type = "l", xlab = "Timesteps", ylab = "Reward per Episode", main = "Complex Reward, 2 Actions")

# Escobar 2020
escobar = read.csv("logs/PPO_CartPoleEnv2_50k_smoothing_error_protection_reward_escobar2020_2024-04-17_11-15-45/observations_rewards_times.csv")

plot(escobar$angle, xlab = "Observations", ylab = "Angle in rad")

# Camera Delay
total_delay = read.csv("logs/PPO_CartPoleEnv2_200k_smoothing_error_protection_reward_ankit_2step_faster_lessweight_2actions_2024-05-14_10-42-44/observations_rewards_times.csv")
camera_delay = read.csv("camera_delay.txt", sep = " ", col.names = c("angle1", "angle2", "angle3", "angle4", "angle5", "angle_velocity", "camera_deay", "pole_up", "time"))

plot(total_delay$delay, xlab = "Observations", ylab = "Delay in seconds")
plot(camera_delay$camera_deay, xlab = "Observations", ylab = "Delay in seconds")
boxplot(cbind(total_delay$delay,camera_delay$camera_deay), ylab = "Delay in seconds", names = c("Total Delay", "Camera Delay"))





