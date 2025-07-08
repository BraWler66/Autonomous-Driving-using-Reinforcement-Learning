import glob
import os
import sys
import random
import time
import numpy as np
import cv2
from threading import Thread
from tqdm import tqdm
import tensorflow as tf

# Local modules
import settings
from modified_tensorboard import ModifiedTensorBoard
from DQN import DQNAgent
from CarUI import CarEnv

# Add CARLA egg to sys.path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

if __name__ == '__main__':
    ep_rewards = [-200]
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    if not os.path.isdir('models'):
        os.makedirs('models')

    agent = DQNAgent()
    env = CarEnv()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    epsilon = settings.EPSILON

    for episode in tqdm(range(1, settings.EPISODES + 1), ascii=True, unit='episodes'):
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1

        current_state = env.reset()
        done = False

        while True:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, 3)
                time.sleep(1/60)

            # Unpack 3 values, not 4
            new_state, reward, done, _ = env.step(action)

            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state

            if done:
                break

        env.destroy_actors()

        ep_rewards.append(episode_reward)

        if not episode % settings.AGGREGATE_STATS_EVERY or episode == 1:
            avg_reward = np.mean(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
            min_reward = np.min(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
            max_reward = np.max(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            if min_reward >= settings.MIN_REWARD:
                agent.model.save(f"models/{settings.MODEL_NAME}__{max_reward:.2f}max_{avg_reward:.2f}avg_{min_reward:.2f}min__{int(time.time())}.model")

        if epsilon > settings.MIN_EPSILON:
            epsilon *= settings.EPSILON_DECAY
            epsilon = max(settings.MIN_EPSILON, epsilon)

    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f"models/{settings.MODEL_NAME}__final__{int(time.time())}.model")
