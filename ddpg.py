import numpy as np
import gym
import tensorflow as tf

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

ENV_NAME = "HalfCheetah-v3"

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.shape[0]

actor = tf.keras.models.Sequential()
actor.add(tf.keras.layers.Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(tf.keras.layers.Dense(32))
actor.add(tf.keras.layers.Activation('relu'))
actor.add(tf.keras.layers.Dense(16))
actor.add(tf.keras.layers.Activation('relu'))
actor.add(tf.keras.layers.Dense(nb_actions))
actor.add(tf.keras.layers.Activation('linear'))
print(actor.summary())

action_input = tf.keras.layers.Input(shape=(nb_actions,), name='action_input')
observation_input = tf.keras.layers.Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = tf.keras.layers.Flatten()(observation_input)
x = tf.keras.layers.Concatenate()([action_input, flattened_observation])
x = tf.keras.layers.Dense(32)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dense(16)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dense(1)(x)
x = tf.keras.layers.Activation('linear')(x)
critic = tf.keras.models.Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input, memory=memory,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(tf.keras.optimizers.Adam(learning_rate=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=100000, visualize=False, verbose=1, nb_max_episode_steps=200)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# agent.load_weights('ddpg_{}_weights.h5f'.format(ENV_NAME))

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)

