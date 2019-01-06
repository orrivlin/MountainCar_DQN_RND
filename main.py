"""
@author: orrivlin
"""

import numpy as np
import gym
from dqn_rnd import DQN_RND
import matplotlib.pyplot as plt
from smooth_signal import smooth
import torch


env = gym.make('MountainCar-v0')

gamma = 0.95
alg = DQN_RND(env,gamma,10000)


num_epochs = 300
for i in range(num_epochs):
    log = alg.run_epoch()
    print('epoch: {}. return: {}'.format(i,np.round(log.get_current('real_return')),2))
    

Y = np.asarray(log.get_log('real_return'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig1 = plt.figure()
ax1 = plt.axes()
ax1.plot(x, Y, Y2)

Y = np.asarray(log.get_log('combined_return'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y, Y2)

obs = env.reset()
for t in range(1000):
    env.render()
    x = torch.Tensor(obs).unsqueeze(0)
    Q = alg.model(x)
    action = Q.argmax().detach().item()
    new_obs, reward, done, info = env.step(action)
    obs = new_obs
    if done:
        break
