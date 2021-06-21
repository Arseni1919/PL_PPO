import random
import numpy as np
import supersuit
from pettingzoo.classic import texas_holdem_v3
import time
import logging

print(np.random.choice(2, p=[0.5,0.5]))

# x = 'bla'
# y = 'bluuuuuuuuittttt'
# print(f'{x:<20}')
# print(f'{y:>20}')
#
# print(f'a{y:^20}b')


#
# logging.basicConfig(filename='sample.log',
#                     format='%(asctime)s | %(levelname)s: %(message)s %(thread)d',
#                     level=0)
# # logging.basicConfig(level=10)
# # logging.basicConfig(level=20)
# # logging.basicConfig(level=30)
# # logging.basicConfig(level=40)
# # logging.basicConfig(level=50)
#
# logging.debug('Here you have some information for debugging.')
# logging.info('Everything is normal. Relax!')
# logging.warning('Something unexpected but not important happend.')
# logging.error('Something unexpected and important happened.')
# logging.critical('OMG!!! A critical error happend and the code cannot run!')

# env = texas_holdem_v3.env()
#
# env.reset()
# for agent in env.agent_iter():
#     observation, reward, done, info = env.last()
#     env.render()
#     print('---')
#     time.sleep(1)
#     if done:
#         break
#     action = random.choice([i for i, e in enumerate(observation['action_mask']) if e != 0])
#     # action = random.choice(list(range(5)))
#     # action = np.random.random(4)
#     env.step(action)


import matplotlib.pyplot as plt
import numpy as np


np.random.seed(19680801)
data = np.random.random((50, 50, 50))
fig, _ = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax = fig.get_axes()
# fig, ax = plt.subplots()

for i in range(len(data)):
    ax[0].cla()
    ax[0].imshow(data[i])
    ax[0].set_title("frame {}".format(i))
    # Note that using time.sleep does *not* work here!
    plt.pause(0.1)