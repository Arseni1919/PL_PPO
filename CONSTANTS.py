import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torch.autograd import Variable
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import gym

ENV = "CartPole-v0"  # gym environment tag
# ENV = 'LunarLander-v2'
# ENV='MountainCar-v0'
NUMBER_OF_GAMES = 10
SAVE_RESULTS = True
# ------------------------------------------- #
# ------------------FOR ALG:----------------- #
# ------------------------------------------- #

MAX_EPOCHS = 1000  # maximum epoch to execute
# BATCH_SIZE = 128  # size of the batches
MAX_LENGTH_OF_A_GAME = 10000
LR = 3e-5  # learning rate
GAMMA = 0.99  # discount factor
ENTROPY_BETA = 0.001
REWARD_STEPS = 4
CLIP_GRAD = 0.1
VAL_CHECKPOINT_INTERVAL = 10
HIDDEN_SIZE = 256

