from CONSTANTS import *

# Neptune.ai Logger
PARAMS = {
    'GAMMA': GAMMA,
    'LR': LR,
    'CLIP_GRAD': CLIP_GRAD,
    'ENV':ENV,
}

run = neptune.init(project='1919ars/Neptune-Tutorials', tags=['PPO', 'try'],
                   name=f'PPO_{time.asctime()}', source_files=['CONSTANTS.py'])
run['parameters'] = PARAMS
