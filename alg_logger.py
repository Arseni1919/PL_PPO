from CONSTANTS import *

# Neptune.ai Logger
PARAMS = {
    'GAMMA': GAMMA,
    'LR': LR,
    'CLIP_GRAD': CLIP_GRAD,
    'ENV': ENV,
    'MAX_EPOCHS': MAX_EPOCHS,
}

run = neptune.init(project='1919ars/PL-implementations',
                   tags=['PPO', ENV, f'{MAX_EPOCHS} epochs'],
                   name=f'PPO_{time.asctime()}',
                   source_files=['CONSTANTS.py'])

run['parameters'] = PARAMS
