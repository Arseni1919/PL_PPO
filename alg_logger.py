from CONSTANTS import *

# Neptune.ai Logger
PARAMS = {
    'GAMMA': GAMMA,
    # 'LR': LR,
    'CLIP_GRAD': CLIP_GRAD,
    'ENV': ENV,
    'MAX_EPOCHS': BIG_EPOCHS,
}

if NEPTUNE:
    run = neptune.init(project='1919ars/PL-implementations',
                       tags=['PPO', ENV, f'{BIG_EPOCHS} epochs'],
                       name=f'PPO_{time.asctime()}',
                       source_files=['CONSTANTS.py'])
else:
    run = {}


run['parameters'] = PARAMS

# run['a'].log(y,x)
