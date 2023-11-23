# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:07:41 2023

@author: fbrev
"""

import numpy as np
from videogame_config import vg_config
conf = vg_config()

epochs = np.zeros([22,5])
for index, platform in enumerate(conf.platform_info):
    # exception for Arcade
    if platform == 143:
        pass
    else:
        for i in range(1,6):
            history = np.load(f'results/history/vg-history-platform{platform}-{conf.MODEL_TYPE}-{i}.npy', allow_pickle=True).item()
            epochs[index,i-1] = len(history['loss'])
m_epochs = np.mean(epochs,axis=1)
np.savetxt('m_epochs.csv', m_epochs, fmt='%.2f', delimiter=', ', newline='\n')        