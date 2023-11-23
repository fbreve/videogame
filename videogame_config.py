# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:59:14 2023

@author: fbrev
"""

class vg_config():
    
    def __init__(self):
           
        self.FAST_RUN = False
        self.IMAGE_CHANNELS=3
        
        self.POOLING = 'avg' # None, 'avg', 'max'
        self.DATA_AUG = False # True, False
        self.DATA_AUG_MULT = 1 # >=1
        self.BATCH_SIZE = 16 # 
        self.FINE_TUN = True
        self.MULTI_OPTIMIZER = False
        self.OPTIMIZER = 'Adam' # 'RMSprop', 'Adam', etc. Note: IJCNN results with rmsprop
        self.REST_BEST_W = False                
        self.DATASET_PATH = "./data/screenshots/"
        #self.MODEL_TYPE_LIST = ['MobileNet', 'DenseNet169', 'EfficientNetB0', 'EfficientNetB2']        

        self.platform_info = {
            28:     "Atari 2600",
            22:     "NES",
            26:     "Master System",
            40:     "PC Engine",
            16:     "Mega Drive",
            15:     "Super Nintendo",
            23:     "Sega Saturn",
            6:      "PlayStation",
            9:      "Nintendo 64",
            8:      "Dreamcast",
            7:      "PlayStation 2",
            14:     "GameCube",
            13:     "Xbox",
            69:     "Xbox 360",
            81:     "PlayStation 3",
            82:     "Wii",
            132:    "Wii U",
            141:    "PlayStation 4",
            142:    "Xbox One",
            203:    "Nintendo Switch",
            289:    "Xbox Series",
            288:    "PlayStation 5",
            143:    "Arcade"
        }
        
        import socket
        hostname = socket.gethostname()
        
        if hostname=='DONALD':
            self.MODEL_TYPE_START = 1
            self.MODEL_TYPE_END = 1
            #self.PLATFORM = 143
            self.PLATFORM_START = 1
            self.PLATFORM_END = 22
            self.MODEL_TYPE_LIST = ['EfficientNetV2S']
            self.MODEL_TYPE = 'EfficientNetB3'
            #self.MODEL = 'EfficientNetV2S'            
            #self.PCA_COMPONENTS = 0.99
            
        elif hostname=='PRECISION':
            self.MODEL_TYPE_START = 1
            self.MODEL_TYPE_END = 1
            self.PLATFORM = 143
            self.PLATFORM_START = 1
            self.PLATFORM_END = 22      
            self.MODEL_TYPE_LIST = ['EfficientNetB3']
            self.MODEL_TYPE = 'EfficientNetB3'
            
        elif hostname=='SNOOPY': 
            self.MODEL_TYPE_START = 1
            self.MODEL_TYPE_END = 1
            #self.PLATFORM = 143
            self.PLATFORM_START = 1
            self.PLATFORM_END = 22
            self.MODEL_TYPE_LIST = ['DenseNet169']
            self.MODEL_TYPE = 'DenseNet169'
            #self.PCA_COMPONENTS = 0.99
        
        else:
            print("ERROR: There is no configuration defined for this host.")        
            import sys
            sys.exit()