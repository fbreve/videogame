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
        self.USE_PROCESS = True
        self.DATA_AUG = False
        self.PLATFORM_START = 1
        self.PLATFORM_END = 22
        self.TF_DETERMINISTIC_OPS = '1' # 0 - DenseNet, 1 - EfficientNet, Swin
        self.SAVE_WEIGHTS_PLATFORM = 23
        self.INTERPOLATION = 'nearest'
        self.MULTI_GPU = False
        
        # With MULTI_GPU = RTX 4060Ti 16GB + RTX2060 SUPER 8GB:
        # Model           MULTI_GPU     RTX 4060Ti alone
        # SwinT           ~500ms/step   ~2s/step
        # DenseNet201     ~500ms/step   ~140ms/step
        # EfficientNetV2S ~560ms/step   ~260ms/step

        self.platform_info = {
            28:     "Atari 2600",       #1
            22:     "NES",              #2
            26:     "Master System",    #3
            40:     "PC Engine",        #4
            16:     "Mega Drive",       #5
            15:     "Super Nintendo",   #6
            23:     "Sega Saturn",      #7
            6:      "PlayStation",      #8
            9:      "Nintendo 64",      #9
            8:      "Dreamcast",        #10
            7:      "PlayStation 2",    #11
            14:     "GameCube",         #12
            13:     "Xbox",             #13
            69:     "Xbox 360",         #14
            81:     "PlayStation 3",    #15
            82:     "Wii",              #16
            132:    "Wii U",            #17
            141:    "PlayStation 4",    #18
            142:    "Xbox One",         #19
            203:    "Nintendo Switch",  #20
            289:    "Xbox Series",      #21
            288:    "PlayStation 5",    #22
            143:    "Arcade"            #23
        }
        
        import socket
        hostname = socket.gethostname()
        
        if hostname=='DONALD':
            self.PLATFORM_START = 1
            self.PLATFORM_END = 22
            self.MODEL_TYPE = 'EfficientNetV2S'
            self.WEIGHTS = 'imagenet' # 'imagenet', None, 'arcade'
            self.TF_DETERMINISTIC_OPS = '1' # 0 - DenseNet201, 1 - Others
            self.USE_PROCESS = True # True will run TensorFlow in separate processes.
            self.DATA_AUG = False # True, False
            #self.INTERPOLATION = 'lanczos'
            self.MULTI_GPU = False
            #self.PCA_COMPONENTS = 0.99
            
        elif hostname=='PRECISION':
            self.PLATFORM_START = 14
            self.PLATFORM_END = 22
            self.MODEL_TYPE = 'EfficientNetV2S'
            self.WEIGHTS = 'imagenet' # 'imagenet', None, 'arcade'
            self.TF_DETERMINISTIC_OPS = '1' # 0 - DenseNet, 1 - EfficientNet
            self.USE_PROCESS = True
            self.DATA_AUG = False # True, False
            self.INTERPOLATION = 'lanczos'
                   
        else:
            print("ERROR: There is no configuration defined for this host.")        
            import sys
            sys.exit()