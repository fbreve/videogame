# -*- coding: utf-8 -*-
"""
Videogame Identification by Screenshots using Convolutional Neural Networks
Created on Aug 2023
@author: Fabricio Breve

based on visually_impaired_aid_tl

Required packages:
    tensorflow
    pandas
    scikit-learn
    pillow
"""

from videogame_config import vg_config
conf = vg_config()

import os
# For some models, when jit_compile='auto' in model.compile, TF_DETERMINISTIC_OPS has to be disabled
# due to error: "GPU MaxPool gradient ops do not yet have a deterministic XLA implementation":
os.environ['TF_DETERMINISTIC_OPS'] = conf.TF_DETERMINISTIC_OPS
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
#from sklearn.metrics import confusion_matrix
import random
from datetime import datetime, timedelta
from time import perf_counter
import PIL
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model    

random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

def load_data(platform):
    gamelist = pd.read_csv("./data/gamelist-" + str(platform) + ".csv", index_col=0)
    screenshotlist = pd.read_csv("./data/screenshotlist-" + str(platform) + ".csv", index_col=0)
    screenshotcount = pd.read_csv("./data/screenshotcount-" + str(platform) + ".csv", index_col=0)
    total_games = len(gamelist)
    total_screenshots = len(screenshotlist)
    selected_games = 0
    selected_screenshots = 0
    
    filenames=[]
    game_ids=[]    
    
    corrupted_screenshot = False
    
    for ssc_index, ssc_row in screenshotcount.iterrows():
        # we won't use games with less than 5 screenshots
        if ssc_row.sc_count>=5:
            selected_games = selected_games + 1
            target_game_id = ssc_row.game_id
            filtered_screenshots = screenshotlist[screenshotlist['game_id'] == target_game_id]
            for fs_index, fs_row in filtered_screenshots.iterrows():
                selected_screenshots = selected_screenshots + 1
                filenames.append(fs_row.filename)    
                game_ids.append(str(fs_row.game_id))
                
                try:
                    img_p = conf.DATASET_PATH + fs_row.filename
                    PIL.Image.open(img_p)
                except PIL.UnidentifiedImageError:
                    print("Screenshot file missing or corrupted: " + fs_row.filename)
                    corrupted_screenshot = True
    
    if corrupted_screenshot == True:
        sys.exit("The screenshots above are missing or corrupted. Please fix them and run again.")   

    #record selected/total games/screenshots to .csv file
    csv_filename = ("./data/stats-" + str(platform) + ".csv")
    with open(csv_filename,"w") as f_csv:
        f_csv.write("%i, %i, %i, %i\n" % (selected_games, total_games, selected_screenshots, total_screenshots))
                   
    df = pd.DataFrame({
        'filename': filenames,
        'category': game_ids
    })

    return df, total_games, total_screenshots, selected_games, selected_screenshots

def create_model(model_type, class_count):
    model_dic = {
        'VGG16': ['vgg16', 224, 224],
        'VGG19': ['vgg19', 224, 224],
        'Xception': ['xception', 299, 299],
        'ResNet50': ['resnet', 224, 224],
        'ResNet101': ['resnet', 224, 224],
        'ResNet152': ['resnet', 224, 224],
        'ResNet50V2': ['resnet_v2', 224, 224],
        'ResNet101V2': ['resnet_v2', 224, 224],
        'ResNet152V2': ['resnet_v2', 224, 224],
        'InceptionV3': ['inception_v3', 299, 299],
        'InceptionResNetV2': ['inception_resnet_v2', 299, 299],
        'MobileNet': ['mobilenet', 224, 224],
        'DenseNet121': ['densenet', 224, 224],
        'DenseNet169': ['densenet', 224, 224],
        'DenseNet201': ['densenet', 224, 224],
        'NASNetLarge': ['nasnet', 331, 331],
        'NASNetMobile': ['nasnet', 224, 224],
        'MobileNetV2': ['mobilenet_v2', 224, 224],
        'EfficientNetB0': ['efficientnet', 224, 224],
        'EfficientNetB1': ['efficientnet', 240, 240],
        'EfficientNetB2': ['efficientnet', 260, 260],
        'EfficientNetB3': ['efficientnet', 300, 300],
        'EfficientNetB4': ['efficientnet', 380, 380],
        'EfficientNetB5': ['efficientnet', 456, 456],
        'EfficientNetB6': ['efficientnet', 528, 528],
        'EfficientNetB7': ['efficientnet', 600, 600],
        'EfficientNetV2B2': ['efficientnet_v2', 260, 260],
        'EfficientNetV2B3': ['efficientnet_v2', 300, 300],
        'EfficientNetV2S': ['efficientnet_v2', 384, 384],
        'EfficientNetV2M': ['efficientnet_v2', 480, 480],
        'ConvNeXtTiny': ['convnext', 224, 224],
        'ConvNeXtSmall': ['convnext', 224, 224],
        'ConvNeXtBase': ['convnext', 224, 224]
    }
    
    model_module = getattr(tf.keras.applications,model_dic[model_type][0])
    model_function = getattr(model_module,model_type)
    image_size = tuple(model_dic[model_type][1:])
    model = model_function(weights='imagenet', include_top=False, pooling=conf.POOLING, input_shape=image_size + (conf.IMAGE_CHANNELS,))
    preprocessing_function = getattr(model_module,'preprocess_input')
    
    # mark loaded layers as not trainable
    if conf.FINE_TUN == False:
        for layer in model.layers:
            layer.trainable = False
	# add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    
    top_dropout_rate = 0.2
    class1 = Dropout(top_dropout_rate, name="top_dropout")(flat1)  

    output = Dense(class_count, activation='softmax')(class1)
    
    # define new model
    model = Model(inputs=model.inputs, outputs=output, name=model_type)    
	
    compile_model(model)
    
    model.summary()
    
    return model, preprocessing_function, image_size

def compile_model(model):
    
    # compile model

    opt_func = getattr(tf.keras.optimizers, conf.OPTIMIZER)
       
    optimizer = opt_func(learning_rate=1e-3)   
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

def train_model(df, model, preprocessing_function, image_size, batch_size, platform):
   
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    earlystop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=conf.REST_BEST_W
    )
   
    if conf.MULTI_OPTIMIZER==True:            
        callbacks = [earlystop]
    else:
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)        
        callbacks = [earlystop, learning_rate_reduction]
                
    train_df, validate_df = train_test_split(df, test_size=0.20, shuffle=True,
                                             stratify=df.category, random_state=SEED)
    
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
          
    if conf.DATA_AUG==True:
        train_datagen = ImageDataGenerator(
            rotation_range=15,    
            #rescale=1./255,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            preprocessing_function=preprocessing_function
        )
    else:
        train_datagen = ImageDataGenerator(
            #rescale=1./255,
            preprocessing_function=preprocessing_function
        )
        
    validation_datagen = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function=preprocessing_function
    )        
       
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        conf.DATASET_PATH, 
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        seed=SEED,
    )
    
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        conf.DATASET_PATH, 
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,    
        seed=SEED,
    )

    #import psutil
    
    epochs=3 if conf.FAST_RUN else 50

    history = model.fit(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=2,
    )

    # save history
    history_filename = "./results/history/vg-history-platform" + str(platform) + "-" + model.name + ".npy"
    os.makedirs(os.path.dirname(history_filename), exist_ok=True)
    np.save(history_filename,history.history)
    
    return model
 
def print_and_log(arg):
     print(arg)
     with open(log_filename,"a+") as f_log:
         f_log.write(arg + "\n")        
 
# Main
if __name__ == "__main__":
    
    # get hostname for log-files
    import socket
    hostname = socket.gethostname()
    
    # create filenames
    log_filename = "videogame-" + conf.MODEL_TYPE + "-" + str(conf.WEIGHTS) + "-saveweights-" + hostname + ".log"
    
    # write log header
    print_and_log("Machine: %s" % hostname)
    now = datetime.now()
    print_and_log(now.strftime("Date: %d/%m/%Y Time: %H:%M:%S"))
    print_and_log("Model: %s" % conf.MODEL_TYPE)
    print_and_log("Weights: %s" % conf.WEIGHTS)
    print_and_log("Pooling Application Layer: %s" % conf.POOLING)
    print_and_log("Data Augmentation: %s" % conf.DATA_AUG)  
    print_and_log("Data Augmentation Multiplier: %s" % conf.DATA_AUG_MULT)
    print_and_log("Fine Tuning: %s" % conf.FINE_TUN)
    print_and_log("Multi Optimizer: %s" % conf.MULTI_OPTIMIZER)
    print_and_log("Optimizer: %s" % conf.OPTIMIZER)    
    print_and_log("Batch Size: %s" % conf.BATCH_SIZE)
    print_and_log("Deterministic OPs: %s" % conf.TF_DETERMINISTIC_OPS)        
    print_and_log("Restore Best Weights: %s\n" % conf.REST_BEST_W)        
         
    df, total_games, total_screenshots, selected_games, selected_screenshots = load_data(conf.SAVE_WEIGHTS_PLATFORM)    
    
    print_and_log("Platform: %s - %s" % (conf.SAVE_WEIGHTS_PLATFORM, conf.platform_info.get(conf.SAVE_WEIGHTS_PLATFORM)))
    print_and_log("Selected Games: %i / %i" % (selected_games, total_games))
    print_and_log("Selected Screenshots: %i / %i\n" % (selected_screenshots, total_screenshots))

    # Even after all images are "validated" (opened by PIL without errors), 
    # sometimes one would still thrown an error "image file is truncated (0 bytes not processed)"
    # when TensorFlow is training. This first appeared in the Xbox 360 screenshots.
    # This setting apparently fix this issue.
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

    t1_start = perf_counter()
                
    batch_size = conf.BATCH_SIZE
    model, preprocessing_function, image_size = create_model(conf.MODEL_TYPE, selected_games)
    model = train_model(df,model,preprocessing_function,image_size, batch_size, conf.SAVE_WEIGHTS_PLATFORM)            

    # remove the classification layer before saving weights
    output = model.layers[-2].output
    model = Model(inputs=model.inputs, outputs=output, name=conf.MODEL_TYPE)
    compile_model(model)
    
    # save weights
    model.save_weights("weights/" + conf.MODEL_TYPE + ".weights.h5")
                    
    t1_stop = perf_counter()
    t1_elapsed = t1_stop-t1_start
    elapsed_time = timedelta(seconds=t1_elapsed)

    # Initialize an empty list to store the time components
    time_components = []
    
    # Check if each time component is greater than zero and add it to the list
    if elapsed_time.days > 0:
        time_components.append(f"{elapsed_time.days} days")
    if elapsed_time.seconds // 3600 > 0:
        time_components.append(f"{elapsed_time.seconds // 3600} hours")
    if (elapsed_time.seconds % 3600) // 60 > 0:
        time_components.append(f"{(elapsed_time.seconds % 3600) // 60} minutes")
    if elapsed_time.seconds % 60 > 0:
        time_components.append(f"{elapsed_time.seconds % 60} seconds")

    # Format the elapsed time
    formatted_time = ', '.join(time_components)
    
    print_and_log("Elapsed time: {}.\n\n".format(formatted_time))