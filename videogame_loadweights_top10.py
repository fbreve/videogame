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

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#from sklearn.metrics import confusion_matrix
import os
import random
from datetime import datetime, timedelta
from time import perf_counter
import PIL
import sys

from videogame_config import vg_config
conf = vg_config()

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

TOP_K = 10

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
    model = model_function(weights='weights/weights-' + model_type + '.h5', include_top=False, pooling=conf.POOLING, input_shape=image_size + (conf.IMAGE_CHANNELS,))
    preprocessing_function = getattr(model_module,'preprocess_input')
    
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.models import Model
    
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

def train_test_model(train_df, test_df, model, preprocessing_function, image_size, split, batch_size, platform):
   
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
                
    train_df, validate_df = train_test_split(train_df, test_size=0.20, shuffle=True,
                                             stratify=train_df.category, random_state=SEED)
    
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    #train_df['category'].value_counts().plot.bar()
    
    #validate_df['category'].value_counts().plot.bar()
    
    #total_train = train_df.shape[0]
    #total_validate = validate_df.shape[0]
    #total_test = test_df.shape[0]
       
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
        verbose='auto',
    )

    # save history
    history_filename = "./results/history/vg-history-loadweights-top10-platform" + str(platform) + "-" + model.name + "-" + str(split) + ".npy"
    os.makedirs(os.path.dirname(history_filename), exist_ok=True)
    np.save(history_filename,history.history)

    test_datagen = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function=preprocessing_function
    )
    test_generator = test_datagen.flow_from_dataframe(
         test_df, 
         conf.DATASET_PATH, 
         x_col='filename',
         y_col='category',
         class_mode='categorical',
         target_size=image_size,
         batch_size=batch_size,
         shuffle=False,
         seed=SEED,
    )
    
    predictions = model.predict(test_generator, verbose=1)
    
    true_labels = test_generator.labels
    
    # Calculate accuracy for top-k predictions
    accuracies = []
    
    for i in range(len(predictions)):
        # Sort the predicted probabilities and get indices of top-k predictions
        top_indices = np.argsort(predictions[i])[::-1][:TOP_K]
        
        accuracy = np.zeros(10)
        # Calculate accuracy for each position
        for j in range(len(top_indices)):
            if top_indices[j] == true_labels[i]:
                accuracy[j] = 1
        
        # add item accuracy to a list                
        accuracies.append(accuracy)
        
    # get the mean accuracy in each position and accumulates it
    acc = np.cumsum(np.mean(accuracies,0))
                  
    trained_epochs = len(history.history['loss'])
    
    return acc, trained_epochs
 
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
    log_filename = "videogame-loadweight-top10-" + conf.MODEL_TYPE + "-" + hostname + ".log"
    csv_acc_filename = "videogame-lw-acc-top10-" + conf.MODEL_TYPE + "-" + hostname + ".csv"
    csv_epc_filename = "videogame-lw-epc-top10-" + conf.MODEL_TYPE + "-" + hostname + ".csv"
    
    # write log header
    print_and_log("Machine: %s" % hostname)
    now = datetime.now()
    print_and_log(now.strftime("Date: %d/%m/%Y Time: %H:%M:%S"))
    print_and_log("Model: %s" % conf.MODEL_TYPE)
    print_and_log("Pooling Application Layer: %s" % conf.POOLING)
    print_and_log("Data Augmentation: %s" % conf.DATA_AUG)  
    print_and_log("Data Augmentation Multiplier: %s" % conf.DATA_AUG_MULT)
    print_and_log("Fine Tuning: %s" % conf.FINE_TUN)
    print_and_log("Multi Optimizer: %s" % conf.MULTI_OPTIMIZER)
    print_and_log("Optimizer: %s" % conf.OPTIMIZER)    
    print_and_log("Batch Size: %s" % conf.BATCH_SIZE)
    print_and_log("Restore Best Weights: %s\n" % conf.REST_BEST_W)

    platform_list = list(conf.platform_info.keys())[conf.PLATFORM_START-1:conf.PLATFORM_END]
    
    for platform in platform_list:
           
        df, total_games, total_screenshots, selected_games, selected_screenshots = load_data(platform)    

        print_and_log("Platform: %s - %s" % (platform, conf.platform_info.get(platform)))
        print_and_log("Selected Games: %i / %i" % (selected_games, total_games))
        print_and_log("Selected Screenshots: %i / %i\n" % (selected_screenshots, total_screenshots))

        # Even after all images are "validated" (opened by PIL without errors), 
        # sometimes one would still thrown an error "image file is truncated (0 bytes not processed)"
        # when TensorFlow is training. This first appeared in the Xbox 360 screenshots.
        # This setting apparently fix this issue.
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

        # creating folds for cross-validation
        from sklearn.model_selection import StratifiedKFold
        kfold_n_splits = 5
        kf = StratifiedKFold(n_splits=kfold_n_splits, shuffle=True, random_state=SEED)
        kf.split(df,df.category)
                           
        t1_start = perf_counter()
                   
        # vectors to hold each fold accuracy and trained epochs
        cvscores = []
        cvepochs = []

        batch_size = conf.BATCH_SIZE
            
        # enumerate allow the usage of the index for prints
        for index, [train, test] in enumerate(kf.split(df,df.category)):    
            train_df = df.loc[train]
            test_df = df.loc[test]            
            
            model, preprocessing_function, image_size = create_model(conf.MODEL_TYPE, selected_games)
            acc, trained_epochs = train_test_model(train_df,test_df,model,preprocessing_function,image_size, index+1, batch_size, platform)            
                      
            cvscores.append(acc)
            cvepochs.append(trained_epochs)
            
            print_and_log(f"Fold: {(index+1)} of {kfold_n_splits}")                      
            str_acc =  "Accuracies:"
            for acci in acc:
                str_acc = str_acc + f" {(acci*100):.2f}%"
            print_and_log(str_acc)
            str_mean = "Mean:      "            
            for meani in np.mean(cvscores,0):
                str_mean = str_mean + f" {(meani*100):.2f}%"
            print_and_log(str_mean)
            print_and_log(f"Epochs: {trained_epochs} "
                          f"Mean: {np.mean(cvepochs):.2f} "
                          f"(+/- {np.std(cvepochs):.2f})"
                          )
                            
        # record results to csv files
        with open(csv_acc_filename,"a+") as f_csv:
            for index, acci in enumerate(np.mean(cvscores,0)):
                if index<TOP_K-1: f_csv.write("%.4f, " % acci)
                else: f_csv.write("%.4f\n" % acci)
        with open(csv_epc_filename,"a+") as f_csv:
            f_csv.write("%.4f\n" % np.mean(cvepochs))
                
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
        
        print_and_log("Elapsed time: {}.\n".format(formatted_time))