# -*- coding: utf-8 -*-
"""
Videogame Identification by Screenshots using Convolutional Neural Networks
Created on Oct 2023
@author: Fabricio Breve

Warning: it is crashing with DirectML (memory issues?) on the NES dataset
when no pooling is used. Workaround: use another Anaconda environment without
DML (TensorFlow running on CPU).

Required packages:
    tensorflow
    pandas
    scikit-learn
    pillow
"""

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
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import multiprocessing
from sklearnex import patch_sklearn
patch_sklearn()

from videogame_config import vg_config
conf = vg_config()

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
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

def create_model(model_type):
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
        'ConvNeXtBase': ['convnext', 224, 224],
        'ConvNeXtXLarge': ['convnext', 224, 224]
    }
    
    model_module = getattr(tf.keras.applications,model_dic[model_type][0])
    model_function = getattr(model_module,model_type)
    image_size = tuple(model_dic[model_type][1:])
    #model = model_function(weights='imagenet', include_top=False, pooling=conf.POOLING, input_shape=image_size + (conf.IMAGE_CHANNELS,))
    model = model_function(weights='weights/weights-' + model_type + '.h5', include_top=False, pooling=conf.POOLING, input_shape=image_size + (conf.IMAGE_CHANNELS,))    
    preprocessing_function = getattr(model_module,'preprocess_input')
    
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model

    output = Flatten()(model.layers[-1].output)   
    model = Model(inputs=model.inputs, outputs=output)

    return model, preprocessing_function, image_size

def extract_features(df, model, preprocessing_function, image_size, batch_size):
           
    datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocessing_function
    )
      
    generator = datagen.flow_from_dataframe(
        df, 
        conf.DATASET_PATH, 
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    features = model.predict(generator)
    
    return features
 
def print_and_log(arg):
    print(arg)
    with open(log_filename,"a+") as f_log:
        f_log.write(arg + "\n")        
 
def train_classifier(train, test, clf, features, labels):    
    clf.fit(features[train], labels[train])
    y_pred = clf.predict(features[test])
    accuracy = accuracy_score(labels[test], y_pred)
    return accuracy
    
# Main
if __name__ == "__main__":
    
    # get hostname for log-files
    import socket
    hostname = socket.gethostname()
    
    # create log filename
    log_filename = "videogame-" + conf.MODEL_TYPE + "-" + str(conf.POOLING) + "-" + str(conf.PCA_COMPONENTS) + "-" + hostname + ".log"
    csv_filename = "videogame-" + conf.MODEL_TYPE + "-" + str(conf.POOLING) + "-" + str(conf.PCA_COMPONENTS) + "-" + hostname + ".csv"
    
    # write log header   
    print_and_log("Machine: %s" % hostname)
    now = datetime.now()
    print_and_log(now.strftime("Date: %d/%m/%Y Time: %H:%M:%S"))
    print_and_log("Model: %s" % conf.MODEL_TYPE)
    print_and_log("Pooling Application Layer: %s" % conf.POOLING)
    print_and_log("Batch Size: %s" % conf.BATCH_SIZE)
    print_and_log("PCA Components: %s\n" % conf.PCA_COMPONENTS)

    platform_list = list(conf.platform_info.keys())[conf.PLATFORM_START-1:conf.PLATFORM_END]

    # Initialize and train multiple classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Support Vector Machines": SVC(),
        "Linear SVM": LinearSVC(dual="auto",max_iter=10000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": HistGradientBoostingClassifier(),
        "Random Forest": RandomForestClassifier(),
    }
    
    print("Creating model...")
    model, preprocessing_function, image_size = create_model(conf.MODEL_TYPE)
    
    batch_size = conf.BATCH_SIZE    
    
    for platform in platform_list:

        t1_start = perf_counter()      
    
        print("Loading data...")
    
        df, total_games, total_screenshots, selected_games, selected_screenshots = load_data(platform)
        
        #labels_filename = "data/features/vg-platform" + str(platform) + "-labels.csv"
        #np.savetxt(labels_filename, df.category, fmt="%s")        
        
        labels = df.category

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
        
        print("Extracting features...")
        features = extract_features(df, model, preprocessing_function, image_size, batch_size)
        
        print_and_log("Original features: %i " % len(features[0]))
               
        # remove features with zero variance
        print("Removing features with zero variance...")
        selector = VarianceThreshold()
        features = selector.fit_transform(features)
        
        print_and_log("Features after removing those with zero variance: %i " % len(features[0]))
        
        # Apply PCA for dimensionality reduction
        print("Applying Principal Component Analysis...")
        pca = PCA(n_components=conf.PCA_COMPONENTS)
        features = pca.fit_transform(features)
        
        print_and_log("Features after applying PCA: %i \n" % len(features[0]))
    
        #data_filename = "data/features/vg-platform" + str(platform) + "-" + model_type + "-data.csv"
        #np.savetxt(data_filename, features, fmt="%s", delimiter=";")        
                    
        pool = multiprocessing.Pool(processes=5)  # Adjust the number of processes as needed
    
        for i_classifier, [clf_name, clf] in enumerate(classifiers.items()):
            results = np.zeros(5)
            print(f"Training {clf_name} Classifier...")
    
            jobs = []
            for i_split, [train, test] in enumerate(kf.split(df, df.category)):
                job = pool.apply_async(train_classifier, (train, test, clf, features, labels))
                jobs.append(job)
    
            for i_split, job in enumerate(jobs):
                accuracy = job.get()
                results[i_split] = accuracy
    
            mean_res = np.mean(results)
            print_and_log(f'{clf_name} Classifier Accuracy: {mean_res:.4f}')
    
            # Record results to the CSV file
            with open(csv_filename, "a+") as f_csv:
                f_csv.write("%.4f" % mean_res)
                if i_classifier != 7:
                    f_csv.write(", ")
                else:
                    f_csv.write("\n")
    
        pool.close()
        pool.join()
    
        print_and_log("\n")
                           
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