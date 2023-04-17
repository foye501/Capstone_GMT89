from utiles import extract_aesthetic_features_utiles, cnn_utiles, OBJ_detection_YOLO_utiles
from utiles.extract_aesthetic_features_utiles import get_aesthetics_one_room
from utiles.cnn_utiles import train_loader, SimpleCNN, image_transform
from utiles.OBJ_detection_YOLO_utiles import get_classes,parse_YOLO_result,get_listing_level_attr

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from tqdm import tqdm
import pickle
import torch.nn as nn
import warnings
from ultralytics import YOLO


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import overpy
import sys

sys.path = sys.path+['./utiles/']
from Airbnb_Capstone_Model import My_Airbnb_Capstone_Model, pre_processing_pipeline
my_capstone_model = pickle.load(open('./trained_models/overall_model.pkl','rb'))
### extract features
img_path_list = ['./new_data/test.jpg', './new_data/test2.jpg']

address = '1st Helms Ave, Culver City, California, United States'
additional_dict = {'bedrooms':2, 
                   'beds':4,
                   'minimum_nights':2, 
                   'maximum_nights':270}
property_dict = {'property_type':'Entire home'} 

# Entire home, Entire rental unit, Private room in home
# Private room in home, Entire guest suite

room_dict = {'room_type':'Entire home/apt'}
#'Entire home/apt', 'Hotel room', 'Private room', 'Shared room'
bathroom_dict = {'bathroom_type':'private', 'bathroom_count':2}
# 'private', 'shared'
amenities_dict = {'wifi':1,'apple tv':1,'essentials':1,'body soap':1}

descriptions = 'Semi-Private, vaccinated only, you will be staying on a queen size bed in my common living room with room dividers to give you some privacy, I have the private bedroom. (UPDATE= from Oct 19 to Nov 19 I will be out of town and the place will be all yours.) Only you and myself at any time.  30 Day minimum, so best for long term guest.  Plenty of street parking.   Please have full time employment or school enrollment.<br /><br /><b>The space</b><br />One of the best locations in Hollywood, central to everything, including trains and buses yet still a quiet neighborhood. Comfy real queen size bed with IKEA room dividers separating it from the living room/kitchen area. 3 laundry rooms in the building.<br /><br /><b>Guest access</b><br />You have access to everything in the apartment except my bedroom.  Unfortunately the buildings pool and hot tub are under reconstruction.<br /><br /><b>Other things to note</b><br />This is more of a roommate situation as I live in the apartment. But I do wo'
### img features
img_features = my_capstone_model.image_processor.process_new_data(img_path_list)
img_features = pd.DataFrame(img_features,index=[0])
img_features.columns = ['Image_'+i for i in img_features.columns]

### loc features
loc_features = my_capstone_model.location_processor.process_new_data(overpy.Overpass(), address)

### amenities features
amenities_features = my_capstone_model.amenities_processor.process_new_data(additional_dict,
                                                           property_dict, 
                                                           room_dict, 
                                                           bathroom_dict, 
                                                           amenities_dict)
amenities_features.columns = ['Amenities_'+i for i in amenities_features.columns]

### descriptions features
descriptions_features = my_capstone_model.nlp_processor.process_new_data(descriptions)
descriptions_features.columns = ['NLP_'+i for i in descriptions_features.columns]

prediciton_features = pd.concat([
    amenities_features,
    loc_features,
    descriptions_features,
    img_features
],axis=1)

prediciton_features = prediciton_features[my_capstone_model.x_names]
my_capstone_model.predict(prediciton_features)


