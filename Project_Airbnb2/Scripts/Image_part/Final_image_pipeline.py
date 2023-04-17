from photo_scrapper import scrape_photos
from extract_aesthetic_features_utiles import get_aesthetics_one_room
from cnn_utiles import train_loader, SimpleCNN, image_transform
from OBJ_detection_YOLO_utiles import get_classes,parse_YOLO_result,get_listing_level_attr

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

warnings.filterwarnings('ignore')

listings = pd.read_csv('../../Data/LA_Airbnb/listings_detailed.csv')
listings['clean_price'] = [float(i.replace('$','').replace(',','')) for i in listings['price']]
room_codes = [i.split('/')[-1] for i in listings['listing_url'].values]

import shutil

#### compile CNN black box features output
cnn_model = pickle.load(open('trained_models/best_CNN.pkl','rb'))
def get_black_box_features(cnn_model,room_code):
    try:
        im_paths = os.listdir(f'LA_photos/{room_code}')
    except:
        return np.array([np.nan])
    
    cnn_model.eval()
    res = []
    for im_path in im_paths:
        try:
            im_tf = image_transform(f'LA_photos/{room_code}/{im_path}')
            
            x = cnn_model.conv1(im_tf)
            x = cnn_model.relu1(x)
            x = cnn_model.pool1(x)
            
            x = cnn_model.conv2(x)
            x = cnn_model.relu2(x)
            x = cnn_model.pool2(x)
            
            x = cnn_model.conv3(x)
            x = cnn_model.relu3(x)
            x = cnn_model.pool3(x)
            
            x = x.view(-1, 64*28*28)
            
            x = cnn_model.fc1(x)
            x = cnn_model.fc2(x)
            res.append(x.detach().numpy())
        except:
            continue
    
    res = np.array(res).mean(axis=0)
    return res



#### compile YOLo obj detection features output
YOLO_model = YOLO('trained_models/YOLOv8_best.pt')
YOLO_classes = get_classes()
def get_YOLO_results(YOLO_model,room_code,YOLO_classes):
    try:
        photos = os.listdir(f'LA_photos/{room_code}')
        photo_paths = [f'LA_photos/{room_code}/{photo}' for photo in photos]
    except:
        return np.nan
    
    if len(photos)==0:
        return {}
    
    photo_paths = [f'LA_photos/{room_code}/{photo}' for photo in photos]
    attrs_all = {}
    for index,path in enumerate(photo_paths):
        try:
            res = YOLO_model(photo_paths, verbose=False)
            attrs = parse_YOLO_result(res)
            attrs_all[index] = attrs[0]
        except Exception as e:
            continue
    
    if len(attrs_all)==0:
        return {}

    listing_level_attrs = get_listing_level_attr(attrs_all,YOLO_classes)
    return listing_level_attrs


#### feature extraction on the fly (streaming)
saving_batch_size = 100
feature_list = []
for room_count,room_code in tqdm(enumerate(room_codes[:20]), total=len(room_codes[:20])):

    try: ### make sure things never stop
        ### part 1: donwload
        def dd(room_code):
            donwloaded = 0
            try:
                donwloaded = scrape_photos(room_code) 
            except Exception as e:
                # print(room_code, e)
                pass
            return donwloaded
            
        donwloaded = dd(room_code)
        if donwloaded<5: ### try again if download fail
            donwloaded = dd(room_code)

        if donwloaded==0:
            continue
        
        ### part 2: aesthetic features
        aes_features = get_aesthetics_one_room(room_code)
        hue_hist_featurs = {f'hist{i+1}':aes_features['hist'][i] for i in range(len(aes_features['hist']))}
        aes_features_others = dict([i for i in aes_features.items() if not i[0]=='hist'])

        ### part 3: black box features
        blackbox_features = get_black_box_features(cnn_model,room_code).flatten()
        blackbox_features = {f'bbf{i+1}':blackbox_features[i] for i in range(len(blackbox_features))}

        ### part 4: object detection features (including the rule of third)
        YOLO_features = get_YOLO_results(YOLO_model,room_code,YOLO_classes)

        ### part 5: identification features
        id_f = {'room_id':room_code,'images_used':donwloaded}

        ### final: combine features
        final_img_features = {**id_f,
                            **blackbox_features, 
                            **aes_features_others, 
                            **hue_hist_featurs, 
                            **YOLO_features}
        
        ### remove that file/folder
        if room_count>3000: ### only store 3000 of them
            shutil.rmtree(f'./LA_photos/{room_code}')

        ### counter & saver
        room_count+=1
        feature_list.append(final_img_features)

        ### saving by batch
        if (room_count%saving_batch_size==0) and (room_count>5) :
            this_batch_num = int(room_count/saving_batch_size)
            pickle.dump(feature_list, open(f'LA_extracted_features1/{this_batch_num}.pkl','wb'))
            feature_list=[]
            
    except Exception as e:
        print(e)
        continue



