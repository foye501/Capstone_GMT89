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

import sys

class Image_processor():
    def __init__(self):
        sys.path = sys.path+['./utiles/']
        self.cnn_model = pickle.load(open('trained_models/best_CNN.pkl','rb'))
        self.YOLO_model =YOLO('trained_models/YOLOv8_best.pt')
        pass
    
    def process_new_data(self,img_path_list):
        warnings.filterwarnings('ignore')
        

        def get_black_box_features(cnn_model,img_path_list):

            cnn_model.eval()
            res = []
            for img_path in img_path_list:
                try:
                    im_tf = image_transform(img_path)
                    
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
                except Exception as e:
                    print(e)
                    raise

            res = np.array(res).mean(axis=0)
            return res


        #### compile YOLo obj detection features output
        YOLO_classes = get_classes()

        def get_YOLO_results(YOLO_model,img_path_list,YOLO_classes):

            attrs_all = {}
            for index,img_path in enumerate(img_path_list):
                try:
                    res = YOLO_model(img_path, verbose=False)
                    attrs = parse_YOLO_result(res)
                    attrs_all[index] = attrs[0]
                except Exception as e:
                    raise

            if len(attrs_all)==0:
                return {}

            listing_level_attrs = get_listing_level_attr(attrs_all,YOLO_classes)
            return listing_level_attrs

        #### feature extraction on the fly (streaming)
        ### part 2: aesthetic features
        aes_features = get_aesthetics_one_room(img_path_list)
        hue_hist_featurs = {f'hist{i+1}':aes_features['hist'][i] for i in range(len(aes_features['hist']))}
        aes_features_others = dict([i for i in aes_features.items() if not i[0]=='hist'])

        ### part 3: black box features
        blackbox_features = get_black_box_features(self.cnn_model,img_path_list).flatten()
        blackbox_features = {f'bbf{i+1}':blackbox_features[i] for i in range(len(blackbox_features))}

        ### part 4: object detection features (including the rule of third)
        YOLO_features = get_YOLO_results(self.YOLO_model,img_path_list,YOLO_classes)

        ### part 5: identification features
        id_f = {'images_used':len(img_path_list)}

        ### final: combine features
        final_img_features = {**id_f,
                            **blackbox_features, 
                            **aes_features_others, 
                            **hue_hist_featurs, 
                            **YOLO_features}
        return final_img_features


