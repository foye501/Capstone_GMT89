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

import pickle

from warnings import filterwarnings
filterwarnings('ignore')
import datetime

#### loading pipelines
from Location_pipeline import Location_processor
from Image_pipeline import Image_processor
from Amenity_pipeline import Amenities_processer
from NLP_pipeline import NLP_processor

def main():
    import pickle
    #### loading original data
    ori_data = pd.read_csv('../../Data/LA_Airbnb/listings_detailed.csv')

    #### 0) trim price
    ori_data['price_clean'] = [float(i.strip().replace('$','').split('.')[0].replace(',','')) for i in ori_data.price]

    #### 1) reading image features (not using Image_processor). **Do not** need to re-extract the features.
    time_start = datetime.datetime.now()
    print('Processing Image Features...')
        
    image_processor = Image_processor()
    image_features = pd.read_csv('./utiles/image_features.csv').rename(columns={'room_id':'id'})
    image_features = image_features.fillna(-1)
    image_features_col = list(image_features.columns)
    image_features_col = [image_features_col[0]] + [f'Image_{i}' for i in image_features_col[1:]]
    image_features.columns = image_features_col

    with open('./saved_pipelines/image_processor.pkl','wb') as f:
        pickle.dump(image_processor, f) ## save
        
    time_end = datetime.datetime.now()
    time_consumption = time_end-time_start
    print(f'done! time use: {round(time_consumption.total_seconds(), 3)}s')


    #### 2) reading location features. **Do not** need to re-extract the features.
    time_start = datetime.datetime.now()
    print('Processing Location Features...')

    import pickle
    location_processor = Location_processor()
    location_features = location_processor.process_airbnb_data(pickle.load(open('./utiles/area_features.pkl','rb')))

    with open('./saved_pipelines/location_processor.pkl','wb') as f:
        pickle.dump(location_processor, f) ## save
        
    time_end = datetime.datetime.now()
    time_consumption = time_end-time_start
    print(f'done! time use: {round(time_consumption.total_seconds(), 3)}s')


    #### 3) reading amenities features. **Need** to re-extract the features.
    #### the amenities_processor will store information like **Labelencoder**, which will be used in prediction phase
    time_start = datetime.datetime.now()
    print('Processing Amenities Features...')

    amenities_processor = Amenities_processer()
    amenities_features = amenities_processor.process_airbnb_data(ori_data)
    amenities_features_col = list(amenities_features.columns)
    amenities_features_col = [amenities_features_col[0]] + [f'Amenities_{i}' for i in amenities_features_col[1:]]
    amenities_features.columns = amenities_features_col

    with open('./saved_pipelines/amenities_processor.pkl','wb') as f:
        pickle.dump(amenities_processor, f) ## save
        
    time_end = datetime.datetime.now()
    time_consumption = time_end-time_start
    print(f'done! time use: {round(time_consumption.total_seconds(), 3)}s')


    #### 4) reading NLP features. **Need** to re-extract the features.
    #### the nlp_processor will store information like **used NERs**, which will be used in prediction phase
    time_start = datetime.datetime.now()
    print('Processing NLP Features...')

    nlp_processor = NLP_processor()
    nlp_features = nlp_processor.process_airbnb_data(ori_data)
    nlp_features_col = list(nlp_features.columns)
    nlp_features_col = [nlp_features_col[0]] + [f'NLP_{i}' for i in nlp_features_col[1:]]
    nlp_features.columns = nlp_features_col

    with open('./saved_pipelines/nlp_processor.pkl','wb') as f:
        pickle.dump(nlp_processor, f) ### save
        
    time_end = datetime.datetime.now()
    time_consumption = time_end-time_start
    print(f'done! time use: {round(time_consumption.total_seconds(), 3)}s')


    ### Merging all
    print('Merging all four part of features...')
    all_features = ori_data[['id','price_clean']].rename(columns={'price_clean':'price'}).merge(
        amenities_features, left_on='id',right_on='id',how='outer'
    ).merge(
        location_features, left_on='id',right_on='id',how='outer'
    ).merge(
        nlp_features, left_on='id',right_on='id',how='outer'
    ).merge(
        image_features, left_on='id',right_on='id',how='outer'
    ).fillna(-1) #### some image data are missing for some listings because we failed to scrape them from the web
    print('done!')

    ### saving all combined features
    print('Saving...')
    all_features.to_csv('./final_features/LA_extracted_all_features_imputed.csv',index=False)
    print('done!')

if __name__=="__main__":
    print('''
This is a **feature processing script** for LA airbnb pricing recommendation system.

It inputs (auto-intake)
    1) ../../Data/LA_Airbnb/listings_detailed.csv: Raw listing dataframes.
    2) ./utiles/image_features.csv: Extracted image features.
    3) ./utiles/area_features.pkl: Extracted location featrues.

It processes:
    1) Generate Image, Location, Amenities, NLP processors.
    2) For Locaiton and Image, it loads the pre-extracted data. For Amenities and NLP, it run throught the extraction pipeline.
    3) Engineering all features and combine them.

It saves:
    1) ./final_features/LA_extracted_all_features_imputed.csv: All combinded data with an "id" columns, a "price" columns, and other 1800+ imputed numeric features.
    2) ./saved_pipelines/[ image | nlp | location | amenities ]_processor.pkl: Four processors that will be used in prediciton phase.

To run the script: python update_features.py
    ''')

    print('\n## Start ##')
    main()
