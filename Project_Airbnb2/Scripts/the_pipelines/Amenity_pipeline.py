import pandas as pd
import numpy as np
import openpyxl
import re
from re import sub
from decimal import Decimal
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from numpy import unique, where
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, silhouette_samples, silhouette_score , completeness_score , homogeneity_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from IPython.display import display, HTML, display_html
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

class Amenities_processer():
    def __init__(self):
        self.amenities_universe = open(r"./utiles/amenities_universe.txt", "r",encoding='unicode escape').read().split('\n')
        self.imputer = SimpleImputer(strategy='most_frequent')
        

    def process_airbnb_data(self,df):
        ### 1) process amenities
        df = df.reset_index(drop=True)
        df['amenities_list'] = [
            [
                e.lower() for e in \
                        list(
                                np.concatenate(
                                        [np.array(
                                            re.findall(r'"(.+)"',i)
                                            ).astype('str') for i in k.split(',')]
                                    )
                                ) \
                            if e.lower() in self.amenities_universe
            ] \
                            for k in df['amenities'].values]
        df['amenities_count'] = [len(i) for i in df.amenities_list.values]

        ### 2) process bathroom
        df['bathrooms_count'] = df['bathrooms_text'].apply(lambda x: re.findall(r"[-+]?(?:\d*\.*\d+)",str(x)) if x != 'half' else [0.5])
        df['bathrooms_count'] = df['bathrooms_count'].apply(lambda x: x[0] if len(x)>0 else 0)
        df['bathrooms_type'] = df['bathrooms_text'].apply(lambda x: 'shared' if 'shared' in str(x) else 'private')

        ### 3) process property types
        prohibitedWords = ['private', 'shared', ' in ', 'entire', 'room'] # added spaces in front and behind 'in' to ensure accuracy
        big_regex = re.compile('|'.join(map(re.escape, prohibitedWords)))
        df['property_type_clean'] =  df['property_type'].apply(lambda x: big_regex.sub("", x).strip())
        df['property_type_clean'] = np.where(df['property_type_clean']=='',df['room_type'],df['property_type_clean'])


        ### 4) additional columns
        self.additional_column_list = ['bedrooms', 'beds','minimum_nights', 'maximum_nights']

        ### 5) compile all
        # amenities df
        amenities_df = df[['id','amenities_list']]
        amenities_df['value'] = 1
        amenities_df = amenities_df.explode('amenities_list').pivot_table(columns='amenities_list',index='id').fillna(0).droplevel(axis=1,level=0)

        # amenities count df
        amenities_count_df = df[['id','amenities_count']]

        # bathrooms count df
        bathroom_count_df = df[['id','bathrooms_count']]

        # bathroom type df (encoding)
        bathrooms_type_df = df[['id','bathrooms_type']]
        bathrooms_type_df['bathrooms_type'] = [i.lower() for i in bathrooms_type_df['bathrooms_type'].values]
        self.bathroom_encoder = LabelEncoder()
        bathrooms_type_df['bathrooms_type_code'] = self.bathroom_encoder.fit_transform(bathrooms_type_df['bathrooms_type'])
        del bathrooms_type_df['bathrooms_type']

        # property type encoder
        property_type_df = df[['id','property_type_clean']]
        property_type_df['property_type_clean'] = [i.lower() for i in property_type_df['property_type_clean'].values]
        self.property_type_encoder = LabelEncoder()
        property_type_df['property_type_code'] = self.property_type_encoder.fit_transform(property_type_df['property_type_clean'])
        del property_type_df['property_type_clean']

        ### room type encoder
        room_type_df = df[['id','room_type']]
        room_type_df['room_type'] = [i.lower() for i in room_type_df['room_type'].values]
        self.room_type_encoder = LabelEncoder()
        room_type_df['room_type_code'] = self.room_type_encoder.fit_transform(df[['room_type']])
        del room_type_df['room_type']
        

        final_df = df[['id']+self.additional_column_list].merge(
            property_type_df, left_on='id', right_on='id', how='outer'
        ).merge(
            room_type_df, left_on='id', right_on='id', how='outer'
        ).merge(
            bathroom_count_df, left_on='id', right_on='id', how='outer'
        ).merge(
            bathrooms_type_df, left_on='id', right_on='id', how='outer'
        ).merge(
            amenities_count_df, left_on='id', right_on='id', how='outer'
        ).merge(
            amenities_df, left_on='id', right_on='id', how='outer'
        )

        # self.imputer

        self.final_x_names = [i for i in list(final_df.columns) if not i=='id']
        final_df[self.final_x_names] = self.imputer.fit_transform(final_df[self.final_x_names])
        return final_df
    
    def process_new_data(self, additional_dict, property_dict, room_dict, bathroom_dict, amenities_dict):
        ## property types:
        if 'property_type' in property_dict:
            if property_dict['property_type'] in self.property_type_encoder.classes_:
                values = self.property_type_encoder.transform(np.array(room_dict['property_type']).reshape(-1,1))
            else:
                values = np.nan
        else:
            values = np.nan
        property_type_df = pd.DataFrame(np.array(values).flatten().reshape(1,-1), columns=['property_type_code'], index=[0])
        
        ## room type
        if 'room_type' in room_dict:
            if room_dict['room_type'] in self.room_type_encoder.classes_:
                values = self.room_type_encoder.transform(np.array(room_dict['room_type']).reshape(-1,1))
            else:
                values = np.nan
        else:
            values = np.nan
        room_type_df = pd.DataFrame(np.array(values).flatten().reshape(1,-1), columns=['room_type_code'], index=[0])

        ## bathroom_count_df
        if 'bathroom_count' in bathroom_dict:
            values = bathroom_dict['bathroom_count']
        else:
            values = np.nan
        bathroom_count_df = pd.DataFrame(np.array(values).flatten().reshape(1,-1), columns=['bathrooms_count'], index=[0])

        ## bathrooms_type_df
        if 'bathrooms_type' in bathroom_dict:
            if bathroom_dict['bathrooms_type'] in self.bathroom_encoder.classes_:
                values = self.bathroom_encoder.transform(np.array(room_dict['bathrooms_type']).reshape(-1,1))
            else:
                values = np.nan
        else:
            values = np.nan
        bathrooms_type_df = pd.DataFrame(np.array(values).flatten().reshape(1,-1), columns=['bathrooms_type_code'], index=[0])


        ## amenities_count_df
        amenities_count = len(amenities_dict)
        amenities_count_df = pd.DataFrame(np.array(amenities_count).reshape(1,-1),columns=['amenities_count'], index=[0])
        
        ## amenities_df
        amenities_df = {i:np.nan for i in self.amenities_universe}
        for k in amenities_dict.keys():
            amenities_df[k] = amenities_dict[k]
        amenities_df = pd.DataFrame(amenities_df, index=[0])

        ### additional columns
        additional_column_dict = {i:np.nan for i in self.additional_column_list}
        for k in additional_dict:
            additional_column_dict[k] = additional_dict[k]
        additional_column_df = pd.DataFrame(additional_column_dict, index=[0])

        final_df = pd.concat([
            additional_column_df, ## ['bedrooms', 'beds','minimum_nights', 'maximum_nights']
            property_type_df, ## 
            room_type_df,
            bathroom_count_df,
            bathrooms_type_df,
            amenities_count_df,
            amenities_df
        ],axis=1)

        final_df = final_df[self.final_x_names]
        final_df[self.final_x_names] = self.imputer.transform(final_df[self.final_x_names])
        
        return final_df





