import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings('ignore')
import datetime
import pickle
from textwrap import dedent

class pre_processing_pipeline():
    def __init__(self):
        ## missing values
        from sklearn.impute import SimpleImputer
        self.imputer = SimpleImputer(strategy='most_frequent')
    def fit(self,X):
        self.imputer.fit(X)
    def transform(self,X):
        return self.imputer.transform(X)
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
            
##### modeling 
class My_Airbnb_Capstone_Model():
    def __init__(self):
        # self.amenities_f_list = amenities_f_list
        # self.location_f_list = location_f_list
        # self.image_f_list = image_f_list
        # self.NLP_f_list = NLP_f_list
    
        ### load pipelines
        self.amenities_processor = pickle.load(open('./saved_pipelines/amenities_processor.pkl','rb'))
        self.image_processor = pickle.load(open('./saved_pipelines/image_processor.pkl','rb'))
        self.location_processor = pickle.load(open('./saved_pipelines/location_processor.pkl','rb'))
        self.nlp_processor = pickle.load(open('./saved_pipelines/nlp_processor.pkl','rb'))

        self.scores = {}
        self.pre_processor = pre_processing_pipeline()
        self.x_names = []
    
    def my_train_test_split(self,data):
        all_features_no_first3000 = data.iloc[3000:,:] ### discard top 3000: they are used to train CNN. Prevent leakage.
        if 'id' in all_features_no_first3000.columns:
            del all_features_no_first3000['id'] ## remove id column

        from sklearn.model_selection import train_test_split
        X = all_features_no_first3000.iloc[:,1:] ## define X
        print('Filling infinite values with -1 ...')
        X[X.columns] = np.where(np.isinf(X[X.columns]), -1, X[X.columns]) #### filling infinite values with -1

        y = all_features_no_first3000.iloc[:,0] ## define y

        ## train test split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

        ### identify the four feature parts
        amenities_f_list = [i for i in X_train.columns if i.startswith('Amenities')] ### starts with 'Amenities'
        location_f_list = [i for i in X_train.columns if i.startswith('Location')]
        image_f_list = [i for i in X_train.columns if i.startswith('Image')]
        NLP_f_list = [i for i in X_train.columns if i.startswith('NLP')]

        print(dedent(f'''
        Amenities raw features count: {len(amenities_f_list)}
        Location raw features count: {len(location_f_list)}
        Image raw features count: {len(image_f_list)}
        NLP raw features count: {len(NLP_f_list)}
        '''))

        print(dedent(f'''
        X_train shape: {X_train.shape}
        X_test shape: {X_test.shape}
        y_train shape: {y_train.shape}
        y_test shape: {y_test.shape}
        '''))

        return X_train, X_test, y_train, y_test


    def scoring(self,pred,truth):
        ## Define scoring functions
        from sklearn.metrics import r2_score, mean_squared_error
        from scipy.stats import spearmanr
        res = {
            'r2': r2_score(pred, truth),
            'mean_squared_error':mean_squared_error(pred, truth),
            'spearmanr':spearmanr(pred, truth)[0]
        }
        print(res)
        return res

    def train(self, X_train, X_test, y_train, y_test):
        from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
        from xgboost import XGBRegressor
        self.x_names = list(X_train.columns)

        print('Feature transformation...')
        new_X_train = self.pre_processor.fit_transform(X_train) ## define pipeline
        print(f'X_train shape after transformation: {new_X_train.shape}')

        # define quantiles
        self.quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

        # fit models for each quantile
        self.models = []
        for q in tqdm(self.quantiles, desc='Training quantile models'):
            model = HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=42)
            model.fit(new_X_train, y_train)
            self.models.append(model)

        for model, quantile in zip(self.models, self.quantiles):
            print('quantile: ',quantile)
            pred = model.predict(self.pre_processor.transform(X_test))  ## predict
            scores = self.scoring(pred, y_test) ## scoring
            self.scores[quantile] = scores

    def predict(self, X):
        X = X[self.x_names]
        new_X = self.pre_processor.transform(X) ## define pipeline
        preds = {}
        for model, quantile in zip(self.models, self.quantiles):
            pred = model.predict(new_X)  ## predict
            preds[quantile] = pred

        return preds



