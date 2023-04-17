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
#         X = X.rename(columns={i:i.replace(',','_').replace('>','_').replace('<','_') for i in X.columns})
        X[X.columns] = self.imputer.transform(X)
        print('Feature Transformed')
        return X
    
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

        #### outlier removal
        temp = all_features_no_first3000.iloc[:,[0]]
        temp = temp.index[
            (temp.iloc[:,0]>0) &\
            (np.log(temp.iloc[:,0]+1) < sorted(np.log(temp.iloc[:,0]+1))[int(temp.shape[0]*0.975)]) &\
            (np.log(temp.iloc[:,0]+1) > sorted(np.log(temp.iloc[:,0]+1))[int(temp.shape[0]*0.025)])
        ]
        all_features_no_first3000 = all_features_no_first3000[all_features_no_first3000.index.isin(list(temp))]
        
        #### train test split
        from sklearn.model_selection import train_test_split
        X = all_features_no_first3000.iloc[:,1:] ## define X
        y = all_features_no_first3000.iloc[:,0] ## define y
        del all_features_no_first3000
        print('Filling infinite values with -1 ...')
        X[X.columns] = np.where(np.isinf(X[X.columns]), -1, X[X.columns]) #### filling infinite values with -1
        print('Filling infinite values with -1 ... Done')
        

        ## train test split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

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
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor

        for n in X_train.columns:
            if '>' in n or '<' in n or ',' in n:
                del X_train[n]
            
        self.x_names = list(X_train.columns)

        print('Feature transformation...')
        self.pre_processor.fit(X_train)
        new_X_train = self.pre_processor.transform(X_train[self.x_names]) ## define pipeline
        new_X_test = self.pre_processor.transform(X_test[self.x_names])
        print(f'X_train shape after transformation: {new_X_train.shape}')
        
        #####
        # define quantiles
        self.quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

        # fit models for each quantile
        self.models = []
        for q in tqdm(self.quantiles, desc='Training quantile models'):
            if q==0.05:
                learning_rate=0.2
                max_depth=7
                num_leaves=48
            elif q==0.25 or q==0.5:
                learning_rate=0.2
                max_depth=11
                num_leaves=60
            elif q==0.75:
                learning_rate=0.1
                max_depth=13
                num_leaves=60
            elif q==0.95:
                learning_rate=0.2
                max_depth=13
                num_leaves=60
            else:
                break
                
            model = LGBMRegressor(random_state=42,objective='quantile', 
                                            alpha=q,learning_rate=learning_rate, max_depth=max_depth, num_leaves=num_leaves)
            model.fit(new_X_train, y_train)
            self.models.append(model)

        for model, quantile in zip(self.models, self.quantiles):
            print('quantile: ',quantile)
            pred = model.predict(new_X_test)  ## predict
            scores = self.scoring(pred, y_test) ## scoring
            self.scores[quantile] = scores
            
        ##### model_for_shap
        self.model_for_shap = self.models[2]
        self.model_for_shap_score = self.scores[0.5]
        

    def predict(self, X):
        
        X = X[self.x_names]
        new_X = self.pre_processor.transform(X) ## define pipeline
        new_X = new_X.apply(pd.to_numeric)
        preds = {}
        for model, quantile in zip(self.models, self.quantiles):
            pred = model.predict(new_X)  ## predict
            preds[quantile] = pred

        return preds
    
    def generate_shap_values(self, X):
        import shap
        X = self.pre_processor.transform(X[self.x_names])
        explainer = shap.Explainer(self.models[2])
        shap_values = explainer(X)
        
#         from types import SimpleNamespace
#         to_pass = SimpleNamespace(**{
#                           'values': np.array(shap_values[0].values),
#                           'data': np.array(shap_values[0].data),
#                           'feature_names': X.columns,
#                           'base_values': shap_values[0].base_values[0]
#             })
        to_pass = shap_values
        return to_pass



