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
import sys
sys.path = sys.path+['./utiles/']

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline


N_JOBS = 4


class pre_processing_pipeline():
    def __init__(self):
        ## missing values
        from sklearn.impute import SimpleImputer
        self.imputer = SimpleImputer(strategy='most_frequent')
        
    def fit(self,X,y=None):
        self.imputer.fit(X)


    def transform(self,X,y=None):
        X[X.columns] = self.imputer.transform(X)
        return X
    
    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)
            
##### modeling 
class My_Airbnb_Capstone_Model():
    def __init__(self):
        # self.amenities_f_list = amenities_f_list
        # self.location_f_list = location_f_list
        # self.image_f_list = image_f_list
        # self.NLP_f_list = NLP_f_list
        from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from sklearn.model_selection import GridSearchCV, KFold, learning_curve
        from sklearn.metrics import mean_squared_error
        from sklearn.pipeline import Pipeline
        
        ### load pipelines
        self.amenities_processor = pickle.load(open('./saved_pipelines/amenities_processor.pkl','rb'))
        self.image_processor = pickle.load(open('./saved_pipelines/image_processor.pkl','rb'))
        self.location_processor = pickle.load(open('./saved_pipelines/location_processor.pkl','rb'))
        self.nlp_processor = pickle.load(open('./saved_pipelines/nlp_processor.pkl','rb'))

        self.scores = {}
        self.pre_processor = pre_processing_pipeline()
        self.x_names = []
        self.pipeline_dict = {}
        self.params = {}
        self.grid_search_dict = {}
        self.learning_curve_dict = {}


    
    def my_train_test_split(self,data):
        all_features_no_first3000 = data.iloc[3000:,:] ### discard top 3000: they are used to train CNN. Prevent leakage.
        if 'id' in all_features_no_first3000.columns:
            del all_features_no_first3000['id'] ## remove id column

        #### outlier removal
        temp = all_features_no_first3000.iloc[:,[0]]
        temp = temp[
            (np.log(temp.iloc[:,0]+1) < sorted(np.log(temp.iloc[:,0]+1))[int(temp.shape[0]*0.99)]) &\
            (np.log(temp.iloc[:,0]+1) > sorted(np.log(temp.iloc[:,0]+1))[int(temp.shape[0]*0.01)])
        ]
        all_features_no_first3000 = all_features_no_first3000[all_features_no_first3000.index.isin(list(temp.index))]
        
        #### train test split
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


    def get_X_y(self,data, outlier_removal=False):
        all_features_no_first3000 = data.iloc[3000:,:] ### discard top 3000: they are used to train CNN. Prevent leakage.
        if 'id' in all_features_no_first3000.columns:
            del all_features_no_first3000['id'] ## remove id column

        #### outlier removal
        temp = all_features_no_first3000.iloc[:,[0]]
        temp = temp[temp['price']>0]
        if outlier_removal:
            temp = temp[
                (np.log(temp.iloc[:,0]+1) < sorted(np.log(temp.iloc[:,0]+1))[int(temp.shape[0]*0.99)]) &\
                (np.log(temp.iloc[:,0]+1) > sorted(np.log(temp.iloc[:,0]+1))[int(temp.shape[0]*0.01)])
            ]
            
        all_features_no_first3000 = all_features_no_first3000[all_features_no_first3000.index.isin(list(temp.index))]
        
        #### train test split
        from sklearn.model_selection import train_test_split
        X = all_features_no_first3000.iloc[:,1:] ## define X
        print('Filling infinite values with -1 ...')
        X[X.columns] = np.where(np.isinf(X[X.columns]), -1, X[X.columns]) #### filling infinite values with -1

        y = all_features_no_first3000.iloc[:,0] ## define y

        ### identify the four feature parts
        amenities_f_list = [i for i in X.columns if i.startswith('Amenities')] ### starts with 'Amenities'
        location_f_list = [i for i in X.columns if i.startswith('Location')]
        image_f_list = [i for i in X.columns if i.startswith('Image')]
        NLP_f_list = [i for i in X.columns if i.startswith('NLP')]

        print(dedent(f'''
        Amenities raw features count: {len(amenities_f_list)}
        Location raw features count: {len(location_f_list)}
        Image raw features count: {len(image_f_list)}
        NLP raw features count: {len(NLP_f_list)}
        '''))

        print(dedent(f'''
        X shape: {X.shape}
        y shape: {y.shape}
        '''))

        return X, y
    
    
    
#     def scorer_dict(self,estimator,X,y):
#         ## Define scoring functions
#         from sklearn.metrics import r2_score, mean_squared_error
#         from scipy.stats import spearmanr
#         from sklearn.metrics import make_scorer
        
#         def spearmanr_score(truth, pred):
#             return spearmanr(truth, pred)[0]
#         pred = estimator.predict(X)
#         scorer_dict = {
#             'r2':make_scorer(r2_score, greater_is_better=True)(y, pred),
#             'mean_squared_error':make_scorer(mean_squared_error, greater_is_better=False)(y, pred),
#             'spearmanr':make_scorer(spearmanr_score, greater_is_better=True)(y, pred)
#         }
        
#         return scorer_dict
    def spearmanr_score(self,truth, pred):
            return spearmanr(truth, pred)[0]
        
    def get_scoring_dict(self):
        ## Define scoring functions
        from sklearn.metrics import r2_score, mean_squared_error
        from scipy.stats import spearmanr
        from sklearn.metrics import make_scorer
        scorer_dict = {
            'r2':make_scorer(r2_score, greater_is_better=True),
            'mean_squared_error':make_scorer(mean_squared_error, greater_is_better=False),
            'spearmanr':make_scorer(self.spearmanr_score, greater_is_better=True),
            'neg_mean_squared_error':'neg_mean_squared_error'
        }
        return scorer_dict

    def train(self, X, y, tuning=False, weight_sample=False):

        for n in X.columns:
            if '>' in n or '<' in n or ',' in n:
                del X[n]
            
        self.x_names = list(X.columns)
#         self.transformed_X_name = list(new_X_train.columns)
        
        #####
        # define quantiles
        self.quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

        # fit models for each quantile
        self.models = []
        self.grid_search_cv_results_dict = {}
        for q in tqdm(self.quantiles, desc='Training quantile models'):
            
            if tuning:
                #### grid searching
                param_grid = {
    #                 'model__learning_rate': [0.01, 0.05],
                    'model__learning_rate': [0.05, 0.1, 0.2],
                    'model__num_leaves': [24, 36, 48, 60],
                    'model__max_depth': [5, 7, 11, 13],
                }
                
                #### k-fold
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                
                #### define the pipeline
                pipeline = Pipeline(steps=[
                    ('preprocessor', self.pre_processor),
                    ('model', LGBMRegressor(random_state=42,objective='quantile', alpha=q))
                ])
                
                #### apply grid search
                print(f'Running GridSearchCV for Quantile {q}')
                grid_search = GridSearchCV(pipeline, 
                                        param_grid, cv=cv, scoring=self.get_scoring_dict(), 
                                        verbose=2, refit='neg_mean_squared_error', error_score='raise',
                                        n_jobs=N_JOBS)
                grid_search.fit(X, y)
                print("Best hyperparameters:", grid_search.best_params_)
                print("Best score:", grid_search.best_score_)
                self.grid_search_dict[q] = grid_search
                self.scores[q] = grid_search.best_score_
                self.params[q] = grid_search.best_params_
                print('Best params:',grid_search.cv_results_)
                
                ### store scores
                self.grid_search_cv_results_dict[q] = grid_search.cv_results_
                
                #### learning curve
                print(f'Running learning curve for Quantile {q}')
                self.learning_curve_dict[q] = learning_curve(
                    pipeline, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring=make_scorer(mean_squared_error, greater_is_better=False), shuffle=True, random_state=42,
                    n_jobs=N_JOBS
                )

                #### final fit
                print(f'Final fitting for Quantile {q}')
                self.pipeline_dict[q] = Pipeline(steps=[
                    ('preprocessor', self.pre_processor),
                    ('model', LGBMRegressor(random_state=42,objective='quantile', alpha=q, **self.params[q]))
                ])
                self.pipeline_dict[q].fit(X, y)
            
            else:
                self.pipeline_dict[q] = Pipeline(steps=[
                    ('preprocessor', self.pre_processor),
                    ('model', LGBMRegressor(random_state=42,objective='quantile', 
                                            alpha=q,learning_rate=0.2,max_depth=7,num_leaves=48))
                ])
                
                if weight_sample:
                    self.pipeline_dict[q].fit(X, y, sample_weight=(y+1)**(1/2))
                else:
                    self.pipeline_dict[q].fit(X, y)

        ##### model_for_shap
        self.model_for_shap = self.pipeline_dict[0.5]
        self.model_for_shap_score = self.scores[0.5]
    

    def predict(self, X):
        
        X = X[self.x_names]
        preds = {}
        for q,pipeline in self.pipeline_dict.items():
            pred = pipeline.predict(X)  ## predict
            preds[q] = pred

        return preds
    
if __name__=='__main__':
    ### read features
    data = pd.read_csv('./final_features/LA_extracted_all_features_imputed.csv')
    data = data[
        (np.log(data['price']+1)>=sorted(np.log(data['price']+1))[int(0.025*len(data))]) &\
        (np.log(data['price']+1)<=sorted(np.log(data['price']+1))[int(0.975*len(data))])
        ] ### remove outliers

    ### define model
    model = My_Airbnb_Capstone_Model()

    ### get X,y
    X, y = model.get_X_y(data)
    del data

    ### training
    model.train(X, y)

    ### saving
    with open('./Feature_Importance_tuning_raw_feature_cv_results.pkl', 'wb') as f:
        pickle.dump(model.grid_search_cv_results_dict, f)
    with open('./Feature_Importance_tuning_raw_feature_model.pkl', 'wb') as f:
        pickle.dump(model, f)



    
