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


N_JOBS = 3 


class PCA_pre_processing_pipeline():
    def __init__(self):
        ## missing values
        from sklearn.impute import SimpleImputer
        self.imputer = SimpleImputer(strategy='most_frequent')
        
    def fit(self,X,y=None):
#         X = X.rename(columns={i:i.replace(',','_').replace('>','_').replace('<','_') for i in X.columns})
        from sklearn.decomposition import PCA
        self.imputer.fit(X)
        #### some homogeneous features can be reduced
        ### 1) Amenities
        ## specials
        self.Amenities_specials = [
         'Amenities_bedrooms',
         'Amenities_beds',
         'Amenities_minimum_nights',
         'Amenities_maximum_nights',
         'Amenities_property_type_code',
         'Amenities_room_type_code',
         'Amenities_bathrooms_count',
         'Amenities_bathrooms_type_code',
         'Amenities_amenities_count']
        self.Amenities_rest = [i for i in X.columns if i.startswith('Amenities_') and i not in self.Amenities_specials]
        self.Amenities_PCA = PCA(n_components=10, random_state=42)
        print('Amenities_PCA...')
        self.Amenities_PCA.fit(X[self.Amenities_rest])
        
        ### 2) Image
        self.Image_specials = [
        'Image_size',
         'Image_sharpness',
         'Image_mean_brightness',
         'Image_contrast',
        'Image_mean_mean_min_dist_r3',
         'Image_mean_mean_weighted_min_dist_r3'
        ]
        self.Image_rest = [i for i in X.columns if i.startswith('Image_') and i not in self.Image_specials]
        self.Image_PCA = PCA(n_components=10, random_state=42)
        print('Image_PCA...')
        self.Image_PCA.fit(X[self.Image_rest])
        
        ### 3) Location
        self.Location_specials = [
        'Location_supermarket_num',
         'Location_restaurant_num',
         'Location_cafe_500m_dis',
         'Location_cafe_500m_num',
         'Location_transport_most_close_dis',
         'Location_transport_500m_num',
         'Location_transport_1000m_num',
        'Location_mean_area_accommodates_price',
         'Location_mean_area_beds_price',
         'Location_real_estate'
        ]
        self.Location_rest = [i for i in X.columns if i.startswith('Location_') and i not in self.Location_specials]
        self.Location_PCA = PCA(n_components=5, random_state=42)
        print('Location_PCA...')
        self.Location_PCA.fit(X[self.Location_rest])
        
        ### 4) NLP
        self.NLP_Embedding_list = [f'NLP_s{i}' for i in range(384)]
        self.NLP_NER_list = [i for i in X.columns if i.startswith('NLP_') and i not in self.NLP_Embedding_list]
        self.NLP_Embedding_PCA = PCA(n_components=20, random_state=42)
        self.NLP_NER_PCA = PCA(n_components=10, random_state=42)
        print('NLP_Embedding_PCA...')
        self.NLP_Embedding_PCA.fit(X[self.NLP_Embedding_list])
        print('NLP_NER_PCA...')
        self.NLP_NER_PCA.fit(X[self.NLP_NER_list])

    def transform(self,X,y=None):
#         X = X.rename(columns={i:i.replace(',','_').replace('>','_').replace('<','_') for i in X.columns})
        X[X.columns] = self.imputer.transform(X)
        ### 1) transform Amenities
        Amenities_pca_df = self.Amenities_PCA.transform(
            X[self.Amenities_rest]
        )
        Amenities_pca_df = pd.DataFrame(
            Amenities_pca_df, columns = [f'Amenities_PC{i+1}' for i in range(Amenities_pca_df.shape[1])]
        )
        
        ### 2) transform Image
        Image_pca_df = self.Image_PCA.transform(
            X[self.Image_rest]
        )
        Image_pca_df = pd.DataFrame(
            Image_pca_df, columns = [f'Image_PC{i+1}' for i in range(Image_pca_df.shape[1])]
        )
        
        ### 3) transform Location
        Location_pca_df = self.Location_PCA.transform(
            X[self.Location_rest]
        )
        Location_pca_df = pd.DataFrame(
            Location_pca_df, columns = [f'Location_PC{i+1}' for i in range(Location_pca_df.shape[1])]
        )
        
        ### 4) transform NLP
        NLP_pca_df1 = self.NLP_Embedding_PCA.transform(
            X[self.NLP_Embedding_list]
        )
        NLP_pca_df1 = pd.DataFrame(
            NLP_pca_df1, columns = [f'NLP_Embedding_PC{i+1}' for i in range(NLP_pca_df1.shape[1])]
        )
        
        NLP_pca_df2 = self.NLP_NER_PCA.transform(
            X[self.NLP_NER_list]
        )
        NLP_pca_df2 = pd.DataFrame(
            NLP_pca_df2, columns = [f'NLP_NER_PC{i+1}' for i in range(NLP_pca_df2.shape[1])]
        )
        
        print(
            'X[self.Amenities_specials].shape:',X[self.Amenities_specials].shape,
            'Amenities_pca_df.shape:',Amenities_pca_df.shape,
            'X[self.Image_specials].shape:',X[self.Image_specials].shape,
            'Image_pca_df.shape:',Image_pca_df.shape,
            'X[self.Location_specials].shape:',X[self.Location_specials].shape,
            'Location_pca_df.shape:',Location_pca_df.shape,
            'NLP_pca_df1.shape:',NLP_pca_df1.shape,
            'NLP_pca_df2.shape:',NLP_pca_df2.shape
        )
        
        X = pd.concat([
            X[self.Amenities_specials].reset_index(drop=True),
            Amenities_pca_df.reset_index(drop=True),
            X[self.Image_specials].reset_index(drop=True),
            Image_pca_df.reset_index(drop=True),
            X[self.Location_specials].reset_index(drop=True),
            Location_pca_df.reset_index(drop=True), 
            NLP_pca_df1.reset_index(drop=True),
            NLP_pca_df2.reset_index(drop=True)
        ], axis=1)
        
        return X
    
    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)
            
##### modeling 
class PCA_My_Airbnb_Capstone_Model():
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
        self.pre_processor = PCA_pre_processing_pipeline()
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


    def get_X_y(self,data,outlier_removal=False):
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
        # self.quantiles = [0.5]
        
        self.quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

        # fit models for each quantile
        self.models = []
        
        self.grid_search_cv_results_dict = {}
        for q in tqdm(self.quantiles, desc='Training quantile models'):
            
            if tuning:
                #### grid searching
                param_grid = {
                    # 'model__learning_rate': [0.01],
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
                                        verbose=2, refit='neg_mean_squared_error', error_score='raise', n_jobs=N_JOBS)
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
                #### define the pipeline
                self.pipeline_dict[q] = Pipeline(steps=[
                    ('preprocessor', self.pre_processor),
                    ('model', LGBMRegressor(random_state=42,objective='quantile', alpha=q,
                                            learning_rate=0.2,
                                            max_depth=11,
                                            num_leaves=60))])
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
    model = PCA_My_Airbnb_Capstone_Model()

    ### get X,y
    X, y = model.get_X_y(data)
    del data

    ### training
    model.train(X, y)

    ### saving
    with open('./Feature_Importance_tuning_PCA_cv_results.pkl', 'wb') as f:
        pickle.dump(model.grid_search_cv_results_dict, f)
    
    with open('./Feature_Importance_tuning_PCA_model.pkl', 'wb') as f:
        pickle.dump(model, f)



    
