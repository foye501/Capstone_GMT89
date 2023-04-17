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

    def transform(self,X):
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
        print('Filling infinite values with -1 ... Done')
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
            model = LGBMRegressor(random_state=42,objective='quantile', alpha=q)
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



