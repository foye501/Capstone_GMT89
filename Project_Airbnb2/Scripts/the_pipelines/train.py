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

def main():
    ### read features
    data = pd.read_csv('./final_features/LA_extracted_all_features_imputed.csv')
    data = data[
        (np.log(data['price']+1)>=sorted(np.log(data['price']+1))[int(0.025*len(data))]) &\
        (np.log(data['price']+1)<=sorted(np.log(data['price']+1))[int(0.975*len(data))])
        ] ### remove outliers
    
    
    
    
    from Airbnb_Capstone_Model import My_Airbnb_Capstone_Model
    my_model = My_Airbnb_Capstone_Model()

    ### train test split
    X_train, X_test,y_train,y_test = my_model.my_train_test_split(data) ### take the LA_extracted_all_features_imputed.csv

    ### training
    my_model.train(X_train, X_test, y_train, y_test)

    ### dump
    with open('./trained_models/overall_model.pkl','wb') as f:
        pickle.dump(my_model, f)
        
    print('Training finshied!')
    print('Model stored in ./trained_models/overall_model.pkl')

if __name__=="__main__":
    print(dedent('''
        This is a training script.
        
        It inputs:
            1) ./final_features/LA_extracted_all_features_imputed.csv: Final combinded features
            
        It processes:
            1) Generate a My_Airbnb_Capstone_Model class
            2) Train-test-split and training quantile regression models
            
        It saves:
            1) A trained "My_Airbnb_Capstone_Model" class model in ./trained_models/overall_model.pkl
            
          '''))
    print("## start ##")
    main()