import streamlit as st
import shap
import pandas as pd
import numpy as np
import os

import pydeck as pdk
import pickle
import sys
import cv2
import overpy
import shap

import matplotlib.pyplot as plt

sys.path = sys.path + ['../the_pipelines'] + ['../the_pipelines/utiles']

from Airbnb_Capstone_Model import My_Airbnb_Capstone_Model,pre_processing_pipeline

'loading model...'
additional_address = ''
address = ''

sorted_amenities_list = pickle.load(open('../the_pipelines/utiles/sorted_amenities.pkl','rb'))
sorted_property_types_list = pickle.load(open('../the_pipelines/utiles/sorted_property_types.pkl','rb'))
sorted_room_types_list = pickle.load(open('../the_pipelines/utiles/sorted_room_types.pkl','rb'))
#
# sorted_amenities_list

st.title("Price Suggestion App")

#### 0) title
st.markdown("### Let's give your house a title")
txt_title = st.text_input('Short title work best. Have fun with it', "Listing_title")

#### 1) Property type
st.markdown('### Select the best description of your property ###')
property_type = st.selectbox(
    '(quotes below shows the frequency of property type)',
    tuple(['Not listed']+[i[0]+f' ({str(round(i[1]*100, 2))}%)' for i in sorted_property_types_list[:10]])
)

if st.checkbox('Not listed? Show other less frequent types'):
    additional_property_type = st.radio(
        '',
        tuple([i[0]+f' ({str(round(i[1]*100, 2))}%)' for i in sorted_property_types_list[10:20]]),
    )


#### 2) Room type
st.markdown("### Select the room type ###")
room_type = st.selectbox(
    '(quotes below shows the frequency of room type)',
    tuple([i[0]+f' ({str(round(i[1]*100, 2))}%)' for i in sorted_room_types_list[:10]])
)


### 3) location
additional_address = ''
st.markdown("### Where's your place located ")
address=st.text_input("",'Enter your address')
if st.checkbox('Select a default one'):
    additional_address = '1st Helms Ave, Culver City, California, United States'



# this is just for the lat and longitude of Los Angel
chart_data = pd.DataFrame(
   np.random.randn(1, 2) / [50, 50] + [34.052235, -118.243683],
   columns=['lat', 'lon'])

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=34.052235,
        longitude=-118.243683,
        zoom=11,
        pitch=50,

    ),
))


### 4) Other basic information
st.markdown("### Shared some basics about your place")
no_guests = st.slider('How many Guests', 1, 10, 2)
no_bedrooms = st.slider('How many Bedrooms', 1, 15, 2)
no_beds = st.slider('How many Beds', 1, 15, 3)
no_bathrooms = st.slider('How many Bathrooms', 1, 4, 2)
bathroom_type = st.selectbox('Bathroom type', ('private','shared'))
minimum_nights = st.number_input('Mininum nights for booking', 1, 365, 1)
maximum_nights = st.number_input('Maximum nights for booking', 1, 365, 365)


### 5) amenities
col5, col6,col7 = st.columns(3)
col5.markdown("### What amenities do you offer")
amenity1 = col5.multiselect(
    '',
    [i[0] for i in sorted_amenities_list[:40]],
    [i[0] for i in sorted_amenities_list[:5]])

col6.markdown("### Do you have any standout amenities")

amenity2 = col6.multiselect(
    'What are your favorite colors',
    ['pool', 'hot hub', 'bbq grill', 'piano',],
    ['pool', 'hot hub'])

col7.markdown("### Do you have any of these safety items")
amenity3 = col7.multiselect(
    'What are your favorite colors',
    ['smoke Alarm', 'first aid kit', 'fire extiguisher', 'carbon monoxide',],
    ['smoke Alarm'])


### 6) Image
st.markdown("### Upload images of your property")
img_count = 0
uploaded_files=st.file_uploader("Choose images (only take .jpg currently)",accept_multiple_files=True)
for count,uploaded_file in enumerate(uploaded_files):
    img_count+=1
    bytes_data = uploaded_file.read()
    st.image(bytes_data)
    
    with open(f"../the_pipelines/new_data/{txt_title}_{img_count}.jpg","wb") as f:
        f.write(uploaded_file.getbuffer())
    
if st.checkbox('Or take a picture'):
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        img_count+=1
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cam_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        with open(f"../the_pipelines/new_data/{txt_title}_{img_count}.jpg","wb") as f:
            f.write(img_file_buffer.getbuffer())
    

#### 7) descriptions
st.markdown("### Next, let's describe your house")

# description = st.multiselect(
#     "Choose up to 2 highlights, We'll use these to get your description started",
#     ['Peaceful', 'unique', 'Family-friendly', 'Stylish',"Central","Spacious"],
#     )

# st.markdown("### Create your description")

txt_des = st.text_area('Share what makes your place special', "This is a nice place!")



def run_model(my_capstone_model,
              txt_title, img_count,
              address,additional_address,
              no_bedrooms, no_beds, minimum_nights, maximum_nights,
              property_type, room_type, 
              bathroom_type, no_bathrooms,
              amenity1, amenity2, amenity3,
              txt_des
              
              ):

    ####### processing input data
    ### extract features
    if img_count==0:
        st.write('using default image')
        img_path_list = ["../the_pipelines/new_data/test.jpg"] ### use test jpg if user did not upload one
    else:
        img_path_list = [f"../the_pipelines/new_data/{txt_title}_{k}.jpg" for k in range(img_count)]
        print('img_used:',img_path_list)

    address = additional_address if additional_address!='' else address
    if address=='Enter your address' or address=='': ### use test address if user did not upload one
        st.write('using default address')
        st.write('1st Helms Ave, Culver City, California, United State')
        address='1st Helms Ave, Culver City, California, United States'
    print(address)
    additional_dict = {'bedrooms':no_bedrooms, 
                    'beds':no_beds,
                    'minimum_nights':minimum_nights, 
                    'maximum_nights':maximum_nights}
    property_dict = {'property_type':property_type.split(' (')[0]} 
    room_dict = {'room_type':room_type.split(' (')[0]}
    bathroom_dict = {'bathroom_type':bathroom_type, 'bathroom_count':no_bathrooms}
    amenities_dict = {i:1 for i in amenity1+amenity2+amenity3}
    descriptions = txt_des

    ####### feature engineering
    ### img features
    print('Processing image data...')
    img_features = my_capstone_model.image_processor.process_new_data(img_path_list)
    img_features = pd.DataFrame(img_features,index=[0])
    img_features.columns = ['Image_'+i for i in img_features.columns]

    ### loc features
    print('Processing location data...')
    try:
        loc_features = my_capstone_model.location_processor.process_new_data(overpy.Overpass(), address)
    except Exception as e:
        print(e)
        st.write('adddress not working, using default address')
        print('adddress not working, using default address')
        loc_features = my_capstone_model.location_processor.process_new_data(overpy.Overpass(),'1st Helms Ave, Culver City, California, United States')
        

    ### amenities features
    print('Processing amenities data...')
    print(additional_dict,property_dict,room_dict,bathroom_dict,amenities_dict)
    amenities_features = my_capstone_model.amenities_processor.process_new_data(additional_dict,
                                                            property_dict, 
                                                            room_dict, 
                                                            bathroom_dict, 
                                                            amenities_dict)
    amenities_features.columns = ['Amenities_'+i for i in amenities_features.columns]

    ### descriptions features
    print('Processing description data...')
    descriptions_features = my_capstone_model.nlp_processor.process_new_data(descriptions)
    descriptions_features.columns = ['NLP_'+i for i in descriptions_features.columns]

    input_features = pd.concat([
        amenities_features,
        loc_features,
        descriptions_features,
        img_features
    ],axis=1)[my_capstone_model.x_names]

    print('Feature engineering done!')
    
    #### prediction
    #print(input_features.columns)
    
    input_features = input_features.apply(pd.to_numeric)
    pred = my_capstone_model.predict(input_features)
    
    #### SHAP
    import shap
    explainer = shap.Explainer(my_capstone_model.model_for_shap)
    shap_values = explainer(my_capstone_model.pre_processor.transform(input_features[my_capstone_model.x_names]))

    
    return pred, shap_values



def plot_t_distribution(quantiles, values):
    import scipy.stats as stats
    import scipy
    
    f = lambda q,nu,mu,sigma: scipy.stats.t.ppf(q=q, df=nu, loc=mu, scale=sigma)

    nu,mu,sigma = scipy.optimize.curve_fit(f,quantiles,values, p0=[50, 200, 30])[0]
    x = [i for i in np.linspace(mu-sigma*3.5,mu+sigma*3.5,100)]
    
    fig = plt.figure(figsize=(10,4))
    plt.plot(
        x,
        scipy.stats.t.pdf(x=x,df=nu,loc=mu,scale=sigma)
    )
    
    for q,v in zip(quantiles, values):
        fitted_v = scipy.stats.t.ppf(q=q,df=nu,loc=mu,scale=sigma) ### price at this quantile
        fitted_prob = scipy.stats.t.pdf(x=fitted_v,df=nu,loc=mu,scale=sigma) ### probability at this quantile
        
        plt.plot([fitted_v,fitted_v], [0,fitted_prob])
        
        ax=plt.gca()
        
        a,b = round(q, 2), round(fitted_v, 2)
#         if q==0.05:
#             x_adj = -70
#         elif q==0.25:
#             x_adj = -60
#         elif q==0.5:
#             x_adj = 0
#         elif q==0.75:
#             x_adj = 20
#         elif q==0.95:
#             x_adj = 30
#         else:
#             x_adj = 0
            
        ax.annotate(f'Quantile: {a}, \nPrice: {b}', xy=(fitted_v, fitted_prob*1.2), xytext=(fitted_v, fitted_prob*1.6),
                    ha='center', va='bottom',
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        if q==0.5: 
            plt.ylim([0,fitted_prob*2])

    plt.title('Probability distribution of the proper price', fontsize=20)
    plt.xlabel('Price ($)')
    plt.tight_layout()
    pricing_df = pd.DataFrame({
        'Quantiles':quantiles,
        'Price ($)':[scipy.stats.t.ppf(q=q,df=nu,loc=mu,scale=sigma) for q in quantiles]
    })
    
    median_price = int(scipy.stats.t.ppf(q=0.5,df=nu,loc=mu,scale=sigma))
    pricing_title = f'<p style="color:Black; font-size: 20px;">Suggested price: $ {median_price}</p>'
    st.markdown(pricing_title, unsafe_allow_html=True)
    
    st.write(pricing_df)
    return fig,pricing_df


##### runing model
if st.button('Run model'):
    st.write('Model running...')
    my_capstone_model = pickle.load(open('../the_pipelines/trained_models/overall_model.pkl','rb'))

    pred, shap_values = run_model(my_capstone_model, ## model
                txt_title,img_count, ### inputs...
                address,additional_address,
                no_bedrooms, no_beds, minimum_nights, maximum_nights,
                property_type, room_type, 
                bathroom_type, no_bathrooms,
                amenity1, amenity2, amenity3,
                txt_des)
    
    #######
    st.write('\n\n\n\n')
    report_title = '<p style="color:Black; font-size: 50px;">Your Pricing Report</p>'
    st.markdown(report_title, unsafe_allow_html=True)
    
    ####### get the pdf
    # Define the quantiles and their corresponding values
    quantiles = np.array([i for i in pred.keys()]).flatten()
    values = np.array([pred[i] for i in quantiles]).flatten()
    fig, pricing_df = plot_t_distribution(quantiles, values)
    st.pyplot(fig)
    st.write('Higher pricing results in higher profit, but lower competitiveness among your rivals; \nLower price increase your chances to be chosen by guests, but lower revenue.\n\n\n\n')

    
    ##### plot shap
    SHAP_title = '<p style="color:Black; font-size: 30px;">Features that have impact on your pricing</p>'
    st.markdown(SHAP_title, unsafe_allow_html=True)
    
    fig = plt.figure(figsize=(10,10))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title('SHAP waterfall plot of feature contributions', fontsize=20)
    st.pyplot(fig)
    st.write('Red/blue arrows indicate the pros/cons of your listing, and how large the impact is on your pricing. The blue arrowed features are potential directions for improvements.\n\n\n\n')


else:
    pass
