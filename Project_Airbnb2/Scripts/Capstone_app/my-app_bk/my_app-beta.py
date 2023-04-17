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

import requests
from geopy.geocoders import Nominatim
from shapely.geometry import Point, Polygon
import json
with open('/home/capstone/Project_Airbnb2/Data/LA_Airbnb/laMap.geojson', 'r') as f:
    geojson = json.load(f)

sys.path = sys.path + ['../the_pipelines'] + ['../the_pipelines/utiles']

'loading model...'
additional_address = ''
address = ''
lat = 34.028580
lon = -118.383470

sorted_amenities_list = pickle.load(open('../the_pipelines/utiles/sorted_amenities.pkl','rb'))
sorted_property_types_list = pickle.load(open('../the_pipelines/utiles/sorted_property_types.pkl','rb'))
sorted_room_types_list = pickle.load(open('../the_pipelines/utiles/sorted_room_types.pkl','rb'))
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
    lat = 34.028580
    lon = -118.383470

url = "https://nominatim.openstreetmap.org/search/" + address + "?format=json"
response = requests.get(url).json()

if response:
    lat = float(response[0]["lat"])
    lon = float(response[0]["lon"])

    point = Point(lon, lat)
    for feature in geojson['features']:
        coordinates = feature['geometry']['coordinates'][0]
        flattened_coordinates = [coord for sublist in coordinates for coord in sublist]
        polygon = Polygon(flattened_coordinates)
        if not polygon.contains(point):
            st.markdown("""**The address is  :red[outside Los Angeles.],
                            Please confirm the address**""")

# this is just for the lat and longitude of Los Angel
# chart_data = pd.DataFrame(
#    np.random.randn(1, 2) / [50, 50] + [34.052235, -118.243683],
#    columns=['lat', 'lon'])

# st.pydeck_chart(pdk.Deck(
#     map_style=None,
#     initial_view_state=pdk.ViewState(
#         latitude=34.052235,
#         longitude=-118.243683,
#         zoom=11,
#         pitch=50,

#     ),
# ))

chart_data = pd.DataFrame(
    data = {'latitude':[lat],
            'longitude':[lon]
           })

st.map(chart_data, zoom=11)

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
img_count = 0
uploaded_files=st.file_uploader("Choose images",accept_multiple_files=True)
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
    ],axis=1)[my_capstone_model.x_names].fillna(-1)
    print('Feature engineering done!')
    pred = my_capstone_model.predict(input_features)
#     to_pass = my_capstone_model.generate_shap_values(input_features)

    #### SHAP
    
    X = my_capstone_model.pre_processor.transform(input_features[my_capstone_model.x_names]).fillna(-1)
    explainer = shap.Explainer(my_capstone_model.models[2])
    shap_values = explainer(X)
    
    st.write(type(shap_values))
    st.write(shap_values)
    

#     from types import SimpleNamespace
#     class to_pass_class():
#         def __init__(self,shap_values,X):
#             self.shap_values = shap_values
#             self.X = X
#             self.base_values = self.shap_values[0].base_values[0]
#             self.values = np.array(self.shap_values[0].values)
#             self.data = np.array(self.shap_values[0].data)
#             self.feature_names = X.columns
#             pass
        
#         def display_data(self):
            
#             return SimpleNamespace(**{'values': np.array(self.shap_values[0].values),
#                       'data': np.array(self.shap_values[0].data),
#                       'feature_names': X.columns,
#                       'base_values': self.shap_values[0].base_values[0]})
        
#     to_pass = to_pass_class(shap_values,X)
# #     to_pass = SimpleNamespace(**{
# #                       'values': np.array(shap_values[0].values),
# #                       'data': np.array(shap_values[0].data),
# #                       'feature_names': X.columns,
# #                       'base_values': shap_values[0].base_values[0]
# #         })
#     print(to_pass)
#     to_pass = shap_values[0]
    to_pass = [explainer.base_values[0], shap_values[0], X[0]]
    st.write(to_pass)
    return pred, to_pass




##### runing model
if st.button('Run model'):
    st.write('Model running...')
    my_capstone_model = pickle.load(open('../the_pipelines/trained_models/overall_model.pkl','rb'))
    try:
        pred, to_pass = run_model(my_capstone_model, ## model
                        txt_title,img_count, ### inputs...
                        address,additional_address,
                        no_bedrooms, no_beds, minimum_nights, maximum_nights,
                        property_type, room_type, 
                        bathroom_type, no_bathrooms,
                        amenity1, amenity2, amenity3,
                        txt_des)
        shap.plots.waterfall(**to_pass)
        pred
    except:
        st.markdown("""**:red[There is an error..],
                        :red[Please contact Customer Center.]**""")        

else:
    pass
