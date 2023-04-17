import pandas as pd
import spacy
import os
from tqdm import tqdm
import numpy as np
from collections import Counter

## class part
import overpy
from geopy.geocoders import Nominatim
import math
from geopy.distance import geodesic
from uszipcode import SearchEngine
import sys
import requests

class Location_processor():
  def __init__(self):
    self.api = overpy.Overpass()
    self.data_with_zip = pd.read_pickle([(root+'/data_with_zip.pkl') for pp in sys.path for root, dirs, files in os.walk(pp) if 'data_with_zip.pkl' in files][0])
    self.df_sightseeing = pd.read_pickle([(root+'/master_sightseeing.pkl') for pp in sys.path for root, dirs, files in os.walk(pp) if 'master_sightseeing.pkl' in files][0])
    self.RealEstateByZip = pd.read_pickle([(root+'/RealEstateByZip.pkl') for pp in sys.path for root, dirs, files in os.walk(pp) if 'RealEstateByZip.pkl' in files][0])
    pass

  ## Function that returns closest distnce and number of the facilities
  def NearbyDisAndNum(self, api, lat, lon, query):
    # self.api = overpy.Overpass()
    if lat == None:
      return (None, None)
    else:
      print('start requesting')
      result = api.query(query.format(lat, lon))
      print('finish requesting')
      closest_distance = math.inf
      for node in result.nodes:
          print('node',node)
          distance = math.sqrt((float(node.lat) - lat)**2 + (float(node.lon) - lon)**2)
          if distance < closest_distance:
              closest_distance = distance
      return (closest_distance,len(result.nodes))

  ## Function that returns numbers of the facilities
  def NearbyNum(self, api, lat, lon, query):
    # self.api = overpy.Overpass()
    if lat == None:
       return None
    else:
      result = api.query(query.format(lat, lon))
      return len(result.nodes)

  ## Function that find zipcode of inputted address
  def get_zipcode(self,lat, lon):
      search = SearchEngine()
      result = search.by_coordinates(lat = lat, lng = lon, returns = 1)
      RealEstateByZip = self.RealEstateByZip
      try :
          zip = result[0].zipcode
          real_estate = RealEstateByZip[RealEstateByZip["zipcode"]==int(zip)].iat[0,1]
          return real_estate
      except IndexError:
          return None

  ## Calculate mean(Price) of same accommodates or beds 
  def average_cal(self,flg, num,lat,lon, data_for_cal):
    if num == None or lat == None:
      return None
    
    if flg == "accommodates":
        data_temp = data_for_cal[data_for_cal["accommodates"]==int(num)]
    elif flg == "beds":
        data_temp = data_for_cal[data_for_cal["beds"]==float(num)]
        data_temp = data_temp[(data_temp["latitude"] > lat - 0.01) & (data_temp["latitude"] < lat + 0.01) ]
        data_temp = data_temp[(data_temp["longitude"] > lon - 0.01) & (data_temp["longitude"] < lon + 0.01) ]
    
    return data_temp["price"].mean()

  ## Calculate distance between other two point(lat, lon)
  def distance(self,airbnb_lat, airbnb_lon, sight_lat, sight_lon):
      if airbnb_lat == None:
        return None
      else:
        dis = round(geodesic((airbnb_lat, airbnb_lon), (sight_lat, sight_lon)).km, 2)
        return dis

  def process_airbnb_data(self,df):
      # from google.colab import drive 
      # drive.mount('/content/drive')
      df_temp = pd.read_pickle([(root+'/area_features.pkl') for pp in sys.path for root, dirs, files in os.walk(pp) if 'area_features.pkl' in files][0])


      extracted_features = pd.merge(df["id"], df_temp.drop(["latitude","longitude"], axis = 1), on = "id", how = "left")
      col = extracted_features.columns
      loc_col = ["Location_" + i for i in col ]
      loc_col[0] = "id"
      extracted_features.columns = loc_col

      return extracted_features
  

  ## Making dict that inculdes all features
  def process_new_data(self, api, address, ac = None, beds = None):
      ## Set a dict that keeps features calculated by the functions

      if address == None:
        result_dict = {}
      else:
        url = "https://nominatim.openstreetmap.org/search/" + address + "?format=json"
        response = requests.get(url).json()
        if response:
           lat = float(response[0]["lat"])
           lon = float(response[0]["lon"])
        else:
            lat = 34.028580
            lon = -118.383470
            print("The address didn't find latitude and longitude. The system use default address.")

        result_dict = {}
        # geolocator = Nominatim(user_agent=str(ut))
        # location = geolocator.geocode(address, timeout=10)
        print('parsed location: lat = {}, lon = {} '.format(lat,lon))
        # lat = location.latitude
        # lon = location.longitude
        result_dict['latitude' ] = lat
        result_dict['longitude'] = lon

      print('Set queries that are used by the function "NearbyDisAndNum"')
      ## Set queries that are used by the function "NearbyDisAndNum"
      transport = {
          "bus_stop_500m" : 'node["highway"="bus_stop"](around:500, {0}, {1}); out;',
          "bus_stop_1000m" : 'node["highway"="bus_stop"](around:1000, {0}, {1}); out;',
          "station_500m" : 'node[railway=station](around:500, {0}, {1}); out;',
          "station_1000m" : 'node[railway=station](around:1000, {0}, {1}); out;',
          "cafe_500m": 'node["amenity"="cafe"](around:500, {0}, {1}); out;'
      }

      print('Extract a distance and number ')
      ## Extract a distance and number 
      for key, value in transport.items():
        print(f'getting... {key}')
        if address == None:
            result_dict["Location_"+key+"_dis"] = None
            result_dict["Location_" + key+"_num"] = None
        else:          
          output = self.NearbyDisAndNum(api, lat, lon, value)
          dis = output[0]
          num = output[1]
          result_dict["Location_"+key+"_dis"] = dis*111
          result_dict["Location_" + key+"_num"] = num
        

      print('Requesting NearbyNum')
      ## Set queries that are used by the function "NearbyNum"
      facility = {
          "restaurant":'node[amenity=restaurant](around:1000, {0}, {1}); out;',
          "supermarket":'node["shop"="supermarket"](around:1000, {0}, {1}); out;'
      }

      print('Extract number of facilities by the function "NearbyNum"')
      ## Extract number of facilities by the function "NearbyNum" 
      for key, value in facility.items():
        if address == None:
          result_dict["Location_" + key+"_num"] = None
        else:          
          output = self.NearbyNum(api, lat,lon, value)
          result_dict["Location_" + key+"_num"] = output

      print('Extract real estate average price')
      ## Extract real estate average price
      if address == None:
        result_dict["Location_real_estate"]  = None
      else:
        result_dict["Location_real_estate"]  = self.get_zipcode(lat, lon)

      print('mean price with same accommodates and beds')
      ## To calculate mean() of price within same accommodates and beds

      #Preparing master dataset
      data = self.data_with_zip

      data_for_cal = data[["id","latitude","longitude","accommodates","beds","price"]]

      #Insert mean of price to result_dict 
      if address ==None:
        result_dict["Location_mean_area_accommodates_price"] = None
        result_dict["Location_mean_area_beds_price"] = None
      else:
        result_dict["Location_mean_area_accommodates_price"] = self.average_cal("accommodates",ac, lat, lon, data_for_cal)
        result_dict["Location_mean_area_beds_price"] = self.average_cal("beds",beds, lat, lon, data_for_cal)

      print('Preparing sightseeng dataset')
      #Preparing master sightseeng dataset
      df_sightseeing = self.df_sightseeing

      #Insert distance bewteen airbnb house and famous sightseeing facility
      for i in range(len(df_sightseeing)):
        if address == None:
          result_dict["Location_"+df_sightseeing.iat[i,0]] = None
        else:
          result_dict["Location_"+df_sightseeing.iat[i,0]] = self.distance(lat, lon, df_sightseeing.iat[i,2], df_sightseeing.iat[i,3])


      # result = pd.DataFrame.from_dict(result_dict)
      print('Preparing final location features')
      df_result = pd.DataFrame(result_dict,index = [0])
      if address == None:
        df_result["Location_transport_most_close_dis"] = None
        df_result["Location_transport_1000m_num"] =  None
        df_result["Location_transport_500m_num"] =  None
      else:       
        df_result["Location_transport_most_close_dis"] = df_result.apply(lambda x : min([x["Location_bus_stop_500m_dis"],x["Location_bus_stop_1000m_dis"],x["Location_station_1000m_dis"],x["Location_station_500m_dis"]]), axis=1)
        df_result["Location_transport_1000m_num"] =  df_result.apply(lambda x : sum([x["Location_bus_stop_1000m_num"],x["Location_station_1000m_num"]]),axis=1)
        df_result["Location_transport_500m_num"] =  df_result.apply(lambda x : sum([x["Location_bus_stop_500m_num"],x["Location_station_500m_num"]]),axis=1)
      df_result = df_result.drop(["Location_bus_stop_500m_dis","Location_bus_stop_500m_num", "Location_bus_stop_1000m_dis",
                                  "Location_bus_stop_1000m_num","Location_station_500m_dis","Location_station_500m_num",
                                  "Location_station_1000m_dis","Location_station_1000m_num"], axis = 1)
#       df_result = df_result[['Location_latitude', 'Location_longitude', 'Location_supermarket_num',
#                              'Location_bus_500m_dis', 'Location_bus_500m_num', 'Location_bus_1000m_dis', 
#                             'Location_bus_1000m_num','Location_station_1000m_dis', 'Location_station_1000m_num',
#                             'Location_station_500m_dis', 'Location_station_500m_num', 'Location_restaurant_num',
#                              'Location_cafe_500m_dis', 'Location_cafe_500m_num', 'Location_Hollywood Walk of Fame',
#                              'Location_Griffith Observatory', 'Location_Universal Studios Hollywood', 'Location_Santa Monica Pier',
#                              'Location_Angeles Stadium', 'Location_Dodgers Stadium', 'Location_California Science Center',
#                              'Location_TCL Chinese Theatre', 'Location_Little Tokyo', 'Location_Venice Beach',
#                              'Location_Norton Simon Museum', 'Location_Melrose Avenue', 'Location_ACMA',
#                              'Location_Rodeo Drive', 'Location_Santa Monica Promenade', 'Location_Farmers Market',
#                              'Location_Abbot Kinney Blvd', 'Location_Los Angeles International Airport', 'Location_mean_area_accommodates_price',
#                              'Location_mean_area_beds_price', 'Location_real_estate', 'Location_transport_most_close_dis',
#                              'Location_transport_1000m_num', 'Location_transport_500m_num']]
      df_result[df_result .columns] = np.where(np.isinf(df_result [df_result .columns]), -1, df_result [df_result .columns]) 
                          
      return df_result 

    
