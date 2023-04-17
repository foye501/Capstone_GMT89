##
#pip install overpy
#pip install uszipcode

import overpy
from geopy.geocoders import Nominatim
import math
from geopy.distance import geodesic
import pandas as pd

from uszipcode import SearchEngine

#from google.colab import drive 
#drive.mount('/content/drive')

##################################
# Functions
##################################

## Function that returns closest distnce and number of the facilities
def NearbyDisAndNum(lat, lon, query):
  result = api.query(query.format(lat, lon))
  closest_distance = math.inf
  for node in result.nodes:
      distance = math.sqrt((float(node.lat) - lat)**2 + (float(node.lon) - lon)**2)
      if distance < closest_distance:
          closest_distance = distance
  return (closest_distance,len(result.nodes))

## Function that returns numbers of the facilities
def NearbyNum(lat, lon, query):
  result = api.query(query.format(lat, lon))
  return len(result.nodes)

## Function that find zipcode of inputted address
def get_zipcode(lat, lon):
    search = SearchEngine()
    RealEstateByZip = pd.read_pickle([(root+'/RealEstateByZip.pkl') for pp in sys.path for root, dirs, files in os.walk(pp) if 'RealEstateByZip.pkl' in files][0])
    result = search.by_coordinates(lat = lat, lng = lon, returns = 1)
    try :
      zip = result[0].zipcode
      real_estate = RealEstateByZip[RealEstateByZip["zipcode"]==zip].iat[0,1]
      return real_estate
    except IndexError:
      return None

## Calculate mean(Price) of same accommodates or beds 
def average_cal(flg, num,lat,lon, data_for_cal):
    if flg == "accommodates":
        data_temp = data_for_cal[data_for_cal["accommodates"]==num]
    elif flg == "beds":
        data_temp = data_for_cal[data_for_cal["beds"]==num]
    data_temp = data_temp[(data_temp["latitude"] > lat - 0.01) & (data_temp["latitude"] < lat + 0.01) ]
    data_temp = data_temp[(data_temp["longitude"] > lon - 0.01) & (data_temp["longitude"] < lon + 0.01) ]
    
    return data_temp["price"].mean()

## Calculate distance between other two point(lat, lon)
def distance(airbnb_lat, airbnb_lon, sight_lat, sight_lon):
  dis = round(geodesic((airbnb_lat, airbnb_lon), (sight_lat, sight_lon)).km, 2)
  return dis

## Making dict that inculdes all features
def main(lat, lon, ac, beds):
    ## Set a dict that keeps features calculated by the functions
    result_dict = {}

    ## Set queries that are used by the function "NearbyDisAndNum"
    transport = {
        "bus_stop_500m" : 'node["highway"="bus_stop"](around:500, {0}, {1}); out;',
        "bus_stop_1000m" : 'node["highway"="bus_stop"](around:1000, {0}, {1}); out;',
        "station_500m" : 'node[railway=station](around:500, {0}, {1}); out;',
        "station_1000m" : 'node[railway=station](around:1000, {0}, {1}); out;',
        "cafe_500m": 'node["amenity"="cafe"](around:500, {0}, {1}); out;'
    }

    ## Extract a distance and number 

    for key, value in transport.items():
        output = NearbyDisAndNum(lat,lon, value)
        print(output)
        dis = output[0]
        num = output[1]
        result_dict[key+"_dis"] = dis
        result_dict[key+"_num"] = num

    ## Set queries that are used by the function "NearbyNum"
    facility = {
        "restaurant":'node[amenity=restaurant](around:1000, {0}, {1}); out;',
        "supermarket":'node["shop"="supermarket"](around:1000, {0}, {1}); out;'
    }

    ## Extract number of facilities by the function "NearbyNum" 

    for key, value in facility.items():
        output = NearbyNum(lat,lon, value)
        result_dict[key+"_num"] = output

    ## Extract real estate average price
    result_dict["real_estate"] = real_estate = get_zipcode(lat, lon)

    ## To calculate mean() of price within same accommodates and beds

    #Preparing master dataset
    data = pd.read_pickle('/content/drive/MyDrive/Colab Notebooks/Capstone/data_zip.pkl')
    data["price"] = data.price.str.replace("$","")
    data["price"] = data.price.str.replace(",","")

    data["price"] = data["price"].astype('float32')

    data_for_cal = data[["id","latitude","longitude","accommodates","beds","price"]]

    #Insert mean of price to result_dict 
    result_dict["mean_area_accommodates_price"] = average_cal("accommodates",ac, lat, lon, data_for_cal)
    result_dict["mean_area_beds_price"] = average_cal("beds",beds, lat, lon, data_for_cal)

    #Preparing master sightseeng dataset
    df_sightseeing = pd.read_pickle('/content/drive/MyDrive/Colab Notebooks/Capstone/master_sightseeing.pkl')

    #Insert distance bewteen airbnb house and famous sightseeing facility
    for i in range(len(df_sightseeing)):
        result_dict[df_sightseeing.iat[i,0]] = distance(lat, lon, df_sightseeing.iat[i,2], df_sightseeing.iat[i,3])

    return result_dict


#####################################
# Main command
#####################################

## Get latitude and longitude from address
api = overpy.Overpass()
geolocator = Nominatim(user_agent="my-application")  

address = input("Please enter your Airbnb address: ")

location = geolocator.geocode(address)
lat = location.latitude
lon = location.longitude

accommodates = input("Please enter accommodates ")
beds = input("Please enter beds ")

result = main(lat, lon, accommodates, beds)

