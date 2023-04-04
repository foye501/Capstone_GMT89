import requests
from bs4 import BeautifulSoup
from soup2dict import convert
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm

def scrape_photos(room_code):
    #### request
    room_urls = f'https://www.airbnb.com/rooms/{room_code}'
    a = requests.get(room_urls)
    time.sleep(1)

    #### get urls
    soup = BeautifulSoup(a.text, 'html.parser')
    b = convert(soup)
    d = b['html'][0]['body'][0]['script'][2]['#text'].split('PHOTO_TOUR_SCROLLABLE"')[1]

    import re
    #### parse urls
    all_urls_found = re.findall(r"(http.*?(jpeg|jpg))",d)

    ### clean urls
    all_url_list= []
    for i in all_urls_found:
        if 'picture' in i[0]:
            all_url_list.append(i[0])
    
    if not os.path.exists(f'./LA_photos/{room_code}'):
        os.mkdir(f'./LA_photos/{room_code}')
    
    
    donwloaded = 0
    for index,url in enumerate(all_url_list[:5]):
        # import cv2
        a = requests.get(url)
        # cv2.imwrite(f'./Boston_photos/{room_code}/{room_code}_{index}.jpg',a.content)

        with open(f'./LA_photos/{room_code}/{room_code}_{index}.jpg', 'wb') as f:
            f.write(a.content)
            donwloaded+=1

    # time.sleep(1)
    return donwloaded













