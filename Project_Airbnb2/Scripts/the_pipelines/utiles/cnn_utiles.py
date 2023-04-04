import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from tqdm import tqdm
import pickle
import torch.nn as nn

def image_transform(im_path):
    im = Image.open(im_path)
    im_transformer = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
    im_transformed = im_transformer(im)
    return im_transformed
    
    

class train_loader():
    def __init__(self,training_room_codes, training_labels, batch_size=50):
        self.room_count = -1
        self.image_list = []
        self.label_list = []
        self.training_room_codes = training_room_codes
        self.training_labels = training_labels
        self.batch_size = batch_size
        self.this_batch_num = 0
        self.transform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    
    def generate_dataset(self):

        for room_count,code in enumerate(self.training_room_codes):
            self.room_count+=1
            try:
                photos = os.listdir(f'LA_photos/{code}')
            except:
                continue
            for photo in photos:
                try:
                    im = Image.open(f'LA_photos/{code}/{photo}')
                    im = self.transform(im)
                    self.image_list.append(im)
                except:
                    continue
                self.label_list.append(np.log(self.training_labels[self.room_count]+1))
                self.this_batch_num+=1

            if self.this_batch_num>=self.batch_size:
                self.this_batch_num=0

                x = self.image_list
                y = self.label_list

                self.image_list = []
                self.label_list = []

                
                yield (torch.stack(x).float(),
                       torch.tensor(y).reshape(-1,1).float(),
                       float((room_count+1)/len(self.training_room_codes)))




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(64*28*28, 64)
        self.fc2 = nn.Linear(64, 10)
        self.final = nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = x.view(-1, 64*28*28)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.final(x)
        
        return x
    


