
import matplotlib.pyplot as plt

import torch
import torchvision
import numpy as np
from skimage import io, color, feature, measure
import pandas as pd
import pickle
import os

import cv2

from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import numpy as np
from tqdm import tqdm
import skimage

import matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO

def get_classes():
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    return classes

def parse_YOLO_result(res):
    attrs = {}
    for index,a_res in enumerate(res):
        a_res_boxes = a_res.boxes.boxes.detach().numpy() ### the boxes, confidence, and label
        a_res_boxes_loc = a_res_boxes[:,:4]
        a_res_conf = a_res_boxes[:,-2]
        a_res_labels = [a_res.names[i] for i in a_res_boxes[:,-1]]

        ### calc points for rule of third
        r3_point1 = (1*(a_res.orig_shape[0]/3), 1*(a_res.orig_shape[1]/3))
        r3_point2 = (2*(a_res.orig_shape[0]/3), 1*(a_res.orig_shape[1]/3))
        r3_point3 = (1*(a_res.orig_shape[0]/3), 2*(a_res.orig_shape[1]/3))
        r3_point4 = (2*(a_res.orig_shape[0]/3), 2*(a_res.orig_shape[1]/3))

        ### calc location of each object
        center_list = []
        min_dist_r3_list = []
        min_dist_r3_weighted_by_size_list = []
        obj_size_list = []
        for i in range(len(a_res_labels)):
            p1 = (a_res_boxes_loc[i,0],a_res.orig_shape[1]-a_res_boxes_loc[i,1])
            p2 = (a_res_boxes_loc[i,2],a_res.orig_shape[1]-a_res_boxes_loc[i,1])
            p3 = (a_res_boxes_loc[i,0],a_res.orig_shape[1]-a_res_boxes_loc[i,3])
            p4 = (a_res_boxes_loc[i,2],a_res.orig_shape[1]-a_res_boxes_loc[i,3])
            center = ((p1[0]+p2[0]+p3[0]+p4[0])/4, (p1[1]+p2[1]+p3[1]+p4[1])/4)
            obj_size = np.abs(p2[0]-p1[0]) * np.abs(p2[1] - p3[1])
            obj_size_list.append(obj_size)
            center_list.append(center)

            ### calc distance of obj center to the closed "third point"
            dists = [((center[0] - r3_point1[0])**2 + (center[1] - r3_point1[1])**2)**(1/2),
                    ((center[0] - r3_point2[0])**2 + (center[1] - r3_point2[1])**2)**(1/2),
                    ((center[0] - r3_point3[0])**2 + (center[1] - r3_point3[1])**2)**(1/2),
                    ((center[0] - r3_point4[0])**2 + (center[1] - r3_point4[1])**2)**(1/2)
                    ]

            min_dist_r3 = np.nanmin(dists)/(a_res.orig_shape[0]*a_res.orig_shape[1])
            min_dist_r3_list.append(min_dist_r3)

        ### mean dist of all obj
        mean_min_dist_r3 = np.nanmean(min_dist_r3_list)
        

        ### weighted mean dist of all obj. Larger obj, larger effect of rule of third
        mean_weighted_min_dist_r3 = [a*(b/(a_res.orig_shape[0] * a_res.orig_shape[1])) for a,b in zip(min_dist_r3_list,obj_size_list)]
        mean_weighted_min_dist_r3 = np.nanmean(mean_weighted_min_dist_r3)
        attrs[index] = {
            'boxes_locs':a_res_boxes_loc,
            'ori_img_shape':a_res.orig_shape,
            'obj_size':obj_size_list,
            'conf':a_res_conf,
            'labels':a_res_labels,
            'center_of_obj':center_list,
            'min_dist_r3':min_dist_r3_list,
            'mean_min_dist_r3':mean_min_dist_r3,
            'mean_weighted_min_dist_r3':mean_weighted_min_dist_r3
        }
    return attrs


def get_listing_level_attr(attrs,classes):
    counter = {it:0 for it in classes}
    mean_mean_min_dist_r3 = 0
    mean_mean_weighted_min_dist_r3 = 0
    for k in attrs:
        for item in attrs[k]['labels']:
            counter[item]+=1
        mean_min_dist_r3 = attrs[k]['mean_min_dist_r3']
        mean_mean_min_dist_r3+=mean_min_dist_r3
        mean_weighted_min_dist_r3=attrs[k]['mean_weighted_min_dist_r3']
        mean_mean_weighted_min_dist_r3+=mean_weighted_min_dist_r3

    listing_level_attr = counter.copy()
    listing_level_attr['mean_mean_min_dist_r3'] = mean_mean_min_dist_r3/len(attrs)
    listing_level_attr['mean_mean_weighted_min_dist_r3'] = mean_mean_weighted_min_dist_r3/len(attrs)
    return listing_level_attr
        


