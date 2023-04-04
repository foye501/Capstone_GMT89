import numpy as np
from skimage import io, color, feature, measure
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import cv2

import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import numpy as np
from tqdm import tqdm
import skimage

def get_aesthetics_one_room(img_path_list):
    features = {
        "size":[],
        "sharpness":[],
        "mean_brightness":[],
        "contrast":[],
        "dissimilarity":[],
        "homogeneity":[],
        "energy":[],
        "correlation":[],
        "num_contours":[],
        "area":[],
        "perimeter":[],
        "num_edges":[],
        "num_contours":[],
        "hist":[],
    }

    

    for img_path in img_path_list:
        try:
            photo_path = img_path

            # Load the image
            img = cv2.imread(photo_path)

            # size
            size = img.shape[0] * img.shape[1]

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Compute the Laplacian of the grayscale image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)

            # Compute the sharpness of the image
            sharpness = np.var(laplacian)

            # Extract the color histogram features
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Compute the mean brightness of the image
            mean_brightness = np.mean(gray)


            # Extract the texture features using gray-level co-occurrence matrix (GLCM)
            glcm = skimage.feature.graycomatrix(np.uint8(gray*255), [5], [0], 256, symmetric=True, normed=True)
            contrast = skimage.feature.graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0]
            energy = skimage.feature.graycoprops(glcm, 'energy')[0, 0]
            correlation = skimage.feature.graycoprops(glcm, 'correlation')[0, 0]

            # Extract the shape features using contour detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_contours = len(contours)
            # area = cv2.contourArea(contours[0])
            # perimeter = cv2.arcLength(contours[0], True)


            # Extract the edge features using Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            num_edges = np.sum(edges > 0)

            # # Extract the HOG features
            # fd, hog_image = feature.hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=False)


            features["size"].append(size)
            features["sharpness"].append(sharpness)
            features["mean_brightness"].append(mean_brightness)
            features["contrast"].append(contrast)
            features["dissimilarity"].append(dissimilarity)
            features["homogeneity"].append(homogeneity)
            features["energy"].append(energy)
            features["correlation"].append(correlation)
            features["num_contours"].append(num_contours)
            # features["area"].append(area)
            # features["perimeter"].append(perimeter)
            features["num_edges"].append(num_edges)
            features["num_contours"].append(num_contours)
            features["hist"].append(hist)
        except:
            continue

    mean_features = {}
    for k in features.keys():
        a = np.array(features[k])
        a = np.mean(a,axis=0)

        mean_features[k] = a

    return mean_features

