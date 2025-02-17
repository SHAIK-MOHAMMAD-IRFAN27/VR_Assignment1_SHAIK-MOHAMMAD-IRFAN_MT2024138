# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:23:50 2025

@author: irfan
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

center=cv2.imread("C:/Users/irfan/OneDrive/Desktop/MTECH-2ND SEM/VR/assignment 1/center.jpeg")
left=cv2.imread("C:/Users/irfan/OneDrive/Desktop/MTECH-2ND SEM/VR/assignment 1/left.jpeg")
right=cv2.imread("C:/Users/irfan/OneDrive/Desktop/MTECH-2ND SEM/VR/assignment 1/right.jpeg")

center=cv2.resize(center, (500,500))
left=cv2.resize(left, (500,500))
right=cv2.resize(right, (500,500))
def pano_creation(center,left):
    gray1 = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    
    
    sift = cv2.SIFT_create()
    
 
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    matched_img = cv2.drawMatches(center, keypoints1, left, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    
    
    height, width, channels = left.shape
    panorama = cv2.warpPerspective(center, H, (width * 2, height))
    panorama[0:height, 0:width] = left
    
    
    average_column = ((panorama[:, 500]/2.0 + panorama[:, 501] /2.0) ).astype(np.uint8)
    
    panorama[:, 500] = average_column
    
    panorama[:, 501] = average_column
    
    blend_width = 5 
    for i in range(blend_width):
        alpha = i / blend_width
        panorama[:, width - blend_width + i] = (panorama[:, width - blend_width + i] * (1 - alpha) + panorama[:, width + i] * alpha).astype(np.uint8)
        # panorama[:, width + blend_width - i] = (panorama[:, width - blend_width + i] * (alpha) + panorama[:, width + i] * (1-alpha)).astype(np.uint8)
    return panorama,matched_img

panorama,matched_1st=pano_creation(center, left)
cv2.imshow('Matched_Keypoints_center_left', matched_1st)
pano_final,matched_2nd=pano_creation(right, panorama)
cv2.imshow('Matched_Keypoints_final',matched_2nd)
cv2.imwrite('C:/Users/irfan/OneDrive/Desktop/MTECH-2ND SEM/VR/assignment 1/panorama_images/pano_output_image/Matched_Keypoints_center_left.jpg', matched_1st)
cv2.imwrite('C:/Users/irfan/OneDrive/Desktop/MTECH-2ND SEM/VR/assignment 1/panorama_images/pano_output_image/Matched_Keypoints_final.jpg', matched_2nd)

cv2.imwrite('C:/Users/irfan/OneDrive/Desktop/MTECH-2ND SEM/VR/assignment 1/panorama_images/pano_output_image/pano_image.jpg', pano_final)



cv2.imshow('center', center)
cv2.imshow('left', left)
cv2.imshow('right', right)
cv2.waitKey(0)
cv2.destroyAllWindows()
