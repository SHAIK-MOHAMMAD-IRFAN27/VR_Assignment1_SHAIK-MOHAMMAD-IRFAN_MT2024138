# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:09:28 2025

@author: irfan
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
image=cv.imread("C:/Users/irfan/OneDrive/Desktop/MTECH-2ND SEM/VR/assignment 1/final-img.jpg")
image=cv.resize(image, (500,500))
img_copy=image.copy()

gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Apply CLAHE to the grayscale image
clahe_equalized = clahe.apply(gray)
# cv.imshow('gray',gray)
# equalized = cv.equalizeHist(gray)
direct_blur=cv.GaussianBlur(gray, (7,7),4)
blurred=cv.GaussianBlur(clahe_equalized, (7,7), 4)
# cv.imshow('equalized',blurred)
cv.imshow('direct_blur',direct_blur)


# white=np.ones((500,500,3),dtype='uint8')*255

# cv.imshow('white',white)
# edges = cv.Laplacian(blurred, cv.CV_64F)
# edges = cv.convertScaleAbs(edges)
# cv.imshow('image',edges)
# cv.imshow('blur',blurred)
non_equivalized_edges=cv.Canny(blurred,90,200,apertureSize=3, L2gradient=False)
cv.imshow('non_equivalized_edges',non_equivalized_edges)
edges=cv.Canny(direct_blur,90,200,apertureSize=3, L2gradient=False)
cv.imshow('edges',edges)

# bitwise_or = cv.bitwise_or(edges, gray)
# cv.imshow('bitwise or', bitwise_or)
mask = np.ones((500, 500, 3), dtype=np.uint8) * 255

ret , thresh = cv.threshold(direct_blur ,200 , 255 , cv.THRESH_BINARY)
contours , _ = cv.findContours(thresh, cv.RETR_TREE , cv.CHAIN_APPROX_NONE)
min_area=19000
contours = [cnt for cnt in contours if cv.contourArea(cnt) < min_area]

mask=cv.drawContours(mask, contours, -1, (0,0,255),thickness=cv.FILLED)
cv.imshow('binary', thresh)
cv.imshow('contours', mask)
  # Weight of the second image
blended_image = cv.addWeighted(image, 0.6, mask, 0.4, 0)
cv.imshow('segmented_image', blended_image)
area={}
for i in range(len(contours)):
    cnt=contours[i]
    ar=cv.contourArea(cnt)
    area[i]=ar
srt = sorted(area.items() , key = lambda x : x[1] , reverse = True)
results = np.array(srt).astype("int")
print(results)
num = np.argwhere(results[: , 1] > 500).shape[0]
print("Number of coins in 1st image is " , num )

# params = cv.SimpleBlobDetector_Params()
# params.filterByCircularity = True
# params.minCircularity = 0.8  # High circularity for coin-like shapes

# detector = cv.SimpleBlobDetector_create(params)
# keypoints = detector.detect(thresh)

# if keypoints:
#     most_circular = max(keypoints, key=lambda k: k.size)
#     print(f'Most circular coin detected with size: {most_circular.size}')
#     image_with_keypoint = cv.drawKeypoints(image, [most_circular], np.array([]), (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     cv.imshow('Most Circular Coin', image_with_keypoint)
# else:
#     print('No circular coins detected')


image2=cv.imread("C:/Users/irfan/OneDrive/Desktop/MTECH-2ND SEM/VR/assignment 1/farcoins.jpg")
image2=cv.resize(image2, (500,500))
origi2=image2.copy()
gray2=cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

sigma1 = 5  
sigma2 =7


blur1 = cv.GaussianBlur(gray2, (9,9), sigma1)
blur2 = cv.GaussianBlur(gray2, (9, 9), sigma2)


dog_image = blur1 - blur2


dog_image = cv.normalize(dog_image, None, 0, 255, cv.NORM_MINMAX)
dog_image = np.uint8(dog_image)

cv.imshow('dog', dog_image)
blurred_image = cv.GaussianBlur(dog_image, (7, 7), 5)
cv.imshow('dog_blur', blurred_image)

# clahenew = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))

# # Apply CLAHE to the grayscale image
# clahe_equalized = clahenew.apply(blurred_image)
# _, thresholded_image = cv.threshold(blurred_image, 50, 255, cv.THRESH_BINARY)
thresholded_image = cv.Canny(blurred_image, 100, 200)

contours , _ = cv.findContours(thresholded_image, cv.RETR_TREE , cv.CHAIN_APPROX_NONE)
mask = np.zeros_like(image)
min_area=70.3
contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]
image2=cv.drawContours(mask, contours, -1, (255,255,255),thickness=1)

cv.imshow('Difference of Gaussians', thresholded_image)
image2=cv.resize(image2, (500,500))
cv.imshow('contours2', image2)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
image2=cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
# Apply CLAHE to the grayscale image
image2=cv.GaussianBlur(image2, (9,9), 0)
clahe_equalized = clahe.apply(image2)
cv.imshow('clahe_equalized_dog', clahe_equalized)
ret , thresh = cv.threshold(clahe_equalized ,45 , 255 , cv.THRESH_BINARY)
cv.imshow('threshdog', thresh)
contours , _ = cv.findContours(thresh, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_NONE)
mask = np.ones((500, 500,3), dtype=np.uint8) * 255
contours= [cnt for cnt in contours if cv.contourArea(cnt) > 4000]
mask=cv.drawContours(mask, contours, -1, (0,255,0),thickness=cv.FILLED)
cv.imshow('DOG_contours',mask)
hell = cv.addWeighted(origi2, 0.7, mask, 0.3, 0)

cv.imshow('blended', hell)

for idx, contour in enumerate(contours):
    mask = np.zeros_like(image2)
    cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)
    segmented_coin = cv.bitwise_and(origi2, origi2, mask=mask)
    
    plt.figure()
    plt.title(f'Segmented Coin {idx+1}')
    plt.axis('off')
    plt.imshow(segmented_coin, cmap='gray')
plt.show()




for i in range(len(contours)):
    cnt = contours[i]
    ar = cv.contourArea(cnt)
    area[i] = ar
srt = sorted(area.items() , key = lambda x : x[1] , reverse = True)
results = np.array(srt).astype("int")
print(results)
num = np.argwhere(results[: , 1] > 500).shape[0]
print("Number of coins in 2nd image is " , num )

cv.waitKey(0)
cv.destroyAllWindows()