# VR_Assignment1_SHAIK-MOHAMMAD-IRFAN_MT2024138


-------------------------------------------------------------------------------------------------


## Part 1: COINS DETECTION & SEGMENTATION & COUNTING USING OPENCV 

## METHOD 1 : CANNY EDGE DETECTOR
 **1. Preprocessing the Images:**
 ###  IMG NAME : coins1.png
- Images are read and resized to 500x500 for consistency.

-	CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to enhance contrast.

-	Gaussian blur ((7,7),4) is applied to reduce noise and improve edge detection.

 **2. Edge Detection:**

-	Canny edge detection is used to detect edges from both the CLAHE-equalized and non-equalized images.
-	Canny edge detection applied on non-equivalized and cache-equivalized image to compare. 

 **3. Contour Detection and Filtering:**

-	Thresholding is applied to generate binary images. If the Intensity of image rises above 200 make all the pixels 255 and else pixels 0.

-	Contours are found and filtered based on area to remove noise and irrelevant details.

 **4. Drawing and Blending Contours:**

-	Contours are drawn on a blank mask and blended with the original image using cv.addWeighted to create segmented outputs. 0.6*img + 0.4*mask

 **5. Finding Area:**
- Kept a threshold on the contour area if the contour area >500,then only consider it as a contour since coins have pretty large area and also to reduce noise.

 **OUTPUT IMAGES:**
- img1_binary.png
- img1_canny_edges.png
- img1_direct_blur.png
- img1_segmented.png


-------------------------------------------------------------------------------------------------


## METHOD 2 : DIFFERENCE OF GAUSSIANS
###  IMG NAME : farcoins.png
-Tried this method to find the edges perfectly (also used Canny over DOG to check performance).

 **1. Difference of Gaussians (DoG):**

-	Two Gaussian blurs with different sigma values are subtracted to highlight the coin edges.
-	sigma1 = 5  sigma2 =7 kernel=(9,9)
-	dog = blur2-blur1
-	The result is normalized and thresholded for contour detection.
-	Did many times blur and histogram equivallization to reduce the noise in the DOG since DOG is very sensitive and has pretty much noise.
  
 **2. Segmentation of Each Coin:**

-	Each detected coin is segmented by creating a mask for its contour and performing a bitwise AND operation with the original image.

-	Segmented coins are displayed and saved as individual images.{segmented_coins_seperated_(1..8).png}.
  
 **3. Counting Coins:**
-	The area of each contour is calculated, and coins are counted based on a minimum area threshold.
 ### How to Run:
1. Ensure you have OpenCV :  
   ```sh
   pip install opencv-python
   pip install numpy
2. Make sure to update the input image path in the code as per directory structure.
3. Open the terminal and run:
   ```sh
   python coin_task.py

 **OUTPUT IMAGES:**
- img2_cache_equivalized_dog.png.
- img2_contours.png.
- img2_edges.png
- img2_segmeted.png
- img2_thresh.png
- segmented_coins_seperated_(1..8).png




-------------------------------------------------------------------------------------------------


## Part 2 : PANAROMA {KEY POINT DETECTION  & IMAGE STICHING}

### Project Steps:
- **Images:**
   -Resized the images to 500x500 pixels to ensure they are the same size before processing.
- **Key points:**
   - Named the images in the order from left to right and converting into gray scale
   - Using SIFT algorithm to detect key points and descriptors of the image.

- **Image Stitching:**
   - Applied FLANN algorithm to find the common keypoints in images.
   - Using Lowe's ratio test found good matches.
   - Drawn the matches between the keypoints of the center and left images and displays them.
   - Computed homographies of the images using RANSAC to deal with outliers and also aligned images for perspective transformation.
   - Estimated transformations to warp images into a common coordinate system and adjusted the output panorama size based on transformed corner coordinates.
   - Created an averaged column for smoother blending between the images at the joining area (around column 500).
   - Performed a weighted average blending along a blend_width (5 pixels) to merge the two images smoothly at the transition area.
   - Warped images using the translation matrix to align images and blended them into a single panoramic view.
   - First, created a panorama between center and left images, then created a final panorama by adding the right image to the already created panorama.


 ### How to Run:
1. Ensure you have OpenCV :  
   ```sh
   pip install opencv-python
   pip install numpy
2. Make sure to update the input image path in the code as per directory structure.
3. Open the terminal and run:
   ```sh
   python panaroma.py

Run the Python file in any IDE or terminal.


 **INPUT IMAGES :**
- center.png
- left.png
- right.png

 **OUTPUT IMAGES :**
- pano_image.png
- Matched_Keypoints_center_left.png
- Matched_Keypoints_final.png
  
