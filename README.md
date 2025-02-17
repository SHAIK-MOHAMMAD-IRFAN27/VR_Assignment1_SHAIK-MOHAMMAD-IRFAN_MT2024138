# VR_Assignment1_SHAIK-MOHAMMAD-IRFAN_MT2024138


-------------------------------------------------------------------------------------------------


## Part1:COINS DETECTION &SEGMENTATION USING OPENCV 

### Project Steps:

 **1. Preprocessing the Images:**

- Images are read and resized to 500x500 for consistency.

-	CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to enhance contrast.

-	Gaussian blur is applied to reduce noise and improve edge detection.

 **2. Edge Detection:**

-	Canny edge detection is used to detect edges from both the CLAHE-equalized and non-equalized images.

 **3. Contour Detection and Filtering:**

-	Thresholding is applied to generate binary images.

-	Contours are found and filtered based on area to remove noise and irrelevant details.

 **4. Drawing and Blending Contours:**

-	Contours are drawn on a blank mask and blended with the original image using cv.addWeighted to create segmented outputs.

 **5. Difference of Gaussians (DoG):**

-	Two Gaussian blurs with different sigma values are subtracted to highlight the coin edges.

-	The result is normalized and thresholded for contour detection.

 **6. Segmentation of Each Coin:**

-	Each detected coin is segmented by creating a mask for its contour and performing a bitwise AND operation with the original image.

-	Segmented coins are displayed and saved as individual images.

 **7. Counting Coins:**

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



-------------------------------------------------------------------------------------------------


## Part 2: Create a panorama from multiple overlapping images.

### Project Steps:








Run the Python file in any IDE or terminal.
