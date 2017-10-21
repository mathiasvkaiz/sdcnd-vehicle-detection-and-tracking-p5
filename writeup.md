##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: car_non_car.png
[image2]: hog_examples.png
[image3]: code_classifier.png
[image4]: windows.png
[image5]: code_windows.png
[image6]: windows_result.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it and here is a link to my [project code](https://github.com/mathiasvkaiz/sdcnd-vehicle-detection-and-tracking-p5/blob/master/Vehicle_Detection_And_Tracking.ipynb)

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first, third (in method `get_hog_features`) and sicth code cells of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and came up to following parameters.

- color_space: YCrCb seems to be far better then the other color spaces, as RGB was the worst.
- orient: According to a pedestrian detection whitepaper the 9 seems to give the best results
- pix_per_cell: Defines the cell size for computing each gradient histogram. So i needed to choose between good results and good performance where 8 has fitted best for me
- cell_per_block: Defines the local histogram normalization in each cell. It was good combination with 8 pix_per_cell to have 2 cell_per_block
- hog_channel: I used all hog channels to get best results as we have a combination of individual channels


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM (placed in code cell seven) based on skit.learn training on all given cars and non-cars `cars` and `non_cars` lists and using the helper function `extract_features()`. in this helper function i iterate over all images. First i do a copy of each image. After that i applied color conversion `COLOR_RGB2YCrCb` as the original image is in RGB. After that i applied `spatial_bin()` to convert the image into a feature vector based on gradient. After that i applied `color_hist()` on the feature_vector image to compute color histograms for each channel separately and combine them into one feature vector.
The two different feature vectors (gradient and color) are combined to one big feature vector. This vector is used to train the SVM classifier. This is all done on a random train set (80% of the data) and random test set (20% of the data). Also i used `StandardScaler()` to normalize the data. This is important for my combined feature vectors so that color or gradient does not overrule the other feature vector part in case we have different value ranges.

Following you can see the code snippet where all this happens:

![alt text][image3]


Based on the given parameters and combined feature extract tools i got a score on  0.9901 what seems to me that this is quite a good result. As it takes quite a long time to train i save the classifier in a pickel file (code cell eight). 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use the sliding window technique (placed on code cell nine) with a 96 size for the windows. This size had the best results in finding the cars compared to window size 32, 64 and 128. Someimes i found too many false positive, sometimes i found nearly no car. But with 96 size i sound every car on my test image set. 

![alt text][image4]

I iterated over the test images in folder `test_images` did a copy of the original image and applied the `slide_window()` helper function to create the windows based on given parameters of start and stop dimensions in x and y directions, window size and allowed overlap of the windows. After that i applied the `each_windows()` function. This function applies the classifier on the given windows and extract single features out of it `single_img_features()`. This function is similar to the `get_hog_features()` functions only that it is applied to one image. The positiove windows, meaning found a car based on the classifiers prediction, are stored in obe list. This `hot_windows` list is used for drawing boxes on using the helper function `draw_boxes()`

Following image show the code snippet for the pipeline.

![alt text][image5]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:



### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:


### Here the resulting bounding boxes are drawn onto the last frame in the series:




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

