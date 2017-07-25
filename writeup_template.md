**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.png "Centered Driving"
[image2]: ./examples/left.png "Left Side Recovery"
[image3]: ./examples/right.png "Right Side Recovery"
[image4]: ./examples/camera.jpg "Normal Image"
[image5]: ./examples/flip.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 66-68) 

The model includes RELU layers with the convolutional layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 62). 

####2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 71). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 16-17). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to adjust the NVIDIA architecture.

My first step was to begin with the NVIDIA convolution neural network model similar. I thought this model might be appropriate because it was created by the autonomous vehicle team at NVIDIA which was made for real-life cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a somewhat higher mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include a dropout layer.

Then I added a maxpooling layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like the turn after the bridge which has no lane line. It also drove off on the turn after that. To improve the driving behavior in these cases, I trained the network to make sharp turns when it goes too close to red and white turn markers as well as the dirt road. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 61-72) consisted of a convolution neural network with the following layers and layer sizes:

Layer (type)                     Output Shape          Param #     Connected to
=======================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 5, 37, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       dropout_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 3, 35, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       maxpooling2d_1[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
=======================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. One of the laps has the car driving in the opposite direction of the road. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from:

The left side

![alt text][image2]

The right side

![alt text][image3]

To augment the data sat, I also flipped images thinking that this would create symmetry so that the car does not favor either side. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

After testing the vehicle with only this data. It still went offroad, so I focused on areas most likely to go offroad (near the bridge and sharp turns) and recorded sharper turns when it got too close to going offroad.

After the collection process, I had 27730 number of data points. I then preprocessed this data by normalizing the data and cropping parts of the top and bottom of the images.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 4 or 5 as evidenced by numerous attempts at training the model. I used 5 epochs just to be sure. I used an adam optimizer so that manually training the learning rate wasn't necessary.
