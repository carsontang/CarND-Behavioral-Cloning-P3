# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2017_08_22_09_21_51_373.jpg "Center Lane"
[image2]: ./examples/center_2017_08_23_08_29_01_357.jpg "Recovering"
[image3]: ./examples/sdc_arch.png "Architecture"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model is based on the Comma.ai model for autonomous vehicles. The model includes ELU layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer.

### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use LeNet first to develop an overall feel of how well a simple convolutional neural network (CNN) would do on this track. I used the Udacity dataset, shuffled it, and split it 80% for training, 20% for validation. The validation dataset was used to combat overfitting. I trained for 10 epochs because any more often led to decreasing accuracy, and I wanted to iterate quickly on the model. I then tested out LeNet with only steering angles and images from the center camera. The CNN did well for most of the track, but it didn't make turns when it should have. I also added data so that the vehicle could learn to recover if needed. Instead of swapping to a different architecture, I incorporated images from the left and right cameras, adjusted the steering corrections, and saw that the new amount of data helped somewhat. However, there were still times when the vehicle didn't stay on the track. Finally, I swapped out LeNet for Comma.ai's autonomous vehicle model, and the vehicle was able to make the proper turns at the right time in the simulator.

### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

![Architecture][image3]

The model has a total of 3,345,009 params, making it relatively easy to transfer between my computer used for training and my laptop used for running the model in a simulator.


### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I started off with the Udacity training set. Then I recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it ever got off track. This images show what a recovery looks like starting from :

![alt text][image2]

Then I repeated this process on track two in order to get more data points.

After the collection process, I shuffled the data, and split the data into 19286 training data points and 4821 validation data points. I then preprocessed this data by normalizing it and cropping the sky above the road and the car's hood out of the images.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the often decreasing accuracy after running more than 10 epochs.  I used an Adam optimizer so that manually training the learning rate wasn't necessary.
