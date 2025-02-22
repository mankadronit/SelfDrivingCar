## SelfDrivingCar
This code helps in getting the steering angle of self driving car. The inspiration is taken from [End to End Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) module from NVIDIA

The End to End Learning for Self-Driving Cars research paper can be found at (https://arxiv.org/abs/1604.07316)
This repository uses convnets to predict steering angle according to the road. 

### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

### Description
An autonomous car (also known as a driverless car, self-driving car, and robotic car) is a vehicle that is capable of sensing its environment and navigating without human input. Autonomous cars combine a variety of techniques to perceive their surroundings, including radar, laser light, GPS, odometry, and computer vision. Advanced control systems interpret sensory information to identify appropriate navigation paths, as well as obstacles and relevant signage

## Self Driving Car V1 (NVIDIA Dataset based on real world)

### Dataset
Download the dataset at [here](https://github.com/SullyChen/driving-datasets) and extract into the repository folder

### Python  Implementation

1) Network Used- Convolutional Network
2) Inspiration - End to End Learning for Self-Driving Cars by Nvidia

If you face any problem, kindly raise an issue

### Procedure

1) First, run `load_data.py` which will get dataset from folder and store it in a pickle file after preprocessing.
2) Now you need to have the data, run `train.py` which will load data from pickle. After this, the training process begins.
3) For testing it on the video, run `run.py`

<img src="https://github.com/mankadronit/SelfDrivingCar/blob/master/run_final.gif">

### References:
 
 - Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba. [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
 - [Behavioral Cloning Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) 
 - This project also took a lot of inspiration from [Autopilot-Tensorflow](https://github.com/SullyChen/Autopilot-TensorFlow)




