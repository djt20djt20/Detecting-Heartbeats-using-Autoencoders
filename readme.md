# Classifying Heartbeats using Autoencoders

In this project, I build an autoencoder in order to distinguish between 'normal' and 'abnormal' heartbeats. 

First, I apply a non-machine learning model to solve the problem, highlighting the fact that the model does a poor job of distinguishing between normal and abnormal heartbeats.

Next, I demonstrate that by training an auto-encoder on normal heartbeats, which learns to compress and then decompress the time series, and then running the time-series through the auto-encoder, we can significantly improve the f1-score, while also being more consistent when performing a 10-fold cross validation.

I build the auto-encoder in Pytorch, but the same technique will work easily enough in Keras and Tensorflow. 

This repo provides the code used in the following article: https://medium.com/@djt20djt20/ed45255bd4fc, where I explain the steps to build the model outlined above in detail.

## Purpose
I built this model because I'm interested in developing algorithms that detect anomolies in time-series data.

## What are auto-encoders?
Auto-encoders are models that compress an input, using an encoder, and then decompress the input, using a decoder. The point is to get back to the original input.

![image](https://user-images.githubusercontent.com/89222838/137792130-5c02e70a-f165-4748-9e3b-d619086ece11.png)

They work for detecting anomolies in time-series, because they expect a time-series with a certain profile. If you give the model a time series that it has never seen before, it fails to reconstruct it. You can then measure the 'distance' between an input, and an auto-encoded output, and use a threshold to classify the heartbeat.  

## Files
### Heart_beat_final.ipynb
The notebook where I wrangle the data, build the model, and perform my analysis
### ECG5000_TEST.arff
The test data set of ecg heartbeats
### ECG5000_TRAIN.arff
The train set of ecg heartbeats

## Libraries used
scipy <br />
pandas <br />
matplotlib <br />
sklearn <br />
torch <br />

If you have any questions about this project, feel free to contact me.
