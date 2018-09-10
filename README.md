![alt text](https://raw.githubusercontent.com/cmusatyalab/openface/master/images/summary.jpg)
(Source: OpenFace Github Repository)

# Face Recognition Using Tensorflow Object Detection API and Coursera Deeplearning.ai Face Recognition Neural Network

The goal is to construct a pipeline of object detection algorithm for face detection and deep neural network for full
function prototype of face recognition model.


## Note about working environment

* In order for the main jupyter notebook to execute properly, it needs to run in a directory where Tensorflow's Object Detection API can execute.
  For further information on how to set up an environment for running the object dection algorithm, refer to either Sentdex or 
  Edje Electronic's tutorial and download necessary files. 
  
  I tried to upload all the necessary files so anyone who wants to try their own project can simply clone the git and go with it. But
  the file sizes were simply too big and it was a major headache to upload everything without Git giving some funky errors

## Pipeline Summary

1) Face Detection using Object Detection
 * I've used Tensorflow API's built in inception model. This model had better accuracy than the mobilenet model which is the lightest
   model that is provided.
   
2) Face Recognition Deep Nueral Network
 * The basic idea of how face recognition NN works can be seen as follows:
 ![alt text](https://i.ytimg.com/vi/6jfw8MuKwpI/maxresdefault.jpg)
 (Source: Coursera Deeplearning.ai lecture on Siamese Network)
 * The idea is to utilize a pre-trained model that has been specifically trained to recognize different faces. The output of
 the DL architecture a 128 dimension vector that represents an encoding of a face image. Then distance metric is used to compare
 different faces and if the distance falls within a certain criteria, we have matching faces.


## Result

You can see in the main Jupyter Notebook file that the model does an excellent job at detecting all faces in the images
and also recognize my face apart from other faces
 
## Suggestions

I have implemented this on my personal hardware (7th Gen core i7 and GTX 1060 6GB) with a built in camera. The frame rate isn't
the greatest but the model can run at an acceptable frame rate and detect faces (and recognize my own face). OpenCV has built in
methods that can automatcally detect any connected camera to your hardware, input a live stream video and return each frame
as individual image represented as array. 

## Useful References

* [Kaggle Competition Description](https://www.kaggle.com/c/dogs-vs-cats)
* [Excellent Youtube tutorial on Keras](https://www.youtube.com/watch?v=LhEMXbjGV_4&t=378s) - Excellent series of MLP and CNN tutorial in Keras.
* [Simple CNN structure](https://pythonprogramming.net/tflearn-machine-learning-tutorial/) - Provided simple 28 x 28 CNN structure
* [Blog on Transfer Learning](https://medium.com/@galen.ballew/transferlearning-b65772083b47) - This blog provides a good overview of XGBoost approach to solve the problem. It also provided a good tip on how the ranking of the dog breed can help improve the accuracy.
* [Paper on how Xception model was conceived](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf) - Paper on Xception
