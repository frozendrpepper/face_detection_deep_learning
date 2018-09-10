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

1) Resources for Object Detection Algorithm
* [Sentdex Tensorflow API Object Detection Youtube](https://www.youtube.com/watch?v=COlbP62-B-U&t=1s)
* [Edje Electronics Tensorflow API Object Detection Youtube](https://www.youtube.com/watch?v=Rgpfk6eYxJA) 

2) Resources for Face Recognition Deep Neural Network
* [Coursera Deeplearning.ai](https://www.coursera.org/specializations/deep-learning)
* [Facenet](https://github.com/davidsandberg/facenet)
* [Medium Blog with Good Explanations](https://medium.com/@vinayakvarrier/building-a-real-time-face-recognition-system-using-pre-trained-facenet-model-f1a277a06947)
* [Open Face](https://cmusatyalab.github.io/openface/)
* [Another Good Github Repo](https://github.com/ageitgey/face_recognition#face-recognition)
