# Sign-labguage-recognition
This is a project developed as a final project for the course Advanced Data Analytics taught by Dr Yi Hong at University of Georgia. It was voted as **Best project in class**.

## Aim
To develop a real time sign language classification with client-server architecture where one can use the application to detect their hands and classify that hand gesture based upon the previously trained model.

## Architecture
<img src="https://github.com/ankit-vaghela30/sign-language-recognition/blob/master/example/Screenshot%202019-03-12%20at%204.30.57%20PM.png">

## Hand Detection
### Dataset
We used EgoHands dataset which is a dataset for hands in complex egocentric interactions. It contains pixel level annotations (>15000 ground truth labels) where hands are located across 4800 images. All images are captured across 48 different environments (indoor, outdoor) and activities (playing cards, chess, jenga, solving puzzles etc).

### Model
* We took an existing model (ssd_mobilenet_v1_coco) from tensorflow object detection api and retrained its final layer to detect hands.
* We saved a frozen copy of the above model
* We integrated it with tensorflow object detection API to detect the hand images.

## Gesture Classification
### Dataset
<img src="https://github.com/ankit-vaghela30/sign-language-recognition/blob/master/example/hands.png">

* We used ASL alphabet dataset from Kaggle which has 87000 images each with 200x200 pixels.
* 29 classes: 26 alphabets and 3 new classes for "space", "delete" and "nothing"

### model
* We used a pretrained VGG16 model which was trained on Imagenet dataset.
* We used transfer learning concept where we added a fully connected layer at the end, made top layers untrainable and then trained it on ASL alphabet dataset.
* We got 91% accuracy on test dataset.
* We saved and loaded this model as h5py file.
<img src="https://github.com/ankit-vaghela30/sign-language-recognition/blob/master/example/training.png">

For more detailed description, Please have a look at our project report: [link](https://github.com/ankit-vaghela30/sign-language-recognition/blob/master/ADA_paper.pdf)
