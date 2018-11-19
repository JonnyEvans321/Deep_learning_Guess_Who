# Guess Who - Celebrities

This project was a bit of fun to try to create an AI playing the board game Guess Who. 

training.py: Creates Inception models trained to recognise each of the 40 attributes in the dataset.
game.py: Plays the game of Guess Who in the console (using the trained Inception models)

![demo_video](https://gfycat.com/IdleChubbyGuineapig)

## Game playing
Firstly, the AI is trained to recognise all 40 attributes that the celebA dataset has been labelled with, using training.py.
Once trained, 24 celebrities in the test dataset are randomly selected for the game, and the AI makes a prediction for each of the 40 attributes of these celebrities.
The game starts with the user selecting their celebrity (the AI randomly selects theirs)
The AI and the user take turns asking questions and answering them about the celebrities, until someone takes a guess.
A player wins if they guess correctly, or their opponent guesses incorrectly.

The AI's question asking 'strategy' is to choose the attribute which will remove as close to half of the remaining celebrities as possible (i.e. half of its remaining celebrities can be removed).

The downside of this game is that many of the questions that can be asked are subjective, and the AI has an average accuracy of  around 70% (I haven't trained it for long on my laptop yet). This means that there can be disagreement over whether a celebrity should have been removed from the board or not, and often results in the final guess of a celebrity being incorrect.

## Dataset
CelebA dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which is available on Kaggle: https://www.kaggle.com/jessicali9530/celeba-dataset

I got the Inception v3 model from Kaggle, but it can also be downloaded from: http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

### Overall
202,599 number of face images of various celebrities 10,177 unique identities, but names of identities are not given 40 binary attribute annotations per image 5 landmark locations

### Data Files
img_align_celeba.zip: All the face images, cropped and aligned
list_eval_partition.csv: Recommended partitioning of images into training, validation, testing sets. Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
list_bbox_celeba.csv: Bounding box information for each image. "x_1" and "y_1" represent the upper left point coordinate of bounding box. "width" and "height" represent the width and height of bounding box
list_landmarks_align_celeba.csv: Image landmarks and their respective coordinates. There are 5 landmarks: left eye, right eye, nose, left mouth, right mouth
list_attr_celeba.csv: Attribute labels for each image. There are 40 attributes. "1" represents positive while "-1" represents negative

## Credits
This script started life as Marcos's gendere recognition kaggle notebook: http://www.kaggle.com/bmarcos/image-recognition-gender-detection-inceptionv3/notebook
thanks Marcos!
