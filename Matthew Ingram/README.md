# Image Classification of Food Items

Name: Matthew Ingram

Original Date: November 3, 2025

Most Recent Update: November 13, 2025

My project for this semester is to create an image classification model which can be given an input of an image of a food item (of various types), and then attempt to classify that food item.

This project uses a modified version of the "Food Image Classification Dataset" available on Kaggle at the following link:
https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset

The modifications of the dataset relate to renaming the topmost folder to "Food Dataset", and making all of the food folders/directories have uniform naming conventions, stripping out underscores and converting all names to lowercase. The exact names used for the folders can be found in the initial values of the "food_types" list.

This project is based on the "Digit_Recognition.ipynb" lab.

The final results of the original version of this project are that the model was only able to get an accuracy of 0.3750, and a val_accuracy of 0.4121. I would assume that due to the learning plateau experienced by the model, it is either suffering from overfitting, or underfitting. The model was able to correctly predict some of the food images that I got from other sources than the training data, but it also failed to correctly predict results quite often.

The newest version of the project was able to get an accuracy of 0.7941, and a val_accuracy of 0.6203. The main changes I made to improve the results were to increase the number of training images allowed, while also adding in weighted training for different classes to help offset the impacts of the increased disparity between classes with a lot of images, and classes with few images. I also increased the complexity and reduced the dropout of the CNN model, which has helped to improve the results, even if it now has a bit of overfitting. Overall, the model is still not highly accurate, but the results have been significantly improved, and I am not sure if there are any other significant ways to improve the project without reworking large portions of it.
