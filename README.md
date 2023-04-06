# ML_P_falciparum_model_training
This Machine Learning model can identify Malaria specie _P. falciparum_ from plasma test image <br>
ALgorithm = ```Logistic Regression```, ```CNN```, and ```Random Forest```

**A General view of How the Logistic Regression Process was implemented** <br>

1. Importing the necessary libraries, which includes ```NumPy```, ```Pandas```, ```OS```, ```PIL```, ```TensorFlow```, and ```scikit-learn```. <br>
2. Defining the path to the folder containing the images, and the size of the images. <br>
3. Loading a pre-trained CNN model (MobileNetV2) from ```TensorFlow```, and defining a function to preprocess images and extract features using the model. <br>
4. Looping through the images in the specified folder, to extract features using the CNN model, and store the features and corresponding labels (assumed to be the first part of the image name) in separate lists. <br>
5. Converting the features and labels to ```NumPy``` arrays, and spliting the data into training and testing sets using scikit-learn's ```train_test_split()``` function. <br>
6. Training a logistic regression model on the training set using scikit-learn's ```LogisticRegression()``` function, and making predictions on the test set. <br>
7. Evaluating the model's accuracy and computing the confusion matrix using scikit-learn's ```accuracy_score()``` and ```confusion_matrix()``` functions. <br>
8. Printing the model's equation using the coefficients and intercept obtained from the logistic regression model. <br>
--------------------------------------------
Model accuracy | Model-Confusion matrix plot
---------------|----------------------------
![Screenshot 2023-03-21 7 06 13 PM](https://user-images.githubusercontent.com/114442903/227508305-a92f1a2c-23e3-48c5-981d-88657406ed8c.png) |
![Screenshot 2023-03-21 7 05 51 PM](https://user-images.githubusercontent.com/114442903/227508324-20f10af2-d902-49d3-8e2a-f2192620e23a.png)
-------------------------------------

This will print the logistic regression equation in the format of
```
y = intercept + coefficient1 * x1 + coefficient2 * x2 + ... + coefficientn * xn.
```
![plasmodium-phone-1182](https://user-images.githubusercontent.com/114442903/227060829-073600e3-df23-4535-867f-fed58d0fa0fb.jpg)
![plasmodium-phone-1178](https://user-images.githubusercontent.com/114442903/227060890-02dea42c-0a93-465a-9b3f-fa11d7b2b527.jpg)
![plasmodium-phone-1177](https://user-images.githubusercontent.com/114442903/227060911-68c8d6ce-664b-4906-a4e1-8bc6009a787d.jpg)


