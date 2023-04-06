y_train_pred = mlp.predict(X_train)
train_accuracy = r2_score(y_train, y_train_pred)

print("Training Accuracy:", train_accuracy)
print("Improvement from Training to Testing:", accuracy - train_accuracy)

"""
To calculate overfitting improvement, you can compare the accuracy between the training set and the testing set. If the accuracy on the training set is significantly higher than on the testing set, it means that the model is likely overfitting.
"""
