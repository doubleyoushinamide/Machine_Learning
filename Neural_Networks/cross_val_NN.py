from sklearn.model_selection import cross_val_score

scores = cross_val_score(mlp, inputs, targets, cv=5) # 5-fold cross validation
print("Cross Validation Score:", np.mean(scores))

"""
For cross-validation, you can use the cross_val_score function from sklearn package. Here's an example above.
"""
