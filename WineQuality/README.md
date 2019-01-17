# wine-quality

This machine learning project is based on the wine quality dataset found [here](https://archive.ics.uci.edu/ml/datasets/wine+quality) along with its description

# Approach

I did some data preparation beforehand as detailed in the .ipynb file through normalization and removing unwanted outliers. Furthermore, I split the dataset into 2 output classes, good vs. bad wine. I then fed training data through a Neural network with 2 hidden layers and performed a binary classification using binary crossentropy loss. 

# Results

My model achieved a test accuracy of about 79.1% and F1 score of about 0.84

# Future

* Increase the accuracy by adding more hidden units along with more Dropout layers to prevent overfitting
* Convert the problem to a multiclass classification to predict wine qualities on a scale of 0-10
