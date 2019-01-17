# boston-housing
A machine learning project to predict median value of owner-occupied homes given some features.

Description and data are [here](https://www.kaggle.com/c/boston-housing)

Approach
========
I trained a Neural Network with 2 hidden units of 8 and 5 nodes respectively and used 
a relu activation followed by a sigmoid activation.

I trained for 20000 epochs with a learning rate of 0.001 and Adam optimizer

Results
=======
The r2 score came out to be about 0.79 with a MSE of 0.0062

Future improvements
===================
Select only the most important features that capture the essence of the relationship to the MEDV target 
and feed those as input features

