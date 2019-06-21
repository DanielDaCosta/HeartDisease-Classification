# CyberLabs_Internship_Prediction

Prediction model built based on Mission_Prediction_Dataset.csv data that can predict the presence of a disease in the patient. 
Model created as part of CYBERLABS Mission: Disease classification, Internship application.


## Technologies
This is created using:
* Numpy: V1.16.4
* Pandas: V0.24.2
* Seaborn: V0.9.0
* Keras: V2.2.4
* Keras-Applications: V1.0.8
* Keras-Preprocessing: V1.1.0
* Scikit-learn: V0.21..2
* Scipy: V1.3.0

## Developing Process
### Preprocessing 
The first step was to analyze the dataset. After verifying the non existence of missing values, the next step was to 
understand how the features were distributed. As some of the variables had similar distributions the dataset was reorganized
so that these features would be side by side.

The output histogram showed that the data was well balanced: 54.45% for sick people and 45.54% for non sick.

In order to analyze each feature contribution to the output, the correlation matrix was plotted. Every feature was correlated
to the output and there wasn't any multicollinearity problem. On the other hand, outliers were detected through a scatter plot
of the data. For a more precise detection, Z-score was used in order to find and remove them.

Since the features have different scales and in order to make the training less sensitive to the scales of variables,
the dataset was normalized. For those with gaussian distributions the best way to rescale them was using standardization.
The binary data don't need normalization so its values weren't changed. For the rest of the data, they were normalized using a 
MinMaxScale function.

### Creating ML Model

For this problem a Neural Network classification model was used. This model was chosen because I already had worked with this kind
of Machine Learning Algorithm. Because of the complexity of the data, with multiples variables, a Multilayer Perceptron 
with two hidden layers was the best configuration found. 

For the activation functions, the inputs range were basically between -1 and 1 (some values were higher due to standardization)
and for that the 'tanh' function was chosen. Since it was a classification problem a 'sigmoid' was used in output layer.

To prevent overfitting methods such as: Droup out and L1 and L2 Regularization, were used, but the result wasn't good 
enough, due to the small neural network configuration (only 7 neurons in each layer). The best result was obtained using 
the early stopping method. The model efficiency was measured using accuracy metric.

30% of the data was used to test the model, 10% as validation set and the rest was used for training.

## Output.txt file

This file contains the prediction results 

## Excecution

The python file 'teste1.py' just have to be executed with all the needed libraries. The output contains the following 
figures:
* Distributions graphs of each feature
* Histogram of the output data
* Correlation Matrix 
* Loss function of training and test sets