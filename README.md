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

##Developing Process
###Preprocessing 
The first step was to analyze the dataset. After verifying the non existence of missing values, the next stop was to 
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

###Creating ML Model

