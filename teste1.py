import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


disease_data = pd.read_csv("/Users/danieldacosta/PycharmProjects/CyberLab/Mission_Prediction_Dataset.csv")

# Analysing Data
print(disease_data.shape)
print(disease_data.describe())

# Analyzing Output
data_size = disease_data.shape[0]
sick = disease_data[disease_data['column14'] == 1]
not_sick = disease_data[disease_data['column14'] == 0]
x = len(sick)/data_size
y = len(not_sick)/data_size
print('Sick :', x*100, '%')
print('Not sick :', y*100, '%')

plt.figure(1)
labels = ['Sick','Not Sick']
grafico = pd.value_counts(disease_data['column14'], sort=True)
grafico.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), labels)
plt.xlabel("Class")
plt.ylabel("Frequency")


### Preprocessing data

# Checking for Missing Values
print(disease_data.isnull().any().sum())

#sns.pairplot(disease_data)
plt.figure(2)
sns.heatmap(disease_data.corr(), annot=True) #Correlation for the dataset


# Checking and Removing Outliers
from scipy import stats
z = np.abs(stats.zscore(disease_data))
threshold = 3
disease_data = disease_data[(z < 3).all(axis=1)]

features = disease_data.iloc[:, 0:13].columns
#for i in range(disease_data.shape[1] - 1):
 #   plt.figure(i+3)
  #  column = [disease_data[features[i]].values]
   # sns.distplot(column)
    #plt.xlabel('Column ' + str(i+1))
#plt.show()
#disease_data.hist()
#from pandas.plotting import scatter_matrix
#scatter_matrix(disease_data.iloc[1:, :]) #Correlacao entre os dados

# Normalizing Data

features = ['column1', 'column4', 'column5', 'column8', 'column3', 'column7', 'column10', 'column11', 'column12', 'column13', 'column2', 'column6','column9','column14']
disease_data = disease_data[features]
print(disease_data.describe())

from sklearn.preprocessing import StandardScaler
gaussian_features = disease_data.iloc[:, 0:4]
sc = StandardScaler()
gaussian_featuresX = sc.fit_transform(gaussian_features)

from sklearn.preprocessing import MinMaxScaler
categorical_features = disease_data.iloc[:, 4:10]
sc = MinMaxScaler(feature_range=(0, 1))
categorical_featuresX = sc.fit_transform(categorical_features)


X = np.concatenate((gaussian_featuresX, categorical_featuresX, disease_data.iloc[:,10:13]), axis=1)
Y = disease_data.iloc[:, 13]
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


### Creting Neural Network

from keras import Sequential
from keras.layers import Dense
from keras.regularizers import l1
from keras.layers import Dropout

classifier = Sequential()
# First Hidden Layer
classifier.add(Dense(7, activation='tanh', kernel_initializer='random_normal', input_dim=13))
# Second  Hidden Layer
classifier.add(Dense(7, activation='tanh', kernel_initializer='random_normal'))
# Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

# Compiling the neural network
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

# Fitting the data to the training dataset
history = classifier.fit(X_train, y_train, batch_size=30, epochs=200, validation_split=0.1, callbacks=[es])
eval_model = classifier.evaluate(X_train, y_train)
print(eval_model)

y_pred=classifier.predict(X_test)
y_pred =(y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac_str = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(ac_str)

# Plot training history
plt.figure(3)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()