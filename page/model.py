import streamlit as st
import pandas as pd
import pickle

import matplotlib.pyplot as plt
Data = pd.read_csv("diabetes_data_upload.csv")

Data.head()


Data.shape


Data.describe()

"""## **Data Preprocessing**

**Check the Missing Values**
"""

Data.isnull().sum()

"""Dataset does not contain any missing value, if dataset has a null value so need to drop or fill the null values

##### **Feature Engineering**

Feature Engineering is the process to create feature/extract the feature from existing features by domain knowledge to increase the performance of machine learning model.

**Drop the attribute which is not useful**

Now, again see the dataset with final attributes
"""

Data.head()

"""again check the shape of the attributes"""

Data.shape

"""**Now, our dataset has a 1000 rows and 13 columns**

**Check the type of the attributes**
"""

Data.dtypes

"""Algorithm will not work on string type of data, so strings may need to be transformed to floating point or integer values.

#####**Encoding**

Encoding is a technique of converting categorical variables into numerical values so that it could be easily fitted to a machine learning model. There are 2 types of encoding

1.Label Encoding

2.One-hot Encoding

**here we are using LabelEnocoding for the transformation**
"""

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Data['Gender']= labelencoder.fit_transform(Data['Gender'])
Data['Polyuria']= labelencoder.fit_transform(Data['Polyuria'])
Data['Polydipsia']= labelencoder.fit_transform(Data['Polydipsia'])
Data['sudden weight loss']= labelencoder.fit_transform(Data['sudden weight loss'])
Data['weakness']= labelencoder.fit_transform(Data['weakness'])
Data['Polyphagia'] = labelencoder.fit_transform(Data['Polyphagia'])
Data['Genital thrush']= labelencoder.fit_transform(Data['Genital thrush'])
Data['visual blurring']= labelencoder.fit_transform(Data['visual blurring'])
Data['Itching']= labelencoder.fit_transform(Data['Itching'])
Data['Irritability']= labelencoder.fit_transform(Data['Irritability'])
Data['delayed healing']= labelencoder.fit_transform(Data['delayed healing'])
Data['partial paresis']= labelencoder.fit_transform(Data['partial paresis'])
Data['muscle stiffness']= labelencoder.fit_transform(Data['muscle stiffness'])
Data['Alopecia']= labelencoder.fit_transform(Data['Alopecia'])
Data['Obesity']= labelencoder.fit_transform(Data['Obesity'])
Data['class']= labelencoder.fit_transform(Data['class'])

"""Now we again check the data type of attributes"""

Data.dtypes

"""now, data does not conatin any object type of attribute

**Independent Varibales-** A Varible whose value does not changes by the effect of other variables and is used to manipualate the dependent variable.it often denoted as X .

**Dependent Variables-** A variable whose value change when there is any manipulation in the values of independent variables. it is often denoted as Y

**In the Dataset the class is a dependent varibale and others are independent variables.so we sotre the dependent and independent varibales into different variables**
"""

X = Data.drop(columns=['class'])
Y = Data['class']

"""Check the shape of the X and Y Variable"""

print(X.shape)  # X has a 1000 rows and 12 columns
print(Y.shape)  # Y has a 1000 rows and 1 column

"""## **Split the Dataset**

To train any machine learning model irrespective what type of dataset is being used you have to split the dataset into training data and testing data. train_test_split() function is used to split the dataset.

So, let us look into how it can be done?

**Here I have used the ‘train_test_split’ to split .**

**test_size:**This is set 0.2 thus defining the test size will be 20% of the dataset.

The data in 80:20 ratio i.e. 80% of the data will be used for training the model while 20% will be used for testing the model that is built out of it.

**random_state:** it controls the shuffling applied to the data before applying the split. Setting random_state a fixed value will guarantee that the same sequence of random numbers are generated each time you run the code.

**Split the dataset into training and testing**
"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2, random_state=1000)

"""**Let's chekc the shape of the x_train, x_test, y_train , y_test**"""

print("shape of the x_train data",x_train.shape)
print("shape of the y_train data",y_train.shape)
print("shape of the x_test data",x_test.shape)
print("shape of the y_test data",y_test.shape)

"""**here, we print the training data of independent attributes**"""

x_training = pd.DataFrame(x_train)
x_training.head()

"""**here, we print the training data of dependent attribute**"""

y_training = pd.DataFrame(y_train)
y_training.head()

"""**here, we print the testing data of independent attributes**"""

x_testing = pd.DataFrame(x_test)
x_testing.head()

"""**here, we print the testing data of dependent attribute**"""

y_testing = pd.DataFrame(y_test)
y_testing.head()

"""## **Model Creation**

**Create a model of Support Vector Machine Classification Algorithm and fitting the algorithm to the trainng set**

import the svc from the sklearn
"""

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)

"""**now, our model is ready for the prediction**

## **Model Prediction**

Model Prediction is a mathematical process used to predict future events or outcomes by analyzing patterns in a given set of input data.

**predicting the test set results**

Prediction perform on testing data of independent variable(x_test)
"""

import time
start = time.time()
y_predicted = model.predict(x_test)
end = time.time()
eval_time = end-start
print("Starting time of the prediction : ",start)
print("Ending time of the predcition : ",end)
print("time in seconds",eval_time)

"""**print the rsult of the prediction**"""

y_predicted

"""## **Model Evaluation**

Model Evaluation is the process through which we quantify the quality of a system’s predictions. To do this, we measure the newly trained model performance on a new and independent dataset. This model will compare label data with it’s own predictions.

**Accuracy Score** - The simplest intuitive performance measure is accuracy, which is just the ratio of accurately predicted observations to total observations.

**confusion matrix**- It is a summary of classification problem prediction outcomes. The number of right and incorrect predictions is broken down by class and summarised with count values.

**classification report** - It is used to evaluate the accuracy of a classification algorithm's predictions. How many of your predictions were correct and how many were incorrect.True Positives, False Negatives, and False Negatives that are used to predict the metrics of a classification report.

**Precision** – Accuracy of positive predictions.

Precision = TP/(TP + FP)

**Recall**- Fraction of positives that were correctly identified.

Recall = TP/(TP+FN)

**F1 score** - It is a weighted harmonic mean of precision and recall

F1 Score = 2*(Recall * Precision) / (Recall + Precision)
"""

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("Accuracy Score is : ", accuracy_score(y_predicted,y_test))
print('\n')
print("confusion_matrix : \n ", confusion_matrix(y_predicted,y_test))
print('\n')
print("classification_report \n  : ", classification_report(y_predicted,y_test))

"""**We got 72 % Accuracy, SVC Basic model give a low accuracy. We can use hyperparameter tuning with the basic model and try to improve the accuracy**

### **Hyperparameter Tuning**
SVM has some hyper-parameters (like which C or gamma values to use), and determining the best hyper-parameter is a difficult task. However, it can be discovered simply by experimenting with various parameters to find which ones perform best. The fundamental concept is to generate a grid of hyper-parameters and then experiment with all of their possible combinations.

**We don't have to do it manually because Scikit-learn provides GridSearchCV built-in.**

**Create a dictionary of parameters**

**C**- C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly.
Increasing C values may lead to overfitting the training data.

**Kernal** - kernel parameters selects the type of hyperplane used to separate the data. Using ‘linear’ will use a linear hyperplane (a line in the case of 2D data). ‘rbf’ uses a non linear hyper-plane.
"""

C =[1,2,3]
kernel=['rbf','linear']
param_grid = dict(C=C, kernel=kernel)

"""**Use GridsearchCV**

GridSearchCV takes a dictionary of parameters that can be used to train a model. The grid of parameters is modelled after a dictionary, with the keys representing the parameters and the values being the settings to be tested.

**Create a model using GridSerachCV**

**param_grid** - The parameter grid to explore, as a dictionary mapping estimator parameters to sequences of allowed values.
"""

from sklearn.model_selection import GridSearchCV
grid_model = GridSearchCV(estimator=model, param_grid = param_grid)

start = time.time()
result = grid_model.fit(x_train,y_train)  # Fit the model
end = time.time()
eval_time = end-start
print("Starting time of the prediction : ",start)
print("Ending time of the predcition : ",end)
print("time in seconds",eval_time)

"""**Predict the model** """

start = time.time()
grid_predictions =grid_model.predict(x_test)
end = time.time()
eval_time = end-start
print("Starting time of the prediction : ",start)
print("Ending time of the predcition : ",end)
print("time in seconds",eval_time)

"""**print the classification report**"""

print(classification_report(y_test, grid_predictions))

"""**We can aslo check the accuracy using best_score_**"""

Accuracy = result.best_score_
print("Accuracy of the model after applying Hyperparameter Tuning: ",Accuracy)

"""**We got 96% Accuracy after Hyperparameter Tuning**

 Hyperparamerter increase the accuracy score,now there are 96% chances that people have diabeties

You can inspect the best parameters found by GridSearchCV in the best_params_ attribute

**print best parameter after tuning**
"""

result.best_params_
