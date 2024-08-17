import mlflow.data.pandas_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
import mlflow

### loading the dataset from data folder.
df=pd.read_csv("Data//iris.csv")

## splitting the data inot independet and dependent features.
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

### splitting the data inot training and testing data.
X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

## parameter.
max_depth=10
n_estimators=100

## set experiment tracking.
mlflow.set_experiment('iris-rf')

mlflow.autolog()
## start mlflow.
with mlflow.start_run():

    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    ### training the data.
    rf.fit(X_train,y_train)

    ## predicting the data
    y_pread=rf.predict(x_test)

    ## finding the accuracy.
    accuracy=accuracy_score(y_test,y_pread)

    # log confusion metrics.
    cm=confusion_matrix(y_test,y_pread)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True,cmap='Blues')
    plt.ylabel("Actual")
    plt.xlabel('Predicted')
    plt.title("Iris-confucion-metrics")

    ## log code.
    mlflow.log_artifact(__file__)

    ## set tag.
    mlflow.set_tag('author','hitesh')
    mlflow.set_tag('model','decision_tree')

    print('accuracy',accuracy)
