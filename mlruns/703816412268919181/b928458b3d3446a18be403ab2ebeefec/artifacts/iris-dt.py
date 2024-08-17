import mlflow.data.pandas_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
max_depth=5

mlflow.set_experiment('iris-dt')
## start mlflow.
with mlflow.start_run():

    dt=DecisionTreeClassifier(max_depth=max_depth)
    ### training the data.
    dt.fit(X_train,y_train)

    ## predicting the data
    y_pread=dt.predict(x_test)

    ## finding the accuracy.
    accuracy=accuracy_score(y_test,y_pread)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)


    # log confusion metrics.
    cm=confusion_matrix(y_test,y_pread)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True,cmap='Blues')
    plt.ylabel("Actual")
    plt.xlabel('Predicted')
    plt.title("Iris-confucion-metrics")


    ## save confusion metrics.
    plt.savefig('confusion-metrics.png')

    ## log confusion-metrics.
    mlflow.log_artifact('confusion-metrics.png')

    ## log code.
    mlflow.log_artifact(__file__)

    ## log model.
    mlflow.sklearn.log_model(dt,'decision_tree')

    ## set tag.
    mlflow.set_tag('author','hitesh')
    mlflow.set_tag('model','decision_tree')

    ### logging dataset.
    train_df=X_train
    train_df['variety']=y_train

    test_df=x_test
    test_df['variety']=y_test

    ## converting dataset into mlflow format.
    train_df=mlflow.data.pandas_dataset(train_df)
    test_df=mlflow.data.pandas_dataset(test_df)

    ## log dataset.
    mlflow.log_input(train_df,'train')
    mlflow.log_input(test_df,'validation')

    print('accuracy',accuracy)
