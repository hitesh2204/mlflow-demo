import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

## reading data.
df=pd.read_csv("Data//diabetes.csv")

### splitting data into independent and dependent features.
X=df.drop('Outcome',axis=1)
y=df['Outcome']

## splitting data into training and testing files.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=41)

### params
param={
    'max_depth':[None,10,8,15,7],
    'n_estimators':[50,60,80,100]
}

### creating random forst model object.
rf=RandomForestClassifier(random_state=40)


### tunning hyper-parametr using gridserachcv.
grid_search=GridSearchCV(estimator=rf,param_grid=param,cv=5,n_jobs=-1,verbose=2)

mlflow.set_experiment('rf-diabetes-hp')
with mlflow.start_run(run_name='parent_run') as parent:
    grid_search.fit(X_train,y_train)

    ## log all combinations.
    for i in range(len(grid_search.cv_results_['params'])):
        print(i)
        with mlflow.start_run(nested=True) as child:

            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric('accuracy',grid_search.cv_results_['mean_test_score'][i])

    ## displaying the best parameter.
    best_param=grid_search.best_params_
    best_score=grid_search.best_score_

    ### log param
    mlflow.log_params(best_param)
    
    ## log metrics
    mlflow.log_metric('accuracy',best_score)

    ## log code.
    #mlflow.log_artifacts(__file__)

    ### data
    train_df=X_train
    train_df['Outcome']=y_train

    test_df=X_test
    test_df['Outcome']=y_test

    ### converting dataset into mlflow format.
    train_df=mlflow.data.from_pandas(train_df)    
    test_df=mlflow.data.from_pandas(test_df)

    ### log dataset.
    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"validation")

    ## log model.
    mlflow.sklearn.log_model(grid_search.best_estimator_,'rf')

    mlflow.set_tag('author','Hitesh')

    print(best_param)
    print(best_score)