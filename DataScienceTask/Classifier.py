import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def getAccuracy():
    df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    df.columns=['SepalLength','SepalWidth','PetalLength','PetalWidth','IrisType']
    y=df.IrisType
    class_names=df.IrisType.unique()#getting the names of the labels (different types)
    x=df.loc[:,df.columns!='IrisType']#dependent variable
    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)#training data is 80%
    clf = DecisionTreeClassifier() #using the decision tree classifier
    clf.fit(X=X_train, y=Y_train)
    y_predict = clf.predict(X_test)
    conf_mat = confusion_matrix(y_predict, Y_test,class_names) #generating the confusion matrix w.r.t labels
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
    print(acc)
    print(conf_mat)
    ax= plt.subplot()
    sns.heatmap(conf_mat, annot=False, ax = ax); #annot=True to annotate cells
    
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(class_names); ax.yaxis.set_ticklabels(class_names);
    plt.show()
    return acc;