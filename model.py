from sklearn.datasets import load_iris
import numpy as np
#import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()

x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(x_train, y_train)

# pickle the trained model
import pickle
with open('model.pkl','wb') as model_file:     #wb means write and buffering model
    pickle.dump(clf,model_file)
    