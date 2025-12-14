"""Training a Model and saving"""
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X,y=load_iris(return_X_y=True)

model=RandomForestClassifier()
model.fit(X,y)

os.makedirs('model',exist_ok=True)
joblib.dump(model,'model/model.joblib')
