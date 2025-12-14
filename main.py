"""Prediction using fast-api """
import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

model=joblib.load('model/model.joblib')

LABELS=['Setosa','Versicolor','Virginica']

class IrisModel(BaseModel):
    """Using to get list of float as iris values are float"""
    features:list[float]

@app.get('/')
def root():
    """Health checkup"""
    return {'message':'Healthy'}

@app.post('/predict')
def prediction(data:IrisModel):
    """prediction using the input value"""
    x=np.array([data.features])
    pred=int(model.predict(x)[0])
    return {
        'class_index':pred,
        'class_name':LABELS[pred]
    }
