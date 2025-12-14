import joblib
import numpy as np 

from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

model=joblib.load('model/model.joblib')

LABELS=['Setosa','Versicolor','Virginica']

class IrisModel(BaseModel):
    features:list[float]
    
@app.get('/')
def root():
    return {'message':'Healthy'}

@app.post('/predict')
def prediction(data:IrisModel):
    x=np.array([data.features])
    pred=int(model.predict(x)[0])
    return {
        'class_index':pred,
        'class_name':LABELS[pred]
    }
    