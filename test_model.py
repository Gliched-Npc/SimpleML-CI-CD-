import os

def test_model_exist():
    return os.path.exists('model/model.joblib'),"Missing Model"