"""Testing if model is existing in the path /model"""
import os

def test_model_exist():
    """Testing model exists"""
    return os.path.exists('model/model.joblib'),"Missing Model"
