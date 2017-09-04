# Ensemble Stacking Class for SKLearn

This repo contains a python code for building stacking ensembles with SKLearn models.  The script can grid search models and find best permuations for building the layers of a stacking ensemble.

Stacking has been shown to give boosts to model performance over single off the shelf models alone.
This class supports classification as well as regression problems, and all loss functions and scoring metrics used in the SKLearn library.


Future features will hopefully include:
1. support for models built in other packages such as XGBoost, LightGBM, Keras and TensorFlow
2. support for building multi-layer stacking ensembles
