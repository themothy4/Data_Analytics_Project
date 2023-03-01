import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data\\parkinsons.data')
df.head()

features = df.loc[:, df.columns != 'status'].values[:, 1]
labels = df.loc[:, 'status'].values
