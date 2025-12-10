import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import csv


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv('second_features.csv', low_memory=False)

# Remove rows with missing color percentage data
df = df.dropna(subset=['red_pct', 'green_pct', 'blue_pct', 'yellow_pct', 'black_pct', 'white_pct'])

df = df.set_index('multiverse_id')

unneeded_columns = ['name', 'layout', 'type', 'subtypes', 'supertypes', 'colors',
                    'color_identity', 'rarity', 'text', 'flavor', 'id', 'image_url', 
                    'mana_cost', 'artist']

features = df.drop(columns=unneeded_columns)

num_input_features, num_columns = features.shape
y_columns = ["is_white", "is_blue", "is_black", "is_red", "is_green"]
y = features[y_columns]

features.drop(y_columns, axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

print(X_train.columns.to_series().groupby(X_train.dtypes).groups)

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
