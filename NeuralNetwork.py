import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import csv


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('all_mtg_cards_feature_abilities_with_colors.csv', low_memory=False)

# Remove rows with missing color percentage data
df = df.dropna(subset=['red_pct', 'green_pct', 'blue_pct', 'yellow_pct', 'black_pct', 'white_pct'])

df = df.set_index('multiverse_id')

unneeded_columns = ['name', 'layout', 'type', 'subtypes', 'supertypes', 'colors',
                    'color_identity', 'rarity', 'text', 'flavor', 'id', 'image_url', 
                    'mana_cost', 'artist', 'number']

features = df.drop(columns=unneeded_columns)

num_input_features, num_columns = features.shape
y_columns = ["is_white", "is_blue", "is_black", "is_red", "is_green"]
y = features[y_columns]

features.drop(y_columns, axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

print(X_train.columns.to_series().groupby(X_train.dtypes).groups)
