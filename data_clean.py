import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast  # for safely evaluating string representation of lists

# Read the dataset
df = pd.read_csv('all_mtg_cards.csv', low_memory=False)

# Print initial number of rows
print(f"Initial number of rows: {len(df)}")

# Remove rows where multiverse_id is missing (NaN or empty)
df = df.dropna(subset=['multiverse_id'])
print(f"Rows after removing missing multiverse_ids: {len(df)}")

# Keep only rows where layout is 'normal'
df = df[df['layout'] == 'normal']
print(f"Rows after keeping only normal layout: {len(df)}")

# Remove rows with missing colors
df = df.dropna(subset=['colors'])
print(f"Rows after removing missing colors: {len(df)}")

# Convert string representation of lists to actual lists
df['colors'] = df['colors'].apply(ast.literal_eval)

# Keep only rows with single color (list length of 1)
df = df[df['colors'].apply(len) == 1]
print(f"Rows after keeping only single-colored cards: {len(df)}")

# Save the cleaned dataset
output_file = 'all_mtg_cards_cleaned_v2.csv'
df.to_csv(output_file, index=False)

# Print final statistics
print(f"Final number of rows in cleaned dataset: {len(df)}")
print(f"Cleaned dataset saved as: {output_file}")