# Data cleaning script for Magic: The Gathering cards dataset
# git test
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

DEFAULT_DROP = [
	'variations', 'rulings', 'watermark', 'printings', 'foreign_names',
	'original_text', 'original_type', 'legalities', 'set_name', 'set', 'loyalty', 
    'number', 'id', 'layout'
]

cols_to_drop = []
for c in df.columns:
    col = df[c]
    # If all values are NA, drop
    if col.isna().all():
        cols_to_drop.append(c)
        continue
    # If all values are empty after string-conversion and strip, treat as empty
    s = col.fillna('').astype(str).str.strip()
    if (s == '').all():
        cols_to_drop.append(c)

# Add user-requested drops (default list + any extra_drop passed)
drops = set(cols_to_drop)

# include default columns to remove even if not empty
for c in DEFAULT_DROP:
    if c in df.columns:
        drops.add(c)

df_clean = df.drop(columns=drops)



# Save the cleaned dataset
output_file = 'all_mtg_cards_cleaned_v1.csv'
df_clean.to_csv(output_file, index=False)

# Print final statistics
print(f"Final number of rows in cleaned dataset: {len(df)}")
print(f"Cleaned dataset saved as: {output_file}")
