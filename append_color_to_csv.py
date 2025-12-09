import os
import glob
import argparse
import csv
import pandas as pd
import cv2
import numpy as np


def color_detect(image):
    """Return a dict of color name -> percentage of pixels in the image.

    Input: image as a NumPy BGR array (as returned by cv2.imread)
    Output: tuple (percentages dict, counts dict, total_pixels)
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    color_ranges = {
        'red': [(0, 100, 100), (10, 255, 255)],
        'green': [(40, 100, 100), (80, 255, 255)],
        'blue': [(100, 100, 100), (140, 255, 255)],
        'yellow': [(20, 100, 100), (40, 255, 255)],
        'black': [(0, 0, 0), (180, 255, 30)],
        'white': [(0, 0, 200), (180, 20, 255)]
    }

    # Create a mask for each color and count pixels
    color_counts = {}
    for color, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_np, upper_np)
        count = int(cv2.countNonZero(mask))
        color_counts[color] = count

    # compute percentage values for each color in the card
    total_pixels = max(1, int(image.shape[0] * image.shape[1]))
    color_percentages = {color: (count / total_pixels) * 100.0 for color, count in color_counts.items()}
    return color_percentages, color_counts, total_pixels


def list_images_in_dir(directory, exts=None):
    """Return a sorted list of image file paths in `directory` matching extensions."""
    if exts is None:
        exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, f'**/*.{ext}'), recursive=True))
    return sorted(files)


def extract_card_name_from_filename(filename):
    """Extract card name from filename (e.g., 'Card Name_hash.jpg' -> 'Card Name')"""
    basename = os.path.basename(filename)
    # Remove extension
    name_with_hash = os.path.splitext(basename)[0]
    # Split on the last underscore (hash is usually after the last _)
    if '_' in name_with_hash:
        card_name = '_'.join(name_with_hash.split('_')[:-1])
        return card_name
    return name_with_hash


def append_color_to_csv(csv_file, image_dir='card_image', name_column='name', output_file=None):
    """
    Read a CSV file, process images from image_dir, and append color data to matching rows.
    
    Args:
        csv_file: path to existing CSV with card data
        image_dir: directory containing card images
        name_column: column name in CSV that contains card names to match
        output_file: output CSV path (default: same as input with _with_colors suffix)
    """
    # Read the CSV
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")
    
    # Get all images
    images = list_images_in_dir(image_dir)
    print(f"Found {len(images)} images in {image_dir}")
    
    # Initialize color columns
    color_columns = [
        'red_count', 'green_count', 'blue_count', 'yellow_count', 'black_count', 'white_count',
        'red_pct', 'green_pct', 'blue_pct', 'yellow_pct', 'black_pct', 'white_pct',
        'red_norm_pct', 'green_norm_pct', 'blue_norm_pct', 'yellow_norm_pct', 'black_norm_pct'
    ]
    
    for col in color_columns:
        df[col] = None  # Initialize with None
    
    # Process each image
    matched = 0
    unmatched = []
    
    for image_path in images:
        # Extract card name from filename
        card_name = extract_card_name_from_filename(image_path)
        
        # Find matching row(s) in CSV
        matching_rows = df[df[name_column].str.lower() == card_name.lower()]
        
        if len(matching_rows) == 0:
            unmatched.append(card_name)
            continue
        
        # Process the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: failed to read image {image_path}, skipping")
            continue
        
        pct, counts, total_pixels = color_detect(img)
        
        # Compute normalized percentages
        white_pct = pct.get('white', 0.0)
        non_white_total = max(0.0, 100.0 - white_pct)
        normalized = {}
        if non_white_total < 0.01:
            for c in ['red', 'green', 'blue', 'yellow', 'black']:
                normalized[c] = None
        else:
            for c in ['red', 'green', 'blue', 'yellow', 'black']:
                normalized[c] = (pct.get(c, 0.0) / non_white_total) * 100.0
        
        # Build color data dict
        color_data = {
            'red_count': counts.get('red', 0),
            'green_count': counts.get('green', 0),
            'blue_count': counts.get('blue', 0),
            'yellow_count': counts.get('yellow', 0),
            'black_count': counts.get('black', 0),
            'white_count': counts.get('white', 0),
            'red_pct': pct.get('red', 0.0),
            'green_pct': pct.get('green', 0.0),
            'blue_pct': pct.get('blue', 0.0),
            'yellow_pct': pct.get('yellow', 0.0),
            'black_pct': pct.get('black', 0.0),
            'white_pct': pct.get('white', 0.0),
            'red_norm_pct': normalized.get('red', None),
            'green_norm_pct': normalized.get('green', None),
            'blue_norm_pct': normalized.get('blue', None),
            'yellow_norm_pct': normalized.get('yellow', None),
            'black_norm_pct': normalized.get('black', None),
        }
        
        # Update all matching rows with color data
        for idx in matching_rows.index:
            for col, val in color_data.items():
                df.at[idx, col] = val
            matched += 1
    
    # Print summary
    print(f"\nMatched {matched} images to rows in CSV")
    if unmatched:
        print(f"Could not match {len(unmatched)} images:")
        for name in unmatched[:10]:  # Show first 10
            print(f"  - {name}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")
    
    # Save output
    if output_file is None:
        base, ext = os.path.splitext(csv_file)
        output_file = f"{base}_with_colors{ext}"
    
    df.to_csv(output_file, index=False)
    print(f"\nWrote output to {output_file}")
    
    return df



if __name__ == '__main__':
    
    append_color_to_csv(
        'all_mtg_cards_feature_abilities.csv',
        'card_image',
        'name',
        'all_mtg_cards_feature_abilities_with_colors.csv'
    )
