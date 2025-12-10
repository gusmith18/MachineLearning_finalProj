import os
import glob
import csv
import pandas as pd
import cv2
import numpy as np
import requests
import tempfile
from urllib.request import urlopen
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')


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


def _process_url_chunk(chunk_data):
    """Process a chunk of URLs and return results. Used for parallel processing."""
    df_chunk, image_url_column = chunk_data
    results = []
    
    for idx, row in df_chunk.iterrows():
        url = row[image_url_column]
        
        if pd.isna(url):
            results.append((idx, None))
            continue
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                temp_path = tmp_file.name
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                tmp_file.write(response.content)
            
            img = cv2.imread(temp_path)
            if img is None:
                results.append((idx, None))
            else:
                pct, counts, total_pixels = color_detect(img)
                color_data = _build_color_data(pct, counts)
                results.append((idx, color_data))
            
            os.unlink(temp_path)
            
        except Exception as e:
            results.append((idx, None))
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    return results


def append_color_to_csv(csv_file, image_dir=None, name_column='name', image_url_column='image_url', output_file=None, chunk_size=50):
    """
    Read a CSV file, process images (either from disk or download from URLs), 
    and append color data to matching rows.
    
    Args:
        csv_file: path to existing CSV with card data
        image_dir: directory containing card images (if None, will download from URLs)
        name_column: column name in CSV that contains card names
        image_url_column: column name in CSV that contains image URLs (used if image_dir is None)
        output_file: output CSV path (default: same as input with _with_colors suffix)
        chunk_size: number of rows to process per parallel worker (default: 50)
    """
    # Read the CSV
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")
    
    # Initialize color columns
    color_columns = [
        'red_count', 'green_count', 'blue_count', 'yellow_count', 'black_count', 'white_count',
        'red_pct', 'green_pct', 'blue_pct', 'yellow_pct', 'black_pct', 'white_pct',
        'red_norm_pct', 'green_norm_pct', 'blue_norm_pct', 'yellow_norm_pct', 'black_norm_pct'
    ]
    
    for col in color_columns:
        df[col] = None  # Initialize with None
    
    matched = 0
    processed = 0
    skipped = 0
    
    if image_dir and os.path.exists(image_dir):
        # Process from local directory
        print(f"Processing images from directory: {image_dir}")
        images = list_images_in_dir(image_dir)
        print(f"Found {len(images)} images in {image_dir}")
        
        for image_path in images:
            card_name = extract_card_name_from_filename(image_path)
            matching_rows = df[df[name_column].str.lower() == card_name.lower()]
            
            if len(matching_rows) == 0:
                continue
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: failed to read image {image_path}, skipping")
                skipped += 1
                continue
            
            pct, counts, total_pixels = color_detect(img)
            color_data = _build_color_data(pct, counts)
            
            for idx in matching_rows.index:
                for col, val in color_data.items():
                    df.at[idx, col] = val
                matched += 1
            processed += 1
    
    elif image_url_column in df.columns:
        # Process from URLs using parallel chunks
        print(f"Processing images from URLs in '{image_url_column}' column")
        
        # Filter rows that need processing
        df_to_process = df[df[image_url_column].notna() & (df['red_count'].isna())].copy()
        
        if len(df_to_process) == 0:
            print("No URLs to process")
        else:
            print(f"Found {len(df_to_process)} images to download")
            
            # Split into chunks and create work items
            chunks = []
            for i in range(0, len(df_to_process), chunk_size):
                chunk = df_to_process.iloc[i:i+chunk_size]
                chunks.append((chunk, image_url_column))
            
            print(f"Processing {len(chunks)} chunks with up to {chunk_size} rows each...")
            
            # Process chunks in parallel
            num_workers = max(1, cpu_count() - 1)
            with Pool(num_workers) as pool:
                all_results = pool.map(_process_url_chunk, chunks)
            
            # Merge results back into main dataframe
            for chunk_results in all_results:
                for idx, color_data in chunk_results:
                    if color_data is not None:
                        for col, val in color_data.items():
                            df.at[idx, col] = val
                        matched += 1
                    else:
                        skipped += 1
                    processed += 1
            
            print(f"Processed {processed} images")
    
    else:
        print("Error: specify image_dir or ensure image_url_column exists in CSV")
        return None
    
    # Print summary
    print(f"\nProcessed {processed} images")
    print(f"Matched color data to {matched} rows")
    print(f"Skipped {skipped} images")
    
    # Save output
    if output_file is None:
        base, ext = os.path.splitext(csv_file)
        output_file = f"{base}_with_colors{ext}"
    
    df.to_csv(output_file, index=False)
    print(f"\nWrote output to {output_file}")
    
    return df


def _build_color_data(pct, counts):
    """Helper function to build color data dict from percentages and counts."""
    white_pct = pct.get('white', 0.0)
    non_white_total = max(0.0, 100.0 - white_pct)
    normalized = {}
    if non_white_total < 0.01:
        for c in ['red', 'green', 'blue', 'yellow', 'black']:
            normalized[c] = None
    else:
        for c in ['red', 'green', 'blue', 'yellow', 'black']:
            normalized[c] = (pct.get(c, 0.0) / non_white_total) * 100.0
    
    return {
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





# Example: Process from local directory
# append_color_to_csv(
#         'all_mtg_cards_feature_abilities.csv',
#         image_dir='card_image',
#         name_column='name',
#         output_file='all_mtg_cards_feature_abilities_with_colors.csv')

# Example: Process from image URLs in CSV
if __name__ == '__main__':
    append_color_to_csv(
            'first_features.csv',
            image_dir=None,
            name_column='name',
            image_url_column='image_url',
            output_file='second_features.csv')
