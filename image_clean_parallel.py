import pandas as pd
import numpy as np
import os
import io
from PIL import Image
import imagehash
import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# Suppress SSL verification warnings
urllib3.disable_warnings(InsecureRequestWarning)

def load_bad_images(bad_image_path):
    """Load bad example image and compute its hash"""
    try:
        print(f"Loading bad example image from: {bad_image_path}")
        img = Image.open(bad_image_path).convert('RGB')
        hash_value = imagehash.average_hash(img)
        print(f"Successfully computed hash for bad example: {hash_value}")
        return hash_value
    except Exception as e:
        print(f"Error loading bad example image: {e}")
        return None

def process_image(args):
    """Process a single image URL and return similarity result"""
    url, bad_hash, threshold = args
    try:
        response = requests.get(url, verify=False, timeout=10)
        response.raise_for_status()
        
        # Load image from response content
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Compute hash
        img_hash = imagehash.average_hash(img)
        
        # Compare hashes
        difference = img_hash - bad_hash
        
        return difference <= threshold, difference
    except Exception as e:
        return False, None

def chunk_list(lst, chunk_size):
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def main():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('all_mtg_cards_cleaned_v1.csv')
    print(f"Loaded {len(df)} rows")
    
    # Load bad example image
    bad_image_path = os.path.join('downloaded_images', 'card_back.jpg')
    bad_hash = load_bad_images(bad_image_path)
    
    if bad_hash is None:
        print("Failed to load bad example image. Exiting.")
        return

    # Prepare arguments for parallel processing
    threshold = 8
    urls = df['image_url'].tolist()
    args = [(url, bad_hash, threshold) for url in urls]
    
    # Calculate optimal chunk size (process 1000 images at a time)
    chunk_size = 1000
    chunks = chunk_list(args, chunk_size)
    
    # Initialize results
    keep_rows = []
    excluded_cards = []
    
    # Setup multiprocessing pool
    num_processes = cpu_count()
    print(f"\nProcessing with {num_processes} CPU cores in parallel...")
    
    start_time = time.time()
    
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        for chunk_idx, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {chunk_idx + 1}/{len(chunks)}")
            
            # Process chunk in parallel
            results = list(tqdm(pool.imap(process_image, chunk), 
                              total=len(chunk),
                              desc="Processing images"))
            
            # Process results for this chunk
            chunk_start = chunk_idx * chunk_size
            for i, (is_similar, difference) in enumerate(results):
                idx = chunk_start + i
                if idx < len(df):  # Ensure we don't go past the end of the dataframe
                    if not is_similar:
                        keep_rows.append(True)
                    else:
                        print(f"Found similar image to exclude: {df.iloc[idx]['name']} (diff: {difference})")
                        excluded_cards.append(df.iloc[idx]['name'])
                        keep_rows.append(False)
            
            # Calculate and display progress
            processed = len(keep_rows)
            total = len(df)
            elapsed_time = time.time() - start_time
            progress = processed / total
            estimated_total_time = elapsed_time / progress if progress > 0 else 0
            time_remaining = estimated_total_time - elapsed_time
            
            print(f"\nProgress: {processed}/{total} rows ({progress*100:.1f}%)")
            print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
            print(f"Estimated time remaining: {time_remaining/60:.1f} minutes")
    
    # Apply filter
    filtered_df = df[keep_rows]
    
    # Print summary of excluded cards
    print("\nExcluded Cards:")
    for card in excluded_cards:
        print(f"- {card}")
    print(f"\nTotal cards excluded: {len(excluded_cards)}")
    
    # Save results
    output_file = 'all_mtg_cards_cleaned_image.csv'
    filtered_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(filtered_df)} rows to {output_file}")

if __name__ == '__main__':
    main()