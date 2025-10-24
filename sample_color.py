import os
import glob
import argparse
import csv

import cv2
import numpy as np


def color_detect(image):
    """Return a dict of color name -> percentage of pixels in the image.

    Input: image as a NumPy BGR array (as returned by cv2.imread)
    Output: dict with keys: red, green, blue, yellow, black, white (percentages)
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
    # Return counts and percentages for downstream CSV output
    return color_percentages, color_counts, total_pixels


def list_images_in_dir(directory, exts=None):
    """Return a sorted list of image file paths in `directory` matching extensions."""
    if exts is None:
        exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, f'**/*.{ext}'), recursive=True))
    return sorted(files)


def process_directory(directory='card_image', output_csv=None):
    """Process all images in `directory` and compute color percentages.

    If `output_csv` is provided, save results there; otherwise print lines.
    """
    images = list_images_in_dir(directory)
    if not images:
        print(f"No images found in {directory}")
        return []

    results = []
    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: failed to read image {path}, skipping")
            continue
        pct, counts, total_pixels = color_detect(img)
        # normalized percentages over non-white pixels (if possible)
        white_pct = pct.get('white', 0.0)
        non_white_total = max(0.0, 100.0 - white_pct)
        normalized = {}
        if non_white_total < 0.01:
            # cannot normalize; leave normalized fields empty
            for c in ['red', 'green', 'blue', 'yellow', 'black']:
                normalized[c] = ''
        else:
            for c in ['red', 'green', 'blue', 'yellow', 'black']:
                normalized[c] = (pct.get(c, 0.0) / non_white_total) * 100.0

        row = {
            'filename': os.path.relpath(path, start=directory),
            'total_pixels': total_pixels,
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
            'red_norm_pct': normalized.get('red', ''),
            'green_norm_pct': normalized.get('green', ''),
            'blue_norm_pct': normalized.get('blue', ''),
            'yellow_norm_pct': normalized.get('yellow', ''),
            'black_norm_pct': normalized.get('black', ''),
        }
        results.append(row)

    # Output
    if output_csv:
        fieldnames = [
            'filename', 'total_pixels',
            'red_count', 'green_count', 'blue_count', 'yellow_count', 'black_count', 'white_count',
            'red_pct', 'green_pct', 'blue_pct', 'yellow_pct', 'black_pct', 'white_pct',
            'red_norm_pct', 'green_norm_pct', 'blue_norm_pct', 'yellow_norm_pct', 'black_norm_pct'
        ]
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Wrote {len(results)} rows to {output_csv}")
    else:
        for r in results:
            print(r)

    return results


def _build_arg_parser():
    p = argparse.ArgumentParser(description='Run color detection on card images')
    p.add_argument('--dir', default='card_image', help='Directory containing card images')
    p.add_argument('--out', default=None, help='Optional output CSV file')
    return p


if __name__ == '__main__':
    parser = _build_arg_parser()
    args = parser.parse_args()
    process_directory(directory=args.dir, output_csv=args.out)
