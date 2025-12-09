import requests
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import hashlib
import warnings
import urllib3
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Suppress only the InsecureRequestWarning from urllib3
warnings.filterwarnings('ignore', category=InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_image(image_url, card_name, folder_name="card_image"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    try:
        # Use a session to maintain settings
        session = requests.Session()
        # Disable SSL verification but keep the connection secure
        session.verify = False
        response = session.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Create a unique filename using the card name and a hash of the URL
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
        # Clean the card name to make it filesystem-friendly
        safe_name = "".join(c for c in card_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        file_extension = '.jpg'  # MTG cards are typically JPG images
        
        file_name = os.path.join(folder_name, f"{safe_name}_{url_hash}{file_extension}")
        
        # Check if file already exists
        if os.path.exists(file_name):
            print(f"Skipping existing file: {file_name}")
            return True

        with open(file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        print(f"Downloaded: {file_name}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading {image_url}: {e}")
        return False
    finally:
        try:
            session.close()
        except:
            pass

# Read the cleaned dataset
df = pd.read_csv('first_features.csv', low_memory=False)
print(f"Total cards in dataset: {len(df)}")

# Take a random sample
random_sample = df.sample(n=1000, random_state=42)  # Starting with a smaller sample for testing
print(f"Downloading {len(random_sample)} random card images...")

# Track success/failure counts
success_count = 0
failure_count = 0

# Download images with progress tracking
for idx, row in random_sample.iterrows():
    if download_image(row['image_url'], row['name']):
        success_count += 1
    else:
        failure_count += 1

    # Print progress every 100 downloads
    print(f"Progress: {success_count + failure_count}/{len(random_sample)} images processed")

print(f"Successfully downloaded: {success_count} images")
print(f"Failed downloads: {failure_count} images")