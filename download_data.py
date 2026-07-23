# download_data.py
import requests
import zipfile
import os
import io

def download_movielens():
    """Download MovieLens Latest Small dataset"""
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    print("📥 Downloading MovieLens dataset...")
    response = requests.get(url)
    
    print("📦 Extracting files...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall('data/')
    
    # Move files to data/ folder for easier access
    source_dir = 'data/ml-latest-small'
    for file in ['movies.csv', 'ratings.csv', 'links.csv', 'tags.csv']:
        src = os.path.join(source_dir, file)
        dst = os.path.join('data', file)
        if os.path.exists(src):
            os.rename(src, dst)
    
    print("✅ Dataset downloaded and extracted successfully!")
    print("📁 Files available:")
    for file in os.listdir('data'):
        if file.endswith('.csv'):
            size = os.path.getsize(os.path.join('data', file)) / 1024
            print(f"   - {file} ({size:.1f} KB)")

if __name__ == "__main__":
    download_movielens()