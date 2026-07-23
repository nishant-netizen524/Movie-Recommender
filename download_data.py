# download_data.py
import requests
import zipfile
import os
import io
import urllib3

# Disable SSL warnings (sometimes causes issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_movielens():
    """Download MovieLens Latest Small dataset with better error handling"""
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    # Add headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print("📥 Downloading MovieLens dataset...")
    print(f"   URL: {url}")
    
    try:
        # Try with verify=False to bypass SSL issues
        response = requests.get(url, headers=headers, verify=False, timeout=60)
        response.raise_for_status()
        
        print("📦 Extracting files...")
        os.makedirs('data', exist_ok=True)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall('data/')
        
        # Move files to data/ folder
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
                
    except requests.exceptions.ConnectionError as e:
        print("\n❌ Connection Error!")
        print("   This usually happens due to network/firewall restrictions.")
        print("\n💡 SOLUTION: Download manually from:")
        print("   https://grouplens.org/datasets/movielens/latest/")
        print("   Then place 'ml-latest-small.zip' in the 'data/' folder and extract it.")
        
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out. The server might be slow.")
        print("   Please try Solution 1 (manual download).")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("   Please try Solution 1 (manual download).")

if __name__ == "__main__":
    download_movielens()