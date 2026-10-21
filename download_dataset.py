"""
NeuroScan AI: Kaggle Dataset Downloader and Extractor
Target Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
"""

import os
import sys
import zipfile
import shutil

DATASET_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

def download_via_kagglehub():
    """Attempt download using kagglehub if available."""
    try:
        import kagglehub
        print("[+] Attempting download via kagglehub...")
        path = kagglehub.dataset_download(DATASET_SLUG)
        print(f"[+] Downloaded by kagglehub to: {path}")
        
        # Copy or link to local dataset directory
        if os.path.exists(path):
            os.makedirs(TARGET_DIR, exist_ok=True)
            for item in os.listdir(path):
                s = os.path.join(path, item)
                d = os.path.join(TARGET_DIR, item)
                if os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            print(f"[+] Dataset successfully organized at: {TARGET_DIR}")
            return True
    except ImportError:
        print("[-] kagglehub package not installed (pip install kagglehub).")
    except Exception as e:
        print(f"[-] kagglehub download failed: {e}")
    return False

def extract_local_zip():
    """Scan root directory for any downloaded zip file and extract it."""
    base_dir = os.path.dirname(__file__)
    possible_zips = [
        f for f in os.listdir(base_dir) 
        if f.lower().endswith(".zip") and ("tumor" in f.lower() or "brain" in f.lower() or "dataset" in f.lower() or "archive" in f.lower())
    ]
    
    if not possible_zips:
        return False
        
    zip_path = os.path.join(base_dir, possible_zips[0])
    print(f"[+] Found downloaded archive: {possible_zips[0]}. Extracting to {TARGET_DIR}...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)
        
    print(f"[+] Successfully extracted {possible_zips[0]} to {TARGET_DIR}")
    return True

def print_manual_instructions():
    """Print step-by-step instructions for manual download."""
    print("=" * 70)
    print("NeuroScan AI: Brain Tumor MRI Dataset Guide")
    print("=" * 70)
    print("You can obtain the dataset from Kaggle via either method:\n")
    print("METHOD 1: Direct Web Download (Recommended & Easiest)")
    print("  1. Visit: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
    print("  2. Click the 'Download' button (top right, ~150 MB).")
    print(f"  3. Save or move the downloaded 'archive.zip' into this directory:")
    print(f"     {os.path.abspath(os.path.dirname(__file__))}")
    print("  4. Re-run this script: python download_dataset.py")
    print("     It will automatically extract it into the 'dataset/' folder.\n")
    print("METHOD 2: Kaggle CLI / Python API")
    print("  1. pip install kagglehub")
    print("  2. python -c \"import kagglehub; kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset')\"")
    print("=" * 70)

if __name__ == "__main__":
    print("Checking for existing dataset or downloaded archives...")
    if os.path.exists(TARGET_DIR) and (os.path.exists(os.path.join(TARGET_DIR, "Training")) or os.path.exists(os.path.join(TARGET_DIR, "Testing"))):
        print(f"[+] Dataset is already present at: {TARGET_DIR}")
        sys.exit(0)
        
    if extract_local_zip():
        sys.exit(0)
        
    if download_via_kagglehub():
        sys.exit(0)
        
    print_manual_instructions()
