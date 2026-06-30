import os
import glob
import zipfile
import pandas as pd
from astropy.io import fits
from astropy.table import Table

# Define where your downloaded ISRO files live
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")

def extract_zips():
    """Finds any .zip files in the raw_data folder and extracts them."""
    zip_files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.zip"), recursive=True)
    
    for zip_path in zip_files:
        extract_folder = zip_path.replace(".zip", "")
        if not os.path.exists(extract_folder):
            print(f"📦 Auto-Extracting {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)

def find_and_load_data(instrument_code, instrument_name):
    """
    Generic function to find extracted data files.
    """
    print(f"\nLooking for {instrument_name} data...")
    
    # --- THE FIX: Force the script to unzip files BEFORE it starts searching ---
    extract_zips()
    # -------------------------------------------------------------------------
    
    all_items = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*"), recursive=True)
    files = []
    search_code = instrument_code.lower()
    
    for item in all_items:
        item_lower = item.lower()
        if os.path.isfile(item) and not item_lower.endswith('.zip'):
            # Looking for SLX, SOLEXS, HLS, or HEL1OS in the file path
            if search_code in item_lower or instrument_name.lower() in item_lower:
                if not item_lower.endswith(('.xml', '.txt', '.pdf', '.png')):
                    files.append(item)
                
    if not files:
        print(f"❌ No {instrument_name} files found! Did you put the zips in data_pipeline/raw_data/?")
        return None
        
    print(f"✅ Found {len(files)} {instrument_name} data files!")
    
    first_file = files[0]
    print(f"📄 Looking inside the first file: {os.path.basename(first_file)}")
    
    if first_file.endswith('.csv'):
        return pd.read_csv(first_file)
    elif first_file.endswith('.parquet'):
        return pd.read_parquet(first_file)
    elif first_file.endswith(('.fits', '.fits.gz', '.lc', '.lc.gz')):
        print(f"🔭 Parsing {instrument_name} astronomical file using Astropy...")
        try:
            with fits.open(first_file) as hdul:
                data_table = Table(hdul[1].data)
                df = data_table.to_pandas()
                print(f"✅ Successfully converted to Pandas DataFrame! Shape: {df.shape}")
                return df
        except Exception as e:
            print(f"❌ Error reading FITS/LC file: {e}")
            return first_file
    else:
        print(f"⚠️ Note: File is a '{first_file.split('.')[-1]}' format. We may need a specific parser.")
        return first_file

if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"Created directory: {RAW_DATA_DIR}")
    else:
        solexs_data = find_and_load_data("SOLEXS", "SoLEXS")
        helios_data = find_and_load_data("hel1os", "HEL1OS")