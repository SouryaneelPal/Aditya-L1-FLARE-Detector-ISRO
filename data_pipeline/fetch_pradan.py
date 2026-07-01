import os
import glob
import zipfile
import pandas as pd
import numpy as np
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
    Finds and loads data files for the range April 2024 - June 2024.
    Returns a concatenated DataFrame of all matching files in that period.
    """
    print(f"\nLooking for {instrument_name} data (Apr-Jun 2024)...")
    
    extract_zips()
    
    all_items = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*"), recursive=True)
    matching_files = []
    search_code = instrument_code.lower()
    
    # Filter files by name/extension and date pattern (202404, 202405, 202406)
    valid_months = ['202404', '202405', '202406']
    
    for item in all_items:
        item_lower = item.lower()
        if os.path.isfile(item) and not item_lower.endswith('.zip'):
            if search_code in item_lower or instrument_name.lower() in item_lower:
                if not item_lower.endswith(('.xml', '.txt', '.pdf', '.png')):
                    if any(month in item_lower for month in valid_months):
                        matching_files.append(item)
                
    if not matching_files:
        print(f"❌ No {instrument_name} files found for Apr-Jun 2024!")
        return None
            
    print(f"📄 Found {len(matching_files)} files for the target period.")
    
    data_frames = []
    for target_file in matching_files:
        try:
            print(f"🔭 Parsing: {os.path.basename(target_file)}")
            if target_file.endswith('.csv'):
                df = pd.read_csv(target_file)
                data_frames.append(df)
            
            elif target_file.endswith(('.fits', '.fits.gz', '.pi', '.pi.gz')):
                with fits.open(target_file) as hdul:
                    for hdu in hdul:
                        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                            df = Table(hdu.data).to_pandas()
                            if 'COUNTS' in df.columns and df['COUNTS'].dtype == 'object':
                                df['COUNTS'] = df['COUNTS'].apply(lambda x: np.sum(x) if isinstance(x, (list, np.ndarray)) else x)
                                df['COUNTS'] = pd.to_numeric(df['COUNTS'])
                            data_frames.append(df)
                            break # Assume first table is the primary one
        except Exception as e:
            print(f"❌ Error reading {os.path.basename(target_file)}: {e}")

    if data_frames:
        print(f"✅ Successfully combined {len(data_frames)} files.")
        return pd.concat(data_frames, ignore_index=True)
    
    return None

if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
    else:
        solexs_data = find_and_load_data("SOLEXS", "SoLEXS")
        helios_data = find_and_load_data("hel1os", "HEL1OS")