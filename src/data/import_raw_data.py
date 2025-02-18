import requests
import os
import logging
from check_structure import check_existing_file, check_existing_folder

def import_raw_data(raw_data_relative_path, filenames, bucket_folder_url):
    """Import filenames from bucket_folder_url into raw_data_relative_path."""
    # Ensure the raw data folder exists
    check_existing_folder(raw_data_relative_path)
    
    # Download each file
    for filename in filenames:
        # Construct the URL (ensure no trailing slash issues)
        input_file = bucket_folder_url.rstrip("/") + "/" + filename
        output_file = os.path.join(raw_data_relative_path, filename)
        
        if check_existing_file(output_file):
            print(f"Downloading {input_file} as {os.path.basename(output_file)}")
            response = requests.get(input_file)
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(response.content)
                print(f"{filename} downloaded successfully.")
            else:
                print(f"Error accessing the object {input_file}: {response.status_code}")

def main(raw_data_relative_path="./dvc/data/raw", 
         filenames=["raw.csv"],
         bucket_folder_url="https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr"):
    """Download raw data from the given bucket URL and store it in the raw data folder."""
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info('Raw data imported.')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
