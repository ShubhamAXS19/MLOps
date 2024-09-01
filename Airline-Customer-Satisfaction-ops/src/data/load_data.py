# src/data/load_data.py
import boto3
import zipfile
import os

def download_and_extract_data(s3_bucket: str, s3_key: str, download_path: str, extract_to: str):
    """
    Download and extract data from an S3 bucket.
    """
    s3 = boto3.client('s3')
    
    # Download the file
    s3.download_file(s3_bucket, s3_key, download_path)
    
    # Extract the file
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Data downloaded and extracted to {extract_to}")

if __name__ == "__main__":
    # Example usage
    s3_bucket = "your-s3-bucket-name"
    s3_key = "path/to/your/data.zip"
    download_path = "data/raw/data.zip"
    extract_to = "data/raw/"
    
    os.makedirs(extract_to, exist_ok=True)
    download_and_extract_data(s3_bucket, s3_key, download_path, extract_to)
