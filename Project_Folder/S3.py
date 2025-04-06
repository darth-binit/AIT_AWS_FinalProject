import os
import boto3
import joblib
import tarfile


def download_and_extract_model_artifacts():
    bucket_name = "finalprojectait-2025"
    s3_key = "models/latest/model.tar"  #

    # Define local paths for the tarball and extraction folder
    local_tar_path = os.path.join(os.getcwd(), "model.tar")  #
    extract_dir = os.path.join(os.getcwd(), "model_artifacts")

    s3 = boto3.client("s3", region_name="us-east-1")
    print(f"Downloading s3://{bucket_name}/{s3_key} to {local_tar_path} ...")
    s3.download_file(bucket_name, s3_key, local_tar_path)

    # Create the extraction directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Extract the tarball (non-gzipped tar file)
    with tarfile.open(local_tar_path, "r:") as tar:
        tar.extractall(path=extract_dir)

    print("Extraction complete. Files are in:", extract_dir)
    return extract_dir


def download_model_artifacts():
    bucket_name = "finalprojectait-2025"
    # Define S3 keys for your artifacts
    s3_keys = {
        "random_forest_model.pkl": "models/random_forest_model.pkl",
        "scaler.pkl": "models/scaler.pkl",
        "brand_means.pkl": "models/brand_means.pkl",
        "onehotencode.pkl": "models/onehotencode.pkl"
    }
    # Create a local directory to save the artifacts
    local_dir = os.path.join(os.getcwd(), "model_artifacts")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    s3 = boto3.client("s3", region_name="us-east-2")  # adjust region if necessary
    for local_filename, s3_key in s3_keys.items():
        local_path = os.path.join(local_dir, local_filename)
        print(f"Downloading {s3_key} to {local_path} ...")
        s3.download_file(bucket_name, s3_key, local_path)

    return local_dir


# On startup, download and load the model artifacts
# model_dir = download_model_artifacts()
#
# model = joblib.load(os.path.join(model_dir, "random_forest_model.pkl"))
# scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
# brand_means = joblib.load(os.path.join(model_dir, "brand_means.pkl"))
# onehot = joblib.load(os.path.join(model_dir, "onehotencode.pkl"))