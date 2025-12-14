def prepare_model_from_s3(
    dest_path: str,
    bucket_name: str,
    model_name: str,
    ) -> str:
    import os
    import boto3

    print("Prepare artifact path... to push to MLflow Model Registry")
    # Copy model from s3 to artifact path
    import boto3
    s3 = boto3.client('s3', endpoint_url=os.getenv("S3_ENDPOINT_URL"), aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"))
    
    # preprocess model name if starts with "s3://"
    if model_name.startswith("s3://"):
        model_name = model_name.split("s3://")[1]
    
    # preprocess model name if starts with bucket_name
    if model_name.startswith(f'{bucket_name}/'):
        model_name = model_name.split(f'{bucket_name}/')[1]
    
    dest_name = os.path.basename(model_name)
    print(f"Copying {model_name} to {dest_name}")
    dest = os.path.join(dest_path, dest_name)
    # Check if dest already exists
    if os.path.exists(dest):
        print(f"{dest_name} model already exists in {dest}")
        return dest
    else:
        s3.download_file(bucket_name, model_name, dest)
        print(f"{dest_name} model copied to {dest}")
        return dest