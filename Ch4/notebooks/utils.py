import boto3
import pickle
import config

def get_client(s3_access_key=config.S3_ACCESS_ID, 
               s3_secret_key=config.S3_ACCESS_KEY, 
               s3_endpoint_url=config.S3_END_POINT):

    client = boto3.client(service_name='s3',
                          aws_access_key_id = s3_access_key,
                          aws_secret_access_key = s3_secret_key, 
                          endpoint_url=s3_endpoint_url,
                          verify=True)

    return client

def write_to_store(bucket, data, key, client):
    if not check_bucket_exists(bucket, client):
        raise ValueError(f"Bucket {bucket} does not exist")

    client.put_object(Body=pickle.dumps(data),
                      Bucket=bucket,
                      Key=key)


def read_from_store(bucket, key, client):
    if not check_bucket_exists(bucket, client):
        raise ValueError(f"Bucket {bucket} does not exist")

    raw_data = client.get_object(Bucket=bucket,
                                 Key=key)['Body']._raw_stream.data

    return pickle.loads(raw_data)

def check_bucket_exists(bucket, client):
    return bucket in [i['Name'] for i in client.list_buckets()['Buckets']]

def create_bucket(bucket, client):
    if not check_bucket_exists(bucket, client):
        client.create_bucket(Bucket=bucket)
