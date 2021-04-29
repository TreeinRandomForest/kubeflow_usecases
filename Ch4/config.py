import os

S3_END_POINT = os.getenv('S3_END_POINT')
S3_ACCESS_ID = os.getenv('AWS_ACCESS_KEY_ID')
S3_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

BASE_IMAGE = 'docker.io/pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime'
BUCKET_NAME = 'opf-datacatalog'
