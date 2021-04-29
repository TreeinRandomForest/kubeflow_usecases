import os

S3_END_POINT = os.getenv('S3_END_POINT', 'http://192.168.1.205:9000')
S3_ACCESS_ID = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
S3_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')

BASE_IMAGE = 'docker.io/pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime'
BUCKET_NAME = 'opf-datacatalog'
