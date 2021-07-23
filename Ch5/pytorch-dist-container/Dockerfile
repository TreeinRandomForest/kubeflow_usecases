FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN chmod 777 /var
WORKDIR /var
ADD pytorch_dist_test2.py /var
ADD requirements.txt /var
RUN pip install --no-cache-dir -r /var/requirements.txt

ENTRYPOINT ["python", "/var/pytorch_dist_test2.py"]
