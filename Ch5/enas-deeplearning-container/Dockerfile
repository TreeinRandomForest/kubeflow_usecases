#FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN chmod 777 /var
WORKDIR /var
ADD enas-deeplearning.py /var
ADD requirements.txt /var
RUN pip3 install --no-cache-dir -r /var/requirements.txt

ENTRYPOINT ["python3", "/var/enas-deeplearning.py"]
