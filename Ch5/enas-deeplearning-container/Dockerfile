FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN chmod 777 /var
WORKDIR /var
ADD enas-deeplearning.py /var
ADD requirements.txt /var
RUN pip install --no-cache-dir -r /var/requirements.txt

ENTRYPOINT ["python", "/var/enas-deeplearning.py"]
